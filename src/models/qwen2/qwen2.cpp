#include "qwen2.hpp"

#include "../../layers/qwen2/qwen2.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/top_k/op.hpp"
#include "../../config.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace llaisys {

namespace {

tensor_t stable_capture(const tensor_t &tensor) {
    return tensor->to(LLAISYS_DEVICE_CPU, 0);
}

} // namespace

Qwen2::Qwen2(Qwen2Meta meta, llaisysDeviceType_t device, int *device_ids, int ndevice) : _meta(meta) {
    this->_device_type = device;
    this->_device_ids = std::vector<int>(device_ids, device_ids + ndevice);

    llaisysDataType_t dtype = meta.dtype;
    const size_t nlayer = meta.nlayer;
    const size_t hs = meta.hs;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t di = meta.di;
    const size_t voc = meta.voc;
    const size_t token_num = KV_CACHE_TOKEN_NUM;
    const size_t block_num = KV_CACHE_BLOCK_NUM;
    const size_t block_size = block_num * token_num * nkvh * dh;
    const size_t batch_max_token_num = BATCH_MAX_TOKEN_NUM;
    const size_t batch_max_seq_num = BATCH_MAX_SEQ_NUM;
    const size_t max_block_num = MAX_BLOCK_NUM;
    int device_id = device_ids[0];

    // 初始化运行时张量
    _tensors._x_norm = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._q = Tensor::create({batch_max_token_num, nh * dh}, dtype, device, device_id);
    _tensors._k = Tensor::create({batch_max_token_num, nkvh * dh}, dtype, device, device_id);
    _tensors._v = Tensor::create({batch_max_token_num, nkvh * dh}, dtype, device, device_id);
    _tensors._q_rope = Tensor::create({batch_max_token_num, nh, dh}, dtype, device, device_id);
    _tensors._k_rope = Tensor::create({batch_max_token_num, nkvh, dh}, dtype, device, device_id);
    _tensors._attn_val = Tensor::create({batch_max_token_num, nh, dh}, dtype, device, device_id);
    _tensors._attn_out = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._x_attn = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._m_norm = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._gate = Tensor::create({batch_max_token_num, di}, dtype, device, device_id);
    _tensors._up = Tensor::create({batch_max_token_num, di}, dtype, device, device_id);
    _tensors._swiglu = Tensor::create({batch_max_token_num, di}, dtype, device, device_id);
    _tensors._down = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._x_mlp = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._dev_block_ids = Tensor::create({batch_max_seq_num * max_block_num}, LLAISYS_DTYPE_I64, device, device_id);
    _tensors._dev_cut_idx = Tensor::create({batch_max_seq_num + 1}, LLAISYS_DTYPE_I64, device, device_id);
    _tensors._dev_tot_len = Tensor::create({batch_max_seq_num}, LLAISYS_DTYPE_I64, device, device_id);
    _tensors._dev_token_ids = Tensor::create({batch_max_token_num}, LLAISYS_DTYPE_I64, device, device_id);
    _tensors._dev_pos_ids = Tensor::create({batch_max_token_num}, LLAISYS_DTYPE_I64, device, device_id);
    _tensors._x = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._x_part = Tensor::create({batch_max_seq_num, hs}, dtype, device, device_id);
    _tensors._x_part_norm = Tensor::create({batch_max_seq_num, hs}, dtype, device, device_id);
    _tensors._logits = Tensor::create({batch_max_seq_num, voc}, dtype, device, device_id);
    _tensors._top_idx = Tensor::create({batch_max_seq_num, MAX_TOP_K}, LLAISYS_DTYPE_I64, device, device_id);
    _tensors._top_val = Tensor::create({batch_max_seq_num, MAX_TOP_K}, dtype, device, device_id);
    _tensors._attn_acc = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, dh}, LLAISYS_DTYPE_F32, device, device_id);
    _tensors._attn_sum = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, 1}, LLAISYS_DTYPE_F32, device, device_id);
    _tensors._attn_max = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, 1}, LLAISYS_DTYPE_F32, device, device_id);

    // 初始化分层 kv cache（写入基类成员）
    this->_k_cache = std::vector<tensor_t>(nlayer, nullptr);
    this->_v_cache = std::vector<tensor_t>(nlayer, nullptr);
    tensor_t cache = Tensor::create({2, nlayer, block_size}, meta.dtype, device, device_id);
    tensor_t k_cache = cache->slice(0, 0, 1)->reshape({nlayer, block_size});
    tensor_t v_cache = cache->slice(0, 1, 2)->reshape({nlayer, block_size});
    for (size_t i = 0; i < nlayer; ++i) {
        this->_k_cache[i] = k_cache->slice(0, i, i + 1)->reshape({block_num, token_num, nkvh, dh});
        this->_v_cache[i] = v_cache->slice(0, i, i + 1)->reshape({block_num, token_num, nkvh, dh});
    }

    // 创建调度器
    this->_scheduler = std::make_shared<Scheduler>(token_num, block_num, BATCH_MAX_TOKEN_NUM, BATCH_MAX_SEQ_NUM, meta.end_token);

    // 初始化 ModelContext
    this->_ctx.meta = core::ModelMeta{meta.dtype, meta.nlayer, meta.hs, meta.nh, meta.nkvh, meta.dh, meta.di,
                                           meta.maxseq, meta.voc, meta.epsilon, meta.theta, meta.end_token};
    this->_ctx.attns.resize(nlayer);
    this->_ctx.ffns.resize(nlayer);
    this->_ctx.k_caches.resize(nlayer);
    this->_ctx.v_caches.resize(nlayer);
}

Qwen2::~Qwen2() = default;

std::vector<int64_t> Qwen2::prepare_block_table(const std::vector<seq_t> &seqs) {
    return core::Model::prepare_block_table(seqs, MAX_BLOCK_NUM);
}

core::BatchPack Qwen2::prepare_prefill(const std::vector<seq_t> &seqs) {
    return core::Model::prepare_prefill(seqs);
}

core::BatchPack Qwen2::prepare_decode(const std::vector<seq_t> &seqs) {
    return core::Model::prepare_decode(seqs);
}

void Qwen2::bind_kv_caches() {
    const size_t nlayer = this->_meta.nlayer;
    for (size_t layer = 0; layer < nlayer; ++layer) {
        this->_ctx.k_caches[layer] = this->_k_cache[layer];
        this->_ctx.v_caches[layer] = this->_v_cache[layer];
    }
}

void Qwen2::bind_context_runtime(core::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) {
    const Qwen2Meta &meta = this->_meta;
    const Qwen2Tensors &ts = this->_tensors;
    core::ModelContext &ctx = this->_ctx;

    const std::vector<int64_t> &token_ids = pack.token_ids;
    const std::vector<int64_t> &pos_ids = pack.pos_ids;
    const std::vector<int64_t> &cut_idx = pack.cut_idx;
    const std::vector<int64_t> &tot_len = pack.tot_len;
    const size_t max_seq_len = pack.max_seq_len;
    const size_t tot_seq_len = token_ids.size();
    const size_t batch_size = tot_len.size();
    const size_t max_block_num = MAX_BLOCK_NUM;
    // block_ids 末行第 0 列是累计块数前缀和 = 整个批次总块数
    const size_t tot_block_num = block_ids[(batch_size - 1) * MAX_BLOCK_NUM];

    const size_t hs = meta.hs;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t di = meta.di;
    const size_t voc = meta.voc;

    ctx.runtime.token_ids = token_ids;
    ctx.runtime.pos_ids = pos_ids;
    ctx.runtime.cut_idx = cut_idx;
    ctx.runtime.tot_len = tot_len;
    ctx.runtime.max_seq_len = max_seq_len;
    ctx.runtime.tot_block_num = tot_block_num;
    ctx.runtime.is_prefill = is_prefill;

    ctx.workspace.x_norm = ts._x_norm->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.q = ts._q->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh * dh});
    ctx.workspace.k = ts._k->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh * dh});
    ctx.workspace.v = ts._v->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh * dh});
    ctx.workspace.q_rope = ts._q_rope->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh, dh});
    ctx.workspace.k_rope = ts._k_rope->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh, dh});
    ctx.workspace.attn_val = ts._attn_val->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh, dh});
    ctx.workspace.attn_out = ts._attn_out->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.x_attn = ts._x_attn->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.m_norm = ts._m_norm->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.gate = ts._gate->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    ctx.workspace.up = ts._up->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    ctx.workspace.swiglu = ts._swiglu->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    ctx.workspace.down = ts._down->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.x_mlp = ts._x_mlp->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.block_ids = ts._dev_block_ids->slice(0, 0, batch_size * max_block_num)->view({batch_size, max_block_num});
    ctx.workspace.cut_idx = ts._dev_cut_idx->slice(0, 0, batch_size + 1)->view({batch_size + 1});
    ctx.workspace.tot_len = ts._dev_tot_len->slice(0, 0, batch_size)->view({batch_size});
    ctx.workspace.pos_ids = ts._dev_pos_ids->slice(0, 0, tot_seq_len)->view({tot_seq_len});
    ctx.workspace.x = ts._x->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.x_part = ts._x_part->slice(0, 0, batch_size)->view({batch_size, hs});
    ctx.workspace.x_part_norm = ts._x_part_norm->slice(0, 0, batch_size)->view({batch_size, hs});
    ctx.workspace.logits = ts._logits->slice(0, 0, batch_size)->view({batch_size, voc});
    ctx.workspace.attn_acc = ts._attn_acc;
    ctx.workspace.attn_sum = ts._attn_sum;
    ctx.workspace.attn_max = ts._attn_max;
    ctx.workspace.capture_post_attn0.reset();
    ctx.workspace.capture_post_ffn0.reset();

    ctx.workspace.block_ids->load(block_ids.data());
    ctx.workspace.cut_idx->load(cut_idx.data());
    ctx.workspace.tot_len->load(tot_len.data());
    ctx.workspace.pos_ids->load(pos_ids.data());
}

std::vector<int64_t> Qwen2::sample_from_logits(tensor_t logits, core::BatchPack &pack) {
    const Qwen2Meta &meta = this->_meta;
    const llaisysDeviceType_t device = this->_device_type;
    const size_t batch_size = pack.tot_len.size();

    const size_t K = static_cast<size_t>(*std::max_element(pack.top_k.begin(), pack.top_k.end()));
    tensor_t top_idx = Tensor::create({batch_size, K}, LLAISYS_DTYPE_I64, device, this->_device_ids[0]);
    tensor_t top_val = Tensor::create({batch_size, K}, meta.dtype, device, this->_device_ids[0]);
    ops::topk(top_idx, top_val, logits, K);
    if (device != LLAISYS_DEVICE_CPU) {
        top_idx = top_idx->to(LLAISYS_DEVICE_CPU, 0);
        top_val = top_val->to(LLAISYS_DEVICE_CPU, 0);
    }
    const std::vector<int64_t> all_idx = top_idx->reshape({batch_size * K})->to_vector<int64_t>();
    const std::vector<float> all_val = top_val->reshape({batch_size * K})->to_vector<float>();
    std::vector<int64_t> result(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<SamplingCandidate> candidates;
        candidates.reserve(K);
        for (size_t j = 0; j < K; ++j) {
            const size_t offset = i * K + j;
            candidates.push_back({all_val[offset], all_idx[offset]});
        }
        result[i] = sample_token(candidates, pack.top_k[i], pack.top_p[i], pack.temperature[i], *pack.rngs[i]);
    }
    return result;
}

tensor_t Qwen2::last_logits(size_t batch_size) const {
    const size_t voc = this->_meta.voc;
    return this->_tensors._logits->slice(0, 0, batch_size)->view({batch_size, voc});
}

std::vector<int64_t> Qwen2::forward(core::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) {
    this->bind_kv_caches();
    this->bind_context_runtime(pack, block_ids, is_prefill);

    core::ModelContext &ctx = this->_ctx;
    ctx.runtime.is_prefill = is_prefill;

    // 显式组装 Layer
    // embedding
    core::Qwen2Embedding{}.forward(ctx);

    // 逐层: Attention → FFN
    for (size_t layer = 0; layer < ctx.meta.nlayer; ++layer) {
        core::Qwen2Attention(&ctx.attns[layer], layer).forward(ctx);
        if (layer == 0 && ctx.enable_layer0_capture) {
            ctx.workspace.capture_post_attn0 = stable_capture(ctx.workspace.x);
        }
        core::Qwen2FFN(&ctx.ffns[layer]).forward(ctx);
        if (layer == 0 && ctx.enable_layer0_capture) {
            ctx.workspace.capture_post_ffn0 = stable_capture(ctx.workspace.x);
        }
    }

    // 先去提取 last-token 隐藏状态
    const size_t batch_size = ctx.runtime.cut_idx.size() - 1;
    auto last_token_indices = ctx.workspace.cut_idx->slice(0, 1, batch_size + 1);
    ops::embedding(ctx.workspace.x_part, last_token_indices, ctx.workspace.x, -1);

    // RMS-norm 均方根归一化
    ops::rms_norm(ctx.workspace.x_part_norm, ctx.workspace.x_part, ctx.out_norm_w, ctx.meta.epsilon);

    // 把隐藏向量映射到词表维度（voc）生成未归一化的打分（logits）
    ops::linear(ctx.workspace.logits, ctx.workspace.x_part_norm, ctx.out_embed, nullptr);

    return this->sample_from_logits(ctx.workspace.logits, pack);
}

} // namespace llaisys
