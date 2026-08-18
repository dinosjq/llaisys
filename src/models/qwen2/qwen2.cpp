#include "qwen2.hpp"

#include "layer/qwen2.hpp"
#include "weight/qwen2_weights.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/top_k/op.hpp"
#include "../../config.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace llaisys {

// 初始化 Qwen2（仅设备；meta / 权重在 set_meta / set_weight）
Qwen2::Qwen2(llaisysDeviceType_t device, int *device_ids, int ndevice, std::unique_ptr<model::Qwen2Context> ctx) {
    // 设置设备类型和设备 id
    this->_device_type = device;
    this->_device_ids = std::vector<int>(device_ids, device_ids + ndevice);
    // 先挂上空壳上下文（Llama 可注入 LlamaContext）；allocate 在 set_meta 后调用
    this->_ctx = ctx ? std::move(ctx) : std::make_unique<model::Qwen2Context>();
}

Qwen2::~Qwen2() = default;

/**
 * 设置 meta
 * 1. 检查 meta_ready
 * 2. 设置 meta
 * 3. 分配上下文内存
 */
 void Qwen2::set_meta(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) {
    if (this->_meta_ready) {
        throw std::runtime_error("set_meta called twice");
    }
    this->_ctx->meta = model::Qwen2Meta::from_map(kv);
    this->allocate_runtime();
}

/**
 * 分配 Qwen2 上下文内存
 * 1. 初始化权重槽位
 * 2. 分配上下文内存
 * 3. 初始化分层 kv cache
 * 4. 创建调度器
 * 5. 设置 meta_ready 为 true
 */
void Qwen2::allocate_runtime() {
    auto &ctx = static_cast<model::Qwen2Context &>(*this->_ctx);
    const auto &meta = model::qwen_meta(ctx);
    // 获取模型参数
    const size_t nlayer = meta.nlayer;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t token_num = KV_CACHE_TOKEN_NUM;
    const size_t block_num = KV_CACHE_BLOCK_NUM;
    const size_t block_size = block_num * token_num * nkvh * dh;
    const int device_id = this->_device_ids[0];

    // 初始化权重槽位
    model::Qwen2Weights::init(ctx);

    // 分配上下文内存，初始化中间张量
    ctx.allocate(this->_device_type, device_id);

    // 初始化分层 kv cache
    this->_k_cache = std::vector<tensor_t>(nlayer, nullptr);
    this->_v_cache = std::vector<tensor_t>(nlayer, nullptr);
    tensor_t cache = Tensor::create({2, nlayer, block_size}, meta.dtype, this->_device_type, device_id);
    tensor_t k_cache = cache->slice(0, 0, 1)->reshape({nlayer, block_size});
    tensor_t v_cache = cache->slice(0, 1, 2)->reshape({nlayer, block_size});
    for (size_t i = 0; i < nlayer; ++i) {
        this->_k_cache[i] = k_cache->slice(0, i, i + 1)->reshape({block_num, token_num, nkvh, dh});
        this->_v_cache[i] = v_cache->slice(0, i, i + 1)->reshape({block_num, token_num, nkvh, dh});
    }

    // 创建调度器
    this->_scheduler = std::make_shared<Scheduler>(token_num, block_num, BATCH_MAX_TOKEN_NUM, BATCH_MAX_SEQ_NUM, meta.end_token);
    this->_meta_ready = true;
}

/**
 * 设置权重
 * 1. 检查 meta_ready 和 ctx
 * 2. 设置权重
 */
 void Qwen2::set_weight(const std::string &hf_name, tensor_t tensor) {
    if (!this->_meta_ready || !this->_ctx) {
        throw std::runtime_error("set_weight before set_meta");
    }
    model::Qwen2Weights::set_weight(*this->_ctx, hf_name, std::move(tensor));
}

/**
 * 前向推理
 * 1. 绑定 kv cache
 * 2. 绑定上下文运行时
 * 3. 设置上下文运行时
 * 4. 前向推理 embedding
 * 5. 前向推理 attention
 * 6. 前向推理 ffn
 */
std::vector<int64_t> Qwen2::forward(model::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) {
    // 获取上下文
    auto &ctx = static_cast<model::Qwen2Context &>(*this->_ctx);
    const auto &meta = model::qwen_meta(ctx);
    
    // 设置上下文推理类型
    ctx.runtime.is_prefill = is_prefill;  

    // 绑定上下文运行时信息
    this->bind_context_runtime(pack, block_ids, is_prefill);

    // 绑定 kv cache
    this->bind_kv_caches();

    // 前向推理 embedding
    model::Qwen2Embedding{}.forward(ctx);

    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        // 前向推理 attention
        model::Qwen2Attention(model::Qwen2Weights::attn(ctx, layer), layer).forward(ctx);
        // 前向推理 ffn
        model::Qwen2FFN(model::Qwen2Weights::ffn(ctx, layer)).forward(ctx);
    }

    // 获取最后一层的 logits
    const size_t batch_size = ctx.runtime.cut_idx.size() - 1;
    auto last_token_indices = ctx.workspace.cut_idx->slice(0, 1, batch_size + 1);

    // 前向推理 embedding
    ops::embedding(ctx.workspace.x_part, last_token_indices, ctx.workspace.x, -1);
    // 前向推理 rms norm
    ops::rms_norm(ctx.workspace.x_part_norm, ctx.workspace.x_part, ctx.out_norm_w, meta.epsilon);
    // 前向推理 linear
    ops::linear(ctx.workspace.logits, ctx.workspace.x_part_norm, ctx.out_embed, nullptr);

    // 采样 token
    return this->sample_from_logits(ctx.workspace.logits, pack);
}

/**
 * 绑定 kv cache
 * 1. 获取模型层数
 * 2. 绑定 kv cache
 */
void Qwen2::bind_kv_caches() {
    const size_t nlayer = model::qwen_meta(*this->_ctx).nlayer;
    for (size_t layer = 0; layer < nlayer; ++layer) {
        this->_ctx->k_caches[layer] = this->_k_cache[layer];
        this->_ctx->v_caches[layer] = this->_v_cache[layer];
    }
}

/**
 * 绑定上下文运行时
 * 1. 绑定上下文运行时信息
 */
void Qwen2::bind_context_runtime(model::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) {
    static_cast<model::Qwen2Context &>(*this->_ctx).bind(pack, block_ids, is_prefill);
}

/**
 * 从 logits 采样 token
 * 1. 获取模型参数
 * 2. 获取设备类型
 * 3. 获取样本数量
 * 4. 创建 topk 索引和值
 * 5. 调用 topk 算子
 * 6. 将 topk 索引和值拷贝到 CPU
 */
std::vector<int64_t> Qwen2::sample_from_logits(tensor_t logits, model::BatchPack &pack) {
    const auto &meta = model::qwen_meta(*this->_ctx);
    const llaisysDeviceType_t device = this->_device_type;
    const size_t batch_size = pack.tot_len.size();

    // 创建 topk 索引和值
    const size_t K = static_cast<size_t>(*std::max_element(pack.top_k.begin(), pack.top_k.end()));
    tensor_t top_idx = Tensor::create({batch_size, K}, LLAISYS_DTYPE_I64, device, this->_device_ids[0]);
    tensor_t top_val = Tensor::create({batch_size, K}, meta.dtype, device, this->_device_ids[0]);
    ops::topk(top_idx, top_val, logits, K);
    // 如果设备不是 CPU，则将 topk 索引和值拷贝到 CPU
    if (device != LLAISYS_DEVICE_CPU) {
        top_idx = top_idx->to(LLAISYS_DEVICE_CPU, 0);
        top_val = top_val->to(LLAISYS_DEVICE_CPU, 0);
    }
    // 将 topk 索引和值展平
    const std::vector<int64_t> all_idx = top_idx->reshape({batch_size * K})->to_vector<int64_t>();
    const std::vector<float> all_val = top_val->reshape({batch_size * K})->to_vector<float>();
    // 创建结果
    std::vector<int64_t> result(batch_size);
    // 遍历每个样本
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

/** 测试用：获取最后一层的 logits 
 * 1. 获取词汇表大小
 * 2. 获取最后一层的 logits
 */
tensor_t Qwen2::last_logits(size_t batch_size) const {
    const size_t voc = model::qwen_meta(*this->_ctx).voc;
    return static_cast<const model::Qwen2Context &>(*this->_ctx)
        .logits_buffer()
        ->slice(0, 0, batch_size)
        ->view({batch_size, voc});
}

} // namespace llaisys
