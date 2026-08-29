#include "qwen2_context.hpp"

namespace llaisys::model {
void Qwen2Context::allocate(llaisysDeviceType_t device, int device_id, Qwen2Meta meta) {
    // 获取模型参数
    const llaisysDataType_t dtype = meta.dtype;
    const size_t nlayer = meta.nlayer;
    const size_t hs = meta.hs;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t di = meta.di;
    const size_t voc = meta.voc;
    const size_t batch_max_token_num = BATCH_MAX_TOKEN_NUM;
    const size_t batch_max_seq_num = BATCH_MAX_SEQ_NUM;
    const size_t max_block_num = MAX_BLOCK_NUM;
    const size_t block_size = KV_CACHE_BLOCK_SIZE; // 每块 token 数
    const size_t block_num = KV_CACHE_BLOCK_NUM;   // 块总数

    // 保存模型配置
    this->meta = std::move(meta);

    // 创建中间张量
    _buf._x_norm = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _buf._q = Tensor::create({batch_max_token_num, nh * dh}, dtype, device, device_id);
    _buf._k = Tensor::create({batch_max_token_num, nkvh * dh}, dtype, device, device_id);
    _buf._v = Tensor::create({batch_max_token_num, nkvh * dh}, dtype, device, device_id);
    _buf._q_rope = Tensor::create({batch_max_token_num, nh, dh}, dtype, device, device_id);
    _buf._k_rope = Tensor::create({batch_max_token_num, nkvh, dh}, dtype, device, device_id);
    _buf._attn_val = Tensor::create({batch_max_token_num, nh, dh}, dtype, device, device_id);
    _buf._attn_out = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _buf._x_attn = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _buf._m_norm = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _buf._gate = Tensor::create({batch_max_token_num, di}, dtype, device, device_id);
    _buf._up = Tensor::create({batch_max_token_num, di}, dtype, device, device_id);
    _buf._swiglu = Tensor::create({batch_max_token_num, di}, dtype, device, device_id);
    _buf._down = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _buf._x_mlp = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _buf._dev_block_ids = Tensor::create({batch_max_seq_num * max_block_num}, LLAISYS_DTYPE_I64, device, device_id);
    _buf._dev_cut_idx = Tensor::create({batch_max_seq_num + 1}, LLAISYS_DTYPE_I64, device, device_id);
    _buf._dev_tot_len = Tensor::create({batch_max_seq_num}, LLAISYS_DTYPE_I64, device, device_id);
    _buf._dev_pos_ids = Tensor::create({batch_max_token_num}, LLAISYS_DTYPE_I64, device, device_id);
    _buf._x = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _buf._x_part = Tensor::create({batch_max_seq_num, hs}, dtype, device, device_id);
    _buf._x_part_norm = Tensor::create({batch_max_seq_num, hs}, dtype, device, device_id);
    _buf._logits = Tensor::create({batch_max_seq_num, voc}, dtype, device, device_id);
    _buf._attn_acc = Tensor::create({batch_max_seq_num * max_block_num, nh, dh}, LLAISYS_DTYPE_F32, device, device_id);
    _buf._attn_sum = Tensor::create({batch_max_seq_num * max_block_num, nh, 1}, LLAISYS_DTYPE_F32, device, device_id);
    _buf._attn_max = Tensor::create({batch_max_seq_num * max_block_num, nh, 1}, LLAISYS_DTYPE_F32, device, device_id);
    _buf._token_ids = Tensor::create({BATCH_MAX_TOKEN_NUM}, LLAISYS_DTYPE_I64, device, device_id);

    // 激活量化中间张量
    _buf._x_norm_q = Tensor::create({batch_max_token_num, hs}, LLAISYS_DTYPE_I8, device, device_id);
    _buf._x_norm_a_scale = Tensor::create({batch_max_token_num}, LLAISYS_DTYPE_F32, device, device_id);
    _buf._attn_val_q = Tensor::create({batch_max_token_num, hs}, LLAISYS_DTYPE_I8, device, device_id);
    _buf._attn_val_a_scale = Tensor::create({batch_max_token_num}, LLAISYS_DTYPE_F32, device, device_id);
    _buf._m_norm_q = Tensor::create({batch_max_token_num, hs}, LLAISYS_DTYPE_I8, device, device_id);
    _buf._m_norm_a_scale = Tensor::create({batch_max_token_num}, LLAISYS_DTYPE_F32, device, device_id);
    _buf._swiglu_q = Tensor::create({batch_max_token_num, di}, LLAISYS_DTYPE_I8, device, device_id);
    _buf._swiglu_a_scale = Tensor::create({batch_max_token_num}, LLAISYS_DTYPE_F32, device, device_id);
    _buf._x_part_norm_q = Tensor::create({batch_max_seq_num, hs}, LLAISYS_DTYPE_I8, device, device_id);
    _buf._x_part_norm_a_scale = Tensor::create({batch_max_seq_num}, LLAISYS_DTYPE_F32, device, device_id);

    // 分层 KV cache (quantize -> I8 cache + F32 scale，统一 INT8 量化)
    const bool quant_kv = meta.quantize;
    const llaisysDataType_t kv_dtype = quant_kv ? LLAISYS_DTYPE_I8 : dtype;
    const size_t cache_numel = block_num * block_size * nkvh * dh;
    this->k_caches = std::vector<tensor_t>(nlayer, nullptr);
    this->v_caches = std::vector<tensor_t>(nlayer, nullptr);
    this->k_scales = std::vector<tensor_t>(nlayer, nullptr);
    this->v_scales = std::vector<tensor_t>(nlayer, nullptr);
    tensor_t cache = Tensor::create({2, nlayer, cache_numel}, kv_dtype, device, device_id);
    tensor_t k_cache = cache->slice(0, 0, 1)->reshape({nlayer, cache_numel});
    tensor_t v_cache = cache->slice(0, 1, 2)->reshape({nlayer, cache_numel});
    for (size_t i = 0; i < nlayer; ++i) {
        this->k_caches[i] = k_cache->slice(0, i, i + 1)->reshape({block_num, block_size, nkvh, dh});
        this->v_caches[i] = v_cache->slice(0, i, i + 1)->reshape({block_num, block_size, nkvh, dh});
    }
    if (quant_kv) {
        const size_t scale_numel = block_num * block_size * nkvh;
        tensor_t k_scale = Tensor::create({nlayer, scale_numel}, LLAISYS_DTYPE_F32, device, device_id);
        tensor_t v_scale = Tensor::create({nlayer, scale_numel}, LLAISYS_DTYPE_F32, device, device_id);
        for (size_t i = 0; i < nlayer; ++i) {
            this->k_scales[i] = k_scale->slice(0, i, i + 1)->reshape({block_num, block_size, nkvh});
            this->v_scales[i] = v_scale->slice(0, i, i + 1)->reshape({block_num, block_size, nkvh});
        }
    }
}

// 绑定上下文运行时
void Qwen2Context::bind(const engine::BatchInput &input, Qwen2Meta meta) {
    auto &pack = std::get<engine::BatchPack>(input);
    // 获取批次信息
    const std::vector<int64_t> &token_ids = pack.token_ids;
    const std::vector<int64_t> &pos_ids = pack.pos_ids;
    const std::vector<int64_t> &cut_idx = pack.cut_idx;
    const std::vector<int64_t> &tot_len = pack.tot_len;
    const std::vector<int64_t> &block_ids = pack.block_ids;
    const size_t tot_task_num = pack.tot_task_num;
    const size_t batch_size = tot_len.size();
    const size_t tot_seq_len = token_ids.size();
    const size_t max_block_num = MAX_BLOCK_NUM;
    // 模型参数
    const size_t hs = meta.hs;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t di = meta.di;
    const size_t voc = meta.voc;

    // 绑定中间张量
    _cur.token_ids = _buf._token_ids->slice(0, 0, tot_seq_len)->view({tot_seq_len});
    _cur.x_norm = _buf._x_norm->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    _cur.q = _buf._q->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh * dh});
    _cur.k = _buf._k->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh * dh});
    _cur.v = _buf._v->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh * dh});
    _cur.q_rope = _buf._q_rope->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh, dh});
    _cur.k_rope = _buf._k_rope->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh, dh});
    _cur.attn_val = _buf._attn_val->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh, dh});
    _cur.attn_out = _buf._attn_out->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    _cur.x_attn = _buf._x_attn->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    _cur.m_norm = _buf._m_norm->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    _cur.gate = _buf._gate->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    _cur.up = _buf._up->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    _cur.swiglu = _buf._swiglu->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    _cur.down = _buf._down->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    _cur.x_mlp = _buf._x_mlp->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    _cur.block_ids = _buf._dev_block_ids->slice(0, 0, batch_size * max_block_num)->view({batch_size, max_block_num});
    _cur.cut_idx = _buf._dev_cut_idx->slice(0, 0, batch_size + 1)->view({batch_size + 1});
    _cur.tot_len = _buf._dev_tot_len->slice(0, 0, batch_size)->view({batch_size});
    _cur.pos_ids = _buf._dev_pos_ids->slice(0, 0, tot_seq_len)->view({tot_seq_len});
    _cur.x = _buf._x->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    _cur.x_part = _buf._x_part->slice(0, 0, batch_size)->view({batch_size, hs});
    _cur.x_part_norm = _buf._x_part_norm->slice(0, 0, batch_size)->view({batch_size, hs});
    _cur.logits = _buf._logits->slice(0, 0, batch_size)->view({batch_size, voc});
    _cur.attn_acc = _buf._attn_acc;
    _cur.attn_sum = _buf._attn_sum;
    _cur.attn_max = _buf._attn_max;
    _cur.x_norm_q = _buf._x_norm_q->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    _cur.x_norm_a_scale = _buf._x_norm_a_scale->slice(0, 0, tot_seq_len)->view({tot_seq_len});
    _cur.attn_val_q = _buf._attn_val_q->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    _cur.attn_val_a_scale = _buf._attn_val_a_scale->slice(0, 0, tot_seq_len)->view({tot_seq_len});
    _cur.m_norm_q = _buf._m_norm_q->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    _cur.m_norm_a_scale = _buf._m_norm_a_scale->slice(0, 0, tot_seq_len)->view({tot_seq_len});
    _cur.swiglu_q = _buf._swiglu_q->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    _cur.swiglu_a_scale = _buf._swiglu_a_scale->slice(0, 0, tot_seq_len)->view({tot_seq_len});
    _cur.x_part_norm_q = _buf._x_part_norm_q->slice(0, 0, batch_size)->view({batch_size, hs});
    _cur.x_part_norm_a_scale = _buf._x_part_norm_a_scale->slice(0, 0, batch_size)->view({batch_size});

    _cur.block_ids->load(block_ids.data());
    _cur.cut_idx->load(cut_idx.data());
    _cur.tot_len->load(tot_len.data());
    _cur.pos_ids->load(pos_ids.data());
    _cur.token_ids->load(token_ids.data());

    // 保存运行时标量 (layer 通过 ctx.runtime() 读取)
    this->_runtime.tot_task_num = tot_task_num;
    this->_runtime.is_prefill = pack.is_prefill;
    this->meta = std::move(meta);
}

} // namespace llaisys::model
