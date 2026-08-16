#include "qwen2_context.hpp"

namespace llaisys::model {

// 分配内存
void Qwen2Context::allocate(llaisysDeviceType_t device, int device_id) {
    // 获取模型参数
    const ModelMeta &m = this->meta;
    const llaisysDataType_t dtype = m.dtype;
    const size_t hs = m.hs;
    const size_t nh = m.nh;
    const size_t nkvh = m.nkvh;
    const size_t dh = m.dh;
    const size_t di = m.di;
    const size_t voc = m.voc;
    const size_t batch_max_token_num = BATCH_MAX_TOKEN_NUM;
    const size_t batch_max_seq_num = BATCH_MAX_SEQ_NUM;
    const size_t max_block_num = MAX_BLOCK_NUM;

    // 创建中间张量
    _x_norm = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _q = Tensor::create({batch_max_token_num, nh * dh}, dtype, device, device_id);
    _k = Tensor::create({batch_max_token_num, nkvh * dh}, dtype, device, device_id);
    _v = Tensor::create({batch_max_token_num, nkvh * dh}, dtype, device, device_id);
    _q_rope = Tensor::create({batch_max_token_num, nh, dh}, dtype, device, device_id);
    _k_rope = Tensor::create({batch_max_token_num, nkvh, dh}, dtype, device, device_id);
    _attn_val = Tensor::create({batch_max_token_num, nh, dh}, dtype, device, device_id);
    _attn_out = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _x_attn = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _m_norm = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _gate = Tensor::create({batch_max_token_num, di}, dtype, device, device_id);
    _up = Tensor::create({batch_max_token_num, di}, dtype, device, device_id);
    _swiglu = Tensor::create({batch_max_token_num, di}, dtype, device, device_id);
    _down = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _x_mlp = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _dev_block_ids = Tensor::create({batch_max_seq_num * max_block_num}, LLAISYS_DTYPE_I64, device, device_id);
    _dev_cut_idx = Tensor::create({batch_max_seq_num + 1}, LLAISYS_DTYPE_I64, device, device_id);
    _dev_tot_len = Tensor::create({batch_max_seq_num}, LLAISYS_DTYPE_I64, device, device_id);
    _dev_pos_ids = Tensor::create({batch_max_token_num}, LLAISYS_DTYPE_I64, device, device_id);
    _x = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _x_part = Tensor::create({batch_max_seq_num, hs}, dtype, device, device_id);
    _x_part_norm = Tensor::create({batch_max_seq_num, hs}, dtype, device, device_id);
    _logits = Tensor::create({batch_max_seq_num, voc}, dtype, device, device_id);
    _attn_acc = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, dh}, LLAISYS_DTYPE_F32, device, device_id);
    _attn_sum = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, 1}, LLAISYS_DTYPE_F32, device, device_id);
    _attn_max = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, 1}, LLAISYS_DTYPE_F32, device, device_id);
}

// 绑定上下文运行时
void Qwen2Context::bind(const BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) {
    // 获取模型参数
    const ModelMeta &m = this->meta;
    const std::vector<int64_t> &token_ids = pack.token_ids;
    const std::vector<int64_t> &pos_ids = pack.pos_ids;
    const std::vector<int64_t> &cut_idx = pack.cut_idx;
    const std::vector<int64_t> &tot_len = pack.tot_len;
    const size_t max_seq_len = pack.max_seq_len;
    const size_t tot_seq_len = token_ids.size();
    const size_t batch_size = tot_len.size();
    const size_t max_block_num = MAX_BLOCK_NUM;
    const size_t tot_block_num = block_ids[(batch_size - 1) * MAX_BLOCK_NUM];

    const size_t hs = m.hs;
    const size_t nh = m.nh;
    const size_t nkvh = m.nkvh;
    const size_t dh = m.dh;
    const size_t di = m.di;
    const size_t voc = m.voc;

    // 设置上下文运行时
    runtime.token_ids = token_ids;
    runtime.pos_ids = pos_ids;
    runtime.cut_idx = cut_idx;
    runtime.tot_len = tot_len;
    runtime.max_seq_len = max_seq_len;
    runtime.tot_block_num = tot_block_num;
    runtime.is_prefill = is_prefill;

    // 绑定 workspace 视图
    workspace.x_norm = _x_norm->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    workspace.q = _q->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh * dh});
    workspace.k = _k->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh * dh});
    workspace.v = _v->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh * dh});
    workspace.q_rope = _q_rope->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh, dh});
    workspace.k_rope = _k_rope->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh, dh});
    workspace.attn_val = _attn_val->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh, dh});
    workspace.attn_out = _attn_out->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    workspace.x_attn = _x_attn->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    workspace.m_norm = _m_norm->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    workspace.gate = _gate->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    workspace.up = _up->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    workspace.swiglu = _swiglu->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    workspace.down = _down->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    workspace.x_mlp = _x_mlp->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    workspace.block_ids = _dev_block_ids->slice(0, 0, batch_size * max_block_num)->view({batch_size, max_block_num});
    workspace.cut_idx = _dev_cut_idx->slice(0, 0, batch_size + 1)->view({batch_size + 1});
    workspace.tot_len = _dev_tot_len->slice(0, 0, batch_size)->view({batch_size});
    workspace.pos_ids = _dev_pos_ids->slice(0, 0, tot_seq_len)->view({tot_seq_len});
    workspace.x = _x->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    workspace.x_part = _x_part->slice(0, 0, batch_size)->view({batch_size, hs});
    workspace.x_part_norm = _x_part_norm->slice(0, 0, batch_size)->view({batch_size, hs});
    workspace.logits = _logits->slice(0, 0, batch_size)->view({batch_size, voc});
    workspace.attn_acc = _attn_acc;
    workspace.attn_sum = _attn_sum;
    workspace.attn_max = _attn_max;

    workspace.block_ids->load(block_ids.data());
    workspace.cut_idx->load(cut_idx.data());
    workspace.tot_len->load(tot_len.data());
    workspace.pos_ids->load(pos_ids.data());
}

} // namespace llaisys::model
