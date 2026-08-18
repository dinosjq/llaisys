#pragma once

#include "../../../model/runtime/model_context.hpp"
#include "../../../config.hpp"

namespace llaisys::model {

// 当前 step 绑定到 Layer 的视图（不拥有大 buffer）
struct WorkspaceViews {
    tensor_t x, x_norm, q, k, v, q_rope, k_rope;
    tensor_t attn_val, attn_acc, attn_sum, attn_max, attn_out, x_attn;
    tensor_t m_norm, gate, up, swiglu, down, x_mlp;
    tensor_t block_ids, cut_idx, tot_len, pos_ids;
    tensor_t x_part, x_part_norm, logits;
};

// Qwen2 decoder workspace：拥有容量 buffer，每步 bind 出 WorkspaceViews
class Qwen2Context : public ModelContext {
public:
    Qwen2Context() = default;
    Qwen2Context(Qwen2Context &&) = default;
    Qwen2Context &operator=(Qwen2Context &&) = default;
    ~Qwen2Context() override = default;

    WorkspaceViews workspace{};

    // 分配内存
    void allocate(llaisysDeviceType_t device, int device_id);
    // 绑定上下文运行时
    void bind(const BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill);
    // 获取 logits 缓存
    tensor_t logits_buffer() const { return _logits; }

protected:
    tensor_t _x_norm, _q, _k, _v, _q_rope, _k_rope;
    tensor_t _attn_val, _attn_out, _x_attn, _m_norm;
    tensor_t _gate, _up, _swiglu, _down, _x_mlp;
    tensor_t _dev_block_ids, _dev_cut_idx, _dev_tot_len, _dev_pos_ids;
    tensor_t _x, _x_part, _x_part_norm, _logits;
    tensor_t _attn_acc, _attn_sum, _attn_max;
};

} // namespace llaisys::model
