#pragma once

#include "../../../model/runtime/model_context.hpp"
#include "../../../config.hpp"
#include "../../../engine/engine.hpp"
#include "../../../tensor/tensor.hpp"
#include "../meta/qwen2_meta.hpp"

namespace llaisys::model {

struct Current {
    tensor_t x, x_norm, q, k, v, q_rope, k_rope;
    tensor_t attn_val, attn_acc, attn_sum, attn_max, attn_out, x_attn;
    tensor_t m_norm, gate, up, swiglu, down, x_mlp;
    tensor_t block_ids, cut_idx, tot_len, pos_ids;
    tensor_t x_part, x_part_norm, logits;
    tensor_t token_ids;
};

struct Buffer {
    tensor_t _x_norm, _q, _k, _v, _q_rope, _k_rope;
    tensor_t _attn_val, _attn_out, _x_attn, _m_norm;
    tensor_t _gate, _up, _swiglu, _down, _x_mlp;
    tensor_t _dev_block_ids, _dev_cut_idx, _dev_tot_len, _dev_pos_ids;
    tensor_t _x, _x_part, _x_part_norm, _logits;
    tensor_t _attn_acc, _attn_sum, _attn_max;
    tensor_t _token_ids;
};

// 每步的运行时标量（bind 时填充）
struct RuntimeInfo {
    size_t tot_task_num = 0;
    bool is_prefill = true;
};

class Qwen2Context : public ModelContext {
public:
    Qwen2Context() = default;
    ~Qwen2Context() override = default;
    Qwen2Context(Qwen2Context &&) = default;
    Qwen2Context &operator=(Qwen2Context &&) = default;

    // 分层 KV cache（allocate 时分配）
    std::vector<tensor_t> k_caches;
    std::vector<tensor_t> v_caches;
    // 模型配置（bind/allocate 时设置，供 layer 读取）
    Qwen2Meta meta{};

    // 分配内存
    void allocate(llaisysDeviceType_t device, int device_id, Qwen2Meta meta);
    // 绑定上下文
    void bind(const engine::BatchInput &input, Qwen2Meta meta);
    // 运行时
    Current &current() { return this->_cur; };
    const Current &current() const { return this->_cur; };
    RuntimeInfo &runtime() { return this->_runtime; };
    const RuntimeInfo &runtime() const { return this->_runtime; };

protected:
    Current _cur{};
    Buffer _buf{};
    RuntimeInfo _runtime{};

};

using qwen_context_t = std::unique_ptr<Qwen2Context>;

} // namespace llaisys::model
