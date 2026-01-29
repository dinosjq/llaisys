#include "llaisys/models/qwen2.h"

#include "../llaisys_tensor.hpp"

#include "../../core/llaisys_core.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rearrange/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include "../../tensor/tensor.hpp"

#include <cmath>
#include <cstring>
#include <vector>

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    LlaisysQwen2Weights weights{};
    llaisysDeviceType_t device{LLAISYS_DEVICE_CPU};
    std::vector<int> device_ids;
    std::vector<llaisys::tensor_t> k_cache;
    std::vector<llaisys::tensor_t> v_cache;
    size_t cache_len{0};
    std::vector<int64_t> cached_tokens;
};

namespace {
static int get_device_id(const LlaisysQwen2Model *model) {
    if (!model || model->device_ids.empty()) {
        return 0;
    }
    return model->device_ids.front();
}

static llaisys::tensor_t to_tensor(llaisysTensor_t t) {
    return t ? t->tensor : nullptr;
}

// 初始化 kv_cache
static void init_kv_cache(LlaisysQwen2Model *model) {
    model->k_cache.resize(model->meta.nlayer);
    model->v_cache.resize(model->meta.nlayer);
    const int device_id = get_device_id(model);
    const size_t maxseq = model->meta.maxseq;
    const size_t nkvh = model->meta.nkvh;
    const size_t dh = model->meta.dh;
    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        model->k_cache[i] = llaisys::Tensor::create({maxseq, nkvh, dh}, model->meta.dtype, model->device, device_id);
        model->v_cache[i] = llaisys::Tensor::create({maxseq, nkvh, dh}, model->meta.dtype, model->device, device_id);
    }
    model->cache_len = 0;
    model->cached_tokens.clear();
}

static void reset_kv_cache(LlaisysQwen2Model *model) {
    model->cache_len = 0;
    model->cached_tokens.clear();
}

/**
 * 初始化权重tensor arrays
 */
static void init_weight_arrays(LlaisysQwen2Weights &w, size_t nlayer) {
    w.attn_norm_w = new llaisysTensor_t[nlayer]();
    w.attn_q_w = new llaisysTensor_t[nlayer]();
    w.attn_q_b = new llaisysTensor_t[nlayer]();
    w.attn_k_w = new llaisysTensor_t[nlayer]();
    w.attn_k_b = new llaisysTensor_t[nlayer]();
    w.attn_v_w = new llaisysTensor_t[nlayer]();
    w.attn_v_b = new llaisysTensor_t[nlayer]();
    w.attn_o_w = new llaisysTensor_t[nlayer]();
    w.mlp_norm_w = new llaisysTensor_t[nlayer]();
    w.mlp_gate_w = new llaisysTensor_t[nlayer]();
    w.mlp_up_w = new llaisysTensor_t[nlayer]();
    w.mlp_down_w = new llaisysTensor_t[nlayer]();
}

/**
 * 释放内存
 */
static void free_weight_arrays(LlaisysQwen2Weights &w) {
    delete[] w.attn_norm_w;
    delete[] w.attn_q_w;
    delete[] w.attn_q_b;
    delete[] w.attn_k_w;
    delete[] w.attn_k_b;
    delete[] w.attn_v_w;
    delete[] w.attn_v_b;
    delete[] w.attn_o_w;
    delete[] w.mlp_norm_w;
    delete[] w.mlp_gate_w;
    delete[] w.mlp_up_w;
    delete[] w.mlp_down_w;
    w.attn_norm_w = nullptr;
    w.attn_q_w = nullptr;
    w.attn_q_b = nullptr;
    w.attn_k_w = nullptr;
    w.attn_k_b = nullptr;
    w.attn_v_w = nullptr;
    w.attn_v_b = nullptr;
    w.attn_o_w = nullptr;
    w.mlp_norm_w = nullptr;
    w.mlp_gate_w = nullptr;
    w.mlp_up_w = nullptr;
    w.mlp_down_w = nullptr;
}
} // namespace

__C {
    /**
     * 根据特征meta信息和设备device信息, 创建初始化模型
     */
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        if (!meta || meta->nlayer == 0) {
            return nullptr;
        }
        if (meta->maxseq == 0) {
            return nullptr;
        }
        auto *model = new LlaisysQwen2Model();
        model->meta = *meta;
        model->device = device;
        if (device_ids && ndevice > 0) {
            model->device_ids.assign(device_ids, device_ids + ndevice);
        }
        init_weight_arrays(model->weights, meta->nlayer);
        init_kv_cache(model);
        return model;
    }

    /**
     * 模型销毁
     */
    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (!model) {
            return;
        }
        free_weight_arrays(model->weights);
        delete model;
    }

    /**
     * 返回模型权重
     */
    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        if (!model) {
            return nullptr;
        }
        return &model->weights;
    }

    /**
     * 执行模型前向推导
     */
    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken) {
        if (!model) {
            return -1;
        }
        if (!token_ids || ntoken == 0) {
            return model->meta.end_token;
        }
        const auto &meta = model->meta;
        const auto &w = model->weights;
        if (ntoken > meta.maxseq) {
            return model->meta.end_token;
        }

        size_t start = 0;
        if (model->cache_len > 0 && ntoken >= model->cache_len) {
            bool prefix_match = true;
            for (size_t i = 0; i < model->cache_len; ++i) {
                if (model->cached_tokens[i] != token_ids[i]) {
                    prefix_match = false;
                    break;
                }
            }
            if (prefix_match) {
                start = model->cache_len;
            } else {
                reset_kv_cache(model);
            }
        }

        const size_t seqlen = ntoken - start;
        if (seqlen == 0) {
            return token_ids[ntoken - 1];
        }
        const size_t hs = meta.hs;
        const size_t nh = meta.nh;
        const size_t nkvh = meta.nkvh;
        const size_t dh = meta.dh;
        const size_t di = meta.di;
        const size_t voc = meta.voc;

        const int device_id = get_device_id(model);
        const llaisysDeviceType_t device = model->device;

        // input ids tensor
        llaisys::tensor_t input_ids = llaisys::Tensor::create({seqlen}, LLAISYS_DTYPE_I64, device, device_id);
        input_ids->load(token_ids + start);

        // position ids tensor
        std::vector<int64_t> pos_ids_vec(seqlen);
        for (size_t i = 0; i < seqlen; ++i) {
            pos_ids_vec[i] = static_cast<int64_t>(start + i);
        }
        llaisys::tensor_t pos_ids = llaisys::Tensor::create({seqlen}, LLAISYS_DTYPE_I64, device, device_id);
        pos_ids->load(pos_ids_vec.data());

        // embedding
        llaisys::tensor_t x = llaisys::Tensor::create({seqlen, hs}, meta.dtype, device, device_id);
        llaisys::ops::embedding(x, input_ids, to_tensor(w.in_embed));

        const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

        for (size_t layer = 0; layer < meta.nlayer; ++layer) {
            // 注意力机制 Attention block
            llaisys::tensor_t x_norm = llaisys::Tensor::create({seqlen, hs}, meta.dtype, device, device_id);
            llaisys::ops::rms_norm(x_norm, x, to_tensor(w.attn_norm_w[layer]), meta.epsilon);

            // 初始化 Q K V
            llaisys::tensor_t q = llaisys::Tensor::create({seqlen, nh * dh}, meta.dtype, device, device_id);
            llaisys::ops::linear(q, x_norm, to_tensor(w.attn_q_w[layer]), to_tensor(w.attn_q_b[layer]));

            llaisys::tensor_t k = llaisys::Tensor::create({seqlen, nkvh * dh}, meta.dtype, device, device_id);
            llaisys::ops::linear(k, x_norm, to_tensor(w.attn_k_w[layer]), to_tensor(w.attn_k_b[layer]));

            llaisys::tensor_t v = llaisys::Tensor::create({seqlen, nkvh * dh}, meta.dtype, device, device_id);
            llaisys::ops::linear(v, x_norm, to_tensor(w.attn_v_w[layer]), to_tensor(w.attn_v_b[layer]));

            llaisys::tensor_t q_view = q->view({seqlen, nh, dh});
            llaisys::tensor_t k_view = k->view({seqlen, nkvh, dh});
            llaisys::tensor_t v_view = v->view({seqlen, nkvh, dh});

            // 旋转位置编码 rope
            llaisys::tensor_t q_rope = llaisys::Tensor::create({seqlen, nh, dh}, meta.dtype, device, device_id);
            llaisys::tensor_t k_rope = llaisys::Tensor::create({seqlen, nkvh, dh}, meta.dtype, device, device_id);
            llaisys::ops::rope(q_rope, q_view, pos_ids, meta.theta);
            llaisys::ops::rope(k_rope, k_view, pos_ids, meta.theta);

            // Update KV cache
            llaisys::tensor_t k_slice = model->k_cache[layer]->slice(0, start, start + seqlen);
            llaisys::tensor_t v_slice = model->v_cache[layer]->slice(0, start, start + seqlen);
            llaisys::ops::rearrange(k_slice, k_rope);
            llaisys::ops::rearrange(v_slice, v_view);

            const size_t total_len = start + seqlen;
            llaisys::tensor_t k_total = model->k_cache[layer]->slice(0, 0, total_len);
            llaisys::tensor_t v_total = model->v_cache[layer]->slice(0, 0, total_len);

            // 自注意力 self_attention
            llaisys::tensor_t attn_val = llaisys::Tensor::create({seqlen, nh, dh}, meta.dtype, device, device_id);
            llaisys::ops::self_attention(attn_val, q_rope, k_total, v_total, scale);

            llaisys::tensor_t attn_merge = attn_val->view({seqlen, nh * dh});
            llaisys::tensor_t attn_out = llaisys::Tensor::create({seqlen, hs}, meta.dtype, device, device_id);
            llaisys::ops::linear(attn_out, attn_merge, to_tensor(w.attn_o_w[layer]), nullptr);

            llaisys::tensor_t x_attn = llaisys::Tensor::create({seqlen, hs}, meta.dtype, device, device_id);
            llaisys::ops::add(x_attn, x, attn_out);
            x = x_attn;

            // 多层感知机 MLP block
            llaisys::tensor_t m_norm = llaisys::Tensor::create({seqlen, hs}, meta.dtype, device, device_id);
            llaisys::ops::rms_norm(m_norm, x, to_tensor(w.mlp_norm_w[layer]), meta.epsilon);

            llaisys::tensor_t gate = llaisys::Tensor::create({seqlen, di}, meta.dtype, device, device_id);
            llaisys::ops::linear(gate, m_norm, to_tensor(w.mlp_gate_w[layer]), nullptr);

            llaisys::tensor_t up = llaisys::Tensor::create({seqlen, di}, meta.dtype, device, device_id);
            llaisys::ops::linear(up, m_norm, to_tensor(w.mlp_up_w[layer]), nullptr);

            llaisys::tensor_t swiglu = llaisys::Tensor::create({seqlen, di}, meta.dtype, device, device_id);
            llaisys::ops::swiglu(swiglu, gate, up);

            llaisys::tensor_t down = llaisys::Tensor::create({seqlen, hs}, meta.dtype, device, device_id);
            llaisys::ops::linear(down, swiglu, to_tensor(w.mlp_down_w[layer]), nullptr);

            llaisys::tensor_t x_mlp = llaisys::Tensor::create({seqlen, hs}, meta.dtype, device, device_id);
            llaisys::ops::add(x_mlp, x, down);
            x = x_mlp;
        }

        // Final norm and logits
        llaisys::tensor_t x_norm = llaisys::Tensor::create({seqlen, hs}, meta.dtype, device, device_id);
        llaisys::ops::rms_norm(x_norm, x, to_tensor(w.out_norm_w), meta.epsilon);

        llaisys::tensor_t logits = llaisys::Tensor::create({seqlen, voc}, meta.dtype, device, device_id);
        llaisys::ops::linear(logits, x_norm, to_tensor(w.out_embed), nullptr);

        llaisys::tensor_t last = logits->slice(0, seqlen - 1, seqlen)->view({voc});
        llaisys::tensor_t max_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
        llaisys::tensor_t max_val = llaisys::Tensor::create({1}, meta.dtype, device, device_id);
        llaisys::ops::argmax(max_idx, max_val, last);

        int64_t result = model->meta.end_token;
        if (device == LLAISYS_DEVICE_CPU) {
            result = reinterpret_cast<int64_t *>(max_idx->data())[0];
        } else {
            llaisys::tensor_t host_idx = llaisys::Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
            llaisys::core::context().setDevice(device, device_id);
            llaisys::core::context().runtime().api()->memcpy_sync(
                host_idx->data(),
                max_idx->data(),
                sizeof(int64_t),
                LLAISYS_MEMCPY_D2H);
            result = reinterpret_cast<int64_t *>(host_idx->data())[0];
        }

        model->cached_tokens.assign(token_ids, token_ids + ntoken);
        model->cache_len = ntoken;
        return result;
    }
}
