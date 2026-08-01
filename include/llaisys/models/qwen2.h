#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };

    struct LlaisysQwen2Model;
    struct LlaisysQwen2Request;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model);

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model);

    // 提交独立请求；返回的 handle 必须通过 llaisysQwen2RequestRelease 释放
    __export struct LlaisysQwen2Request *llaisysQwen2RequestSubmit(struct LlaisysQwen2Model *model,
        int64_t *token_ids, size_t ntoken, int64_t max_new_tokens, int top_k, float top_p, float temperature);
    // 阻塞等待当前请求生成下一个 token；取消返回 -2，错误返回 -1
    __export int64_t llaisysQwen2RequestAwait(struct LlaisysQwen2Request *request);
    // 取消当前请求，不影响其他请求
    __export void llaisysQwen2RequestAbort(struct LlaisysQwen2Request *request);
    // 释放请求 handle；未完成请求会先被取消
    __export void llaisysQwen2RequestRelease(struct LlaisysQwen2Request *request);

}
#endif // LLAISYS_MODELS_QWEN2_H
