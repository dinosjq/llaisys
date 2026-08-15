#ifndef LLAISYS_MODELS_MODEL_H
#define LLAISYS_MODELS_MODEL_H

#include "../tensor.h"

__C {
    typedef enum LlaisysModelArch {
        LLAISYS_MODEL_QWEN2 = 0,
        LLAISYS_MODEL_LLAMA = 1,
    } LlaisysModelArch;

    /* Must match llaisys::core::WeightRole order/values. */
    typedef enum LlaisysWeightRole {
        LLAISYS_WEIGHT_IN_EMBED = 0,
        LLAISYS_WEIGHT_OUT_EMBED = 1,
        LLAISYS_WEIGHT_OUT_NORM = 2,
        LLAISYS_WEIGHT_ATTN_NORM = 3,
        LLAISYS_WEIGHT_ATTN_Q_W = 4,
        LLAISYS_WEIGHT_ATTN_Q_B = 5,
        LLAISYS_WEIGHT_ATTN_K_W = 6,
        LLAISYS_WEIGHT_ATTN_K_B = 7,
        LLAISYS_WEIGHT_ATTN_V_W = 8,
        LLAISYS_WEIGHT_ATTN_V_B = 9,
        LLAISYS_WEIGHT_ATTN_O_W = 10,
        LLAISYS_WEIGHT_MLP_NORM = 11,
        LLAISYS_WEIGHT_MLP_GATE_W = 12,
        LLAISYS_WEIGHT_MLP_UP_W = 13,
        LLAISYS_WEIGHT_MLP_DOWN_W = 14,
    } LlaisysWeightRole;

    struct LlaisysModelMeta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };

    struct LlaisysModel;
    struct LlaisysRequest;

    __export struct LlaisysModel *llaisysModelCreate(LlaisysModelArch arch, LlaisysModelMeta *meta,
                                                     llaisysDeviceType_t device, int *device_ids, int ndevice);

    __export void llaisysModelDestroy(struct LlaisysModel *model);

    __export void llaisysModelSetWeight(struct LlaisysModel *model, LlaisysWeightRole role, size_t layer,
                                        llaisysTensor_t tensor);

    __export struct LlaisysRequest *llaisysModelRequestSubmit(struct LlaisysModel *model, int64_t *token_ids,
                                                             size_t ntoken, int64_t max_new_tokens, int top_k,
                                                             float top_p, float temperature);

    __export int64_t llaisysModelRequestAwait(struct LlaisysRequest *request);

    __export void llaisysModelRequestAbort(struct LlaisysRequest *request);

    __export void llaisysModelRequestRelease(struct LlaisysRequest *request);
}
#endif // LLAISYS_MODELS_MODEL_H
