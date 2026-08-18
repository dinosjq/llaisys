#ifndef LLAISYS_MODELS_MODEL_H
#define LLAISYS_MODELS_MODEL_H

#include "../tensor.h"

__C {
    struct LlaisysModel;
    struct LlaisysRequest;

    __export struct LlaisysModel *llaisysModelCreate(const char *arch, llaisysDeviceType_t device, int *device_ids,
                                                     int ndevice);

    __export void llaisysModelDestroy(struct LlaisysModel *model);

    // vals[i] is an opaque byte blob of length val_nbytes[i] (not necessarily NUL-terminated).
    __export int llaisysModelSetMeta(struct LlaisysModel *model, const char *const *keys, const void *const *vals,
                                     const size_t *val_nbytes, size_t n);

    __export int llaisysModelSetWeight(struct LlaisysModel *model, const char *name, llaisysTensor_t tensor);

    __export struct LlaisysRequest *llaisysModelRequestSubmit(struct LlaisysModel *model, int64_t *token_ids,
                                                             size_t ntoken, int64_t max_new_tokens, int top_k,
                                                             float top_p, float temperature);

    __export int64_t llaisysModelRequestAwait(struct LlaisysRequest *request);

    __export void llaisysModelRequestAbort(struct LlaisysRequest *request);

    __export void llaisysModelRequestRelease(struct LlaisysRequest *request);
}
#endif // LLAISYS_MODELS_MODEL_H
