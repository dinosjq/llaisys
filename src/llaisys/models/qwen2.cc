#include "llaisys/models/qwen2.h"

#include "../llaisys_tensor.hpp"
#include "../../models/qwen2.hpp"

struct LlaisysQwen2Model {
    llaisys::Qwen2_t qwen2;
};
 
struct LlaisysQwen2Request {
    llaisys::Qwen2_t qwen2;
    llaisys::qwen2_request_t request;
};

__C {
    /** 初始化模型 */
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        if (!meta) return nullptr;
        auto *model = new LlaisysQwen2Model();
        model->qwen2 = std::make_shared<llaisys::Qwen2>(*reinterpret_cast<llaisys::Qwen2Meta *>(meta), device, device_ids, ndevice);
        return model;
    }

    /** 模型销毁 */
    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (!model) return;
        model->qwen2->stop();
        delete model;
    }

    /** 模型权重指针 */
    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        if (!model) return nullptr;
        return &model->qwen2->weights();
    }

    struct LlaisysQwen2Request *llaisysQwen2RequestSubmit(struct LlaisysQwen2Model *model, int64_t *token_ids,
                                                            size_t ntoken, int64_t max_new_tokens, int top_k,
                                                            float top_p, float temperature) {
        if (!model || !model->qwen2) return nullptr;
        llaisys::qwen2_request_t submitted;
        try {
            submitted = model->qwen2->submit(token_ids, ntoken, max_new_tokens, top_k, top_p, temperature);
            auto request = std::make_unique<LlaisysQwen2Request>();
            request->qwen2 = model->qwen2;
            request->request = std::move(submitted);
            return request.release();
        } catch (...) {
            if (submitted) model->qwen2->release(submitted);
            return nullptr;
        }
    }

    int64_t llaisysQwen2RequestAwait(struct LlaisysQwen2Request *request) {
        if (!request || !request->qwen2 || !request->request) return -1;
        try {
            return request->qwen2->await(request->request);
        } catch (...) {
            return -1;
        }
    }

    void llaisysQwen2RequestAbort(struct LlaisysQwen2Request *request) {
        if (request && request->qwen2 && request->request) request->qwen2->abort(request->request);
    }

    void llaisysQwen2RequestRelease(struct LlaisysQwen2Request *request) {
        if (!request) return;
        if (request->qwen2 && request->request) request->qwen2->release(request->request);
        delete request;
    }

}
