#include "llaisys/models/model.h"

#include "../llaisys_tensor.hpp"
#include "../../models/qwen2/qwen2.hpp"
#include "../../models/llama/llama.hpp"

#include <memory>

struct LlaisysModel {
    std::shared_ptr<llaisys::model::Model> model;
};

struct LlaisysRequest {
    std::shared_ptr<llaisys::model::Model> model;
    llaisys::model::model_request_t request;
};

namespace {

llaisys::Qwen2Meta to_qwen2_meta(const LlaisysModelMeta *meta) {
    llaisys::Qwen2Meta out{};
    out.dtype = meta->dtype;
    out.nlayer = meta->nlayer;
    out.hs = meta->hs;
    out.nh = meta->nh;
    out.nkvh = meta->nkvh;
    out.dh = meta->dh;
    out.di = meta->di;
    out.maxseq = meta->maxseq;
    out.voc = meta->voc;
    out.epsilon = meta->epsilon;
    out.theta = meta->theta;
    out.end_token = meta->end_token;
    out.rope_scale = meta->rope_scale > 0.f ? meta->rope_scale : 1.f;
    return out;
}

} // namespace

__C {
    struct LlaisysModel *llaisysModelCreate(LlaisysModelArch arch, LlaisysModelMeta *meta, llaisysDeviceType_t device,
                                            int *device_ids, int ndevice) {
        if (!meta || !device_ids || ndevice <= 0) {
            return nullptr;
        }
        auto *handle = new LlaisysModel();
        const llaisys::Qwen2Meta cpp_meta = to_qwen2_meta(meta);
        try {
            if (arch == LLAISYS_MODEL_QWEN2) {
                handle->model = std::make_shared<llaisys::Qwen2>(cpp_meta, device, device_ids, ndevice);
            } else if (arch == LLAISYS_MODEL_LLAMA) {
                handle->model = std::make_shared<llaisys::Llama>(cpp_meta, device, device_ids, ndevice);
            } else {
                delete handle;
                return nullptr;
            }
        } catch (...) {
            delete handle;
            return nullptr;
        }
        return handle;
    }

    void llaisysModelDestroy(struct LlaisysModel *model) {
        if (!model) {
            return;
        }
        if (model->model) {
            model->model->stop();
        }
        delete model;
    }

    void llaisysModelSetWeight(struct LlaisysModel *model, LlaisysWeightRole role, size_t layer, llaisysTensor_t tensor) {
        if (!model || !model->model || !tensor) {
            return;
        }
        llaisys::model::Model::set_weight(model->model->ctx(), static_cast<llaisys::model::WeightRole>(role),
                                              layer, tensor->tensor);
    }

    struct LlaisysRequest *llaisysModelRequestSubmit(struct LlaisysModel *model, int64_t *token_ids, size_t ntoken,
                                                     int64_t max_new_tokens, int top_k, float top_p, float temperature) {
        if (!model || !model->model || !token_ids) {
            return nullptr;
        }
        llaisys::model::model_request_t submitted = nullptr;
        try {
            submitted = model->model->submit(token_ids, ntoken, max_new_tokens, top_k, top_p, temperature);
            auto *request = new LlaisysRequest();
            request->model = model->model;
            request->request = std::move(submitted);
            return request;
        } catch (...) {
            if (submitted) {
                model->model->release(submitted);
            }
            return nullptr;
        }
    }

    int64_t llaisysModelRequestAwait(struct LlaisysRequest *request) {
        if (!request || !request->model || !request->request) {
            return -1;
        }
        try {
            return request->model->await(request->request);
        } catch (...) {
            return -1;
        }
    }

    void llaisysModelRequestAbort(struct LlaisysRequest *request) {
        if (request && request->model && request->request) {
            request->model->abort(request->request);
        }
    }

    void llaisysModelRequestRelease(struct LlaisysRequest *request) {
        if (!request) {
            return;
        }
        if (request->model && request->request) {
            request->model->release(request->request);
        }
        delete request;
    }
}
