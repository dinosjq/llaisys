#include "llaisys/models/model.h"

#include "../llaisys_tensor.hpp"
#include "../../model/registry/model_registry.hpp"
#include "../../models/qwen2/qwen2.hpp"
#include "../../models/llama/llama.hpp"
#include "../../utils/byte_map.hpp"

#include <memory>
#include <mutex>

struct LlaisysModel {
    std::shared_ptr<llaisys::model::Model> model;
};

struct LlaisysRequest {
    std::shared_ptr<llaisys::model::Model> model;
    llaisys::model::model_request_t request;
};

namespace {

std::once_flag g_register_once;

void register_builtin_models() {
    llaisys::model::ModelRegistry::register_model(
        "qwen2", [](llaisysDeviceType_t device, int *device_ids, int ndevice) {
            return std::shared_ptr<llaisys::model::Model>(
                std::make_shared<llaisys::Qwen2>(device, device_ids, ndevice));
        });
    llaisys::model::ModelRegistry::register_model(
        "llama", [](llaisysDeviceType_t device, int *device_ids, int ndevice) {
            return std::shared_ptr<llaisys::model::Model>(
                std::make_shared<llaisys::Llama>(device, device_ids, ndevice));
        });
}

} // namespace

__C {
    struct LlaisysModel *llaisysModelCreate(const char *arch, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        if (!arch || !device_ids || ndevice <= 0) {
            return nullptr;
        }
        std::call_once(g_register_once, register_builtin_models);
        auto *handle = new LlaisysModel();
        try {
            handle->model = llaisys::model::ModelRegistry::create(arch, device, device_ids, ndevice);
            if (!handle->model) {
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

    int llaisysModelSetMeta(struct LlaisysModel *model, const char *const *keys, const void *const *vals,
                            const size_t *val_nbytes, size_t n) {
        if (!model || !model->model) {
            return -1;
        }
        try {
            auto kv = llaisys::utils::make_byte_map(keys, vals, val_nbytes, n);
            model->model->set_meta(kv);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int llaisysModelSetWeight(struct LlaisysModel *model, const char *name, llaisysTensor_t tensor) {
        if (!model || !model->model || !name || !tensor) {
            return -1;
        }
        try {
            model->model->set_weight(name, tensor->tensor);
            return 0;
        } catch (...) {
            return -1;
        }
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
        if (!request || !request->model || !request->request) {
            return;
        }
        request->model->abort(request->request);
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
