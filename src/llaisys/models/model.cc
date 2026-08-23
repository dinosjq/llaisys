#include "llaisys/models/model.h"

#include "../llaisys_tensor.hpp"
#include "../../engine/registry/model_registry.hpp"
#include "../../engine/engine.hpp"
#include "../../models/qwen2/qwen2.hpp"
#include "../../models/llama/llama.hpp"
#include "../../utils/byte_map.hpp"

#include <memory>
#include <mutex>

struct LlaisysModel {
    std::shared_ptr<llaisys::engine::Engine> engine;
};

struct LlaisysRequest {
    std::shared_ptr<llaisys::engine::Engine> engine;
    llaisys::engine::request_t request;
};

namespace {

std::once_flag g_register_once;

void register_builtin_models() {
    llaisys::model::ModelRegistry::register_model(
        "qwen2", [](llaisysDeviceType_t device, int device_id) {
            return std::make_unique<llaisys::model::Qwen2>(device, device_id);
        });
    llaisys::model::ModelRegistry::register_model(
        "llama", [](llaisysDeviceType_t device, int device_id) {
            return std::make_unique<llaisys::model::Llama>(device, device_id);
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
            auto model = llaisys::model::ModelRegistry::create(arch, device, device_ids[0]);
            if (!model) {
                delete handle;
                return nullptr;
            }
            handle->engine = std::make_shared<llaisys::engine::Engine>(device, device_ids[0], std::move(model));
            if (!handle->engine) {
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
        // Engine 析构会 stop + join 工作线程
        delete model;
    }

    int llaisysModelSetMeta(struct LlaisysModel *model, const char *const *keys, const void *const *vals,
                            const size_t *val_nbytes, size_t n) {
        if (!model || !model->engine) {
            return -1;
        }
        try {
            auto kv = llaisys::utils::make_byte_map(keys, vals, val_nbytes, n);
            model->engine->model().set_meta(kv);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    int llaisysModelSetWeight(struct LlaisysModel *model, const char *name, llaisysTensor_t tensor) {
        if (!model || !model->engine || !name || !tensor) {
            return -1;
        }
        try {
            model->engine->model().set_weight(name, tensor->tensor);
            return 0;
        } catch (...) {
            return -1;
        }
    }

    struct LlaisysRequest *llaisysModelRequestSubmit(struct LlaisysModel *model, int64_t *token_ids, size_t ntoken,
                                                     int64_t max_new_tokens, int top_k, float top_p, float temperature) {
        if (!model || !model->engine || !token_ids) {
            return nullptr;
        }
        try {
            auto *request = new LlaisysRequest();
            request->engine = model->engine;
            request->request = model->engine->submit(token_ids, ntoken, max_new_tokens, top_k, top_p, temperature);
            if (!request->request) {
                delete request;
                return nullptr;
            }
            return request;
        } catch (...) {
            return nullptr;
        }
    }

    int64_t llaisysModelRequestAwait(struct LlaisysRequest *request) {
        if (!request || !request->engine || !request->request) {
            return -1;
        }
        try {
            return request->engine->await(request->request);
        } catch (...) {
            return -1;
        }
    }

    void llaisysModelRequestAbort(struct LlaisysRequest *request) {
        if (!request || !request->engine || !request->request) {
            return;
        }
        request->engine->abort(request->request);
    }

    void llaisysModelRequestRelease(struct LlaisysRequest *request) {
        if (!request) {
            return;
        }
        if (request->engine && request->request) {
            request->engine->release(request->request);
        }
        delete request;
    }
}
