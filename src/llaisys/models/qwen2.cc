#include "llaisys/models/qwen2.h"

#include "../llaisys_tensor.hpp"
#include "../../models/qwen2.hpp"
#include "../../kv_cache/block_manager.hpp"

struct LlaisysQwen2Model {
    llaisys::Qwen2_t qwen2;
};
 
__C {
    /** 初始化模型 */
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        ASSERT(device_ids != nullptr, "llaisysQwen2ModelCreate: device_ids eq nullptr");
        if (!meta) return nullptr;
        size_t token_dim = (meta->nlayer) * (meta->nkvh) * (meta->dh);
        llaisys::BlockManager_t manager = llaisys::BlockManager::create(token_dim, 32, 8, meta->dtype, device, device_ids[0]);
        auto *model = new LlaisysQwen2Model();
        model->qwen2 = llaisys::Qwen2::model(*reinterpret_cast<llaisys::Qwen2Meta *>(meta), device, device_ids, ndevice);
        model->qwen2->KVcacheManager() = manager;
        return model;
    }

    /** 模型销毁 */
    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (!model) return;
        delete model;
    }

    /** 模型权重指针 */
    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        if (!model) return nullptr;
        return &model->qwen2->weights();
    }

    /** 一次完整前向传播 */
    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        if (!model || !model->qwen2) return -1;
        return model->qwen2->forward(token_ids, ntoken);
    }
}
