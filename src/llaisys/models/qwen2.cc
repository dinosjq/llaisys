#include "llaisys/models/qwen2.h"

#include "../llaisys_tensor.hpp"
#include "../../models/qwen2.hpp"
#include "../../kv_cache/block_manager.hpp"
#include "../../config.hpp"

struct LlaisysQwen2Model {
    llaisys::Qwen2_t qwen2;
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

    /** 一次完整前向传播 */
    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken, int64_t max_new_tokens) {
        if (!model || !model->qwen2) return -1;
        return model->qwen2->request(token_ids, ntoken, max_new_tokens);
    }

    /** 可用 block 数 */
    size_t llaisysQwen2GetFreeBlockCount(struct LlaisysQwen2Model *model) {
        if (!model || !model->qwen2) return 0;
        return model->qwen2->free_block_count();
    }

    /** 已用 block 数 */
    size_t llaisysQwen2GetUsedBlockCount(struct LlaisysQwen2Model *model) {
        if (!model || !model->qwen2) return 0;
        return model->qwen2->used_block_count();
    }

    /** 调度器中 running 序列数 */
    size_t llaisysQwen2GetRunningCount(struct LlaisysQwen2Model *model) {
        if (!model || !model->qwen2) return 0;
        return model->qwen2->running_count();
    }
}
