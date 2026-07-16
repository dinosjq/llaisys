#include "llaisys/models/qwen2.h"

#include "../llaisys_tensor.hpp"
#include "../../models/qwen2.hpp"

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
    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken, int64_t max_new_tokens,
                                   int top_k, float top_p, float temperature) {
        if (!model || !model->qwen2) return -1;
        return model->qwen2->request(token_ids, ntoken, max_new_tokens, top_k, top_p, temperature);
    }

    /** 取消请求 */
    void llaisysQwen2ModelAbort(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        if (!model || !model->qwen2) return;
        model->qwen2->abort(token_ids, ntoken);
    }

    /** 获取上一步 logprobs (top-n token + logprob) */
    int llaisysQwen2ModelGetLogprobs(struct LlaisysQwen2Model *model,
                                     float *out_logprobs, int n_vocab,
                                     int *out_tokens, int n_tokens, int batch_idx) {
        if (!model || !model->qwen2) return -1;
        return model->qwen2->get_logprobs(out_logprobs, n_vocab, out_tokens, n_tokens, batch_idx);
    }
}
