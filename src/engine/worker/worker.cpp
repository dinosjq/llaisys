#include "worker.hpp"

namespace llaisys::engine{

// prefill 分块粒度，与 paged_attention_nvidia.cu 的 BLOCK_M 保持一致
static constexpr size_t PREFILL_BLOCK_M = 8;

Worker::Worker(llaisysDeviceType_t device_type, int device_id, model::model_t model)
    : _device_type(device_type), _device_id(device_id), _model(std::move(model)) {}

/**
 * 预处理
 * 输入: 本次调度要计算的序列, 块表宽度, 阶段
 * 输出: forward 批次输入信息
 *
 * 块表: [[row_size, block_index_0, ...], ...]
 * 将每行的第一个元素定义为: 行的有效块表索引数, 即有效块数
 */
BatchInput Worker::prepare(const std::vector<seq_t> &seqs, const size_t table_width, const bool is_prefill) {
    // 批次大小
    const size_t batch_size = seqs.size();
    std::vector<int64_t> token_ids;     // token
    std::vector<int64_t> pos_ids;       // 位置
    std::vector<int64_t> cut_idx(1, 0); // 截断
    std::vector<int64_t> tot_len;       // 总长
    // 块表
    std::vector<int64_t> block_ids(batch_size * table_width, -1);
    // 批次信息
    size_t tot_block_num = 0;
    size_t max_seq_len = 0;
    // 采样参数
    std::vector<int> top_ks;
    std::vector<float> top_ps;
    std::vector<float> temperatures;
    std::vector<std::mt19937 *> rngs;

    // 预处理块表
    for (size_t i = 0; i < batch_size; ++i) {
        const size_t offset = i * table_width;
        // 块数
        block_ids[offset] = seqs[i]->block_ids().size();
        if (i > 0) {
            block_ids[offset] += block_ids[offset - table_width];
        }
        // 复制
        std::copy(seqs[i]->block_ids().begin(), seqs[i]->block_ids().end(), block_ids.begin() + offset + 1);
    }

    // 预处理序列信息
    for (size_t i = 0; i < batch_size; ++i) {
        seq_t seq = seqs[i];
        const size_t seq_len = seq->scheduled_token_num();
        const size_t begin = seq->cached_token_num();
        const size_t end = begin + seq_len - 1;
        const std::vector<int64_t> &_token_ids = seq->token_ids();
        for (size_t j = begin; j <= end; ++j) {
            token_ids.push_back(_token_ids[j]);
            pos_ids.push_back(j);
        }
        cut_idx.push_back(cut_idx.back() + seq_len);
        tot_len.push_back(begin + seq_len);
        tot_block_num += seqs[i]->block_ids().size();
        max_seq_len = std::max(max_seq_len, seq_len);
    }

    // 采样参数
    for (size_t i = 0; i < batch_size; ++i) {
        top_ks.push_back(seqs[i]->top_k());
        top_ps.push_back(seqs[i]->top_p());
        temperatures.push_back(seqs[i]->temperature());
        rngs.push_back(&seqs[i]->rng());
    }
    return BatchPack{token_ids, pos_ids, cut_idx, tot_len, block_ids, top_ks, top_ps, temperatures, rngs, batch_size, tot_block_num, max_seq_len, is_prefill};
}

BatchOutput Worker::forward(const BatchInput &input) {
    return this->_model->forward(input);
}
};