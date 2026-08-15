#include "model.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace llaisys::framework {

namespace {

void validate_attn_layer(const ModelContext &ctx, size_t layer) {
    if (layer >= ctx.meta.nlayer || layer >= ctx.attns.size()) {
        throw std::invalid_argument("attention weight layer is out of range");
    }
}

void validate_ffn_layer(const ModelContext &ctx, size_t layer) {
    if (layer >= ctx.meta.nlayer || layer >= ctx.ffns.size()) {
        throw std::invalid_argument("FFN weight layer is out of range");
    }
}

} // namespace

void Model::set_weight(ModelContext &ctx, WeightRole role, size_t layer, tensor_t tensor) {
    switch (role) {
    case WeightRole::InEmbed:
        ctx.in_embed = std::move(tensor);
        return;
    case WeightRole::OutEmbed:
        ctx.out_embed = std::move(tensor);
        return;
    case WeightRole::OutNorm:
        ctx.out_norm_w = std::move(tensor);
        return;
    case WeightRole::AttnNorm:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].norm_w = std::move(tensor);
        return;
    case WeightRole::AttnQ_W:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].q_w = std::move(tensor);
        return;
    case WeightRole::AttnQ_B:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].q_b = std::move(tensor);
        return;
    case WeightRole::AttnK_W:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].k_w = std::move(tensor);
        return;
    case WeightRole::AttnK_B:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].k_b = std::move(tensor);
        return;
    case WeightRole::AttnV_W:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].v_w = std::move(tensor);
        return;
    case WeightRole::AttnV_B:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].v_b = std::move(tensor);
        return;
    case WeightRole::AttnO_W:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].o_w = std::move(tensor);
        return;
    case WeightRole::MlpNorm:
        validate_ffn_layer(ctx, layer);
        ctx.ffns[layer].norm_w = std::move(tensor);
        return;
    case WeightRole::MlpGate_W:
        validate_ffn_layer(ctx, layer);
        ctx.ffns[layer].gate_w = std::move(tensor);
        return;
    case WeightRole::MlpUp_W:
        validate_ffn_layer(ctx, layer);
        ctx.ffns[layer].up_w = std::move(tensor);
        return;
    case WeightRole::MlpDown_W:
        validate_ffn_layer(ctx, layer);
        ctx.ffns[layer].down_w = std::move(tensor);
        return;
    }
}

/**
 * 预处理块表
 * 输入: 本次调度要计算的序列
 * 输出: 合并后的总块表，每个序列按照最长块表长度对齐
 *
 * 重定义块表：[[row_size, block_table_index_0, ...], ...]
 * 将每行的第一个元素定义为：行的有效块表索引数，即有效块数
 */
std::vector<int64_t> Model::prepare_block_table(const std::vector<seq_t> &seqs, size_t block_table_width) {
    const size_t batch_size = seqs.size();
    std::vector<int64_t> block_table(batch_size * block_table_width, -1);
    for (size_t i = 0; i < batch_size; ++i) {
        const size_t offset = i * block_table_width;
        // 新增：块数
        block_table[offset] = seqs[i]->block_ids().size();
        if (i > 0) {
            block_table[offset] += block_table[offset - block_table_width];
        }
        // 复制到指定位置
        std::copy(seqs[i]->block_ids().begin(), seqs[i]->block_ids().end(), block_table.begin() + offset + 1);
    }
    return block_table;
}

/**
 * prefill预处理
 * prefill阶段每个seq的计算长度可能不同，导致直接对tensor添加batch维度会无法对齐
 * 所以这里的做法是: 直接将所有要计算的seq拼接在一起，并记录对应的端点信息；
 * 原因是：除了 attention 和 采样算子(top_k) 之外的其他算子天然支持并行计算，
 *        并且 top_k 可以先将要处理的信息先按 batch 划分，然后执行批次计算，
 *        对 attention 则需要记录间断点（load q）和最长序列信息（算子分块）还需要已计算部分的总长信息（totlen），
 *        也可以进行批处理改造
 */
Qwen2Pack Model::prepare_prefill(const std::vector<seq_t> &seqs) {
    const size_t batch_size = seqs.size();
    std::vector<int64_t> token_ids; // token信息
    std::vector<int64_t> pos_ids;   // 位置信息
    std::vector<int64_t> cut_idx(1, 0); // 切断点
    std::vector<int64_t> tot_len;       // 总长信息
    size_t max_seq_len = 0;         // 最大序列长度
    for (size_t i = 0; i < batch_size; ++i) {
        seq_t seq = seqs[i];
        const size_t seq_len = seq->scheduled_token_num();
        const size_t begin = seq->cached_token_num();
        const size_t end = begin + seq_len - 1;
        const std::vector<int64_t> &seq_token_ids = seq->token_ids();
        for (size_t j = begin; j <= end; ++j) {
            token_ids.push_back(seq_token_ids[j]);
            pos_ids.push_back(j);
        }
        cut_idx.push_back(cut_idx.back() + seq_len);
        tot_len.push_back(begin + seq_len);
        max_seq_len = std::max(max_seq_len, seq_len);
    }
    // per-seq 采样参数
    std::vector<int> top_ks;
    std::vector<float> top_ps, temps;
    std::vector<std::mt19937 *> rngs;
    for (size_t i = 0; i < batch_size; ++i) {
        top_ks.push_back(seqs[i]->top_k());
        top_ps.push_back(seqs[i]->top_p());
        temps.push_back(seqs[i]->temperature());
        rngs.push_back(&seqs[i]->rng());
    }
    return Qwen2Pack{token_ids, pos_ids, cut_idx, tot_len, top_ks, top_ps, temps, rngs, max_seq_len};
}

/**
 * decode预处理
 * 相较于 prefill 更加简单，因为所有 seq 当前的计算长度 seqlen 都是 1
 */
Qwen2Pack Model::prepare_decode(const std::vector<seq_t> &seqs) {
    const size_t batch_size = seqs.size();
    std::vector<int64_t> token_ids;
    std::vector<int64_t> pos_ids;
    std::vector<int64_t> cut_idx(1, 0);
    std::vector<int64_t> tot_len;
    for (size_t i = 0; i < batch_size; ++i) {
        seq_t seq = seqs[i];
        const size_t idx = seq->cached_token_num();
        const std::vector<int64_t> &seq_token_ids = seq->token_ids();
        token_ids.push_back(seq_token_ids[idx]);
        pos_ids.push_back(idx);
        cut_idx.push_back(cut_idx.back() + 1);
        tot_len.push_back(seq->token_num());
    }
    // per-seq 采样参数
    std::vector<int> top_ks;
    std::vector<float> top_ps, temps;
    std::vector<std::mt19937 *> rngs;
    for (size_t i = 0; i < batch_size; ++i) {
        top_ks.push_back(seqs[i]->top_k());
        top_ps.push_back(seqs[i]->top_p());
        temps.push_back(seqs[i]->temperature());
        rngs.push_back(&seqs[i]->rng());
    }
    return Qwen2Pack{token_ids, pos_ids, cut_idx, tot_len, top_ks, top_ps, temps, rngs, 1};
}

} // namespace llaisys::framework
