#include "model.hpp"

#include "../../utils/check.hpp"

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

Model::~Model() {
    this->_ctx.workspace.capture_post_attn0.reset();
    this->_ctx.workspace.capture_post_ffn0.reset();
    this->stop();
    if (this->_worker.joinable()) {
        this->_worker.join();
    }
}

void Model::start() {
    if (this->_running) {
        return;
    }
    // 权重加载结束后、worker 启动前
    this->sync_weights();
    auto loop = [](Model *model) -> void {
        while (model->_running) {
            // 先根据调度器决定本次计算的序列
            auto [seqs, is_prefill] = model->_scheduler->schedule();
            if (seqs.empty())
                continue;
            // 预处理块表
            auto block_ids = Model::prepare_block_table(seqs, MAX_BLOCK_NUM);
            // prefill / decode 预处理
            BatchPack pack = is_prefill ? Model::prepare_prefill(seqs) : Model::prepare_decode(seqs);
            // 子类 forward（Layer 组装）
            std::vector<int64_t> token_ids = model->forward(pack, block_ids, is_prefill);
            // 后处理更新 kv cache 信息
            model->_scheduler->postprocess(seqs, token_ids, is_prefill);
        }
    };
    // Genshen启动!
    this->_running = true;
    this->_worker = std::thread(loop, this);
}

void Model::stop() {
    this->_running = false;
}

model_request_t Model::submit(int64_t *token_ids, size_t ntoken, int64_t max_new_tokens, int top_k, float top_p,
                              float temperature) {
    // 首次提交时再 sync + 启动 worker
    this->start();
    size_t limit = (max_new_tokens < 0) ? MAX_TOKEN_NUM : ntoken + static_cast<size_t>(max_new_tokens);
    auto request = std::make_shared<ModelRequest>();
    request->sequence = std::make_shared<Sequence>(token_ids, ntoken, limit, top_k, top_p, temperature);
    request->observed_tokens = ntoken;
    {
        std::lock_guard<std::mutex> lock(this->_request_lock);
        this->_active_requests.push_back(request);
    }
    this->_scheduler->add(request->sequence);
    return request;
}

int64_t Model::await(const model_request_t &request) {
    CHECK_ARGUMENT(request != nullptr && request->sequence != nullptr, "invalid request handle");
    std::lock_guard<std::mutex> lock(request->mutex);
    if (!request->sequence->wait_for_token(request->observed_tokens)) {
        return -2;
    }
    const int64_t token = request->sequence->token_at(request->observed_tokens);
    ++request->observed_tokens;
    return token;
}

void Model::abort(const model_request_t &request) {
    if (request && request->sequence) {
        this->_scheduler->abort(request->sequence);
    }
}

void Model::release(const model_request_t &request) {
    if (!request) {
        return;
    }
    if (request->sequence && !request->sequence->is_finish()) {
        this->_scheduler->abort(request->sequence);
    }
    std::lock_guard<std::mutex> lock(this->_request_lock);
    auto it = std::find(this->_active_requests.begin(), this->_active_requests.end(), request);
    if (it != this->_active_requests.end()) {
        this->_active_requests.erase(it);
    }
}

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
 */
BatchPack Model::prepare_prefill(const std::vector<seq_t> &seqs) {
    const size_t batch_size = seqs.size();
    std::vector<int64_t> token_ids;
    std::vector<int64_t> pos_ids;
    std::vector<int64_t> cut_idx(1, 0);
    std::vector<int64_t> tot_len;
    size_t max_seq_len = 0;
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
    return BatchPack{token_ids, pos_ids, cut_idx, tot_len, top_ks, top_ps, temps, rngs, max_seq_len};
}

/**
 * decode预处理
 * 相较于 prefill 更加简单，因为所有 seq 当前的计算长度 seqlen 都是 1
 */
BatchPack Model::prepare_decode(const std::vector<seq_t> &seqs) {
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
    std::vector<int> top_ks;
    std::vector<float> top_ps, temps;
    std::vector<std::mt19937 *> rngs;
    for (size_t i = 0; i < batch_size; ++i) {
        top_ks.push_back(seqs[i]->top_k());
        top_ps.push_back(seqs[i]->top_p());
        temps.push_back(seqs[i]->temperature());
        rngs.push_back(&seqs[i]->rng());
    }
    return BatchPack{token_ids, pos_ids, cut_idx, tot_len, top_ks, top_ps, temps, rngs, 1};
}

} // namespace llaisys::framework
