#include "../../src/config.hpp"
#include "../../src/model/model.hpp"
#include "../../src/model/model_context.hpp"
#include "../../src/sequence/sequence.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Oracle: copy of pre-lift Qwen2::prepare_* behavior (locked expectations for Model).
std::vector<int64_t> oracle_prepare_block_table(const std::vector<llaisys::seq_t> &seqs, size_t block_table_width) {
    const size_t batch_size = seqs.size();
    std::vector<int64_t> block_table(batch_size * block_table_width, -1);
    for (size_t i = 0; i < batch_size; ++i) {
        const size_t offset = i * block_table_width;
        block_table[offset] = seqs[i]->block_ids().size();
        if (i > 0) {
            block_table[offset] += block_table[offset - block_table_width];
        }
        std::copy(seqs[i]->block_ids().begin(), seqs[i]->block_ids().end(), block_table.begin() + offset + 1);
    }
    return block_table;
}

llaisys::model::BatchPack oracle_prepare_prefill(const std::vector<llaisys::seq_t> &seqs) {
    const size_t batch_size = seqs.size();
    std::vector<int64_t> token_ids;
    std::vector<int64_t> pos_ids;
    std::vector<int64_t> cut_idx(1, 0);
    std::vector<int64_t> tot_len;
    size_t max_seq_len = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        llaisys::seq_t seq = seqs[i];
        const size_t seq_len = seq->scheduled_token_num();
        const size_t begin = seq->cached_token_num();
        const size_t end = begin + seq_len - 1;
        const std::vector<int64_t> &seq_token_ids = seq->token_ids();
        for (size_t j = begin; j <= end; ++j) {
            token_ids.push_back(seq_token_ids[j]);
            pos_ids.push_back(static_cast<int64_t>(j));
        }
        cut_idx.push_back(cut_idx.back() + static_cast<int64_t>(seq_len));
        tot_len.push_back(static_cast<int64_t>(begin + seq_len));
        max_seq_len = std::max(max_seq_len, seq_len);
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
    return llaisys::model::BatchPack{token_ids, pos_ids, cut_idx, tot_len, top_ks, top_ps, temps, rngs, max_seq_len};
}

llaisys::model::BatchPack oracle_prepare_decode(const std::vector<llaisys::seq_t> &seqs) {
    const size_t batch_size = seqs.size();
    std::vector<int64_t> token_ids;
    std::vector<int64_t> pos_ids;
    std::vector<int64_t> cut_idx(1, 0);
    std::vector<int64_t> tot_len;
    for (size_t i = 0; i < batch_size; ++i) {
        llaisys::seq_t seq = seqs[i];
        const size_t idx = seq->cached_token_num();
        const std::vector<int64_t> &seq_token_ids = seq->token_ids();
        token_ids.push_back(seq_token_ids[idx]);
        pos_ids.push_back(static_cast<int64_t>(idx));
        cut_idx.push_back(cut_idx.back() + 1);
        tot_len.push_back(static_cast<int64_t>(seq->token_num()));
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
    return llaisys::model::BatchPack{token_ids, pos_ids, cut_idx, tot_len, top_ks, top_ps, temps, rngs, 1};
}

void require_vec_eq(const std::vector<int64_t> &a, const std::vector<int64_t> &b, const char *msg) {
    require(a.size() == b.size(), msg);
    for (size_t i = 0; i < a.size(); ++i) {
        require(a[i] == b[i], msg);
    }
}

void require_pack_eq(const llaisys::model::BatchPack &a, const llaisys::model::BatchPack &b, const char *msg) {
    require_vec_eq(a.token_ids, b.token_ids, msg);
    require_vec_eq(a.pos_ids, b.pos_ids, msg);
    require_vec_eq(a.cut_idx, b.cut_idx, msg);
    require_vec_eq(a.tot_len, b.tot_len, msg);
    require(a.top_k.size() == b.top_k.size(), msg);
    require(a.top_p.size() == b.top_p.size(), msg);
    require(a.temperature.size() == b.temperature.size(), msg);
    require(a.rngs.size() == b.rngs.size(), msg);
    for (size_t i = 0; i < a.top_k.size(); ++i) {
        require(a.top_k[i] == b.top_k[i], msg);
        require(a.top_p[i] == b.top_p[i], msg);
        require(a.temperature[i] == b.temperature[i], msg);
        require(a.rngs[i] == b.rngs[i], msg);
    }
    require(a.max_seq_len == b.max_seq_len, msg);
}

std::vector<llaisys::seq_t> make_two_prefill_seqs() {
    // Seq0: cached=2, scheduled=3 over tokens [10..50]; blocks {7,8}
    std::vector<int64_t> tokens0{10, 20, 30, 40, 50};
    auto s0 = std::make_shared<llaisys::Sequence>(tokens0.data(), tokens0.size(), 64, 3, 0.8f, 0.5f);
    s0->cached_token_num() = 2;
    s0->scheduled_token_num() = 3;
    s0->block_ids() = {7, 8};

    // Seq1: cached=0, scheduled=4 over tokens [1..4]; blocks {9}
    std::vector<int64_t> tokens1{1, 2, 3, 4};
    auto s1 = std::make_shared<llaisys::Sequence>(tokens1.data(), tokens1.size(), 64, 5, 0.95f, 1.2f);
    s1->cached_token_num() = 0;
    s1->scheduled_token_num() = 4;
    s1->block_ids() = {9};

    return {s0, s1};
}

std::vector<llaisys::seq_t> make_two_decode_seqs() {
    // Decode: one new token position each; different sampling params.
    std::vector<int64_t> tokens0{10, 20, 30, 40, 50, 60};
    auto s0 = std::make_shared<llaisys::Sequence>(tokens0.data(), tokens0.size(), 64, 2, 0.7f, 0.0f);
    s0->cached_token_num() = 5;
    s0->scheduled_token_num() = 1;
    s0->block_ids() = {1, 2, 3};

    std::vector<int64_t> tokens1{100, 101, 102};
    auto s1 = std::make_shared<llaisys::Sequence>(tokens1.data(), tokens1.size(), 64, 4, 0.9f, 0.8f);
    s1->cached_token_num() = 2;
    s1->scheduled_token_num() = 1;
    s1->block_ids() = {4};

    return {s0, s1};
}

void test_model_prepare_matches_qwen2_oracle() {
    constexpr size_t width = 8;
    auto prefill_seqs = make_two_prefill_seqs();
    auto decode_seqs = make_two_decode_seqs();

    require_vec_eq(llaisys::model::Model::prepare_block_table(prefill_seqs, width),
                   oracle_prepare_block_table(prefill_seqs, width),
                   "prepare_block_table prefill parity");
    require_vec_eq(llaisys::model::Model::prepare_block_table(decode_seqs, width),
                   oracle_prepare_block_table(decode_seqs, width),
                   "prepare_block_table decode parity");

    require_pack_eq(llaisys::model::Model::prepare_prefill(prefill_seqs),
                    oracle_prepare_prefill(prefill_seqs),
                    "prepare_prefill parity");
    require_pack_eq(llaisys::model::Model::prepare_decode(decode_seqs),
                    oracle_prepare_decode(decode_seqs),
                    "prepare_decode parity");

    // Sanity: production width path with MAX_BLOCK_NUM still matches oracle.
    require_vec_eq(llaisys::model::Model::prepare_block_table(prefill_seqs, MAX_BLOCK_NUM),
                   oracle_prepare_block_table(prefill_seqs, MAX_BLOCK_NUM),
                   "prepare_block_table MAX_BLOCK_NUM parity");
}

} // namespace

int main() {
    test_model_prepare_matches_qwen2_oracle();
    std::cout << "prepare parity test passed\n";
    return 0;
}
