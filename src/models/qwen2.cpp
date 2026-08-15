
#include "../sequence/sequence.hpp"
#include "qwen2.hpp"
#include "framework/model.hpp"
#include "layers/qwen2/stack.hpp"
#include "../llaisys/llaisys_tensor.hpp"
#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/paged_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/top_k/op.hpp"
#include "../config.hpp"
#include "../ops/kv_cache_move/op.hpp"
#include "sampling.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <functional>
#include <thread> 

namespace llaisys{

namespace {
static tensor_t to_tensor(llaisysTensor_t t) {
    return t ? t->tensor : nullptr;
}

static void init_weight_arrays(Qwen2Weights &w, size_t nlayer) {
    w.in_embed = nullptr;
    w.out_embed = nullptr;
    w.out_norm_w = nullptr;
    w.attn_norm_w = new llaisysTensor_t[nlayer]();
    w.attn_q_w = new llaisysTensor_t[nlayer]();
    w.attn_q_b = new llaisysTensor_t[nlayer]();
    w.attn_k_w = new llaisysTensor_t[nlayer]();
    w.attn_k_b = new llaisysTensor_t[nlayer]();
    w.attn_v_w = new llaisysTensor_t[nlayer]();
    w.attn_v_b = new llaisysTensor_t[nlayer]();
    w.attn_o_w = new llaisysTensor_t[nlayer]();
    w.mlp_norm_w = new llaisysTensor_t[nlayer]();
    w.mlp_gate_w = new llaisysTensor_t[nlayer]();
    w.mlp_up_w = new llaisysTensor_t[nlayer]();
    w.mlp_down_w = new llaisysTensor_t[nlayer]();
}

static void free_weight_arrays(Qwen2Weights &w) {
    delete[] w.attn_norm_w;
    delete[] w.attn_q_w;
    delete[] w.attn_q_b;
    delete[] w.attn_k_w;
    delete[] w.attn_k_b;
    delete[] w.attn_v_w;
    delete[] w.attn_v_b;
    delete[] w.attn_o_w;
    delete[] w.mlp_norm_w;
    delete[] w.mlp_gate_w;
    delete[] w.mlp_up_w;
    delete[] w.mlp_down_w;
    w.in_embed = nullptr;
    w.out_embed = nullptr;
    w.out_norm_w = nullptr;
    w.attn_norm_w = nullptr;
    w.attn_q_w = nullptr;
    w.attn_q_b = nullptr;
    w.attn_k_w = nullptr;
    w.attn_k_b = nullptr;
    w.attn_v_w = nullptr;
    w.attn_v_b = nullptr;
    w.attn_o_w = nullptr;
    w.mlp_norm_w = nullptr;
    w.mlp_gate_w = nullptr;
    w.mlp_up_w = nullptr;
    w.mlp_down_w = nullptr;
}

// LLAISYS_QWEN2_LAYER_FORWARD unset/0 → legacy；非 0 → layer stack
static bool env_use_layer_forward() {
    const char *value = std::getenv("LLAISYS_QWEN2_LAYER_FORWARD");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return std::strcmp(value, "0") != 0;
}

} // namespace

Qwen2::Qwen2(Qwen2Meta meta, 
             llaisysDeviceType_t device, 
             int *device_ids, 
             int ndevice):
             _meta(meta),
             _device_type(device),
             _device_ids(std::vector<int>(device_ids, device_ids + ndevice)),
             _running(false)

{
    // 加载一些数据
    llaisysDataType_t dtype = meta.dtype;
    const size_t nlayer = meta.nlayer;
    const size_t hs = meta.hs;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t di = meta.di;
    const size_t voc = meta.voc;
    const size_t token_num = KV_CACHE_TOKEN_NUM;
    const size_t block_num = KV_CACHE_BLOCK_NUM;
    const size_t block_size = block_num * token_num * nkvh * dh;
    const size_t batch_max_token_num = BATCH_MAX_TOKEN_NUM;
    const size_t batch_max_seq_num = BATCH_MAX_SEQ_NUM;
    const size_t max_block_num = MAX_BLOCK_NUM;
    int device_id = device_ids[0];
    // 初始化权重数组
    init_weight_arrays(this->_weights, nlayer);
    // 初始化运行时张量
    _tensors._x_norm = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._q = Tensor::create({batch_max_token_num, nh * dh}, dtype, device, device_id);
    _tensors._k = Tensor::create({batch_max_token_num, nkvh * dh}, dtype, device, device_id);
    _tensors._v = Tensor::create({batch_max_token_num, nkvh * dh}, dtype, device, device_id);
    _tensors._q_rope = Tensor::create({batch_max_token_num, nh, dh}, dtype, device, device_id);
    _tensors._k_rope = Tensor::create({batch_max_token_num, nkvh, dh}, dtype, device, device_id);
    _tensors._attn_val = Tensor::create({batch_max_token_num, nh, dh}, dtype, device, device_id);
    _tensors._attn_out = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._x_attn = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._m_norm = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._gate = Tensor::create({batch_max_token_num, di}, dtype, device, device_id);
    _tensors._up = Tensor::create({batch_max_token_num, di}, dtype, device, device_id);
    _tensors._swiglu = Tensor::create({batch_max_token_num, di}, dtype, device, device_id);
    _tensors._down = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._x_mlp = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._dev_block_ids = Tensor::create({batch_max_seq_num * max_block_num}, LLAISYS_DTYPE_I64, device, device_id);
    _tensors._dev_cut_idx = Tensor::create({batch_max_seq_num + 1}, LLAISYS_DTYPE_I64, device, device_id);
    _tensors._dev_tot_len = Tensor::create({batch_max_seq_num}, LLAISYS_DTYPE_I64, device, device_id);
    _tensors._dev_token_ids = Tensor::create({batch_max_token_num}, LLAISYS_DTYPE_I64, device, device_id);
    _tensors._dev_pos_ids = Tensor::create({batch_max_token_num}, LLAISYS_DTYPE_I64, device, device_id);
    _tensors._x = Tensor::create({batch_max_token_num, hs}, dtype, device, device_id);
    _tensors._x_part = Tensor::create({batch_max_seq_num, hs}, dtype, device, device_id);
    _tensors._x_part_norm = Tensor::create({batch_max_seq_num, hs}, dtype, device, device_id);
    _tensors._logits = Tensor::create({batch_max_seq_num, voc}, dtype, device, device_id);
    _tensors._top_idx = Tensor::create({batch_max_seq_num, MAX_TOP_K}, LLAISYS_DTYPE_I64, device, device_id);
    _tensors._top_val = Tensor::create({batch_max_seq_num, MAX_TOP_K}, dtype, device, device_id);
    _tensors._attn_acc = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, dh}, LLAISYS_DTYPE_F32, device, device_id);
    _tensors._attn_sum = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, 1}, LLAISYS_DTYPE_F32, device, device_id);
    _tensors._attn_max = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, 1}, LLAISYS_DTYPE_F32, device, device_id);
    // 初始化分层 kv cache
    this->_k_cache = std::vector<tensor_t>(nlayer, nullptr);
    this->_v_cache = std::vector<tensor_t>(nlayer, nullptr);
    tensor_t cache = Tensor::create({2, nlayer, block_size}, meta.dtype, device, device_id);
    tensor_t k_cache = cache->slice(0, 0, 1)->reshape({nlayer, block_size});
    tensor_t v_cache = cache->slice(0, 1, 2)->reshape({nlayer, block_size});
    for(size_t i = 0; i < nlayer; ++ i){
        this->_k_cache[i] = k_cache->slice(0, i, i + 1)->reshape({block_num, token_num, nkvh, dh});
        this->_v_cache[i] = v_cache->slice(0, i, i + 1)->reshape({block_num, token_num, nkvh, dh});
    }
    // 创建调度器
    this->_scheduler = std::make_shared<Scheduler>(token_num, block_num, BATCH_MAX_TOKEN_NUM, BATCH_MAX_SEQ_NUM, meta.end_token);
    // 初始化分层 ModelContext（权重稍后从 legacy Weights 绑定）
    this->_ctx.meta = framework::ModelMeta{meta.dtype, meta.nlayer, meta.hs, meta.nh, meta.nkvh, meta.dh, meta.di,
                                            meta.maxseq, meta.voc, meta.epsilon, meta.theta, meta.end_token};
    this->_ctx.attns.resize(nlayer);
    this->_ctx.ffns.resize(nlayer);
    this->_ctx.k_caches.resize(nlayer);
    this->_ctx.v_caches.resize(nlayer);
    this->use_layer_forward_ = env_use_layer_forward();
    // 启动时间循环
    this->start();
}

Qwen2::~Qwen2() {
    // 先终止事件循环
    this->stop();
    // 暂停对应的生产者线程
    if (this->_worker.joinable()) {
        this->_worker.join();
    }
    // 释放权重所占内存
    free_weight_arrays(this->_weights);
}  

void Qwen2::start(){
    if (this->_running) {
        return;
    }
    // 事件循环方法
    auto loop = [](Qwen2 *model)->void{
        while(model->_running){
            // 先根据调度器的调度方法决定本次计算的序列
            auto [seqs, is_prefill] = model->_scheduler->schedule();
            if(seqs.empty()) continue;
            // 预处理块表
            auto block_ids = model->prepare_block_table(seqs);
            // 根据是 prefill 还是 decode 分别进行预处理
            Qwen2Pack pack = is_prefill ? model->prepare_prefill(seqs) : model->prepare_decode(seqs);
            // 执行一次批处理前向传播（默认 legacy；LLAISYS_QWEN2_LAYER_FORWARD 开启时走 layer）
            std::vector<int64_t> token_ids = model->use_layer_forward_
                                                ? model->forward_layers(pack, block_ids, is_prefill)
                                                : model->forward_legacy(pack, block_ids, is_prefill);
            // 后处理更新kv cache信息
            model->_scheduler->postprocess(seqs, token_ids, is_prefill);
        }
    };
    // Genshen启动!
    this->_running = true;
    this->_worker = std::thread(loop, this);
}

// 暂停事件循环
void Qwen2::stop(){
    this->_running = false;
}

qwen2_request_t Qwen2::submit(int64_t *token_ids, size_t ntoken, int64_t max_new_tokens, int top_k, float top_p, float temperature) {
    size_t limit = (max_new_tokens < 0) ? MAX_TOKEN_NUM : ntoken + static_cast<size_t>(max_new_tokens);
    auto request = std::make_shared<Qwen2Request>();
    request->sequence = std::make_shared<Sequence>(token_ids, ntoken, limit, top_k, top_p, temperature);
    request->observed_tokens = ntoken;
    {
        std::lock_guard<std::mutex> lock(this->_request_lock);
        this->_active_requests.push_back(request);
    }
    this->_scheduler->add(request->sequence);
    return request;
}

int64_t Qwen2::await(const qwen2_request_t &request) {
    CHECK_ARGUMENT(request != nullptr && request->sequence != nullptr, "invalid request handle");
    std::lock_guard<std::mutex> lock(request->mutex);
    if (!request->sequence->wait_for_token(request->observed_tokens)) {
        return -2;
    }
    const int64_t token = request->sequence->token_at(request->observed_tokens);
    ++request->observed_tokens;
    return token;
}

void Qwen2::abort(const qwen2_request_t &request) {
    if (request && request->sequence) {
        this->_scheduler->abort(request->sequence);
    }
}

void Qwen2::release(const qwen2_request_t &request) {
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

std::vector<int64_t> Qwen2::prepare_block_table(const std::vector<seq_t> &seqs){
    return framework::Model::prepare_block_table(seqs, MAX_BLOCK_NUM);
}

Qwen2Pack Qwen2::prepare_prefill(const std::vector<seq_t> &seqs){
    return framework::Model::prepare_prefill(seqs);
}

Qwen2Pack Qwen2::prepare_decode(const std::vector<seq_t> &seqs){
    return framework::Model::prepare_decode(seqs);
}

void Qwen2::bind_context_weights() {
    const Qwen2Weights &w = this->_weights;
    framework::ModelContext &ctx = this->_ctx;
    const size_t nlayer = this->_meta.nlayer;

    // 从 legacy Weights 绑定到分层 Context（每次 forward_layers 调用，权重可能刚被 Python 填入）
    ctx.in_embed = to_tensor(w.in_embed);
    ctx.out_embed = to_tensor(w.out_embed);
    ctx.out_norm_w = to_tensor(w.out_norm_w);
    for (size_t layer = 0; layer < nlayer; ++layer) {
        auto &attn = ctx.attns[layer];
        attn.norm_w = to_tensor(w.attn_norm_w[layer]);
        attn.q_w = to_tensor(w.attn_q_w[layer]);
        attn.q_b = to_tensor(w.attn_q_b[layer]);
        attn.k_w = to_tensor(w.attn_k_w[layer]);
        attn.k_b = to_tensor(w.attn_k_b[layer]);
        attn.v_w = to_tensor(w.attn_v_w[layer]);
        attn.v_b = to_tensor(w.attn_v_b[layer]);
        attn.o_w = to_tensor(w.attn_o_w[layer]);

        auto &ffn = ctx.ffns[layer];
        ffn.norm_w = to_tensor(w.mlp_norm_w[layer]);
        ffn.gate_w = to_tensor(w.mlp_gate_w[layer]);
        ffn.up_w = to_tensor(w.mlp_up_w[layer]);
        ffn.down_w = to_tensor(w.mlp_down_w[layer]);

        ctx.k_caches[layer] = this->_k_cache[layer];
        ctx.v_caches[layer] = this->_v_cache[layer];
    }
}

void Qwen2::bind_context_runtime(Qwen2Pack &pack, std::vector<int64_t> &block_ids, bool is_prefill) {
    const Qwen2Meta &meta = this->_meta;
    const Qwen2Tensors &ts = this->_tensors;
    framework::ModelContext &ctx = this->_ctx;

    const std::vector<int64_t> &token_ids = pack.token_ids;
    const std::vector<int64_t> &pos_ids = pack.pos_ids;
    const std::vector<int64_t> &cut_idx = pack.cut_idx;
    const std::vector<int64_t> &tot_len = pack.tot_len;
    const size_t max_seq_len = pack.max_seq_len;
    const size_t tot_seq_len = token_ids.size();
    const size_t batch_size = tot_len.size();
    const size_t max_block_num = MAX_BLOCK_NUM;
    // block_ids 末行第 0 列是累计块数前缀和 = 整个批次总块数
    const size_t tot_block_num = block_ids[(batch_size - 1) * MAX_BLOCK_NUM];

    const size_t hs = meta.hs;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t di = meta.di;
    const size_t voc = meta.voc;

    // 填 BatchRuntime（embedding 从 runtime.token_ids 建 device 张量）
    ctx.runtime.token_ids = token_ids;
    ctx.runtime.pos_ids = pos_ids;
    ctx.runtime.cut_idx = cut_idx;
    ctx.runtime.tot_len = tot_len;
    ctx.runtime.max_seq_len = max_seq_len;
    ctx.runtime.tot_block_num = tot_block_num;
    ctx.runtime.is_prefill = is_prefill;

    // 切片绑定到与 legacy 相同的预分配 workspace
    ctx.workspace.x_norm = ts._x_norm->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.q = ts._q->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh * dh});
    ctx.workspace.k = ts._k->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh * dh});
    ctx.workspace.v = ts._v->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh * dh});
    ctx.workspace.q_rope = ts._q_rope->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh, dh});
    ctx.workspace.k_rope = ts._k_rope->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh, dh});
    ctx.workspace.attn_val = ts._attn_val->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh, dh});
    ctx.workspace.attn_out = ts._attn_out->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.x_attn = ts._x_attn->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.m_norm = ts._m_norm->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.gate = ts._gate->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    ctx.workspace.up = ts._up->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    ctx.workspace.swiglu = ts._swiglu->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    ctx.workspace.down = ts._down->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.x_mlp = ts._x_mlp->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.block_ids = ts._dev_block_ids->slice(0, 0, batch_size * max_block_num)->view({batch_size, max_block_num});
    ctx.workspace.cut_idx = ts._dev_cut_idx->slice(0, 0, batch_size + 1)->view({batch_size + 1});
    ctx.workspace.tot_len = ts._dev_tot_len->slice(0, 0, batch_size)->view({batch_size});
    ctx.workspace.pos_ids = ts._dev_pos_ids->slice(0, 0, tot_seq_len)->view({tot_seq_len});
    ctx.workspace.x = ts._x->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    ctx.workspace.x_part = ts._x_part->slice(0, 0, batch_size)->view({batch_size, hs});
    ctx.workspace.x_part_norm = ts._x_part_norm->slice(0, 0, batch_size)->view({batch_size, hs});
    ctx.workspace.logits = ts._logits->slice(0, 0, batch_size)->view({batch_size, voc});
    ctx.workspace.attn_acc = ts._attn_acc;
    ctx.workspace.attn_sum = ts._attn_sum;
    ctx.workspace.attn_max = ts._attn_max;
    ctx.workspace.capture_post_attn0.reset();
    ctx.workspace.capture_post_ffn0.reset();

    // 初始化设备端元数据（H2D）；token_ids 由 Embedding 层从 runtime 加载
    ctx.workspace.block_ids->load(block_ids.data());
    ctx.workspace.cut_idx->load(cut_idx.data());
    ctx.workspace.tot_len->load(tot_len.data());
    ctx.workspace.pos_ids->load(pos_ids.data());
}

std::vector<int64_t> Qwen2::sample_from_logits(tensor_t logits, Qwen2Pack &pack) {
    const Qwen2Meta &meta = this->_meta;
    const llaisysDeviceType_t device = this->_device_type;
    const size_t batch_size = pack.tot_len.size();

    // top_k
    const size_t K = static_cast<size_t>(*std::max_element(pack.top_k.begin(), pack.top_k.end()));
    tensor_t top_idx = Tensor::create({batch_size, K}, LLAISYS_DTYPE_I64, device, this->_device_ids[0]);
    tensor_t top_val = Tensor::create({batch_size, K}, meta.dtype, device, this->_device_ids[0]);
    ops::topk(top_idx, top_val, logits, K);
    // 搬到cpu
    if (device != LLAISYS_DEVICE_CPU) {
        top_idx = top_idx->to(LLAISYS_DEVICE_CPU, 0);
        top_val = top_val->to(LLAISYS_DEVICE_CPU, 0);
    }
    const std::vector<int64_t> all_idx = top_idx->reshape({batch_size * K})->to_vector<int64_t>();
    const std::vector<float> all_val = top_val->reshape({batch_size * K})->to_vector<float>();
    // 逐个进行随机 top_p 采样 (per-seq top_p / temperature)
    std::vector<int64_t> result(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<SamplingCandidate> candidates;
        candidates.reserve(K);
        for (size_t j = 0; j < K; ++j) {
            const size_t offset = i * K + j;
            candidates.push_back({all_val[offset], all_idx[offset]});
        }
        result[i] = sample_token(candidates, pack.top_k[i], pack.top_p[i], pack.temperature[i], *pack.rngs[i]);
    }
    return result;
}

tensor_t Qwen2::last_logits(size_t batch_size) const {
    const size_t voc = this->_meta.voc;
    return this->_tensors._logits->slice(0, 0, batch_size)->view({batch_size, voc});
}

std::vector<int64_t> Qwen2::forward(Qwen2Pack &pack, std::vector<int64_t> &block_ids, bool is_prefill) {
    if (this->use_layer_forward_) {
        return forward_layers(pack, block_ids, is_prefill);
    }
    return forward_legacy(pack, block_ids, is_prefill);
}

std::vector<int64_t> Qwen2::forward_layers(Qwen2Pack &pack, std::vector<int64_t> &block_ids, bool is_prefill) {
    // 绑定权重 / KV，再写入本步 runtime + workspace
    bind_context_weights();
    bind_context_runtime(pack, block_ids, is_prefill);
    // 执行分层 stack（产出 logits；层内 residual 后 CPU stable_capture）
    framework::run_qwen2_layer_stack(this->_ctx, is_prefill);
    // 与 legacy 相同的 topk + sample 后处理
    return sample_from_logits(this->_ctx.workspace.logits, pack);
}

std::vector<int64_t> Qwen2::forward_legacy(Qwen2Pack &pack, std::vector<int64_t> &block_ids, bool is_prefill){
    // 加载元数据 + 权重
    const Qwen2Meta &meta = this->_meta;
    const Qwen2Weights &w = this->_weights;
    const Qwen2Tensors &ts = this->_tensors;

    // 预处理信息
    const std::vector<int64_t> &token_ids = pack.token_ids;
    const std::vector<int64_t> &pos_ids = pack.pos_ids;
    const std::vector<int64_t> &cut_idx = pack.cut_idx;
    const std::vector<int64_t> &tot_len = pack.tot_len;
    const size_t max_seq_len = pack.max_seq_len; 
    const size_t tot_seq_len = token_ids.size();
    const size_t batch_size = tot_len.size();
    const size_t max_block_num = MAX_BLOCK_NUM;
    // block_ids 末行第 0 列是累计块数前缀和 = 整个批次总块数
    const size_t tot_block_num = block_ids[(batch_size - 1) * MAX_BLOCK_NUM];

    // 加载其他参数
    const size_t nlayer = meta.nlayer;
    const size_t hs = meta.hs;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t di = meta.di;
    const size_t voc = meta.voc;
    const float epsilon = meta.epsilon;
    const float theta = meta.theta;

    // 初始化中间张量
    tensor_t x_norm = ts._x_norm->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    tensor_t q = ts._q->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh * dh});
    tensor_t k = ts._k->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh * dh});
    tensor_t v = ts._v->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh * dh}); 
    tensor_t q_rope = ts._q_rope->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh, dh});
    tensor_t k_rope = ts._k_rope->slice(0, 0, tot_seq_len)->view({tot_seq_len, nkvh, dh});
    tensor_t attn_val = ts._attn_val->slice(0, 0, tot_seq_len)->view({tot_seq_len, nh, dh});
    tensor_t attn_out = ts._attn_out->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    tensor_t x_attn = ts._x_attn->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    tensor_t m_norm = ts._m_norm->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    tensor_t gate = ts._gate->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    tensor_t up = ts._up->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    tensor_t swiglu = ts._swiglu->slice(0, 0, tot_seq_len)->view({tot_seq_len, di});
    tensor_t down =  ts._down->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    tensor_t x_mlp = ts._x_mlp->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});
    tensor_t dev_block_ids = ts._dev_block_ids->slice(0, 0, batch_size * max_block_num)->view({batch_size, max_block_num});
    tensor_t dev_cut_idx = ts._dev_cut_idx->slice(0, 0, batch_size + 1)->view({batch_size + 1});
    tensor_t dev_tot_len = ts._dev_tot_len->slice(0, 0, batch_size)->view({batch_size});
    tensor_t dev_token_ids = ts._dev_token_ids->slice(0, 0, tot_seq_len)->view({tot_seq_len});
    tensor_t dev_pos_ids = ts._dev_pos_ids->slice(0, 0, tot_seq_len)->view({tot_seq_len});
    tensor_t x =  ts._x->slice(0, 0, tot_seq_len)->view({tot_seq_len, hs});

    // 初始化设备端数据
    dev_block_ids->load(block_ids.data());
    dev_cut_idx->load(cut_idx.data());
    dev_tot_len->load(tot_len.data());
    dev_token_ids->load(token_ids.data());
    dev_pos_ids->load(pos_ids.data());

    // embedding: 根据 token ids 从 embedding 权重矩阵 中提取对应的语意向量
    ops::embedding(x, dev_token_ids, to_tensor(w.in_embed));

    // 计算 self_attention 所需的参数 scale
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    // 逐层执行 transformer 前向推导
    for (size_t layer = 0; layer < nlayer; ++layer) {
        // RMS-norm 均方根归一化
        ops::rms_norm(x_norm, x, to_tensor(w.attn_norm_w[layer]), epsilon);

        // 初始化 Q K V
        ops::linear(q, x_norm, to_tensor(w.attn_q_w[layer]), to_tensor(w.attn_q_b[layer]));
        ops::linear(k, x_norm, to_tensor(w.attn_k_w[layer]), to_tensor(w.attn_k_b[layer]));
        ops::linear(v, x_norm, to_tensor(w.attn_v_w[layer]), to_tensor(w.attn_v_b[layer]));

        tensor_t q_view = q->view({tot_seq_len, nh, dh});
        tensor_t k_view = k->view({tot_seq_len, nkvh, dh});
        tensor_t v_view = v->view({tot_seq_len, nkvh, dh});

        // 旋转位置编码 rope: 将位置信息融入进 Q / K 当中
        ops::rope(q_rope, q_view, dev_pos_ids, theta);
        ops::rope(k_rope, k_view, dev_pos_ids, theta);

        // 按块加载到 kv cache
        tensor_t k_layer = this->_k_cache[layer];
        tensor_t v_layer = this->_v_cache[layer];
        ops::kv_cache_move(k_layer, k_rope, dev_block_ids, dev_cut_idx, dev_pos_ids, max_seq_len);
        ops::kv_cache_move(v_layer, v_view, dev_block_ids, dev_cut_idx, dev_pos_ids, max_seq_len);

        // 自注意力: paged_attention(flash v2)
        ops::paged_attention(attn_val, q_rope, k_layer, v_layer, dev_block_ids, dev_cut_idx, dev_tot_len, max_seq_len, tot_block_num, scale, is_prefill,
                             ts._attn_acc, ts._attn_sum, ts._attn_max);

        // 得到多头注意力输出投影
        tensor_t attn_merge = attn_val->view({tot_seq_len, nh * dh});
        ops::linear(attn_out, attn_merge, to_tensor(w.attn_o_w[layer]), nullptr);

        // 实现残差连接: x(上一层信息) + attn_out(当前层信息)
        ops::add(x_attn, x, attn_out);
        std::swap(x, x_attn);

        // 多层感知机 MLP block
        ops::rms_norm(m_norm, x, to_tensor(w.mlp_norm_w[layer]), epsilon);
        ops::linear(gate, m_norm, to_tensor(w.mlp_gate_w[layer]), nullptr);
        ops::linear(up, m_norm, to_tensor(w.mlp_up_w[layer]), nullptr);
        ops::swiglu(swiglu, gate, up);
        ops::linear(down, swiglu, to_tensor(w.mlp_down_w[layer]), nullptr);
        ops::add(x_mlp, x, down);
        std::swap(x, x_mlp);
    }
    // 先去提取
    tensor_t x_part = ts._x_part->slice(0, 0, batch_size)->view({batch_size, hs});
    ops::embedding(x_part, dev_cut_idx->slice(0, 1, batch_size + 1), x, -1);

    // RMS-norm 均方根归一化
    tensor_t x_part_norm = ts._x_part_norm->slice(0, 0, batch_size)->view({batch_size, hs});
    ops::rms_norm(x_part_norm, x_part, to_tensor(w.out_norm_w), epsilon);

    // 把隐藏向量映射到词表维度（voc）生成未归一化的打分（logits）
    tensor_t logits = ts._logits->slice(0, 0, batch_size)->view({batch_size, voc});
    ops::linear(logits, x_part_norm, to_tensor(w.out_embed), nullptr);

    // 与 layer 路径共用采样后处理
    return sample_from_logits(logits, pack);
}

}