
#include "../sequence/sequence.hpp"
#include "qwen2.hpp"
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
            // 执行一次批处理前向传播
            std::vector<int64_t> token_ids = model->forward(pack, block_ids, is_prefill);
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

/**
 * 预处理块表 
 * 输入: 本次调度要计算的序列
 * 输出: 合并后的总块表，每个序列按照最长块表长度对齐
 * 
 * 重定义块表：[[row_size, block_table_index_0, ...], ...]
 * 将每行的第一个元素定义为：行的有效块表索引数，即有效块数
 */
std::vector<int64_t> Qwen2::prepare_block_table(const std::vector<seq_t> &seqs){
    const size_t batch_size = seqs.size();
    std::vector<int64_t> block_table(batch_size * MAX_BLOCK_NUM, -1);
    for(size_t i = 0; i < batch_size; ++ i){
        const size_t offset = i * MAX_BLOCK_NUM;
        // 新增：块数
        block_table[offset] = seqs[i]->block_ids().size();
        if(i > 0){
            block_table[offset] += block_table[offset - MAX_BLOCK_NUM];
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
Qwen2Pack Qwen2::prepare_prefill(const std::vector<seq_t> &seqs){
    const size_t batch_size = seqs.size();
    std::vector<int64_t> token_ids; // token信息
    std::vector<int64_t> pos_ids;   // 位置信息
    std::vector<int64_t> cut_idx(1, 0); // 切断点
    std::vector<int64_t> tot_len;       // 总长信息
    size_t max_seq_len = 0;         // 最大序列长度
    for(size_t i = 0; i < batch_size; ++ i){
        seq_t seq = seqs[i];
        const size_t seq_len = seq->scheduled_token_num();
        const size_t begin = seq->cached_token_num();
        const size_t end = begin + seq_len - 1;
        const std::vector<int64_t> &seq_token_ids = seq->token_ids();
        for(size_t j = begin; j <= end; ++ j){
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
    for(size_t i = 0; i < batch_size; ++ i){
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
Qwen2Pack Qwen2::prepare_decode(const std::vector<seq_t> &seqs){
    const size_t batch_size = seqs.size();
    std::vector<int64_t> token_ids;
    std::vector<int64_t> pos_ids;
    std::vector<int64_t> cut_idx(1, 0);
    std::vector<int64_t> tot_len;
    for(size_t i = 0; i < batch_size; ++ i){
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
    for(size_t i = 0; i < batch_size; ++ i){
        top_ks.push_back(seqs[i]->top_k());
        top_ps.push_back(seqs[i]->top_p());
        temps.push_back(seqs[i]->temperature());
        rngs.push_back(&seqs[i]->rng());
    }
    return Qwen2Pack{token_ids, pos_ids, cut_idx, tot_len, top_ks, top_ps, temps, rngs, 1};
}

std::vector<int64_t> Qwen2::forward(Qwen2Pack &pack, std::vector<int64_t> &block_ids, bool is_prefill){
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

    // 加载设备
    const llaisysDeviceType_t device = this->_device_type;

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
        ops::paged_attention(attn_val, q_rope, k_layer, v_layer, dev_block_ids, dev_cut_idx, dev_tot_len, max_seq_len, scale, is_prefill,
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
    // argmax
    // tensor_t max_idx = Tensor::create({batch_size, 1}, LLAISYS_DTYPE_I64, device, device_id);
    // tensor_t max_val = Tensor::create({batch_size, 1}, dtype, device, device_id);
    // ops::argmax(max_idx, max_val, last);
    // if (device != LLAISYS_DEVICE_CPU) {
    //     max_idx = max_idx->to(LLAISYS_DEVICE_CPU, 0);
    // }
    // int64_t result = reinterpret_cast<int64_t *>(max_idx->data())[0];

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
    for(size_t i = 0; i < batch_size; ++ i){
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

}