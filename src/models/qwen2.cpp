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

static int64_t _random_sample(const std::vector<int64_t> &candidates, const std::vector<float> &weights, float top_p){
    const size_t n = candidates.size();
    std::vector<std::pair<float, int64_t>> p(n);
    for(size_t i = 0; i < n; ++ i){
        p[i] = std::make_pair(weights[i], candidates[i]);
    }
    auto cmp = [](const std::pair<float, int64_t> &p1, const std::pair<float, int64_t> &p2) -> bool{
        return p1.first > p2.first;
    };
    std::sort(p.begin(), p.end(), cmp);
    for(size_t i = 1; i < n; ++ i){
        p[i].first += p[i - 1].first;
    }
    size_t lim = lower_bound(p.begin(), p.end(), std::make_pair(top_p, int64_t(0))) - p.begin() + 1;
    return p[rand() % lim].second;
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
    const size_t nlayer = meta.nlayer;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t token_num = KV_CACHE_TOKEN_NUM;
    const size_t block_num = KV_CACHE_BLOCK_NUM;
    const size_t block_size = block_num * token_num * nkvh * dh;
    init_weight_arrays(this->_weights, nlayer);
    this->_k_cache = std::vector<tensor_t>(nlayer, nullptr);
    this->_v_cache = std::vector<tensor_t>(nlayer, nullptr);
    tensor_t cache = Tensor::create({2, nlayer, block_size}, meta.dtype, device, device_ids[0]);
    tensor_t k_cache = cache->slice(0, 0, 1)->reshape({nlayer, block_size});
    tensor_t v_cache = cache->slice(0, 1, 2)->reshape({nlayer, block_size});
    for(size_t i = 0; i < nlayer; ++ i){
        this->_k_cache[i] = k_cache->slice(0, i, i + 1)->reshape({block_num, token_num, nkvh, dh});
        this->_v_cache[i] = v_cache->slice(0, i, i + 1)->reshape({block_num, token_num, nkvh, dh});
    }
    this->_scheduler = std::make_shared<Scheduler>(token_num, block_num, BATCH_MAX_TOKEN_NUM, BATCH_MAX_SEQ_NUM, meta.end_token);
    this->start();
}

Qwen2::~Qwen2() {
    this->stop();
    if (this->_worker.joinable()) {
        this->_worker.join();
    }
    free_weight_arrays(this->_weights);
}  

void Qwen2::start(){
    if (this->_running) {
        return;
    }
    auto loop = [](Qwen2 *model)->void{
        while(model->_running){
            auto [seqs, is_prefill] = model->_scheduler->schedule();
            if(seqs.empty()) continue;
            auto block_ids = model->prepare_block_table(seqs);
            Qwen2Pack pack = is_prefill ? model->prepare_prefill(seqs) : model->prepare_decode(seqs);
            std::vector<int64_t> token_ids = model->forward(pack, block_ids);
            model->_scheduler->postprocess(seqs, token_ids, is_prefill);
        }
    };
    this->_running = true;
    this->_worker = std::thread(loop, this);
}

void Qwen2::stop(){
    this->_running = false;
}

int64_t Qwen2::request(int64_t *token_ids, size_t ntoken){
    long long hash = 0;
    std::shared_ptr<Sequence> seq;
    {
        std::lock_guard<std::mutex> lk(this->_hash_lock);
        for(size_t i = 0; i < ntoken; ++ i){
            hash = (hash * Prime + token_ids[i]) % mod;
            if(_hash_2_seq.count(hash)){
                seq = _hash_2_seq[hash];
            }
        }
        if(seq == nullptr){
            seq = std::make_shared<Sequence>(token_ids, ntoken);
            _hash_2_seq[seq->prompt_hash()] = seq;
            this->_scheduler->add(seq);
        }
    }

    while(seq->token_num() <= ntoken){
        std::this_thread::yield();
    }

    int64_t token = seq->token_ids()[ntoken];
    if(token == this->_meta.end_token){
        std::lock_guard<std::mutex> lk(this->_hash_lock);
        this->_hash_2_seq.erase(seq->prompt_hash());
    }
    return token;
}

std::vector<int> Qwen2::prepare_block_table(const std::vector<seq_t> &seqs){
    const size_t batch_size = seqs.size();
    size_t max_block_num = 0;
    for(size_t i = 0; i < batch_size; ++ i){
        max_block_num = std::max(max_block_num, seqs[i]->block_ids().size());
    }
    std::vector<int> block_table(batch_size * max_block_num);
    for(size_t i = 0; i < batch_size; ++ i){
        std::copy(seqs[i]->block_ids().begin(), seqs[i]->block_ids().end(), block_table.begin() + i * max_block_num);
    }
    return block_table;
}

Qwen2Pack Qwen2::prepare_prefill(const std::vector<seq_t> &seqs){
    const size_t batch_size = seqs.size();
    std::vector<int64_t> token_ids;
    std::vector<int64_t> pos_ids;
    std::vector<int> cut_idx(1, 0);
    std::vector<int> tot_len;
    size_t max_seq_len = 0;
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
    return Qwen2Pack{token_ids, pos_ids, cut_idx, tot_len, max_seq_len};
}

Qwen2Pack Qwen2::prepare_decode(const std::vector<seq_t> &seqs){
    const size_t batch_size = seqs.size();
    std::vector<int64_t> token_ids;
    std::vector<int64_t> pos_ids;
    std::vector<int> cut_idx(1, 0);
    std::vector<int> tot_len;
    for(size_t i = 0; i < batch_size; ++ i){
        seq_t seq = seqs[i];
        const size_t idx = seq->cached_token_num();
        const std::vector<int64_t> &seq_token_ids = seq->token_ids();
        token_ids.push_back(seq_token_ids[idx]);
        pos_ids.push_back(idx);
        cut_idx.push_back(cut_idx.back() + 1);
        tot_len.push_back(seq->token_num());
    }
    return Qwen2Pack{token_ids, pos_ids, cut_idx, tot_len, 1};
}

std::vector<int64_t> Qwen2::forward(Qwen2Pack &pack, std::vector<int> &block_ids){
    // 加载元数据 + 权重
    const Qwen2Meta &meta = this->_meta;
    const Qwen2Weights &w = this->_weights;

    // 预处理信息
    const std::vector<int64_t> &token_ids = pack.token_ids;
    const std::vector<int64_t> &pos_ids = pack.pos_ids;
    const std::vector<int> &cut_idx = pack.cut_idx;
    const std::vector<int> &tot_len = pack.tot_len;
    const size_t &max_seq_len = pack.max_seq_len; 
    const size_t tot_seq_len = token_ids.size();
    const size_t batch_size = tot_len.size();
    const size_t max_block_num = block_ids.size() / batch_size;

    // 加载其他参数
    llaisysDataType_t dtype = meta.dtype;
    const size_t nlayer = meta.nlayer;
    const size_t hs = meta.hs;
    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t di = meta.di;
    const size_t voc = meta.voc;
    const float epsilon = meta.epsilon;
    const float theta = meta.theta;
    const size_t token_num = this->_scheduler->token_num();

    // 加载设备
    const int device_id = this->_device_ids[0];
    const llaisysDeviceType_t device = this->_device_type;

    // 初始化中间张量
    tensor_t x_norm = Tensor::create({tot_seq_len, hs}, dtype, device, device_id);
    tensor_t q = Tensor::create({tot_seq_len, nh * dh}, dtype, device, device_id);
    tensor_t k = Tensor::create({tot_seq_len, nkvh * dh}, dtype, device, device_id);
    tensor_t v = Tensor::create({tot_seq_len, nkvh * dh}, dtype, device, device_id);
    tensor_t q_rope = Tensor::create({tot_seq_len, nh, dh}, dtype, device, device_id);
    tensor_t k_rope = Tensor::create({tot_seq_len, nkvh, dh}, dtype, device, device_id);
    tensor_t attn_val = Tensor::create({tot_seq_len, nh, dh}, dtype, device, device_id);
    tensor_t attn_out = Tensor::create({tot_seq_len, hs}, dtype, device, device_id);
    tensor_t x_attn = Tensor::create({tot_seq_len, hs}, dtype, device, device_id);
    tensor_t m_norm = Tensor::create({tot_seq_len, hs}, dtype, device, device_id);
    tensor_t gate = Tensor::create({tot_seq_len, di}, dtype, device, device_id);
    tensor_t up = Tensor::create({tot_seq_len, di}, dtype, device, device_id);
    tensor_t swiglu = Tensor::create({tot_seq_len, di}, dtype, device, device_id);
    tensor_t down = Tensor::create({tot_seq_len, hs}, dtype, device, device_id);
    tensor_t x_mlp = Tensor::create({tot_seq_len, hs}, dtype, device, device_id);

    // 初始化设备端数据
    tensor_t dev_block_ids = Tensor::create({batch_size, max_block_num}, LLAISYS_DTYPE_I32, device, device_id);
    dev_block_ids->load(block_ids.data());

    tensor_t dev_cut_idx = Tensor::create({batch_size + 1}, LLAISYS_DTYPE_I32, device, device_id);
    dev_cut_idx->load(cut_idx.data());

    tensor_t dev_tot_len = Tensor::create({batch_size}, LLAISYS_DTYPE_I32, device, device_id);
    dev_tot_len->load(tot_len.data());

    // 创建待处理的token ids张量: 输入信息
    tensor_t dev_token_ids = Tensor::create({tot_seq_len}, LLAISYS_DTYPE_I64, device, device_id);
    dev_token_ids->load(token_ids.data());

    // 创建待处理的position ids张量: 位置信息
    tensor_t dev_pos_ids = Tensor::create({tot_seq_len}, LLAISYS_DTYPE_I64, device, device_id);
    dev_pos_ids->load(pos_ids.data());

    // embedding: 根据 token ids 从 embedding 权重矩阵 中提取对应的语意向量
    tensor_t x = Tensor::create({tot_seq_len, hs}, meta.dtype, device, device_id);
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

        for(size_t i = 0; i < batch_size; ++ i){
            size_t offset = cut_idx[i];
            size_t begin = pos_ids[cut_idx[i]];
            size_t end = pos_ids[cut_idx[i + 1] - 1];
            for(size_t j = begin; j <= end; j += token_num){
                size_t b = j / token_num;
                size_t l = std::max(begin, b * token_num);
                size_t r = std::min(end, (b + 1) * token_num - 1);
                int block_id = block_ids[i * max_block_num + b];
                if(block_id == -1) break;
                tensor_t k_block = k_layer->slice(0, block_id, block_id + 1)->reshape({token_num, nkvh, dh});
                tensor_t v_block = v_layer->slice(0, block_id, block_id + 1)->reshape({token_num, nkvh, dh});
                k_block = k_block->slice(0, l % token_num, r % token_num + 1)->reshape({r - l + 1, nkvh, dh});
                v_block = v_block->slice(0, l % token_num, r % token_num + 1)->reshape({r - l + 1, nkvh, dh});
                // 张量切分
                tensor_t k_slice = k_rope->slice(0, offset + l - begin, offset + r - begin + 1);
                tensor_t v_slice = v_view->slice(0, offset + l - begin, offset + r - begin + 1);
                ops::rearrange(k_block, k_slice);
                ops::rearrange(v_block, v_slice);
            }
        }

        // 自注意力: paged_attention(flash v2)
        ops::paged_attention(attn_val, q_rope, k_layer, v_layer, dev_block_ids, dev_cut_idx, dev_tot_len, max_seq_len, scale);

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

    // RMS-norm 均方根归一化
    ops::rms_norm(x_norm, x, to_tensor(w.out_norm_w), epsilon);

    // 把隐藏向量映射到词表维度（voc）生成未归一化的打分（logits）
    tensor_t logits = Tensor::create({tot_seq_len, voc}, dtype, device, device_id);
    ops::linear(logits, x_norm, to_tensor(w.out_embed), nullptr);
    
    // 选出最后一个 token 的预测结果
    tensor_t last = Tensor::create({batch_size, voc}, dtype, device, device_id);
    for(size_t i = 0; i < batch_size; ++ i){
        tensor_t target = logits->slice(0, cut_idx[i + 1] - 1, cut_idx[i + 1])->view({voc});
        last->slice(0, i, i + 1)->reshape({voc})->load(target->data());
    }

    // argmax
    // tensor_t max_idx = Tensor::create({batch_size, 1}, LLAISYS_DTYPE_I64, device, device_id);
    // tensor_t max_val = Tensor::create({batch_size, 1}, dtype, device, device_id);
    // ops::argmax(max_idx, max_val, last);
    // if (device != LLAISYS_DEVICE_CPU) {
    //     max_idx = max_idx->to(LLAISYS_DEVICE_CPU, 0);
    // }
    // int64_t result = reinterpret_cast<int64_t *>(max_idx->data())[0];

    // top_k
    constexpr size_t K = TOP_K;
    tensor_t top_idx = Tensor::create({batch_size, K}, LLAISYS_DTYPE_I64, device, device_id);
    tensor_t top_val = Tensor::create({batch_size, K}, dtype, device, device_id);
    ops::topk(top_idx, top_val, last, K);
    if (device != LLAISYS_DEVICE_CPU) {
        top_idx = top_idx->to(LLAISYS_DEVICE_CPU, 0);
        top_val = top_val->to(LLAISYS_DEVICE_CPU, 0);
    }
    std::vector<int64_t> result(batch_size);
    for(size_t i = 0; i < batch_size; ++ i){
        std::vector<int64_t> idx = top_idx->slice(0, i, i + 1)->reshape({K})->to_vector<int64_t>();
        std::vector<float> val = top_val->slice(0, i, i + 1)->reshape({K})->to_vector<float>();
        result[i] = _random_sample(idx, val, TOP_P);
    }
    return result;
}   

}