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

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>

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
             _device_ids(std::vector<int>(device_ids, device_ids + ndevice))
{
    init_weight_arrays(this->_weights, meta.nlayer);
}

Qwen2::~Qwen2() {
    free_weight_arrays(this->_weights);
}

Qwen2_t Qwen2::model(Qwen2Meta meta, llaisysDeviceType_t device, int *device_ids, int ndevice)
{
    return Qwen2_t(new Qwen2(meta, device, device_ids, ndevice));
}

int64_t Qwen2::forward(const int64_t *token_ids, const size_t ntoken){
    if (!token_ids || ntoken == 0) {
        return this->_meta.end_token;
    }
    const Qwen2Meta &meta = this->_meta;
    const Qwen2Weights &w = this->_weights;
    if (ntoken > meta.maxseq) {
        return meta.end_token;
    }
    // kv cache前缀匹配
    std::vector<int> block_ids;
    size_t start = this->_manager->allocate(token_ids, ntoken, block_ids);

    // 计算需要处理的长度: 总长-kv_cache缓存的长度
    const size_t seqlen = ntoken - start;
    if (seqlen == 0) {
        return token_ids[ntoken - 1];
    }

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
    const int64_t end_token = meta.end_token;

    // 加载设备
    const int device_id = this->device_id();
    const llaisysDeviceType_t device = this->_device_type;

    // 创建待处理的input ids张量: 输入信息
    tensor_t input_ids = Tensor::create({seqlen}, LLAISYS_DTYPE_I64, device, device_id);
    input_ids->load(token_ids + start);

    // 创建待处理的position ids张量: 位置信息
    std::vector<int64_t> pos_ids_vec(seqlen);
    std::iota(pos_ids_vec.begin(), pos_ids_vec.end(), start);
    
    tensor_t pos_ids = Tensor::create({seqlen}, LLAISYS_DTYPE_I64, device, device_id);
    pos_ids->load(pos_ids_vec.data());

    // embedding: 根据 input_ids 从 embedding 权重矩阵 中提取对应的语意向量
    tensor_t x = Tensor::create({seqlen, hs}, meta.dtype, device, device_id);
    ops::embedding(x, input_ids, to_tensor(w.in_embed));

    // 计算 self_attention 所需的参数 scale
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    // 初始化中间张量
    tensor_t x_norm = Tensor::create({seqlen, hs}, dtype, device, device_id);
    tensor_t q = Tensor::create({seqlen, nh * dh}, dtype, device, device_id);
    tensor_t k = Tensor::create({seqlen, nkvh * dh}, dtype, device, device_id);
    tensor_t v = Tensor::create({seqlen, nkvh * dh}, dtype, device, device_id);
    tensor_t q_rope = Tensor::create({seqlen, nh, dh}, dtype, device, device_id);
    tensor_t k_rope = Tensor::create({seqlen, nkvh, dh}, dtype, device, device_id);
    tensor_t attn_val = Tensor::create({seqlen, nh, dh}, dtype, device, device_id);
    tensor_t attn_out = Tensor::create({seqlen, hs}, dtype, device, device_id);
    tensor_t x_attn = Tensor::create({seqlen, hs}, dtype, device, device_id);
    tensor_t m_norm = Tensor::create({seqlen, hs}, dtype, device, device_id);
    tensor_t gate = Tensor::create({seqlen, di}, dtype, device, device_id);
    tensor_t up = Tensor::create({seqlen, di}, dtype, device, device_id);
    tensor_t swiglu = Tensor::create({seqlen, di}, dtype, device, device_id);
    tensor_t down = Tensor::create({seqlen, hs}, dtype, device, device_id);
    tensor_t x_mlp = Tensor::create({seqlen, hs}, dtype, device, device_id);

    // 加载设备块表
    tensor_t dev_block_ids = Tensor::create({block_ids.size()}, LLAISYS_DTYPE_I32, device, device_id);
    dev_block_ids->load(block_ids.data());

    // kv cache
    tensor_t k_cache = this->_manager->k_cache();
    tensor_t v_cache = this->_manager->v_cache();
    const size_t block_num = this->_manager->block_num();
    const size_t token_num = this->_manager->token_num();
    k_cache = k_cache->reshape({nlayer, block_num, token_num, nkvh, dh});
    v_cache = v_cache->reshape({nlayer, block_num, token_num, nkvh, dh});

    // 逐层执行 transformer 前向推导
    for (size_t layer = 0; layer < nlayer; ++layer) {
        // 自注意力机制 Attention
        ops::rms_norm(x_norm, x, to_tensor(w.attn_norm_w[layer]), epsilon);

        // 初始化 Q K V
        ops::linear(q, x_norm, to_tensor(w.attn_q_w[layer]), to_tensor(w.attn_q_b[layer]));
        ops::linear(k, x_norm, to_tensor(w.attn_k_w[layer]), to_tensor(w.attn_k_b[layer]));
        ops::linear(v, x_norm, to_tensor(w.attn_v_w[layer]), to_tensor(w.attn_v_b[layer]));

        tensor_t q_view = q->view({seqlen, nh, dh});
        tensor_t k_view = k->view({seqlen, nkvh, dh});
        tensor_t v_view = v->view({seqlen, nkvh, dh});

        // 旋转位置编码 rope: 将位置信息融入进 Q / K 当中
        ops::rope(q_rope, q_view, pos_ids, theta);
        ops::rope(k_rope, k_view, pos_ids, theta);

        // 更新 KV cache: slice 返回视图, rearrange 将新数据写入对应的位置
        tensor_t k_layer = k_cache->slice(0, layer, layer + 1)->reshape({block_num, token_num, nkvh, dh});
        tensor_t v_layer = v_cache->slice(0, layer, layer + 1)->reshape({block_num, token_num, nkvh, dh});

        // 按块加载到 kv cache
        for(size_t i = start; i < ntoken; i += token_num){
            size_t b = i / token_num;
            size_t l = std::max(start, b * token_num);
            size_t r = std::min(ntoken - 1, (b + 1) * token_num - 1);
            // 缓存块切分
            int block_id = block_ids[b];
            tensor_t k_block = k_layer->slice(0, block_id, block_id + 1)->reshape({token_num, nkvh, dh});
            tensor_t v_block = v_layer->slice(0, block_id, block_id + 1)->reshape({token_num, nkvh, dh});
            k_block = k_block->slice(0, l % token_num, r % token_num + 1)->reshape({r - l + 1, nkvh, dh});
            v_block = v_block->slice(0, l % token_num, r % token_num + 1)->reshape({r - l + 1, nkvh, dh});
            // 张量切分
            tensor_t k_slice = k_rope->slice(0, l - start, r - start + 1);
            tensor_t v_slice = v_view->slice(0, l - start, r - start + 1);
            ops::rearrange(k_block, k_slice);
            ops::rearrange(v_block, v_slice);
        } 

        /**
         * 推理流程 debug 复盘：
         * 
         *  先对整个链路去除 kv cache、paged attention 确保其他算子和流程的正确性
         * 
         *  由于现在还是单请求，并且 kv cache 就是连续的块，所以分块搬等价于连续搬
         * 
         *  所以我构造了一种情况：直接拿现成的 kv cache 构造成可被 flash attention 可用的情况
         * 
         *  验证了链路在连续搬运 kv cache 且 无 paged attention 的情况下是正确的
         * 
         *  然后替换为 paged attention 发现也可以跑通了 所以 paged attention 也是对的
         * 
         *  所以定位问题在分块搬运的过程中
         * 
         *  tensor 的 reshape 要求张量要连续 isContigous 连续会返回视图 复用内存 否则会调用 contigous 创建临时张量
         *  
         *  而分块搬运连续两次 slice 导致张量不再连续 并且在搬之前还 slice 了一次
         * 
         *  导致张量直接 reshape 就会调用 contigous 创建临时张量 而不是写入块中
         * 
         *  解决方案：每次 slice 了，就 reshape 一下 
         */

        // 去掉 layer 维度
        // k_layer = k_layer->reshape({block_num * token_num, nkvh, dh});
        // v_layer = v_layer->reshape({block_num * token_num, nkvh, dh});

        // tensor_t k_slice = k_layer->slice(0, start, ntoken);
        // tensor_t v_slice = v_layer->slice(0, start, ntoken);

        // ops::rearrange(k_slice, k_rope);
        // ops::rearrange(v_slice, v_view);

        // k_layer = k_layer->slice(0, 0, ntoken);
        // v_layer = v_layer->slice(0, 0, ntoken);

        // 自注意力: paged_attention(flash v2)
        ops::paged_attention(attn_val, q_rope, k_layer, v_layer, dev_block_ids, block_ids.size(), ntoken, scale);

        // ops::self_attention(attn_val, q_rope, k_layer, v_layer, scale);

        // 得到多头注意力输出投影
        tensor_t attn_merge = attn_val->view({seqlen, nh * dh});
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
    // Final norm and logits
    ops::rms_norm(x_norm, x, to_tensor(w.out_norm_w), epsilon);

    // 把隐藏向量映射到词表维度（voc）生成未归一化的打分（logits）
    tensor_t logits = Tensor::create({seqlen, voc}, dtype, device, device_id);
    ops::linear(logits, x_norm, to_tensor(w.out_embed), nullptr);

    // 选出最后一个 token 的预测结果
    tensor_t last = logits->slice(0, seqlen - 1, seqlen)->view({voc});
    tensor_t max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device, device_id);
    tensor_t max_val = Tensor::create({1}, dtype, device, device_id);
    ops::argmax(max_idx, max_val, last);

    int64_t result = end_token;
    if (device == LLAISYS_DEVICE_CPU) {
        result = reinterpret_cast<int64_t *>(max_idx->data())[0];
    } else {
        tensor_t host_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);
        core::context().setDevice(device, device_id);
        core::context().runtime().api()->memcpy_sync(
            host_idx->data(),
            max_idx->data(),
            sizeof(int64_t),
            LLAISYS_MEMCPY_D2H);
        result = reinterpret_cast<int64_t *>(host_idx->data())[0];
    }
    return result;
}

};