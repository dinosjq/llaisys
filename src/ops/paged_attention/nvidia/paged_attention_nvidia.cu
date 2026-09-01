#include "../../../utils.hpp"
#include "paged_attention_nvidia.cuh"
#include "utils/nvidia_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <mma.h>

using namespace nvcuda;     // wmma / fragment 等
using bf16 = __nv_bfloat16;
using fp16 = __nv_half;

constexpr size_t WMMA_M = 16;
constexpr size_t WMMA_N = 16;
constexpr size_t WMMA_K = 16;
constexpr size_t WARP_NUM = 8;
constexpr size_t PA_WARP_SIZE = 32;        // 前缀避免与 paged_attention_nvidia.cu 顶层常量冲突
constexpr size_t PA_HEAD_DIM_MAX = 128;

template<typename T, size_t HEAD_DIM, typename IN>
__device__ __forceinline__ void load_kv_cache(const IN *__restrict__ _k_cache, 
                                              const IN *__restrict__ _v_cache, 
                                              const float *__restrict__ _k_scale,
                                              const float *__restrict__ _v_scale,
                                              const int64_t *block_ids,
                                              T (*s_k)[HEAD_DIM],
                                              T (*s_v)[HEAD_DIM],
                                              const size_t _block_size,
                                              const size_t _nkvh,
                                              const size_t nkvh,
                                              const size_t tile_id,
                                              const size_t warp_id,
                                              const size_t lane_id,
                                              const size_t totlen) 
{
    if (warp_id < WARP_NUM / 2) return;
#pragma unroll
    for (size_t row = (warp_id & 3); row < WMMA_N; row += WARP_NUM / 2) {
        const size_t token_id = tile_id * WMMA_N + row;
        const size_t col = lane_id << 2;
        if (token_id >= totlen) {
            // 越界 KV 槽（tile 尾部不足 16 个 token）：置 0，避免垃圾/NaN 经 PV 的 0*NaN=NaN 污染累加器
#pragma unroll
            for (size_t j = 0; j < 4; ++j) {
                s_k[row][col + j] = T(0);
                s_v[row][col + j] = T(0);
            }
            continue;
        }
        const size_t table_id = token_id / _block_size + 1;
        const size_t block_id = block_ids[table_id];
        const size_t kv_offset = (block_id * _block_size + token_id % _block_size) * _nkvh + nkvh;
        if constexpr (std::is_same_v<T, IN>) {
            // 非量化 (bf16/fp16, 2B): float2 一次拷 4 个元素 (8B 向量化访存)
            const IN *k_cache = _k_cache + kv_offset * HEAD_DIM;
            const IN *v_cache = _v_cache + kv_offset * HEAD_DIM;
            *reinterpret_cast<float2 *>(&s_k[row][col]) = *reinterpret_cast<const float2 *>(&k_cache[col]);
            *reinterpret_cast<float2 *>(&s_v[row][col]) = *reinterpret_cast<const float2 *>(&v_cache[col]);
        } else {  // 量化 (IN = int8): int 打包 4 个 int8 一次加载 + 反量化
            const IN *k_cache = _k_cache + kv_offset * HEAD_DIM;
            const IN *v_cache = _v_cache + kv_offset * HEAD_DIM;
            const float k_scale = _k_scale[kv_offset];
            const float v_scale = _v_scale[kv_offset];
            const int k4 = *reinterpret_cast<const int *>(&k_cache[col]);
            const int v4 = *reinterpret_cast<const int *>(&v_cache[col]);
            const int8_t *kb = reinterpret_cast<const int8_t *>(&k4);
            const int8_t *vb = reinterpret_cast<const int8_t *>(&v4);
#pragma unroll
            for (size_t j = 0; j < 4; ++j) {
                s_k[row][col + j] = T((float)kb[j] * k_scale);
                s_v[row][col + j] = T((float)vb[j] * v_scale);
            }
        }
    }
}

// 自定义 paged_attention 以 flash_attention_v2 为基础
// 使用 tensor core 进行优化, 暂不支持量化
template <typename T, size_t HEAD_DIM, bool QUANT>
__global__ void paged_attention_kernel(T *__restrict__ _attn, 
                                       const T *__restrict__ _q, 
                                       const void *__restrict__ _k_cache, 
                                       const void *__restrict__ _v_cache, 
                                       const float *__restrict__ _k_scale,
                                       const float *__restrict__ _v_scale,
                                       const int64_t *_block_ids,
                                       const int64_t *_cut_idx,
                                       const int64_t *_tot_len,
                                       const size_t _block_size,
                                       const size_t _batch_size,
                                       const size_t _table_width,
                                       const float _scale, 
                                       const size_t _nh, 
                                       const size_t _nkvh) 
{
    __shared__ T s_q[WARP_NUM / 2][WMMA_M][HEAD_DIM];
    __shared__ T s_k[2][WMMA_N][HEAD_DIM];
    __shared__ T s_v[2][WMMA_N][HEAD_DIM];
    __shared__ T s_p[WARP_NUM / 2][WMMA_M][WMMA_N];
    __shared__ float s_score[WARP_NUM / 2][WMMA_M][WMMA_N];
    __shared__ float s_max[WARP_NUM / 2][WMMA_M];
    __shared__ float s_sum[WARP_NUM / 2][WMMA_M];
    __shared__ float s_rescale[WARP_NUM / 2][WMMA_M];

    const size_t tid = threadIdx.x;
    const size_t lane_id = tid & 31;
    const size_t warp_id = tid >> 5;

    const bool is_consumer = warp_id < (WARP_NUM / 2);
    const bool is_producer = !is_consumer;

    const size_t task_id = blockIdx.x;
    const size_t nh = blockIdx.y;
    const size_t batch_id = blockIdx.z;

    const size_t nkvh = nh / (_nh / _nkvh);
    const size_t prelen = _cut_idx[batch_id];
    const size_t seqlen = _cut_idx[batch_id + 1] - prelen;
    const size_t totlen = _tot_len[batch_id];
    const int64_t *block_ids = _block_ids + batch_id * _table_width;

    const size_t begin_id = (task_id * (WARP_NUM / 2) + warp_id) * WMMA_M;
    const bool valid = is_consumer && (begin_id < seqlen);

    const size_t q_end = (begin_id + WMMA_M < seqlen) ? (begin_id + WMMA_M) : seqlen;
    const size_t kv_end = totlen - seqlen + q_end;
    const size_t kv_tile_num = (totlen + WMMA_N - 1) / WMMA_N;

    // load q (global -> smem)
    if (valid) {
#pragma unroll
        for (size_t row = 0; row < WMMA_M; ++ row) {
            if (begin_id + row < seqlen) {
                const T *q = _q + ((prelen + begin_id + row) * _nh + nh) * HEAD_DIM;
#pragma unroll
                for (size_t col = lane_id; col < HEAD_DIM; col += 32) {
                    s_q[warp_id][row][col] = q[col];
                }
            } else {
#pragma unroll
                for (size_t col = lane_id; col < HEAD_DIM; col += 32) {
                    s_q[warp_id][row][col] = T(0);
                }
            }
        }
    }

    // wmma fragments
    constexpr size_t FRAG_NUM = HEAD_DIM / WMMA_K;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> q_frag[FRAG_NUM];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[FRAG_NUM];

    // load q (smem -> frag(register))
    if (valid) {
#pragma unroll
        for (size_t i = 0; i < FRAG_NUM; ++ i) {
            wmma::load_matrix_sync(q_frag[i], reinterpret_cast<T *>(s_q[warp_id]) + i * WMMA_K, HEAD_DIM);
        }
        if (lane_id < WMMA_M) {
            s_max[warp_id][lane_id] = -INFINITY;
            s_sum[warp_id][lane_id] = 0.0f;
            s_rescale[warp_id][lane_id] = 1.0;
        }
        // init acc, max, sum
#pragma unroll
        for (size_t i = 0; i < FRAG_NUM; ++ i) {
            wmma::fill_fragment(acc_frag[i], 0.0f);
        }
    }

    // pre load kv cache
    if (is_producer) {
        if(QUANT) load_kv_cache<T, HEAD_DIM, int8_t>(reinterpret_cast<const int8_t*>(_k_cache), reinterpret_cast<const int8_t*>(_v_cache), _k_scale, _v_scale, block_ids, s_k[0], s_v[0], _block_size, _nkvh, nkvh, 0, warp_id, lane_id, totlen);
        else load_kv_cache<T, HEAD_DIM, T>(reinterpret_cast<const T*>(_k_cache), reinterpret_cast<const T*>(_v_cache), _k_scale, _v_scale, block_ids, s_k[0], s_v[0], _block_size, _nkvh, nkvh, 0, warp_id, lane_id, totlen);
    }
    __syncthreads();

    for (size_t tile_id = 0; tile_id < kv_tile_num; ++ tile_id) {
        const size_t stage = tile_id & 1;
        
        // load next stage kv cache
        if(is_producer && tile_id + 1 < kv_tile_num){
            if(QUANT) load_kv_cache<T, HEAD_DIM, int8_t>(reinterpret_cast<const int8_t*>(_k_cache), reinterpret_cast<const int8_t*>(_v_cache), _k_scale, _v_scale, block_ids, s_k[stage ^ 1], s_v[stage ^ 1], _block_size, _nkvh, nkvh, tile_id + 1, warp_id, lane_id, totlen);
            else load_kv_cache<T, HEAD_DIM, T>(reinterpret_cast<const T*>(_k_cache), reinterpret_cast<const T*>(_v_cache), _k_scale, _v_scale, block_ids, s_k[stage ^ 1], s_v[stage ^ 1], _block_size, _nkvh, nkvh, tile_id + 1, warp_id, lane_id, totlen);
        }

        // compute current stage kv tile
        if (valid) {
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::col_major> k_frag;
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;

            wmma::fill_fragment(s_frag, 0.0f);
#pragma unroll
            for (size_t i = 0; i < FRAG_NUM; ++ i) {
                wmma::load_matrix_sync(k_frag, reinterpret_cast<T *>(s_k[stage]) + i * WMMA_K, HEAD_DIM);
                wmma::mma_sync(s_frag, q_frag[i], k_frag, s_frag);
            }
            wmma::store_matrix_sync(reinterpret_cast<float *>(s_score[warp_id]), s_frag, WMMA_N, wmma::mem_row_major);
            __syncwarp();

            if (lane_id < WMMA_M) {
                const size_t row_kv_end = (totlen - seqlen) + begin_id + lane_id + 1;   // 该行因果边界
                float row_max = -INFINITY;
#pragma unroll
                for (size_t i = 0; i < WMMA_N; ++ i) {
                    const size_t token_id = tile_id * WMMA_N + i;
                    // 掩码 + row_max 都只用有效（过去）token；未来 token 记 -INF 不抬高 max
                    const float val = (token_id < row_kv_end) ? s_score[warp_id][lane_id][i] * _scale : -INFINITY;
                    s_score[warp_id][lane_id][i] = val;
                    row_max = fmaxf(row_max, val);
                }
                const float old_max = s_max[warp_id][lane_id];
                const float new_max = fmaxf(row_max, old_max);
                const float alpha = __expf(old_max - new_max);
                float row_sum = 0.0f;
#pragma unroll
                for (size_t i = 0; i < WMMA_N; ++ i) {
                    const float val = __expf(s_score[warp_id][lane_id][i] - new_max);
                    s_score[warp_id][lane_id][i] = val;
                    s_p[warp_id][lane_id][i] = T(val);
                    row_sum += val;
                }
                s_max[warp_id][lane_id] = new_max;
                s_sum[warp_id][lane_id] = s_sum[warp_id][lane_id] * alpha + row_sum;
                s_rescale[warp_id][lane_id] = alpha;
            }
            __syncwarp();

#pragma unroll
            for (size_t i = 0; i < FRAG_NUM; ++ i) {
                for (size_t j = 0; j < acc_frag[i].num_elements; ++ j) {
                    int row = (lane_id >> 2) + 8 * ((j & 3) >> 1);
                    acc_frag[i].x[j] *= s_rescale[warp_id][row];
                }
            }

            wmma::load_matrix_sync(p_frag, reinterpret_cast<T *>(s_p[warp_id]), WMMA_N);
#pragma unroll
            for (size_t i = 0; i < FRAG_NUM; ++ i) {
                wmma::load_matrix_sync(v_frag, reinterpret_cast<T *>(s_v[stage]) + i * WMMA_K, HEAD_DIM);
                wmma::mma_sync(acc_frag[i], p_frag, v_frag, acc_frag[i]);
            }
        }
        __syncthreads();
    }

    if (valid) {
#pragma unroll
        for (size_t i = 0; i < FRAG_NUM; ++ i) {
            for (size_t j = 0; j < acc_frag[i].num_elements; ++ j) {
                size_t row = (lane_id >> 2) + 8 * ((j & 3) >> 1);
                size_t col = (j >> 2) * 8 + (lane_id & 3) * 2 + (j & 1);
                if (begin_id + row < seqlen) {
                    size_t offset = ((prelen + begin_id + row) * _nh + nh) * HEAD_DIM + (i * WMMA_K) + col;
                    _attn[offset] = T(acc_frag[i].x[j] / s_sum[warp_id][row]);
                }
            }
        }
    }
}

template <typename T>
void paged_attention_launch(std::byte *attn_val,       // 输出 (tot_seqlen, nh, dv)
                            const std::byte *q,        // q (tot_seqlen, nh, d)
                            const std::byte *k_cache,  // k底层缓存 (tot_len, nkvh, d) 
                            const std::byte *v_cache,  // q底层缓存 (tot_len, nkvh, dv)
                            const std::byte *block_ids,// 块表 (batch, block_ids_size | block_ids)  kv cache布局: (block_num, block_size, nkvh, d | dv) 
                            const std::byte *cut_idx,  // 记录对应序列起始位置，也可计算seqlen
                            const std::byte *tot_len,  // 已计算kv总长
                            const size_t block_size,   // 每个块的token数 (批次无关)
                            const size_t batch_size,   // 批次大小 (批次有关)
                            const size_t table_width,  // 块表最多块数 (批次无关)
                            const size_t tot_block_num,// 批次总 kv 块数
                            const size_t max_seq_len,  // 最大序列长
                            const float &scale,        // 缩放因子
                            const size_t &nh,          // 其他参数 ...
                            const size_t &dv, 
                            const size_t &d, 
                            const size_t &nkvh,
                            const std::byte *k_scale = nullptr,
                            const std::byte *v_scale = nullptr)
{
    if (d == 0 || dv == 0 || d > PA_HEAD_DIM_MAX || dv > PA_HEAD_DIM_MAX || d % 32 != 0 || dv % 32 != 0 || block_size == 0 || nkvh == 0 || nh % nkvh != 0) {
        throw std::invalid_argument("paged_attention requires d,dv in {32,64,96,128}, " "block_size > 0, nkvh > 0, and nh divisible by nkvh");
    }
    if (batch_size == 0 || nh == 0 || tot_block_num == 0) {
        return;
    }

    auto *d_attn = reinterpret_cast<T *>(attn_val);
    const auto *d_q = reinterpret_cast<const T *>(q);
    const void *d_k_cache = reinterpret_cast<const void *>(k_cache);
    const void *d_v_cache = reinterpret_cast<const void *>(v_cache);
    const auto *d_k_scale = reinterpret_cast<const float *>(k_scale);
    const auto *d_v_scale = reinterpret_cast<const float *>(v_scale);
    const auto *d_block_ids = reinterpret_cast<const int64_t *>(block_ids);
    const auto *d_cut_idx = reinterpret_cast<const int64_t *>(cut_idx);
    const auto *d_tot_len = reinterpret_cast<const int64_t *>(tot_len);

    const size_t max_tile_num = (max_seq_len + WMMA_M - 1) / WMMA_M;
    const size_t max_task_num = (max_tile_num + WARP_NUM / 2 - 1) / (WARP_NUM / 2);

    dim3 blockDim(WARP_NUM * PA_WARP_SIZE);
    dim3 gridDim(max_task_num, nh, batch_size);

    if (k_scale != nullptr) {
        paged_attention_kernel<T, 128, true><<<gridDim, blockDim>>>(
            d_attn, d_q, d_k_cache, d_v_cache, d_k_scale, d_v_scale, d_block_ids, d_cut_idx,
            d_tot_len, block_size, batch_size, table_width, scale, nh, nkvh);
    } else {
        paged_attention_kernel<T, 128, false><<<gridDim, blockDim>>>(
            d_attn, d_q, d_k_cache, d_v_cache, d_k_scale, d_v_scale, d_block_ids, d_cut_idx,
            d_tot_len, block_size, batch_size, table_width, scale, nh, nkvh);
    }

    CUDA_CHECK(cudaGetLastError());
}


namespace llaisys::ops::nvidia {
void paged_attention(std::byte *attn_val,       // 输出 (tot_seqlen, nh, dv)
                     const std::byte *q,        // q (tot_seqlen, nh, d)
                     const std::byte *k_cache,  // k底层缓存 (tot_len, nkvh, d) 
                     const std::byte *v_cache,  // q底层缓存 (tot_len, nkvh, dv)
                     const std::byte *block_ids,// 块表 (batch, block_ids_size | block_ids)  kv cache布局: (block_num, block_size, nkvh, d | dv) 
                     const std::byte *cut_idx,  // 记录对应序列起始位置，也可计算seqlen
                     const std::byte *tot_len,  // 已计算kv总长
                     const size_t block_size,   // 每个块的token数
                     const size_t batch_size,   // 批次大小
                     const size_t table_width,  // 块表最多块数
                     const size_t tot_block_num,// 批次总 kv 块数
                     const size_t max_seq_len,  // 最大序列长度
                     llaisysDataType_t dtype,   // 权重数据类型
                     const float &scale,        // 缩放因子
                     const size_t &nh,          // 其他参数 ...
                     const size_t &dv, 
                     const size_t &d, 
                     const size_t &nkvh,
                     const std::byte *k_scale,
                     const std::byte *v_scale)
{
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        // WMMA m16n16k16 tensor core 不支持 fp32 (仅 bf16/fp16), F32 走不了本 kernel
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    case LLAISYS_DTYPE_BF16:
        return paged_attention_launch<bf16>(attn_val, q, k_cache, v_cache, block_ids, cut_idx, tot_len, block_size,
                                             batch_size, table_width, tot_block_num, max_seq_len, scale, nh, dv, d, nkvh, k_scale, v_scale);
    case LLAISYS_DTYPE_F16:
        return paged_attention_launch<fp16>(attn_val, q, k_cache, v_cache, block_ids, cut_idx, tot_len, block_size,
                                             batch_size, table_width, tot_block_num, max_seq_len, scale, nh, dv, d, nkvh, k_scale, v_scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia