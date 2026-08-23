#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>


#include "flash_decoding_nvidia.cuh"
#include "utils/nvidia_utils.cuh"
#include "utils.hpp"

__device__ __forceinline__ float warp_reduce(float val) {
#pragma unroll
    for (size_t offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

static constexpr size_t BLOCK_DIM = 256;

/**
 *  flash decoding based on:
 *  1). flash attention v2;
 *  2). paged_attention;
 *          => kv cache block: token_num = 32
 *  3). two buffer: consumer + producer;
 *  4). less register/smem
 *          => warp contain data
 *  5). totlen dim cut: parallel + reduce;
 *  6). cp.async;
 *  7). grid change-able, coalesce, template... 
 */
template <typename T, 
          size_t HEAD_DIM, 
          size_t HEAD_NUM, 
          size_t TILE_K, 
          size_t COALESCE>
__global__ void flash_decoding_parallel_kernel(float *__restrict__ _attn_acc, 
                                               float *__restrict__ _attn_sum,
                                               float *__restrict__ _attn_max,
                                               const T *__restrict__ _q, 
                                               const T *__restrict__ _k_cache, 
                                               const T *__restrict__ _v_cache, 
                                               const int64_t *_block_ids,
                                               const int64_t *_cut_idx,
                                               const int64_t *_tot_len,
                                               const size_t _token_num,
                                               const size_t _batch_size,
                                               const size_t _max_block_num,
                                               const float _scale, 
                                               const size_t _nh,
                                               const size_t _nkvh) 
{
    // shared memory
    __shared__ T *s_k[2][TILE_K * HEAD_DIM];
    __shared__ T *s_v[2][TILE_K * HEAD_DIM];
    __shared__ float s_sum[HEAD_NUM];
    __shared__ float s_max[HEAD_NUM];

    // register
    float reg_q[(HEAD_DIM + 31) >> 5];
    float reg_acc[(HEAD_DIM + 31) >> 5];
    float reg_sum = 0.0;
    float reg_max = -INFINITY;
    float alpha = 1.0f;
    float beta = 0.0f;

    const size_t tid = threadIdx.x;
    const size_t warp_id = tid >> 5;
    const size_t lane_id = tid & 31;

    // calc: batch_id
    const size_t task_id = blockIdx.x;
    size_t batch_id = -1;
    size_t pre_task_num = 0;
    for (size_t i = 0; i < _batch_size; ++ i) {
        size_t task_num = (static_cast<size_t>(_block_ids[i * _max_block_num]) + COALESCE - 1) / COALESCE;
        if (pre_task_num + task_num > task_id) {
            batch_id = i;
            break;
        }
        pre_task_num += task_num;
    }
    if (batch_id == -1) {
        return;
    }

    // calc: begin table_id, block_num  
    const size_t begin_id = (task_id - pre_task_num) * COALESCE;
    size_t tot_block_num = _block_ids[i * _max_block_num] - ((batch_id > 0) ? _block_ids[(i - 1) * _max_block_num] : 0);
    tot_block_num = min(tot_block_num - begin_id, COALESCE);

    // calc: nh, nkvh
    const size_t nh = blockIdx.y * HEAD_NUM + warp_id;
    const size_t nkvh = nh / (_nh / _nkvh);

    // prepare    
    const size_t totlen = _totlen[batch_id];
    const size_t limit = totlen - 1;

    // load: q; init: reg_acc
    const T *q = _q + (_cut_idx[batch_id] * _nh + nh) * HEAD_DIM;
#pragma unroll
    for (size_t i = 0; i < ((HEAD_DIM + 31) >> 5); ++ i) {
        size_t offset = (i << 5) + lane_id;
        reg_q[i] = (offset < HEAD_DIM) ? llaisys::utils::nvidia::cast<float>(q + offset) : 0.0f;
        reg_acc[i] = 0.0f;
    }
    
    // pre_calc(to cp.async): kv cache block global pointer
    const T *k_ptrs[COALESCE];
    const T *v_ptrs[COALESCE];
    size_t block_token_offsets[COALESCE];
#pragma unroll
    for (size_t i = 0; i < COALESCE; ++ i) {
        if (i >= block_num) break;
        size_t table_id = begin_id + i;
        size_t block_id = _block_ids[table_id + 1];
        size_t cache_offset = (block_id * _token_num * _nkvh + nkvh) * HEAD_DIM;
        k_ptrs[i] = _k_cache + cache_offset;
        v_ptrs[i] = _v_cache + cache_offset;
        block_token_offsets[i] = table_id * _token_num;
    }

    // copy one tile
    const ELEMS_PER_BLOCK = 16 / sizeof(T);
    const BLOCKS_PER_ROW = HEAD_DIM / ELEMS_PER_BLOCK;
    const size_t tile_num = (_token_num + TILE_K - 1) / TILE_K;
    const size_t tot_tile_num = tot_block_num * tile_num;

    auto _cp_tile = [&](size_t stage, size_t tile_id) {
        size_t block_id = tile_id / tile_num;
        size_t inner_tile_id = tile_id % tile_num;
        const T *k_ptr = k_ptrs[i];
        const T *v_ptr = v_ptrs[i];
        size_t block_offset = block_token_offsets[block_id];
        for (size_t i = tid; i < BLOCKS_PER_ROW * TILE_K; i += blockDim.x) {
            size_t t = i / BLOCKS_PER_ROW;
            size_t 
        }
    };
    
    // prefetch kv cache
    _cp_tile(0, 0); __pipeline_commit();

    // compute and fetch loop
    for (size_t tile_id = 0, now = 0; tile_id < tot_tile_num; ++ tile_id, now ^= 1) {
        if (tile_id + 1 < tot_tile_num) {
            _cp_tile((tile_id + 1) & 1, tile_id + 1);
        }
        __pipeline_commit();
        __pipeline_wait_prior((tile_id + 1 < tot_tile_num) ? 1 : 0);
        __syncthreads();

        size_t block_id = tile_id / tile_num;
        size_t inner_tile_id = tile_id % tile_num;
        size_t block_token_offset = block_token_offsets[block_id];

#pragma unroll
        for (size_t tile_token_id = 0; tile_token_id < TILE_K; ++ tile_token_id) {
            size_t token_id = block_token_offset + inner_tile_id * TILE_K + tile_token_id;
            if (token_id <= limit && inner_tile_id * TILE_K + tile_token_id < _token_num) {
                const T *sk_cur = s_k + (now * TILE_K + tile_token_id) * HEAD_DIM;
                const T *sv_cur = s_v + (now * TILE_K + tile_token_id) * HEAD_DIM;
                
            }
        }



    }
    __syncthreads();

    // warp prefetch and compute loop
    const size_t token_num_padded = (_token_num + 3) / 4 * 4;
    for (size_t inner_token_id = warp_id & 3; inner_token_id < token_num_padded; inner_token_id += 4) {
        const size_t token_id = token_offset + inner_token_id;
        // online softmax
        if (warp_id < 4) {
            if (token_id <= limit && inner_token_id < _token_num) {
                float dot = 0.0f;
                const float *sk_cur = s_k[now];
#pragma unroll
                for (size_t i = lane_id; i < _d; i += 32) {
                    dot += s_q[i] * sk_cur[i];
                }
                dot = warp_reduce(dot);
                dot = dot * _scale;

                const float max_old = reg_max;
                const float max_new = fmaxf(max_old, dot);

                alpha = __expf(max_old - max_new);
                beta = __expf(dot - max_new);
                reg_max = max_new;
                reg_sum = alpha * reg_sum + beta;

                const float *sv_cur = s_v[now];
                for (size_t i = lane_id; i < _dv; i += 32) {
                    reg_acc[i >> 5] = alpha * reg_acc[i >> 5] + beta * sv_cur[i];
                }
            } else {
                alpha = 1.0f;
                beta = 0.0f;
            }
        }

        // load k / v
        if (warp_id >= 4 && token_id + 4 < totlen && inner_token_id + 4 < _token_num) {
            // global memory
            const T *k_cur = k_cache + (inner_token_id + 4) * _nkvh * _d;
            const T *v_cur = v_cache + (inner_token_id + 4) * _nkvh * _dv;
            float *sk_cur = s_k[now ^ 4];
            float *sv_cur = s_v[now ^ 4];

            // vec load: warp init
#pragma unroll
            for (size_t col = lane_id; col < _d; col += 32) {
                sk_cur[col] = llaisys::utils::nvidia::cast<float>(k_cur[col]);
            }
#pragma unroll
            for (size_t col = lane_id; col < _dv; col += 32) {
                 sv_cur[col] = llaisys::utils::nvidia::cast<float>(v_cur[col]);
            }
        }

        __syncthreads();
        now ^= 4;
    }

    // warp 0~3 inner reduce, 1~3 load sm, warp 0 compute then load back
    if (warp_id > 0 && warp_id < 4) {
        float *s_acc_cur = s_acc[warp_id];
#pragma unroll
        for (size_t i = lane_id; i < _dv; i += 32) {
            s_acc_cur[i] = reg_acc[i >> 5];
        }    
        s_sum[warp_id] = reg_sum;
        s_max[warp_id] = reg_max; 
    }

    __syncthreads();
    
    // warp 0 compute then write out
    if (0 == warp_id) {
#pragma unroll
        for (size_t i = 1; i <= 3; ++ i) {
            float sum_cur = s_sum[i];
            float max_cur = s_max[i];

            float max_new = fmaxf(reg_max, max_cur);

            alpha = __expf(reg_max - max_new);
            beta = __expf(max_cur - max_new);

            reg_sum = reg_sum * alpha + sum_cur * beta;
            reg_max = max_new;

            float *s_acc_cur = s_acc[i];
#pragma unroll
            for (size_t j = lane_id; j < _dv; j += 32) {
                reg_acc[j >> 5] = reg_acc[j >> 5] * alpha + s_acc_cur[j] * beta;
            }
        }

// DONE: host 端 _block_ids 第 0 列总块数计算前缀和
        int64_t offset = batch_id > 0 ? _block_ids[(batch_id - 1) * _max_block_num] : 0;
        offset += table_id;

        // thread load max / sum
        offset = (offset * _nh + nh) * 1;
        if (lane_id == 0) {
            _attn_max[offset] = reg_max;
            _attn_sum[offset] = reg_sum;
        }

        // naive: warp load back acc
        offset = offset * _dv;
#pragma unroll
        for (size_t i = lane_id; i < _dv; i += 32) {
            _attn_acc[offset + i] = reg_acc[i >> 5];
        }
    }
}

// 双缓冲 + block reduce
template<typename T, size_t _dv = 128>
__global__ void flash_decoding_reduce_kernel(T *__restrict__ _attn, 
                                             float *__restrict__ _attn_acc, 
                                             float *__restrict__ _attn_sum, 
                                             float *__restrict__ _attn_max, 
                                             const int64_t *_block_ids, 
                                             const size_t _max_block_num,
                                             const size_t _nh)
{
    __shared__ float s_acc[8 * _dv];    // (8 X dv) 缓冲区
    __shared__ float s_sum[4];          // (1 X 4)
    __shared__ float s_max[4];          // (1 X 4)

    float reg_acc[(_dv + 31) >> 5];
    float reg_sum = 0.0f;
    float reg_max = -INFINITY;
    float alpha = 1.0f;
    float beta = 0.0f;

    const size_t tid = threadIdx.x;
    const size_t nh = blockIdx.y;
    const size_t batch_id = blockIdx.z;
    
    const size_t warp_id = tid >> 5; // warp_id: 0~3 计算 4~7预取
    const size_t lane_id = tid & 31;

// DONE: host 端 _block_ids 第 0 列总块数计算前缀和
    int64_t offset = batch_id > 0 ? _block_ids[(batch_id - 1) * _max_block_num] : 0;
    offset = offset * _nh + nh; 

    float *attn_acc = _attn_acc + offset * _dv;
    float *attn_sum = _attn_sum + offset;
    float *attn_max = _attn_max + offset;

// DONE: host 端 _block_ids 第 0 列总块数计算前缀和
    const int64_t block_num = batch_id > 0 ? _block_ids[batch_id *  _max_block_num] - _block_ids[(batch_id - 1) * _max_block_num] : _block_ids[0]; // to: seq

    // warp 0~3 init reg_acc, warp 4~7 prefetch data
    size_t now = warp_id & 3; // 0, 1, 2, 3
    if (warp_id < 4) {
#pragma unroll
        for (size_t i = lane_id; i < _dv; i += 32) {
            reg_acc[i >> 5] = 0.0f;
        }
    } else if (now < block_num) {
        float *s_acc_cur = s_acc + now * _dv;
        float *attn_acc_cur = attn_acc + (warp_id & 3) * _nh * _dv;
#pragma unroll
        for (size_t i = lane_id; i < _dv; i += 32) {
            s_acc_cur[i] = attn_acc_cur[i];
        }
    }

    __syncthreads();
    
    // warp compute and prefetch loop
    const size_t block_num_padded = (block_num + 3) / 4 * 4;
    for (size_t block_id = (warp_id & 3); block_id < block_num_padded; block_id += 4) {
        // prefetch next block
        if (warp_id >= 4 && block_id + 4 < block_num) {
            float *s_acc_cur = s_acc + (now ^ 4) * _dv;
            float *attn_acc_cur = attn_acc + (block_id + 4) * _nh * _dv; 
#pragma unroll
            for (size_t i = lane_id; i < _dv; i += 32) {
                s_acc_cur[i] = attn_acc_cur[i];
            }
        }
        // compute current block
        if (warp_id < 4 && block_id < block_num) {
            float max_cur = attn_max[block_id * _nh];
            float sum_cur = attn_sum[block_id * _nh];

            float max_new = fmaxf(reg_max, max_cur);
            
            alpha = __expf(reg_max - max_new);
            beta = __expf(max_cur - max_new);

            reg_sum = reg_sum * alpha + sum_cur * beta;
            reg_max = max_new;

            float *s_acc_cur = s_acc + now * _dv;
#pragma unroll
            for (size_t i = lane_id; i < _dv; i += 32) {
                reg_acc[i >> 5] = reg_acc[i >> 5] * alpha + s_acc_cur[i] * beta;
            }
        }
        __syncthreads();
        now ^= 4;   // 0 <-> 4, 1 <-> 5, ..., 3 <-> 7
    }

    // warp 0~3 inner reduce, 1~3 load sm, warp 0 compute then load back
    if (warp_id > 0 && warp_id < 4) {
        float *s_acc_cur = s_acc + warp_id * _dv;
#pragma unroll
        for (size_t i = lane_id; i < _dv; i += 32) {
            s_acc_cur[i] = reg_acc[i >> 5];
        }
        s_sum[warp_id] = reg_sum;
        s_max[warp_id] = reg_max;
    }
    __syncthreads();
    if (warp_id == 0) {
#pragma unroll
        for (size_t i = 1; i <= 3; ++ i) {
            float max_cur = s_max[i];
            float sum_cur = s_sum[i];

            float max_new = fmaxf(reg_max, max_cur);

            alpha = __expf(reg_max - max_new);
            beta = __expf(max_cur - max_new);

            reg_sum = reg_sum * alpha + sum_cur * beta;
            reg_max = max_new;

            float *s_acc_cur = s_acc + i * _dv;
#pragma unroll
            for (size_t j = lane_id; j < _dv; j += 32) {
                reg_acc[j >> 5] = reg_acc[j >> 5] * alpha + s_acc_cur[j] * beta;
            }
        }
        float den = reg_sum > 0.0f ? 1.0f / reg_sum : 1.0;
        size_t attn_offset = (batch_id * _nh + nh) * _dv; 
        T *attn_cur = _attn + attn_offset;
#pragma unroll
        for (size_t i = lane_id; i < _dv; i += 32) {
            attn_cur[i] = llaisys::utils::nvidia::cast<T>(reg_acc[i >> 5] * den);
        }
    }
}
 
// decode stage 特殊的有: per_seq_len = 1, tot_seqlen = batch_size
template <typename T>
void flash_decoding_launch(std::byte *attn_val,        // 输出 (tot_seqlen, nh, dv) => (batch_size, nh, dv)
                            std::byte *attn_acc,       // 输出 (batch_block_num, nh, dv) 
                            std::byte *attn_sum,       // 输出 (batch_block_num, nh, 1)
                            std::byte *attn_max,       // 输出 (batch_block_num, nh, 1)
                            const std::byte *q,        // q (tot_seqlen, nh, d) => (batch_size, nh, d)
                            const std::byte *k_cache,  // k底层缓存 (block_num * token_num, nkvh, d)  
                            const std::byte *v_cache,  // q底层缓存 (block_num * token_num, nkvh, dv)
                            const std::byte *block_ids,// 块表 (batch, block_ids_size | block_ids)  kv cache布局: (block_num, token_num, nkvh, d | dv) 
                            const std::byte *cut_idx,  // 记录对应序列起始位置，也可计算 seqlen
                            const std::byte *tot_len,  // 已计算kv总长
                            const size_t token_num,    // 每个块的token数 (批次无关)
                            const size_t batch_size,   // 批次大小 (批次有关)
                            const size_t max_block_num,// 单个序列的块表最多块数 (批次无关)
                            const size_t tot_block_num,// 整个批次所有序列所用块数
                            const size_t max_seq_len,  // 最大序列长度 用于分块 (批次相关)
                            const float &scale,        // 缩放因子
                            const size_t &nh,          // 其他参数 ...
                            const size_t &dv, 
                            const size_t &d, 
                            const size_t &nkvh)
{
    auto *d_attn = reinterpret_cast<T *>(attn_val);
    // (max_batch_size * max_block_num = 2048, nh, dv) => (batch_max_block_num = 191, nh, dv) 不等式计算取 max
    auto *d_attn_acc = reinterpret_cast<float *>(attn_acc);
    auto *d_attn_sum = reinterpret_cast<float *>(attn_sum);
    auto *d_attn_max = reinterpret_cast<float *>(attn_max);
    const auto *d_q = reinterpret_cast<const T *>(q);
    const auto *d_k_cache = reinterpret_cast<const T *>(k_cache);
    const auto *d_v_cache = reinterpret_cast<const T *>(v_cache);
    const auto *d_block_ids = reinterpret_cast<const int64_t *>(block_ids);
    const auto *d_cut_idx = reinterpret_cast<const int64_t *>(cut_idx);
    const auto *d_tot_len = reinterpret_cast<const int64_t *>(tot_len);

    ASSERT(dv == 128 && d == 128, "variable illegal.");
    // 参数
    constexpr size_t HEAD_DIM = 128; // 维度
    constexpr size_t HEAD_NUM = 6;      // 每block负责q的行数 => HEAD_NUM | nh       
    constexpr size_t TILE_K = 4;     // 双缓冲每次copy到smem的kv行数
    constexpr size_t COALESCE = 1;   // 若干kv block组成一个task 一个task中最多包含 coalesce 个kv block

    // 根据 coalesce 非精确估算 task 数
    const size_t tot_task_num = tot_block_num / COALESCE + batch_size;
    const size_t warp_size = 32;

    dim3 blockDim_0(HEAD_NUM * warp_size);
    dim3 gridDim_0(tot_task_num, nh / HEAD_NUM);

    dim3 blockDim_1(BLOCK_DIM);
    dim3 gridDim_1(1, nh, batch_size);

    // 模板实例化
#define LLAISYS_DISPATCH(HEAD_DIM, HEADS, TILE_K, COALESCE)                                                         \
    flash_decoding_parallel_kernel<T, HEAD_DIM, HEADS, TILE_K, COALESCE><<<gridDim_0, blockDim_0>>>(               \
        d_attn_acc, d_attn_sum, d_attn_max, d_q, d_k_cache, d_v_cache, d_block_ids,     \
        d_cut_idx, d_tot_len, token_num, batch_size, max_block_num, scale, nh, nkvh);   \
    flash_decoding_reduce_kernel<T, HEAD_DIM, HEADS, TILE_K, COALESCE><<<gridDim_1, blockDim_1>>>(                     \
        d_attn, d_attn_acc, d_attn_sum, d_attn_max, d_block_ids, max_block_num, nh)

    if (d == 128 && dv == 128) {
        LLAISYS_DISPATCH_HD(128);
    } else if (d == 64 && dv == 64) {
        LLAISYS_DISPATCH_HD(64);
    } else if (d == 32 && dv == 32) {
        LLAISYS_DISPATCH_HD(32);
    } else if (d == 16 && dv == 16) {
        LLAISYS_DISPATCH_HD(16);
    } else if (d == 8 && dv == 8) {
        LLAISYS_DISPATCH_HD(8);
    } else {
        ASSERT(false, "flash_decoding: unsupported head_dim (only d==dv in {8,16,32,64,128}).");
    }
#undef LLAISYS_DISPATCH_HD

    CUDA_CHECK(cudaGetLastError());
}

namespace llaisys::ops::nvidia {
void flash_decoding(std::byte *attn_val,       // 输出 (tot_seqlen, nh, dv) => (batch_size, nh, dv)
                    std::byte *attn_acc,       // 输出 (batch_block_num: sum{[[block_num_0], [block_num_1], ...]}, nh, dv) 
                    std::byte *attn_sum,       // 输出 (batch_block_num, nh, 1)
                    std::byte *attn_max,       // 输出 (batch_block_num, nh, 1)
                    const std::byte *q,        // q (tot_seqlen, nh, d) => (batch_size, nh, d)
                    const std::byte *k_cache,  // k底层缓存 (block_num * token_num, nkvh, d)  
                    const std::byte *v_cache,  // q底层缓存 (block_num * token_num, nkvh, dv)
                    const std::byte *block_ids,// 块表 (batch, block_ids) 记录块号 (block_num, token_num, nkvh, d | dv) 
                    const std::byte *cut_idx,  // 记录对应序列起始位置，也可计算seqlen
                    const std::byte *tot_len,  // 已计算kv总长
                    const size_t token_num,    // 每个块的token数 (批次无关)
                    const size_t batch_size,   // 批次大小 (批次有关)
                    const size_t max_block_num,// 单个序列的块表最多块数 (批次无关)
                    const size_t tot_block_num,// 整个批次所有序列所用块数
                    const size_t max_seq_len,  // 最大序列长度 用于分块 (批次相关)
                    llaisysDataType_t dtype,   // 权重数据类型
                    const float &scale,        // 缩放因子
                    const size_t &nh,          // 其他参数 ...
                    const size_t &dv, 
                    const size_t &d, 
                    const size_t &nkvh)
{
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return flash_decoding_launch<float>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, block_ids, cut_idx, tot_len, token_num,
                                             batch_size, max_block_num, tot_block_num, max_seq_len, scale, nh, dv, d, nkvh);
    case LLAISYS_DTYPE_BF16:
        return flash_decoding_launch<llaisys::bf16_t>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, block_ids, cut_idx, tot_len, token_num,
                                             batch_size, max_block_num, tot_block_num, max_seq_len, scale, nh, dv, d, nkvh);
    case LLAISYS_DTYPE_F16:
        return flash_decoding_launch<llaisys::fp16_t>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, block_ids, cut_idx, tot_len, token_num,
                                             batch_size, max_block_num, tot_block_num, max_seq_len, scale, nh, dv, d, nkvh);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
