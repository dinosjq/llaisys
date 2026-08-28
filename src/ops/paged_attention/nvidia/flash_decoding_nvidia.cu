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
 *  8). kv smem reuse
 * 
 * physical cache overview
 *  1. [all blocks -> block -> token]
 *  2. [task / coalesce blocks -> block -> tile -> token -> chunk]
 * 
 * virtual table overview
 *  [table -> (tile) -> token]
 * 
 *  table_id(virtual) ={block_ids map}=> block_id(physical) 
 */

constexpr size_t WARP_SIZE = 32;        // 固定
constexpr size_t BLOCK_DIM_0 = 128;     // 固定
constexpr size_t BLOCK_DIM_1 = 256;     // 固定
constexpr size_t WARP_NUM = 4;          // 固定

static_assert(WARP_SIZE * WARP_NUM == BLOCK_DIM_0, "illegal");

template <typename T,       // value type
          size_t HEAD_DIM,  // attn head dim
          size_t HEAD_NUM,  // the count of q row  => (nk / nkvh)
          size_t TILE_K,    // split physical block into tile
          size_t COALESCE   // merge blocks count
          >
__global__ void flash_decoding_parallel_kernel(float *__restrict__ _attn_acc, 
                                               float *__restrict__ _attn_sum,
                                               float *__restrict__ _attn_max,
                                               const T *__restrict__ _q, 
                                               const T *__restrict__ _k_cache, 
                                               const T *__restrict__ _v_cache, 
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
    // shared memory __align__(16) cp.async 16B 拷贝要求 src/dst 16B 对齐
    __shared__ __align__(16) T s_k[2][TILE_K][HEAD_DIM];
    __shared__ __align__(16) T s_v[2][TILE_K][HEAD_DIM];
    __shared__ __align__(16) T s_q[HEAD_NUM][HEAD_DIM];
    __shared__ __align__(16) float s_acc[HEAD_NUM][WARP_NUM][HEAD_DIM];
    __shared__ float s_sum[HEAD_NUM][WARP_NUM];
    __shared__ float s_max[HEAD_NUM][WARP_NUM];

    // register
    float4 reg_acc[HEAD_NUM];
    float reg_sum[HEAD_NUM];
    float reg_max[HEAD_NUM];

    const size_t tid = threadIdx.x;
    const size_t warp_id = tid >> 5;
    const size_t lane_id = tid & 31;

    // init acc sum max
#pragma unroll
    for (size_t h = 0; h < HEAD_NUM; ++ h) {
        reg_acc[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        reg_sum[h] = 0.0f;
        reg_max[h] = -INFINITY;
    }

    // info
    const size_t task_id = blockIdx.x;
    const size_t rep = _nh / _nkvh;                     
    const size_t q_groups = (rep + HEAD_NUM - 1) / HEAD_NUM;
    const size_t nkvh = blockIdx.y / q_groups;         
    const size_t q_group = blockIdx.y % q_groups;      
    size_t batch_id = -1;
    size_t pre_task_num = 0;
    size_t last = 0;
    for (size_t i = 0; i < _batch_size; ++ i) {
        size_t now = static_cast<size_t>(_block_ids[i * _table_width]);
        size_t task_num = (now - last + COALESCE - 1) / COALESCE;
        if (pre_task_num + task_num > task_id) {
            batch_id = i;
            break;
        }
        pre_task_num += task_num;
        last = now;
    }
    if (batch_id == -1) {
        return;
    }
    const size_t totlen = _tot_len[batch_id];
    const size_t limit = totlen - 1;
    const size_t begin_id = (task_id - pre_task_num) * COALESCE;
    size_t task_size = _block_ids[batch_id * _table_width] - ((batch_id > 0) ? _block_ids[(batch_id - 1) * _table_width] : 0);
    task_size = min(task_size - begin_id, COALESCE);

    // load q (global -> smem)
#pragma unroll
    for (size_t h = warp_id; h < HEAD_NUM; h += WARP_NUM) {
        if (q_group * HEAD_NUM + h >= rep) continue;
        size_t nh = nkvh * rep + q_group * HEAD_NUM + h;
        const T *q = _q + (batch_id * _nh + nh) * HEAD_DIM;
        llaisys::utils::nvidia::copy_4d(q + (lane_id << 2), s_q[h] + (lane_id << 2));
    }
    
    // pre_calc(to cp.async): kv cache block global pointer
    const T *k_ptrs[COALESCE];
    const T *v_ptrs[COALESCE];
    size_t table_offsets[COALESCE];
#pragma unroll
    for (size_t i = 0; i < COALESCE; ++ i) {
        if (i >= task_size) break;
        size_t table_id = begin_id + i;
        size_t block_id = _block_ids[batch_id * _table_width + table_id + 1];
        size_t cache_offset = (block_id * _block_size * _nkvh + nkvh) * HEAD_DIM;
        k_ptrs[i] = _k_cache + cache_offset;
        v_ptrs[i] = _v_cache + cache_offset;
        table_offsets[i] = table_id * _block_size;
    }

    // copy one chunk
    constexpr size_t ELES_PER_CHUNK = 16 / sizeof(T);               // one thread copy 16 bytes
    constexpr size_t CHUNKS_PER_ROW = HEAD_DIM / ELES_PER_CHUNK;    // chunk num per head dim
    const size_t tile_num = (_block_size + TILE_K - 1) / TILE_K;    // tile num per block
    const size_t tot_tile_num = task_size * tile_num;               // total tile num for all blocks in this task  

    // cp async method (global -> smem)
    auto _cp_tile = [&](size_t stage, size_t tile_id) {
        size_t table_id = tile_id / tile_num;
        size_t inner_id = tile_id % tile_num;
        const T *k_ptr = k_ptrs[table_id];
        const T *v_ptr = v_ptrs[table_id];
        size_t table_offset = table_offsets[table_id];
        size_t tile_offset = inner_id * TILE_K;
        for (size_t i = tid; i < TILE_K * CHUNKS_PER_ROW; i += blockDim.x) {
            size_t row = i / CHUNKS_PER_ROW;
            size_t col = i % CHUNKS_PER_ROW;
            size_t inner_token_id = tile_offset + row; 
            if (inner_token_id < _block_size && inner_token_id + table_offset < totlen) {
                __pipeline_memcpy_async(&s_k[stage][row][col * ELES_PER_CHUNK],
                                        k_ptr + inner_token_id * _nkvh * HEAD_DIM + col * ELES_PER_CHUNK, 16);
                __pipeline_memcpy_async(&s_v[stage][row][col * ELES_PER_CHUNK],
                                        v_ptr + inner_token_id * _nkvh * HEAD_DIM + col * ELES_PER_CHUNK, 16);
            }
        }
    };
    
    // prefetch kv cache
    _cp_tile(0, 0); __pipeline_commit();

    // compute and fetch loop
    for (size_t tile_id = 0; tile_id < tot_tile_num; ++ tile_id) {
        // current stage
        const size_t now = tile_id & 1;

        // fetch
        if (tile_id + 1 < tot_tile_num) {
            _cp_tile(now ^ 1, tile_id + 1);
        }
        __pipeline_commit();
        __pipeline_wait_prior((tile_id + 1 < tot_tile_num) ? 1 : 0);
        __syncthreads();

        size_t table_id = tile_id / tile_num;
        size_t inner_id = tile_id % tile_num;
        size_t table_offset = table_offsets[table_id];
        
        // compute
#pragma unroll
        for (size_t row = warp_id; row < TILE_K; row += WARP_NUM) {
            size_t inner_token_id = inner_id * TILE_K + row;
            if (inner_token_id < _block_size && inner_token_id + table_offset <= limit) {
                // load kv (smem -> register)
                const T *sp_k = s_k[now][row];
                const T *sp_v = s_v[now][row];
                float4 k_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 v_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                llaisys::utils::nvidia::copy_4d(sp_k + (lane_id << 2), &k_vec.x);
                llaisys::utils::nvidia::copy_4d(sp_v + (lane_id << 2), &v_vec.x);
#pragma unroll
                for (size_t h = 0; h < HEAD_NUM; ++ h) {
                    float4 q_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    llaisys::utils::nvidia::copy_4d(s_q[h] + (lane_id << 2), &q_vec.x);

                    float dot = q_vec.x * k_vec.x + q_vec.y * k_vec.y + q_vec.z * k_vec.z + q_vec.w * k_vec.w;
                    dot = warp_reduce(dot) * _scale;

                    float max_new = fmaxf(reg_max[h], dot);
                    float alpha = __expf(reg_max[h] - max_new);
                    float beta = __expf(dot - max_new);
                    reg_max[h] = max_new;
                    reg_sum[h] = alpha * reg_sum[h] + beta;

                    reg_acc[h].x = alpha * reg_acc[h].x + beta * v_vec.x;
                    reg_acc[h].y = alpha * reg_acc[h].y + beta * v_vec.y;
                    reg_acc[h].z = alpha * reg_acc[h].z + beta * v_vec.z;
                    reg_acc[h].w = alpha * reg_acc[h].w + beta * v_vec.w;
                }
            }
        }
        __syncthreads();
    }

    // store acc (register -> smem)
#pragma unroll
    for (size_t h = 0; h < HEAD_NUM; ++ h) {
        if (h % WARP_NUM == warp_id) continue;
        *reinterpret_cast<float4 *>(&s_acc[h][warp_id][lane_id << 2]) = reg_acc[h];
        if (lane_id == 0) {
            s_sum[h][warp_id] = reg_sum[h];
            s_max[h][warp_id] = reg_max[h];
        }
    }
    __syncthreads();

    // compute then write back
#pragma unroll
    for (size_t h = warp_id; h < HEAD_NUM; h += WARP_NUM) {
        if (q_group * HEAD_NUM + h >= rep) continue;
        float sum_old = reg_sum[h];
        float max_old = reg_max[h];
#pragma unroll
        for (size_t i = 0; i < WARP_NUM; ++ i) {
            if (i == warp_id) continue;
            float sum_cur = s_sum[h][i];
            float max_cur = s_max[h][i];

            float max_new = fmaxf(max_old, max_cur);
            float alpha = __expf(max_old - max_new);
            float beta = __expf(max_cur - max_new);

            max_old = max_new;
            sum_old = alpha * sum_old + beta * sum_cur;
            float4 b = *reinterpret_cast<float4 *>(&s_acc[h][i][lane_id << 2]);
            reg_acc[h].x = alpha * reg_acc[h].x + beta * b.x;
            reg_acc[h].y = alpha * reg_acc[h].y + beta * b.y;
            reg_acc[h].z = alpha * reg_acc[h].z + beta * b.z;
            reg_acc[h].w = alpha * reg_acc[h].w + beta * b.w;
        }

        size_t nh = nkvh * rep + q_group * HEAD_NUM + h;
        size_t offset = task_id * _nh + nh;
        if (lane_id == 0) {
            _attn_max[offset] = max_old;
            _attn_sum[offset] = sum_old;
        }

        offset *= HEAD_DIM;
        reinterpret_cast<float4 *>(_attn_acc + offset)[lane_id] = reg_acc[h];
    }
}

// 双缓冲 + block reduce
template<typename T, 
         size_t HEAD_DIM,
         size_t COALESCE>
__global__ void flash_decoding_reduce_kernel(T *__restrict__ _attn, 
                                             float *__restrict__ _attn_acc, 
                                             float *__restrict__ _attn_sum, 
                                             float *__restrict__ _attn_max, 
                                             const int64_t *_block_ids, 
                                             const size_t _table_width,
                                             const size_t _nh)
{
    __shared__ float s_acc[8 * HEAD_DIM];   // (8 X dv) 缓冲区
    __shared__ float s_sum[4];              // (1 X 4)
    __shared__ float s_max[4];              // (1 X 4)

    float reg_acc[(HEAD_DIM + 31) >> 5];
    float reg_sum = 0.0f;
    float reg_max = -INFINITY;
    float alpha = 1.0f;
    float beta = 0.0f;

    const size_t tid = threadIdx.x;
    const size_t nh = blockIdx.x;
    const size_t batch_id = blockIdx.y;
    
    const size_t warp_id = tid >> 5; // warp_id: 0~3 计算 4~7预取
    const size_t lane_id = tid & 31;

    const size_t offset = (batch_id > 0) ? (static_cast<size_t>(_block_ids[(batch_id - 1) * _table_width]) + COALESCE - 1) / COALESCE : 0;

    float *attn_acc = _attn_acc + offset * _nh * HEAD_DIM;
    float *attn_sum = _attn_sum + offset * _nh;
    float *attn_max = _attn_max + offset * _nh;

    size_t task_num = batch_id > 0 ? _block_ids[batch_id * _table_width] - _block_ids[(batch_id - 1) * _table_width] : _block_ids[0]; // to: seq
    task_num = (task_num + COALESCE - 1) / COALESCE;

    // warp 0~3 init reg_acc, warp 4~7 prefetch data
    size_t now = warp_id & 3; // 0, 1, 2, 3
    if (warp_id < 4) {
#pragma unroll
        for (size_t i = lane_id; i < HEAD_DIM; i += 32) {
            reg_acc[i >> 5] = 0.0f;
        }
    } else if (now < task_num) {
        float *s_acc_cur = s_acc + now * HEAD_DIM;
        float *attn_acc_cur = attn_acc + (warp_id & 3) * _nh * HEAD_DIM + nh * HEAD_DIM;
#pragma unroll
        for (size_t i = lane_id; i < HEAD_DIM; i += 32) {
            s_acc_cur[i] = attn_acc_cur[i];
        }
    }

    __syncthreads();
    
    // warp compute and prefetch loop
    const size_t task_num_padded = (task_num + 3) / 4 * 4;
    for (size_t task_id = (warp_id & 3); task_id < task_num_padded; task_id += 4) {
        // prefetch next block
        if (warp_id >= 4 && task_id + 4 < task_num) {
            float *s_acc_cur = s_acc + (now ^ 4) * HEAD_DIM;
            float *attn_acc_cur = attn_acc + (task_id + 4) * _nh * HEAD_DIM + nh * HEAD_DIM;
#pragma unroll
            for (size_t i = lane_id; i < HEAD_DIM; i += 32) {
                s_acc_cur[i] = attn_acc_cur[i];
            }
        }
        // compute current block
        if (warp_id < 4 && task_id < task_num) {
            float max_cur = attn_max[task_id * _nh + nh];
            float sum_cur = attn_sum[task_id * _nh + nh];

            float max_new = fmaxf(reg_max, max_cur);
            
            alpha = __expf(reg_max - max_new);
            beta = __expf(max_cur - max_new);

            reg_sum = reg_sum * alpha + sum_cur * beta;
            reg_max = max_new;

            float *s_acc_cur = s_acc + now * HEAD_DIM;
#pragma unroll
            for (size_t i = lane_id; i < HEAD_DIM; i += 32) {
                reg_acc[i >> 5] = reg_acc[i >> 5] * alpha + s_acc_cur[i] * beta;
            }
        }
        __syncthreads();
        now ^= 4;   // 0 <-> 4, 1 <-> 5, ..., 3 <-> 7
    }

    // warp 0~3 inner reduce, 1~3 load sm, warp 0 compute then load back
    if (warp_id > 0 && warp_id < 4) {
        float *s_acc_cur = s_acc + warp_id * HEAD_DIM;
#pragma unroll
        for (size_t i = lane_id; i < HEAD_DIM; i += 32) {
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

            float *s_acc_cur = s_acc + i * HEAD_DIM;
#pragma unroll
            for (size_t j = lane_id; j < HEAD_DIM; j += 32) {
                reg_acc[j >> 5] = reg_acc[j >> 5] * alpha + s_acc_cur[j] * beta;
            }
        }
        float den = reg_sum > 0.0f ? 1.0f / reg_sum : 1.0;
        size_t attn_offset = (batch_id * _nh + nh) * HEAD_DIM; 
        T *attn_cur = _attn + attn_offset;
#pragma unroll
        for (size_t i = lane_id; i < HEAD_DIM; i += 32) {
            attn_cur[i] = llaisys::utils::nvidia::cast<T>(reg_acc[i >> 5] * den);
        }
    }
}

// decode stage 特殊的有: per_seq_len = 1, tot_seqlen = batch_size
template <typename T, size_t HD, size_t HEADS, size_t TILE_K, size_t COALESCE>
void flash_decoding_launch(std::byte *attn_val,        // 输出 (batch_size, nh, dv)
                            std::byte *attn_acc,       // 输出 (batch_task_num, nh, dv) 
                            std::byte *attn_sum,       // 输出 (batch_task_num, nh, 1)
                            std::byte *attn_max,       // 输出 (batch_task_num, nh, 1)
                            const std::byte *q,        // q (batch_size, nh, d)
                            const std::byte *k_cache,  // k底层缓存 (block_num * block_size, nkvh, d)  
                            const std::byte *v_cache,  // q底层缓存 (block_num * block_size, nkvh, dv)
                            const std::byte *block_ids,// 块表 (batch_size, block_ids_size | block_ids)  kv cache 布局: (block_num, block_size, nkvh, d | dv) 
                            const std::byte *cut_idx,  // 序列起始位置 (batch_size + 1)
                            const std::byte *tot_len,  // 已计算 kv 总长 (batch_size)
                            const size_t block_size,   // 一个 block 的 token 数 (批次无关)
                            const size_t batch_size,   // 批次大小 (批次有关)
                            const size_t table_width,  // 块表宽度 (批次无关)
                            const size_t tot_task_num, // task 数
                            const float &scale,        // 缩放因子
                            const size_t &nh,          // 其他参数 ...
                            const size_t &nkvh)
{
    auto *d_attn = reinterpret_cast<T *>(attn_val);
    auto *d_attn_acc = reinterpret_cast<float *>(attn_acc);
    auto *d_attn_sum = reinterpret_cast<float *>(attn_sum);
    auto *d_attn_max = reinterpret_cast<float *>(attn_max);
    const auto *d_q = reinterpret_cast<const T *>(q);
    const auto *d_k_cache = reinterpret_cast<const T *>(k_cache);
    const auto *d_v_cache = reinterpret_cast<const T *>(v_cache);
    const auto *d_block_ids = reinterpret_cast<const int64_t *>(block_ids);
    const auto *d_cut_idx = reinterpret_cast<const int64_t *>(cut_idx);
    const auto *d_tot_len = reinterpret_cast<const int64_t *>(tot_len);

    const size_t q_groups = ((nh / nkvh) + HEADS - 1) / HEADS;  // 上取整，最后一块可不满

    dim3 blockDim_0(BLOCK_DIM_0);
    dim3 gridDim_0(tot_task_num, nkvh * q_groups);

    dim3 blockDim_1(BLOCK_DIM_1);
    dim3 gridDim_1(nh, batch_size);

    flash_decoding_parallel_kernel<T, HD, HEADS, TILE_K, COALESCE><<<gridDim_0, blockDim_0>>>(
        d_attn_acc, d_attn_sum, d_attn_max, d_q, d_k_cache, d_v_cache, d_block_ids,
        d_cut_idx, d_tot_len, block_size, batch_size, table_width, scale, nh, nkvh);
    flash_decoding_reduce_kernel<T, HD, COALESCE><<<gridDim_1, blockDim_1>>>(
        d_attn, d_attn_acc, d_attn_sum, d_attn_max, d_block_ids, table_width, nh);

    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void flash_decoding_dispatch(std::byte *attn_val,      // 输出 (batch_size, nh, dv)
                            std::byte *attn_acc,       // 输出 (batch_task_num, nh, dv) 
                            std::byte *attn_sum,       // 输出 (batch_task_num, nh, 1)
                            std::byte *attn_max,       // 输出 (batch_task_num, nh, 1)
                            const std::byte *q,        // q (batch_size, nh, d)
                            const std::byte *k_cache,  // k底层缓存 (block_num * block_size, nkvh, d)  
                            const std::byte *v_cache,  // q底层缓存 (block_num * block_size, nkvh, dv)
                            const std::byte *block_ids,// 块表 (batch_size, block_ids_size | block_ids)  kv cache 布局: (block_num, block_size, nkvh, d | dv) 
                            const std::byte *cut_idx,  // 序列起始位置 (batch_size + 1)
                            const std::byte *tot_len,  // 已计算 kv 总长 (batch_size)
                            const size_t block_size,   // 一个 block 的 token 数 (批次无关)
                            const size_t batch_size,   // 批次大小 (批次有关)
                            const size_t table_width,  // 块表宽度 (批次无关)
                            const size_t tot_task_num, // task 数
                            const float &scale,        // 缩放因子
                            const size_t &nh,          // 其他参数 ...
                            const size_t &dv, 
                            const size_t &d,
                            const size_t &nkvh)
{
    ASSERT(d == dv, "flash_decoding: d must equal dv.");
    ASSERT(d == 128, "flash_decoding: only hd==128 supported (Qwen2-1.5B / Llama-3.2-3B).");
    const int heads = static_cast<int>(nh / nkvh);   // HEAD_NUM = rep（整组最优）

#define FD_LAUNCH(H)                                                                   \
    flash_decoding_launch<T, 128, H, 16, 1>(                                           \
        attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, block_ids,        \
        cut_idx, tot_len, block_size, batch_size, table_width, tot_task_num,           \
        scale, nh, nkvh)

    if (heads == 6)      FD_LAUNCH(6);   // Qwen2-1.5B
    else if (heads == 3) FD_LAUNCH(3);   // Llama-3.2-3B
    else ASSERT(false, "flash_decoding: unsupported GQA ratio (nh/nkvh must be 6 or 3).");
#undef FD_LAUNCH
}

namespace llaisys::ops::nvidia {
void flash_decoding(std::byte *attn_val,       // 输出 (tot_seqlen, nh, dv) => (batch_size, nh, dv)
                    std::byte *attn_acc,       // 输出 (batch_block_num: sum{[[block_num_0], [block_num_1], ...]}, nh, dv) 
                    std::byte *attn_sum,       // 输出 (batch_block_num, nh, 1)
                    std::byte *attn_max,       // 输出 (batch_block_num, nh, 1)
                    const std::byte *q,        // q (tot_seqlen, nh, d) => (batch_size, nh, d)
                    const std::byte *k_cache,  // k底层缓存 (block_num * block_size, nkvh, d)  
                    const std::byte *v_cache,  // q底层缓存 (block_num * block_size, nkvh, dv)
                    const std::byte *block_ids,// 块表 (batch, block_ids) 记录块号 (block_num, block_size, nkvh, d | dv) 
                    const std::byte *cut_idx,  // 记录对应序列起始位置，也可计算seqlen
                    const std::byte *tot_len,  // 已计算kv总长
                    const size_t block_size,   // 每个块的token数 (批次无关)
                    const size_t batch_size,   // 批次大小 (批次有关)
                    const size_t table_width,  // 块表宽度 (批次无关)
                    const size_t tot_task_num, // 精确 task 数
                    llaisysDataType_t dtype,   // 权重数据类型
                    const float &scale,        // 缩放因子
                    const size_t &nh,          // 其他参数 ...
                    const size_t &dv, 
                    const size_t &d, 
                    const size_t &nkvh)
{
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return flash_decoding_dispatch<float>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, block_ids, cut_idx, tot_len, block_size,
                                             batch_size, table_width, tot_task_num, scale, nh, dv, d, nkvh);
    case LLAISYS_DTYPE_BF16:
        return flash_decoding_dispatch<llaisys::bf16_t>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, block_ids, cut_idx, tot_len, block_size,
                                             batch_size, table_width, tot_task_num, scale, nh, dv, d, nkvh);
    case LLAISYS_DTYPE_F16:
        return flash_decoding_dispatch<llaisys::fp16_t>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, block_ids, cut_idx, tot_len, block_size,
                                             batch_size, table_width, tot_task_num, scale, nh, dv, d, nkvh);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
