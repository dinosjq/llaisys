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
 *  3). consumer + producer;
 *  4). less register/smem => warp contain data
 *  5). parallel + reduce;
 *  6). cp.async, assembly line;
 *  7). coalesce kv block
 *  8). kv smem reuse
 *  9). less register/smem => warp group
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

// 对 block 内的 warp 进行分组
constexpr size_t WARP_SIZE = 32;    // 固定: 单个 warp 的 thread 个数

template <typename T,       // value type (attn 输出 / q)
          size_t HEAD_DIM,  // attn head dim
          size_t HEAD_NUM,  // _nk / _nkvh
          size_t WARP_NUM,  // 可变: 一个 group 的 warp 数目
          size_t TILE_M,    // 可变: 一个 group 最多处理几行 q
          size_t TILE_K,    // split physical block into tile
          size_t COALESCE,  // merge blocks count
          bool QUANT        // 量化 KV cache (i8 + per-token scale)
          >
__global__ void flash_decoding_parallel_kernel(float *__restrict__ _attn_acc, 
                                               float *__restrict__ _attn_sum,
                                               float *__restrict__ _attn_max,
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
    // cp.async 16B 拷贝要求 src/dst 16B 对齐
    __shared__ __align__(16) char s_k[2][TILE_K][HEAD_DIM * (QUANT ? 1 : sizeof(T))];
    __shared__ __align__(16) char s_v[2][TILE_K][HEAD_DIM * (QUANT ? 1 : sizeof(T))];
    __shared__ __align__(16) float s_scale[2][TILE_K][2];   // 量化时每 tile 每 token 的 {k_scale, v_scale}
    __shared__ __align__(16) T s_q[HEAD_NUM][HEAD_DIM];
    __shared__ __align__(16) float s_acc[HEAD_NUM][WARP_NUM][HEAD_DIM];
    __shared__ float s_sum[HEAD_NUM][WARP_NUM];
    __shared__ float s_max[HEAD_NUM][WARP_NUM];

    // register
    float4 reg_acc[TILE_M];
    float reg_sum[TILE_M];
    float reg_max[TILE_M];
#pragma unroll
    for (size_t h = 0; h < TILE_M; ++ h) {
        reg_acc[h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        reg_sum[h] = 0.0f;
        reg_max[h] = -INFINITY;
    }

    const size_t tid = threadIdx.x;
    const size_t task_id = blockIdx.x;
    const size_t nkvh = blockIdx.y; 

    const size_t warp_id = tid >> 5;
    const size_t lane_id = tid & 31;
    const size_t group_id = warp_id / WARP_NUM;

    size_t batch_id = -1, pre_task_num = 0, last = 0;
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
    if (batch_id == -1) return;
    const size_t begin_id = (task_id - pre_task_num) * COALESCE;

    const size_t totlen = _tot_len[batch_id];
    const size_t limit = totlen - 1;
    const int64_t *block_ids = _block_ids + batch_id * _table_width;
    size_t task_size = _block_ids[batch_id * _table_width] - ((batch_id > 0) ? _block_ids[(batch_id - 1) * _table_width] : 0);
    task_size = min(task_size - begin_id, COALESCE);

    // load q (global -> smem)
#pragma unroll
    for (size_t i = warp_id % WARP_NUM; i < TILE_M; i += WARP_NUM) {
        if (group_id * TILE_M + i >= HEAD_NUM) break;
        const size_t h = group_id * TILE_M + i;
        const size_t nh = nkvh * HEAD_NUM + h;
        const T *q = _q + (batch_id * _nh + nh) * HEAD_DIM;
        llaisys::utils::nvidia::copy_4d(q + (lane_id << 2), s_q[h] + (lane_id << 2));
    }
    
    // kv cache physical block pointer (字节地址)
    constexpr size_t ELEM_BYTES = QUANT ? 1 : sizeof(T);
    const void *k_ptrs[COALESCE];
    const void *v_ptrs[COALESCE];
    size_t table_offsets[COALESCE];
#pragma unroll
    for (size_t i = 0; i < COALESCE; ++ i) {
        if (i >= task_size) break;
        size_t table_id = begin_id + i;
        size_t block_id = block_ids[table_id + 1];
        size_t cache_offset = (block_id * _block_size * _nkvh + nkvh) * HEAD_DIM * ELEM_BYTES;
        k_ptrs[i] = reinterpret_cast<const void *>(reinterpret_cast<const char *>(_k_cache) + cache_offset);
        v_ptrs[i] = reinterpret_cast<const void *>(reinterpret_cast<const char *>(_v_cache) + cache_offset);
        table_offsets[i] = table_id * _block_size;
    }

    // copy one chunk
    constexpr size_t ELES_PER_CHUNK = 16 / sizeof(T);               // one thread copy 16 bytes (非量化元素数)
    constexpr size_t CHUNKS_PER_ROW = HEAD_DIM / ELES_PER_CHUNK;    // chunk num per head dim
    constexpr size_t I8_CHUNKS_PER_ROW = HEAD_DIM / 16;             // 量化：每 16B 拷 16 个 i8
    const size_t tile_num = (_block_size + TILE_K - 1) / TILE_K;    // tile num per block
    const size_t tot_tile_num = task_size * tile_num;               // total tile num for all blocks in this task  

    // cp async method (global -> smem)
    auto _cp_tile = [&](size_t stage, size_t tile_id) {
        size_t table_id = tile_id / tile_num;
        size_t inner_id = tile_id % tile_num;
        size_t table_offset = table_offsets[table_id];
        size_t tile_offset = inner_id * TILE_K;
        if constexpr (QUANT) {
            const int8_t *k_ptr = reinterpret_cast<const int8_t *>(k_ptrs[table_id]);
            const int8_t *v_ptr = reinterpret_cast<const int8_t *>(v_ptrs[table_id]);
            size_t block_id = block_ids[table_id + 1];
            const float *k_scale = _k_scale + block_id * _block_size * _nkvh;
            const float *v_scale = _v_scale + block_id * _block_size * _nkvh;
            // kv 数据：16B = 16 个 i8
            for (size_t i = tid; i < TILE_K * I8_CHUNKS_PER_ROW; i += blockDim.x) {
                size_t row = i / I8_CHUNKS_PER_ROW;
                size_t col = i % I8_CHUNKS_PER_ROW;
                size_t inner_token_id = tile_offset + row;
                if (inner_token_id < _block_size && inner_token_id + table_offset < totlen) {
                    __pipeline_memcpy_async(&s_k[stage][row][col * 16],
                                            k_ptr + inner_token_id * _nkvh * HEAD_DIM + col * 16, 16);
                    __pipeline_memcpy_async(&s_v[stage][row][col * 16],
                                            v_ptr + inner_token_id * _nkvh * HEAD_DIM + col * 16, 16);
                }
            }
            // scale：每 token 当前 nkvh 的 {k,v} scale
            for (size_t i = tid; i < TILE_K; i += blockDim.x) {
                size_t inner_token_id = tile_offset + i;
                if (inner_token_id < _block_size && inner_token_id + table_offset < totlen) {
                    s_scale[stage][i][0] = k_scale[inner_token_id * _nkvh + nkvh];
                    s_scale[stage][i][1] = v_scale[inner_token_id * _nkvh + nkvh];
                }
            }
        } else {
            const T *k_ptr = reinterpret_cast<const T *>(k_ptrs[table_id]);
            const T *v_ptr = reinterpret_cast<const T *>(v_ptrs[table_id]);
            for (size_t i = tid; i < TILE_K * CHUNKS_PER_ROW; i += blockDim.x) {
                size_t row = i / CHUNKS_PER_ROW;
                size_t col = i % CHUNKS_PER_ROW;
                size_t inner_token_id = tile_offset + row; 
                if (inner_token_id < _block_size && inner_token_id + table_offset < totlen) {
                    __pipeline_memcpy_async(&s_k[stage][row][col * ELES_PER_CHUNK * sizeof(T)],
                                            k_ptr + inner_token_id * _nkvh * HEAD_DIM + col * ELES_PER_CHUNK, 16);
                    __pipeline_memcpy_async(&s_v[stage][row][col * ELES_PER_CHUNK * sizeof(T)],
                                            v_ptr + inner_token_id * _nkvh * HEAD_DIM + col * ELES_PER_CHUNK, 16);
                }
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
        for (size_t row = warp_id % WARP_NUM; row < TILE_K; row += WARP_NUM) {
            size_t inner_token_id = inner_id * TILE_K + row;
            if (inner_token_id < _block_size && inner_token_id + table_offset <= limit) {
                // load kv (smem -> register) 反量化
                float4 k_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 v_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if constexpr (QUANT) {
                    const int8_t *sp_k = reinterpret_cast<const int8_t *>(s_k[now][row]);
                    const int8_t *sp_v = reinterpret_cast<const int8_t *>(s_v[now][row]);
                    const float k_sc = s_scale[now][row][0];
                    const float v_sc = s_scale[now][row][1];
                    const size_t q4 = lane_id << 2;
                    k_vec.x = static_cast<float>(sp_k[q4 + 0]) * k_sc;
                    k_vec.y = static_cast<float>(sp_k[q4 + 1]) * k_sc;
                    k_vec.z = static_cast<float>(sp_k[q4 + 2]) * k_sc;
                    k_vec.w = static_cast<float>(sp_k[q4 + 3]) * k_sc;
                    v_vec.x = static_cast<float>(sp_v[q4 + 0]) * v_sc;
                    v_vec.y = static_cast<float>(sp_v[q4 + 1]) * v_sc;
                    v_vec.z = static_cast<float>(sp_v[q4 + 2]) * v_sc;
                    v_vec.w = static_cast<float>(sp_v[q4 + 3]) * v_sc;
                } else {
                    const T *sp_k = reinterpret_cast<const T *>(s_k[now][row]);
                    const T *sp_v = reinterpret_cast<const T *>(s_v[now][row]);
                    llaisys::utils::nvidia::copy_4d(sp_k + (lane_id << 2), &k_vec.x);
                    llaisys::utils::nvidia::copy_4d(sp_v + (lane_id << 2), &v_vec.x);
                }

#pragma unroll
                for (size_t i = 0; i < TILE_M; ++ i) {
                    float4 q_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    const size_t h = group_id * TILE_M + i;
                    if (h >= HEAD_NUM) break;
                    llaisys::utils::nvidia::copy_4d(s_q[h] + (lane_id << 2), &q_vec.x);

                    float dot = q_vec.x * k_vec.x + q_vec.y * k_vec.y + q_vec.z * k_vec.z + q_vec.w * k_vec.w;
                    dot = warp_reduce(dot) * _scale;

                    float max_new = fmaxf(reg_max[i], dot);
                    float alpha = __expf(reg_max[i] - max_new);
                    float beta = __expf(dot - max_new);
                    reg_max[i] = max_new;
                    reg_sum[i] = alpha * reg_sum[i] + beta;

                    reg_acc[i].x = alpha * reg_acc[i].x + beta * v_vec.x;
                    reg_acc[i].y = alpha * reg_acc[i].y + beta * v_vec.y;
                    reg_acc[i].z = alpha * reg_acc[i].z + beta * v_vec.z;
                    reg_acc[i].w = alpha * reg_acc[i].w + beta * v_vec.w;
                }
            }
        }
        __syncthreads();
    }

    // store acc (register -> smem)
#pragma unroll
    for (size_t i = 0; i < TILE_M; ++ i) {
        const size_t h = group_id * TILE_M + i;
        if (h >= HEAD_NUM) break;
        *reinterpret_cast<float4 *>(&s_acc[h][warp_id % WARP_NUM][lane_id << 2]) = reg_acc[i];
        if (lane_id == 0) {
            s_sum[h][warp_id % WARP_NUM] = reg_sum[i];
            s_max[h][warp_id % WARP_NUM] = reg_max[i];
        }
    }
    __syncthreads();

    // compute then write back
    if (warp_id % WARP_NUM == 0) {
#pragma unroll
        for (size_t hh = 0; hh < TILE_M; ++ hh) {
            const size_t h = group_id * TILE_M + hh;
            if (h >= HEAD_NUM) break;
            float sum_old = reg_sum[hh];
            float max_old = reg_max[hh];
#pragma unroll
            for (size_t i = 0; i < WARP_NUM; ++ i) {
                if (i == warp_id % WARP_NUM) continue;
                float sum_cur = s_sum[h][i];
                float max_cur = s_max[h][i];

                float max_new = fmaxf(max_old, max_cur);
                float alpha = __expf(max_old - max_new);
                float beta = __expf(max_cur - max_new);

                max_old = max_new;
                sum_old = alpha * sum_old + beta * sum_cur;
                float4 b = *reinterpret_cast<float4 *>(&s_acc[h][i][lane_id << 2]);
                reg_acc[hh].x = alpha * reg_acc[hh].x + beta * b.x;
                reg_acc[hh].y = alpha * reg_acc[hh].y + beta * b.y;
                reg_acc[hh].z = alpha * reg_acc[hh].z + beta * b.z;
                reg_acc[hh].w = alpha * reg_acc[hh].w + beta * b.w;
            }

            size_t nh = nkvh * HEAD_NUM + h;
            size_t offset = task_id * _nh + nh;
            if (lane_id == 0) {
                _attn_max[offset] = max_old;
                _attn_sum[offset] = sum_old;
            }

            offset *= HEAD_DIM;
            reinterpret_cast<float4 *>(_attn_acc + offset)[lane_id] = reg_acc[hh];
        }
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
template <typename T, size_t HEAD_DIM, size_t HEAD_NUM, size_t WARP_NUM, size_t TILE_M, size_t TILE_K, size_t COALESCE, bool QUANT>
void flash_decoding_launch(std::byte *attn_val,        // 输出 (batch_size, nh, dv)
                            std::byte *attn_acc,       // 输出 (batch_task_num, nh, dv) 
                            std::byte *attn_sum,       // 输出 (batch_task_num, nh, 1)
                            std::byte *attn_max,       // 输出 (batch_task_num, nh, 1)
                            const std::byte *q,        // q (batch_size, nh, d)
                            const std::byte *k_cache,  // k底层缓存 (block_num * block_size, nkvh, d)  
                            const std::byte *v_cache,  // q底层缓存 (block_num * block_size, nkvh, dv)
                            const std::byte *k_scale,  // k量化 scale (block_num*block_size, nkvh)，非量化为 nullptr
                            const std::byte *v_scale,  // v量化 scale (block_num*block_size, nkvh)，非量化为 nullptr
                            const std::byte *block_ids,// 块表 (batch_size, block_ids_size | block_ids)  kv cache 布局: (block_num, block_size, nkvh, d | dv) 
                            const std::byte *cut_idx,  // 序列起始位置 (batch_size + 1)
                            const std::byte *tot_len,  // 已计算 kv 总长 (batch_size)
                            const size_t block_size,   // 一个 block 的 token 数 (批次无关)
                            const size_t batch_size,   // 批次大小 (批次有关)
                            const size_t table_width,  // 块表宽度 (批次无关)
                            const size_t tot_block_num,// task 数
                            const float &scale,        // 缩放因子
                            const size_t &nh,          // 其他参数 ...
                            const size_t &nkvh)
{
    ASSERT(nh == HEAD_NUM * nkvh, "flash_decoding: nh != HEAD_NUM * nkvh");

    auto *d_attn = reinterpret_cast<T *>(attn_val);
    auto *d_attn_acc = reinterpret_cast<float *>(attn_acc);
    auto *d_attn_sum = reinterpret_cast<float *>(attn_sum);
    auto *d_attn_max = reinterpret_cast<float *>(attn_max);
    const auto *d_q = reinterpret_cast<const T *>(q);
    const void *d_k_cache = reinterpret_cast<const void *>(k_cache);
    const void *d_v_cache = reinterpret_cast<const void *>(v_cache);
    const auto *d_k_scale = reinterpret_cast<const float *>(k_scale);
    const auto *d_v_scale = reinterpret_cast<const float *>(v_scale);
    const auto *d_block_ids = reinterpret_cast<const int64_t *>(block_ids);
    const auto *d_cut_idx = reinterpret_cast<const int64_t *>(cut_idx);
    const auto *d_tot_len = reinterpret_cast<const int64_t *>(tot_len);

    const size_t group_num = (HEAD_NUM + TILE_M - 1) / TILE_M;
    const size_t block_dim = group_num * WARP_NUM * WARP_SIZE;
    const size_t tot_task_num = tot_block_num / COALESCE + batch_size;

    dim3 blockDim_0(block_dim);
    dim3 gridDim_0(tot_task_num, nkvh);

    dim3 blockDim_1(256);
    dim3 gridDim_1(nh, batch_size);

    flash_decoding_parallel_kernel<T, HEAD_DIM, HEAD_NUM, WARP_NUM, TILE_M, TILE_K, COALESCE, QUANT><<<gridDim_0, blockDim_0>>>(
        d_attn_acc, d_attn_sum, d_attn_max, d_q, d_k_cache, d_v_cache, d_k_scale, d_v_scale,
        d_block_ids, d_cut_idx, d_tot_len, block_size, batch_size, table_width, scale, nh, nkvh);
    CUDA_CHECK(cudaGetLastError());   // 隔离并行 kernel 错误
    flash_decoding_reduce_kernel<T, HEAD_DIM, COALESCE><<<gridDim_1, blockDim_1>>>(
        d_attn, d_attn_acc, d_attn_sum, d_attn_max, d_block_ids, table_width, nh);
    CUDA_CHECK(cudaGetLastError());   // 隔离 reduce kernel 错误
}

template <typename T>
void flash_decoding_dispatch(std::byte *attn_val,      // 输出 (batch_size, nh, dv)
                            std::byte *attn_acc,       // 输出 (batch_task_num, nh, dv) 
                            std::byte *attn_sum,       // 输出 (batch_task_num, nh, 1)
                            std::byte *attn_max,       // 输出 (batch_task_num, nh, 1)
                            const std::byte *q,        // q (batch_size, nh, d)
                            const std::byte *k_cache,  // k底层缓存 (block_num * block_size, nkvh, d)  
                            const std::byte *v_cache,  // q底层缓存 (block_num * block_size, nkvh, dv)
                            const std::byte *k_scale,  // k量化 scale，非量化为 nullptr
                            const std::byte *v_scale,  // v量化 scale，非量化为 nullptr
                            const std::byte *block_ids,// 块表 (batch_size, block_ids_size | block_ids)  kv cache 布局: (block_num, block_size, nkvh, d | dv) 
                            const std::byte *cut_idx,  // 序列起始位置 (batch_size + 1)
                            const std::byte *tot_len,  // 已计算 kv 总长 (batch_size)
                            const size_t block_size,   // 一个 block 的 token 数 (批次无关)
                            const size_t batch_size,   // 批次大小 (批次有关)
                            const size_t table_width,  // 块表宽度 (批次无关)
                            const size_t tot_block_num, // task 数
                            const float &scale,        // 缩放因子
                            const size_t &nh,          // 其他参数 ...
                            const size_t &dv, 
                            const size_t &d,
                            const size_t &nkvh)
{
    ASSERT(d == dv, "flash_decoding: d must equal dv.");
    ASSERT(d == 128, "flash_decoding: only HEAD_DIM==128 supported (Qwen2-1.5B / Llama-3.2-3B / Qwen2-7B).");
    const int heads = static_cast<int>(nh / nkvh);   // HEAD_NUM = rep（整组最优）
    const bool quant = (k_scale != nullptr);

    // ---- 选参器：按 GPU 架构(Blackwell sm_10x vs 其它) + 工作量(tot_task_num/batch) 选
    //     块内 warp 分组 (WARP_NUM, TILE_M) 与 TILE_K/COALESCE。阈值由 fd_wn_sweep 扫描拟合：
    //     - B300(sm_103)：单组 4 warp 处理全部 q 行 (WN=4,TM=6,128 线程) 全负载最优（2026-09-02 扫描）。
    //     - 其它(sm_89 等)：任务少(tot≤512)用 TM3/WN3，任务多用 TM2/WN2（省 smem 提占用）。
    //     选参阈值依赖具体架构，sm_89 拟合不能直接搬 sm_103 → 运行时按 device capability 分流。
    static const bool blackwell = []() {
        int dev = 0; cudaGetDevice(&dev);
        cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
        return prop.major >= 10;
    }();
    const size_t tot = tot_block_num;      // 总物理块数
    const size_t b = batch_size;
    int wn, tm, tk, co;
    if (blackwell) {
        // B300：bf16/f16（sizeof≤2）用 WN=4/TM=6 最优；f32 的 (4,6) 静态 smem ~47KB 超 48KB
        // 会被 nvcc 静默丢弃 → "named symbol not found"，故 f32 退到 (3,3)。
        wn = (sizeof(T) <= 2) ? 4 : 3;
        tm = (sizeof(T) <= 2) ? 6 : 3;
        tk = 16;
        co = (tot <= 512) ? 1 : (tot <= 2048) ? (b >= 128 ? 4 : 2) : 4;
    } else {
        wn = tm = (tot <= 512) ? 3 : 2;
        tk = 16;
        co = (tot <= 128) ? 1 : (tot <= 512) ? 2 : (tot <= 1024) ? 4 : 8;
    }

#define FD_LAUNCH(H, WN, TM, TK, CO, Q)                                                \
    flash_decoding_launch<T, 128, H, WN, TM, TK, CO, Q>(                                \
        attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, k_scale, v_scale, \
        block_ids, cut_idx, tot_len, block_size, batch_size, table_width, tot_block_num, \
        scale, nh, nkvh)
#define FD_CO(H, WN, TM, TK, Q)                                                        \
    if (co == 1)       FD_LAUNCH(H, WN, TM, TK, 1, Q);                                 \
    else if (co == 2)  FD_LAUNCH(H, WN, TM, TK, 2, Q);                                 \
    else if (co == 4)  FD_LAUNCH(H, WN, TM, TK, 4, Q);                                 \
    else               FD_LAUNCH(H, WN, TM, TK, 8, Q)
#define FD_TK(H, WN, TM, Q)                                                            \
    if (tk == 16)      FD_CO(H, WN, TM, 16, Q);                                        \
    else               FD_CO(H, WN, TM, 8, Q)
// (4,6) 仅对 bf16/f16（sizeof(T)<=2）实例化：f32 的 HEAD_NUM=7 静态 smem 50912B > 48KB 编不过；
// 且 (4,6) 运行时只在 Blackwell 且 sizeof(T)<=2 时才被选（f32 恒退 (3,3)），故 f32 不实例化 (4,6)。
#define FD_WT(H, Q)                                                                    \
    if (wn == 2 && tm == 2)      FD_TK(H, 2, 2, Q);                                    \
    else if (wn == 3 && tm == 3) FD_TK(H, 3, 3, Q);                                    \
    else if constexpr (sizeof(T) <= 2) FD_TK(H, 4, 6, Q);                              \
    else                                FD_TK(H, 3, 3, Q)

    if (heads == 6) {        // Qwen2-1.5B
        if (quant) FD_WT(6, true);
        else       FD_WT(6, false);
    } else if (heads == 3) { // Llama-3.2-3B
        if (quant) FD_WT(3, true);
        else       FD_WT(3, false);
    } else if (heads == 7) { // Qwen2-7B (28/4): HEAD_NUM=7（全 rep，块内 warp 分组）
        if (quant) FD_WT(7, true);
        else       FD_WT(7, false);
    } else {
        ASSERT(false, "flash_decoding: unsupported GQA ratio (nh/nkvh must be 6, 3 or 7).");
    }
#undef FD_WT
#undef FD_TK
#undef FD_CO
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
                    const size_t tot_block_num, // 精确 task 数
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
        return flash_decoding_dispatch<float>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, k_scale, v_scale, block_ids, cut_idx, tot_len, block_size,
                                             batch_size, table_width, tot_block_num, scale, nh, dv, d, nkvh);
    case LLAISYS_DTYPE_BF16:
        return flash_decoding_dispatch<llaisys::bf16_t>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, k_scale, v_scale, block_ids, cut_idx, tot_len, block_size,
                                             batch_size, table_width, tot_block_num, scale, nh, dv, d, nkvh);
    case LLAISYS_DTYPE_F16:
        return flash_decoding_dispatch<llaisys::fp16_t>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, k_scale, v_scale, block_ids, cut_idx, tot_len, block_size,
                                             batch_size, table_width, tot_block_num, scale, nh, dv, d, nkvh);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
