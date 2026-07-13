#pragma once

#include "types.hpp"

#include <cstdint>
#include <cstdlib>
#include <cuda_fp16.h>    // __half2float, __float2half, __half_raw
#include <cuda_bf16.h>    // __bfloat162float, __float2bfloat16, __nv_bfloat16_raw
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(call)                                                 \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << "\n";       \
            exit(1);                                                     \
        }                                                                \
    }

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define USHORT4(value) (reinterpret_cast<ushort4 *>(&(value))[0])

#define FLOAT2(value) (reinterpret_cast<float2 *>(&(value))[0])

namespace llaisys::utils::nvidia {

__device__ __forceinline__ float nv_bf16_to_f32(bf16_t val) {
    __nv_bfloat16_raw raw;
    raw.x = val._v;
    return __bfloat162float(raw);
}

__device__ __forceinline__ bf16_t nv_f32_to_bf16(float val) {
    __nv_bfloat16_raw raw = __float2bfloat16(val);
    return bf16_t{raw.x};
}

__device__ __forceinline__ float nv_f16_to_f32(fp16_t val) {
    __half_raw raw;
    raw.x = val._v;
    return __half2float(raw);
}

__device__ __forceinline__ fp16_t nv_f32_to_f16(float val) {
    __half_raw raw = __float2half(val);
    return fp16_t{raw.x};
}

template <typename TypeTo, typename TypeFrom>
__device__ __forceinline__ TypeTo cast(TypeFrom val) {
    if constexpr (std::is_same<TypeTo, TypeFrom>::value) {
        return val;
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && std::is_same<TypeFrom, float>::value) {
        return nv_f32_to_f16(val);
    } else if constexpr (std::is_same<TypeTo, fp16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return nv_f32_to_f16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && std::is_same<TypeTo, float>::value) {
        return nv_f16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, fp16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(nv_f16_to_f32(val));
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && std::is_same<TypeFrom, float>::value) {
        return nv_f32_to_bf16(val);
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && !std::is_same<TypeFrom, float>::value) {
        return nv_f32_to_bf16(static_cast<float>(val));
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && std::is_same<TypeTo, float>::value) {
        return nv_bf16_to_f32(val);
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && !std::is_same<TypeTo, float>::value) {
        return static_cast<TypeTo>(nv_bf16_to_f32(val));
    } else {
        return static_cast<TypeTo>(val);
    }
}

/**
 * 多数据类型向量化访存
 */
template <typename T>
__device__ __forceinline__ float4 load_4d(const T *data) {
    if constexpr (std::is_same_v<T, float>) {
        return reinterpret_cast<const float4 *>(data)[0];
    } else if constexpr (std::is_same_v<T, bf16_t>) {
        const ushort4 v4 = reinterpret_cast<const ushort4 *>(data)[0];
        float4 flo4;
        flo4.x = nv_bf16_to_f32(bf16_t{v4.x});
        flo4.y = nv_bf16_to_f32(bf16_t{v4.y});
        flo4.z = nv_bf16_to_f32(bf16_t{v4.z});
        flo4.w = nv_bf16_to_f32(bf16_t{v4.w});
        return flo4;
    } else {
        const ushort4 v4 = reinterpret_cast<const ushort4 *>(data)[0];
        float4 flo4;
        flo4.x = nv_f16_to_f32(fp16_t{v4.x});
        flo4.y = nv_f16_to_f32(fp16_t{v4.y});
        flo4.z = nv_f16_to_f32(fp16_t{v4.z});
        flo4.w = nv_f16_to_f32(fp16_t{v4.w});
        return flo4;
    }
}

/**
 * 由于内存存在没对齐的情况，导致使用向量化访存传输回去反而更慢
 */
template <typename T>
__device__ __forceinline__ void save_4d(T *data, const float4 &flo4) {
    if constexpr (std::is_same_v<T, float>) {
        reinterpret_cast<float4 *>(data)[0] = flo4;
    } else if constexpr (std::is_same_v<T, bf16_t>) {
        ushort4 v4;
        v4.x = nv_f32_to_bf16(flo4.x)._v;
        v4.y = nv_f32_to_bf16(flo4.y)._v;
        v4.z = nv_f32_to_bf16(flo4.z)._v;
        v4.w = nv_f32_to_bf16(flo4.w)._v;
        reinterpret_cast<ushort4 *>(data)[0] = v4;
    } else {
        ushort4 v4;
        v4.x = nv_f32_to_f16(flo4.x)._v;
        v4.y = nv_f32_to_f16(flo4.y)._v;
        v4.z = nv_f32_to_f16(flo4.z)._v;
        v4.w = nv_f32_to_f16(flo4.w)._v;
        reinterpret_cast<ushort4 *>(data)[0] = v4;
    }
}

template <typename T>
__device__ __forceinline__ T add(const T &a, const T &b) {
    if constexpr (std::is_same_v<T, float>) {
        return a + b;
    } else if constexpr (std::is_same_v<T, float2>) {
        return make_float2(a.x + b.x, a.y + b.y);
    } else if constexpr (std::is_same_v<T, float3>) {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    } else if constexpr (std::is_same_v<T, float4>) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}
} // namespace llaisys::utils::nvidia
