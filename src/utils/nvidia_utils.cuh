#pragma once

#include "types.hpp"

#include <cstdint>
#include <cstdlib>
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
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;
    return __uint_as_float(bits32);
}

__device__ __forceinline__ bf16_t nv_f32_to_bf16(float val) {
    uint32_t bits32 = __float_as_uint(val);
    const uint32_t rounding_bias = 0x00007FFFu + ((bits32 >> 16) & 1u);
    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);
    return bf16_t{bf16_bits};
}

__device__ __forceinline__ float nv_f16_to_f32(fp16_t val) {
    uint16_t h = val._v;
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FFu;

    uint32_t f32;
    if (exponent == 31) {
        f32 = sign | 0x7F800000u | (mantissa << 13);
    } else if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign;
        } else {
            exponent = -14;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FFu;
            f32 = sign | (static_cast<uint32_t>(exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        f32 = sign | (static_cast<uint32_t>(exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    return __uint_as_float(f32);
}

__device__ __forceinline__ fp16_t nv_f32_to_f16(float val) {
    uint32_t f32 = __float_as_uint(val);
    uint16_t sign = static_cast<uint16_t>((f32 >> 16) & 0x8000u);
    int32_t exponent = static_cast<int32_t>((f32 >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32 & 0x7FFFFFu;

    if (exponent >= 16) {
        if (exponent == 128 && mantissa != 0) {
            return fp16_t{static_cast<uint16_t>(sign | 0x7E00u)};
        }
        return fp16_t{static_cast<uint16_t>(sign | 0x7C00u)};
    } else if (exponent >= -14) {
        return fp16_t{static_cast<uint16_t>(sign | ((exponent + 15) << 10) | (mantissa >> 13))};
    } else if (exponent >= -24) {
        mantissa |= 0x800000u;
        mantissa >>= (-14 - exponent);
        return fp16_t{static_cast<uint16_t>(sign | (mantissa >> 13))};
    } else {
        return fp16_t{sign};
    }
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
