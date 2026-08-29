#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
// W8A8 量化线性: 输入激活已量化 (in_q I8 + a_scale F32 per-token), cuBLAS INT8 GEMM + epilogue
void linear_w8a8(std::byte *out, const std::byte *in_q, const std::byte *a_scale, const std::byte *weight_i8,
                 const std::byte *w_scale, const std::byte *bias, llaisysDataType_t dtype, size_t m, size_t n,
                 size_t k);
} // namespace llaisys::ops::nvidia
