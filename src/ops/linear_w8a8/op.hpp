#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// W8A8 量化线性（NVIDIA only）：输入激活已量化（in_q I8 + a_scale F32 per-token），
// 权重 int8[n,k] + per-channel F32 scale。流程：cuBLAS INT8 GEMM(acc_i32) → epilogue(acc*a_scale*w_scale+bias→out)。
// acc 为算子内部静态 workspace，接口不含工作区。
void linear_w8a8(tensor_t out, tensor_t in_q, tensor_t a_scale, tensor_t weight_i8, tensor_t w_scale, tensor_t bias);
} // namespace llaisys::ops
