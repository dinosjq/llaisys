#include "../../../utils.hpp"
#include "topk_cpu.hpp"

#include <cmath>
#include <queue>
#include <utility>
#include <type_traits>

template <typename T>
void _topk(size_t *out_idx, T *out_val, const T *vals, size_t numel, size_t k) {
    auto cmp = [&](const size_t &i1, const size_t &i2)-> bool {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>){
            return llaisys::utils::cast<float>(vals[i1]) > llaisys::utils::cast<float> (vals[i2]);
        }
        else {
            return vals[i1] > vals[i2];
        }
    };
    std::priority_queue<size_t, std::vector<size_t>, decltype(cmp)> heap(cmp);
    for(size_t i = 0; i < numel; ++ i){
        heap.push(i);
        if(heap.size() > k) heap.pop();
    }
    for(int i = k - 1; i >= 0; -- i){
        size_t p = heap.top();
        heap.pop();
        out_idx[i] = p;
        out_val[i] = vals[p];
    }
}

template <typename T>
void _topk_launch(std::byte *out_idx, std::byte *out_val, const std::byte *vals, size_t numel, size_t k){
    size_t *d_out_idx = reinterpret_cast<size_t *>(out_idx);
    T *d_out_val = reinterpret_cast<T *>(out_val);
    const T *d_vals = reinterpret_cast<const T *>(vals);
    return _topk<T>(d_out_idx, d_out_val, d_vals, numel, k);
}

namespace llaisys::ops::cpu {
void topk(std::byte *out_idx, std::byte *out_val, const std::byte *vals, llaisysDataType_t type, size_t numel, size_t k){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return _topk_launch<float>(out_idx, out_val, vals, numel, k);
    case LLAISYS_DTYPE_BF16:
        return _topk_launch<llaisys::bf16_t>(out_idx, out_val, vals, numel, k);
    case LLAISYS_DTYPE_F16:
        return _topk_launch<llaisys::fp16_t>(out_idx, out_val, vals, numel, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
