#include "rearrange_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t dtype,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides) {
    const size_t ndim = shape.size();
    if (ndim == 0) {
        return;
    }
    ASSERT(out_strides.size() == ndim && in_strides.size() == ndim, "rearrange: strides size mismatch.");

    size_t numel = 1;
    for (size_t d = 0; d < ndim; ++d) {
        numel *= shape[d];
    }
    if (numel == 0) {
        return;
    }

    const size_t elem_size = llaisys::utils::dsize(dtype);
    std::vector<size_t> coord(ndim, 0);

    for (size_t idx = 0; idx < numel; ++idx) {
        size_t tmp = idx;
        for (size_t d = ndim; d-- > 0;) {
            coord[d] = tmp % shape[d];
            tmp /= shape[d];
        }

        ptrdiff_t in_offset = 0;
        ptrdiff_t out_offset = 0;
        for (size_t d = 0; d < ndim; ++d) {
            in_offset += static_cast<ptrdiff_t>(coord[d]) * in_strides[d];
            out_offset += static_cast<ptrdiff_t>(coord[d]) * out_strides[d];
        }

        std::memcpy(out + out_offset * static_cast<ptrdiff_t>(elem_size),
                    in + in_offset * static_cast<ptrdiff_t>(elem_size),
                    elem_size);
    }
}
} // namespace llaisys::ops::cpu
