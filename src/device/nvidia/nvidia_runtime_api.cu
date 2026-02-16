#include "../runtime_api.hpp"
#include "../../utils/nvidia_utils.cuh"

#include <cstdlib>
#include <cstring>

namespace llaisys::device::nvidia {

namespace runtime_api {

/**
 * cuda device prop
 */
int getDeviceCount() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

/**
 * cuda stream
 */
void deviceSynchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    return static_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    CUDA_CHECK(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
}

void streamSynchronize(llaisysStream_t stream) {
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

/**
 * cuda memory
 */
void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    CUDA_CHECK(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, static_cast<cudaMemcpyKind>(kind)));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, size, static_cast<cudaMemcpyKind>(kind), static_cast<cudaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
