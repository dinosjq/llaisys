#include "tensor.hpp"

#include "../ops/rearrange/op.hpp"
#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

/**
 * @brief Tensor 构造函数，根据元信息、存储对象和偏移量初始化一个张量对象。
 * @param meta    张量的元信息（数据类型、形状、步长等）
 * @param storage 底层数据存储对象，负责实际内存分配和管理
 * @param offset  数据在存储中的偏移量，支持视图等操作
 */
Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

/**
 * @brief 静态工厂方法，根据指定参数创建并返回一个新的张量对象。
 * @param shape       张量的形状（如 {2, 3, 4} 表示三维张量）
 * @param dtype       张量的数据类型（如 float、int32 等）
 * @param device_type 张量存储在哪种设备（如 CPU、GPU）
 * @param device      设备编号（如第几个 GPU）
 * @return            返回一个持有新分配内存和元信息的 Tensor 智能指针
 *
 * 主要流程：
 * 1. 计算步长（strides），用于多维数组的内存布局。
 * 2. 构造元信息结构体 TensorMeta。
 * 3. 计算总元素数和每个元素的字节数。
 * 4. 根据设备类型分配内存（主机或设备）。
 * 5. 创建并返回一个新的 Tensor 智能指针。
 */
tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    // 1. 计算步长（strides），用于多维数组的内存布局
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    // 2. 构造元信息结构体
    TensorMeta meta{dtype, shape, strides};
    // 3. 计算总元素数和每个元素的字节数
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    // 4. 根据设备类型分配内存
    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        // 分配主机内存（CPU）
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        // 转换当前的环境
        core::context().setDevice(device_type, device);
        // 分配设备内存（如 GPU）
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

/**
 * @brief 获取张量数据的可写指针。
 * @return 指向张量数据起始位置的 std::byte* 指针。
 */
std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

/**
 * @brief 获取张量数据的只读指针。
 * @return 指向张量数据起始位置的 const std::byte* 指针。
 */
const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

/**
 * @brief 获取张量的维度数。
 * @return 张量的维度数量。
 */
size_t Tensor::ndim() const {
    return _meta.shape.size();
}

/**
 * @brief 获取张量的形状（每一维的大小）。
 * @return 形状的常量引用。
 */
const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

/**
 * @brief 获取张量的步长信息。
 * @return 步长的常量引用。
 */
const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

/**
 * @brief 获取张量的数据类型。
 * @return 数据类型枚举值。
 */
llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

/**
 * @brief 获取张量所在的设备类型（如 CPU、GPU）。
 * @return 设备类型枚举值。
 */
llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

/**
 * @brief 获取张量所在的设备编号。
 * @return 设备编号。
 */
int Tensor::deviceId() const {
    return _storage->deviceId();
}

/**
 * @brief 获取张量的元素总数。
 * @return 元素数量。
 */
size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

/**
 * @brief 获取单个元素的字节数。
 * @return 元素字节数。
 */
size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

/**
 * @brief 获取张量的描述信息字符串。
 * @return 包含形状、步长、数据类型等信息的字符串。
 */
std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

/**
 * @brief 打印张量的调试信息和内容。
 *        若在CPU上直接打印，若在其他设备则先拷贝到主机再打印。
 */
void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

/**
 * @brief 判断张量是否为连续内存布局。
 *        检查每一维的stride是否等于上一维长度的累乘，只有全部匹配才认为是contiguous。
 * @return 如果张量是连续内存布局则返回true，否则返回false。
 */
bool Tensor::isContiguous() const {
    const auto &shape = this->shape();
    const auto &strides = this->strides();
    size_t ndim = this->ndim();
    size_t stride = 1;
    for (size_t i = 1; i <= ndim; ++i) {
        if (strides[ndim - i] != static_cast<ptrdiff_t>(stride)) {
            return false;
        }
        stride *= shape[ndim - i];
    }
    return true;
}

/**
 * @brief 生成一个新的张量，其维度顺序根据给定的 order 重新排列（不拷贝数据，仅改变 shape 和 strides）。
 *        新张量的每一维 shape 和 stride 都按照 order 重新排列，底层数据与原张量共享。
 * @param order 维度的新顺序（如 {1, 0, 2} 表示交换第0和第1维）
 * @return 新的 Tensor 视图对象
 * @throws std::runtime_error 如果 order 长度与张量维度不一致
 */
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t ndim = this->ndim();
    // 验证大小
    if (order.size() != ndim) {
        throw std::runtime_error("Tensor::permute : order size error");
    }
    // 验证序列
    std::vector<char> flag(ndim, 0);
    for (auto o : order) {
        if (o >= ndim || flag[o]) {
            throw std::runtime_error("Tensor::permute : order error");
        }
        flag[o] = 1;
    }
    std::vector<size_t> nshape(ndim);
    std::vector<ptrdiff_t> nstrides(ndim);
    const auto &shape = this->shape();
    const auto &strides = this->strides();
    for (size_t i = 0; i < ndim; ++i) {
        nshape[i] = shape[order[i]];     // 新 shape 按 order 排列
        nstrides[i] = strides[order[i]]; // 新 strides 也按 order 排列
    }
    TensorMeta nmeta{this->dtype(), nshape, nstrides};
    return std::shared_ptr<Tensor>(new Tensor(nmeta, _storage, _offset));
}

/**
 * @brief 创建一个新的张量视图（view），仅改变形状和步长，不拷贝数据。
 *        只有当原张量为连续内存布局且新形状与原元素总数一致时才允许创建视图，否则抛出异常。
 * @param shape 新视图的形状
 * @return 新的 Tensor 视图对象（与原张量共享底层存储）
 * @throws std::runtime_error 如果原张量不是连续内存或新形状元素总数不一致
 */
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 检查新形状元素总数是否与原张量一致，且原张量必须是连续内存
    if (this->numel() != std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>()) || !this->isContiguous()) {
        throw std::runtime_error("Tensor::view error.");
    }
    // 在此之前，应该先检查总数是否一致，还有原tensor内存分布是否连续，否则抛出异常
    // 重新计算新视图的 strides
    size_t ndim = shape.size();
    size_t stride = 1;
    std::vector<ptrdiff_t> strides(ndim);
    for (size_t i = 1; i <= ndim; ++i) {
        strides[ndim - i] = stride;
        stride *= shape[ndim - i];
    }
    TensorMeta _new_meta{this->dtype(), shape, strides};
    // 创建新视图对象，底层存储与原张量共享
    return std::shared_ptr<Tensor>(new Tensor(_new_meta, _storage, _offset));
}

/**
 * @brief 沿指定维度对张量进行切片，返回新的视图（不拷贝数据）。
 * @param dim   要切片的维度
 * @param start 起始索引（包含）
 * @param end   结束索引（不包含）
 * @return 新的 Tensor 视图对象
 */
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 增加范围校验
    const auto &shape = this->shape();
    if (dim >= shape.size() || start > end || end > shape[dim]) {
        throw std::runtime_error("Tensor::slice error");
    }
    std::vector<size_t> nshape(this->shape());
    nshape[dim] = end - start;
    // offest 是整体的偏移
    size_t noffset = _offset + start * this->strides()[dim] * this->elementSize();
    TensorMeta nmeta{this->dtype(), nshape, this->strides()};
    return std::shared_ptr<Tensor>(new Tensor(nmeta, _storage, noffset));
}

/**
 * @brief 从主机（CPU）内存加载数据到当前张量对象。
 *        如果张量在CPU上，直接拷贝；如果张量在设备（如GPU）上，使用runtime的memcpy_sync进行主机到设备的数据拷贝。
 * @param src_ 指向源数据的常量指针（通常为主机内存中的原始数据）。
 */
void Tensor::load(const void *src_) {
    size_t total_elems = this->numel() * this->elementSize();
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // 张量在CPU上，直接拷贝
        std::memcpy(this->data(), src_, total_elems);
    } else {
        // 张量在设备上，使用 runtime 的 memcpy_sync
        core::context().setDevice(this->deviceType(), this->deviceId());
        core::context().runtime().api()->memcpy_sync(
            this->data(),      // 目标：设备内存
            src_,              // 源：主机内存
            total_elems,       // 拷贝字节数
            LLAISYS_MEMCPY_H2D // 主机到设备
        );
    }
}

/**
 * @brief 返回一个连续内存布局的张量副本。
 *        如果当前张量已是连续内存，则直接返回视图；否则分配新内存并重排数据。
 * @return 连续内存布局的张量对象。
 */
tensor_t Tensor::contiguous() const {
    if (this->isContiguous()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }
    tensor_t out = Tensor::create(this->shape(), this->dtype(), this->deviceType(), this->deviceId());
    tensor_t in_view = std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    llaisys::ops::rearrange(out, in_view);
    return out;
}

/**
 * @brief 返回一个新形状的张量视图或副本。
 *        若原张量为连续内存，直接返回视图；否则先转为连续再视图。
 * @param shape 新形状
 * @return 新形状的张量对象
 * @throws std::runtime_error 元素数量不一致时报错
 */
tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
    if (new_numel != this->numel()) {
        throw std::runtime_error("Tensor::reshape : shape size mismatch");
    }
    if (this->isContiguous()) {
        return this->view(shape);
    }
    return this->contiguous()->view(shape);
}

/**
 * @brief 返回一个拷贝到指定设备的新张量。
 *        若目标设备与当前一致，返回视图；否则分配新内存并拷贝数据。
 * @param device_type 目标设备类型（如 CPU、GPU）
 * @param device 目标设备编号，<0 时自动推断
 * @return 新设备上的张量对象
 */
tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    int target_device = device;
    if (target_device < 0) {
        target_device = (device_type == this->deviceType()) ? this->deviceId() : 0;
    }

    if (device_type == this->deviceType() && target_device == this->deviceId()) {
        return std::shared_ptr<Tensor>(new Tensor(_meta, _storage, _offset));
    }

    tensor_t out = Tensor::create(this->shape(), this->dtype(), device_type, target_device);
    size_t total_bytes = this->numel() * this->elementSize();
    llaisysMemcpyKind_t kind = LLAISYS_MEMCPY_H2H;

    if (this->deviceType() == LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2H;
    } else if (this->deviceType() == LLAISYS_DEVICE_CPU && device_type != LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2D;
    } else if (this->deviceType() != LLAISYS_DEVICE_CPU && device_type == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_D2H;
    } else {
        kind = LLAISYS_MEMCPY_D2D;
    }

    if (kind == LLAISYS_MEMCPY_D2H) {
        core::context().setDevice(this->deviceType(), this->deviceId());
    } else if (kind != LLAISYS_MEMCPY_H2H) {
        core::context().setDevice(device_type, target_device);
    }

    core::context().runtime().api()->memcpy_sync(
        out->data(),
        this->data(),
        total_bytes,
        kind);

    return out;
}

} // namespace llaisys
