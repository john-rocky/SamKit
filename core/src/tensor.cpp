#include "samkit/tensor.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cstring>

namespace samkit {

Tensor::Tensor() : dtype_(DataType::FLOAT32), numel_(0) {
}

Tensor::Tensor(const std::vector<int>& shape, DataType dtype)
    : shape_(shape), dtype_(dtype) {
    computeStrides();
    allocate();
}

Tensor::Tensor(std::initializer_list<int> shape, DataType dtype)
    : shape_(shape), dtype_(dtype) {
    computeStrides();
    allocate();
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), strides_(other.strides_), 
      dtype_(other.dtype_), numel_(other.numel_) {
    allocate();
    if (other.data_) {
        copyFrom(other.data_.get());
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      dtype_(other.dtype_), numel_(other.numel_), data_(std::move(other.data_)) {
    other.numel_ = 0;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        shape_ = other.shape_;
        strides_ = other.strides_;
        dtype_ = other.dtype_;
        numel_ = other.numel_;
        allocate();
        if (other.data_) {
            copyFrom(other.data_.get());
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        dtype_ = other.dtype_;
        numel_ = other.numel_;
        data_ = std::move(other.data_);
        other.numel_ = 0;
    }
    return *this;
}

Tensor::~Tensor() = default;

int Tensor::dim(int axis) const {
    if (axis < 0) {
        axis += rank();
    }
    if (axis < 0 || axis >= rank()) {
        throw std::out_of_range("Tensor dimension out of range");
    }
    return shape_[axis];
}

float& Tensor::at(const std::vector<int>& indices) {
    if (dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("at() only supports FLOAT32 tensors");
    }
    size_t offset = computeOffset(indices);
    return static_cast<float*>(data_.get())[offset];
}

const float& Tensor::at(const std::vector<int>& indices) const {
    if (dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("at() only supports FLOAT32 tensors");
    }
    size_t offset = computeOffset(indices);
    return static_cast<const float*>(data_.get())[offset];
}

float& Tensor::operator[](size_t index) {
    if (dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("operator[] only supports FLOAT32 tensors");
    }
    if (index >= numel_) {
        throw std::out_of_range("Tensor index out of range");
    }
    return static_cast<float*>(data_.get())[index];
}

const float& Tensor::operator[](size_t index) const {
    if (dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("operator[] only supports FLOAT32 tensors");
    }
    if (index >= numel_) {
        throw std::out_of_range("Tensor index out of range");
    }
    return static_cast<const float*>(data_.get())[index];
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    // Calculate the product of new shape
    size_t new_numel = 1;
    int infer_idx = -1;
    
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (infer_idx != -1) {
                throw std::invalid_argument("Only one dimension can be inferred");
            }
            infer_idx = static_cast<int>(i);
        } else {
            new_numel *= new_shape[i];
        }
    }
    
    std::vector<int> final_shape = new_shape;
    if (infer_idx != -1) {
        final_shape[infer_idx] = static_cast<int>(numel_ / new_numel);
        new_numel = numel_;
    }
    
    if (new_numel != numel_) {
        throw std::invalid_argument("Cannot reshape tensor: size mismatch");
    }
    
    Tensor result(final_shape, dtype_);
    result.copyFrom(data_.get());
    return result;
}

Tensor Tensor::view(const std::vector<int>& new_shape) const {
    return reshape(new_shape);
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    if (dim0 < 0) dim0 += rank();
    if (dim1 < 0) dim1 += rank();
    
    if (dim0 >= rank() || dim1 >= rank()) {
        throw std::out_of_range("Transpose dimensions out of range");
    }
    
    std::vector<int> perm(rank());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[dim0], perm[dim1]);
    
    return permute(perm);
}

Tensor Tensor::permute(const std::vector<int>& dims) const {
    if (dims.size() != static_cast<size_t>(rank())) {
        throw std::invalid_argument("Permute dimensions mismatch");
    }
    
    std::vector<int> new_shape(rank());
    for (size_t i = 0; i < dims.size(); ++i) {
        new_shape[i] = shape_[dims[i]];
    }
    
    Tensor result(new_shape, dtype_);
    
    // Simplified implementation for common cases
    if (dtype_ == DataType::FLOAT32) {
        float* src = static_cast<float*>(data_.get());
        float* dst = static_cast<float*>(result.data_.get());
        
        // For now, implement a simple but inefficient permutation
        // This should be optimized for specific cases (e.g., NCHW -> NHWC)
        std::vector<int> src_indices(rank());
        std::vector<int> dst_indices(rank());
        
        size_t dst_idx = 0;
        std::function<void(int)> permute_recursive = [&](int depth) {
            if (depth == rank()) {
                // Map dst indices to src indices
                for (int i = 0; i < rank(); ++i) {
                    src_indices[dims[i]] = dst_indices[i];
                }
                size_t src_offset = computeOffset(src_indices);
                dst[dst_idx++] = src[src_offset];
            } else {
                for (int i = 0; i < new_shape[depth]; ++i) {
                    dst_indices[depth] = i;
                    permute_recursive(depth + 1);
                }
            }
        };
        
        permute_recursive(0);
    }
    
    return result;
}

Tensor Tensor::slice(int dim, int start, int end) const {
    if (dim < 0) dim += rank();
    if (dim >= rank()) {
        throw std::out_of_range("Slice dimension out of range");
    }
    
    if (start < 0) start += shape_[dim];
    if (end < 0) end += shape_[dim];
    
    start = std::max(0, start);
    end = std::min(shape_[dim], end);
    
    if (start >= end) {
        throw std::invalid_argument("Invalid slice range");
    }
    
    std::vector<int> new_shape = shape_;
    new_shape[dim] = end - start;
    
    Tensor result(new_shape, dtype_);
    
    // Simplified implementation
    // TODO: Optimize for contiguous slices
    if (dtype_ == DataType::FLOAT32) {
        float* src = static_cast<float*>(data_.get());
        float* dst = static_cast<float*>(result.data_.get());
        
        std::vector<int> indices(rank(), 0);
        size_t dst_idx = 0;
        
        std::function<void(int)> copy_recursive = [&](int depth) {
            if (depth == rank()) {
                std::vector<int> src_indices = indices;
                src_indices[dim] += start;
                size_t src_offset = computeOffset(src_indices);
                dst[dst_idx++] = src[src_offset];
            } else if (depth == dim) {
                for (int i = 0; i < new_shape[depth]; ++i) {
                    indices[depth] = i;
                    copy_recursive(depth + 1);
                }
            } else {
                for (int i = 0; i < shape_[depth]; ++i) {
                    indices[depth] = i;
                    copy_recursive(depth + 1);
                }
            }
        };
        
        copy_recursive(0);
    }
    
    return result;
}

Tensor Tensor::zeros(const std::vector<int>& shape, DataType dtype) {
    Tensor result(shape, dtype);
    std::memset(result.data_.get(), 0, result.bytes());
    return result;
}

Tensor Tensor::ones(const std::vector<int>& shape, DataType dtype) {
    Tensor result(shape, dtype);
    if (dtype == DataType::FLOAT32) {
        result.fill(1.0f);
    } else {
        // For other types, implement as needed
        throw std::runtime_error("ones() not implemented for this dtype");
    }
    return result;
}

Tensor Tensor::fromFloat(const float* data, const std::vector<int>& shape) {
    Tensor result(shape, DataType::FLOAT32);
    result.copyFrom(data);
    return result;
}

Tensor Tensor::fromUint8(const uint8_t* data, const std::vector<int>& shape) {
    Tensor result(shape, DataType::UINT8);
    result.copyFrom(data);
    return result;
}

void Tensor::fill(float value) {
    if (dtype_ != DataType::FLOAT32) {
        throw std::runtime_error("fill() only supports FLOAT32 tensors");
    }
    float* ptr = static_cast<float*>(data_.get());
    std::fill(ptr, ptr + numel_, value);
}

void Tensor::copyFrom(const void* src) {
    if (src && data_) {
        std::memcpy(data_.get(), src, bytes());
    }
}

void Tensor::copyTo(void* dst) const {
    if (dst && data_) {
        std::memcpy(dst, data_.get(), bytes());
    }
}

void Tensor::computeStrides() {
    strides_.resize(shape_.size());
    numel_ = 1;
    
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        strides_[i] = static_cast<int>(numel_);
        numel_ *= shape_[i];
    }
}

size_t Tensor::elementSize() const {
    switch (dtype_) {
        case DataType::FLOAT32:
        case DataType::INT32:
            return 4;
        case DataType::FLOAT16:
            return 2;
        case DataType::INT64:
            return 8;
        case DataType::UINT8:
            return 1;
        default:
            return 4;
    }
}

size_t Tensor::computeOffset(const std::vector<int>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index dimension mismatch");
    }
    
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] < 0 || indices[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of range");
        }
        offset += indices[i] * strides_[i];
    }
    
    return offset;
}

void Tensor::allocate() {
    size_t total_bytes = bytes();
    if (total_bytes > 0) {
        // Use aligned allocation for better performance
        void* raw_ptr = nullptr;
        if (posix_memalign(&raw_ptr, 64, total_bytes) != 0) {
            throw std::bad_alloc();
        }
        data_ = std::shared_ptr<void>(raw_ptr, free);
    }
}

// TensorOps implementation

void TensorOps::normalize(Tensor& tensor, const std::vector<float>& mean,
                         const std::vector<float>& std) {
    if (tensor.dtype() != DataType::FLOAT32) {
        throw std::runtime_error("normalize() only supports FLOAT32 tensors");
    }
    
    if (tensor.rank() < 3) {
        throw std::invalid_argument("Tensor must have at least 3 dimensions (CHW or HWC)");
    }
    
    // Assume channels are the third-to-last dimension (CHW format)
    int channels = tensor.dim(-3);
    int height = tensor.dim(-2);
    int width = tensor.dim(-1);
    
    if (static_cast<size_t>(channels) != mean.size() || 
        static_cast<size_t>(channels) != std.size()) {
        throw std::invalid_argument("Mean/std size mismatch with channels");
    }
    
    float* data = tensor.data<float>();
    
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                size_t idx = c * height * width + h * width + w;
                data[idx] = (data[idx] - mean[c]) / std[c];
            }
        }
    }
}

Tensor TensorOps::uint8ToFloat(const Tensor& input, float scale) {
    if (input.dtype() != DataType::UINT8) {
        throw std::runtime_error("Input must be UINT8 tensor");
    }
    
    Tensor result(input.shape(), DataType::FLOAT32);
    const uint8_t* src = input.data<uint8_t>();
    float* dst = result.data<float>();
    
    for (size_t i = 0; i < input.numel(); ++i) {
        dst[i] = src[i] * scale;
    }
    
    return result;
}

Tensor TensorOps::floatToUint8(const Tensor& input, float scale) {
    if (input.dtype() != DataType::FLOAT32) {
        throw std::runtime_error("Input must be FLOAT32 tensor");
    }
    
    Tensor result(input.shape(), DataType::UINT8);
    const float* src = input.data<float>();
    uint8_t* dst = result.data<uint8_t>();
    
    for (size_t i = 0; i < input.numel(); ++i) {
        float val = src[i] * scale;
        val = std::max(0.0f, std::min(255.0f, val));
        dst[i] = static_cast<uint8_t>(val + 0.5f);
    }
    
    return result;
}

} // namespace samkit