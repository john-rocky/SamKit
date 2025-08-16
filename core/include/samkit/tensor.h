#ifndef SAMKIT_TENSOR_H
#define SAMKIT_TENSOR_H

#include <vector>
#include <memory>
#include <cstdint>
#include <initializer_list>

namespace samkit {

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT64,
    UINT8
};

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<int>& shape, DataType dtype = DataType::FLOAT32);
    Tensor(std::initializer_list<int> shape, DataType dtype = DataType::FLOAT32);
    
    // Copy and move
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(const Tensor& other);
    Tensor& operator=(Tensor&& other) noexcept;
    
    ~Tensor();
    
    // Shape operations
    const std::vector<int>& shape() const { return shape_; }
    int rank() const { return static_cast<int>(shape_.size()); }
    int dim(int axis) const;
    size_t numel() const { return numel_; }
    
    // Data access
    DataType dtype() const { return dtype_; }
    size_t bytes() const { return numel_ * elementSize(); }
    void* data() { return data_.get(); }
    const void* data() const { return data_.get(); }
    
    // Typed data access
    template<typename T>
    T* data() { return static_cast<T*>(data_.get()); }
    
    template<typename T>
    const T* data() const { return static_cast<const T*>(data_.get()); }
    
    // Element access (for float tensors)
    float& at(const std::vector<int>& indices);
    const float& at(const std::vector<int>& indices) const;
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    
    // Reshape
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor view(const std::vector<int>& new_shape) const;
    
    // Operations
    Tensor transpose(int dim0, int dim1) const;
    Tensor permute(const std::vector<int>& dims) const;
    Tensor slice(int dim, int start, int end) const;
    
    // Factory methods
    static Tensor zeros(const std::vector<int>& shape, DataType dtype = DataType::FLOAT32);
    static Tensor ones(const std::vector<int>& shape, DataType dtype = DataType::FLOAT32);
    static Tensor fromFloat(const float* data, const std::vector<int>& shape);
    static Tensor fromUint8(const uint8_t* data, const std::vector<int>& shape);
    
    // Utilities
    void fill(float value);
    void copyFrom(const void* src);
    void copyTo(void* dst) const;
    bool isContiguous() const { return true; } // Simplified for now
    
private:
    std::vector<int> shape_;
    std::vector<int> strides_;
    DataType dtype_;
    size_t numel_;
    std::shared_ptr<void> data_;
    
    // Helper methods
    void computeStrides();
    size_t elementSize() const;
    size_t computeOffset(const std::vector<int>& indices) const;
    void allocate();
};

// Tensor operations
namespace TensorOps {
    
    // Normalization
    void normalize(Tensor& tensor, const std::vector<float>& mean, 
                  const std::vector<float>& std);
    
    // Conversion
    Tensor uint8ToFloat(const Tensor& input, float scale = 1.0f/255.0f);
    Tensor floatToUint8(const Tensor& input, float scale = 255.0f);
    
    // Resizing
    Tensor resize(const Tensor& input, int new_height, int new_width);
    Tensor pad(const Tensor& input, int pad_top, int pad_bottom, 
              int pad_left, int pad_right, float pad_value = 0.0f);
    
    // Channel operations
    Tensor permuteChannels(const Tensor& input, const std::vector<int>& order);
    
} // namespace TensorOps

} // namespace samkit

#endif // SAMKIT_TENSOR_H