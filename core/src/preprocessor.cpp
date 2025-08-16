#include "samkit/preprocessor.h"
#include <algorithm>
#include <cmath>

namespace samkit {

Preprocessor::Preprocessor(int model_size) 
    : model_size_(model_size) {
    // Default normalization parameters for SAM
    normalize_params_ = NormalizeParams(
        {123.675f, 116.28f, 103.53f},
        {58.395f, 57.12f, 57.375f}
    );
}

Preprocessor::~Preprocessor() = default;

void Preprocessor::setNormalization(const NormalizeParams& params) {
    normalize_params_ = params;
}

std::pair<Tensor, TransformParams> Preprocessor::process(const Image& image) const {
    // 1. Resize and pad the image
    auto [padded_image, transform] = resizeAndPad(image);
    
    // 2. Convert to tensor
    Tensor tensor = imageToTensor(padded_image);
    
    // 3. Normalize
    normalizeTensor(tensor);
    
    return {tensor, transform};
}

std::pair<Image, TransformParams> Preprocessor::resizeAndPad(const Image& image) const {
    TransformParams transform = computeTransform(image.width(), image.height());
    
    // Calculate scaled dimensions
    int scaled_width = static_cast<int>(image.width() * transform.scale);
    int scaled_height = static_cast<int>(image.height() * transform.scale);
    
    // Resize image
    Image resized = image.resize(scaled_width, scaled_height);
    
    // Create padded image
    Image padded(model_size_, model_size_, image.channels(), image.format());
    
    // Fill with zeros (black padding)
    std::fill(padded.data(), padded.data() + padded.size(), 0);
    
    // Copy resized image to center of padded image
    int pad_left = static_cast<int>(transform.pad_x);
    int pad_top = static_cast<int>(transform.pad_y);
    
    for (int y = 0; y < scaled_height; ++y) {
        for (int x = 0; x < scaled_width; ++x) {
            for (int c = 0; c < image.channels(); ++c) {
                padded.at(x + pad_left, y + pad_top, c) = resized.at(x, y, c);
            }
        }
    }
    
    return {padded, transform};
}

Tensor Preprocessor::imageToTensor(const Image& image) const {
    // Convert image to RGB if needed
    Image rgb_image = image;
    if (image.format() != ImageFormat::RGB) {
        rgb_image = image.convertTo(ImageFormat::RGB);
    }
    
    // Create tensor in CHW format (3, H, W)
    Tensor tensor({3, model_size_, model_size_}, DataType::FLOAT32);
    float* data = tensor.data<float>();
    
    // Copy and convert uint8 to float
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < model_size_; ++y) {
            for (int x = 0; x < model_size_; ++x) {
                int tensor_idx = c * model_size_ * model_size_ + y * model_size_ + x;
                data[tensor_idx] = static_cast<float>(rgb_image.at(x, y, c));
            }
        }
    }
    
    return tensor;
}

void Preprocessor::normalizeTensor(Tensor& tensor) const {
    if (tensor.rank() != 3 || tensor.dim(0) != 3) {
        throw std::invalid_argument("Expected tensor shape (3, H, W)");
    }
    
    float* data = tensor.data<float>();
    int height = tensor.dim(1);
    int width = tensor.dim(2);
    
    for (int c = 0; c < 3; ++c) {
        float mean = normalize_params_.mean[c];
        float std = normalize_params_.std[c];
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int idx = c * height * width + y * width + x;
                data[idx] = (data[idx] - mean) / std;
            }
        }
    }
}

TransformParams Preprocessor::computeTransform(int orig_width, int orig_height) const {
    TransformParams transform;
    transform.original_width = orig_width;
    transform.original_height = orig_height;
    transform.model_size = model_size_;
    
    // Compute scale to fit the longer side to model_size
    transform.scale = computeScale(orig_width, orig_height);
    
    // Compute scaled dimensions
    int scaled_width = static_cast<int>(orig_width * transform.scale);
    int scaled_height = static_cast<int>(orig_height * transform.scale);
    
    // Compute padding
    auto [pad_x, pad_y] = computePadding(scaled_width, scaled_height);
    transform.pad_x = static_cast<float>(pad_x);
    transform.pad_y = static_cast<float>(pad_y);
    
    return transform;
}

Tensor Preprocessor::encodePoints(const std::vector<Point>& points,
                                  const TransformParams& transform) const {
    if (points.empty()) {
        // Return empty tensor
        return Tensor({0, 3}, DataType::FLOAT32);
    }
    
    // Shape: (N, 3) where each row is [x, y, label]
    Tensor tensor({static_cast<int>(points.size()), 3}, DataType::FLOAT32);
    float* data = tensor.data<float>();
    
    for (size_t i = 0; i < points.size(); ++i) {
        Point model_point = transform.toModel(points[i]);
        data[i * 3 + 0] = model_point.x;
        data[i * 3 + 1] = model_point.y;
        data[i * 3 + 2] = static_cast<float>(model_point.label);
    }
    
    return tensor;
}

Tensor Preprocessor::encodeBox(const Box& box, const TransformParams& transform) const {
    if (box.empty()) {
        // Return empty tensor
        return Tensor({0, 4}, DataType::FLOAT32);
    }
    
    // Shape: (1, 4) where row is [x0, y0, x1, y1]
    Tensor tensor({1, 4}, DataType::FLOAT32);
    float* data = tensor.data<float>();
    
    Box model_box = transform.toModel(box);
    data[0] = model_box.x0;
    data[1] = model_box.y0;
    data[2] = model_box.x1;
    data[3] = model_box.y1;
    
    return tensor;
}

Tensor Preprocessor::encodeMask(const Mask& mask, const TransformParams& transform) const {
    if (mask.width == 0 || mask.height == 0) {
        // Return empty tensor
        return Tensor({1, model_size_, model_size_}, DataType::FLOAT32);
    }
    
    // Create tensor for mask input
    Tensor tensor({1, model_size_, model_size_}, DataType::FLOAT32);
    tensor.fill(0.0f);
    
    float* data = tensor.data<float>();
    
    // Resample mask to model size
    // This is a simplified bilinear interpolation
    float x_scale = static_cast<float>(mask.width) / model_size_;
    float y_scale = static_cast<float>(mask.height) / model_size_;
    
    for (int y = 0; y < model_size_; ++y) {
        for (int x = 0; x < model_size_; ++x) {
            float src_x = x * x_scale;
            float src_y = y * y_scale;
            
            int x0 = static_cast<int>(src_x);
            int y0 = static_cast<int>(src_y);
            int x1 = std::min(x0 + 1, mask.width - 1);
            int y1 = std::min(y0 + 1, mask.height - 1);
            
            float dx = src_x - x0;
            float dy = src_y - y0;
            
            // Get mask values (assuming logits if available, otherwise alpha)
            float v00, v01, v10, v11;
            if (!mask.logits.empty()) {
                v00 = mask.logits[y0 * mask.width + x0];
                v01 = mask.logits[y0 * mask.width + x1];
                v10 = mask.logits[y1 * mask.width + x0];
                v11 = mask.logits[y1 * mask.width + x1];
            } else {
                v00 = mask.alpha[y0 * mask.width + x0] / 255.0f;
                v01 = mask.alpha[y0 * mask.width + x1] / 255.0f;
                v10 = mask.alpha[y1 * mask.width + x0] / 255.0f;
                v11 = mask.alpha[y1 * mask.width + x1] / 255.0f;
            }
            
            // Bilinear interpolation
            float v0 = v00 * (1 - dx) + v01 * dx;
            float v1 = v10 * (1 - dx) + v11 * dx;
            float v = v0 * (1 - dy) + v1 * dy;
            
            data[y * model_size_ + x] = v;
        }
    }
    
    return tensor;
}

float Preprocessor::computeScale(int width, int height) const {
    int long_side = std::max(width, height);
    return static_cast<float>(model_size_) / long_side;
}

std::pair<int, int> Preprocessor::computePadding(int scaled_width, int scaled_height) const {
    int pad_x = (model_size_ - scaled_width) / 2;
    int pad_y = (model_size_ - scaled_height) / 2;
    return {pad_x, pad_y};
}

} // namespace samkit