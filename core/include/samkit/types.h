#ifndef SAMKIT_TYPES_H
#define SAMKIT_TYPES_H

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

namespace samkit {

// Forward declarations
class Image;
class Tensor;

// Point prompt for SAM
struct Point {
    float x;
    float y;
    int label;  // 1 = positive, 0 = negative
    
    Point(float x_, float y_, int label_ = 1) 
        : x(x_), y(y_), label(label_) {}
};

// Bounding box
struct Box {
    float x0, y0;  // Top-left corner
    float x1, y1;  // Bottom-right corner
    
    Box() : x0(0), y0(0), x1(0), y1(0) {}
    Box(float x0_, float y0_, float x1_, float y1_)
        : x0(x0_), y0(y0_), x1(x1_), y1(y1_) {}
    
    float width() const { return x1 - x0; }
    float height() const { return y1 - y0; }
    bool empty() const { return width() <= 0 || height() <= 0; }
};

// SAM inference options
struct Options {
    bool multimask_output = true;
    bool return_logits = false;
    float mask_threshold = 0.0f;
    int max_masks = 3;
};

// Single mask result
struct Mask {
    int width;
    int height;
    std::vector<float> logits;    // Optional: raw logits
    std::vector<uint8_t> alpha;   // 0-255 alpha mask
    float score;                   // IoU/confidence score
    
    Mask() : width(0), height(0), score(0.0f) {}
};

// SAM inference result
struct Result {
    std::vector<Mask> masks;
    bool success;
    std::string error_message;
    
    Result() : success(true) {}
};

// Image format
enum class ImageFormat {
    RGB,
    RGBA,
    BGR,
    BGRA,
    GRAY
};

// Model type
enum class ModelType {
    SAM2_1_TINY,
    SAM2_1_SMALL,
    SAM2_1_BASE,
    SAM2_1_LARGE,
    SAM2_1_BASE_PLUS,
    MOBILE_SAM,
    CUSTOM
};

// Runtime configuration
struct RuntimeConfig {
    enum ComputeUnit {
        CPU_ONLY,
        GPU_PREFERRED,
        NEURAL_ENGINE_PREFERRED,
        BEST_AVAILABLE
    };
    
    ComputeUnit compute_unit = BEST_AVAILABLE;
    int num_threads = 4;
    bool enable_fp16 = true;
    bool enable_quantization = false;
};

// Transform parameters for coordinate mapping
struct TransformParams {
    float scale;
    float pad_x;
    float pad_y;
    int original_width;
    int original_height;
    int model_size;
    
    TransformParams() 
        : scale(1.0f), pad_x(0), pad_y(0), 
          original_width(0), original_height(0), model_size(1024) {}
    
    // Convert from image coordinates to model coordinates
    Point toModel(const Point& p) const {
        return Point(
            p.x * scale + pad_x,
            p.y * scale + pad_y,
            p.label
        );
    }
    
    // Convert from model coordinates to image coordinates
    Point toImage(const Point& p) const {
        return Point(
            (p.x - pad_x) / scale,
            (p.y - pad_y) / scale,
            p.label
        );
    }
    
    Box toModel(const Box& b) const {
        return Box(
            b.x0 * scale + pad_x,
            b.y0 * scale + pad_y,
            b.x1 * scale + pad_x,
            b.y1 * scale + pad_y
        );
    }
    
    Box toImage(const Box& b) const {
        return Box(
            (b.x0 - pad_x) / scale,
            (b.y0 - pad_y) / scale,
            (b.x1 - pad_x) / scale,
            (b.y1 - pad_y) / scale
        );
    }
};

// Model normalization parameters
struct NormalizeParams {
    std::vector<float> mean;
    std::vector<float> std;
    
    NormalizeParams() 
        : mean({123.675f, 116.28f, 103.53f}),
          std({58.395f, 57.12f, 57.375f}) {}
    
    NormalizeParams(const std::vector<float>& m, const std::vector<float>& s)
        : mean(m), std(s) {}
};

} // namespace samkit

#endif // SAMKIT_TYPES_H