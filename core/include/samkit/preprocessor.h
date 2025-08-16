#ifndef SAMKIT_PREPROCESSOR_H
#define SAMKIT_PREPROCESSOR_H

#include "samkit/types.h"
#include "samkit/image.h"
#include "samkit/tensor.h"

namespace samkit {

class Preprocessor {
public:
    explicit Preprocessor(int model_size = 1024);
    ~Preprocessor();
    
    // Set normalization parameters
    void setNormalization(const NormalizeParams& params);
    
    // Main preprocessing pipeline
    // Returns preprocessed tensor and transform parameters
    std::pair<Tensor, TransformParams> process(const Image& image) const;
    
    // Individual preprocessing steps
    std::pair<Image, TransformParams> resizeAndPad(const Image& image) const;
    Tensor imageToTensor(const Image& image) const;
    void normalizeTensor(Tensor& tensor) const;
    
    // Coordinate transformation
    TransformParams computeTransform(int orig_width, int orig_height) const;
    
    // Prompt encoding
    Tensor encodePoints(const std::vector<Point>& points, 
                       const TransformParams& transform) const;
    Tensor encodeBox(const Box& box, const TransformParams& transform) const;
    Tensor encodeMask(const Mask& mask, const TransformParams& transform) const;
    
private:
    int model_size_;
    NormalizeParams normalize_params_;
    
    // Helper methods
    float computeScale(int width, int height) const;
    std::pair<int, int> computePadding(int scaled_width, int scaled_height) const;
};

} // namespace samkit

#endif // SAMKIT_PREPROCESSOR_H