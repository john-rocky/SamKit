#ifndef SAMKIT_POSTPROCESSOR_H
#define SAMKIT_POSTPROCESSOR_H

#include "samkit/types.h"
#include "samkit/tensor.h"

namespace samkit {

class Postprocessor {
public:
    explicit Postprocessor(int model_size = 1024);
    ~Postprocessor();
    
    // Main postprocessing pipeline
    Result process(const Tensor& mask_logits, 
                  const Tensor& iou_predictions,
                  const TransformParams& transform,
                  const Options& options) const;
    
    // Individual postprocessing steps
    Tensor upsampleMasks(const Tensor& masks) const;
    Tensor removePadding(const Tensor& masks, const TransformParams& transform) const;
    Tensor resizeToOriginal(const Tensor& masks, const TransformParams& transform) const;
    
    // Mask operations
    std::vector<Mask> extractMasks(const Tensor& masks, 
                                   const Tensor& scores,
                                   const Options& options) const;
    
    // Thresholding and conversion
    Tensor applyThreshold(const Tensor& logits, float threshold) const;
    std::vector<uint8_t> tensorToAlpha(const Tensor& tensor) const;
    
    // Mask refinement
    Tensor refineMask(const Tensor& mask) const;
    
private:
    int model_size_;
    
    // Helper methods
    Tensor bilinearUpsample(const Tensor& input, int scale_factor) const;
    std::vector<int> sortMasksByScore(const std::vector<float>& scores) const;
};

} // namespace samkit

#endif // SAMKIT_POSTPROCESSOR_H