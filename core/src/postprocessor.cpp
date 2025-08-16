#include "samkit/postprocessor.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace samkit {

Postprocessor::Postprocessor(int model_size)
    : model_size_(model_size) {
}

Postprocessor::~Postprocessor() = default;

Result Postprocessor::process(const Tensor& mask_logits,
                             const Tensor& iou_predictions,
                             const TransformParams& transform,
                             const Options& options) const {
    Result result;
    
    try {
        // 1. Upsample masks from low resolution to model size
        Tensor upsampled = upsampleMasks(mask_logits);
        
        // 2. Remove padding
        Tensor depadded = removePadding(upsampled, transform);
        
        // 3. Resize to original image size
        Tensor final_masks = resizeToOriginal(depadded, transform);
        
        // 4. Extract individual masks with scores
        result.masks = extractMasks(final_masks, iou_predictions, options);
        
        result.success = true;
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = e.what();
    }
    
    return result;
}

Tensor Postprocessor::upsampleMasks(const Tensor& masks) const {
    // Assuming masks shape is (N, 1, H, W) where H, W are low resolution (e.g., 256x256)
    // We need to upsample to model_size (e.g., 1024x1024)
    
    if (masks.rank() != 4) {
        throw std::invalid_argument("Expected masks tensor with rank 4 (N, 1, H, W)");
    }
    
    int num_masks = masks.dim(0);
    int low_res_h = masks.dim(2);
    int low_res_w = masks.dim(3);
    
    // Calculate scale factor
    int scale_factor = model_size_ / low_res_h;  // Assuming square masks
    
    // Create upsampled tensor
    Tensor upsampled({num_masks, 1, model_size_, model_size_}, DataType::FLOAT32);
    
    const float* src = masks.data<float>();
    float* dst = upsampled.data<float>();
    
    // Bilinear upsampling for each mask
    for (int n = 0; n < num_masks; ++n) {
        const float* mask_src = src + n * low_res_h * low_res_w;
        float* mask_dst = dst + n * model_size_ * model_size_;
        
        for (int y = 0; y < model_size_; ++y) {
            float src_y = static_cast<float>(y) / scale_factor;
            int y0 = static_cast<int>(src_y);
            int y1 = std::min(y0 + 1, low_res_h - 1);
            float dy = src_y - y0;
            
            for (int x = 0; x < model_size_; ++x) {
                float src_x = static_cast<float>(x) / scale_factor;
                int x0 = static_cast<int>(src_x);
                int x1 = std::min(x0 + 1, low_res_w - 1);
                float dx = src_x - x0;
                
                // Bilinear interpolation
                float v00 = mask_src[y0 * low_res_w + x0];
                float v01 = mask_src[y0 * low_res_w + x1];
                float v10 = mask_src[y1 * low_res_w + x0];
                float v11 = mask_src[y1 * low_res_w + x1];
                
                float v0 = v00 * (1 - dx) + v01 * dx;
                float v1 = v10 * (1 - dx) + v11 * dx;
                float v = v0 * (1 - dy) + v1 * dy;
                
                mask_dst[y * model_size_ + x] = v;
            }
        }
    }
    
    return upsampled;
}

Tensor Postprocessor::removePadding(const Tensor& masks, const TransformParams& transform) const {
    int num_masks = masks.dim(0);
    
    // Calculate the actual image size in model space
    int scaled_width = static_cast<int>(transform.original_width * transform.scale);
    int scaled_height = static_cast<int>(transform.original_height * transform.scale);
    
    // Calculate padding
    int pad_left = static_cast<int>(transform.pad_x);
    int pad_top = static_cast<int>(transform.pad_y);
    
    // Create depadded tensor
    Tensor depadded({num_masks, 1, scaled_height, scaled_width}, DataType::FLOAT32);
    
    const float* src = masks.data<float>();
    float* dst = depadded.data<float>();
    
    // Copy unpadded region for each mask
    for (int n = 0; n < num_masks; ++n) {
        const float* mask_src = src + n * model_size_ * model_size_;
        float* mask_dst = dst + n * scaled_height * scaled_width;
        
        for (int y = 0; y < scaled_height; ++y) {
            for (int x = 0; x < scaled_width; ++x) {
                int src_idx = (y + pad_top) * model_size_ + (x + pad_left);
                int dst_idx = y * scaled_width + x;
                mask_dst[dst_idx] = mask_src[src_idx];
            }
        }
    }
    
    return depadded;
}

Tensor Postprocessor::resizeToOriginal(const Tensor& masks, const TransformParams& transform) const {
    int num_masks = masks.dim(0);
    int scaled_height = masks.dim(2);
    int scaled_width = masks.dim(3);
    
    // Create final tensor at original resolution
    Tensor final_masks({num_masks, 1, transform.original_height, transform.original_width}, 
                      DataType::FLOAT32);
    
    const float* src = masks.data<float>();
    float* dst = final_masks.data<float>();
    
    // Resize each mask to original size
    float x_scale = static_cast<float>(scaled_width) / transform.original_width;
    float y_scale = static_cast<float>(scaled_height) / transform.original_height;
    
    for (int n = 0; n < num_masks; ++n) {
        const float* mask_src = src + n * scaled_height * scaled_width;
        float* mask_dst = dst + n * transform.original_height * transform.original_width;
        
        for (int y = 0; y < transform.original_height; ++y) {
            float src_y = y * y_scale;
            int y0 = static_cast<int>(src_y);
            int y1 = std::min(y0 + 1, scaled_height - 1);
            float dy = src_y - y0;
            
            for (int x = 0; x < transform.original_width; ++x) {
                float src_x = x * x_scale;
                int x0 = static_cast<int>(src_x);
                int x1 = std::min(x0 + 1, scaled_width - 1);
                float dx = src_x - x0;
                
                // Bilinear interpolation
                float v00 = mask_src[y0 * scaled_width + x0];
                float v01 = mask_src[y0 * scaled_width + x1];
                float v10 = mask_src[y1 * scaled_width + x0];
                float v11 = mask_src[y1 * scaled_width + x1];
                
                float v0 = v00 * (1 - dx) + v01 * dx;
                float v1 = v10 * (1 - dx) + v11 * dx;
                float v = v0 * (1 - dy) + v1 * dy;
                
                mask_dst[y * transform.original_width + x] = v;
            }
        }
    }
    
    return final_masks;
}

std::vector<Mask> Postprocessor::extractMasks(const Tensor& masks,
                                              const Tensor& scores,
                                              const Options& options) const {
    std::vector<Mask> result_masks;
    
    int num_masks = masks.dim(0);
    int height = masks.dim(2);
    int width = masks.dim(3);
    
    const float* mask_data = masks.data<float>();
    const float* score_data = scores.data<float>();
    
    // Get indices sorted by score
    std::vector<int> indices = sortMasksByScore(
        std::vector<float>(score_data, score_data + num_masks)
    );
    
    // Determine how many masks to return
    int masks_to_return = options.multimask_output ? 
        std::min(options.max_masks, num_masks) : 1;
    
    for (int i = 0; i < masks_to_return; ++i) {
        int idx = indices[i];
        Mask mask;
        mask.width = width;
        mask.height = height;
        mask.score = score_data[idx];
        
        const float* mask_ptr = mask_data + idx * height * width;
        
        if (options.return_logits) {
            // Return raw logits
            mask.logits.resize(height * width);
            std::copy(mask_ptr, mask_ptr + height * width, mask.logits.begin());
        }
        
        // Always compute alpha mask
        mask.alpha.resize(height * width);
        
        if (options.mask_threshold > 0) {
            // Apply threshold
            for (int j = 0; j < height * width; ++j) {
                mask.alpha[j] = (mask_ptr[j] > options.mask_threshold) ? 255 : 0;
            }
        } else {
            // Convert logits to alpha using sigmoid
            for (int j = 0; j < height * width; ++j) {
                float sigmoid = 1.0f / (1.0f + std::exp(-mask_ptr[j]));
                mask.alpha[j] = static_cast<uint8_t>(sigmoid * 255.0f + 0.5f);
            }
        }
        
        result_masks.push_back(std::move(mask));
    }
    
    return result_masks;
}

Tensor Postprocessor::applyThreshold(const Tensor& logits, float threshold) const {
    Tensor binary = Tensor::zeros(logits.shape(), DataType::FLOAT32);
    
    const float* src = logits.data<float>();
    float* dst = binary.data<float>();
    
    for (size_t i = 0; i < logits.numel(); ++i) {
        dst[i] = (src[i] > threshold) ? 1.0f : 0.0f;
    }
    
    return binary;
}

std::vector<uint8_t> Postprocessor::tensorToAlpha(const Tensor& tensor) const {
    std::vector<uint8_t> alpha(tensor.numel());
    const float* data = tensor.data<float>();
    
    for (size_t i = 0; i < tensor.numel(); ++i) {
        // Apply sigmoid if needed, then convert to uint8
        float val = data[i];
        if (val < -10.0f || val > 10.0f) {
            // Likely logits, apply sigmoid
            val = 1.0f / (1.0f + std::exp(-val));
        }
        alpha[i] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, val * 255.0f)));
    }
    
    return alpha;
}

Tensor Postprocessor::refineMask(const Tensor& mask) const {
    // Simple mask refinement using morphological operations
    // This is a placeholder - actual implementation would use proper morphology
    return mask;
}

Tensor Postprocessor::bilinearUpsample(const Tensor& input, int scale_factor) const {
    if (input.rank() != 4) {
        throw std::invalid_argument("Expected 4D tensor for upsampling");
    }
    
    int batch = input.dim(0);
    int channels = input.dim(1);
    int height = input.dim(2);
    int width = input.dim(3);
    
    int new_height = height * scale_factor;
    int new_width = width * scale_factor;
    
    Tensor output({batch, channels, new_height, new_width}, DataType::FLOAT32);
    
    const float* src = input.data<float>();
    float* dst = output.data<float>();
    
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            const float* channel_src = src + (b * channels + c) * height * width;
            float* channel_dst = dst + (b * channels + c) * new_height * new_width;
            
            for (int y = 0; y < new_height; ++y) {
                float src_y = static_cast<float>(y) / scale_factor;
                int y0 = static_cast<int>(src_y);
                int y1 = std::min(y0 + 1, height - 1);
                float dy = src_y - y0;
                
                for (int x = 0; x < new_width; ++x) {
                    float src_x = static_cast<float>(x) / scale_factor;
                    int x0 = static_cast<int>(src_x);
                    int x1 = std::min(x0 + 1, width - 1);
                    float dx = src_x - x0;
                    
                    float v00 = channel_src[y0 * width + x0];
                    float v01 = channel_src[y0 * width + x1];
                    float v10 = channel_src[y1 * width + x0];
                    float v11 = channel_src[y1 * width + x1];
                    
                    float v0 = v00 * (1 - dx) + v01 * dx;
                    float v1 = v10 * (1 - dx) + v11 * dx;
                    float v = v0 * (1 - dy) + v1 * dy;
                    
                    channel_dst[y * new_width + x] = v;
                }
            }
        }
    }
    
    return output;
}

std::vector<int> Postprocessor::sortMasksByScore(const std::vector<float>& scores) const {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(),
              [&scores](int a, int b) { return scores[a] > scores[b]; });
    
    return indices;
}

} // namespace samkit