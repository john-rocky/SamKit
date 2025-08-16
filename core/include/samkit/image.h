#ifndef SAMKIT_IMAGE_H
#define SAMKIT_IMAGE_H

#include "samkit/types.h"
#include <memory>
#include <cstring>

namespace samkit {

class Image {
public:
    // Constructors
    Image();
    Image(int width, int height, int channels, ImageFormat format = ImageFormat::RGB);
    Image(const uint8_t* data, int width, int height, int channels, 
          ImageFormat format = ImageFormat::RGB);
    
    // Copy and move constructors
    Image(const Image& other);
    Image(Image&& other) noexcept;
    Image& operator=(const Image& other);
    Image& operator=(Image&& other) noexcept;
    
    ~Image();
    
    // Getters
    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }
    ImageFormat format() const { return format_; }
    size_t size() const { return width_ * height_ * channels_; }
    uint8_t* data() { return data_.get(); }
    const uint8_t* data() const { return data_.get(); }
    
    // Pixel access
    uint8_t& at(int x, int y, int c);
    const uint8_t& at(int x, int y, int c) const;
    
    // Image operations
    Image resize(int new_width, int new_height) const;
    Image pad(int target_size, uint8_t pad_value = 0) const;
    Image crop(const Box& box) const;
    Image convertTo(ImageFormat new_format) const;
    
    // Create from common formats
    static Image fromRGB(const uint8_t* data, int width, int height);
    static Image fromRGBA(const uint8_t* data, int width, int height);
    static Image fromBGR(const uint8_t* data, int width, int height);
    static Image fromGray(const uint8_t* data, int width, int height);
    
    // Validation
    bool isValid() const { return data_ != nullptr && width_ > 0 && height_ > 0; }
    
private:
    int width_;
    int height_;
    int channels_;
    ImageFormat format_;
    std::unique_ptr<uint8_t[]> data_;
    
    // Helper methods
    void allocate();
    void copyFrom(const uint8_t* src);
    int getChannelsForFormat(ImageFormat fmt) const;
};

// Image utilities
namespace ImageUtils {
    
    // Bilinear interpolation for resize
    void bilinearResize(const uint8_t* src, int src_width, int src_height,
                       uint8_t* dst, int dst_width, int dst_height, 
                       int channels);
    
    // Convert between color formats
    void rgbToBgr(const uint8_t* src, uint8_t* dst, int pixels);
    void rgbaToRgb(const uint8_t* src, uint8_t* dst, int pixels);
    void bgrToRgb(const uint8_t* src, uint8_t* dst, int pixels);
    void grayToRgb(const uint8_t* src, uint8_t* dst, int pixels);
    
    // Padding utilities
    void padImage(const uint8_t* src, int src_width, int src_height,
                 uint8_t* dst, int dst_width, int dst_height,
                 int pad_left, int pad_top, int channels, uint8_t pad_value);
    
} // namespace ImageUtils

} // namespace samkit

#endif // SAMKIT_IMAGE_H