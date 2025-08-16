#include "samkit/image.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace samkit {

Image::Image() 
    : width_(0), height_(0), channels_(0), format_(ImageFormat::RGB), data_(nullptr) {
}

Image::Image(int width, int height, int channels, ImageFormat format)
    : width_(width), height_(height), channels_(channels), format_(format) {
    allocate();
}

Image::Image(const uint8_t* data, int width, int height, int channels, ImageFormat format)
    : width_(width), height_(height), channels_(channels), format_(format) {
    allocate();
    if (data) {
        copyFrom(data);
    }
}

Image::Image(const Image& other)
    : width_(other.width_), height_(other.height_), 
      channels_(other.channels_), format_(other.format_) {
    allocate();
    if (other.data_) {
        copyFrom(other.data_.get());
    }
}

Image::Image(Image&& other) noexcept
    : width_(other.width_), height_(other.height_),
      channels_(other.channels_), format_(other.format_),
      data_(std::move(other.data_)) {
    other.width_ = 0;
    other.height_ = 0;
    other.channels_ = 0;
}

Image& Image::operator=(const Image& other) {
    if (this != &other) {
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        format_ = other.format_;
        allocate();
        if (other.data_) {
            copyFrom(other.data_.get());
        }
    }
    return *this;
}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        format_ = other.format_;
        data_ = std::move(other.data_);
        
        other.width_ = 0;
        other.height_ = 0;
        other.channels_ = 0;
    }
    return *this;
}

Image::~Image() = default;

uint8_t& Image::at(int x, int y, int c) {
    if (x < 0 || x >= width_ || y < 0 || y >= height_ || c < 0 || c >= channels_) {
        throw std::out_of_range("Image pixel access out of range");
    }
    return data_[(y * width_ + x) * channels_ + c];
}

const uint8_t& Image::at(int x, int y, int c) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_ || c < 0 || c >= channels_) {
        throw std::out_of_range("Image pixel access out of range");
    }
    return data_[(y * width_ + x) * channels_ + c];
}

Image Image::resize(int new_width, int new_height) const {
    Image result(new_width, new_height, channels_, format_);
    ImageUtils::bilinearResize(data_.get(), width_, height_,
                               result.data_.get(), new_width, new_height,
                               channels_);
    return result;
}

Image Image::pad(int target_size, uint8_t pad_value) const {
    int pad_width = target_size;
    int pad_height = target_size;
    
    float scale = static_cast<float>(target_size) / std::max(width_, height_);
    int scaled_width = static_cast<int>(width_ * scale);
    int scaled_height = static_cast<int>(height_ * scale);
    
    int pad_left = (pad_width - scaled_width) / 2;
    int pad_top = (pad_height - scaled_height) / 2;
    
    // First resize
    Image resized = resize(scaled_width, scaled_height);
    
    // Then pad
    Image padded(pad_width, pad_height, channels_, format_);
    
    // Fill with pad value
    std::fill(padded.data_.get(), 
             padded.data_.get() + pad_width * pad_height * channels_,
             pad_value);
    
    // Copy resized image to center
    ImageUtils::padImage(resized.data_.get(), scaled_width, scaled_height,
                        padded.data_.get(), pad_width, pad_height,
                        pad_left, pad_top, channels_, pad_value);
    
    return padded;
}

Image Image::crop(const Box& box) const {
    int x0 = std::max(0, static_cast<int>(box.x0));
    int y0 = std::max(0, static_cast<int>(box.y0));
    int x1 = std::min(width_, static_cast<int>(box.x1));
    int y1 = std::min(height_, static_cast<int>(box.y1));
    
    int crop_width = x1 - x0;
    int crop_height = y1 - y0;
    
    if (crop_width <= 0 || crop_height <= 0) {
        return Image();
    }
    
    Image result(crop_width, crop_height, channels_, format_);
    
    for (int y = 0; y < crop_height; ++y) {
        for (int x = 0; x < crop_width; ++x) {
            for (int c = 0; c < channels_; ++c) {
                result.at(x, y, c) = at(x0 + x, y0 + y, c);
            }
        }
    }
    
    return result;
}

Image Image::convertTo(ImageFormat new_format) const {
    if (format_ == new_format) {
        return *this;
    }
    
    int new_channels = getChannelsForFormat(new_format);
    Image result(width_, height_, new_channels, new_format);
    
    int pixels = width_ * height_;
    
    if (format_ == ImageFormat::RGB && new_format == ImageFormat::BGR) {
        ImageUtils::rgbToBgr(data_.get(), result.data_.get(), pixels);
    } else if (format_ == ImageFormat::BGR && new_format == ImageFormat::RGB) {
        ImageUtils::bgrToRgb(data_.get(), result.data_.get(), pixels);
    } else if (format_ == ImageFormat::RGBA && new_format == ImageFormat::RGB) {
        ImageUtils::rgbaToRgb(data_.get(), result.data_.get(), pixels);
    } else if (format_ == ImageFormat::GRAY && new_format == ImageFormat::RGB) {
        ImageUtils::grayToRgb(data_.get(), result.data_.get(), pixels);
    } else {
        // Default: copy what we can
        int min_channels = std::min(channels_, new_channels);
        for (int i = 0; i < pixels; ++i) {
            for (int c = 0; c < min_channels; ++c) {
                result.data_[i * new_channels + c] = data_[i * channels_ + c];
            }
        }
    }
    
    return result;
}

Image Image::fromRGB(const uint8_t* data, int width, int height) {
    return Image(data, width, height, 3, ImageFormat::RGB);
}

Image Image::fromRGBA(const uint8_t* data, int width, int height) {
    return Image(data, width, height, 4, ImageFormat::RGBA);
}

Image Image::fromBGR(const uint8_t* data, int width, int height) {
    return Image(data, width, height, 3, ImageFormat::BGR);
}

Image Image::fromGray(const uint8_t* data, int width, int height) {
    return Image(data, width, height, 1, ImageFormat::GRAY);
}

void Image::allocate() {
    size_t total_size = size();
    if (total_size > 0) {
        data_ = std::make_unique<uint8_t[]>(total_size);
    }
}

void Image::copyFrom(const uint8_t* src) {
    if (src && data_) {
        std::memcpy(data_.get(), src, size());
    }
}

int Image::getChannelsForFormat(ImageFormat fmt) const {
    switch (fmt) {
        case ImageFormat::RGB:
        case ImageFormat::BGR:
            return 3;
        case ImageFormat::RGBA:
        case ImageFormat::BGRA:
            return 4;
        case ImageFormat::GRAY:
            return 1;
        default:
            return 3;
    }
}

// ImageUtils implementation

void ImageUtils::bilinearResize(const uint8_t* src, int src_width, int src_height,
                                uint8_t* dst, int dst_width, int dst_height,
                                int channels) {
    float x_scale = static_cast<float>(src_width) / dst_width;
    float y_scale = static_cast<float>(src_height) / dst_height;
    
    for (int y = 0; y < dst_height; ++y) {
        float src_y = y * y_scale;
        int y0 = static_cast<int>(src_y);
        int y1 = std::min(y0 + 1, src_height - 1);
        float dy = src_y - y0;
        
        for (int x = 0; x < dst_width; ++x) {
            float src_x = x * x_scale;
            int x0 = static_cast<int>(src_x);
            int x1 = std::min(x0 + 1, src_width - 1);
            float dx = src_x - x0;
            
            for (int c = 0; c < channels; ++c) {
                float v00 = src[(y0 * src_width + x0) * channels + c];
                float v01 = src[(y0 * src_width + x1) * channels + c];
                float v10 = src[(y1 * src_width + x0) * channels + c];
                float v11 = src[(y1 * src_width + x1) * channels + c];
                
                float v0 = v00 * (1 - dx) + v01 * dx;
                float v1 = v10 * (1 - dx) + v11 * dx;
                float v = v0 * (1 - dy) + v1 * dy;
                
                dst[(y * dst_width + x) * channels + c] = static_cast<uint8_t>(v + 0.5f);
            }
        }
    }
}

void ImageUtils::rgbToBgr(const uint8_t* src, uint8_t* dst, int pixels) {
    for (int i = 0; i < pixels; ++i) {
        dst[i * 3 + 0] = src[i * 3 + 2];  // B = R
        dst[i * 3 + 1] = src[i * 3 + 1];  // G = G
        dst[i * 3 + 2] = src[i * 3 + 0];  // R = B
    }
}

void ImageUtils::bgrToRgb(const uint8_t* src, uint8_t* dst, int pixels) {
    rgbToBgr(src, dst, pixels);  // Same operation
}

void ImageUtils::rgbaToRgb(const uint8_t* src, uint8_t* dst, int pixels) {
    for (int i = 0; i < pixels; ++i) {
        dst[i * 3 + 0] = src[i * 4 + 0];
        dst[i * 3 + 1] = src[i * 4 + 1];
        dst[i * 3 + 2] = src[i * 4 + 2];
    }
}

void ImageUtils::grayToRgb(const uint8_t* src, uint8_t* dst, int pixels) {
    for (int i = 0; i < pixels; ++i) {
        uint8_t gray = src[i];
        dst[i * 3 + 0] = gray;
        dst[i * 3 + 1] = gray;
        dst[i * 3 + 2] = gray;
    }
}

void ImageUtils::padImage(const uint8_t* src, int src_width, int src_height,
                         uint8_t* dst, int dst_width, int dst_height,
                         int pad_left, int pad_top, int channels, uint8_t pad_value) {
    // Copy src image to padded location in dst
    for (int y = 0; y < src_height; ++y) {
        for (int x = 0; x < src_width; ++x) {
            int src_idx = (y * src_width + x) * channels;
            int dst_idx = ((y + pad_top) * dst_width + (x + pad_left)) * channels;
            for (int c = 0; c < channels; ++c) {
                dst[dst_idx + c] = src[src_idx + c];
            }
        }
    }
}

} // namespace samkit