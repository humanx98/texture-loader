#include <hip/hip_runtime.h>
#include "ImageSource/OIIOReader.h"
#include "../DemandLoading/Internal/ImageData.h"
#include <OpenImageIO/imageio.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace hip_demand {
namespace {

hipArray_Format imageFormat(OIIO::TypeDesc type) {
    switch (type.basetype) {
    case OIIO::TypeDesc::UINT8: return HIP_AD_FORMAT_UNSIGNED_INT8;
    case OIIO::TypeDesc::INT8: return HIP_AD_FORMAT_SIGNED_INT8;
    case OIIO::TypeDesc::UINT16: return HIP_AD_FORMAT_UNSIGNED_INT16;
    case OIIO::TypeDesc::INT16: return HIP_AD_FORMAT_SIGNED_INT16;
    case OIIO::TypeDesc::UINT32: return HIP_AD_FORMAT_UNSIGNED_INT32;
    case OIIO::TypeDesc::INT32: return HIP_AD_FORMAT_SIGNED_INT32;
    case OIIO::TypeDesc::HALF: return HIP_AD_FORMAT_HALF;
    default: return HIP_AD_FORMAT_FLOAT;
    }
}

OIIO::TypeDesc pixelType(hipArray_Format format) {
    switch (format) {
    case HIP_AD_FORMAT_UNSIGNED_INT8: return OIIO::TypeDesc::UINT8;
    case HIP_AD_FORMAT_SIGNED_INT8: return OIIO::TypeDesc::INT8;
    case HIP_AD_FORMAT_UNSIGNED_INT16: return OIIO::TypeDesc::UINT16;
    case HIP_AD_FORMAT_SIGNED_INT16: return OIIO::TypeDesc::INT16;
    case HIP_AD_FORMAT_UNSIGNED_INT32: return OIIO::TypeDesc::UINT32;
    case HIP_AD_FORMAT_SIGNED_INT32: return OIIO::TypeDesc::INT32;
    case HIP_AD_FORMAT_HALF: return OIIO::TypeDesc::HALF;
    case HIP_AD_FORMAT_FLOAT: return OIIO::TypeDesc::FLOAT;
    default: throw std::invalid_argument("Unsupported OIIO image channel format");
    }
}

size_t levelBytes(const TextureInfo& info, unsigned int width, unsigned int height) {
    return internal::imageByteSize(width, height, info.numChannels,
                                   getBytesPerChannel(info.format));
}

} // namespace

OIIOReader::OIIOReader(const std::string& filename) : filename_(filename) {}
OIIOReader::~OIIOReader() { close(); }

void OIIOReader::open(TextureInfo* info) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (isOpen_) {
        if (info) *info = info_;
        return;
    }
    const auto start = std::chrono::high_resolution_clock::now();
    auto input = OIIO::ImageInput::open(filename_);
    if (!input) throw std::runtime_error("Failed to open image: " + filename_);
    const OIIO::ImageSpec& spec = input->spec();
    if (spec.width <= 0 || spec.height <= 0 || spec.depth != 1 ||
        spec.nchannels < 1 || spec.nchannels > 4)
        throw std::runtime_error("Unsupported OIIO texture dimensions or channels: " + filename_);

    TextureInfo opened{};
    opened.width = static_cast<unsigned int>(spec.width);
    opened.height = static_cast<unsigned int>(spec.height);
    opened.numChannels = static_cast<unsigned int>(spec.nchannels);
    opened.format = imageFormat(spec.format);
    opened.isTiled = spec.tile_width > 0;
    opened.numMipLevels = 1;
    const unsigned int fullLevels = calculateNumMipLevels(opened.width, opened.height);
    for (unsigned int level = 1; level < fullLevels; ++level) {
        OIIO::ImageSpec mipSpec;
        if (!input->seek_subimage(0, static_cast<int>(level), mipSpec)) break;
        if (mipSpec.width != static_cast<int>(std::max(1u, opened.width >> level)) ||
            mipSpec.height != static_cast<int>(std::max(1u, opened.height >> level)) ||
            mipSpec.depth != 1 || mipSpec.nchannels != static_cast<int>(opened.numChannels))
            // HIP arrays halve dimensions with rounding down. A valid EXR can round up;
            // expose its compatible prefix and let the loader generate the remaining levels.
            break;
        ++opened.numMipLevels;
    }
    opened.isValid = true;
    levelBytes(opened, opened.width, opened.height);
    input->close();
    info_ = opened;
    isOpen_ = true;
    if (info) *info = info_;
    totalReadTime_ += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();
}

void OIIOReader::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    mipLevels_.clear();
    isOpen_ = false;
}

bool OIIOReader::isOpen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return isOpen_;
}

const TextureInfo& OIIOReader::getInfo() const { return info_; }

bool OIIOReader::loadImage() {
    if (!isOpen_) return false;
    const auto start = std::chrono::high_resolution_clock::now();
    auto input = OIIO::ImageInput::open(filename_);
    if (!input) return false;
    const OIIO::TypeDesc type = pixelType(info_.format);
    std::vector<std::vector<unsigned char>> levels(info_.numMipLevels);
    unsigned long long bytesRead = 0;
    unsigned int width = info_.width;
    unsigned int height = info_.height;
    for (unsigned int level = 0; level < info_.numMipLevels; ++level) {
        OIIO::ImageSpec spec;
        if (input->seek_subimage(0, static_cast<int>(level), spec)) {
            if (spec.width != static_cast<int>(width) || spec.height != static_cast<int>(height) ||
                spec.depth != 1 || spec.nchannels != static_cast<int>(info_.numChannels))
                return false;
            levels[level].resize(levelBytes(info_, width, height));
            if (!input->read_image(0, static_cast<int>(level), 0, spec.nchannels,
                                   type, levels[level].data()))
                return false;
            bytesRead += levels[level].size();
        } else {
            return false;
        }
        width = std::max(1u, width / 2);
        height = std::max(1u, height / 2);
    }
    input->close();
    mipLevels_.swap(levels);
    bytesRead_ += bytesRead;
    totalReadTime_ += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start).count();
    return true;
}

bool OIIOReader::readMipLevel(char* dest, unsigned int level, unsigned int expectedWidth,
                              unsigned int expectedHeight, hipStream_t /*stream*/) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!dest || !isOpen_ || level >= info_.numMipLevels) return false;
    const unsigned int width = std::max(1u, info_.width >> level);
    const unsigned int height = std::max(1u, info_.height >> level);
    if (width != expectedWidth || height != expectedHeight) return false;
    if (mipLevels_.empty() && !loadImage()) return false;
    const size_t size = levelBytes(info_, width, height);
    if (mipLevels_[level].size() != size) return false;
    std::memcpy(dest, mipLevels_[level].data(), size);
    return true;
}

bool OIIOReader::readBaseColor(float4& dest) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!isOpen_ || std::max(1u, info_.width >> (info_.numMipLevels - 1)) != 1 ||
        std::max(1u, info_.height >> (info_.numMipLevels - 1)) != 1)
        return false;
    if (mipLevels_.empty() && !loadImage()) return false;
    TextureInfo levelInfo = info_;
    levelInfo.width = levelInfo.height = 1;
    const internal::ImageData pixel = internal::decodeImagePixels(mipLevels_.back().data(), levelInfo);
    if (pixel.isFloat())
        dest = make_float4(pixel.floats[0], pixel.floats[1], pixel.floats[2], pixel.floats[3]);
    else
        dest = make_float4(pixel.bytes[0] / 255.0f, pixel.bytes[1] / 255.0f,
                          pixel.bytes[2] / 255.0f, pixel.bytes[3] / 255.0f);
    return true;
}

unsigned long long OIIOReader::getNumBytesRead() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytesRead_;
}

double OIIOReader::getTotalReadTime() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totalReadTime_;
}

unsigned long long OIIOReader::getHash(hipStream_t /*stream*/) const {
    return static_cast<unsigned long long>(std::hash<std::string>{}(filename_));
}

std::unique_ptr<ImageSource> createImageSource(const std::string& filename) {
    return std::make_unique<OIIOReader>(filename);
}

} // namespace hip_demand
