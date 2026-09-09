// SPDX-License-Identifier: MIT
#pragma once
#include <gtest/gtest.h>
#include "DemandLoading/Internal/ImageData.h"
#include <array>

namespace hip_demand { namespace test {
class TypedImageSource : public ImageSource {
public:
    TextureInfo info;
    std::vector<unsigned char> pixels;
    std::vector<std::vector<unsigned char>> mipPixels;
    std::vector<unsigned int> readLevels;
    void* destination = nullptr;
    bool opened = false;
    bool failRead = false;
    unsigned int reads = 0;
    unsigned int baseColorReads = 0;

    void open(TextureInfo* result) override { opened = true; if (result) *result = info; }
    void close() override { opened = false; }
    bool isOpen() const override { return opened; }
    const TextureInfo& getInfo() const override { return info; }
    bool readMipLevel(char* dest, unsigned int level, unsigned int width,
                        unsigned int height, hipStream_t) override {
        ++reads;
        readLevels.push_back(level);
        destination = dest;
        EXPECT_EQ(width, std::max(1u, info.width >> level));
        EXPECT_EQ(height, std::max(1u, info.height >> level));
        if (failRead) return false;
        if (mipPixels.empty()) {
            EXPECT_EQ(level, 0u);
            std::memcpy(dest, pixels.data(), pixels.size());
        } else {
            if (level >= mipPixels.size()) return false;
            std::memcpy(dest, mipPixels[level].data(), mipPixels[level].size());
        }
        return true;
    }
    bool readBaseColor(float4&) override { ++baseColorReads; return false; }
    unsigned long long getNumBytesRead() const override { return reads * pixels.size(); }
    double getTotalReadTime() const override { return 0.0; }
};

template<class T> std::vector<unsigned char> packed(std::array<T, 4> values) {
    std::vector<unsigned char> bytes(sizeof(values));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

struct FormatCase {
    hipArray_Format format;
    std::vector<unsigned char> values;
    std::array<float, 4> expected;
};

const std::array<FormatCase, 8> formats = {{
    {HIP_AD_FORMAT_UNSIGNED_INT8, packed<uint8_t>({0, 128, 255, 64}), {0, 128.f/255, 1, 64.f/255}},
    {HIP_AD_FORMAT_SIGNED_INT8, packed<int8_t>({-128, -64, 127, 0}), {-1, -64.f/127, 1, 0}},
    {HIP_AD_FORMAT_UNSIGNED_INT16, packed<uint16_t>({0, 32768, 65535, 1}), {0, 32768.f/65535, 1, 1.f/65535}},
    {HIP_AD_FORMAT_SIGNED_INT16, packed<int16_t>({-32768, -16384, 32767, 0}), {-1, -16384.f/32767, 1, 0}},
    {HIP_AD_FORMAT_UNSIGNED_INT32, packed<uint32_t>({0, 2147483648u, 4294967295u, 1}), {0, .5f, 1, 1.f/4294967295.0f}},
    {HIP_AD_FORMAT_SIGNED_INT32, packed<int32_t>({std::numeric_limits<int32_t>::min(), -1073741824, 2147483647, 0}), {-1, -.5f, 1, 0}},
    {HIP_AD_FORMAT_HALF, packed<uint16_t>({0xc000, 0x3800, 0x4200, 0x0001}), {-2, .5f, 3, 0x1p-24f}},
    {HIP_AD_FORMAT_FLOAT, packed<float>({-.25f, 2.5f, .5f, 1}), {-.25f, 2.5f, .5f, 1}}
}};

inline TypedImageSource makeSource(unsigned int formatIndex, unsigned int channels) {
    TypedImageSource source;
    const auto& format = formats.at(formatIndex);
    source.info.width = 3;
    source.info.height = 2;
    source.info.numChannels = channels;
    source.info.numMipLevels = 1;
    source.info.format = format.format;
    source.info.isValid = true;
    const size_t channelBytes = getBytesPerChannel(format.format);
    source.pixels.resize(3 * 2 * channels * channelBytes);
    for (size_t i = 0; i < 6; ++i)
        for (size_t c = 0; c < channels; ++c)
            std::memcpy(source.pixels.data() + (i * channels + c) * channelBytes,
                           format.values.data() + ((i + c) % 4) * channelBytes, channelBytes);
    return source;
}


} }
