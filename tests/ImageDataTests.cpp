// SPDX-License-Identifier: MIT
// These tests exercise image decoding without creating a HIP device or stream.
#include <gtest/gtest.h>
#include "ImageDataTestUtils.h"

#include <array>
#include <tuple>

namespace hip_demand {
namespace test {
namespace {

class ImageDataFormatTest : public ::testing::TestWithParam<std::tuple<unsigned int, unsigned int>> {};

TEST_P(ImageDataFormatTest, NativeAllocationDecodedValuesAndUploadLayout) {
    const auto [formatIndex, channels] = GetParam();
    auto source = makeSource(formatIndex, channels);
    internal::ImageData image;
    ASSERT_TRUE(internal::readImageSource(source, image));
    EXPECT_EQ(source.reads, 1u);
    EXPECT_EQ(source.baseColorReads, 0u);
    EXPECT_EQ(image.width, 3u);
    EXPECT_EQ(image.height, 2u);
    const bool floating = formats[formatIndex].format != HIP_AD_FORMAT_UNSIGNED_INT8;
    EXPECT_EQ(image.isFloat(), floating);
    EXPECT_EQ(image.rowBytes(), 3u * 4 * (floating ? 4 : 1));
    EXPECT_EQ(image.sizeBytes(), source.info.width * source.info.height * 4u * (floating ? 4 : 1));
    const auto desc = image.channelDesc();
    EXPECT_EQ(desc.x, floating ? 32 : 8);
    EXPECT_EQ(desc.y, desc.x);
    EXPECT_EQ(desc.z, desc.x);
    EXPECT_EQ(desc.w, desc.x);
    EXPECT_EQ(desc.f, floating ? hipChannelFormatKindFloat : hipChannelFormatKindUnsigned);
    EXPECT_EQ(image.readMode(), floating ? hipReadModeElementType : hipReadModeNormalizedFloat);
    for (size_t i = 0; i < 6; ++i) {
        for (size_t c = 0; c < 4; ++c) {
            const float expected = c < channels ? formats[formatIndex].expected[(i + c) % 4]
                                                : (c == 3 ? 1.f : 0.f);
            const float actual = floating ? image.floats[i * 4 + c] : image.bytes[i * 4 + c] / 255.f;
            EXPECT_FLOAT_EQ(actual, expected) << "pixel=" << i << " channel=" << c;
        }
    }
    if (channels == 4 && (formatIndex == 0 || formatIndex == 7))
        EXPECT_EQ(source.destination, image.data()) << "Packed RGBA should decode directly into upload storage";
}

INSTANTIATE_TEST_SUITE_P(AllFormatsAndChannels, ImageDataFormatTest,
                         ::testing::Combine(::testing::Range(0u, 8u), ::testing::Range(1u, 5u)));

TEST(ImageDataTest, HalfSpecialValues) {
    EXPECT_EQ(internal::halfToFloat(0x7c00), std::numeric_limits<float>::infinity());
    EXPECT_EQ(internal::halfToFloat(0xfc00), -std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::isnan(internal::halfToFloat(0x7e01)));
    EXPECT_TRUE(std::signbit(internal::halfToFloat(0x8000)));
    EXPECT_FLOAT_EQ(internal::halfToFloat(0x03ff), 1023.f * 0x1p-24f);
    EXPECT_FLOAT_EQ(internal::halfToFloat(0x0400), 0x1p-14f);
}

TEST(ImageDataTest, SizeValidationPrecedesRead) {
    EXPECT_EQ(internal::imageByteSize(65536, 65536, 4, 1), size_t{17179869184ull});
    EXPECT_THROW(internal::imageByteSize(std::numeric_limits<int>::max(),
                                          std::numeric_limits<int>::max(), 4, 4), std::overflow_error);
    EXPECT_THROW(internal::imageByteSize(0, 1, 4, 4), std::invalid_argument);
    EXPECT_THROW(internal::imageByteSize(1, 0, 4, 4), std::invalid_argument);
    EXPECT_THROW(internal::imageByteSize(1, 1, 5, 4), std::invalid_argument);
    EXPECT_THROW(internal::imageByteSize(1, 1, 4, 0), std::invalid_argument);
    auto source = makeSource(7, 4);
    source.info.width = std::numeric_limits<int>::max();
    source.info.height = std::numeric_limits<int>::max();
    internal::ImageData image;
    EXPECT_THROW(internal::readImageSource(source, image), std::overflow_error);
    EXPECT_EQ(source.reads, 0u);
    source.info.width = source.info.height = 1;
    source.info.format = static_cast<hipArray_Format>(0);
    EXPECT_THROW(internal::readImageSource(source, image), std::invalid_argument);
    EXPECT_EQ(source.reads, 0u);
}

TEST(ImageDataTest, ReadFailureIsReported) {
    auto source = makeSource(7, 4);
    source.failRead = true;
    internal::ImageData image;
    EXPECT_FALSE(internal::readImageSource(source, image));
}

TEST(ImageDataTest, FloatMipPreservesHDRAndRectangularAccounting) {
    internal::ImageData image;
    image.reset(4, 1, true);
    image.floats = {-2, 2, 6, .5f, 2, 6, 10, .5f, 6, 10, 14, .5f, 10, 14, 18, .5f};
    auto next = internal::downsampleImage(image);
    EXPECT_EQ(next.width, 2u);
    EXPECT_EQ(next.height, 1u);
    EXPECT_FLOAT_EQ(next.floats[0], 0);
    EXPECT_FLOAT_EQ(next.floats[2], 8);
    EXPECT_FLOAT_EQ(next.floats[3], .5f);
    EXPECT_FLOAT_EQ(next.floats[4], 8);
    EXPECT_EQ(internal::mipImageByteSize(image, 3), (4u + 2u + 1u) * 16);
    EXPECT_EQ(internal::mipImageByteSize(image, 2), (4u + 2u) * 16);
}

TEST(ImageDataTest, ByteMipPreservesIntegerFiltering) {
    internal::ImageData image;
    image.reset(2, 2, false);
    image.bytes = {1, 3, 5, 255, 2, 4, 6, 255, 3, 5, 7, 255, 4, 6, 8, 255};
    auto next = internal::downsampleImage(image);
    EXPECT_EQ(next.bytes, (std::vector<unsigned char>{2, 4, 6, 255}));
    EXPECT_EQ(internal::mipImageByteSize(image, 2), 20u);
}

TEST(ImageDataTest, FloatSRGBPreservesAlphaAndHDR) {
    internal::ImageData image;
    image.reset(1, 1, true);
    image.floats = {0.04045f, .5f, 2.f, .25f};
    internal::linearizeFloatSRGB(image);
    EXPECT_FLOAT_EQ(image.floats[0], .04045f / 12.92f);
    EXPECT_NEAR(image.floats[1], .21404114f, 1e-6f);
    EXPECT_GT(image.floats[2], 1.f);
    EXPECT_FLOAT_EQ(image.floats[3], .25f);
}

TEST(ImageDataTest, OddMipFootprintIncludesLastRowAndColumn) {
    for (bool vertical : {false, true}) {
        internal::ImageData image;
        image.reset(vertical ? 1 : 3, vertical ? 3 : 1, true);
        image.floats = {0, 0, 0, 1, 0, 0, 0, 1, 99, 99, 99, 1};
        auto next = internal::downsampleImage(image);
        EXPECT_EQ(next.floats, (std::vector<float>{33, 33, 33, 1}));
    }
    internal::ImageData image;
    image.reset(5, 3, true);
    for (size_t i = 0; i < 15; ++i) {
        const float value = static_cast<float>(i % 5);
        image.floats[i * 4] = value;
        image.floats[i * 4 + 1] = value;
        image.floats[i * 4 + 2] = value;
        image.floats[i * 4 + 3] = 1;
    }
    auto next = internal::downsampleImage(image);
    EXPECT_FLOAT_EQ(next.floats[0], .8f);
    EXPECT_FLOAT_EQ(next.floats[4], 3.2f);
    EXPECT_FLOAT_EQ(internal::downsampleImage(next).floats[0], 2.f);
    image.reset(5, 3, false);
    std::fill(image.bytes.begin(), image.bytes.end(), 255);
    next = internal::downsampleImage(image);
    for (unsigned char value : next.bytes) EXPECT_EQ(value, 255);
}

TEST(ImageDataTest, SuppliedMipPixelsAreReadInsteadOfRegenerated) {
    auto source = makeSource(7, 4);
    source.info.width = source.info.height = 2;
    source.info.numMipLevels = 2;
    source.mipPixels = {std::vector<unsigned char>(4 * 16), packed<float>({7, -2, .25f, .5f})};
    internal::ImageData image;
    ASSERT_TRUE(internal::readImageSource(source, image, 1));
    EXPECT_EQ(image.width, 1u);
    EXPECT_EQ(image.height, 1u);
    EXPECT_EQ(image.floats, (std::vector<float>{7, -2, .25f, .5f}));
    EXPECT_EQ(source.readLevels, (std::vector<unsigned int>{1}));
    EXPECT_EQ(source.baseColorReads, 0u);
    EXPECT_FALSE(internal::readImageSource(source, image, 2));
}

TEST(ImageDataTest, GeneratedByteSRGBMipFiltersInLinearSpace) {
    internal::ImageData image;
    image.reset(2, 1, false);
    image.bytes = {0, 0, 0, 128, 255, 255, 255, 0};
    auto next = internal::downsampleImage(image, true);
    EXPECT_EQ(next.bytes, (std::vector<unsigned char>{188, 188, 188, 64}));
    EXPECT_NEAR(internal::srgbToLinear(next.bytes[0] / 255.f), .5f, .004f);
    image.reset(2, 1, true);
    image.floats = {0, 0, 0, 1, 1, 1, 1, 1};
    next = internal::downsampleImage(image, true);
    EXPECT_FLOAT_EQ(next.floats[0], .5f);
}

} // namespace
} // namespace test
} // namespace hip_demand
