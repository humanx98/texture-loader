// SPDX-License-Identifier: MIT
#include "TestUtils.h"
#include "ImageDataTestUtils.h"
#include <DemandLoading/DeviceContext.h>
#include <chrono>
#include <fstream>

namespace hip_demand {
namespace test {
namespace {

void loadRequestedTexture(DemandTextureLoader& loader, const TextureHandle& handle,
                            hipTextureObject_t& texture) {
    ASSERT_TRUE(handle.valid);
    auto context = loader.getDeviceContext();
    const uint32_t count = 1;
    ASSERT_EQ(hipMemcpy(context.requests, &handle.id, sizeof(handle.id), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(context.requestCount, &count, sizeof(count), hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(loader.processRequests(nullptr, context), 1u);
    loader.launchPrepare(nullptr);
    ASSERT_EQ(hipMemcpy(&texture, context.textures + handle.id, sizeof(texture), hipMemcpyDeviceToHost), hipSuccess);
    ASSERT_NE(texture, hipTextureObject_t{});
}

TEST_F(LoaderTestFixture, ImageSourceFormatsUseCorrectGpuStorage) {
    size_t expectedMemory = 0;
    for (unsigned int formatIndex = 0; formatIndex < formats.size(); ++formatIndex) {
        for (unsigned int channels = 1; channels <= 4; ++channels) {
            SCOPED_TRACE(::testing::Message() << "format=" << formatIndex << " channels=" << channels);
            auto source = std::make_shared<TypedImageSource>(makeSource(formatIndex, channels));
            TextureDesc desc;
            desc.generateMipmaps = false;
            desc.filterMode = hipFilterModePoint;
            const auto handle = loader_->createTexture(source, desc);
            hipTextureObject_t texture = 0;
            loadRequestedTexture(*loader_, handle, texture);
            ASSERT_NE(texture, hipTextureObject_t{});

            hipResourceDesc resource{};
            hipTextureDesc sampler{};
            ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
            ASSERT_EQ(hipGetTextureObjectTextureDesc(&sampler, texture), hipSuccess);
            ASSERT_EQ(resource.resType, hipResourceTypeArray);
            hipChannelFormatDesc channelDesc{};
            ASSERT_EQ(hipGetChannelDesc(&channelDesc, resource.res.array.array), hipSuccess);
            const bool floating = formatIndex != 0;
            EXPECT_EQ(channelDesc.x, floating ? 32 : 8);
            EXPECT_EQ(channelDesc.y, channelDesc.x);
            EXPECT_EQ(channelDesc.z, channelDesc.x);
            EXPECT_EQ(channelDesc.w, channelDesc.x);
            EXPECT_EQ(channelDesc.f, floating ? hipChannelFormatKindFloat : hipChannelFormatKindUnsigned);
            EXPECT_EQ(sampler.readMode, floating ? hipReadModeElementType : hipReadModeNormalizedFloat);

            const size_t rowBytes = 3 * 4 * (floating ? sizeof(float) : 1);
            std::vector<unsigned char> pixels(rowBytes * 2);
            ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), rowBytes, resource.res.array.array,
                                              0, 0, rowBytes, 2, hipMemcpyDeviceToHost), hipSuccess);
            for (size_t i = 0; i < 6; ++i) {
                for (size_t c = 0; c < 4; ++c) {
                    const float expected = c < channels ? formats[formatIndex].expected[(i + c) % 4]
                                                        : (c == 3 ? 1.f : 0.f);
                    const float actual = floating ? internal::readChannel<float>(pixels.data() + (i * 4 + c) * 4)
                                                  : pixels[i * 4 + c] / 255.f;
                    EXPECT_FLOAT_EQ(actual, expected) << "pixel=" << i << " channel=" << c;
                }
            }
            expectedMemory += pixels.size();
            EXPECT_EQ(loader_->getTotalTextureMemory(), expectedMemory);
        }
    }
    EXPECT_EQ(loader_->getResidentTextureCount(), 32u);
}

TEST_F(LoaderTestFixture, FloatMipUploadPreservesHdrValues) {
    auto source = std::make_shared<TypedImageSource>(makeSource(7, 4));
    source->info.width = 4;
    source->info.height = 1;
    const std::vector<float> colors = {-2, 2, 6, .5f, 2, 6, 10, .5f, 6, 10, 14, .5f, 10, 14, 18, .5f};
    source->pixels.resize(colors.size() * sizeof(float));
    std::memcpy(source->pixels.data(), colors.data(), source->pixels.size());
    TextureDesc desc;
    desc.generateMipmaps = true;
    auto handle = loader_->createTexture(source, desc);
    hipTextureObject_t texture = 0;
    loadRequestedTexture(*loader_, handle, texture);
    ASSERT_NE(texture, hipTextureObject_t{});
    hipResourceDesc resource{};
    ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
    if (resource.resType == hipResourceTypeArray) {
        // Keep coverage of the supported non-mip fallback on older devices.
        EXPECT_EQ(loader_->getTotalTextureMemory(), 4u * 16);
        std::vector<float> pixels(colors.size());
        ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), 4 * 16, resource.res.array.array,
                                          0, 0, 4 * 16, 1, hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(pixels, colors);
    } else {
        ASSERT_EQ(resource.resType, hipResourceTypeMipmappedArray);
        hipArray_t level{};
        ASSERT_EQ(hipGetMipmappedArrayLevel(&level, resource.res.mipmap.mipmap, 1), hipSuccess);
        std::vector<float> pixels(8);
        ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), 2 * 16, level, 0, 0, 2 * 16, 1,
                                          hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(pixels, (std::vector<float>{0, 4, 8, .5f, 8, 12, 16, .5f}));
        EXPECT_EQ(loader_->getTotalTextureMemory(), (4u + 2u + 1u) * 16);
    }
}

TEST_F(LoaderTestFixture, NativeChannelMemoryTextureCanReloadAfterUnload) {
    for (unsigned int channels = 1; channels <= 3; ++channels) {
        SCOPED_TRACE(channels);
        auto source = makeSource(0, channels);
        TextureDesc desc;
        desc.generateMipmaps = false;
        auto handle = loader_->createTextureFromMemory(source.pixels.data(), 3, 2, channels, desc);
        for (unsigned int pass = 0; pass < 2; ++pass) {
            hipTextureObject_t texture{};
            loadRequestedTexture(*loader_, handle, texture);
            hipResourceDesc resource{};
            ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
            ASSERT_EQ(resource.resType, hipResourceTypeArray);
            std::vector<unsigned char> pixels(6 * 4);
            ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), 3 * 4, resource.res.array.array,
                                              0, 0, 3 * 4, 2, hipMemcpyDeviceToHost), hipSuccess);
            for (size_t i = 0; i < 6; ++i)
                for (size_t c = 0; c < 4; ++c)
                    EXPECT_EQ(pixels[i * 4 + c], c < channels ? source.pixels[i * channels + c]
                                                             : (c == 3 ? 255 : 0));
            loader_->unloadTexture(handle.id);
            EXPECT_EQ(loader_->getTotalTextureMemory(), 0u);
        }
    }
}

TEST_F(LoaderTestFixture, SuppliedMipLevelsSurviveUploadAndLargeMipLimit) {
    auto source = std::make_shared<TypedImageSource>(makeSource(7, 4));
    source->info.width = source->info.height = 4;
    source->info.numMipLevels = 3;
    source->mipPixels = {std::vector<unsigned char>(16 * 16),
                         std::vector<unsigned char>(4 * 16), packed<float>({42, -3, .5f, .75f})};
    const auto authored = packed<float>({9, -2, .125f, .25f});
    for (size_t i = 0; i < 4; ++i)
        std::memcpy(source->mipPixels[1].data() + i * 16, authored.data(), 16);
    TextureDesc desc;
    desc.generateMipmaps = true;
    desc.maxMipLevel = std::numeric_limits<unsigned int>::max();
    auto handle = loader_->createTexture(source, desc);
    hipTextureObject_t texture{};
    loadRequestedTexture(*loader_, handle, texture);
    hipResourceDesc resource{};
    ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
    if (resource.resType == hipResourceTypeArray)
        GTEST_SKIP() << "Device uses the supported non-mipmapped fallback";
    ASSERT_EQ(resource.resType, hipResourceTypeMipmappedArray);
    EXPECT_EQ(source->readLevels, (std::vector<unsigned int>{0, 1, 2}));
    for (unsigned int index = 1; index < 3; ++index) {
        hipArray_t level{};
        ASSERT_EQ(hipGetMipmappedArrayLevel(&level, resource.res.mipmap.mipmap, index), hipSuccess);
        const unsigned int width = 4 >> index;
        std::vector<unsigned char> pixels(width * width * 16);
        ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), width * 16, level, 0, 0,
                                          width * 16, width, hipMemcpyDeviceToHost), hipSuccess);
        EXPECT_EQ(pixels, source->mipPixels[index]);
    }
    EXPECT_EQ(loader_->getTotalTextureMemory(), (16u + 4u + 1u) * 16);
}

TEST_F(LoaderTestFixture, FilenameHdrAnd16BitPixelsKeepTheirPrecision) {
    const auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    for (bool hdr : {true, false}) {
        const auto path = std::filesystem::temp_directory_path() /
            ("hip-demand-typed-" + suffix + (hdr ? ".hdr" : ".pgm"));
        struct RemoveFile {
            std::filesystem::path path;
            ~RemoveFile() { std::error_code error; std::filesystem::remove(path, error); }
        } remove{path};
        {
            std::ofstream file(path, std::ios::binary);
            ASSERT_TRUE(file.good());
            if (hdr) {
                file << "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n";
                const unsigned char values[] = {128, 64, 32, 131, 64, 128, 32, 129};
                file.write(reinterpret_cast<const char*>(values), sizeof(values));
            } else {
                file << "P5\n2 1\n65535\n";
                const unsigned char values[] = {0x40, 0, 0xc0, 0};
                file.write(reinterpret_cast<const char*>(values), sizeof(values));
            }
        }
        TextureDesc desc;
        desc.generateMipmaps = false;
        auto handle = loader_->createTexture(path.string(), desc);
        hipTextureObject_t texture{};
        loadRequestedTexture(*loader_, handle, texture);
        hipResourceDesc resource{};
        ASSERT_EQ(hipGetTextureObjectResourceDesc(&resource, texture), hipSuccess);
        ASSERT_EQ(resource.resType, hipResourceTypeArray);
        hipChannelFormatDesc channelDesc{};
        ASSERT_EQ(hipGetChannelDesc(&channelDesc, resource.res.array.array), hipSuccess);
        EXPECT_EQ(channelDesc.f, hipChannelFormatKindFloat);
        std::vector<float> pixels(8);
        ASSERT_EQ(hipMemcpy2DFromArray(pixels.data(), 2 * 16, resource.res.array.array,
                                          0, 0, 2 * 16, 1, hipMemcpyDeviceToHost), hipSuccess);
        if (hdr) {
            EXPECT_EQ(pixels, (std::vector<float>{4, 2, 1, 1, .5f, 1, .25f, 1}));
        } else {
            EXPECT_FLOAT_EQ(pixels[0], 16384.f / 65535);
            EXPECT_FLOAT_EQ(pixels[4], 49152.f / 65535);
            EXPECT_FLOAT_EQ(pixels[3], 1);
            EXPECT_FLOAT_EQ(pixels[7], 1);
        }
    }
}

} // namespace
} // namespace test
} // namespace hip_demand
