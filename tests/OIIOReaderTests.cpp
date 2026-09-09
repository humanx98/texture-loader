#include <hip/hip_runtime.h>
#include <gtest/gtest.h>
#include <ImageSource/OIIOReader.h>
#include <OpenImageIO/imageio.h>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <tuple>
#include <vector>

namespace {

struct FormatCase {
    OIIO::TypeDesc type;
    hipArray_Format format;
    const char* name;
    const char* extension;
};
const FormatCase formats[] = {
    {OIIO::TypeDesc::UINT8, HIP_AD_FORMAT_UNSIGNED_INT8, "UInt8", ".tif"},
    {OIIO::TypeDesc::INT8, HIP_AD_FORMAT_SIGNED_INT8, "Int8", ".tif"},
    {OIIO::TypeDesc::UINT16, HIP_AD_FORMAT_UNSIGNED_INT16, "UInt16", ".tif"},
    {OIIO::TypeDesc::INT16, HIP_AD_FORMAT_SIGNED_INT16, "Int16", ".tif"},
    {OIIO::TypeDesc::UINT32, HIP_AD_FORMAT_UNSIGNED_INT32, "UInt32", ".tif"},
    {OIIO::TypeDesc::INT32, HIP_AD_FORMAT_SIGNED_INT32, "Int32", ".tif"},
    {OIIO::TypeDesc::HALF, HIP_AD_FORMAT_HALF, "Half", ".exr"},
    {OIIO::TypeDesc::FLOAT, HIP_AD_FORMAT_FLOAT, "Float", ".exr"},
};

class OIIOReaderFiles : public ::testing::Test {
protected:
    std::filesystem::path directory;
    void SetUp() override {
        static std::atomic<unsigned int> sequence{0};
        directory = std::filesystem::temp_directory_path() /
            ("hip_demand_oiio_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
             std::to_string(sequence.fetch_add(1)));
        ASSERT_TRUE(std::filesystem::create_directory(directory));
    }
    void TearDown() override {
        if (!directory.empty()) std::filesystem::remove_all(directory);
    }
};

class OIIOReaderNativeFile : public OIIOReaderFiles,
                           public ::testing::WithParamInterface<std::tuple<FormatCase, unsigned int>> {};

TEST_P(OIIOReaderNativeFile, MetadataPayloadAndBaseColorAgree) {
    const auto [format, channels] = GetParam();
    const std::array<float, 4> values{{2.5f, -0.5f, 0.25f, 0.75f}};
    std::vector<unsigned char> native(channels * format.type.size());
    ASSERT_TRUE(OIIO::convert_types(OIIO::TypeDesc::FLOAT, values.data(),
                                    format.type, native.data(), channels));
    const auto filename = (directory / (std::string("typed") + format.extension)).string();
    auto output = OIIO::ImageOutput::create(filename);
    ASSERT_NE(output, nullptr);
    OIIO::ImageSpec spec(1, 1, static_cast<int>(channels), format.type);
    ASSERT_TRUE(output->open(filename, spec)) << output->geterror();
    ASSERT_TRUE(output->write_image(format.type, native.data())) << output->geterror();
    ASSERT_TRUE(output->close()) << output->geterror();

    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    ASSERT_NO_THROW(reader.open(&info));
    ASSERT_TRUE(info.isValid);
    ASSERT_EQ(info.format, format.format);
    ASSERT_EQ(info.numChannels, channels);
    ASSERT_EQ(info.numMipLevels, 1u);
    ASSERT_EQ(hip_demand::getBytesPerChannel(info.format) * info.numChannels, native.size());
    constexpr size_t guard = 16;
    std::vector<unsigned char> copied(native.size() + 2 * guard, 0xa5);
    ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(copied.data() + guard), 0, 1, 1));
    EXPECT_EQ(std::vector<unsigned char>(copied.begin() + guard, copied.end() - guard), native);
    EXPECT_EQ(std::vector<unsigned char>(copied.begin(), copied.begin() + guard),
              std::vector<unsigned char>(guard, 0xa5));
    EXPECT_EQ(std::vector<unsigned char>(copied.end() - guard, copied.end()),
              std::vector<unsigned char>(guard, 0xa5));
    EXPECT_EQ(reader.getNumBytesRead(), native.size());
    EXPECT_FALSE(reader.readMipLevel(nullptr, 0, 1, 1));
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(copied.data()), 0, 2, 1));
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(copied.data()), 1, 1, 1));

    std::array<float, 4> expected{{0.f, 0.f, 0.f, 1.f}};
    ASSERT_TRUE(OIIO::convert_types(format.type, native.data(), OIIO::TypeDesc::FLOAT,
                                    expected.data(), channels));
    float4 color{};
    ASSERT_TRUE(reader.readBaseColor(color));
    EXPECT_FLOAT_EQ(color.x, expected[0]);
    EXPECT_FLOAT_EQ(color.y, expected[1]);
    EXPECT_FLOAT_EQ(color.z, expected[2]);
    EXPECT_FLOAT_EQ(color.w, expected[3]);
    reader.close();
    EXPECT_FALSE(reader.readBaseColor(color));
}

INSTANTIATE_TEST_SUITE_P(FormatsAndChannels, OIIOReaderNativeFile,
    ::testing::Combine(::testing::ValuesIn(formats), ::testing::Range(1u, 5u)),
    [](const auto& parameter) {
        return std::string(std::get<0>(parameter.param).name) + "C" +
               std::to_string(std::get<1>(parameter.param));
    });

TEST_F(OIIOReaderFiles, ReportsOnlyAuthoredLevelsSoLoaderOwnsFallbackFiltering) {
    const auto filename = (directory / "base.exr").string();
    auto output = OIIO::ImageOutput::create(filename);
    ASSERT_NE(output, nullptr);
    OIIO::ImageSpec spec(3, 1, 1, OIIO::TypeDesc::FLOAT);
    const std::array<float, 3> pixels{{-2.f, 0.5f, 90.f}};
    ASSERT_TRUE(output->open(filename, spec));
    ASSERT_TRUE(output->write_image(OIIO::TypeDesc::FLOAT, pixels.data()));
    ASSERT_TRUE(output->close());
    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    reader.open(&info);
    ASSERT_EQ(info.numMipLevels, 1u);
    std::array<float, 3> actual{};
    ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(actual.data()), 0, 3, 1));
    EXPECT_EQ(actual, pixels);
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(actual.data()), 1, 1, 1));
    float4 color{};
    EXPECT_FALSE(reader.readBaseColor(color));
}

TEST_F(OIIOReaderFiles, ReadsAuthoredHdrMipValuesInsteadOfRegeneratingThem) {
    const auto filename = (directory / "authored.exr").string();
    auto output = OIIO::ImageOutput::create(filename);
    ASSERT_NE(output, nullptr);
    ASSERT_TRUE(output->supports("mipmap"));
    const std::array<float, 3> values{{2.f, 10.f, -3.5f}};
    for (int level = 0; level < 3; ++level) {
        const int size = 4 >> level;
        OIIO::ImageSpec spec(size, size, 4, OIIO::TypeDesc::FLOAT);
        spec.tile_width = spec.tile_height = 16;
        spec.attribute("textureformat", "Plain Texture");
        spec.attribute("openexr:levelmode", 1);
        spec.attribute("openexr:roundingmode", 0);
        ASSERT_TRUE(output->open(filename, spec,
            level == 0 ? OIIO::ImageOutput::Create : OIIO::ImageOutput::AppendMIPLevel))
            << output->geterror();
        const std::vector<float> pixels(static_cast<size_t>(size) * size * 4, values[level]);
        ASSERT_TRUE(output->write_image(OIIO::TypeDesc::FLOAT, pixels.data())) << output->geterror();
    }
    ASSERT_TRUE(output->close()) << output->geterror();
    output.reset();
    auto input = OIIO::ImageInput::open(filename);
    ASSERT_NE(input, nullptr) << OIIO::geterror();
    input->close();
    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    reader.open(&info);
    ASSERT_EQ(info.numMipLevels, 3u);
    ASSERT_TRUE(info.isTiled);
    for (unsigned int level = 0; level < 3; ++level) {
        const unsigned int size = 4 >> level;
        std::vector<float> pixels(static_cast<size_t>(size) * size * 4);
        ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(pixels.data()), level, size, size));
        for (const float value : pixels) EXPECT_FLOAT_EQ(value, values[level]);
    }
    EXPECT_EQ(reader.getNumBytesRead(), (16u + 4u + 1u) * 4u * sizeof(float));
    float4 color{};
    ASSERT_TRUE(reader.readBaseColor(color));
    EXPECT_FLOAT_EQ(color.x, -3.5f);
    EXPECT_FLOAT_EQ(color.w, -3.5f);
}

TEST_F(OIIOReaderFiles, KeepsValidBaseImageWhenAuthoredMipDimensionsRoundUp) {
    const auto filename = (directory / "rounded-up.exr").string();
    auto output = OIIO::ImageOutput::create(filename);
    ASSERT_NE(output, nullptr);
    const std::array<int, 3> widths{{3, 2, 1}};
    for (int level = 0; level < 3; ++level) {
        OIIO::ImageSpec spec(widths[level], 1, 4, OIIO::TypeDesc::FLOAT);
        spec.tile_width = spec.tile_height = 16;
        spec.attribute("textureformat", "Plain Texture");
        spec.attribute("openexr:levelmode", 1);
        spec.attribute("openexr:roundingmode", 1);
        ASSERT_TRUE(output->open(filename, spec,
            level == 0 ? OIIO::ImageOutput::Create : OIIO::ImageOutput::AppendMIPLevel))
            << output->geterror();
        const std::vector<float> pixels(static_cast<size_t>(widths[level]) * 4, 6.f + level);
        ASSERT_TRUE(output->write_image(OIIO::TypeDesc::FLOAT, pixels.data())) << output->geterror();
    }
    ASSERT_TRUE(output->close()) << output->geterror();
    output.reset();
    hip_demand::OIIOReader reader(filename);
    hip_demand::TextureInfo info;
    ASSERT_NO_THROW(reader.open(&info));
    ASSERT_EQ(info.width, 3u);
    ASSERT_EQ(info.numMipLevels, 1u);
    std::array<float, 12> pixels{};
    ASSERT_TRUE(reader.readMipLevel(reinterpret_cast<char*>(pixels.data()), 0, 3, 1));
    for (const float value : pixels) EXPECT_FLOAT_EQ(value, 6.f);
    EXPECT_FALSE(reader.readMipLevel(reinterpret_cast<char*>(pixels.data()), 1, 1, 1));
}

} // namespace
