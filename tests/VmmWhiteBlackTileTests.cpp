// SPDX-License-Identifier: MIT
// Tests for the VMM loader's sharing of white and black tiles (Options::coalesceWhiteBlackTiles).

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <DemandLoading/vmm/DemandTextureLoaderImpl.h>
#include <DemandLoading/vmm/WhiteBlackTile.h>
#include "VmmLoaderTestUtils.h"
#include <ImageSource/ImageSource.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace hip_demand::vmm::test {

namespace {

constexpr size_t pageBytes = PageSystem::PAGE_BYTES;

template <class T>
std::vector<uint8_t> texelBytes( std::initializer_list<T> channels )
{
    std::vector<uint8_t> texel( channels.size() * sizeof( T ) );
    std::memcpy( texel.data(), channels.begin(), texel.size() );
    return texel;
}

std::vector<uint8_t> pageOf( const std::vector<uint8_t>& texel )
{
    std::vector<uint8_t> page;
    page.reserve( pageBytes );
    while( page.size() < pageBytes )
        page.insert( page.end(), texel.begin(), texel.end() );
    return page;
}

// Whether a page of the texel is a white or black tile, and that its key is the texel.
template <class T>
bool isWhiteBlack( hipArray_Format format, std::initializer_list<T> channels )
{
    const std::vector<uint8_t> texel = texelBytes( channels );
    const std::vector<uint8_t> page  = pageOf( texel );
    std::string                key;
    if( !whiteBlackTileKey( page.data(), page.size(), format, static_cast<uint32_t>( channels.size() ), key ) )
        return false;
    EXPECT_EQ( key, std::string( texel.begin(), texel.end() ) );
    return true;
}

}  // namespace

TEST( VmmWhiteBlackTileKeyTest, WhiteAndBlackTexelsOfEveryChannelCount )
{
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 0 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 0, 0 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 255 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 0, 0, 0 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 255, 255 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 0, 0, 0, 0 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 0, 0, 0, 255 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 255, 255, 0 } ) );
    EXPECT_TRUE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 255, 255, 255 } ) );
}

TEST( VmmWhiteBlackTileKeyTest, OneIsWhatTheSamplerNormalizesToOne )
{
    EXPECT_TRUE( isWhiteBlack<int8_t>( HIP_AD_FORMAT_SIGNED_INT8, { 127, 127, 127, 0 } ) );
    EXPECT_TRUE( isWhiteBlack<uint16_t>( HIP_AD_FORMAT_UNSIGNED_INT16, { 65535 } ) );
    EXPECT_TRUE( isWhiteBlack<int16_t>( HIP_AD_FORMAT_SIGNED_INT16, { 32767, 32767 } ) );
    EXPECT_TRUE( isWhiteBlack<uint32_t>( HIP_AD_FORMAT_UNSIGNED_INT32, { 0xFFFFFFFFu } ) );
    EXPECT_TRUE( isWhiteBlack<int32_t>( HIP_AD_FORMAT_SIGNED_INT32, { 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF } ) );
    EXPECT_TRUE( isWhiteBlack<uint16_t>( HIP_AD_FORMAT_HALF, { 0x3C00, 0x3C00, 0x3C00, 0x0000 } ) );  // 1.0h
    EXPECT_TRUE( isWhiteBlack<float>( HIP_AD_FORMAT_FLOAT, { 1.f, 1.f, 1.f, 1.f } ) );
    EXPECT_TRUE( isWhiteBlack<float>( HIP_AD_FORMAT_FLOAT, { 0.f } ) );

    EXPECT_FALSE( isWhiteBlack<int8_t>( HIP_AD_FORMAT_SIGNED_INT8, { -128 } ) );
    EXPECT_FALSE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_SIGNED_INT8, { 255 } ) );
    EXPECT_FALSE( isWhiteBlack<uint16_t>( HIP_AD_FORMAT_HALF, { 0x3800 } ) );  // 0.5h
    EXPECT_FALSE( isWhiteBlack<float>( HIP_AD_FORMAT_FLOAT, { -1.f } ) );
}

TEST( VmmWhiteBlackTileKeyTest, OtherTexelsAreNotWhiteOrBlack )
{
    EXPECT_FALSE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 128 } ) );
    EXPECT_FALSE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 0 } ) );
    EXPECT_FALSE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 255, 0 } ) );
    EXPECT_FALSE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 255, 0, 0, 255 } ) );
    EXPECT_FALSE( isWhiteBlack<uint8_t>( HIP_AD_FORMAT_UNSIGNED_INT8, { 0, 0, 0, 128 } ) );
    EXPECT_FALSE( isWhiteBlack<float>( HIP_AD_FORMAT_FLOAT, { 0.5f, 0.5f, 0.5f, 1.f } ) );
}

TEST( VmmWhiteBlackTileKeyTest, OtherPagesHaveNoKey )
{
    std::string          key;
    std::vector<uint8_t> page = pageOf( { 255 } );
    page[pageBytes - 1]       = 254;
    EXPECT_FALSE( whiteBlackTileKey( page.data(), page.size(), HIP_AD_FORMAT_UNSIGNED_INT8, 1, key ) );
    page[pageBytes - 1] = 255;
    page[0]             = 0;
    EXPECT_FALSE( whiteBlackTileKey( page.data(), page.size(), HIP_AD_FORMAT_UNSIGNED_INT8, 1, key ) );

    // A page of alternating texels is not uniform.
    const std::vector<uint8_t> stripes = pageOf( { 0, 255 } );
    EXPECT_FALSE( whiteBlackTileKey( stripes.data(), stripes.size(), HIP_AD_FORMAT_UNSIGNED_INT8, 1, key ) );

    const std::vector<uint8_t> white = pageOf( { 255 } );
    EXPECT_FALSE( whiteBlackTileKey( nullptr, pageBytes, HIP_AD_FORMAT_UNSIGNED_INT8, 1, key ) );
    EXPECT_FALSE( whiteBlackTileKey( white.data(), white.size(), HIP_AD_FORMAT_UNSIGNED_INT8, 0, key ) );
    EXPECT_FALSE( whiteBlackTileKey( white.data(), white.size(), HIP_AD_FORMAT_UNSIGNED_INT8, 5, key ) );
    EXPECT_FALSE( whiteBlackTileKey( white.data(), white.size(), static_cast<hipArray_Format>( 0xFF ), 1, key ) );
}

class VmmWhiteBlackTileTest : public VmmLoaderTest
{
};

TEST_F( VmmWhiteBlackTileTest, WhiteTilesShareOnePhysicalPage )
{
    createLoader( true );
    // 1024x1024 8-bit tiles are 256x256: 16 tiles on the finest level.
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 255 ), TextureDescriptor{} );
    const DeviceTextureInfo     info    = loadTextureInfo( texture );
    const std::vector<uint32_t> pages   = loadLevel0( info );
    ASSERT_EQ( pages.size(), 16u );

    EXPECT_EQ( impl_->physicalPageCount(), 1u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 1u );
    for( uint32_t pageId : pages )
    {
        EXPECT_TRUE( isResident( pageId ) ) << "page " << pageId;
        EXPECT_EQ( pageValue( pageId ), 255 ) << "page " << pageId;
    }
}

TEST_F( VmmWhiteBlackTileTest, TexturesShareThePageOfATexel )
{
    createLoader( true );
    const DemandTexture& white         = loader_->createTexture( std::make_shared<TestImage>( 1024, 512, 1, true, 255 ), TextureDescriptor{} );
    const DemandTexture& whiteToo      = loader_->createTexture( std::make_shared<TestImage>( 512, 512, 1, true, 255 ), TextureDescriptor{} );
    const DemandTexture& black         = loader_->createTexture( std::make_shared<TestImage>( 512, 512, 1, true, 0 ), TextureDescriptor{} );
    const auto           whitePages    = loadLevel0( loadTextureInfo( white ) );
    const auto           whiteTooPages = loadLevel0( loadTextureInfo( whiteToo ) );
    const auto           blackPages    = loadLevel0( loadTextureInfo( black ) );
    ASSERT_EQ( whitePages.size(), 8u );
    ASSERT_EQ( whiteTooPages.size(), 4u );
    ASSERT_EQ( blackPages.size(), 4u );

    EXPECT_EQ( impl_->physicalPageCount(), 2u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 2u );
    for( uint32_t pageId : whiteTooPages )
        EXPECT_EQ( pageValue( pageId ), 255 ) << "page " << pageId;
    for( uint32_t pageId : blackPages )
        EXPECT_EQ( pageValue( pageId ), 0 ) << "page " << pageId;
}

TEST_F( VmmWhiteBlackTileTest, GreyTilesGetTheirOwnPages )
{
    createLoader( true );
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 128 ), TextureDescriptor{} );
    const std::vector<uint32_t> pages   = loadLevel0( loadTextureInfo( texture ) );
    ASSERT_EQ( pages.size(), 16u );

    EXPECT_EQ( impl_->physicalPageCount(), 16u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 0u );
    for( uint32_t pageId : pages )
        EXPECT_EQ( pageValue( pageId ), 128 ) << "page " << pageId;
}

TEST_F( VmmWhiteBlackTileTest, OtherTilesGetTheirOwnPages )
{
    createLoader( true );
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, false, 0 ), TextureDescriptor{} );
    const std::vector<uint32_t> pages   = loadLevel0( loadTextureInfo( texture ) );
    ASSERT_EQ( pages.size(), 16u );

    EXPECT_EQ( impl_->physicalPageCount(), 16u );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 0u );
    for( uint32_t pageId : pages )
        EXPECT_EQ( pageValue( pageId ), -1 ) << "page " << pageId;
}

TEST_F( VmmWhiteBlackTileTest, SharingIsOffByDefault )
{
    createLoader( false );
    const DemandTexture&        texture = loader_->createTexture( std::make_shared<TestImage>( 1024, 1024, 1, true, 255 ), TextureDescriptor{} );
    const std::vector<uint32_t> pages   = loadLevel0( loadTextureInfo( texture ) );

    EXPECT_EQ( impl_->physicalPageCount(), pages.size() );
    EXPECT_EQ( impl_->whiteBlackTileCount(), 0u );
}

}  // namespace hip_demand::vmm::test
