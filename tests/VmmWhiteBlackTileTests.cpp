// SPDX-License-Identifier: MIT
// Tests for the VMM loader's sharing of white and black tiles (Options::coalesceWhiteBlackTiles).

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <DemandLoading/vmm/DemandTextureLoaderImpl.h>
#include <DemandLoading/vmm/WhiteBlackTile.h>
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

namespace {

// An in-memory single-channel 8-bit image whose tiles are all one value, or each a byte ramp when not uniform.
class TestImage : public ImageSource
{
  public:
    TestImage( uint32_t width, uint32_t height, uint32_t numMipLevels, bool uniform, uint8_t value )
        : uniform_( uniform )
        , value_( value )
    {
        info_.width        = width;
        info_.height       = height;
        info_.numChannels  = 1;
        info_.format       = HIP_AD_FORMAT_UNSIGNED_INT8;
        info_.numMipLevels = numMipLevels;
        info_.isTiled      = true;
        info_.isValid      = true;
    }

    void open( TextureInfo* info ) override
    {
        open_ = true;
        if( info )
            *info = info_;
    }
    void               close() override { open_ = false; }
    bool               isOpen() const override { return open_; }
    const TextureInfo& getInfo() const override { return info_; }

    bool readMipLevel( char* dest, unsigned int mipLevel, unsigned int width, unsigned int height, hipStream_t ) override
    {
        if( mipLevel >= info_.numMipLevels )
            return false;
        fill( dest, static_cast<size_t>( width ) * height );
        return true;
    }

    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, hipStream_t ) override
    {
        if( mipLevel >= info_.numMipLevels )
            return false;
        fill( dest, static_cast<size_t>( tile.width ) * tile.height );
        return true;
    }

    bool               readBaseColor( float4& ) override { return false; }
    unsigned long long getNumBytesRead() const override { return 0; }
    double             getTotalReadTime() const override { return 0.0; }

  private:
    void fill( char* dest, size_t count ) const
    {
        for( size_t i = 0; i < count; ++i )
            dest[i] = uniform_ ? static_cast<char>( value_ ) : static_cast<char>( i );
    }

    TextureInfo info_{};
    bool        uniform_ = true;
    uint8_t     value_   = 0;
    bool        open_    = false;
};

class VmmWhiteBlackTileTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        int deviceCount = 0;
        if( hipGetDeviceCount( &deviceCount ) != hipSuccess || deviceCount == 0 )
            GTEST_SKIP() << "No HIP device";
        ASSERT_EQ( hipSetDevice( 0 ), hipSuccess );
        int vmmSupported = 0;
        ASSERT_EQ( hipDeviceGetAttribute( &vmmSupported, hipDeviceAttributeVirtualMemoryManagementSupported, 0 ), hipSuccess );
        if( !vmmSupported )
            GTEST_SKIP() << "No virtual memory management";
        ASSERT_EQ( hipStreamCreate( &stream_ ), hipSuccess );
    }

    void TearDown() override
    {
        loader_.reset();
        if( stream_ )
            ASSERT_EQ( hipStreamDestroy( stream_ ), hipSuccess );
    }

    void createLoader( bool coalesceWhiteBlackTiles )
    {
        Options options{};
        options.maxVirtualPages         = 4096;
        options.maxPhysicalPages        = 4096;
        options.maxTextures             = 16;
        options.maxRequests             = 256;
        options.enableEviction          = false;
        options.maxThreads              = 2;
        options.coalesceWhiteBlackTiles = coalesceWhiteBlackTiles;
        loader_                         = createDemandTextureLoader( options );
        impl_                           = dynamic_cast<DemandTextureLoaderImpl*>( loader_.get() );
        ASSERT_NE( impl_, nullptr );
        // Fills the context with the resource table the request ids are computed from.
        loader_->launchPrepare( stream_, context_ );
        ASSERT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
    }

    // Runs one request pass for the given resources, as a sampling kernel would request them.
    void requestResources( const std::vector<uint32_t>& resourceIds )
    {
        loader_->launchPrepare( stream_, context_ );
        ASSERT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
        std::vector<uint32_t> words( context_.referenceBits.len, 0 );
        for( uint32_t resourceId : resourceIds )
            words[resourceId / 32] |= 1u << ( resourceId % 32 );
        ASSERT_EQ( hipMemcpy( context_.referenceBits.ptr, words.data(), words.size() * sizeof( uint32_t ), hipMemcpyHostToDevice ), hipSuccess );
        loader_->processRequests( stream_, context_ ).wait();
        ASSERT_EQ( hipStreamSynchronize( stream_ ), hipSuccess );
    }

    DeviceTextureInfo loadTextureInfo( const DemandTexture& texture )
    {
        requestResources( { context_.resourceTable.textureInfos.getResourceId( texture.getId() ) } );
        DeviceTextureInfo info{};
        EXPECT_EQ( hipMemcpy( &info, context_.textureInfos.ptr + texture.getId(), sizeof( info ), hipMemcpyDeviceToHost ), hipSuccess );
        EXPECT_TRUE( info.isLoaded() );
        return info;
    }

    // Loads every tile of the finest level and returns their page ids.
    std::vector<uint32_t> loadLevel0( const DeviceTextureInfo& info )
    {
        const DeviceMipLevel  mip = info.getMipLevel( 0 );
        std::vector<uint32_t> pageIds, resourceIds;
        for( uint32_t page = 0; page < mip.pageCount(); ++page )
        {
            pageIds.push_back( mip.startPage + page );
            resourceIds.push_back( context_.resourceTable.textureTiles.getResourceId( mip.startPage + page ) );
        }
        requestResources( resourceIds );
        return pageIds;
    }

    bool isResident( uint32_t pageId )
    {
        const uint32_t resourceId = context_.resourceTable.textureTiles.getResourceId( pageId );
        uint32_t       word       = 0;
        EXPECT_EQ( hipMemcpy( &word, context_.residenceBits.ptr + resourceId / 32, sizeof( word ), hipMemcpyDeviceToHost ), hipSuccess );
        return ( word >> ( resourceId % 32 ) ) & 1u;
    }

    // The value a page holds, or -1 when it holds more than one.
    int pageValue( uint32_t pageId )
    {
        std::vector<uint8_t> host( context_.pageSize );
        if( hipMemcpy( host.data(), context_.pageMemory.ptr + pageId * context_.pageSize, host.size(), hipMemcpyDeviceToHost ) != hipSuccess )
            return -2;
        for( uint8_t byte : host )
        {
            if( byte != host[0] )
                return -1;
        }
        return host[0];
    }

    hipStream_t                          stream_ = nullptr;
    std::unique_ptr<DemandTextureLoader> loader_;
    DemandTextureLoaderImpl*             impl_ = nullptr;
    DeviceContext                        context_{};
};

}  // namespace

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
