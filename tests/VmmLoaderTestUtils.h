// SPDX-License-Identifier: MIT
// Shared fixture for tests that drive the VMM loader with in-memory images.

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <DemandLoading/vmm/DemandTextureLoaderImpl.h>
#include <ImageSource/ImageSource.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace hip_demand::vmm::test {

// An in-memory 8-bit image whose tiles are all one value in every channel, or each a byte ramp when not uniform.
// The part of a tile past the edge of its level is always a byte ramp, as an image source may leave anything there.
class TestImage : public ImageSource
{
  public:
    TestImage( uint32_t width, uint32_t height, uint32_t numMipLevels, bool uniform, uint8_t value, uint32_t numChannels = 1 )
        : uniform_( uniform )
        , value_( value )
    {
        info_.width        = width;
        info_.height       = height;
        info_.numChannels  = numChannels;
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
        fill( dest, static_cast<size_t>( width ) * height * info_.numChannels );
        return true;
    }

    bool readTile( char* dest, unsigned int mipLevel, const Tile& tile, hipStream_t ) override
    {
        if( mipLevel >= info_.numMipLevels )
            return false;
        const size_t rowBytes = static_cast<size_t>( tile.width ) * info_.numChannels;
        fill( dest, rowBytes * tile.height );
        const uint32_t levelWidth  = std::max( info_.width >> mipLevel, 1u );
        const uint32_t levelHeight = std::max( info_.height >> mipLevel, 1u );
        const uint32_t visibleWidth  = std::min( tile.width, levelWidth - tile.x * tile.width );
        const uint32_t visibleHeight = std::min( tile.height, levelHeight - tile.y * tile.height );
        for( uint32_t row = 0; row < tile.height; ++row )
        {
            char*        line   = dest + row * rowBytes;
            const size_t offset = row < visibleHeight ? visibleWidth * info_.numChannels : 0;
            for( size_t i = offset; i < rowBytes; ++i )
                line[i] = static_cast<char>( row + i );
        }
        numTilesRead_++;
        return true;
    }

    bool               readBaseColor( float4& ) override { return false; }
    unsigned long long getNumTilesRead() const override { return numTilesRead_; }
    unsigned long long getNumBytesRead() const override { return numBytesRead_; }
    double             getTotalReadTime() const override { return 0.0; }

  private:
    void fill( char* dest, size_t count )
    {
        for( size_t i = 0; i < count; ++i )
            dest[i] = uniform_ ? static_cast<char>( value_ ) : static_cast<char>( i );
        numBytesRead_ += count;
    }

    TextureInfo                     info_{};
    bool                            uniform_ = true;
    uint8_t                         value_   = 0;
    bool                            open_    = false;
    std::atomic<unsigned long long> numTilesRead_{ 0 };
    std::atomic<unsigned long long> numBytesRead_{ 0 };
};

class VmmLoaderTest : public ::testing::Test
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
        options_                        = options;
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
        return loadLevel0Tiles( info, []( uint32_t, uint32_t ) { return true; } );
    }

    // Loads the tiles of the finest level that the predicate accepts by their (x, y) and returns their page ids.
    template <class Predicate>
    std::vector<uint32_t> loadLevel0Tiles( const DeviceTextureInfo& info, Predicate accept )
    {
        const DeviceMipLevel  mip = info.getMipLevel( 0 );
        std::vector<uint32_t> pageIds, resourceIds;
        for( uint32_t y = 0; y < mip.tilesY; ++y )
        {
            for( uint32_t x = 0; x < mip.tilesX; ++x )
            {
                if( !accept( x, y ) )
                    continue;
                const uint32_t pageId = mip.startPage + y * mip.tilesX + x;
                pageIds.push_back( pageId );
                resourceIds.push_back( context_.resourceTable.textureTiles.getResourceId( pageId ) );
            }
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

    // The value the first bytes of a page hold (all of them by default), or -1 when they hold more than one.
    int pageValue( uint32_t pageId, size_t bytes = 0 )
    {
        std::vector<uint8_t> host( bytes ? bytes : context_.pageSize );
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
    Options                              options_{};
    std::unique_ptr<DemandTextureLoader> loader_;
    DemandTextureLoaderImpl*             impl_ = nullptr;
    DeviceContext                        context_{};
};


}  // namespace hip_demand::vmm::test
