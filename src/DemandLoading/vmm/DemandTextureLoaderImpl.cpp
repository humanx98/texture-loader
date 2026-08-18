#include "../Internal/HipCheck.h"
#include "../Internal/Utils.h"
#include "Allocator.h"
#include "PageSystem.h"
// #include "TicketImpl.h"
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <ImageSource/TextureInfo.h>
#include <algorithm>
#include <array>
// #include <condition_variable>
// #include <exception>
// #include <functional>
#include <mutex>
// #include <queue>
// #include <stdexcept>
// #include <thread>

namespace hip_demand::vmm {

using internal::allocArray;
using internal::Bitset;
using internal::ceilDiv;
using internal::freeArray;
using internal::memcpyDtoH;
using internal::memcpyHtoD;
using internal::memset;
using internal::mipDimensions;
using internal::NonCopyble;
using internal::safeAdd;
using internal::sizeInBytes;
using internal::tileShapeForGranularity;

class RequestQueue : NonCopyble
{
  public:
  private:
};

class ThreadPool : NonCopyble
{
  public:
  private:
};

struct Resource
{
    enum class Type
    {
        TextureId,
        Tile,
        MipTail
    };

    struct Tile
    {
        uint32_t textureId = 0;
        uint32_t mipLevel  = 0;
        uint32_t tileX     = 0;
        uint32_t tileY     = 0;
        uint32_t pageId    = 0;
    };

    struct MipTail
    {
        uint32_t textureId = 0;
        uint32_t pageId    = 0;
    };

    Type type;
    union
    {
        uint32_t textureId;
        Tile     tile;
        MipTail  mipTail;
    };

    Resource() {}
};

class DemandTextureImpl : public DemandTexture, NonCopyble
{
  public:
    DemandTextureImpl( uint32_t textureId, std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& textureDescriptor )
        : id( textureId )
        , image( imageSource )
        , descriptor( textureDescriptor )
    {
    }
    uint32_t getId() const override { return id; }

    uint32_t                     id                  = INVALID_TEXTURE;
    uint32_t                     loadedTextureInfoId = INVALID_TEXTURE;
    std::shared_ptr<ImageSource> image{};
    TextureDescriptor            descriptor;
};

class DemandTextureLoaderImpl : public DemandTextureLoader, NonCopyble
{
  public:
    explicit DemandTextureLoaderImpl( const Options& options );
    ~DemandTextureLoaderImpl() override;

    const DemandTexture& createTexture( std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& textureDesc ) override;
    void launchPrepare( hipStream_t stream, DeviceContext& deviceContext ) override;
    void processRequests( hipStream_t stream, const DeviceContext& deviceContext ) override;
    void processRequestsCallback( const DeviceContext& deviceContext );

  private:
    void     initPageTable( uint32_t& resourceCount );
    void     initDeviceContext( DeviceContext& deviceContext, hipStream_t stream );
    void     freeDeviceContext( const DeviceContext& deviceContext );
    Resource decode( uint32_t resourceId );
    void     processTextureInfo( uint32_t textureId, hipStream_t stream, const DeviceContext& deviceContext );
    void     processTile( const Resource::Tile& tile, hipStream_t stream, const DeviceContext& deviceContext );
    void     processMipTail( const Resource::MipTail& mipTail, hipStream_t stream, const DeviceContext& deviceContext );

    mutable std::mutex    mutex_;
    Options               options_{};
    std::atomic<bool>     residentBitsDirty_{ false };
    Bitset                residentBits_{};
    std::vector<uint32_t> requestedResources_{};
    std::vector<uint8_t>  tmpPageBuffer_{};
    PageSystem            pageSystem_;
    PageTable             pageTable_{};

    std::array<uint32_t, static_cast<size_t>( CounterIndex::NumCounters )> counters_{};

    std::vector<std::unique_ptr<DemandTextureImpl>> textures_{};
    // note that this list should be accessed by texture.loadedTextureInfoId
    // and it's ordered by startPage in order to use std::upper_bound
    std::vector<DeviceTextureInfo> loadedTextureInfos_{};
    Allocator<DeviceTextureInfo>   textureInfoAllocator_;

    std::vector<std::unique_ptr<DeviceContext>> deviceContextPool_{};
    std::vector<size_t>                         freeDeviceContextList_{};
};

class ProcessRequestCallback
{
  public:
    explicit ProcessRequestCallback( DemandTextureLoaderImpl* loader, DeviceContext deviceContext )
        : loader_( loader )
        , deviceContext_( deviceContext )
    {
    }

    static void enqueue( hipStream_t stream, ProcessRequestCallback* callback )
    {
        HIP_CHECK( hipLaunchHostFunc( stream, &staticCallback, callback ) );
    }

  private:
    void callback() { loader_->processRequestsCallback( deviceContext_ ); }

    static void staticCallback( void* arg )
    {
        std::unique_ptr<ProcessRequestCallback> callback( static_cast<ProcessRequestCallback*>( arg ) );
        callback->callback();
    }

    DemandTextureLoaderImpl* loader_;
    DeviceContext            deviceContext_;
};

DemandTextureLoaderImpl::DemandTextureLoaderImpl( const Options& options )
    : options_( options )
    , pageSystem_( options_.maxVirtualPages, options_.maxPhysicalPages )
    , textureInfoAllocator_( pageSystem_ )
{
    if( options_.maxRequests == 0 )
        throw std::invalid_argument( "maxRequests cannot be 0" );

    uint32_t resourceCount = 0;
    initPageTable( resourceCount );
    textureInfoAllocator_.setRange( pageTable_.textureInfos );

    tmpPageBuffer_.resize( pageSystem_.pageBytes() );
    residentBits_.resize( resourceCount );
    requestedResources_.resize( options_.maxRequests );
}

void DemandTextureLoaderImpl::initPageTable( uint32_t& resourceCount )
{
    pageTable_.pageSize    = pageSystem_.pageBytes();
    pageTable_.maxTextures = options_.maxTextures;

    const size_t   maxTextureInfoSize  = options_.maxTextures * sizeof( DeviceTextureInfo );
    const uint32_t maxTextureInfoPages = static_cast<uint32_t>( ceilDiv( maxTextureInfoSize, pageTable_.pageSize ) );
    if( maxTextureInfoPages > options_.maxVirtualPages )
        throw std::invalid_argument( "maxVirtualPages is too small to store texture metadata" );
    const uint32_t maxTextureTilePages = options_.maxVirtualPages - maxTextureInfoPages;

    pageTable_.textureInfos = PageTable::Range( 0, maxTextureInfoPages );
    pageTable_.textureTiles = PageTable::Range( pageTable_.textureInfos.pageCount, maxTextureTilePages );

    resourceCount = 0;
    if( !safeAdd( options_.maxTextures, pageTable_.textureTiles.pageCount, resourceCount ) )
        throw std::overflow_error(
            "Cannot create demand texture loader: total resource count exceeds the uint32_t limit" );
}

DemandTextureLoaderImpl::~DemandTextureLoaderImpl()
{
    if( !deviceContextPool_.empty() )
    {
        freeArray( deviceContextPool_[0]->residentBits );
        freeArray( deviceContextPool_[0]->textureInfos );
        for( auto& ctx : deviceContextPool_ )
        {
            freeArray( ctx->requestedBits );
            freeArray( ctx->requestedResources );
            freeArray( ctx->counters );
        }
    }
}

const DemandTexture& DemandTextureLoaderImpl::createTexture( std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& descriptor )
{
    std::lock_guard<std::mutex> lock( mutex_ );

    if( !imageSource )
        throw std::invalid_argument( "Image source is null" );

    if( textures_.size() >= options_.maxTextures )
        throw std::runtime_error( "Maximum demand texture count exceeded" );

    const uint32_t textureId = static_cast<uint32_t>( textures_.size() );
    textures_.emplace_back( std::make_unique<DemandTextureImpl>( textureId, imageSource, descriptor ) );
    return *textures_.back();
}

void DemandTextureLoaderImpl::launchPrepare( hipStream_t stream, DeviceContext& deviceContext )
{
    std::lock_guard<std::mutex> lock( mutex_ );

    initDeviceContext( deviceContext, stream );

    if( !textures_.empty() )
    {
        memset( deviceContext.requestedBits, 0, stream );
        memset( deviceContext.counters, 0, stream );

        if( residentBitsDirty_.exchange( false, std::memory_order_acquire ) )
            memcpyHtoD( deviceContext.residentBits, residentBits_.words(), stream );

        deviceContext.textureInfos.len = static_cast<uint32_t>( textures_.size() );
    }
}

void DemandTextureLoaderImpl::initDeviceContext( DeviceContext& deviceContext, hipStream_t stream )
{
    if( !freeDeviceContextList_.empty() )
    {
        deviceContext = *deviceContextPool_.at( freeDeviceContextList_.back() ).get();
        freeDeviceContextList_.pop_back();
        return;
    }

    deviceContextPool_.push_back( std::make_unique<DeviceContext>() );
    deviceContext = *deviceContextPool_.back().get();

    // set data per stream
    deviceContext.poolIndex          = deviceContextPool_.size() - 1;
    deviceContext.requestedBits      = allocArray<uint32_t>( residentBits_.wordCount(), true, stream );
    deviceContext.requestedResources = allocArray<uint32_t>( requestedResources_.size(), true, stream );
    deviceContext.counters           = allocArray<uint32_t>( counters_.size(), true, stream );

    // set data per loader
    deviceContext.pageTable  = pageTable_;
    deviceContext.pageMemory = pageSystem_.virtualAddressSpace();
    if( deviceContextPool_.empty() )
    {
        deviceContext.residentBits = allocArray<uint32_t>( residentBits_.wordCount(), true, stream );
        deviceContext.textureInfos = allocArray<DeviceTextureInfo*>( options_.maxTextures, true, stream );
    }
    else
    {
        deviceContext.residentBits = deviceContextPool_.front()->residentBits;
        deviceContext.textureInfos = deviceContextPool_.front()->textureInfos;
    }
}

void DemandTextureLoaderImpl::freeDeviceContext( const DeviceContext& deviceContext )
{
    assert( deviceContext.poolIndex < deviceContextPool_.size() );
    freeDeviceContextList_.push_back( deviceContext.poolIndex );
}

void DemandTextureLoaderImpl::processRequests( hipStream_t stream, const DeviceContext& deviceContext )
{
    std::lock_guard<std::mutex> lock( mutex_ );

    assert( deviceContext.poolIndex < deviceContextPool_.size() );

    if( textures_.empty() )
        return;

    memcpyDtoH( requestedResources_, deviceContext.requestedResources, stream );
    memcpyDtoH( counters_, deviceContext.counters, stream );
    ProcessRequestCallback::enqueue( stream, new ProcessRequestCallback( this, deviceContext ) );
    HIP_CHECK( hipStreamSynchronize( stream ) );

    const uint32_t requestCount = counters_[static_cast<uint32_t>( CounterIndex::RequestedResources )];
    for( size_t i = 0; i < requestCount; i++ )
    {
        const uint32_t resourceId = requestedResources_.at( i );

        const Resource resource = decode( resourceId );
        switch( resource.type )
        {
            case Resource::Type::TextureId: {
                processTextureInfo( resource.textureId, stream, deviceContext );
                break;
            }
            case Resource::Type::Tile: {
                processTile( resource.tile, stream, deviceContext );
                break;
            }
            case Resource::Type::MipTail: {
                processMipTail( resource.mipTail, stream, deviceContext );
                break;
            }
            default: {
                throw std::logic_error( "Unhandled resource type: " + std::to_string( static_cast<uint32_t>( resource.type ) ) );
            }
        }

        residentBits_.set( resourceId, true );
        residentBitsDirty_.store(true, std::memory_order_release);
    }
}

void DemandTextureLoaderImpl::processRequestsCallback( const DeviceContext& deviceContext ) {
    freeDeviceContext( deviceContext );
}

void DemandTextureLoaderImpl::processTextureInfo( uint32_t textureId, hipStream_t stream, const DeviceContext& deviceContext )
{
    DemandTextureImpl& texture = *textures_.at( textureId );
    TextureInfo        hostInfo{};
    texture.image->open( &hostInfo );
    hostInfo.numMipLevels = std::min( hostInfo.numMipLevels, MAX_TEXTURE_MIP_LEVELS );
    assert( hostInfo.isValid );

    const uint32_t bytesPerChannel = getBytesPerChannel( hostInfo.format );
    // TODO_BS: handle images with width = 0, height = 0, mips = 0, unsupported format
    if( bytesPerChannel == 0 )
        throw std::invalid_argument( "Unsupported hipArray_Format" );
    if( hostInfo.width == 0 )
        throw std::runtime_error( "Unsupported width: 0" );
    if( hostInfo.height == 0 )
        throw std::runtime_error( "Unsupported height: 0" );
    if( hostInfo.numMipLevels == 0 )
        throw std::runtime_error( "Unsupported mip count: " + std::to_string( hostInfo.numMipLevels ) );

    const uint32_t bytesPerTexel = bytesPerChannel * hostInfo.numChannels;
    const uint2    tileShape     = tileShapeForGranularity( pageSystem_.granularity(), bytesPerTexel );

    std::array<size_t, MAX_TEXTURE_MIP_LEVELS> mipTailOffsets{};

    uint32_t mipTailFirstLevel = hostInfo.numMipLevels;
    size_t   tailBytes         = 0;
    for( uint32_t mipLevel = hostInfo.numMipLevels; mipLevel-- > 0; )
    {
        const uint2  dimensions = mipDimensions( make_uint2( hostInfo.width, hostInfo.height ), mipLevel );
        const size_t levelBytes = dimensions.x * dimensions.y * bytesPerTexel;

        if( levelBytes > pageSystem_.pageBytes() - tailBytes )
            break;

        tailBytes += levelBytes;
        mipTailFirstLevel = mipLevel;
    }

    size_t mipTailOffset = 0;
    for( uint32_t mipLevel = mipTailFirstLevel; mipLevel < hostInfo.numMipLevels; ++mipLevel )
    {
        const uint2  dimensions = mipDimensions( make_uint2( hostInfo.width, hostInfo.height ), mipLevel );
        const size_t levelBytes = dimensions.x * dimensions.y * bytesPerTexel;

        mipTailOffsets[mipLevel] = mipTailOffset;
        mipTailOffset += levelBytes;
    }
    size_t mipTailSize = mipTailOffset;

    DevicePtr<DeviceTextureInfo> dstInfo = textureInfoAllocator_.alloc();
    DeviceTextureInfo            deviceInfo{};
    deviceInfo.textureId        = textureId;
    deviceInfo.addressMode[0]   = texture.descriptor.addressMode[0];
    deviceInfo.addressMode[1]   = texture.descriptor.addressMode[1];
    deviceInfo.filterMode       = texture.descriptor.filterMode;
    deviceInfo.mipmapFilterMode = texture.descriptor.mipmapFilterMode;
    deviceInfo.normalizedCoords = texture.descriptor.normalizedCoords ? 1u : 0u;
    deviceInfo.format           = hostInfo.format;
    deviceInfo.numChannels      = hostInfo.numChannels;
    deviceInfo.tileWidth        = tileShape.x;
    deviceInfo.tileHeight       = tileShape.y;
    deviceInfo.bytesPerTexel    = bytesPerTexel;
    deviceInfo.width            = hostInfo.width;
    deviceInfo.height           = hostInfo.height;
    deviceInfo.startPage        = pageTable_.textureTiles.nextAvailablePage;

    deviceInfo.mipCount          = hostInfo.numMipLevels;
    deviceInfo.mipTailFirstLevel = mipTailFirstLevel;
    deviceInfo.mipTailSize       = static_cast<uint32_t>( mipTailSize );

    uint32_t pageCount = 0;
    for( uint32_t mip = 0; mip < deviceInfo.mipTailFirstLevel; ++mip )
    {
        auto& level     = deviceInfo.mips[mip];
        level.startPage = pageTable_.textureTiles.nextAvailablePage + pageCount;
        // we can call getMipLevel after we init deviceInfo.mips[mip]
        pageCount += deviceInfo.getMipLevel( mip ).pageCount();
    }

    if( deviceInfo.mipTailFirstLevel < deviceInfo.mipCount )
    {
        deviceInfo.mipTailPage = pageTable_.textureTiles.nextAvailablePage + pageCount;
        pageCount++;

        for( uint32_t mip = deviceInfo.mipTailFirstLevel; mip < deviceInfo.mipCount; ++mip )
        {
            auto& level         = deviceInfo.mips[mip];
            level.startPage     = deviceInfo.mipTailPage;
            level.mipTailOffset = static_cast<uint32_t>( mipTailOffsets[mip] );
        }
    }

    // TODO_BS: how to handle it?
    if( pageTable_.textureTiles.nextAvailablePage + pageCount > options_.maxVirtualPages )
        throw std::runtime_error( "Maximum demand virtual page count exceeded" );

    texture.loadedTextureInfoId = static_cast<uint32_t>( loadedTextureInfos_.size() );
    loadedTextureInfos_.push_back( deviceInfo );

    pageTable_.textureTiles.nextAvailablePage += pageCount;

    HIP_CHECK( hipMemcpyHtoDAsync( dstInfo, &deviceInfo, sizeof( deviceInfo ), stream ) );
    HIP_CHECK( hipMemcpyHtoDAsync( deviceContext.textureInfos.ptr + textureId, &dstInfo, sizeof( dstInfo ), stream ) );
    HIP_CHECK( hipStreamSynchronize( stream ) );
}

void DemandTextureLoaderImpl::processTile( const Resource::Tile& tile, hipStream_t stream, const DeviceContext& deviceContext )
{
    const DemandTextureImpl& texture = *textures_.at( tile.textureId );
    const DeviceTextureInfo& info    = loadedTextureInfos_.at( texture.loadedTextureInfoId );

    if( !pageSystem_.mapped( tile.pageId ) )
        pageSystem_.map( tile.pageId );

    Tile t{};
    t.x      = tile.tileX;
    t.y      = tile.tileY;
    t.width  = info.tileWidth;
    t.height = info.tileHeight;
    assert( static_cast<size_t>( t.width ) * t.height * info.bytesPerTexel <= sizeInBytes( tmpPageBuffer_ ) );
    if( !texture.image->readTile( reinterpret_cast<char*>( tmpPageBuffer_.data() ), tile.mipLevel, t, stream ) )
    {
        throw std::runtime_error( "Failed to read texture tile for texture " + std::to_string( tile.textureId )
                                  + ", mip " + std::to_string( tile.mipLevel ) + ", tile ("
                                  + std::to_string( tile.tileX ) + ", " + std::to_string( tile.tileY ) + ")" );
    }

    memcpyHtoD( pageSystem_.page( tile.pageId ), tmpPageBuffer_, stream );
    HIP_CHECK( hipStreamSynchronize( stream ) );
}

void DemandTextureLoaderImpl::processMipTail( const Resource::MipTail& mipTail, hipStream_t stream, const DeviceContext& deviceContext )
{
    const DemandTextureImpl& texture = *textures_.at( mipTail.textureId );
    const DeviceTextureInfo& info    = loadedTextureInfos_.at( texture.loadedTextureInfoId );
    assert( info.mipTailFirstLevel < info.mipCount );

    if( !pageSystem_.mapped( mipTail.pageId ) )
        pageSystem_.map( mipTail.pageId );

    std::fill( tmpPageBuffer_.begin(), tmpPageBuffer_.end(), 0 );
    for( uint32_t mipLevel = info.mipTailFirstLevel; mipLevel < info.mipCount; ++mipLevel )
    {
        const DeviceMipLevel mip = info.getMipLevel( mipLevel );
        char*                dst = reinterpret_cast<char*>( tmpPageBuffer_.data() + mip.mipTailOffset );
        if( !texture.image->readMipLevel( dst, mipLevel, mip.width, mip.height, stream ) )
        {
            throw std::runtime_error( "Failed to read mip tail level " + std::to_string( mipLevel ) + " for texture "
                                      + std::to_string( mipTail.textureId ) );
        }
    }

    memcpyHtoD( pageSystem_.page( mipTail.pageId ), tmpPageBuffer_, stream );
    HIP_CHECK( hipStreamSynchronize( stream ) );
}

Resource DemandTextureLoaderImpl::decode( uint32_t resourceId )
{
    if( resourceId < options_.maxTextures )
    {
        Resource resource{};
        resource.type      = Resource::Type::TextureId;
        resource.textureId = pageTable_.getTextureIdByResourceId( resourceId );
        return resource;
    }
    else
    {
        const uint32_t pageId = pageTable_.getTextureTilePageByResourceId( resourceId );
        const auto     it =
            std::upper_bound( loadedTextureInfos_.cbegin(), loadedTextureInfos_.cend(), pageId,
                              []( uint32_t page, const DeviceTextureInfo& info ) { return page < info.startPage; } );

        if( it == loadedTextureInfos_.begin() )
            throw std::out_of_range( "Cannot decode resourceId " + std::to_string( resourceId )
                                     + ": it does not belong to any registered texture" );

        const auto infoIt = std::prev( it );

        const DeviceTextureInfo& info = *infoIt;
        if( pageId == info.mipTailPage )
        {
            Resource resource{};
            resource.type              = Resource::Type::MipTail;
            resource.mipTail.textureId = info.textureId;
            resource.mipTail.pageId    = pageId;
            return resource;
        }

        for( uint32_t mipLevel = 0; mipLevel < info.mipTailFirstLevel; ++mipLevel )
        {
            const DeviceMipLevel level = info.getMipLevel( mipLevel );
            if( level.startPage <= pageId && pageId < level.startPage + level.pageCount() )
            {
                const uint32_t pageInLevel = pageId - level.startPage;
                const uint32_t tileX       = pageInLevel % level.tilesX;
                const uint32_t tileY       = pageInLevel / level.tilesX;

                Resource resource{};
                resource.type           = Resource::Type::Tile;
                resource.tile.textureId = info.textureId;
                resource.tile.mipLevel  = mipLevel;
                resource.tile.tileX     = tileX;
                resource.tile.tileY     = tileY;
                resource.tile.pageId    = pageId;
                return resource;
            }
        }

        // we should never be here!
        throw std::out_of_range( "Cannot decode resourceId " + std::to_string( resourceId )
                                 + ": it is outside the mip ranges of texture " + std::to_string( info.textureId ) );
    }
}

std::unique_ptr<DemandTextureLoader> createDemandTextureLoader( const Options& options )
{
    return std::make_unique<DemandTextureLoaderImpl>( options );
}


}  // namespace hip_demand::vmm
