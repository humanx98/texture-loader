#include "DemandTextureLoaderImpl.h"
#include <ImageSource/TextureInfo.h>

namespace hip_demand::vmm {

class HipCallback
{
  public:
    virtual ~HipCallback() = default;
    static void enqueue( hipStream_t stream, HipCallback* callback )
    {
        HIP_CHECK( hipLaunchHostFunc( stream, &staticCallback, callback ) );
    }

    static void staticCallback( void* arg )
    {
        std::unique_ptr<HipCallback> callback( static_cast<HipCallback*>( arg ) );
        callback->callback();
    }

  protected:
    virtual void callback() = 0;
};

class ProcessRequestCallback : public HipCallback
{
  public:
    explicit ProcessRequestCallback( DemandTextureLoaderImpl& loader, DeviceContext deviceContext, Ticket ticket )
        : loader_( loader )
        , deviceContext_( deviceContext )
        , ticket_( std::move( ticket ) )
    {
    }

  protected:
    void callback() { loader_.processRequestsCallback( deviceContext_, std::move( ticket_ ) ); }

  private:
    DemandTextureLoaderImpl& loader_;
    Ticket                   ticket_;
    DeviceContext            deviceContext_;
};

struct DeviceContextImpl : DeviceContext
{
  public:
    uint32_t* h_requestedResources = nullptr;
    uint32_t* h_counters           = nullptr;
};

static PageTable createPageTable( const Options& options, size_t pageSize )
{
    PageTable pageTable{};
    pageTable.pageSize    = pageSize;
    pageTable.maxTextures = options.maxTextures;

    const size_t   maxTextureInfoSize  = pageTable.maxTextures * sizeof( DeviceTextureInfo );
    const uint32_t maxTextureInfoPages = static_cast<uint32_t>( ceilDiv( maxTextureInfoSize, pageTable.pageSize ) );
    if( maxTextureInfoPages > options.maxVirtualPages )
        throw std::invalid_argument( "maxVirtualPages is too small to store texture metadata" );
    const uint32_t maxTextureTilePages = options.maxVirtualPages - maxTextureInfoPages;

    pageTable.textureInfos = PageTable::Range( 0, maxTextureInfoPages );
    pageTable.textureTiles = PageTable::Range( pageTable.textureInfos.pageCount, maxTextureTilePages );
    return pageTable;
}

static uint32_t getResourceCount( const PageTable& pageTable )
{
    uint32_t resourceCount = 0;
    if( !safeAdd( pageTable.maxTextures, pageTable.textureTiles.pageCount, resourceCount ) )
        throw std::overflow_error(
            "Cannot create demand texture loader: total resource count exceeds the uint32_t limit" );
    return resourceCount;
}

DemandTextureLoaderImpl::DemandTextureLoaderImpl( const Options& options )
    : options_( options )
    , eventPool_( 10 )
    , pageSystem_( options_.maxVirtualPages, options_.maxPhysicalPages )
    , pageBufferPinnedAllocator_( pageSystem_.pageBytes(), eventPool_ )
    , textureInfoAllocator_( pageSystem_ )
    , textureInfoPinnedAllocator_( eventPool_ )
    , ptrTextureInfoPinnedAllocator_( eventPool_ )
    , pageTable_( createPageTable( options, pageSystem_.pageBytes() ) )
    , requestProcessor_( this, eventPool_, getResourceCount( pageTable_ ), options.maxThreads, options_.maxRequestQueue )
{
    if( options_.maxRequests == 0 )
        throw std::invalid_argument( "maxRequests cannot be 0" );

    textureInfoAllocator_.setRange( pageTable_.textureInfos );
}

DemandTextureLoaderImpl::~DemandTextureLoaderImpl()
{
    requestProcessor_.stop();

    if( !inFlight_.empty() )
    {

        freeArray( residentBits_ );
        freeArray( textureInfos_ );
        for( auto& f : inFlight_ )
        {
            freeArray( f.deviceContext.requestedBits );
            freeArray( f.deviceContext.requestedResources );
            freeArray( f.deviceContext.counters );

            hostFreeArray( f.requestedResources );
            hostFreeArray( f.counters );
        }
    }

    if( transferStream_ )
    {
        HIP_WARN( hipStreamDestroy( transferStream_ ) );
        transferStream_ = nullptr;
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
    {
        std::scoped_lock lock( mutex_, metadataMutex_ );

        initDeviceContext( deviceContext, stream );
        if( textures_.empty() )
            return;

        deviceContext.textureInfos.len = static_cast<uint32_t>( textures_.size() );
    }

    memset( deviceContext.requestedBits, 0, stream );
    memset( deviceContext.counters, 0, stream );
    requestProcessor_.uploadResidentBits( deviceContext.residentBits, stream );
}

void DemandTextureLoaderImpl::initDeviceContext( DeviceContext& deviceContext, hipStream_t stream )
{
    if( !freeDeviceContextList_.empty() )
    {
        deviceContext = inFlight_.at( freeDeviceContextList_.back() ).deviceContext;
        freeDeviceContextList_.pop_back();
        return;
    }

    if( inFlight_.empty() )
    {
        residentBits_ = allocArray<uint32_t>( requestProcessor_.residentWordCount(), true, stream );
        textureInfos_ = allocArray<DeviceTextureInfo*>( options_.maxTextures, true, stream );
    }

    inFlight_.push_back( {} );
    InFlight& flight = inFlight_.back();

    // set data per stream
    flight.deviceContext.poolIndex     = inFlight_.size() - 1;
    flight.deviceContext.requestedBits = allocArray<uint32_t>( requestProcessor_.residentWordCount(), true, stream );
    flight.deviceContext.requestedResources = allocArray<uint32_t>( options_.maxRequests, true, stream );
    flight.deviceContext.counters = allocArray<uint32_t>( static_cast<size_t>( CounterIndex::NumCounters ), true, stream );

    flight.requestedResources = hostAllocArray<uint32_t>( flight.deviceContext.requestedResources.len );
    flight.counters           = hostAllocArray<uint32_t>( flight.deviceContext.counters.len );

    // set data per loader
    flight.deviceContext.pageTable    = pageTable_;
    flight.deviceContext.pageMemory   = pageSystem_.virtualAddressSpace();
    flight.deviceContext.residentBits = residentBits_;
    flight.deviceContext.textureInfos = textureInfos_;

    deviceContext = flight.deviceContext;
}

void DemandTextureLoaderImpl::freeDeviceContext( DeviceContext& deviceContext, bool needLock )
{
    std::unique_lock<std::mutex> lock{ mutex_, std::defer_lock };

    if( needLock )
        lock.lock();

    assert( deviceContext.poolIndex < inFlight_.size() );
    freeDeviceContextList_.push_back( deviceContext.poolIndex );
}

Ticket DemandTextureLoaderImpl::processRequests( hipStream_t stream, const DeviceContext& deviceContext )
{
    std::lock_guard<std::mutex> lock( mutex_ );
    assert( deviceContext.poolIndex < inFlight_.size() );

    Ticket ticket = TicketImpl::create( stream );

    if( textures_.empty() )
    {
        TicketImpl::getImpl( ticket )->initialize( 0 );
        DeviceContext freeContext = deviceContext;
        freeDeviceContext( freeContext, false );
        return ticket;
    }

    InFlight& flight = inFlight_.at( deviceContext.poolIndex );
    memcpyDtoH( flight.requestedResources, deviceContext.requestedResources, stream );
    memcpyDtoH( flight.counters, deviceContext.counters, stream );
    ProcessRequestCallback::enqueue( stream, new ProcessRequestCallback( *this, deviceContext, ticket ) );
    return ticket;
}

void DemandTextureLoaderImpl::processRequestsCallback( DeviceContext& deviceContext, Ticket ticket )
{
    std::lock_guard<std::mutex> lock( mutex_ );

    InFlight& flight       = inFlight_.at( deviceContext.poolIndex );
    uint32_t  requestCount = flight.counters.ptr[static_cast<uint32_t>( CounterIndex::RequestedResources )];
    requestProcessor_.submit( flight.requestedResources.ptr, requestCount, std::move( ticket ) );
    freeDeviceContext( deviceContext, false );
}

void DemandTextureLoaderImpl::processRequest( hipStream_t stream, uint32_t resourceId )
{
    const Resource resource = decode( resourceId );
    switch( resource.type )
    {
        case Resource::Type::TextureId: {
            processTextureInfo( stream, resource.textureId );
            break;
        }
        case Resource::Type::Tile: {
            processTile( stream, resource.tile );
            break;
        }
        case Resource::Type::MipTail: {
            processMipTail( stream, resource.mipTail );
            break;
        }
        default: {
            throw std::logic_error( "Unhandled resource type: " + std::to_string( static_cast<uint32_t>( resource.type ) ) );
        }
    }
}

void DemandTextureLoaderImpl::processTextureInfo( hipStream_t stream, uint32_t textureId )
{
    DemandTextureImpl* texture = nullptr;
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        texture = textures_.at( textureId ).get();
    }
    TextureInfo imageInfo{};
    texture->image->open( &imageInfo );
    imageInfo.numMipLevels = std::min( imageInfo.numMipLevels, MAX_TEXTURE_MIP_LEVELS );
    assert( imageInfo.isValid );

    const uint32_t bytesPerChannel = getBytesPerChannel( imageInfo.format );
    // TODO_BS: handle images with width = 0, height = 0, mips = 0, unsupported format
    if( bytesPerChannel == 0 )
        throw std::invalid_argument( "Unsupported hipArray_Format" );
    if( imageInfo.width == 0 )
        throw std::runtime_error( "Unsupported width: 0" );
    if( imageInfo.height == 0 )
        throw std::runtime_error( "Unsupported height: 0" );
    if( imageInfo.numMipLevels == 0 )
        throw std::runtime_error( "Unsupported mip count: " + std::to_string( imageInfo.numMipLevels ) );
    if( imageInfo.numChannels == 0 )
        throw std::runtime_error( "Unsupported channel count: 0" );

    const uint32_t bytesPerTexel = bytesPerChannel * imageInfo.numChannels;
    const uint2    tileShape     = tileShapeForGranularity( pageSystem_.granularity(), bytesPerTexel );

    std::array<size_t, MAX_TEXTURE_MIP_LEVELS> mipTailOffsets{};

    uint32_t mipTailFirstLevel = imageInfo.numMipLevels;
    size_t   tailBytes         = 0;
    for( uint32_t mipLevel = imageInfo.numMipLevels; mipLevel-- > 0; )
    {
        const uint2  dimensions = mipDimensions( make_uint2( imageInfo.width, imageInfo.height ), mipLevel );
        const size_t levelBytes = static_cast<size_t>( bytesPerTexel ) * dimensions.x * dimensions.y;

        if( levelBytes > pageSystem_.pageBytes() - tailBytes )
            break;

        tailBytes += levelBytes;
        mipTailFirstLevel = mipLevel;
    }

    size_t mipTailOffset = 0;
    for( uint32_t mipLevel = mipTailFirstLevel; mipLevel < imageInfo.numMipLevels; ++mipLevel )
    {
        const uint2  dimensions = mipDimensions( make_uint2( imageInfo.width, imageInfo.height ), mipLevel );
        const size_t levelBytes = static_cast<size_t>( bytesPerTexel ) * dimensions.x * dimensions.y;

        mipTailOffsets[mipLevel] = mipTailOffset;
        mipTailOffset += levelBytes;
    }
    size_t mipTailSize = mipTailOffset;

    auto h_info              = textureInfoPinnedAllocator_.allocScoped( stream );
    h_info->textureId        = textureId;
    h_info->addressMode[0]   = texture->descriptor.addressMode[0];
    h_info->addressMode[1]   = texture->descriptor.addressMode[1];
    h_info->filterMode       = texture->descriptor.filterMode;
    h_info->mipmapFilterMode = texture->descriptor.mipmapFilterMode;
    h_info->normalizedCoords = texture->descriptor.normalizedCoords ? 1u : 0u;
    h_info->format           = imageInfo.format;
    h_info->numChannels      = imageInfo.numChannels;
    h_info->tileWidth        = tileShape.x;
    h_info->tileHeight       = tileShape.y;
    h_info->bytesPerTexel    = bytesPerTexel;
    h_info->width            = imageInfo.width;
    h_info->height           = imageInfo.height;

    h_info->mipCount          = imageInfo.numMipLevels;
    h_info->mipTailFirstLevel = mipTailFirstLevel;
    h_info->mipTailSize       = static_cast<uint32_t>( mipTailSize );

    {
        std::lock_guard<std::mutex> lock( metadataMutex_ );

        h_info->startPage = pageTable_.textureTiles.nextAvailablePage;

        uint32_t pageCount = 0;
        for( uint32_t mip = 0; mip < h_info->mipTailFirstLevel; ++mip )
        {
            auto& level     = h_info->mips[mip];
            level.startPage = pageTable_.textureTiles.nextAvailablePage + pageCount;
            // we can call getMipLevel after we init pinnedDeviceInfo->mips[mip]
            pageCount += h_info->getMipLevel( mip ).pageCount();
        }

        if( h_info->mipTailFirstLevel < h_info->mipCount )
        {
            h_info->mipTailPage = pageTable_.textureTiles.nextAvailablePage + pageCount;
            pageCount++;

            for( uint32_t mip = h_info->mipTailFirstLevel; mip < h_info->mipCount; ++mip )
            {
                auto& level         = h_info->mips[mip];
                level.startPage     = h_info->mipTailPage;
                level.mipTailOffset = static_cast<uint32_t>( mipTailOffsets[mip] );
            }
        }

        // TODO_BS: how to handle it?
        if( pageTable_.textureTiles.nextAvailablePage + pageCount > options_.maxVirtualPages )
            throw std::runtime_error( "Maximum demand virtual page count exceeded" );

        texture->loadedTextureInfoId = static_cast<uint32_t>( loadedTextureInfos_.size() );
        loadedTextureInfos_.push_back( *h_info );

        pageTable_.textureTiles.nextAvailablePage += pageCount;

        // copy struct pinned memory to device memory
        DevicePtr<DeviceTextureInfo> d_info = textureInfoAllocator_.alloc();
        memcpyHtoD( DeviceSpan<DeviceTextureInfo>( d_info, 1 ), HostSpan<DeviceTextureInfo>( h_info.get(), 1 ), stream );

        // copy device pointer to deivce pointers
        auto stagedTextureInfoPointer = ptrTextureInfoPinnedAllocator_.allocScoped( stream );
        *stagedTextureInfoPointer     = d_info;
        memcpyHtoD( DeviceSpan<DeviceTextureInfo*>( textureInfos_.ptr + textureId, 1 ), stagedTextureInfoPointer.span(), stream );
    }
}

void DemandTextureLoaderImpl::processTile( hipStream_t stream, const Resource::Tile& tile )
{
    std::shared_ptr<ImageSource> image;
    DeviceTextureInfo            info{};
    {
        std::scoped_lock         lock( mutex_, metadataMutex_ );
        const DemandTextureImpl& texture = *textures_.at( tile.textureId );
        image                            = texture.image;
        info                             = loadedTextureInfos_.at( texture.loadedTextureInfoId );
    }

    Tile t{};
    t.x      = tile.tileX;
    t.y      = tile.tileY;
    t.width  = info.tileWidth;
    t.height = info.tileHeight;
    assert( static_cast<size_t>( t.width ) * t.height * info.bytesPerTexel <= pageBufferPinnedAllocator_.bufferSize() );
    auto pageBuffer = pageBufferPinnedAllocator_.allocScoped( stream );
    if( !image->readTile( reinterpret_cast<char*>( pageBuffer.get() ), tile.mipLevel, t, stream ) )
    {
        throw std::runtime_error( "Failed to read texture tile for texture " + std::to_string( tile.textureId )
                                  + ", mip " + std::to_string( tile.mipLevel ) + ", tile ("
                                  + std::to_string( tile.tileX ) + ", " + std::to_string( tile.tileY ) + ")" );
    }

    if( !pageSystem_.mapped( tile.pageId ) )
        pageSystem_.map( tile.pageId );

    memcpyHtoD( pageSystem_.page( tile.pageId ), pageBuffer.span(), stream );
}

void DemandTextureLoaderImpl::processMipTail( hipStream_t stream, const Resource::MipTail& mipTail )
{
    std::shared_ptr<ImageSource> image;
    DeviceTextureInfo            info{};
    {
        std::scoped_lock         lock( mutex_, metadataMutex_ );
        const DemandTextureImpl& texture = *textures_.at( mipTail.textureId );
        image                            = texture.image;
        info                             = loadedTextureInfos_.at( texture.loadedTextureInfoId );
    }

    assert( info.mipTailFirstLevel < info.mipCount );

    auto pageBuffer = pageBufferPinnedAllocator_.allocScoped( stream );
    std::memset( pageBuffer.get(), 0, pageBufferPinnedAllocator_.bufferSize() );
    for( uint32_t mipLevel = info.mipTailFirstLevel; mipLevel < info.mipCount; ++mipLevel )
    {
        const DeviceMipLevel mip = info.getMipLevel( mipLevel );
        char*                dst = reinterpret_cast<char*>( pageBuffer.get() + mip.mipTailOffset );
        if( !image->readMipLevel( dst, mipLevel, mip.width, mip.height, stream ) )
        {
            throw std::runtime_error( "Failed to read mip tail level " + std::to_string( mipLevel ) + " for texture "
                                      + std::to_string( mipTail.textureId ) );
        }
    }

    if( !pageSystem_.mapped( mipTail.pageId ) )
        pageSystem_.map( mipTail.pageId );

    memcpyHtoD( pageSystem_.page( mipTail.pageId ), pageBuffer.span(), stream );
}

Resource DemandTextureLoaderImpl::decode( uint32_t resourceId )
{
    std::lock_guard<std::mutex> lock( metadataMutex_ );

    if( resourceId < options_.maxTextures )
    {
        return Resource( pageTable_.getTextureIdByResourceId( resourceId ) );
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
            return Resource( Resource::MipTail{ info.textureId, pageId } );

        for( uint32_t mipLevel = 0; mipLevel < info.mipTailFirstLevel; ++mipLevel )
        {
            const DeviceMipLevel level = info.getMipLevel( mipLevel );
            if( level.startPage <= pageId && pageId < level.startPage + level.pageCount() )
            {
                const uint32_t pageInLevel = pageId - level.startPage;
                const uint32_t tileX       = pageInLevel % level.tilesX;
                const uint32_t tileY       = pageInLevel / level.tilesX;

                return Resource( Resource::Tile{ info.textureId, mipLevel, tileX, tileY, pageId } );
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
