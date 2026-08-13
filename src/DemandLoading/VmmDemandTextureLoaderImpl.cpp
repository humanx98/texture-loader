#include "Internal/HipCheck.h"
#include "Internal/Utils.h"
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <ImageSource/TextureInfo.h>
#include <algorithm>
#include <array>
#include <mutex>

namespace hip_demand::vmm {

using internal::Bitset;
using internal::calculateMipLevels;
using internal::ceilDiv;
using internal::memcpyDtoHAsync;
using internal::memcpyHtoDAsync;
using internal::memset;
using internal::memsetAsync;
using internal::mipDimension;
using internal::NonCopyble;
using internal::safeAdd;
using internal::tileShapeForGranularity;
using internal::sizeInBytes;

class HipGC : NonCopyble
{
  public:
    template <typename T>
    DeviceSpan<T> allocArray( size_t count, bool zero = false )
    {
        const size_t size = count * sizeof( T );
        if( size == 0 )
            return {};

        T* ptr = nullptr;
        HIP_CHECK( hipMalloc( &ptr, size ) );
        allocations_.push_back( ptr );

        DeviceSpan<T> result( ptr, count );
        if( zero )
            memset( result, 0 );

        return result;
    }

    ~HipGC()
    {
        for( hipDeviceptr_t a : allocations_ )
            HIP_WARN( hipFree( a ) );
    }

  private:
    std::vector<hipDeviceptr_t> allocations_;
};

inline uint32_t pixelSize( hipArray_Format format, uint32_t numChannels )
{
    if( numChannels == 0 || numChannels > 4 )
        throw std::invalid_argument( "Texture channel count must be between 1 and 4" );

    const uint32_t bytesPerChannel = getBytesPerChannel( format );
    if( bytesPerChannel == 0 )
        throw std::invalid_argument( "Unsupported hipArray_Format" );

    return bytesPerChannel * numChannels;
}

struct ResourceTile
{
    uint32_t textureId = 0;
    uint32_t mipLevel  = 0;
    uint32_t tileX     = 0;
    uint32_t tileY     = 0;
    uint32_t pageId    = 0;
};

enum class ResourceType
{
    TextureInfo,
    TextureTile
};

struct Resource
{
    ResourceType type;
    union
    {
        struct
        {
            uint32_t textureId;
        } textureInfo;
        ResourceTile tile;
    };

    static Resource TextureInfo( uint32_t textureId )
    {
        Resource resource{};
        resource.type                  = ResourceType::TextureInfo;
        resource.textureInfo.textureId = textureId;
        return resource;
    }

    static Resource TextureTile( uint32_t pageId, uint32_t textureId, uint32_t mipLevel, uint32_t tileX, uint32_t tileY )
    {
        Resource resource{};
        resource.type           = ResourceType::TextureTile;
        resource.tile.textureId = textureId;
        resource.tile.mipLevel  = mipLevel;
        resource.tile.tileX     = tileX;
        resource.tile.tileY     = tileY;
        resource.tile.pageId    = pageId;
        return resource;
    }

  private:
    Resource() {}
};

struct VmmPhysicalPage
{
    hipMemGenericAllocationHandle_t handle        = nullptr;
    uint32_t                        virtualPageId = INVALID_PAGE;
};

class VmmPageSystem : NonCopyble
{
  public:
    explicit VmmPageSystem( uint32_t maxVirtualPages, uint32_t maxPhysicalPages );
    ~VmmPageSystem();
    void                map( uint32_t pageId );
    void                unmap( uint32_t pageId );
    bool                mapped( uint32_t pageId ) const { return virtualIdToPhysicalId_.at( pageId ) != INVALID_PAGE; }
    DeviceSpan<uint8_t> page( uint32_t pageId ) const
    {
        return DeviceSpan<uint8_t>( virtualAddressSpace_.ptr + pageId * pageBytes_, pageBytes_ );
    }
    DeviceSpan<uint8_t> virtualAddressSpace() const { return virtualAddressSpace_; }
    size_t              granularity() const { return granularity_; }
    size_t              pageBytes() const { return pageBytes_; }

  private:
    uint32_t                     maxVirtualPages_  = 0;
    uint32_t                     maxPhysicalPages_ = 0;
    int                          device_           = 0;
    size_t                       granularity_      = 0;
    size_t                       pageBytes_        = 0;
    hipMemAllocationProp         allocationProp_{};
    std::vector<uint32_t>        virtualIdToPhysicalId_{};
    std::vector<VmmPhysicalPage> physicalPages_{};
    std::vector<uint32_t>        freePhysicalPages_{};
    DeviceSpan<uint8_t>          virtualAddressSpace_{};
};

VmmPageSystem::VmmPageSystem( uint32_t maxVirtualPages, uint32_t maxPhysicalPages )
    : maxVirtualPages_( maxVirtualPages )
    , maxPhysicalPages_( maxPhysicalPages )
{
    HIP_CHECK( hipGetDevice( &device_ ) );
    int vmmSupported = 0;
    HIP_CHECK( hipDeviceGetAttribute( &vmmSupported, hipDeviceAttributeVirtualMemoryManagementSupported, device_ ) );
    if( vmmSupported == 0 )
        throw std::runtime_error( "The active HIP device does not support virtual memory management" );

    allocationProp_.type          = hipMemAllocationTypePinned;
    allocationProp_.location.type = hipMemLocationTypeDevice;
    allocationProp_.location.id   = device_;
    HIP_CHECK( hipMemGetAllocationGranularity( &granularity_, &allocationProp_, hipMemAllocationGranularityMinimum ) );
    if( granularity_ == 0 )
        throw std::runtime_error( "hipMemAllocationGranularityMinimum is zero" );

    pageBytes_               = granularity_;
    virtualAddressSpace_.len = maxVirtualPages_ * pageBytes_;
    HIP_CHECK( hipMemAddressReserve( reinterpret_cast<void**>( &virtualAddressSpace_.ptr ), virtualAddressSpace_.len,
                                     granularity_, nullptr, 0 ) );

    virtualIdToPhysicalId_.assign( maxVirtualPages_, INVALID_PAGE );
}

VmmPageSystem::~VmmPageSystem()
{
    for( VmmPhysicalPage& physicalPage : physicalPages_ )
    {
        if( physicalPage.virtualPageId != INVALID_PAGE )
        {
            DeviceSpan<uint8_t> virtualPage = page( physicalPage.virtualPageId );
            HIP_WARN( hipMemUnmap( virtualPage.ptr, virtualPage.len ) );
        }

        HIP_WARN( hipMemRelease( physicalPage.handle ) );
    }

    if( virtualAddressSpace_.ptr )
        HIP_WARN( hipMemAddressFree( virtualAddressSpace_.ptr, virtualAddressSpace_.len ) );
}

void VmmPageSystem::map( uint32_t pageId )
{
    if( mapped( pageId ) )
        return;

    uint32_t physicalPageId = INVALID_PAGE;
    if( freePhysicalPages_.empty() )
    {
        // TODO_BS: handle max physical pages
        if( physicalPages_.size() >= maxPhysicalPages_ )
            throw std::runtime_error( "Maximum demand physical page count exceeded" );

        physicalPageId                         = static_cast<uint32_t>( physicalPages_.size() );
        hipMemGenericAllocationHandle_t handle = nullptr;
        HIP_CHECK( hipMemCreate( &handle, pageBytes_, &allocationProp_, 0 ) );
        physicalPages_.push_back( VmmPhysicalPage{ handle, INVALID_PAGE } );
    }
    else
    {
        physicalPageId = freePhysicalPages_.back();
        freePhysicalPages_.pop_back();
    }

    VmmPhysicalPage&    physicalPage = physicalPages_.at( physicalPageId );
    DeviceSpan<uint8_t> virtualPage  = page( pageId );
    HIP_CHECK( hipMemMap( virtualPage.ptr, virtualPage.len, 0, physicalPage.handle, 0 ) );

    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id   = device_;
    access.flags         = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK( hipMemSetAccess( virtualPage.ptr, virtualPage.len, &access, 1 ) );

    virtualIdToPhysicalId_.at( pageId ) = physicalPageId;
    physicalPage.virtualPageId          = pageId;
}

void VmmPageSystem::unmap( uint32_t pageId )
{
    if( !mapped( pageId ) )
        return;

    const uint32_t            physicalPageId = virtualIdToPhysicalId_.at( pageId );
    const DeviceSpan<uint8_t> virtualPage    = page( pageId );
    HIP_CHECK( hipMemUnmap( virtualPage.ptr, virtualPage.len ) );
    virtualIdToPhysicalId_.at( pageId )               = INVALID_PAGE;
    physicalPages_.at( physicalPageId ).virtualPageId = INVALID_PAGE;
    freePhysicalPages_.push_back( physicalPageId );
}

template <typename T>
class VmmAllocator : NonCopyble
{
  public:
    explicit VmmAllocator( VmmPageSystem& pageSystem );

    DevicePtr<T> alloc() { return reinterpret_cast<DevicePtr<T>>( alloc( sizeof( T ) ) ); }

    void setRange( const PageTable::Range& pageRange )
    {
        pageRange_          = pageRange;
        availablePageBytes_ = pageSystem_.pageBytes();
    }

  private:
    hipDeviceptr_t alloc( size_t size );

    VmmPageSystem&   pageSystem_;
    PageTable::Range pageRange_{};
    size_t           availablePageBytes_ = 0;
};

template <typename T>
VmmAllocator<T>::VmmAllocator( VmmPageSystem& pageSystem )
    : pageSystem_( pageSystem )
{
}

template <typename T>
hipDeviceptr_t VmmAllocator<T>::alloc( size_t size )
{
    assert( size > 0 );

    if( pageRange_.nextAvailablePage < pageRange_.startPage )
        throw std::logic_error( "VmmAllocator page range has an invalid next available page" );

    const uint32_t usedPages = pageRange_.nextAvailablePage - pageRange_.startPage;
    if( usedPages >= pageRange_.pageCount )
        throw std::bad_alloc{};

    const size_t   pageBytes      = pageSystem_.pageBytes();
    const uint32_t remainingPages = pageRange_.pageCount - usedPages;

    size_t bytesAfterCurrentPage = 0;
    size_t additionalPages       = 0;
    if( size > availablePageBytes_ )
    {
        bytesAfterCurrentPage = size - availablePageBytes_;
        additionalPages       = 1 + ( bytesAfterCurrentPage - 1 ) / pageBytes;
    }

    if( additionalPages > remainingPages - 1 )
        throw std::bad_alloc{};

    for( size_t pageOffset = 0; pageOffset < additionalPages + 1; ++pageOffset )
    {
        const uint32_t page = static_cast<uint32_t>( pageRange_.nextAvailablePage + pageOffset );
        if( !pageSystem_.mapped( page ) )
            pageSystem_.map( page );
    }

    hipDeviceptr_t result = pageSystem_.page( pageRange_.nextAvailablePage ).ptr + pageBytes - availablePageBytes_;

    if( size < availablePageBytes_ )
    {
        availablePageBytes_ -= size;
    }
    else
    {
        const uint32_t pagesAdvanced = static_cast<uint32_t>( 1 + bytesAfterCurrentPage / pageBytes );
        pageRange_.nextAvailablePage += pagesAdvanced;
        availablePageBytes_ = pageBytes - ( bytesAfterCurrentPage % pageBytes );
    }

    return result;
}

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

  private:
    void     initPageTable( uint32_t& resourceCount );
    Resource decode( uint32_t resourceId );
    void     processTextureInfo( uint32_t textureId, hipStream_t stream, const DeviceContext& deviceContext );
    void     processTextureTile( const ResourceTile& tile, hipStream_t stream, const DeviceContext& deviceContext );

    mutable std::mutex    mutex_;
    Options               options_{};
    bool                  residentBitsDirty_ = false;
    Bitset                residentBits_{};
    std::vector<uint32_t> requestedResources_{};
    std::vector<uint8_t>  tmpPageBuffer_{};
    VmmPageSystem         pageSystem_;
    PageTable             pageTable_{};

    std::array<uint32_t, static_cast<size_t>( CounterIndex::NumCounters )> counters_{};

    std::vector<std::unique_ptr<DemandTextureImpl>> textures_{};
    // note that this list should be accessed by texture.loadedTextureInfoId
    // and it's ordered by startPage in order to use std::upper_bound
    std::vector<DeviceTextureInfo>  loadedTextureInfos_{};
    VmmAllocator<DeviceTextureInfo> textureInfoAllocator_;

    HipGC         hipGC_{};
    DeviceContext deviceContext_{};
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

    deviceContext_.pageMemory         = pageSystem_.virtualAddressSpace();
    deviceContext_.requestedBits      = hipGC_.allocArray<uint32_t>( residentBits_.wordCount(), true );
    deviceContext_.residentBits       = hipGC_.allocArray<uint32_t>( residentBits_.wordCount(), true );
    deviceContext_.requestedResources = hipGC_.allocArray<uint32_t>( requestedResources_.size(), true );
    deviceContext_.counters           = hipGC_.allocArray<uint32_t>( counters_.size(), true );
    deviceContext_.textureInfos       = hipGC_.allocArray<DeviceTextureInfo*>( options.maxTextures, true );
    deviceContext_.pageTable          = pageTable_;
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

DemandTextureLoaderImpl::~DemandTextureLoaderImpl() {}

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

    if( !textures_.empty() )
    {
        memsetAsync( deviceContext_.requestedBits, 0, stream );
        memsetAsync( deviceContext_.counters, 0, stream );

        if( residentBitsDirty_ )
        {
            memcpyHtoDAsync( deviceContext_.residentBits, residentBits_.words(), stream );
            residentBitsDirty_ = false;
        }

        deviceContext_.textureInfos.len = static_cast<uint32_t>( textures_.size() );
    }

    deviceContext = deviceContext_;
}

void DemandTextureLoaderImpl::processRequests( hipStream_t stream, const DeviceContext& deviceContext )
{
    std::lock_guard<std::mutex> lock( mutex_ );

    if( textures_.empty() )
        return;

    if( deviceContext.pageMemory.ptr != deviceContext_.pageMemory.ptr
        || deviceContext.requestedBits.ptr != deviceContext_.requestedBits.ptr
        || deviceContext.residentBits.ptr != deviceContext_.residentBits.ptr
        || deviceContext.requestedResources.ptr != deviceContext_.requestedResources.ptr
        || deviceContext.counters.ptr != deviceContext_.counters.ptr  // TODO_BS: || deviceContext.pageTable != deviceContext_.pageTable
        || deviceContext.textureInfos.ptr != deviceContext_.textureInfos.ptr )
    {
        throw std::invalid_argument( "DeviceContext does not belong to this demand texture loader" );
    }

    memcpyDtoHAsync( requestedResources_, deviceContext.requestedResources, stream );
    memcpyDtoHAsync( counters_, deviceContext.counters, stream );
    HIP_CHECK( hipStreamSynchronize( stream ) );

    const uint32_t requestCount = counters_[static_cast<uint32_t>( CounterIndex::RequestedResources )];
    for( size_t i = 0; i < requestCount; i++ )
    {
        const uint32_t resourceId = requestedResources_.at( i );

        const Resource resource = decode( resourceId );
        switch( resource.type )
        {
            case ResourceType::TextureInfo: {
                processTextureInfo( resource.textureInfo.textureId, stream, deviceContext );
                break;
            }
            case ResourceType::TextureTile: {
                processTextureTile( resource.tile, stream, deviceContext );
                break;
            }
            default: {
                throw std::logic_error( "Unhandled resource type: " + std::to_string( static_cast<uint32_t>( resource.type ) ) );
            }
        }

        residentBits_.set( resourceId, true );
        residentBitsDirty_ = true;
    }
}

void DemandTextureLoaderImpl::processTextureInfo( uint32_t textureId, hipStream_t stream, const DeviceContext& deviceContext )
{
    DemandTextureImpl& texture = *textures_.at( textureId );
    TextureInfo        hostInfo{};
    texture.image->open( &hostInfo );
    assert( hostInfo.isValid );

    const uint32_t bytesPerTexel = pixelSize( hostInfo.format, hostInfo.numChannels );
    const uint2    tileShape     = tileShapeForGranularity( pageSystem_.granularity(), bytesPerTexel );

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
    deviceInfo.mipCount         = hostInfo.numMipLevels;

    uint32_t pageCount = 0;
    for( uint32_t mip = 0; mip < deviceInfo.mipCount; ++mip )
    {
        auto& level     = deviceInfo.mips[mip];
        level.width     = mipDimension( deviceInfo.width, mip );
        level.height    = mipDimension( deviceInfo.height, mip );
        level.tilesX    = ceilDiv( level.width, deviceInfo.tileWidth );
        level.tilesY    = ceilDiv( level.height, deviceInfo.tileHeight );
        level.startPage = pageTable_.textureTiles.nextAvailablePage + pageCount;

        pageCount += level.pageCount();
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

void DemandTextureLoaderImpl::processTextureTile( const ResourceTile& tile, hipStream_t stream, const DeviceContext& deviceContext )
{
    if( !pageSystem_.mapped( tile.pageId ) )
        pageSystem_.map( tile.pageId );

    const DemandTextureImpl& texture = *textures_.at( tile.textureId );
    const DeviceTextureInfo& info    = loadedTextureInfos_.at( texture.loadedTextureInfoId );
    
    Tile t{};
    t.x      = tile.tileX;
    t.y      = tile.tileY;
    t.width  = info.tileWidth;
    t.height = info.tileHeight;
    assert( static_cast<size_t>( t.width ) * t.height * info.bytesPerTexel <= sizeInBytes( tmpPageBuffer_ ) );
    assert( texture.image->readTile( reinterpret_cast<char*>( tmpPageBuffer_.data() ), tile.mipLevel, t, stream ) );

    memcpyHtoDAsync( pageSystem_.page( tile.pageId ), tmpPageBuffer_, stream );
    HIP_CHECK( hipStreamSynchronize( stream ) );
}

Resource DemandTextureLoaderImpl::decode( uint32_t resourceId )
{
    if( resourceId < options_.maxTextures )
    {
        return Resource::TextureInfo( pageTable_.getTextureIdByResourceId( resourceId ) );
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
        for( uint32_t mipLevel = 0; mipLevel < info.mipCount; ++mipLevel )
        {
            const DeviceMipLevel& level = info.mips[mipLevel];
            if( level.startPage <= pageId && pageId < level.startPage + level.pageCount() )
            {
                const uint32_t pageInLevel = pageId - level.startPage;
                const uint32_t tileX       = pageInLevel % level.tilesX;
                const uint32_t tileY       = pageInLevel / level.tilesX;
                return Resource::TextureTile( pageId, info.textureId, mipLevel, tileX, tileY );
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
