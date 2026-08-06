#include "Internal/HipCheck.h"
#include "Internal/Utils.h"
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <algorithm>
#include <mutex>

namespace hip_demand::vmm {

using internal::Bitset;
using internal::calculateMipLevels;
using internal::ceilDiv;
using internal::mipDimension;
using internal::memcpyDtoHAsync;
using internal::memcpyHtoDAsync;
using internal::memset;
using internal::memsetAsync;
using internal::NonCopyble;
using internal::tileShapeForGranularity;

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

inline uint32_t pixelSize( TextureFormat format )
{
    switch( format )
    {
        case TextureFormat::R8Unorm:
            return 1 * sizeof( uint8_t );
        case TextureFormat::RG8Unorm:
            return 2 * sizeof( uint8_t );
        case TextureFormat::RGBA8Unorm:
            return 4 * sizeof( uint8_t );
        case TextureFormat::R16Unorm:
            return 1 * sizeof( uint16_t );
        case TextureFormat::RG16Unorm:
            return 2 * sizeof( uint16_t );
        case TextureFormat::RGBA16Unorm:
            return 4 * sizeof( uint16_t );
        case TextureFormat::R32Float:
            return 1 * sizeof( float );
        case TextureFormat::RG32Float:
            return 2 * sizeof( float );
        case TextureFormat::RGBA32Float:
            return 4 * sizeof( float );
        default:
            throw std::invalid_argument( "Unsupported texture format" );
    }
}

struct VmmTileKey
{
    uint32_t textureId = 0;
    uint32_t mipLevel  = 0;
    uint32_t tileX     = 0;
    uint32_t tileY     = 0;

    bool operator==( const VmmTileKey& other ) const
    {
        return textureId == other.textureId && mipLevel == other.mipLevel && tileX == other.tileX && tileY == other.tileY;
    }
};

inline void readTile( const VmmTileKey& key, const ImageSource& image, const DeviceTextureInfo& info, std::vector<uint8_t>& pageBuffer )
{
    assert( info.tileWidth * info.tileHeight * info.bytesPerTexel <= pageBuffer.size() );

    const size_t   mipWidth   = info.mips[key.mipLevel].width;
    const size_t   mipHeight  = info.mips[key.mipLevel].height;
    const size_t   firstX     = key.tileX * info.tileWidth;
    const size_t   firstY     = key.tileY * info.tileHeight;
    const size_t   copyWidth  = std::min( static_cast<size_t>( info.tileWidth ), mipWidth - firstX );
    const size_t   copyHeight = std::min( static_cast<size_t>( info.tileHeight ), mipHeight - firstY );
    const uint8_t* source     = image.data.data();
    uint8_t*       output     = pageBuffer.data();

    for( size_t row = 0; row < copyHeight; ++row )
    {
        const size_t sourceOffset = ( ( firstY + row ) * mipWidth + firstX ) * info.bytesPerTexel;
        const size_t outputOffset = row * info.tileWidth * info.bytesPerTexel;
        std::memcpy( output + outputOffset, source + sourceOffset, copyWidth * info.bytesPerTexel );
    }
}

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

    uint32_t            physicalPageId = virtualIdToPhysicalId_[pageId];
    DeviceSpan<uint8_t> virtualPage    = page( pageId );
    HIP_CHECK( hipMemUnmap( virtualPage.ptr, virtualPage.len ) );
    virtualIdToPhysicalId_.at( pageId )               = INVALID_PAGE;
    physicalPages_.at( physicalPageId ).virtualPageId = INVALID_PAGE;
    freePhysicalPages_.push_back( physicalPageId );
}

class DemandTextureImpl : public DemandTexture, NonCopyble
{
  public:
    DemandTextureImpl( uint32_t textureId, std::shared_ptr<ImageSource> imageSource )
        : id( textureId )
        , image( imageSource )
    {
    }
    uint32_t getId() const override { return id; }

    uint32_t                     id = 0;
    std::shared_ptr<ImageSource> image{};
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
    VmmTileKey decodePage( uint32_t pageId );

    mutable std::mutex    mutex_;
    Options               options_{};
    uint32_t              nextAvailablePage_      = 0;
    uint32_t              frame_                  = 0;
    bool                  textureInfosDirty_      = false;
    bool                  residentPageFlagsDirty_ = false;
    Bitset                residentPageBitFlags_{};
    Bitset                requestedPageBitFlags_{};
    std::vector<uint32_t> requestedPages_{};
    uint32_t              requestedPageCount_ = 0;
    std::vector<uint8_t>  tmpPageBuffer_{};
    VmmPageSystem         pageSystem_;

    std::vector<std::unique_ptr<DemandTextureImpl>> textures_{};
    std::vector<DeviceTextureInfo>                  textureInfos_{};

    HipGC         hipGC_;
    DeviceContext deviceContext_{};
};

DemandTextureLoaderImpl::DemandTextureLoaderImpl( const Options& options )
    : options_( options )
    , pageSystem_( options_.maxVirtualPages, options_.maxPhysicalPages )
{
    tmpPageBuffer_.resize( pageSystem_.pageBytes() );
    residentPageBitFlags_.resize( options_.maxVirtualPages );
    requestedPageBitFlags_.resize( options_.maxVirtualPages );
    requestedPages_.resize( options_.maxRequestedPages );

    deviceContext_.pageMemory            = pageSystem_.virtualAddressSpace();
    deviceContext_.requestedPageBitFlags = hipGC_.allocArray<uint32_t>( requestedPageBitFlags_.wordCount(), true );
    deviceContext_.residentPageBitFlags  = hipGC_.allocArray<uint32_t>( residentPageBitFlags_.wordCount(), true );
    deviceContext_.textureInfos          = hipGC_.allocArray<DeviceTextureInfo>( options_.maxTextures, true );
    deviceContext_.counters = hipGC_.allocArray<uint32_t>( static_cast<uint32_t>( CounterIndex::NumCounters ), true );
    deviceContext_.pageSize = pageSystem_.pageBytes();
}

DemandTextureLoaderImpl::~DemandTextureLoaderImpl() {}

const DemandTexture& DemandTextureLoaderImpl::createTexture( std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& descriptor )
{
    if( !imageSource || imageSource->data.empty() )
        throw std::invalid_argument( "createTexture requires an image source" );

    if( imageSource->width == 0 || imageSource->height == 0 )
        throw std::invalid_argument( "createTexture requires width > 0 and height > 0" );

    std::lock_guard<std::mutex> lock( mutex_ );
    if( textures_.size() >= options_.maxTextures )
        throw std::runtime_error( "Maximum demand texture count exceeded" );

    const uint32_t bytesPerTexel = pixelSize( descriptor.format );
    const uint2    tileShape     = tileShapeForGranularity( pageSystem_.granularity(), bytesPerTexel );

    const uint32_t textureId = static_cast<uint32_t>( textureInfos_.size() );
    // TODO_BS: implement mip count > 1
    const uint32_t mipCount = 1;
    if( mipCount > MAX_TEXTURE_MIP_LEVELS )
        throw std::runtime_error( "Texture has more mip levels than DeviceTextureInfo can store" );

    DeviceTextureInfo textureInfo{};
    textureInfo.width            = imageSource->width;
    textureInfo.height           = imageSource->height;
    textureInfo.tileWidth        = tileShape.x;
    textureInfo.tileHeight       = tileShape.y;
    textureInfo.startPage        = nextAvailablePage_;
    textureInfo.mipCount         = mipCount;
    textureInfo.addressMode[0]   = descriptor.addressMode[0];
    textureInfo.addressMode[1]   = descriptor.addressMode[1];
    textureInfo.filterMode       = descriptor.filterMode;
    textureInfo.mipmapFilterMode = descriptor.mipmapFilterMode;
    textureInfo.normalizedCoords = descriptor.normalizedCoords ? 1u : 0u;
    textureInfo.format           = descriptor.format;
    textureInfo.bytesPerTexel    = bytesPerTexel;

    uint32_t pageCount = 0;
    for( uint32_t mip = 0; mip < mipCount; ++mip )
    {
        auto& level     = textureInfo.mips[mip];
        level.width     = mipDimension( textureInfo.width, mip );
        level.height    = mipDimension( textureInfo.height, mip );
        level.tilesX    = ceilDiv( level.width, textureInfo.tileWidth );
        level.tilesY    = ceilDiv( level.height, textureInfo.tileHeight );
        level.startPage = nextAvailablePage_ + pageCount;

        pageCount += level.pageCount();
    }

    if( nextAvailablePage_ + pageCount > options_.maxVirtualPages )
        throw std::runtime_error( "Maximum demand virtual page count exceeded" );

    textures_.emplace_back( std::make_unique<DemandTextureImpl>( textureId, imageSource ) );
    textureInfos_.push_back( textureInfo );
    nextAvailablePage_ += pageCount;
    textureInfosDirty_ = true;
    return *textures_.back();
}

void DemandTextureLoaderImpl::launchPrepare( hipStream_t stream, DeviceContext& deviceContext )
{
    std::lock_guard<std::mutex> lock( mutex_ );

    memsetAsync( deviceContext_.requestedPageBitFlags, 0, stream );
    memsetAsync( deviceContext_.counters, 0, stream );

    if( textureInfosDirty_ )
    {
        memcpyHtoDAsync( deviceContext_.textureInfos, textureInfos_, textureInfos_.size(), stream );
        // change len because for device we preallocated for maxTextures
        deviceContext_.textureInfos.len = textureInfos_.size();
        textureInfosDirty_ = false;
    }

    if( residentPageFlagsDirty_ )
    {
        memcpyHtoDAsync( deviceContext_.residentPageBitFlags, residentPageBitFlags_.words(), stream );
        residentPageFlagsDirty_ = false;
    }

    deviceContext = deviceContext_;
}

void DemandTextureLoaderImpl::processRequests( hipStream_t stream, const DeviceContext& deviceContext )
{
    if( deviceContext.pageMemory.ptr != deviceContext_.pageMemory.ptr
        || deviceContext.requestedPageBitFlags.ptr != deviceContext_.requestedPageBitFlags.ptr
        || deviceContext.residentPageBitFlags.ptr != deviceContext_.residentPageBitFlags.ptr
        || deviceContext.textureInfos.ptr != deviceContext_.textureInfos.ptr
        || deviceContext.counters.ptr != deviceContext_.counters.ptr || deviceContext.pageSize != deviceContext_.pageSize )
    {
        throw std::invalid_argument( "DeviceContext does not belong to this demand texture loader" );
    }

    std::lock_guard<std::mutex> lock( mutex_ );

    // TODO_BS: move getting pageIds to GPU
    memcpyDtoHAsync( requestedPageBitFlags_.words(), deviceContext.requestedPageBitFlags, stream );
    HIP_CHECK( hipStreamSynchronize( stream ) );
    requestedPageCount_ = 0;
    for( uint32_t wordIndex = 0; wordIndex < requestedPageBitFlags_.wordCount(); ++wordIndex )
    {
        uint32_t word = requestedPageBitFlags_.words()[wordIndex];
        while( word != 0 )
        {
#if defined( _MSC_VER )
            unsigned long bit = 0;
            _BitScanForward( &bit, word );
            const uint32_t bitIndex = static_cast<uint32_t>( bit );
#else
            const uint32_t bitIndex = static_cast<uint32_t>( __builtin_ctz( word ) );
#endif
            const uint32_t pageId = wordIndex * 32 + bitIndex;
            if( pageId < options_.maxVirtualPages )
                requestedPages_.at( requestedPageCount_++ ) = pageId;
            word &= word - 1;
        }
    }

    for( size_t i = 0; i < requestedPageCount_; i++ )
    {
        const uint32_t pageId = requestedPages_[i];
        assert( pageId < options_.maxVirtualPages );

        const VmmTileKey tileKey = decodePage( pageId );
        if( !pageSystem_.mapped( pageId ) )
            pageSystem_.map( pageId );

        readTile( tileKey, *textures_.at( tileKey.textureId )->image, textureInfos_.at( tileKey.textureId ), tmpPageBuffer_ );

        memcpyHtoDAsync( pageSystem_.page( pageId ), tmpPageBuffer_, stream );
        HIP_CHECK( hipStreamSynchronize( stream ) );

        residentPageBitFlags_.set( pageId, true );
        residentPageFlagsDirty_ = true;
    }
}

VmmTileKey DemandTextureLoaderImpl::decodePage( uint32_t pageId )
{
    const auto it =
        std::upper_bound( textureInfos_.cbegin(), textureInfos_.cend(), pageId,
                          []( uint32_t page, const DeviceTextureInfo& info ) { return page < info.startPage; } );

    if( it == textureInfos_.begin() )
        throw std::out_of_range( "Cannot decode virtual page " + std::to_string( pageId )
                                 + ": it does not belong to any registered texture" );

    const auto     infoIt    = std::prev( it );
    const uint32_t textureId = static_cast<uint32_t>( std::distance( textureInfos_.cbegin(), infoIt ) );

    const DeviceTextureInfo& info = *infoIt;
    for( uint32_t mipLevel = 0; mipLevel < info.mipCount; ++mipLevel )
    {
        const DeviceMipLevel& level = info.mips[mipLevel];
        if( level.startPage <= pageId && pageId < level.startPage + level.pageCount() )
        {
            const uint32_t pageInLevel = pageId - level.startPage;
            return VmmTileKey{ textureId, mipLevel, pageInLevel % level.tilesX, pageInLevel / level.tilesX };
        }
    }

    // we should never be here!
    throw std::out_of_range( "Cannot decode virtual page " + std::to_string( pageId )
                             + ": it is outside the mip ranges of texture " + std::to_string( textureId ) );
}

std::unique_ptr<DemandTextureLoader> createDemandTextureLoader( const Options& options )
{
    return std::make_unique<DemandTextureLoaderImpl>( options );
}


}  // namespace hip_demand::vmm
