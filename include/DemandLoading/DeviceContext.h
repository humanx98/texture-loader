#pragma once

/// @file DeviceContext.h
/// @brief Device context for GPU texture sampling.
/// @note No HIP headers required. This header uses platform-agnostic types.

#include <cstdint>
#include <hip/hip_runtime.h>

#if defined( __HIPCC__ )
#define HIP_DEMAND_INLINE __device__ __forceinline__
#else
#define HIP_DEMAND_INLINE inline
#endif

namespace hip_demand {

/// Platform-agnostic texture object handle.
/// This is binary-compatible with hipTextureObject_t (unsigned long long).
using TextureObject = unsigned long long;

// Device context passed to kernels
// This structure contains GPU-accessible data for texture sampling
struct DeviceContext
{
    uint32_t*      residentFlags;    // Bit flags for texture residency
    TextureObject* textures;         // Array of texture objects
    uint32_t*      requests;         // Request buffer
    uint32_t*      requestCount;     // Atomic counter for requests
    uint32_t*      requestOverflow;  // Flag set when request buffer overflows
    uint32_t       maxTextures;
    uint32_t       maxRequests;
};

template <typename T>
using DevicePtr = T*;

template <typename T>
struct DeviceSpan
{
    DevicePtr<T> ptr = nullptr;
    size_t       len = 0;

    DeviceSpan() = default;

    explicit DeviceSpan( DevicePtr<T> ptr_, size_t len_ )
        : ptr( ptr_ )
        , len( len_ )
    {
    }

    HIP_DEMAND_INLINE size_t sizeInBytes() const { return len * sizeof( T ); }
};

}  // namespace hip_demand

namespace hip_demand::vmm {

constexpr uint32_t INVALID_TEXTURE        = ~0u;
constexpr uint32_t INVALID_PAGE           = ~0u;
constexpr uint32_t MAX_TEXTURE_MIP_LEVELS = 15;

struct DeviceMipLevel
{
    uint32_t width         = 0;
    uint32_t height        = 0;
    uint32_t tilesX        = 0;
    uint32_t tilesY        = 0;
    uint32_t startPage     = INVALID_PAGE;
    uint32_t mipTailOffset = 0;
    bool     mipTail       = false;

    HIP_DEMAND_INLINE uint32_t pageCount() const { return tilesX * tilesY; }
};

struct DeviceTextureInfo
{
    uint32_t        textureId        = INVALID_TEXTURE;
    uint32_t        width            = 0;
    uint32_t        height           = 0;
    uint32_t        tileWidth        = 0;
    uint32_t        tileHeight       = 0;
    uint32_t        addressMode[2]   = { hipAddressModeWrap, hipAddressModeWrap };
    uint32_t        filterMode       = hipFilterModeLinear;
    uint32_t        mipmapFilterMode = hipFilterModeLinear;
    uint32_t        normalizedCoords = 1;
    hipArray_Format format           = HIP_AD_FORMAT_UNSIGNED_INT8;
    uint32_t        numChannels      = 4;
    uint32_t        bytesPerTexel    = 4;
    uint32_t        startPage        = INVALID_PAGE;

    uint32_t mipCount          = 0;
    uint32_t mipTailFirstLevel = MAX_TEXTURE_MIP_LEVELS;
    uint32_t mipTailPage       = INVALID_PAGE;
    uint32_t mipTailSize       = 0;
    struct
    {
        uint32_t startPage = INVALID_PAGE;
        uint32_t mipTailOffset = 0;  // Byte offset within the 64 KiB mip-tail page; uint32_t keeps each mip descriptor compact.
    } mips[MAX_TEXTURE_MIP_LEVELS]{};

    HIP_DEMAND_INLINE DeviceMipLevel getMipLevel( uint32_t level ) const
    {
        DeviceMipLevel mip{};
        mip.startPage     = mips[level].startPage;
        mip.mipTailOffset = mips[level].mipTailOffset;
        mip.width         = std::max( width >> level, 1u );
        mip.height        = std::max( height >> level, 1u );
        mip.tilesX        = mip.width / tileWidth + static_cast<uint32_t>( mip.width % tileWidth != 0 );
        mip.tilesY        = mip.height / tileHeight + static_cast<uint32_t>( mip.height % tileHeight != 0 );
        mip.mipTail       = level >= mipTailFirstLevel;
        return mip;
    }
};

struct PageTable
{
    struct Range
    {
        uint32_t startPage         = 0;
        uint32_t pageCount         = 0;
        uint32_t nextAvailablePage = 0;

        HIP_DEMAND_INLINE Range() {}
        HIP_DEMAND_INLINE Range( uint32_t start, uint32_t count )
            : startPage( start )
            , pageCount( count )
            , nextAvailablePage( start )
        {
        }
    };

    size_t   pageSize = 0;
    Range    textureInfos{};
    uint32_t maxTextures = 0;
    Range    textureTiles{};

    HIP_DEMAND_INLINE uint32_t getTextureIdByResourceId( uint32_t resourceId ) const { return resourceId; }

    HIP_DEMAND_INLINE uint32_t getTextureTilePageByResourceId( uint32_t resourceId ) const
    {
        return textureTiles.startPage + resourceId - maxTextures;
    }


    HIP_DEMAND_INLINE uint32_t getResourceIdByTextureId( uint32_t textureId ) const { return textureId; }
    HIP_DEMAND_INLINE uint32_t getResourceIdByTextureTilePage( uint32_t pageId ) const
    {
        return pageId - textureTiles.startPage + maxTextures;
    }
};

enum class CounterIndex : uint32_t
{
    RequestedResources = 0,
    NumCounters
};

struct DeviceContext
{
    PageTable                      pageTable{};
    DeviceSpan<uint8_t>            pageMemory{};
    DeviceSpan<uint32_t>           residentBits{};
    DeviceSpan<DeviceTextureInfo*> textureInfos{};
    DeviceSpan<uint32_t>           requestedBits{};
    DeviceSpan<uint32_t>           requestedResources{};
    DeviceSpan<uint32_t>           counters{};
    size_t                         poolIndex = 0;
};

}  // namespace hip_demand::vmm
