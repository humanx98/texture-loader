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

enum class TextureFormat : uint32_t
{
    R8Unorm,
    RG8Unorm,
    RGBA8Unorm,
    R16Unorm,
    RG16Unorm,
    RGBA16Unorm,
    R32Float,
    RG32Float,
    RGBA32Float,
};

template <typename T>
struct DeviceSpan
{
    T*     ptr = nullptr;
    size_t len = 0;

    DeviceSpan() = default;

    explicit DeviceSpan( T* ptr_, size_t len_ )
        : ptr( ptr_ )
        , len( len_ )
    {
    }

    HIP_DEMAND_INLINE size_t sizeInBytes() const { return len * sizeof( T ); }
};

}  // namespace hip_demand

namespace hip_demand::vmm {

constexpr uint32_t INVALID_PAGE           = ~0u;
constexpr uint32_t MAX_TEXTURE_MIP_LEVELS = 15;

struct DeviceMipLevel
{
    uint32_t width     = 0;
    uint32_t height    = 0;
    uint32_t tilesX    = 0;
    uint32_t tilesY    = 0;
    uint32_t startPage = 0;

    HIP_DEMAND_INLINE uint32_t pageCount() const { return tilesX * tilesY; }
};

struct DeviceTextureInfo
{
    uint32_t       width            = 0;
    uint32_t       height           = 0;
    uint32_t       tileWidth        = 0;
    uint32_t       tileHeight       = 0;
    uint32_t       startPage        = 0;
    uint32_t       mipCount         = 0;
    uint32_t       addressMode[2]   = { hipAddressModeWrap, hipAddressModeWrap };
    uint32_t       filterMode       = hipFilterModeLinear;
    uint32_t       mipmapFilterMode = hipFilterModeLinear;
    uint32_t       normalizedCoords = 1;
    TextureFormat  format           = TextureFormat::RGBA8Unorm;
    uint32_t       bytesPerTexel    = 4;
    DeviceMipLevel mips[MAX_TEXTURE_MIP_LEVELS]{};
};

enum class CounterIndex : uint32_t
{
    RequestedPages = 0,
    NumCounters
};

struct DeviceContext
{
    DeviceSpan<uint8_t>           pageMemory{};
    DeviceSpan<uint32_t>          requestedPageBitFlags{};
    DeviceSpan<uint32_t>          requestedPages;
    DeviceSpan<uint32_t>          residentPageBitFlags{};
    DeviceSpan<DeviceTextureInfo> textureInfos{};
    DeviceSpan<uint32_t>          counters{};
    size_t                        pageSize = 0;
};

}  // namespace hip_demand::vmm
