#pragma once

#include "DeviceContext.h"
#include <hip/hip_runtime.h>


namespace hip_demand::vmm {

#if !defined( __HIPCC__ )

HIP_DEMAND_INLINE uint32_t atomicOr( uint32_t* address, uint32_t value )
{
    const uint32_t oldValue = *address;
    *address                = oldValue | value;
    return oldValue;
}

#endif

HIP_DEMAND_INLINE void getWordIdxAndBitIdx( uint32_t idx, uint32_t& wordIdx, uint32_t& bitIdx )
{
    wordIdx = idx >> 5;   // idx / 32
    bitIdx  = idx & 31u;  // idx % 32
}

HIP_DEMAND_INLINE void recordPageRequest( const DeviceContext& context, uint32_t pageId )
{
    uint32_t wordIdx = 0;
    uint32_t bitIdx  = 0;
    getWordIdxAndBitIdx( pageId, wordIdx, bitIdx );
    atomicOr( &context.requestedPageBitFlags.ptr[wordIdx], 1u << bitIdx );
}

HIP_DEMAND_INLINE bool isPageResident( const DeviceContext& context, uint32_t pageId )
{
    uint32_t wordIdx = 0;
    uint32_t bitIdx  = 0;
    getWordIdxAndBitIdx( pageId, wordIdx, bitIdx );
    return ( context.residentPageBitFlags.ptr[wordIdx] & ( 1u << bitIdx ) ) != 0;
}

HIP_DEMAND_INLINE int applyAddressMode( int coordinate, int extent, uint32_t mode, bool& valid )
{
    switch( static_cast<hipTextureAddressMode>( mode ) )
    {
        case hipAddressModeWrap:
            coordinate %= extent;
            return coordinate < 0 ? coordinate + extent : coordinate;

        case hipAddressModeMirror: {
            const int period = extent * 2;
            coordinate %= period;
            if( coordinate < 0 )
                coordinate += period;
            return coordinate < extent ? coordinate : period - coordinate - 1;
        }

        case hipAddressModeBorder:
            if( coordinate < 0 || coordinate >= extent )
            {
                valid = false;
                return 0;
            }
            return coordinate;

        case hipAddressModeClamp:
        default:
            return std::clamp( coordinate, 0, extent - 1 );
    }
}

HIP_DEMAND_INLINE float4
fetchDemandTexel( const DeviceContext& context, const DeviceTextureInfo& texture, uint32_t mipLevel, int x, int y, bool& resident )
{
    resident = false;
    if( mipLevel >= texture.mipCount || mipLevel >= MAX_TEXTURE_MIP_LEVELS )
        return make_float4( 0.0f, 0.0f, 0.0f, 0.0f );

    const DeviceMipLevel& mip       = texture.mips[mipLevel];
    const uint32_t        mipWidth  = mipDimension( texture.width, mipLevel );
    const uint32_t        mipHeight = mipDimension( texture.height, mipLevel );
    bool                  valid     = true;
    x = applyAddressMode( x, static_cast<int>( mipWidth ), texture.addressMode[0], valid );
    y = applyAddressMode( y, static_cast<int>( mipHeight ), texture.addressMode[1], valid );
    if( !valid )
        return make_float4( 0.0f, 0.0f, 0.0f, 0.0f );

    const uint32_t tileX = static_cast<uint32_t>( x ) / texture.tileWidth;
    const uint32_t tileY = static_cast<uint32_t>( y ) / texture.tileHeight;
    if( tileX >= mip.tilesX || tileY >= mip.tilesY )
        return make_float4( 0.0f, 0.0f, 0.0f, 0.0f );

    const uint32_t pageId = mip.startPage + tileY * mip.tilesX + tileX;
    if( !isPageResident( context, pageId ) )
    {
        recordPageRequest( context, pageId );
        return make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
    }

    const uint32_t localX = static_cast<uint32_t>( x ) % texture.tileWidth;
    const uint32_t localY = static_cast<uint32_t>( y ) % texture.tileHeight;
    const uint64_t byteOffset = pageId * context.pageSize + ( localY * texture.tileWidth + localX ) * texture.bytesPerTexel;
    const uint8_t* texel = context.pageMemory.ptr + byteOffset;
    resident             = true;

    switch( texture.format )
    {
        case TextureFormat::R8Unorm:
            return make_float4( texel[0] / 255.0f, 0.0f, 0.0f, 1.0f );
        case TextureFormat::RG8Unorm:
            return make_float4( texel[0] / 255.0f, texel[1] / 255.0f, 0.0f, 1.0f );
        case TextureFormat::RGBA8Unorm:
            return make_float4( texel[0] / 255.0f, texel[1] / 255.0f, texel[2] / 255.0f, texel[3] / 255.0f );

        case TextureFormat::R16Unorm: {
            const uint16_t* value = reinterpret_cast<const uint16_t*>( texel );
            return make_float4( value[0] / 65535.0f, 0.0f, 0.0f, 1.0f );
        }
        case TextureFormat::RG16Unorm: {
            const uint16_t* value = reinterpret_cast<const uint16_t*>( texel );
            return make_float4( value[0] / 65535.0f, value[1] / 65535.0f, 0.0f, 1.0f );
        }
        case TextureFormat::RGBA16Unorm: {
            const uint16_t* value = reinterpret_cast<const uint16_t*>( texel );
            return make_float4( value[0] / 65535.0f, value[1] / 65535.0f, value[2] / 65535.0f, value[3] / 65535.0f );
        }

        case TextureFormat::R32Float: {
            const float* value = reinterpret_cast<const float*>( texel );
            return make_float4( value[0], 0.0f, 0.0f, 1.0f );
        }
        case TextureFormat::RG32Float: {
            const float* value = reinterpret_cast<const float*>( texel );
            return make_float4( value[0], value[1], 0.0f, 1.0f );
        }
        case TextureFormat::RGBA32Float: {
            const float* value = reinterpret_cast<const float*>( texel );
            return make_float4( value[0], value[1], value[2], value[3] );
        }

        default:
            resident = false;
            return make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
    }
}

// HIP_DEMAND_INLINE float4 sampleDemandTexture( const DeviceContext& context, uint32_t textureId, float u, float v, uint32_t mipLevel = 0 )
// {
//     if( textureId >= context.textureCount || context.textures == nullptr )
//         return make_float4( 0.0f, 0.0f, 0.0f, 0.0f );

//     const DeviceTextureInfo& texture = context.textures[textureId];
//     if( texture.mipCount == 0 )
//         return make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
//     if( mipLevel >= texture.mipCount )
//         mipLevel = texture.mipCount - 1;

//     const DeviceMipLevel& mip = texture.mips[mipLevel];
//     const float           x   = texture.normalizedCoords ? u * mip.width - 0.5f : u - 0.5f;
//     const float           y   = texture.normalizedCoords ? v * mip.height - 0.5f : v - 0.5f;

//     if( texture.filterMode == hipFilterModePoint )
//     {
//         bool resident = false;
//         return fetchDemandTexel( context, texture, mipLevel, static_cast<int>( floorf( x + 0.5f ) ),
//                                  static_cast<int>( floorf( y + 0.5f ) ), resident );
//     }

//     const int    x0         = static_cast<int>( floorf( x ) );
//     const int    y0         = static_cast<int>( floorf( y ) );
//     const float  tx         = x - x0;
//     const float  ty         = y - y0;
//     bool         resident00 = false;
//     bool         resident10 = false;
//     bool         resident01 = false;
//     bool         resident11 = false;
//     const float4 c00        = fetchDemandTexel( context, texture, mipLevel, x0, y0, resident00 );
//     const float4 c10        = fetchDemandTexel( context, texture, mipLevel, x0 + 1, y0, resident10 );
//     const float4 c01        = fetchDemandTexel( context, texture, mipLevel, x0, y0 + 1, resident01 );
//     const float4 c11        = fetchDemandTexel( context, texture, mipLevel, x0 + 1, y0 + 1, resident11 );

//     const float4 cx0 = make_float4( c00.x + ( c10.x - c00.x ) * tx, c00.y + ( c10.y - c00.y ) * tx,
//                                     c00.z + ( c10.z - c00.z ) * tx, c00.w + ( c10.w - c00.w ) * tx );
//     const float4 cx1 = make_float4( c01.x + ( c11.x - c01.x ) * tx, c01.y + ( c11.y - c01.y ) * tx,
//                                     c01.z + ( c11.z - c01.z ) * tx, c01.w + ( c11.w - c01.w ) * tx );
//     return make_float4( cx0.x + ( cx1.x - cx0.x ) * ty, cx0.y + ( cx1.y - cx0.y ) * ty, cx0.z + ( cx1.z - cx0.z ) * ty,
//                         cx0.w + ( cx1.w - cx0.w ) * ty );
// }

// #endif

}  // namespace hip_demand::vmm
