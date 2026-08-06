#pragma once

#include "DeviceContext.h"
#include <algorithm>
#include <cmath>
#include <hip/hip_runtime.h>
#include <type_traits>


namespace hip_demand::vmm {

#if !defined( __HIPCC__ )

HIP_DEMAND_INLINE uint32_t atomicOr( uint32_t* address, uint32_t value )
{
    const uint32_t oldValue = *address;
    *address                = oldValue | value;
    return oldValue;
}

HIP_DEMAND_INLINE float4 operator+( const float4& a, const float4& b )
{
    return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

HIP_DEMAND_INLINE float4 operator*( const float4& value, float scalar )
{
    return make_float4( value.x * scalar, value.y * scalar, value.z * scalar, value.w * scalar );
}

HIP_DEMAND_INLINE float4 operator*( float scalar, const float4& value )
{
    return value * scalar;
}

#endif

HIP_DEMAND_INLINE float lerp( float a, float b, float t )
{
    return a + ( b - a ) * t;
}

HIP_DEMAND_INLINE float4 lerp( const float4& a, const float4& b, float t )
{
    return make_float4( lerp( a.x, b.x, t ), lerp( a.y, b.y, t ), lerp( a.z, b.z, t ), lerp( a.w, b.w, t ) );
}

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

HIP_DEMAND_INLINE float4 fetchTexel( const DeviceContext& context, const DeviceTextureInfo& texture, uint32_t mipLevel, int x, int y, bool& resident )
{
    resident = false;
    if( mipLevel >= texture.mipCount || mipLevel >= MAX_TEXTURE_MIP_LEVELS )
        return make_float4( 0.0f, 0.0f, 0.0f, 0.0f );

    const auto& mip   = texture.mips[mipLevel];
    bool        valid = true;

    x = applyAddressMode( x, static_cast<int>( mip.width ), texture.addressMode[0], valid );
    y = applyAddressMode( y, static_cast<int>( mip.height ), texture.addressMode[1], valid );
    if( !valid )
    {
        resident = true;
        return make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
    }

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

HIP_DEMAND_INLINE float4
sampleMipLevel( const DeviceContext& context, const DeviceTextureInfo& texture, uint32_t mipLevel, float x, float y, bool& isResident )
{
    isResident = false;

    if( texture.normalizedCoords )
    {
        x = x * static_cast<float>( texture.mips[mipLevel].width );
        y = y * static_cast<float>( texture.mips[mipLevel].height );
    }

    if( texture.filterMode == hipFilterModePoint )
    {
        x = std::floorf( x );
        y = std::floorf( y );
        return fetchTexel( context, texture, mipLevel, static_cast<int>( x ), static_cast<int>( y ), isResident );
    }

    x = x - 0.5f;
    y = y - 0.5f;

    const int   x0 = static_cast<int>( std::floorf( x ) );
    const int   y0 = static_cast<int>( std::floorf( y ) );
    const float a  = x - static_cast<float>( x0 );
    const float b  = y - static_cast<float>( y0 );

    bool resident00 = false;
    bool resident10 = false;
    bool resident01 = false;
    bool resident11 = false;

    const float4 t00 = fetchTexel( context, texture, mipLevel, x0, y0, resident00 );
    const float4 t10 = fetchTexel( context, texture, mipLevel, x0 + 1, y0, resident10 );
    const float4 t01 = fetchTexel( context, texture, mipLevel, x0, y0 + 1, resident01 );
    const float4 t11 = fetchTexel( context, texture, mipLevel, x0 + 1, y0 + 1, resident11 );

    isResident = resident00 && resident10 && resident01 && resident11;
    return ( 1.0f - a ) * ( 1.0f - b ) * t00 + a * ( 1.0f - b ) * t10 + ( 1.0f - a ) * b * t01 + a * b * t11;
}

HIP_DEMAND_INLINE float4 sampleTexture( const DeviceContext& context, uint32_t textureId, float u, float v, uint32_t mipLevel = 0 )
{
    if( textureId >= context.textureInfos.len )
        return make_float4( 0.0f, 0.0f, 0.0f, 0.0f );

    const DeviceTextureInfo& texture = context.textureInfos.ptr[textureId];
    if( texture.mipCount == 0 )
        return make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
    if( mipLevel >= texture.mipCount )
        mipLevel = texture.mipCount - 1;

    bool isResident = false;
    return sampleMipLevel( context, texture, mipLevel, u, v, isResident );
}

template <class Sample>
HIP_DEMAND_INLINE Sample tex2DLod( const DeviceContext& context, uint32_t textureId, float x, float y, float lod, bool& isResident )
{
    static_assert( std::is_same<Sample, float4>::value, "VMM tex2DLod currently supports Sample = float4 only" );

    isResident = false;
    if( textureId >= context.textureInfos.len )
        return Sample{};

    const DeviceTextureInfo& texture = context.textureInfos.ptr[textureId];

    lod = std::clamp( lod, 0.0f, static_cast<float>( texture.mipCount - 1 ) );
    if( texture.mipmapFilterMode == hipFilterModePoint )
    {
        const uint32_t mipLevel = static_cast<uint32_t>( std::floorf( lod + 0.5f ) );
        return sampleMipLevel( context, texture, mipLevel, x, y, isResident );
    }

    const uint32_t mipLevel0 = static_cast<uint32_t>( std::floorf( lod ) );
    const uint32_t mipLevel1 = mipLevel0 + 1 < texture.mipCount ? mipLevel0 + 1 : mipLevel0;

    bool         resident0 = false;
    const float4 sample0   = sampleMipLevel( context, texture, mipLevel0, x, y, resident0 );
    if( mipLevel0 == mipLevel1 )
    {
        isResident = resident0;
        return sample0;
    }

    bool         resident1 = false;
    const float4 sample1   = sampleMipLevel( context, texture, mipLevel1, x, y, resident1 );

    isResident = resident0 && resident1;
    return lerp( sample0, sample1, lod - static_cast<float>( mipLevel0 ) );
}

}  // namespace hip_demand::vmm
