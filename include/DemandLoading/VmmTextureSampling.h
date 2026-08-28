#pragma once

#include "DeviceContext.h"
#include <algorithm>
#include <cmath>
#include <hip/hip_fp16.h>
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

// float4 operators for cpu debugging

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

// float3 operators for cpu debugging

HIP_DEMAND_INLINE float3 operator+( const float3& a, const float3& b )
{
    return make_float3( a.x + b.x, a.y + b.y, a.z + b.z );
}

HIP_DEMAND_INLINE float3 operator*( const float3& value, float scalar )
{
    return make_float3( value.x * scalar, value.y * scalar, value.z * scalar );
}

HIP_DEMAND_INLINE float3 operator*( float scalar, const float3& value )
{
    return value * scalar;
}

// float2 operators for cpu debugging

HIP_DEMAND_INLINE float2 operator+( const float2& a, const float2& b )
{
    return make_float2( a.x + b.x, a.y + b.y );
}

HIP_DEMAND_INLINE float2 operator*( const float2& value, float scalar )
{
    return make_float2( value.x * scalar, value.y * scalar );
}

HIP_DEMAND_INLINE float2 operator*( float scalar, const float2& value )
{
    return value * scalar;
}

#endif

HIP_DEMAND_INLINE void getWordIdxAndBitIdx( uint32_t idx, uint32_t& wordIdx, uint32_t& bitIdx )
{
    wordIdx = idx >> 5;   // idx / 32
    bitIdx  = idx & 31u;  // idx % 32
}

HIP_DEMAND_INLINE void recordRequest( const DeviceContext& context, uint32_t resourceId )
{
    uint32_t wordIdx = 0;
    uint32_t bitIdx  = 0;
    getWordIdxAndBitIdx( resourceId, wordIdx, bitIdx );
    atomicOr( &context.requestedBits.ptr[wordIdx], 1u << bitIdx );
}

HIP_DEMAND_INLINE bool isResourceResident( const DeviceContext& context, uint32_t resourceId )
{
    uint32_t wordIdx = 0;
    uint32_t bitIdx  = 0;
    getWordIdxAndBitIdx( resourceId, wordIdx, bitIdx );
    return ( context.residentBits.ptr[wordIdx] & ( 1u << bitIdx ) ) != 0;
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

HIP_DEMAND_INLINE float decodeTexelChannel( const uint8_t* texel, hipArray_Format format, uint32_t channel )
{
    switch( format )
    {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            return reinterpret_cast<const uint8_t*>( texel )[channel] / 255.0f;
        case HIP_AD_FORMAT_SIGNED_INT8:
            return fmaxf( -1.0f, reinterpret_cast<const int8_t*>( texel )[channel] / 127.0f );
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            return reinterpret_cast<const uint16_t*>( texel )[channel] / 65535.0f;
        case HIP_AD_FORMAT_SIGNED_INT16:
            return fmaxf( -1.0f, reinterpret_cast<const int16_t*>( texel )[channel] / 32767.0f );
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            return static_cast<float>( reinterpret_cast<const uint32_t*>( texel )[channel] ) / 4294967295.0f;
        case HIP_AD_FORMAT_SIGNED_INT32:
            return fmaxf( -1.0f, static_cast<float>( reinterpret_cast<const int32_t*>( texel )[channel] ) / 2147483647.0f );
        case HIP_AD_FORMAT_HALF:
            return __half2float( reinterpret_cast<const __half*>( texel )[channel] );
        case HIP_AD_FORMAT_FLOAT:
            return reinterpret_cast<const float*>( texel )[channel];
        default:
            return 0.0f;
    }
}

template <class Sample>
HIP_DEMAND_INLINE Sample decodeTexel( const uint8_t* texel, hipArray_Format format, uint32_t numChannels )
{
    if constexpr( std::is_same<Sample, float>::value )
    {
        return decodeTexelChannel( texel, format, 0 );
    }
    else if constexpr( std::is_same<Sample, float2>::value )
    {
        float2 result;
        result.x = decodeTexelChannel( texel, format, 0 );
        if( numChannels > 0 )
            result.y = decodeTexelChannel( texel, format, 1 );
        return result;
    }
    else if constexpr( std::is_same<Sample, float3>::value )
    {
        float3 result;
        result.x = decodeTexelChannel( texel, format, 0 );
        if( numChannels > 0 )
            result.y = decodeTexelChannel( texel, format, 1 );
        if( numChannels > 1 )
            result.z = decodeTexelChannel( texel, format, 2 );
        return result;
    }
    else if constexpr( std::is_same<Sample, float4>::value )
    {
        float4 result;
        result.x = decodeTexelChannel( texel, format, 0 );
        if( numChannels > 0 )
            result.y = decodeTexelChannel( texel, format, 1 );
        if( numChannels > 1 )
            result.z = decodeTexelChannel( texel, format, 2 );
        if( numChannels > 2 )
            result.w = decodeTexelChannel( texel, format, 3 );
        return result;
    }
    else
    {
        static_assert( false, "decodeTexel supports Sample = float, float2, float3, or float4" );
    }
}

template <class Sample>
HIP_DEMAND_INLINE Sample
fetchTexel( const DeviceContext& context, const DeviceTextureInfo& texture, const DeviceMipLevel& mip, int x, int y, bool& resident )
{
    resident   = false;
    bool valid = true;

    x = applyAddressMode( x, static_cast<int>( mip.width ), texture.addressMode[0], valid );
    y = applyAddressMode( y, static_cast<int>( mip.height ), texture.addressMode[1], valid );
    if( !valid )
    {
        // TODO_BS: do we need to have a texture border color here when addressMode == hipAddressModeBorder?
        resident = true;
        return Sample{};
    }

    uint32_t pageId         = INVALID_PAGE;
    size_t   pageByteOffset = 0;
    if( mip.mipTail )
    {
        pageId = texture.mipTailPage;
        pageByteOffset =
            mip.mipTailOffset + ( static_cast<size_t>( y ) * mip.width + static_cast<size_t>( x ) ) * texture.bytesPerTexel;
    }
    else
    {
        const uint32_t tileX = static_cast<uint32_t>( x ) / texture.tileWidth;
        const uint32_t tileY = static_cast<uint32_t>( y ) / texture.tileHeight;
        pageId               = mip.startPage + tileY * mip.tilesX + tileX;
        const size_t localX  = static_cast<size_t>( x ) % texture.tileWidth;
        const size_t localY  = static_cast<size_t>( y ) % texture.tileHeight;
        pageByteOffset       = ( localY * texture.tileWidth + localX ) * texture.bytesPerTexel;
    }

    const uint32_t resourceId = context.resourceTable.textureTiles.getResourceId( pageId );
    if( !isResourceResident( context, resourceId ) )
    {
        recordRequest( context, resourceId );
        return Sample{};
    }

    const size_t byteOffset = static_cast<size_t>( pageId ) * context.pageSize + pageByteOffset;
    resident                = true;
    return decodeTexel<Sample>( context.pageMemory.ptr + byteOffset, texture.format, texture.numChannels );
}

template <class Sample>
HIP_DEMAND_INLINE Sample
sampleMipLevel( const DeviceContext& context, const DeviceTextureInfo& texture, uint32_t mipLevel, float x, float y, bool& isResident )
{
    isResident               = false;
    const DeviceMipLevel mip = texture.getMipLevel( mipLevel );
    if( texture.normalizedCoords )
    {
        x = x * static_cast<float>( mip.width );
        y = y * static_cast<float>( mip.height );
    }

    if( texture.filterMode == hipFilterModePoint )
    {
        x = std::floorf( x );
        y = std::floorf( y );
        return fetchTexel<Sample>( context, texture, mip, static_cast<int>( x ), static_cast<int>( y ), isResident );
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

    const Sample t00 = fetchTexel<Sample>( context, texture, mip, x0, y0, resident00 );
    const Sample t10 = fetchTexel<Sample>( context, texture, mip, x0 + 1, y0, resident10 );
    const Sample t01 = fetchTexel<Sample>( context, texture, mip, x0, y0 + 1, resident01 );
    const Sample t11 = fetchTexel<Sample>( context, texture, mip, x0 + 1, y0 + 1, resident11 );

    isResident = resident00 && resident10 && resident01 && resident11;
    return ( 1.0f - a ) * ( 1.0f - b ) * t00 + a * ( 1.0f - b ) * t10 + ( 1.0f - a ) * b * t01 + a * b * t11;
}

template <class Sample>
HIP_DEMAND_INLINE Sample tex2DLod( const DeviceContext& context, uint32_t textureId, float x, float y, float lod, bool& isResident )
{
    isResident = false;
    if( textureId >= context.textureInfos.len )
    {
        isResident = true;
        return Sample{};
    }

    const uint32_t resourceId = context.resourceTable.textureInfos.getResourceId( textureId );
    if( !isResourceResident( context, resourceId ) )
    {
        recordRequest( context, resourceId );
        return Sample{};
    }

    const DeviceTextureInfo& texture = *context.textureInfos.ptr[textureId];

    lod = std::clamp( lod, 0.0f, static_cast<float>( texture.mipCount - 1 ) );
    if( texture.mipmapFilterMode == hipFilterModePoint )
    {
        const uint32_t mipLevel = static_cast<uint32_t>( std::floorf( lod + 0.5f ) );
        return sampleMipLevel<Sample>( context, texture, mipLevel, x, y, isResident );
    }

    const uint32_t mipLevel0 = static_cast<uint32_t>( std::floorf( lod ) );
    const uint32_t mipLevel1 = mipLevel0 + 1 < texture.mipCount ? mipLevel0 + 1 : mipLevel0;

    bool         resident0 = false;
    const Sample sample0   = sampleMipLevel<Sample>( context, texture, mipLevel0, x, y, resident0 );
    if( mipLevel0 == mipLevel1 )
    {
        isResident = resident0;
        return sample0;
    }

    bool         resident1 = false;
    const Sample sample1   = sampleMipLevel<Sample>( context, texture, mipLevel1, x, y, resident1 );

    isResident = resident0 && resident1;
    float t    = lod - static_cast<float>( mipLevel0 );
    return ( 1.0f - t ) * sample0 + t * sample1;
}

// TODO_BS: this implementation is generated by chatGPT. I'll need to do more research about it
template <class Sample>
HIP_DEMAND_INLINE Sample tex2DGrad( const DeviceContext& context, uint32_t textureId, float x, float y, float2 ddx, float2 ddy, bool& isResident )
{
    isResident = false;
    if( textureId >= context.textureInfos.len )
    {
        isResident = true;
        return Sample{};
    }

    const uint32_t resourceId = context.resourceTable.textureInfos.getResourceId( textureId );
    if( !isResourceResident( context, resourceId ) )
    {
        recordRequest( context, resourceId );
        return Sample{};
    }

    const DeviceTextureInfo& texture = *context.textureInfos.ptr[textureId];
    if( texture.normalizedCoords )
    {
        const DeviceMipLevel baseMip = texture.getMipLevel( 0 );
        ddx.x *= static_cast<float>( baseMip.width );
        ddx.y *= static_cast<float>( baseMip.height );
        ddy.x *= static_cast<float>( baseMip.width );
        ddy.y *= static_cast<float>( baseMip.height );
    }

    const float footprintXSquared = ddx.x * ddx.x + ddx.y * ddx.y;
    const float footprintYSquared = ddy.x * ddy.x + ddy.y * ddy.y;
    const float footprintSquared  = fmaxf( footprintXSquared, footprintYSquared );
    const float lod               = 0.5f * log2f( fmaxf( footprintSquared, 1.0f ) );

    return tex2DLod<Sample>( context, textureId, x, y, lod, isResident );
}

template <class Sample>
HIP_DEMAND_INLINE Sample tex2D( const DeviceContext& context, uint32_t textureId, float x, float y, bool& isResident )
{
    return tex2DLod<Sample>( context, textureId, x, y, 0.0f, isResident );
}

}  // namespace hip_demand::vmm
