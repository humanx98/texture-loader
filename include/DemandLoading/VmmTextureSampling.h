#pragma once

#include "DeviceContext.h"
#include "VmmDeviceUtils.h"
#include <algorithm>
#include <cmath>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <type_traits>


namespace hip_demand::vmm {

HIP_DEMAND_INLINE void recordRequest( const DeviceContext& context, uint32_t resourceId )
{
// #if defined( HIP_ENABLE_WARP_SYNC_BUILTINS )
//     const uint64_t activeLanes   = __activemask();
//     const uint64_t matchingLanes = __match_any_sync( activeLanes, resourceId );
//     const uint32_t leaderLane    = static_cast<uint32_t>( __ffsll( static_cast<long long>( matchingLanes ) ) - 1 );
//     if( static_cast<uint32_t>( __lane_id() ) != leaderLane )
//         return;
// #endif

    atomicSetBit( context.referenceBits, resourceId );
}
HIP_DEMAND_INLINE bool isResourceResident( const DeviceContext& context, uint32_t resourceId ) { return checkBitSet(context.residenceBits, resourceId); }

HIP_DEMAND_INLINE int wrapCoordinate( int coordinate, int extent )
{
    coordinate %= extent;
    return coordinate < 0 ? coordinate + extent : coordinate;
}

HIP_DEMAND_INLINE int mirrorCoordinate( int coordinate, int extent )
{
    const int period = extent * 2;
    coordinate %= period;
    if( coordinate < 0 )
        coordinate += period;
    return coordinate < extent ? coordinate : period - coordinate - 1;
}

HIP_DEMAND_INLINE int applyAddressMode( int coordinate, int extent, uint32_t mode )
{
    switch( static_cast<hipTextureAddressMode>( mode ) )
    {
        case hipAddressModeWrap:
            return wrapCoordinate( coordinate, extent );
        case hipAddressModeMirror:
            return mirrorCoordinate( coordinate, extent );
        case hipAddressModeClamp:
        default:
            return std::clamp( coordinate, 0, extent - 1 );
    }
}

HIP_DEMAND_INLINE int2 applyAddressModePair( int coordinate0, int coordinate1, int extent, uint32_t mode )
{
    switch( static_cast<hipTextureAddressMode>( mode ) )
    {
        case hipAddressModeWrap:
            return make_int2( wrapCoordinate( coordinate0, extent ), wrapCoordinate( coordinate1, extent ) );
        case hipAddressModeMirror:
            return make_int2( mirrorCoordinate( coordinate0, extent ), mirrorCoordinate( coordinate1, extent ) );
        case hipAddressModeClamp:
        default:
            return make_int2( std::clamp( coordinate0, 0, extent - 1 ), std::clamp( coordinate1, 0, extent - 1 ) );
    }
}

HIP_DEMAND_INLINE float decodeChannelValue( uint8_t value )
{
    return value * ( 1.0f / 255.0f );
}
HIP_DEMAND_INLINE float decodeChannelValue( int8_t value )
{
    return fmaxf( -1.0f, value * ( 1.0f / 127.0f ) );
}
HIP_DEMAND_INLINE float decodeChannelValue( uint16_t value )
{
    return value * ( 1.0f / 65535.0f );
}
HIP_DEMAND_INLINE float decodeChannelValue( int16_t value )
{
    return fmaxf( -1.0f, value * ( 1.0f / 32767.0f ) );
}
HIP_DEMAND_INLINE float decodeChannelValue( uint32_t value )
{
    return static_cast<float>( value ) * ( 1.0f / 4294967295.0f );
}
HIP_DEMAND_INLINE float decodeChannelValue( int32_t value )
{
    return fmaxf( -1.0f, static_cast<float>( value ) * ( 1.0f / 2147483647.0f ) );
}
HIP_DEMAND_INLINE float decodeChannelValue( __half value )
{
    return __half2float( value );
}
HIP_DEMAND_INLINE float decodeChannelValue( float value )
{
    return value;
}

template <class Sample, uint32_t NumChannels, class Vector>
HIP_DEMAND_INLINE Sample decodeVectorTexel( const Vector& value )
{
    const float x = decodeChannelValue( value.x );
    if constexpr( std::is_same<Sample, float>::value )
        return x;
    else
    {
        float y = 0.0f;
        if constexpr( NumChannels > 1 )
            y = decodeChannelValue( value.y );
        if constexpr( std::is_same<Sample, float2>::value )
            return make_float2( x, y );
        else
        {
            float z = 0.0f;
            if constexpr( NumChannels > 2 )
                z = decodeChannelValue( value.z );
            if constexpr( std::is_same<Sample, float3>::value )
                return make_float3( x, y, z );
            else if constexpr( std::is_same<Sample, float4>::value )
            {
                float w = 0.0f;
                if constexpr( NumChannels > 3 )
                    w = decodeChannelValue( value.w );
                return make_float4( x, y, z, w );
            }
            else
                static_assert( std::is_same<Sample, void>::value, "Unsupported texture sample type" );
        }
    }
}

template <class Sample, uint32_t NumChannels, class Channel>
HIP_DEMAND_INLINE Sample decodePackedChannels( const uint8_t* texel )
{
    using PackedChannels       = HIP_vector_type<Channel, NumChannels>;
    const PackedChannels value = *reinterpret_cast<const PackedChannels*>( texel );
    return decodeVectorTexel<Sample, NumChannels>( value );
}

template <class Sample, uint32_t NumChannels>
HIP_DEMAND_INLINE Sample decodeHalfChannels( const uint8_t* texel )
{
    const __half* channels = reinterpret_cast<const __half*>( texel );
    const float   x        = decodeChannelValue( channels[0] );
    if constexpr( std::is_same<Sample, float>::value )
        return x;
    else
    {
        float y = 0.0f;
        if constexpr( NumChannels > 1 )
            y = decodeChannelValue( channels[1] );
        if constexpr( std::is_same<Sample, float2>::value )
            return make_float2( x, y );
        else
        {
            float z = 0.0f;
            if constexpr( NumChannels > 2 )
                z = decodeChannelValue( channels[2] );
            if constexpr( std::is_same<Sample, float3>::value )
                return make_float3( x, y, z );
            else if constexpr( std::is_same<Sample, float4>::value )
            {
                float w = 0.0f;
                if constexpr( NumChannels > 3 )
                    w = decodeChannelValue( channels[3] );
                return make_float4( x, y, z, w );
            }
            else
                static_assert( std::is_same<Sample, void>::value, "Unsupported texture sample type" );
        }
    }
}

template <class Sample, uint32_t NumChannels>
HIP_DEMAND_INLINE Sample decodeTexelChannels( const uint8_t* texel, hipArray_Format format )
{
    switch( format )
    {
        case HIP_AD_FORMAT_UNSIGNED_INT8:
            return decodePackedChannels<Sample, NumChannels, uint8_t>( texel );
        case HIP_AD_FORMAT_SIGNED_INT8:
            return decodePackedChannels<Sample, NumChannels, int8_t>( texel );
        case HIP_AD_FORMAT_UNSIGNED_INT16:
            return decodePackedChannels<Sample, NumChannels, uint16_t>( texel );
        case HIP_AD_FORMAT_SIGNED_INT16:
            return decodePackedChannels<Sample, NumChannels, int16_t>( texel );
        case HIP_AD_FORMAT_UNSIGNED_INT32:
            return decodePackedChannels<Sample, NumChannels, uint32_t>( texel );
        case HIP_AD_FORMAT_SIGNED_INT32:
            return decodePackedChannels<Sample, NumChannels, int32_t>( texel );
        case HIP_AD_FORMAT_HALF:
            return decodeHalfChannels<Sample, NumChannels>( texel );
        case HIP_AD_FORMAT_FLOAT:
            return decodePackedChannels<Sample, NumChannels, float>( texel );
        default:
            return Sample{};
    }
}

template <class Sample>
HIP_DEMAND_INLINE Sample decodeTexel( const uint8_t* texel, hipArray_Format format, uint32_t numChannels )
{
    switch( numChannels )
    {
        case 1:
            return decodeTexelChannels<Sample, 1>( texel, format );
        case 2:
            return decodeTexelChannels<Sample, 2>( texel, format );
        case 3:
            return decodeTexelChannels<Sample, 3>( texel, format );
        case 4:
            return decodeTexelChannels<Sample, 4>( texel, format );
        default:
            return Sample{};
    }
}

template <class Sample>
HIP_DEMAND_INLINE Sample
fetchTexel( const DeviceContext& context, const DeviceTextureInfo& texture, const DeviceMipLevel& mip, int x, int y, bool& resident )
{
    resident = false;

    x = applyAddressMode( x, static_cast<int>( mip.width ), texture.addressMode[0] );
    y = applyAddressMode( y, static_cast<int>( mip.height ), texture.addressMode[1] );

    uint32_t pageId         = INVALID_PAGE;
    uint32_t pageByteOffset = 0;
    if( mip.mipTail )
    {
        pageId = texture.mipTailPage;
        pageByteOffset =
            mip.mipTailOffset + ( static_cast<uint32_t>( y ) * mip.width + static_cast<uint32_t>( x ) ) * texture.bytesPerTexel;
    }
    else
    {
        const uint32_t tileX  = static_cast<uint32_t>( x ) >> texture.tileWidthShift;
        const uint32_t tileY  = static_cast<uint32_t>( y ) >> texture.tileHeightShift;
        pageId                = mip.startPage + tileY * mip.tilesX + tileX;
        const uint32_t localX = static_cast<uint32_t>( x ) & texture.tileWidthMask;
        const uint32_t localY = static_cast<uint32_t>( y ) & texture.tileHeightMask;
        pageByteOffset        = ( localY * texture.tileWidth + localX ) * texture.bytesPerTexel;
    }

    const uint32_t resourceId = context.resourceTable.textureTiles.getResourceId( pageId );
    resident                  = isResourceResident( context, resourceId );
    if( !resident )
    {
        recordRequest( context, resourceId );
        return Sample{};
    }
    else if( context.requestIfResident )
    {
        recordRequest( context, resourceId );
    }

    const size_t byteOffset = static_cast<size_t>( pageId ) * context.pageSize + pageByteOffset;
    resident                = true;
    return decodeTexel<Sample>( context.pageMemory.ptr + byteOffset, texture.format, texture.numChannels );
}

HIP_DEMAND_INLINE const uint8_t* resolveTexturePage( const DeviceContext& context, uint32_t pageId )
{
    const uint32_t resourceId = context.resourceTable.textureTiles.getResourceId( pageId );
    const bool     resident   = isResourceResident( context, resourceId );
    if( !resident || context.requestIfResident )
        recordRequest( context, resourceId );

    return resident ? context.pageMemory.ptr + static_cast<size_t>( pageId ) * context.pageSize : nullptr;
}

template <class Sample>
HIP_DEMAND_INLINE void fetchBilinearTexels( const DeviceContext&     context,
                                            const DeviceTextureInfo& texture,
                                            const DeviceMipLevel&    mip,
                                            int                      x0,
                                            int                      y0,
                                            Sample&                  t00,
                                            Sample&                  t10,
                                            Sample&                  t01,
                                            Sample&                  t11,
                                            bool&                    resident )
{
    const int2 x = applyAddressModePair( x0, x0 + 1, static_cast<int>( mip.width ), texture.addressMode[0] );
    const int2 y = applyAddressModePair( y0, y0 + 1, static_cast<int>( mip.height ), texture.addressMode[1] );
    x0           = x.x;
    y0           = y.x;
    const int x1 = x.y;
    const int y1 = y.y;

    uint32_t pageId = INVALID_PAGE;
    uint32_t offset00{};
    uint32_t offset10{};
    uint32_t offset01{};
    uint32_t offset11{};

    if( mip.mipTail )
    {
        pageId                  = texture.mipTailPage;
        const uint32_t rowBytes = mip.width * texture.bytesPerTexel;
        offset00 = mip.mipTailOffset + static_cast<uint32_t>( y0 ) * rowBytes + static_cast<uint32_t>( x0 ) * texture.bytesPerTexel;
        offset10 = mip.mipTailOffset + static_cast<uint32_t>( y0 ) * rowBytes + static_cast<uint32_t>( x1 ) * texture.bytesPerTexel;
        offset01 = mip.mipTailOffset + static_cast<uint32_t>( y1 ) * rowBytes + static_cast<uint32_t>( x0 ) * texture.bytesPerTexel;
        offset11 = mip.mipTailOffset + static_cast<uint32_t>( y1 ) * rowBytes + static_cast<uint32_t>( x1 ) * texture.bytesPerTexel;
    }
    else
    {
        const uint32_t tileX0   = static_cast<uint32_t>( x0 ) >> texture.tileWidthShift;
        const uint32_t tileX1   = static_cast<uint32_t>( x1 ) >> texture.tileWidthShift;
        const uint32_t tileY0   = static_cast<uint32_t>( y0 ) >> texture.tileHeightShift;
        const uint32_t tileY1   = static_cast<uint32_t>( y1 ) >> texture.tileHeightShift;
        const uint32_t localX0  = static_cast<uint32_t>( x0 ) & texture.tileWidthMask;
        const uint32_t localX1  = static_cast<uint32_t>( x1 ) & texture.tileWidthMask;
        const uint32_t localY0  = static_cast<uint32_t>( y0 ) & texture.tileHeightMask;
        const uint32_t localY1  = static_cast<uint32_t>( y1 ) & texture.tileHeightMask;
        const uint32_t rowBytes = texture.tileWidth * texture.bytesPerTexel;
        offset00                = localY0 * rowBytes + localX0 * texture.bytesPerTexel;
        offset10                = localY0 * rowBytes + localX1 * texture.bytesPerTexel;
        offset01                = localY1 * rowBytes + localX0 * texture.bytesPerTexel;
        offset11                = localY1 * rowBytes + localX1 * texture.bytesPerTexel;

        const uint32_t pageRow0 = mip.startPage + tileY0 * mip.tilesX;
        pageId                  = pageRow0 + tileX0;

        if( tileX0 != tileX1 || tileY0 != tileY1 )
        {
            const uint32_t pageRow1 = mip.startPage + tileY1 * mip.tilesX;
            const uint32_t pageId10 = pageRow0 + tileX1;
            const uint32_t pageId01 = pageRow1 + tileX0;
            const uint32_t pageId11 = pageRow1 + tileX1;

            const uint8_t* page00 = resolveTexturePage( context, pageId );
            const uint8_t* page10 = pageId10 == pageId ? page00 : resolveTexturePage( context, pageId10 );
            const uint8_t* page01 = pageId01 == pageId ? page00 : resolveTexturePage( context, pageId01 );
            const uint8_t* page11{};
            if( pageId11 == pageId10 )
                page11 = page10;
            else if( pageId11 == pageId01 )
                page11 = page01;
            else
                page11 = resolveTexturePage( context, pageId11 );

            resident = page00 && page10 && page01 && page11;
            if( resident )
            {
                t00 = decodeTexel<Sample>( page00 + offset00, texture.format, texture.numChannels );
                t10 = decodeTexel<Sample>( page10 + offset10, texture.format, texture.numChannels );
                t01 = decodeTexel<Sample>( page01 + offset01, texture.format, texture.numChannels );
                t11 = decodeTexel<Sample>( page11 + offset11, texture.format, texture.numChannels );
            }
            return;
        }
    }

    const uint32_t resourceId = context.resourceTable.textureTiles.getResourceId( pageId );
    resident                  = isResourceResident( context, resourceId );
    if( !resident )
    {
        recordRequest( context, resourceId );
        return;
    }
    if( context.requestIfResident )
        recordRequest( context, resourceId );

    const uint8_t* page = context.pageMemory.ptr + static_cast<size_t>( pageId ) * context.pageSize;
    t00                 = decodeTexel<Sample>( page + offset00, texture.format, texture.numChannels );
    t10                 = decodeTexel<Sample>( page + offset10, texture.format, texture.numChannels );
    t01                 = decodeTexel<Sample>( page + offset01, texture.format, texture.numChannels );
    t11                 = decodeTexel<Sample>( page + offset11, texture.format, texture.numChannels );
}

template <class Sample>
HIP_DEMAND_INLINE Sample
sampleMipLevel( const DeviceContext& context, const DeviceTextureInfo& texture, uint32_t mipLevel, float x, float y, bool& resident )
{
    resident                 = false;
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
        return fetchTexel<Sample>( context, texture, mip, static_cast<int>( x ), static_cast<int>( y ), resident );
    }

    x = x - 0.5f;
    y = y - 0.5f;

    const int   x0 = static_cast<int>( std::floorf( x ) );
    const int   y0 = static_cast<int>( std::floorf( y ) );
    const float a  = x - static_cast<float>( x0 );
    const float b  = y - static_cast<float>( y0 );

    Sample t00{};
    Sample t10{};
    Sample t01{};
    Sample t11{};
    fetchBilinearTexels( context, texture, mip, x0, y0, t00, t10, t01, t11, resident );

    return ( 1.0f - a ) * ( 1.0f - b ) * t00 + a * ( 1.0f - b ) * t10 + ( 1.0f - a ) * b * t01 + a * b * t11;
}

template <class Sample>
HIP_DEMAND_INLINE Sample tex2DLod( const DeviceContext& context, uint32_t textureId, float x, float y, float lod, bool& resident )
{
    resident = false;
    if( textureId >= context.textureInfos.len )
    {
        resident = true;
        return Sample{};
    }

    // no need to check isResourceResident if we could just try to load texture info and check if it's null
    // const uint32_t resourceId = context.resourceTable.textureInfos.getResourceId( textureId );
    // resident                  = isResourceResident( context, resourceId );

    const DeviceTextureInfo* texture = context.textureInfos.ptr[textureId];
    if( !texture )
    {
        recordRequest( context, context.resourceTable.textureInfos.getResourceId( textureId ) );
        return Sample{};
    }
    else if( context.requestIfResident )
    {
        recordRequest( context, context.resourceTable.textureInfos.getResourceId( textureId ) );
    }

    resident = true;

    lod = std::clamp( lod, 0.0f, static_cast<float>( texture->mipCount - 1 ) );
    if( texture->mipmapFilterMode == hipFilterModePoint )
    {
        const uint32_t mipLevel = static_cast<uint32_t>( std::floorf( lod + 0.5f ) );
        return sampleMipLevel<Sample>( context, *texture, mipLevel, x, y, resident );
    }

    const uint32_t mipLevel0 = static_cast<uint32_t>( std::floorf( lod ) );
    const uint32_t mipLevel1 = mipLevel0 + 1 < texture->mipCount ? mipLevel0 + 1 : mipLevel0;

    bool         resident0 = false;
    const Sample sample0   = sampleMipLevel<Sample>( context, *texture, mipLevel0, x, y, resident0 );
    if( mipLevel0 == mipLevel1 )
    {
        resident = resident0;
        return sample0;
    }

    bool         resident1 = false;
    const Sample sample1   = sampleMipLevel<Sample>( context, *texture, mipLevel1, x, y, resident1 );

    resident = resident0 && resident1;
    float t    = lod - static_cast<float>( mipLevel0 );
    return ( 1.0f - t ) * sample0 + t * sample1;
}

// TODO_BS: this implementation is generated by chatGPT. I'll need to do more research about it
template <class Sample>
HIP_DEMAND_INLINE Sample tex2DGrad( const DeviceContext& context, uint32_t textureId, float x, float y, float2 ddx, float2 ddy, bool& resident )
{
    resident = false;
    if( textureId >= context.textureInfos.len )
    {
        resident = true;
        return Sample{};
    }

    const uint32_t resourceId = context.resourceTable.textureInfos.getResourceId( textureId );
    resident = isResourceResident( context, resourceId );
    if( !resident )
    {
        recordRequest( context, resourceId );
        return Sample{};
    }
    else if (context.requestIfResident)
    {
        recordRequest( context, resourceId );
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

    return tex2DLod<Sample>( context, textureId, x, y, lod, resident );
}

template <class Sample>
HIP_DEMAND_INLINE Sample tex2D( const DeviceContext& context, uint32_t textureId, float x, float y, bool& resident )
{
    return tex2DLod<Sample>( context, textureId, x, y, 0.0f, resident );
}

}  // namespace hip_demand::vmm
