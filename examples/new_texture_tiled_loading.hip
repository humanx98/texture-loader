#include <DemandLoading/DeviceContext.h>
#include <DemandLoading/VmmTextureSampling.h>

#include <hip/hip_runtime.h>

namespace {

__device__ __forceinline__ uint8_t toUnorm8( float value )
{
    value = fminf( 1.0f, fmaxf( 0.0f, value ) );
    return static_cast<uint8_t>( value * 255.0f + 0.5f );
}

}  // namespace

extern "C" __global__ void renderVmmTexture( hip_demand::vmm::DeviceContext context,
                                              uint32_t textureId,
                                              uint8_t* output,
                                              uint32_t width,
                                              uint32_t height)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= width || y >= height )
        return;

    const size_t outputOffset = ( static_cast<size_t>( y ) * width + x ) * 4;
    bool resident = false;
    const float u = 2.0f * ( static_cast<float>( x ) + 0.5f ) / static_cast<float>( width );
    const float v = 2.0f * ( static_cast<float>( y ) + 0.5f ) / static_cast<float>( height );
    const float2 ddx = make_float2( 2.0f / static_cast<float>( width ), 0.0f );
    const float2 ddy = make_float2( 0.0f, 2.0f / static_cast<float>( height ) );
    const auto color = hip_demand::vmm::tex2DGrad<float3>( context, textureId, u, v, ddx, ddy, resident );
    if( resident )
    {
        output[outputOffset + 0] = toUnorm8( color.x );
        output[outputOffset + 1] = toUnorm8( color.y );
        output[outputOffset + 2] = toUnorm8( color.z );
        output[outputOffset + 3] = toUnorm8( 1.0f );
    } else
    {
        output[outputOffset + 0] = toUnorm8( 1.0f );
        output[outputOffset + 1] = toUnorm8( 0.0f );
        output[outputOffset + 2] = toUnorm8( 1.0f );
        output[outputOffset + 3] = toUnorm8( 1.0f );
    }
}

extern "C" __global__ void renderVmmTextureGrid( hip_demand::vmm::DeviceContext context,
                                                  uint8_t* output,
                                                  uint32_t outputWidth,
                                                  uint32_t outputHeight,
                                                  uint32_t textureCount,
                                                  uint32_t columnCount,
                                                  uint32_t rowCount )
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= outputWidth || y >= outputHeight )
        return;

    const uint32_t column    = x * columnCount / outputWidth;
    const uint32_t row       = y * rowCount / outputHeight;
    const uint32_t textureId = row * columnCount + column;
    if( textureId >= textureCount )
        return;

    const uint32_t cellX0     = column * outputWidth / columnCount;
    const uint32_t cellX1     = ( column + 1 ) * outputWidth / columnCount;
    const uint32_t cellY0     = row * outputHeight / rowCount;
    const uint32_t cellY1     = ( row + 1 ) * outputHeight / rowCount;
    const uint32_t cellWidth  = cellX1 - cellX0;
    const uint32_t cellHeight = cellY1 - cellY0;

    bool         resident = false;
    const float  u        = ( static_cast<float>( x - cellX0 ) + 0.5f ) / static_cast<float>( cellWidth );
    const float  v        = ( static_cast<float>( y - cellY0 ) + 0.5f ) / static_cast<float>( cellHeight );
    const float2 ddx      = make_float2( 1.0f / static_cast<float>( cellWidth ), 0.0f );
    const float2 ddy      = make_float2( 0.0f, 1.0f / static_cast<float>( cellHeight ) );
    const auto   color   = hip_demand::vmm::tex2DGrad<float3>( context, textureId, u, v, ddx, ddy, resident );

    const size_t outputOffset = ( static_cast<size_t>( y ) * outputWidth + x ) * 4;
    if( resident )
    {
        output[outputOffset + 0] = toUnorm8( color.x );
        output[outputOffset + 1] = toUnorm8( color.y );
        output[outputOffset + 2] = toUnorm8( color.z );
        output[outputOffset + 3] = toUnorm8( 1.0f );
    }
    else
    {
        output[outputOffset + 0] = toUnorm8( 1.0f );
        output[outputOffset + 1] = toUnorm8( 0.0f );
        output[outputOffset + 2] = toUnorm8( 1.0f );
        output[outputOffset + 3] = toUnorm8( 1.0f );
    }
}

extern "C" __global__ void renderVmmTextureMipGrid( hip_demand::vmm::DeviceContext context,
                                                     uint8_t* output,
                                                     uint32_t outputWidth,
                                                     uint32_t outputHeight,
                                                     uint32_t textureCount,
                                                     uint32_t columnCount,
                                                     uint32_t rowCount )
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= outputWidth || y >= outputHeight )
        return;

    const uint32_t column    = x * columnCount / outputWidth;
    const uint32_t row       = y * rowCount / outputHeight;
    const uint32_t textureId = row * columnCount + column;
    if( textureId >= textureCount )
        return;

    const uint32_t cellX0     = column * outputWidth / columnCount;
    const uint32_t cellX1     = ( column + 1 ) * outputWidth / columnCount;
    const uint32_t cellY0     = row * outputHeight / rowCount;
    const uint32_t cellY1     = ( row + 1 ) * outputHeight / rowCount;
    const uint32_t cellWidth  = cellX1 - cellX0;
    const uint32_t cellHeight = cellY1 - cellY0;
    const uint32_t localX     = x - cellX0;
    const uint32_t localY     = y - cellY0;
    const size_t   outputOffset = ( static_cast<size_t>( y ) * outputWidth + x ) * 4;

    // The texture metadata must be resident before the mip count can be inspected.
    const uint32_t textureResourceId = context.resourceTable.textureInfos.getResourceId( textureId );
    if( !hip_demand::vmm::isResourceResident( context, textureResourceId ) )
    {
        bool resident = false;
        hip_demand::vmm::tex2DLod<float3>( context, textureId, 0.5f, 0.5f, 0.0f, resident );
        output[outputOffset + 0] = 255;
        output[outputOffset + 1] = 0;
        output[outputOffset + 2] = 255;
        output[outputOffset + 3] = 255;
        return;
    }

    const hip_demand::vmm::DeviceTextureInfo& texture = *context.textureInfos.ptr[textureId];

    // Pack a conventional mip chain into each grid cell: mip 0 fills the left
    // two thirds and all smaller levels are stacked, at relative size, on the right.
    const uint32_t baseWidth = max( 1u, cellWidth * 2u / 3u );
    uint32_t       mipLevel  = 0;
    uint32_t       mipX      = 0;
    uint32_t       mipY      = 0;
    uint32_t       mipWidth  = baseWidth;
    uint32_t       mipHeight = cellHeight;
    bool           inMip     = localX < baseWidth;

    if( !inMip )
    {
        mipY = 0;
        for( uint32_t level = 1; level < texture.mipCount; ++level )
        {
            const uint32_t levelWidth  = max( 1u, baseWidth >> level );
            const uint32_t levelHeight = max( 1u, cellHeight >> level );
            if( localX >= baseWidth && localX < baseWidth + levelWidth && localY >= mipY
                && localY < mipY + levelHeight )
            {
                mipLevel  = level;
                mipX      = baseWidth;
                mipWidth  = levelWidth;
                mipHeight = levelHeight;
                inMip     = true;
                break;
            }
            mipY += levelHeight;
        }
    }

    if( !inMip )
    {
        output[outputOffset + 0] = 0;
        output[outputOffset + 1] = 0;
        output[outputOffset + 2] = 0;
        output[outputOffset + 3] = 255;
        return;
    }

    const float u = ( static_cast<float>( localX - mipX ) + 0.5f ) / static_cast<float>( mipWidth );
    const float v = ( static_cast<float>( localY - mipY ) + 0.5f ) / static_cast<float>( mipHeight );
    bool        resident = false;
    const auto  color = hip_demand::vmm::tex2DLod<float3>( context, textureId, u, v,
                                                           static_cast<float>( mipLevel ), resident );

    if( resident )
    {
        output[outputOffset + 0] = toUnorm8( color.x );
        output[outputOffset + 1] = toUnorm8( color.y );
        output[outputOffset + 2] = toUnorm8( color.z );
        output[outputOffset + 3] = 255;
    }
    else
    {
        output[outputOffset + 0] = 255;
        output[outputOffset + 1] = 0;
        output[outputOffset + 2] = 255;
        output[outputOffset + 3] = 255;
    }
}
