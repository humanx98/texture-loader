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
