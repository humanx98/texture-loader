#pragma once

#include "DeviceContext.h"


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

HIP_DEMAND_INLINE void atomicSetBit( const DeviceSpan<uint32_t>& span, uint32_t index, int memoryOrder )
{
    uint32_t wordIdx = 0;
    uint32_t bitIdx  = 0;
    getWordIdxAndBitIdx( index, wordIdx, bitIdx );
#if defined( __HIPCC__ )
    __atomic_fetch_or( &span.ptr[wordIdx], 1u << bitIdx, memoryOrder );
#else
    atomicOr( &span.ptr[wordIdx], 1u << bitIdx );
#endif
}

HIP_DEMAND_INLINE void atomicUnsetBit(const DeviceSpan<uint32_t>& span, uint32_t index)
{
    uint32_t wordIdx = 0;
    uint32_t bitIdx  = 0;
    getWordIdxAndBitIdx( index, wordIdx, bitIdx );
    atomicAnd( &span.ptr[wordIdx], ~( 1u << bitIdx ) );
}

HIP_DEMAND_INLINE bool atomicCheckBit(const DeviceSpan<uint32_t>& span, uint32_t index, int memoryOrder)
{
    uint32_t wordIdx = 0;
    uint32_t bitIdx  = 0;
    getWordIdxAndBitIdx( index, wordIdx, bitIdx );
#if defined( __HIPCC__ )
    const uint32_t word = __atomic_load_n( &span.ptr[wordIdx], memoryOrder );
#else
    const uint32_t word = span.ptr[wordIdx];
#endif
    return ( word & ( 1u << bitIdx ) ) != 0;
}

HIP_DEMAND_INLINE uint32_t getUint4( const DeviceSpan<uint32_t>& words, const uint32_t index )
{
    const uint32_t wordIndex = index >> 3;
#if defined( __HIPCC__ )
    const uint32_t word = __atomic_load_n( &words.ptr[wordIndex], __ATOMIC_RELAXED );
#else
    const uint32_t word = words.ptr[wordIndex];
#endif
    return ( word >> 4u * ( index & 0x7u ) ) & 0xf;
}

HIP_DEMAND_INLINE void atomicClearUint4( const DeviceSpan<uint32_t>& words, const uint32_t index )
{
    const uint32_t wordIndex = index >> 3u;
    atomicAnd( &words.ptr[wordIndex], ~( 0xf << ( 4u * ( index & 0x7u ) ) ) );
}

}  // namespace hip_demand::vmm
