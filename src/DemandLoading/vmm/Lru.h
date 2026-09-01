#pragma once
#include <DemandLoading/DeviceContext.h>

namespace hip_demand::vmm {

inline constexpr uint32_t lruMax          = 14u;
inline constexpr uint32_t lruNonEvictable = 15u;
inline constexpr uint32_t lruThresholdMin = 2u;

HIP_DEMAND_INLINE uint32_t getLru( const uint32_t index, uint32_t* words )
{
    const uint32_t wordIndex = index >> 3;
    const uint32_t shiftVal  = 4 * ( index & 0x7 );
    return ( words[wordIndex] >> shiftVal ) & 0xf;
}

HIP_DEMAND_INLINE void clearLru( const uint32_t index, uint32_t* words )
{
    const uint32_t wordIndex = index >> 3u;
    words[wordIndex] &= ~( 0xf << ( 4u * ( index & 0x7u ) ) );
}

HIP_DEMAND_INLINE void addLru( const uint32_t index, const uint32_t val, uint32_t* words )
{
    const uint32_t wordIndex = index >> 3;
    words[wordIndex] += val << ( 4u * ( index & 0x7u ) );
}

}  // namespace hip_demand::vmm