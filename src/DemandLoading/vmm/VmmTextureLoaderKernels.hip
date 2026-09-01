#include <DemandLoading/DeviceContext.h>
#include "Lru.h"
#include <cstdint>
#include <hip/hip_runtime.h>

namespace hip_demand::vmm {

HIP_DEMAND_INLINE uint32_t incrementLru( const uint32_t count, const uint32_t launchNum, const uint32_t resourceId )
{
    /**
     * Increments the LRU counter logarithmically. The increment interval doubles
     * as the counter grows.
     *
     * Example for resourceId = 0 and an initial LRU of 0:
     *
     *   Launch:  1  2  3  4  5  6  7  8
     *   LRU:     1  2  2  3  3  3  3  4
     *
     * `resourceId` shifts the increment timing so resources do not all increment
     * during the same launch. For example, at LRU = 2, each resource increments
     * once every 4 launches, but at a different time:
     *
     *   Launch:       1  2  3  4  5  6  7  8
     *   Resource 0:   -  -  -  +  -  -  -  +
     *   Resource 1:   -  -  +  -  -  -  +  -
     *   Resource 2:   -  +  -  -  -  +  -  -
     *   Resource 3:   +  -  -  -  +  -  -  -
     *
     * Here, `+` indicates an LRU increment.
     */
    if (count >= lruMax)
        return count;
        
    const uint32_t mask = ( 1u << count ) - 1;
    return ( ( mask & (launchNum + resourceId) ) == 0 ) ? count + 1u : count;
}

HIP_DEMAND_INLINE uint32_t calcListStartIndex( const uint32_t numSetBits, uint32_t* counter )
{
    const uint32_t laneId = __lane_id();
    const uint64_t activeMask = __activemask();
    uint32_t index = numSetBits;

    #pragma unroll
    for( uint32_t i = 1; i < __builtin_amdgcn_wavefrontsize(); i *= 2 )
    {
        const uint32_t n = __shfl_up_sync( activeMask, index, i );
        if( laneId >= i )
            index += n;
    }

    const uint32_t waveRequestCount = __reduce_add_sync( activeMask, numSetBits );
    index = __shfl_up_sync( activeMask, index, 1 );

    uint32_t startIndex = 0;
    if( laneId == 0 )
    {
        index = 0;
        if( waveRequestCount != 0 )
            startIndex = atomicAdd( counter, waveRequestCount );
    }

    return __shfl_sync(activeMask, startIndex, 0 ) + index;
}

HIP_DEMAND_INLINE void collectRequests( DeviceContext& context, uint32_t requestBits, const uint32_t resourceBitOffset )
{
    const uint32_t numRequestBits = __popc( requestBits );

    uint32_t requestIndex = calcListStartIndex( numRequestBits, &context.counters.ptr[static_cast<uint32_t>( CounterIndex::RequestedResources )] );
    // get requested resource Ids
    while( requestBits != 0 && ( requestIndex < context.requestedResources.len ) )
    {
        const uint32_t bitIndex = __ffs( requestBits ) - 1;
        requestBits ^= ( 1U << bitIndex );
        context.requestedResources.ptr[requestIndex++] = resourceBitOffset + bitIndex;
    }
}

HIP_DEMAND_INLINE void collectLruEvictionCandidates( DeviceContext& context,
    const uint32_t launchNum,
    const uint32_t lruThreshold,
    const uint32_t referenceWord,
    const uint32_t residenceWord,
    const uint32_t resourceBitOffset )
{
    // reset lru values
    {
        uint32_t resetLruBits = referenceWord & residenceWord;
        while( resetLruBits != 0 )
        {
            const uint32_t bitIndex = __ffs( resetLruBits ) - 1;
            resetLruBits ^= ( 1U << bitIndex );
            const uint32_t resourceId = resourceBitOffset + bitIndex;
            const uint32_t lruVal = getLru( resourceId, context.lru.ptr );
            if( lruVal != 0 && lruVal != lruNonEvictable )
                clearLru( resourceId, context.lru.ptr );
        }
    }

    // num bits bigger than threshold
    {
        const uint32_t evictionBits = ~referenceWord & residenceWord;
        uint32_t numEvictionBits = 0;
        uint32_t bits = evictionBits;
        while( bits != 0 )
        {
            const uint32_t bitIndex = __ffs( bits ) - 1;
            bits ^= ( 1U << bitIndex );
            const uint32_t resourceId = resourceBitOffset + bitIndex;

            const uint32_t oldVal = getLru( resourceId, context.lru.ptr );
            const uint32_t newVal = incrementLru( oldVal, launchNum, resourceId );
            if( newVal != oldVal )
                addLru( resourceId, 1, context.lru.ptr );
            if( newVal >= lruThreshold && newVal < lruNonEvictable )
                numEvictionBits++;
        }

        uint32_t candidateIndex = calcListStartIndex( numEvictionBits, &context.counters.ptr[static_cast<uint32_t>( CounterIndex::EvictionCandidates )] );
        bits = evictionBits;
        while( bits != 0 && ( candidateIndex < context.evictionCandidates.len ) )
        {
            const uint32_t bitIndex = __ffs( bits ) - 1;
            bits ^= ( 1U << bitIndex );
            const uint32_t resourceId = resourceBitOffset + bitIndex;

            const uint32_t lruVal = getLru( resourceId, context.lru.ptr );
            if( lruVal >= lruThreshold && lruVal != lruNonEvictable )
                context.evictionCandidates.ptr[candidateIndex++] = { resourceId, lruVal };
        }
    }
}

template<bool enableEviction>
HIP_DEMAND_INLINE void collectRequestsAndEvictionCandidates( DeviceContext& context, uint32_t launchNum, uint32_t lruThreshold )
{
    const uint32_t startIndex = 0;
    const uint32_t endIndex = context.referenceBits.len;

    uint32_t wordIdx = blockIdx.x * blockDim.x + threadIdx.x;

    while( wordIdx < endIndex )
    {
        const uint32_t resourceBitOffset = wordIdx * 32;
        const uint32_t referenceWord = context.referenceBits.ptr[wordIdx];
        const uint32_t residenceWord = context.residenceBits.ptr[wordIdx];
        collectRequests(context, referenceWord & ~residenceWord, resourceBitOffset);
        if constexpr (enableEviction)
        {
            collectLruEvictionCandidates(context, launchNum, lruThreshold, referenceWord, residenceWord, resourceBitOffset);
        }

        wordIdx += gridDim.x * blockDim.x;
    }
}

}

extern "C" __global__ void collectRequestsKernel( hip_demand::vmm::DeviceContext context, uint32_t launchNum, uint32_t lruThreshold )
{
    hip_demand::vmm::collectRequestsAndEvictionCandidates<false>(context, launchNum, lruThreshold);
}

extern "C" __global__ void collectRequestsAndEvictionCandidatesKernel( hip_demand::vmm::DeviceContext context, uint32_t launchNum, uint32_t lruThreshold )
{
    hip_demand::vmm::collectRequestsAndEvictionCandidates<true>(context, launchNum, lruThreshold);
}
