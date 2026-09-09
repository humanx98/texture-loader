#include <DemandLoading/DeviceContext.h>
#include <DemandLoading/VmmDeviceUtils.h>
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

HIP_DEMAND_INLINE uint32_t atomicIncrementLru( const DeviceSpan<uint32_t>& lru, const uint32_t pageId, const uint32_t launchNum )
{
    const uint32_t wordIndex = pageId >> 3u;
    const uint32_t shift     = 4u * ( pageId & 0x7u );
    const uint32_t valueMask = 0xfu << shift;
    uint32_t*      word      = &lru.ptr[wordIndex];

    uint32_t expected = __atomic_load_n( word, __ATOMIC_RELAXED );
    while( true )
    {
        const uint32_t oldValue = ( expected & valueMask ) >> shift;
        const uint32_t newValue = incrementLru( oldValue, launchNum, pageId );
        if( newValue == oldValue )
            return oldValue;

        const uint32_t desired = ( expected & ~valueMask ) | ( newValue << shift );
        if( __atomic_compare_exchange_n( word, &expected, desired, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED ) )
            return newValue;
    }
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

HIP_DEMAND_INLINE void collectEvictionCandidates( DeviceContext& context,
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
            
            // eviction logic should be applied only to texture tiles
            const uint32_t resourceId = resourceBitOffset + bitIndex;
            if( !context.resourceTable.textureTiles.contains(resourceId) )
                continue;

            const uint32_t pageId = context.resourceTable.textureTiles.getLocalId(resourceId);
            const uint32_t lruVal = getUint4( context.lru, pageId );
            if( lruVal != 0 && lruVal != lruNonEvictable )
                atomicClearUint4( context.lru, pageId );
        }
    }

    // num bits bigger than threshold
    {
        const uint32_t evictionBits  = ~referenceWord & residenceWord;
        uint32_t       candidateBits = 0;
        uint32_t       bits          = evictionBits;
        while( bits != 0 )
        {
            const uint32_t bitIndex = __ffs( bits ) - 1;
            bits ^= ( 1U << bitIndex );

            // eviction logic should be applied only to texture tiles
            const uint32_t resourceId = resourceBitOffset + bitIndex;
            if( !context.resourceTable.textureTiles.contains( resourceId ) )
                continue;

            const uint32_t pageId = context.resourceTable.textureTiles.getLocalId( resourceId );
            const uint32_t newVal = atomicIncrementLru( context.lru, pageId, launchNum );
            if( newVal >= lruThreshold && newVal < lruNonEvictable )
                candidateBits |= 1u << bitIndex;
        }

        const uint32_t numEvictionBits = __popc( candidateBits );
        uint32_t       candidateIndex =
            calcListStartIndex( numEvictionBits, &context.counters.ptr[static_cast<uint32_t>( CounterIndex::EvictionCandidates )] );
        bits = candidateBits;
        while( bits != 0 && ( candidateIndex < context.evictionCandidates.len ) )
        {
            const uint32_t bitIndex = __ffs( bits ) - 1;
            bits ^= ( 1U << bitIndex );

            // eviction logic should be applied only to texture tiles
            const uint32_t resourceId = resourceBitOffset + bitIndex;
            if( !context.resourceTable.textureTiles.contains( resourceId ) )
                continue;

            const uint32_t pageId = context.resourceTable.textureTiles.getLocalId( resourceId );
            const uint32_t lruVal = getUint4( context.lru, pageId );
            context.evictionCandidates.ptr[candidateIndex++] = { pageId, lruVal };
        }
    }
}

HIP_DEMAND_INLINE void collectRequestsAndEvictionCandidates( DeviceContext& context, uint32_t launchNum, uint32_t lruThreshold )
{
    uint32_t wordIdx = blockIdx.x * blockDim.x + threadIdx.x;

    while( wordIdx < context.referenceBits.len )
    {
        const uint32_t resourceBitOffset = wordIdx * 32;
        const uint32_t referenceWord = context.referenceBits.ptr[wordIdx];
        const uint32_t residenceWord = __atomic_load_n( &context.residenceBits.ptr[wordIdx], __ATOMIC_RELAXED );
        collectRequests(context, referenceWord & ~residenceWord, resourceBitOffset);
        if( context.lru.len > 0 )
        {
            collectEvictionCandidates(context, launchNum, lruThreshold, referenceWord, residenceWord, resourceBitOffset);
        }

        wordIdx += gridDim.x * blockDim.x;
    }
}

HIP_DEMAND_INLINE void updateProcessedResources( const ResourceTable& resourceTable, const DeviceSpan<uint32_t>& residenceBits, const DeviceSpan<DeviceTextureInfo*>& textureInfos, const DeviceSpan<uint32_t>& lru, const DeviceSpan<ProcessedResource>& processedResources )
{
    uint32_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    while( globalIdx < processedResources.len )
    {
        ProcessedResource resource = processedResources.ptr[globalIdx];
        if( resourceTable.textureTiles.contains( resource.resourceId ) )
        {
            if( lru.len > 0 )
            {
                const uint32_t pageId = resourceTable.textureTiles.getLocalId(resource.resourceId);
                atomicClearUint4(lru, pageId);
            }
        }
        else
        {
            const uint32_t textureId = resourceTable.textureInfos.getLocalId(resource.resourceId);
            textureInfos.ptr[textureId] = resource.textureInfo.ptr;
        }

        atomicSetBit( residenceBits, resource.resourceId, __ATOMIC_RELEASE );
        globalIdx += gridDim.x * blockDim.x;
    }
}

HIP_DEMAND_INLINE void updateEvictedPagesKernel( DeviceContext& context, uint32_t evictionCount )
{
    uint32_t globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    while( globalIdx < evictionCount )
    {
        const uint32_t pageId = context.evictionCandidates.ptr[globalIdx].pageId;
        const uint32_t resourceId = context.resourceTable.textureTiles.getResourceId(pageId);
        atomicUnsetBit(context.residenceBits, resourceId);
        globalIdx += gridDim.x * blockDim.x;
    }
}

}

extern "C" __global__ void collectRequestsKernel( hip_demand::vmm::DeviceContext context, uint32_t launchNum, uint32_t lruThreshold )
{
    hip_demand::vmm::collectRequestsAndEvictionCandidates(context, launchNum, lruThreshold);
}

extern "C" __global__ void updateProcessedResourcesKernel(
    hip_demand::vmm::ResourceTable resourceTable,
    hip_demand::DeviceSpan<uint32_t> residenceBits,
    hip_demand::DeviceSpan<hip_demand::vmm::DeviceTextureInfo*> textureInfos,
    hip_demand::DeviceSpan<uint32_t> lru,
    hip_demand::DeviceSpan<hip_demand::vmm::ProcessedResource> processedResources
)
{
    hip_demand::vmm::updateProcessedResources(resourceTable, residenceBits, textureInfos, lru, processedResources);
}

extern "C" __global__ void updateEvictedPagesKernel( hip_demand::vmm::DeviceContext context, uint32_t evictionCount )
{
    hip_demand::vmm::updateEvictedPagesKernel(context, evictionCount);
}
