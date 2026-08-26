#include <DemandLoading/DeviceContext.h>
#include <cstdint>
#include <hip/hip_runtime.h>

using hip_demand::vmm::CounterIndex;
using hip_demand::vmm::DeviceContext;

__device__ __forceinline__ uint32_t calcListStartIndex( const uint32_t laneId, uint32_t numSetBits, uint32_t* counter )
{
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

extern "C" __global__ void collectRequests( DeviceContext context )
{
    const uint32_t startIndex = 0;
    const uint32_t endIndex = context.requestedBits.len;

    uint32_t wordIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t laneId = __lane_id();

    while( wordIdx < endIndex )
    {
        const uint32_t referenceWord = context.requestedBits.ptr[wordIdx];
        const uint32_t residenceWord = context.residentBits.ptr[wordIdx];
        uint32_t requestBits = referenceWord & ~residenceWord;
        const uint32_t numRequestBits = __popc( requestBits );

        const uint32_t resourceBitOffset = wordIdx * 32;
        uint32_t requestIndex = calcListStartIndex( laneId, numRequestBits, &context.counters.ptr[static_cast<uint32_t>( CounterIndex::RequestedResources )] );
        while( requestBits != 0 && ( requestIndex < context.requestedResources.len ) )
        {
            const uint32_t bitIndex = __ffs( requestBits ) - 1;
            requestBits ^= ( 1U << bitIndex );
            context.requestedResources.ptr[requestIndex++] = resourceBitOffset + bitIndex;
        }

        wordIdx += gridDim.x * blockDim.x;
    }
}
