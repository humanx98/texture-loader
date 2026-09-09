#include "InFlightContextPool.h"
#include "DemandTextureLoaderImpl.h"

#include <algorithm>
#include <cassert>

namespace hip_demand::vmm {

InFlightContext* InFlightContextPool::alloc( hipStream_t stream, const DemandTextureLoaderImpl& loader )
{
    std::lock_guard<std::mutex> lock( mutex_ );

    if( !free_.empty() )
    {
        InFlightContext* context = contexts_.at( free_.back() ).get();
        free_.pop_back();
        return context;
    }

    // allocate per device data
    // this data is used by all contexts and accessed by atomics on device
    if( contexts_.empty() )
    {
        residenceBits_ = allocArray<uint32_t>( loader.bits_.wordCount(), true, stream );
        textureInfos_  = allocArray<DeviceTextureInfo*>( loader.options_.maxTextures, true, stream );

        // eviction logic should be applied only to texture tiles
        // 4 bits per textute tile page, 8 values per uint32_t
        if( loader.options_.enableEviction )
            lru_ = allocArray<uint32_t>( ceilDiv( loader.resourceTable_.textureTiles.count, 8u ), true, stream );

        // sync beacause it's shared across streams
        HIP_CHECK( hipStreamSynchronize( stream ) );
    }

    contexts_.push_back( std::make_unique<InFlightContext>() );
    InFlightContext* context = contexts_.back().get();
    *context                 = {};

    context->deviceContext.resourceTable      = loader.resourceTable_;
    context->deviceContext.pageSize           = loader.pageSystem_.pageBytes();
    context->deviceContext.poolIndex          = contexts_.size() - 1;
    context->deviceContext.pageMemory         = loader.pageSystem_.virtualAddressSpace();
    context->deviceContext.residenceBits      = residenceBits_;
    context->deviceContext.textureInfos       = textureInfos_;
    context->deviceContext.referenceBits      = allocArray<uint32_t>( loader.bits_.wordCount(), true, stream );
    context->deviceContext.requestedResources = allocArray<uint32_t>( loader.options_.maxRequests, false, stream );
    context->deviceContext.counters =
        allocArray<uint32_t>( static_cast<size_t>( CounterIndex::NumCounters ), true, stream );
    if( loader.options_.enableEviction )
    {
        context->deviceContext.requestIfResident = true;
        context->deviceContext.evictionCandidates =
            allocArray<EvictionCandidate>( loader.options_.maxEvictedPages, false, stream );
        context->deviceContext.lru = lru_;
    }

    // set host transfer buffers
    context->requestedResources = hostAllocArray<uint32_t>( context->deviceContext.requestedResources.len );
    context->evictionCandidates.previous = hostAllocArray<EvictionCandidate>( context->deviceContext.evictionCandidates.len );
    context->evictionCandidates.current = hostAllocArray<EvictionCandidate>( context->deviceContext.evictionCandidates.len );
    context->counters = hostAllocArray<uint32_t>( context->deviceContext.counters.len, true );

    return context;
}

void InFlightContextPool::free( InFlightContext* context )
{
    assert( context );
    std::lock_guard<std::mutex> lock( mutex_ );

    assert( context->deviceContext.poolIndex < contexts_.size() );
    assert( std::find( free_.begin(), free_.end(), context->deviceContext.poolIndex ) == free_.end() );
    free_.push_back( context->deviceContext.poolIndex );
}

InFlightContext* InFlightContextPool::get( size_t poolIndex )
{
    std::lock_guard<std::mutex> lock( mutex_ );
    return contexts_.at( poolIndex ).get();
}

InFlightContextPool::~InFlightContextPool()
{
    freeArray( residenceBits_ );
    freeArray( lru_ );
    freeArray( textureInfos_ );

    for( auto& context : contexts_ )
    {
        freeArray( context->deviceContext.referenceBits );
        freeArray( context->deviceContext.requestedResources );
        freeArray( context->deviceContext.evictionCandidates );
        freeArray( context->deviceContext.counters );

        hostFreeArray( context->requestedResources );
        hostFreeArray( context->evictionCandidates.previous );
        hostFreeArray( context->evictionCandidates.current );
        hostFreeArray( context->counters );
    }
}

}  // namespace hip_demand::vmm
