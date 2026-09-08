#include "ProcessedBatchPool.h"

#include <algorithm>
#include <cassert>

namespace hip_demand::vmm {

ProcessedBatch::ProcessedBatch( size_t poolIndex, uint32_t maxCapacity )
    : maxCapacity_( maxCapacity )
    , poolIndex_( poolIndex )
{
}

ProcessedBatch::~ProcessedBatch()
{
    hostFreeArray( hostBuffer_ );
    freeArray( deviceBuffer_ );
}

void ProcessedBatch::append( ProcessedResource resource )
{
    std::lock_guard<std::mutex> lock( mutex_ );

    if( !hostBuffer_.ptr )
        hostBuffer_   = hostAllocArray<ProcessedResource>( maxCapacity_ );

    if( !deviceBuffer_.ptr )
        deviceBuffer_ = allocArray<ProcessedResource>( maxCapacity_ );

    assert( count_ < maxCapacity_ );
    hostBuffer_.ptr[count_++] = resource;
}

ProcessedBatchPool::ProcessedBatchPool( uint32_t maxProcessedResources )
    : maxProcessedResources_( maxProcessedResources )
{
}

ProcessedBatch* ProcessedBatchPool::alloc()
{
    std::lock_guard<std::mutex> lock( mutex_ );

    if( !free_.empty() )
    {
        ProcessedBatch* batch = batches_.at( free_.back() ).get();
        free_.pop_back();
        return batch;
    }

    batches_.push_back( std::make_unique<ProcessedBatch>( batches_.size(), maxProcessedResources_ ) );
    return batches_.back().get();
}

void ProcessedBatchPool::free( ProcessedBatch* batch )
{
    assert( batch );
    std::lock_guard<std::mutex> lock( mutex_ );

    assert( batch->poolIndex() < batches_.size() );
    assert( std::find( free_.begin(), free_.end(), batch->poolIndex() ) == free_.end() );
    free_.push_back( batch->poolIndex() );
}

}  // namespace hip_demand::vmm
