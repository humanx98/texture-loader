#pragma once

#include "../Internal/Utils.h"
#include <DemandLoading/DeviceContext.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace hip_demand::vmm {

class ProcessedBatch : NonCopyble
{
  public:
    explicit ProcessedBatch( size_t poolIndex, uint32_t maxCapacity );
    ~ProcessedBatch();

    void append( ProcessedResource resource );

    void clear() { count_ = 0; }
    size_t poolIndex() const { return poolIndex_; }
    uint32_t count() const { return count_; }
    HostSpan<ProcessedResource> hostBuffer() const { return hostBuffer_; }
    DeviceSpan<ProcessedResource> deviceBuffer() const { return deviceBuffer_; }

  private:
    mutable std::mutex            mutex_;
    HostSpan<ProcessedResource>   hostBuffer_{};
    DeviceSpan<ProcessedResource> deviceBuffer_{};
    uint32_t                      maxCapacity_ = 0;
    uint32_t                      count_       = 0;
    size_t                        poolIndex_   = 0;
};

class ProcessedBatchPool : NonCopyble
{
  public:
    explicit ProcessedBatchPool( uint32_t maxProcessedResources );

    ProcessedBatch* alloc();
    void            free( ProcessedBatch* batch );

  private:
    mutable std::mutex                            mutex_;
    uint32_t                                      maxProcessedResources_;
    std::vector<std::unique_ptr<ProcessedBatch>> batches_{};
    std::vector<size_t>                           free_{};
};

}  // namespace hip_demand::vmm
