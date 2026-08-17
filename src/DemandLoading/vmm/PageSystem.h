#pragma once

#include "../Internal/Utils.h"
#include <DemandLoading/DeviceContext.h>
#include <cstdint>
#include <hip/hip_runtime.h>

namespace hip_demand::vmm {

using internal::NonCopyble;

class PageSystem : NonCopyble
{
  public:
    explicit PageSystem( uint32_t maxVirtualPages, uint32_t maxPhysicalPages );
    ~PageSystem();
    void                map( uint32_t pageId );
    void                unmap( uint32_t pageId );
    bool                mapped( uint32_t pageId ) const { return virtualIdToPhysicalId_.at( pageId ) != INVALID_PAGE; }
    DeviceSpan<uint8_t> page( uint32_t pageId ) const
    {
        return DeviceSpan<uint8_t>( virtualAddressSpace_.ptr + pageId * pageBytes_, pageBytes_ );
    }
    DeviceSpan<uint8_t> virtualAddressSpace() const { return virtualAddressSpace_; }
    size_t              granularity() const { return granularity_; }
    size_t              pageBytes() const { return pageBytes_; }

  private:
    struct PhysicalPage
    {
        hipMemGenericAllocationHandle_t handle        = nullptr;
        uint32_t                        virtualPageId = INVALID_PAGE;
    };

    uint32_t                  maxVirtualPages_  = 0;
    uint32_t                  maxPhysicalPages_ = 0;
    int                       device_           = 0;
    size_t                    granularity_      = 0;
    size_t                    pageBytes_        = 0;
    hipMemAllocationProp      allocationProp_{};
    std::vector<uint32_t>     virtualIdToPhysicalId_{};
    std::vector<PhysicalPage> physicalPages_{};
    std::vector<uint32_t>     freePhysicalPages_{};
    DeviceSpan<uint8_t>       virtualAddressSpace_{};
};

}  // namespace hip_demand::vmm