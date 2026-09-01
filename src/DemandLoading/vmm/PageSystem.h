#pragma once

#include "../Internal/Utils.h"
#include <DemandLoading/DeviceContext.h>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <mutex>

namespace hip_demand::vmm {

struct PageTable
{
    struct Range
    {
        uint32_t startPage         = 0;
        uint32_t pageCount         = 0;
        uint32_t nextAvailablePage = 0;

        Range() {}
        Range( uint32_t start, uint32_t count )
            : startPage( start )
            , pageCount( count )
            , nextAvailablePage( start )
        {
        }
    };

    Range textureInfos{};
    Range textureTiles{};
};

class PageSystem : NonCopyble
{
  public:
    explicit PageSystem( uint32_t maxVirtualPages, uint32_t maxPhysicalPages );
    ~PageSystem();
    void map( uint32_t pageId );
    void enqueueEvictedPages( const EvictionCandidate* evictedPages, uint32_t count );

    DeviceSpan<uint8_t> page( uint32_t pageId ) const
    {
        return DeviceSpan<uint8_t>( virtualAddressSpace_.ptr + pageId * pageBytes_, pageBytes_ );
    }
    DeviceSpan<uint8_t> virtualAddressSpace() const { return virtualAddressSpace_; }
    size_t              granularity() const { return granularity_; }
    size_t              pageBytes() const { return pageBytes_; }
    int                 device() const { return device_; }

  private:
    void processPendingEvictedPages();

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
    std::vector<uint32_t>     pendingEvictedPages_{};
    DeviceSpan<uint8_t>       virtualAddressSpace_{};
    mutable std::mutex        mutex_;
};

}  // namespace hip_demand::vmm