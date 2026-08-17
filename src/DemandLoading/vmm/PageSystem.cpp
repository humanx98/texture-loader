#include "PageSystem.h"
#include "../Internal/HipCheck.h"

namespace hip_demand::vmm {

PageSystem::PageSystem( uint32_t maxVirtualPages, uint32_t maxPhysicalPages )
    : maxVirtualPages_( maxVirtualPages )
    , maxPhysicalPages_( maxPhysicalPages )
{
    HIP_CHECK( hipGetDevice( &device_ ) );
    int vmmSupported = 0;
    HIP_CHECK( hipDeviceGetAttribute( &vmmSupported, hipDeviceAttributeVirtualMemoryManagementSupported, device_ ) );
    if( vmmSupported == 0 )
        throw std::runtime_error( "The active HIP device does not support virtual memory management" );

    allocationProp_.type          = hipMemAllocationTypePinned;
    allocationProp_.location.type = hipMemLocationTypeDevice;
    allocationProp_.location.id   = device_;
    HIP_CHECK( hipMemGetAllocationGranularity( &granularity_, &allocationProp_, hipMemAllocationGranularityMinimum ) );
    if( granularity_ == 0 )
        throw std::runtime_error( "hipMemAllocationGranularityMinimum is zero" );

    pageBytes_               = granularity_;
    virtualAddressSpace_.len = maxVirtualPages_ * pageBytes_;
    HIP_CHECK( hipMemAddressReserve( reinterpret_cast<void**>( &virtualAddressSpace_.ptr ), virtualAddressSpace_.len,
                                     granularity_, nullptr, 0 ) );

    virtualIdToPhysicalId_.assign( maxVirtualPages_, INVALID_PAGE );
}

PageSystem::~PageSystem()
{
    for( PhysicalPage& physicalPage : physicalPages_ )
    {
        if( physicalPage.virtualPageId != INVALID_PAGE )
        {
            DeviceSpan<uint8_t> virtualPage = page( physicalPage.virtualPageId );
            HIP_WARN( hipMemUnmap( virtualPage.ptr, virtualPage.len ) );
        }

        HIP_WARN( hipMemRelease( physicalPage.handle ) );
    }

    if( virtualAddressSpace_.ptr )
        HIP_WARN( hipMemAddressFree( virtualAddressSpace_.ptr, virtualAddressSpace_.len ) );
}

void PageSystem::map( uint32_t pageId )
{
    if( mapped( pageId ) )
        return;

    uint32_t physicalPageId = INVALID_PAGE;
    if( freePhysicalPages_.empty() )
    {
        // TODO_BS: handle max physical pages
        if( physicalPages_.size() >= maxPhysicalPages_ )
            throw std::runtime_error( "Maximum demand physical page count exceeded" );

        physicalPageId                         = static_cast<uint32_t>( physicalPages_.size() );
        hipMemGenericAllocationHandle_t handle = nullptr;
        HIP_CHECK( hipMemCreate( &handle, pageBytes_, &allocationProp_, 0 ) );
        physicalPages_.push_back( PhysicalPage{ handle, INVALID_PAGE } );
    }
    else
    {
        physicalPageId = freePhysicalPages_.back();
        freePhysicalPages_.pop_back();
    }

    PhysicalPage&    physicalPage = physicalPages_.at( physicalPageId );
    DeviceSpan<uint8_t> virtualPage  = page( pageId );
    HIP_CHECK( hipMemMap( virtualPage.ptr, virtualPage.len, 0, physicalPage.handle, 0 ) );

    hipMemAccessDesc access{};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id   = device_;
    access.flags         = hipMemAccessFlagsProtReadWrite;
    HIP_CHECK( hipMemSetAccess( virtualPage.ptr, virtualPage.len, &access, 1 ) );

    virtualIdToPhysicalId_.at( pageId ) = physicalPageId;
    physicalPage.virtualPageId          = pageId;
}

void PageSystem::unmap( uint32_t pageId )
{
    if( !mapped( pageId ) )
        return;

    const uint32_t            physicalPageId = virtualIdToPhysicalId_.at( pageId );
    const DeviceSpan<uint8_t> virtualPage    = page( pageId );
    HIP_CHECK( hipMemUnmap( virtualPage.ptr, virtualPage.len ) );
    virtualIdToPhysicalId_.at( pageId )               = INVALID_PAGE;
    physicalPages_.at( physicalPageId ).virtualPageId = INVALID_PAGE;
    freePhysicalPages_.push_back( physicalPageId );
}

}