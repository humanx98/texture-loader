#pragma once

#include <DemandLoading/DeviceContext.h>
#include <ImageSource/ImageSource.h>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <utility>

namespace hip_demand::vmm {

struct TextureDescriptor
{
    hipTextureAddressMode addressMode[2]   = { hipAddressModeWrap, hipAddressModeWrap };
    hipTextureFilterMode  filterMode       = hipFilterModeLinear;
    hipTextureFilterMode  mipmapFilterMode = hipFilterModeLinear;
    bool                  normalizedCoords = true;
};

class Ticket
{
  public:
    Ticket() {}
    int  numTasksTotal() const;
    int  numTasksRemaining() const;
    void wait( hipEvent_t* event = nullptr );

  private:
    std::shared_ptr<class TicketImpl> impl_;
    friend class TicketImpl;
    Ticket( std::shared_ptr<TicketImpl>&& impl )
        : impl_( std::move( impl ) )
    {
    }
};

struct Options
{
    uint32_t maxTextures      = 1024;
    uint32_t maxVirtualPages  = 32 * 1024;
    uint32_t maxPhysicalPages = 1024;  // 64KB * 1024
    uint32_t maxRequests      = 1024;
    uint32_t maxRequestQueue  = 1024;
    uint32_t maxThreads       = 0;
    bool     enableEviction   = true;
    //uint32_t minResidentFrames  = 3;
};

class DemandTexture
{
  public:
    virtual ~DemandTexture()       = default;
    virtual uint32_t getId() const = 0;
};

class DemandTextureLoader
{
  public:
    virtual ~DemandTextureLoader() = default;

    virtual const DemandTexture& createTexture( std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& textureDesc ) = 0;
    virtual void   launchPrepare( hipStream_t stream, DeviceContext& deviceContext )         = 0;
    virtual Ticket processRequests( hipStream_t stream, const DeviceContext& deviceContext ) = 0;
    virtual void   freeDeviceContext( DeviceContext& deviceContext )                         = 0;
};

std::unique_ptr<DemandTextureLoader> createDemandTextureLoader( const Options& options );

}  // namespace hip_demand::vmm
