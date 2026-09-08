#pragma once

#include "DemandTextureLoaderImpl.h"
#include <DemandLoading/DeviceContext.h>

namespace hip_demand::vmm {

class HipCallback
{
  public:
    virtual ~HipCallback() = default;

  protected:
    virtual void callback() = 0;

    static void enqueueCallback( hipStream_t stream, HipCallback* callback )
    {
        HIP_CHECK( hipLaunchHostFunc( stream, &staticCallback, callback ) );
    }

  private:
    static void staticCallback( void* arg )
    {
        std::unique_ptr<HipCallback> callback( static_cast<HipCallback*>( arg ) );
        callback->callback();
    }
};

class ProcessRequestCallback : public HipCallback
{
  public:
    explicit ProcessRequestCallback( DemandTextureLoaderImpl& loader, Ticket ticket, size_t inFlightIndex )
        : loader_( loader )
        , ticket_( std::move( ticket ) )
        , inFlightIndex_( inFlightIndex )
    {
    }

    static void enqueue( hipStream_t stream, DemandTextureLoaderImpl& loader, Ticket ticket, size_t inFlightIndex )
    {
        enqueueCallback( stream, new ProcessRequestCallback( loader, std::move( ticket ), inFlightIndex ) );
    }

  protected:
    void callback() override { loader_.processRequestsCallback( std::move( ticket_ ), inFlightIndex_ ); }

  private:
    DemandTextureLoaderImpl& loader_;
    Ticket                   ticket_;
    size_t                   inFlightIndex_;
};

class UpdateProcessedResourceCallback : public HipCallback
{
  public:
    explicit UpdateProcessedResourceCallback( DemandTextureLoaderImpl& loader, ProcessedBatch* batch )
        : loader_( loader )
        , batch_( batch )
    {
    }

    static void enqueue( hipStream_t stream, DemandTextureLoaderImpl& loader, ProcessedBatch* batch )
    {
        enqueueCallback( stream, new UpdateProcessedResourceCallback( loader, batch ) );
    }

  protected:
    void callback() override { loader_.recycleProcessedBatch( batch_ ); }

  private:
    DemandTextureLoaderImpl& loader_;
    ProcessedBatch*          batch_;
};

}  // namespace hip_demand::vmm
