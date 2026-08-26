#pragma once

#include "RequestProcessor.h"
#include "DemandTextureLoaderImpl.h"

namespace hip_demand::vmm {

RequestProcessor::RequestProcessor( DemandTextureLoaderImpl* loader, HipEventPool& eventPool, uint32_t resourceCount, uint32_t maxThreads, uint32_t maxQueueSize )
    : queue_( maxQueueSize )
    , residentBits_( resourceCount )
    , loadingBits_( resourceCount )
    , eventPool_( eventPool )
    , loader_( loader )
{
    residentBitsUploadDone_ = eventPool_.acquire();

    if( maxThreads == 0 )
        maxThreads = std::thread::hardware_concurrency();

    const uint32_t threadCount = std::clamp( maxThreads, 1u, 16u );
    try
    {
        workers_.reserve( threadCount );
        for( uint32_t i = 0; i < threadCount; ++i )
            workers_.emplace_back( [this] { workerLoop(); } );
    }
    catch( ... )
    {
        queue_.close();
        for( std::thread& worker : workers_ )
            worker.join();
        if( residentBitsUploadDone_ )
            eventPool_.release( residentBitsUploadDone_ );
        residentBitsUploadDone_ = nullptr;
        throw;
    }
}

RequestProcessor::~RequestProcessor()
{
    stop();

    if( residentBitsUploadDone_ )
        eventPool_.release( residentBitsUploadDone_ );
}

void RequestProcessor::stop()
{
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        stopped_ = true;
    }
    queue_.close();

    for( std::thread& worker : workers_ )
    {
        if( worker.joinable() )
            worker.join();
    }

    std::unique_lock<std::mutex> lock( mutex_ );
    waitForResidentBitsUpload();
}

void RequestProcessor::submit( const uint32_t* resourceIds, uint32_t count, Ticket ticket )
{
    std::lock_guard<std::mutex> lock( mutex_ );
    if( stopped_ )
    {
        TicketImpl::getImpl( ticket )->initialize( 0 );
        return;
    }

    queue_.push( resourceIds, count, ticket );
}

void RequestProcessor::uploadResidentBits( DeviceSpan<uint32_t>& destination, hipStream_t stream )
{
    std::unique_lock<std::mutex> lock( mutex_ );
    if( !residentBitsDirty_ )
        return;

    waitForResidentBitsUpload();
    memcpyHtoD( destination, residentBits_.words(), stream );
    HIP_CHECK( hipEventRecord( residentBitsUploadDone_, stream ) );
    residentBitsUploadInFlight_ = true;
    residentBitsDirty_          = false;
}

void RequestProcessor::waitForResidentBitsUpload()
{
    if( !residentBitsUploadInFlight_ )
        return;

    HIP_WARN( hipEventSynchronize( residentBitsUploadDone_ ) );
    residentBitsUploadInFlight_ = false;
}

void RequestProcessor::workerLoop()
{
    try
    {
        HIP_CHECK( hipSetDevice( loader_->device() ) );
        while( true )
        {
            ResourceRequest request;
            if( !queue_.pop( request ) )
                break;

            const uint32_t               resourceId = request.resourceId;
            std::shared_ptr<TicketImpl>& ticket     = TicketImpl::getImpl( request.ticket );
            bool                         shouldLoad = false;

            {
                std::unique_lock<std::mutex> lock( mutex_ );

                loading_.wait( lock, [this, resourceId] { return !loadingBits_.test( resourceId ); } );

                if( !stopped_ && !residentBits_.test( resourceId ) )
                {
                    loadingBits_.set( resourceId, true );
                    shouldLoad = true;
                }
            }

            if( shouldLoad )
            {
                bool success = false;
                try
                {
                    loader_->processRequest( ticket->getStream(), resourceId );
                    success = true;
                }
                catch( const std::exception& e )
                {
                    std::cerr << "Error: " << e.what() << std::endl;
                }

                std::unique_lock<std::mutex> lock( mutex_ );
                if( success )
                {
                    waitForResidentBitsUpload();
                    residentBits_.set( resourceId, true );
                    residentBitsDirty_ = true;
                }
                loadingBits_.set( resourceId, false );
                loading_.notify_all();
            }

            ticket->notify();
        }
    }
    catch( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

}  // namespace hip_demand::vmm
