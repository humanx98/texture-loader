#pragma once

#include "RequestProcessor.h"
#include "DemandTextureLoaderImpl.h"

namespace hip_demand::vmm {

RequestProcessor::RequestProcessor( DemandTextureLoaderImpl* loader, uint32_t resourceCount, uint32_t maxThreads, uint32_t maxQueueSize )
    : queue_( maxQueueSize )
    , residenceBits_( resourceCount )
    , loadingBits_( resourceCount )
    , loader_( loader )
{
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
        throw;
    }
}

RequestProcessor::~RequestProcessor()
{
    stop();
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

void RequestProcessor::uploadResidenceBits( DeviceSpan<uint32_t>& destination, hipStream_t stream )
{
    std::unique_lock<std::mutex> lock( mutex_ );
    memcpyHtoD( destination, residenceBits_.words(), stream );
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

                if( !stopped_ && !residenceBits_.test( resourceId ) )
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
                    residenceBits_.set( resourceId, true );
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
