#pragma once

#include "RequestProcessor.h"
#include "Callbacks.h"
#include "DemandTextureLoaderImpl.h"

namespace hip_demand::vmm {

RequestProcessor::RequestProcessor( DemandTextureLoaderImpl* loader, uint32_t maxThreads, uint32_t maxQueueSize )
    : queue_( maxQueueSize )
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

size_t RequestProcessor::submit( const HostSpan<uint32_t>& resourceIds, ProcessedBatch* batch, Ticket ticket )
{
    std::lock_guard<std::mutex> lock( mutex_ );
    if( stopped_ )
    {
        TicketImpl::getImpl( ticket )->initialize( 0 );
        return 0;
    }

    return queue_.push( resourceIds, batch, ticket );
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

            std::shared_ptr<TicketImpl>& ticket = TicketImpl::getImpl( request.ticket );

            hipStream_t stream = ticket->getStream();
            loader_->processRequest( stream, request.batch, request.resourceId );
            if( ticket->finishTaskAndClaimFinalization() )
            {
                loader_->publishProcessedBatch( stream, request.batch );
                ticket->publishCompletion();
            }
        }
    }
    catch( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

}  // namespace hip_demand::vmm
