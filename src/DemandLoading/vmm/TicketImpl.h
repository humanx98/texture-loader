#pragma once

#include "../Internal/HipCheck.h"
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <cassert>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <utility>

namespace hip_demand::vmm {

class TicketImpl
{
  public:
    static Ticket create( hipStream_t stream ) { return Ticket( std::make_shared<TicketImpl>( stream ) ); }
    static std::shared_ptr<TicketImpl>& getImpl( Ticket& ticket ) { return ticket.impl_; }

    TicketImpl( hipStream_t stream )
        : m_stream( stream )
    {
    }

    void update( unsigned int numTasks )
    {
        {
            std::unique_lock<std::mutex> lock( m_mutex );
            m_numTasksTotal     = numTasks;
            m_numTasksRemaining = numTasks;
        }
        if( numTasks == 0 )
            m_isDone.notify_all();
    }

    hipStream_t getStream() const { return m_stream; }
    int numTasksTotal() const
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_numTasksTotal;
    }

    int         numTasksRemaining() const
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        return m_numTasksRemaining;
    }

    void wait( hipEvent_t* event = nullptr )
    {
        std::unique_lock<std::mutex> lock( m_mutex );
        m_isDone.wait( lock, [this] { return m_numTasksRemaining == 0; } );

        const std::exception_ptr exception = m_exception;
        lock.unlock();

        if( exception )
            std::rethrow_exception( exception );

        if( event )
        {
            HIP_CHECK( hipEventRecord( *event, m_stream ) );
        }
    }

    void notify( std::exception_ptr exception = {} )
    {
        std::unique_lock<std::mutex> lock( m_mutex );

        assert( m_numTasksRemaining > 0 );
        if( exception && !m_exception )
            m_exception = std::move( exception );
        --m_numTasksRemaining;

        if( m_numTasksRemaining == 0 )
            m_isDone.notify_all();
    }

  private:
    const hipStream_t       m_stream{};
    int                     m_numTasksTotal{ -1 };
    int                     m_numTasksRemaining{ -1 };
    mutable std::mutex      m_mutex;
    std::condition_variable m_isDone;
    std::exception_ptr      m_exception;
};

}  // namespace hip_demand::vmm
