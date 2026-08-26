#pragma once

#include "../Internal/HipCheck.h"
#include "../Internal/Utils.h"
#include "Allocator.h"
#include "HipEventPool.h"
#include "PageSystem.h"
#include "RequestProcessor.h"
#include <DemandLoading/VmmDemandTextureLoader.h>
#include <cassert>
#include <deque>
#include <map>
#include <mutex>
#include <utility>

namespace hip_demand::vmm {

class HostAllocator : NonCopyble
{
  public:
    HostAllocator( size_t maxPending, HipEventPool& eventPool )
        : maxPending_( maxPending )
        , eventPool_( eventPool )
    {
    }

    ~HostAllocator()
    {
        freePending( true );
        for( auto& kv : available_ )
        {
            for( auto& allocations : kv.second )
                hostFreeArray( allocations );
        }
    }

    template <typename T>
    HostSpan<T> allocArray( size_t len )
    {
        HostSpan<uint8_t> buffer = allocBuffer( len * sizeof( T ) );
        return HostSpan<T>( reinterpret_cast<T*>( buffer.ptr ), len );
    }

    template <typename T>
    void freeArray( HostSpan<T>& span, hipStream_t stream )
    {
        freeBuffer( HostSpan<uint8_t>( reinterpret_cast<uint8_t*>( span.ptr ), span.sizeInBytes() ), stream );
        span = {};
    }

    template <typename T>
    class Scoped
    {
      public:
        explicit Scoped( HostAllocator* allocator, HostSpan<T> memory, hipStream_t stream )
            : allocator_( allocator )
            , memory_( memory )
            , stream_( stream )
        {
        }

        ~Scoped() { reset(); }

        Scoped( const Scoped& )            = delete;
        Scoped& operator=( const Scoped& ) = delete;

        Scoped( Scoped&& other ) noexcept
            : allocator_( std::exchange( other.allocator_, nullptr ) )
            , memory_( std::exchange( other.memory_, nullptr ) )
            , stream_( std::exchange( other.stream_, nullptr ) )
        {
        }

        Scoped& operator=( Scoped&& other ) noexcept
        {
            if( this != &other )
            {
                reset();

                allocator_ = std::exchange( other.allocator_, nullptr );
                memory_    = std::exchange( other.memory_, nullptr );
                stream_    = std::exchange( other.stream_, nullptr );
            }

            return *this;
        }

        void reset()
        {
            if( allocator_ && memory_.ptr )
                allocator_->freeArray( memory_, stream_ );

            allocator_ = nullptr;
            memory_    = {};
            stream_    = nullptr;
        }

        HostSpan<T> span() const noexcept { return memory_; }
        T*          operator->() noexcept { return memory_.ptr; }
        const T*    operator->() const noexcept { return memory_.ptr; }

        T& operator*() noexcept
        {
            assert( memory_.ptr );
            return *memory_.ptr;
        }

        const T& operator*() const noexcept
        {
            assert( memory_.ptr );
            return *memory_.ptr;
        }

      private:
        HostAllocator* allocator_{};
        HostSpan<T>    memory_{};
        hipStream_t    stream_{};
    };

    template <typename T>
    Scoped<T> allocScopedArray( size_t len, hipStream_t stream )
    {
        return Scoped( this, allocArray<T>( len ), stream );
    }


  private:
    HostSpan<uint8_t> allocBuffer( size_t size )
    {
        assert( size > 0 );

        {
            std::lock_guard<std::mutex> lock( mutex_ );

            freePending( false );

            auto& allocations = available_[size];
            if( !allocations.empty() )
            {
                HostSpan<uint8_t> result = allocations.back();
                allocations.pop_back();
                return result;
            }
        }

        return hostAllocArray<uint8_t>( size );
    }

    void freeBuffer( HostSpan<uint8_t> allocation, hipStream_t stream )
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        assert( allocation.ptr );
        assert( stream );

        Pending async = { allocation, eventPool_.acquire() };
        HIP_CHECK( hipEventRecord( async.event, stream ) );
        pendingQueue_.push_back( async );
    }

    void freePending( bool waitOnEvents )
    {
        while( !pendingQueue_.empty() )
        {
            const Pending& pendingRelease   = pendingQueue_.front();
            hipError_t     eventQueryStatus = hipEventQuery( pendingRelease.event );
            if( eventQueryStatus == hipErrorNotReady )
            {
                if( waitOnEvents || pendingQueue_.size() > maxPending_ )
                {
                    HIP_WARN( hipEventSynchronize( pendingRelease.event ) );
                }
                else
                {
                    break;
                }
            }
            else
            {
                HIP_WARN( eventQueryStatus );
            }

            eventPool_.release( pendingRelease.event );
            available_.at( pendingRelease.memory.len ).push_back( pendingRelease.memory );
            pendingQueue_.pop_front();
        }
    }

    struct Pending
    {
        HostSpan<uint8_t> memory{};
        hipEvent_t        event{};
    };

    mutable std::mutex  mutex_;
    size_t              maxPending_ = 0;
    std::deque<Pending> pendingQueue_{};
    HipEventPool&       eventPool_;

    std::map<size_t, std::vector<HostSpan<uint8_t>>> available_{};
};

struct Resource
{
    enum class Type
    {
        TextureId,
        Tile,
        MipTail
    };

    struct Tile
    {
        uint32_t textureId = 0;
        uint32_t mipLevel  = 0;
        uint32_t tileX     = 0;
        uint32_t tileY     = 0;
        uint32_t pageId    = 0;
    };

    struct MipTail
    {
        uint32_t textureId = 0;
        uint32_t pageId    = 0;
    };

    Type type;
    union
    {
        uint32_t textureId;
        Tile     tile;
        MipTail  mipTail;
    };

    explicit Resource( uint32_t id )
        : type( Type::TextureId )
        , textureId( id )
    {
    }

    explicit Resource( Tile value )
        : type( Type::Tile )
        , tile( value )
    {
    }

    explicit Resource( MipTail value )
        : type( Type::MipTail )
        , mipTail( value )
    {
    }
};

class DemandTextureImpl : public DemandTexture, NonCopyble
{
  public:
    DemandTextureImpl( uint32_t textureId, std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& textureDescriptor )
        : id( textureId )
        , image( imageSource )
        , descriptor( textureDescriptor )
    {
    }
    uint32_t getId() const override { return id; }

    uint32_t                     id                  = INVALID_TEXTURE;
    uint32_t                     loadedTextureInfoId = INVALID_TEXTURE;
    std::shared_ptr<ImageSource> image{};
    TextureDescriptor            descriptor;
};

class DemandTextureLoaderImpl : public DemandTextureLoader, NonCopyble
{
  public:
    explicit DemandTextureLoaderImpl( const Options& options );
    ~DemandTextureLoaderImpl() override;

    const DemandTexture& createTexture( std::shared_ptr<ImageSource> imageSource, const TextureDescriptor& textureDesc ) override;
    void   launchPrepare( hipStream_t stream, DeviceContext& deviceContext ) override;
    Ticket processRequests( hipStream_t stream, const DeviceContext& deviceContext ) override;
    void   processRequestsCallback( DeviceContext& deviceContext, Ticket ticket );
    void   processRequest( hipStream_t stream, uint32_t resourceId );
    void   freeDeviceContext( DeviceContext& deviceContext ) override { freeDeviceContext( deviceContext, true ); }
    int    device() const { return pageSystem_.device(); }

  private:
    void     initDeviceContext( DeviceContext& deviceContext, hipStream_t stream );
    void     freeDeviceContext( DeviceContext& deviceContext, bool needLock );
    Resource decode( uint32_t resourceId );
    void     processTextureInfo( hipStream_t stream, uint32_t textureId );
    void     processTile( hipStream_t stream, const Resource::Tile& tile );
    void     processMipTail( hipStream_t stream, const Resource::MipTail& mipTail );

    struct InFlight
    {
        DeviceContext      deviceContext{};
        HostSpan<uint32_t> requestedResources{};
        HostSpan<uint32_t> counters{};
    };

    mutable std::mutex mutex_;
    Options            options_{};
    HipEventPool       eventPool_;
    PageSystem         pageSystem_;

    DeviceSpan<uint32_t>           residentBits_{};
    DeviceSpan<DeviceTextureInfo*> textureInfos_{};

    std::vector<InFlight>                           inFlight_{};
    std::vector<size_t>                             freeDeviceContextList_{};
    std::vector<std::unique_ptr<DemandTextureImpl>> textures_{};

    // metadata
    mutable std::mutex           metadataMutex_;
    Allocator<DeviceTextureInfo> textureInfoAllocator_;
    PageTable                    pageTable_{};
    uint32_t                     maxResources_ = 0;
    // note that this list should be accessed by texture.loadedTextureInfoId
    // and it's ordered by startPage in order to use std::upper_bound
    std::vector<DeviceTextureInfo> loadedTextureInfos_{};

    RequestProcessor requestProcessor_;
    HostAllocator    hostAllocator_;
    struct {
        hipModule_t module;
        hipFunction_t collectRequests;
    } kernels_{};
};

}  // namespace hip_demand::vmm
