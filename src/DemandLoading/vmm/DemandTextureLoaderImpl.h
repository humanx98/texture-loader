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
#include <mutex>
#include <utility>

namespace hip_demand::vmm {

class PinnedBufferAllocator : NonCopyble
{
  public:
    PinnedBufferAllocator( size_t buffSize, HipEventPool& eventPool )
        : buffSize_( buffSize )
        , eventPool_( eventPool )
    {
    }

    ~PinnedBufferAllocator()
    {
        freePending( true );
        for( void* a : available_ )
            HIP_WARN( hipHostFree( a ) );
    }

    uint8_t* alloc()
    {
        std::lock_guard<std::mutex> lock( mutex_ );

        uint8_t* allocation = nullptr;
        if( available_.empty() )
        {
            HIP_CHECK( hipHostMalloc( &allocation, buffSize_ ) );
        }
        else
        {
            allocation = available_.back();
            available_.pop_back();
        }

        return allocation;
    }

    void free( uint8_t* allocation, hipStream_t stream )
    {
        std::lock_guard<std::mutex> lock( mutex_ );
        assert( allocation );
        assert( stream );
        assert( std::find( available_.begin(), available_.end(), allocation ) == available_.end() );
        assert( std::none_of( pendingQueue_.begin(), pendingQueue_.end(),
                              [allocation]( const Pending& item ) { return item.memory == allocation; } ) );

        freePending( false );

        Pending async = { allocation, eventPool_.acquire() };
        HIP_CHECK( hipEventRecord( async.event, stream ) );
        pendingQueue_.push_back( async );
    }

    size_t bufferSize() { return buffSize_; }

    class Scoped
    {
      public:
        explicit Scoped( PinnedBufferAllocator* allocator, uint8_t* memory, hipStream_t stream )
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
            if( memory_ )
                allocator_->free( memory_, stream_ );

            allocator_ = nullptr;
            memory_    = nullptr;
            stream_    = nullptr;
        }

        uint8_t*          get() const noexcept { return memory_; }
        HostSpan<uint8_t> span() const noexcept { return HostSpan<uint8_t>( memory_, allocator_->bufferSize() ); }

      private:
        PinnedBufferAllocator* allocator_ = nullptr;
        uint8_t*               memory_    = nullptr;
        hipStream_t            stream_    = nullptr;
    };

    Scoped allocScoped( hipStream_t stream ) { return Scoped( this, alloc(), stream ); }


  private:
    void freePending( bool waitOnEvents )
    {
        while( !pendingQueue_.empty() )
        {
            const Pending& pendingRelease   = pendingQueue_.front();
            hipError_t     eventQueryStatus = hipEventQuery( pendingRelease.event );
            if( eventQueryStatus == hipErrorNotReady )
            {
                if( waitOnEvents )
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
            available_.push_back( pendingRelease.memory );
            pendingQueue_.pop_front();
        }
    }

    struct Pending
    {
        uint8_t*   memory = nullptr;
        hipEvent_t event  = nullptr;
    };

    mutable std::mutex    mutex_;
    size_t                buffSize_;
    std::vector<uint8_t*> available_;
    std::deque<Pending>   pendingQueue_;
    HipEventPool&         eventPool_;
};

template <typename T>
class PinnedAllocator : NonCopyble
{
  public:
    class Scoped
    {
      public:
        explicit Scoped( PinnedBufferAllocator::Scoped allocation )
            : allocation_( std::move( allocation ) )
        {
        }

        Scoped( Scoped&& ) noexcept            = default;
        Scoped& operator=( Scoped&& ) noexcept = default;
        Scoped( const Scoped& )                = delete;
        Scoped& operator=( const Scoped& )     = delete;

        void reset() { allocation_.reset(); }

        T*          get() noexcept { return reinterpret_cast<T*>( allocation_.get() ); }
        const T*    get() const noexcept { return reinterpret_cast<const T*>( allocation_.get() ); }
        T*          operator->() noexcept { return get(); }
        const T*    operator->() const noexcept { return get(); }
        HostSpan<T> span() noexcept { return HostSpan<T>( get(), 1 ); }

        T& operator*() noexcept
        {
            assert( get() );
            return *get();
        }

        const T& operator*() const noexcept
        {
            assert( get() );
            return *get();
        }

      private:
        PinnedBufferAllocator::Scoped allocation_;
    };

    PinnedAllocator( HipEventPool& eventPool )
        : bufferAllocator_( sizeof( T ), eventPool )
    {
    }

    T* alloc()
    {
        T* allocation = reinterpret_cast<T*>( bufferAllocator_.alloc() );
        *allocation   = {};
        return allocation;
    }

    void free( T* allocation, hipStream_t stream )
    {
        bufferAllocator_.free( reinterpret_cast<uint8_t*>( allocation ), stream );
    }

    Scoped allocScoped( hipStream_t stream )
    {
        PinnedBufferAllocator::Scoped allocation  = bufferAllocator_.allocScoped( stream );
        *reinterpret_cast<T*>( allocation.get() ) = {};
        return Scoped( std::move( allocation ) );
    }

  private:
    PinnedBufferAllocator bufferAllocator_;
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


    PinnedBufferAllocator pageBufferPinnedAllocator_;

    // metadata
    mutable std::mutex                  metadataMutex_;
    Allocator<DeviceTextureInfo>        textureInfoAllocator_;
    PinnedAllocator<DeviceTextureInfo>  textureInfoPinnedAllocator_;
    PinnedAllocator<DeviceTextureInfo*> ptrTextureInfoPinnedAllocator_;
    PageTable                           pageTable_{};
    // note that this list should be accessed by texture.loadedTextureInfoId
    // and it's ordered by startPage in order to use std::upper_bound
    std::vector<DeviceTextureInfo> loadedTextureInfos_{};

    RequestProcessor requestProcessor_;
};

}  // namespace hip_demand::vmm
