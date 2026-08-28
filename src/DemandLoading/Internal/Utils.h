// SPDX-License-Identifier: MIT
// Internal utility functions

#pragma once

#include "HipCheck.h"
#include <DemandLoading/DeviceContext.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace hip_demand {

class NonCopyble
{
    NonCopyble( const NonCopyble& )            = delete;
    NonCopyble& operator=( const NonCopyble& ) = delete;

  protected:
    NonCopyble() = default;
};

HIP_DEMAND_INLINE uint32_t ceilDiv( uint32_t value, uint32_t divisor )
{
    assert( divisor != 0 );
    return value / divisor + static_cast<uint32_t>( value % divisor != 0 );
}

HIP_DEMAND_INLINE size_t ceilDiv( size_t value, size_t divisor )
{
    assert( divisor != 0 );
    return value / divisor + static_cast<size_t>( value % divisor != 0 );
}

/// Calculate total memory needed for mipmaps
inline size_t calculateMipmapMemory( int width, int height, int bytesPerPixel )
{
    size_t total = 0;
    while( width > 0 && height > 0 )
    {
        total += width * height * bytesPerPixel;
        width /= 2;
        height /= 2;
    }
    return total;
}

/// Calculate number of mip levels
inline uint32_t calculateMipLevels( uint32_t width, uint32_t height )
{
    uint32_t levels = 1;
    while( width > 1 || height > 1 )
    {
        width  = std::max( 1u, width / 2 );
        height = std::max( 1u, height / 2 );
        levels++;
    }
    return levels;
}

template <typename T>
inline size_t sizeInBytes( const std::vector<T>& dst )
{
    return dst.size() * sizeof( T );
}

template <typename T, size_t N>
inline size_t sizeInBytes( const std::array<T, N>& dst )
{
    return dst.size() * sizeof( T );
}

inline size_t highestPowerOfTwoAtMost( size_t value )
{
    size_t result = 1;
    while( result <= value / 2 )
        result *= 2;
    return result;
}

inline bool safeAdd( uint32_t a, uint32_t b, uint32_t& result )
{
    if( a > std::numeric_limits<uint32_t>::max() - b )
        return false;

    result = a + b;
    return true;
}

inline uint2 tileShapeForGranularity( size_t granularity, uint32_t bytesPerTexel )
{
    assert( bytesPerTexel );
    assert( granularity >= bytesPerTexel );

    const size_t availableTexels = granularity / bytesPerTexel;

    size_t width = 1;
    while( width <= availableTexels / width / 4 )
        width *= 2;

    size_t height = highestPowerOfTwoAtMost( availableTexels / width );
    if( height > width )
        std::swap( width, height );

    return { static_cast<uint32_t>( width ), static_cast<uint32_t>( height ) };
}

inline uint32_t mipDimension( uint32_t baseDimension, uint32_t mipLevel )
{
    return std::max( baseDimension >> mipLevel, 1u );
}

inline uint2 mipDimensions( uint2 dimensions, uint32_t mipLevel )
{
    return make_uint2( mipDimension( dimensions.x, mipLevel ), mipDimension( dimensions.y, mipLevel ) );
}

template <typename T>
static DeviceSpan<T> allocArray( size_t count, bool zero = false, hipStream_t stream = nullptr )
{
    const size_t size = count * sizeof( T );
    if( size == 0 )
        return {};

    T* ptr = nullptr;
    HIP_CHECK( hipMalloc( &ptr, size ) );

    DeviceSpan<T> result( ptr, count );
    if( zero )
        memset( result, 0, stream );

    return result;
}

template <typename T>
static void freeArray( DeviceSpan<T>& span )
{
    if( span.ptr )
        HIP_CHECK( hipFree( span.ptr ) );

    span.ptr = nullptr;
    span.len = 0;
}

template <typename T>
struct HostSpan
{
    T*     ptr = nullptr;
    size_t len = 0;

    HostSpan() = default;

    explicit HostSpan( T* ptr_, size_t len_ )
        : ptr( ptr_ )
        , len( len_ )
    {
    }

    HIP_DEMAND_INLINE size_t sizeInBytes() const { return len * sizeof( T ); }
};

template <typename T>
static HostSpan<T> hostAllocArray( size_t count )
{
    const size_t size = count * sizeof( T );
    if( size == 0 )
        return {};

    T* ptr = nullptr;
    HIP_CHECK( hipHostAlloc( &ptr, size ) );

    HostSpan<T> result( ptr, count );
    return result;
}

template <typename T>
static void hostFreeArray( HostSpan<T>& span )
{
    if( span.ptr )
        HIP_CHECK( hipHostFree( span.ptr ) );

    span.ptr = nullptr;
    span.len = 0;
}

template <bool PinnedMemory>
class Bitset : NonCopyble
{
  public:
    Bitset( uint32_t count = 0 )
    {
        count_ = count;
        if constexpr( PinnedMemory )
        {
            words_ = hostAllocArray<uint32_t>( ceilDiv( count, 32u ) );
        }
        else
        {
            words_.ptr = new uint32_t[count];
            words_.len = count;
        }
        std::memset( words_.ptr, 0, words_.sizeInBytes() );
    }

    ~Bitset()
    {
        count_ = 0;
        if constexpr( PinnedMemory )
        {
            hostFreeArray( words_ );
        }
        else
        {
            delete[] words_.ptr;
            words_.ptr = nullptr;
        }
    }

    void set( uint32_t index, bool value )
    {
        validate( index );
        const uint32_t mask = 1u << ( index % 32u );
        if( value )
            words_.ptr[index / 32] |= mask;
        else
            words_.ptr[index / 32] &= ~mask;
    }

    bool test( uint32_t index ) const
    {
        validate( index );
        return ( words_.ptr[index / 32] & ( 1u << ( index % 32u ) ) ) != 0;
    }

    uint32_t                  bitCount() const { return count_; }
    uint32_t                  wordCount() const { return static_cast<uint32_t>( words_.len ); }
    const HostSpan<uint32_t>& words() const { return words_; }

  private:
    void validate( uint32_t index ) const
    {
        if( index >= count_ )
            throw std::out_of_range( "index is outside the bitset" );
    }

    uint32_t           count_ = 0;
    HostSpan<uint32_t> words_;
};

template <typename T>
static void memcpyDtoH( HostSpan<T>& dst, const DeviceSpan<T>& src, hipStream_t stream = nullptr )
{
    assert( dst.sizeInBytes() == src.sizeInBytes() );
    HIP_CHECK( hipMemcpyAsync( dst.ptr, src.ptr, dst.sizeInBytes(), hipMemcpyDeviceToHost, stream ) );
}

template <typename T>
static void memcpyHtoD( DeviceSpan<T>& dst, const HostSpan<T>& src, hipStream_t stream = nullptr )
{
    assert( dst.sizeInBytes() == src.sizeInBytes() );
    HIP_CHECK( hipMemcpyAsync( dst.ptr, src.ptr, dst.sizeInBytes(), hipMemcpyHostToDevice, stream ) );
}

template <typename T>
static void memset( const DeviceSpan<T>& dst, int value, hipStream_t stream = nullptr )
{
    HIP_CHECK( hipMemsetAsync( dst.ptr, value, dst.sizeInBytes(), stream ) );
}

}  // namespace hip_demand
