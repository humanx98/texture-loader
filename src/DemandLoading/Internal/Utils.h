// SPDX-License-Identifier: MIT
// Internal utility functions

#pragma once

#include <DemandLoading/DeviceContext.h>
#include <array>
#include <cassert>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <limits>
#include <optional>
#include <vector>

namespace hip_demand {
namespace internal {

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

class Bitset : NonCopyble
{
  public:
    void resize( uint32_t count )
    {
        count_ = count;
        words_.resize( ceilDiv( count, 32 ), 0 );
    }

    void set( uint32_t index, bool value )
    {
        validate( index );
        const uint32_t mask = 1u << ( index % 32u );
        if( value )
            words_[index / 32] |= mask;
        else
            words_[index / 32] &= ~mask;
    }

    bool test( uint32_t index ) const
    {
        validate( index );
        return ( words_[index / 32] & ( 1u << ( index % 32u ) ) ) != 0;
    }

    uint32_t                     bitCount() const { return count_; }
    uint32_t                     wordCount() const { return static_cast<uint32_t>( words_.size() ); }
    const std::vector<uint32_t>& words() const { return words_; }
    std::vector<uint32_t>&       words() { return words_; }

  private:
    void validate( uint32_t index ) const
    {
        if( index >= count_ )
            throw std::out_of_range( "index is outside the bitset" );
    }

    uint32_t              count_ = 0;
    std::vector<uint32_t> words_;
};

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
    const uint32_t dimension = baseDimension >> mipLevel;
    return dimension == 0 ? 1u : dimension;
}

template <typename T>
static void memcpyHtoD( const DeviceSpan<T>& dst, const std::vector<T>& src )
{
    assert( dst.sizeInBytes() == sizeInBytes( src ) );
    HIP_CHECK( hipMemcpy( dst.ptr, src.data(), dst.sizeInBytes(), hipMemcpyHostToDevice ) );
}

template <typename T>
static void memcpyHtoDAsync( const DeviceSpan<T>& dst, const std::vector<T>& src, hipStream_t stream )
{
    assert( dst.sizeInBytes() == sizeInBytes( src ) );
    HIP_CHECK( hipMemcpyAsync( dst.ptr, src.data(), dst.sizeInBytes(), hipMemcpyHostToDevice, stream ) );
}

template <typename T>
static void memcpyDtoH( std::vector<T>& dst, const DeviceSpan<T>& src )
{
    assert( sizeInBytes( dst ) == src.sizeInBytes() );
    HIP_CHECK( hipMemcpy( dst.data(), src.ptr, sizeInBytes( dst ), hipMemcpyDeviceToHost ) );
}

template <typename T>
static void memcpyDtoHAsync( std::vector<T>& dst, const DeviceSpan<T>& src, hipStream_t stream )
{
    assert( sizeInBytes( dst ) == src.sizeInBytes() );
    HIP_CHECK( hipMemcpyAsync( dst.data(), src.ptr, sizeInBytes( dst ), hipMemcpyDeviceToHost, stream ) );
}

template <typename T, size_t N>
static void memcpyDtoHAsync( std::array<T, N>& dst, const DeviceSpan<T>& src, hipStream_t stream )
{
    assert( sizeInBytes( dst ) == src.sizeInBytes() );
    HIP_CHECK( hipMemcpyAsync( dst.data(), src.ptr, sizeInBytes( dst ), hipMemcpyDeviceToHost, stream ) );
}

template <typename T>
static void memcpyHtoD( const DeviceSpan<T>& dst, const std::vector<T>& src, size_t count )
{
    size_t bytes = count * sizeof( T );
    assert( bytes <= sizeInBytes( src ) );
    assert( bytes <= dst.sizeInBytes() );
    HIP_CHECK( hipMemcpy( dst.ptr, src.data(), bytes, hipMemcpyHostToDevice ) );
}

template <typename T>
static void memcpyHtoDAsync( const DeviceSpan<T>& dst, const std::vector<T>& src, size_t count, hipStream_t stream )
{
    size_t bytes = count * sizeof( T );
    assert( bytes <= sizeInBytes( src ) );
    assert( bytes <= dst.sizeInBytes() );
    HIP_CHECK( hipMemcpyAsync( dst.ptr, src.data(), bytes, hipMemcpyHostToDevice, stream ) );
}

template <typename T>
static void memcpyDtoH( std::vector<T>& dst, const DeviceSpan<T>& src, size_t count )
{
    size_t bytes = count * sizeof( T );
    assert( bytes <= src.sizeInBytes() );
    assert( bytes <= sizeInBytes( dst ) );
    HIP_CHECK( hipMemcpy( dst.data(), src.ptr, bytes, hipMemcpyDeviceToHost ) );
}

template <typename T>
static void memcpyDtoHAsync( std::vector<T>& dst, const DeviceSpan<T>& src, size_t count, hipStream_t stream )
{
    size_t bytes = count * sizeof( T );
    assert( bytes <= src.sizeInBytes() );
    assert( bytes <= sizeInBytes( dst ) );
    HIP_CHECK( hipMemcpyAsync( dst.data(), src.ptr, bytes, hipMemcpyDeviceToHost, stream ) );
}

template <typename T>
static void memset( const DeviceSpan<T>& dst, int value )
{
    HIP_CHECK( hipMemset( dst.ptr, value, dst.sizeInBytes() ) );
}

template <typename T>
static void memsetAsync( const DeviceSpan<T>& dst, int value, hipStream_t stream )
{
    HIP_CHECK( hipMemsetAsync( dst.ptr, value, dst.sizeInBytes(), stream ) );
}

}  // namespace internal
}  // namespace hip_demand
