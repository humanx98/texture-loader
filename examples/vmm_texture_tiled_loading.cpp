#include <DemandLoading/VmmDemandTextureLoader.h>
#include <ImageSource/OIIOReader.h>
#include <ImageSource/TextureInfo.h>

#include "hip_check.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <mempulse/mempulse.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef TEST_IMAGES_DIR
#error "TEST_IMAGES_DIR must be defined by the build target"
#endif

#ifndef USE_OIIO
#error "This example requires a build configured with USE_OIIO=ON"
#endif

namespace fs = std::filesystem;

namespace {

class KernelModule
{
  public:
    explicit KernelModule( const fs::path& path )
    {
        HIP_CHECK( hipModuleLoad( &module_, path.string().c_str() ) );
        HIP_CHECK( hipModuleGetFunction( &regionKernel_, module_, "renderVmmTextureRegion" ) );
        HIP_CHECK( hipModuleGetFunction( &vmmTex2DLodKernel_, module_, "renderVmmTex2DLodBenchmark" ) );
        HIP_CHECK( hipModuleGetFunction( &hipTex2DLodKernel_, module_, "renderHipTex2DLodBenchmark" ) );
    }

    ~KernelModule()
    {
        if( module_ != nullptr )
            HIP_WARN( hipModuleUnload( module_ ) );
    }

    hipFunction_t regionKernel() const { return regionKernel_; }
    hipFunction_t vmmTex2DLodKernel() const { return vmmTex2DLodKernel_; }
    hipFunction_t hipTex2DLodKernel() const { return hipTex2DLodKernel_; }

    KernelModule( const KernelModule& )            = delete;
    KernelModule& operator=( const KernelModule& ) = delete;

  private:
    hipModule_t   module_            = nullptr;
    hipFunction_t regionKernel_      = nullptr;
    hipFunction_t vmmTex2DLodKernel_ = nullptr;
    hipFunction_t hipTex2DLodKernel_ = nullptr;
};

struct GpuResources
{
    explicit GpuResources( size_t outputBytes )
    {
        HIP_CHECK( hipStreamCreate( &stream ) );
        HIP_CHECK( hipMalloc( reinterpret_cast<void**>( &output ), outputBytes ) );
    }

    ~GpuResources()
    {
        // A render pass may still be in flight if writing the PNG throws.
        HIP_WARN( hipStreamSynchronize( stream ) );
        HIP_WARN( hipFree( output ) );
        HIP_WARN( hipStreamDestroy( stream ) );
    }

    GpuResources( const GpuResources& )            = delete;
    GpuResources& operator=( const GpuResources& ) = delete;

    hipStream_t stream = nullptr;
    uint8_t*    output = nullptr;
};

void saveGpuOutput( const fs::path& outputPath, hipStream_t stream, const uint8_t* deviceOutput, uint32_t width, uint32_t height, uint32_t channels )
{
    const size_t         byteCount = static_cast<size_t>( width ) * height * channels;
    std::vector<uint8_t> hostOutput( byteCount );
    HIP_CHECK( hipMemcpyAsync( hostOutput.data(), deviceOutput, byteCount, hipMemcpyDeviceToHost, stream ) );
    HIP_CHECK( hipStreamSynchronize( stream ) );

    if( stbi_write_png( outputPath.string().c_str(), static_cast<int>( width ), static_cast<int>( height ),
                        static_cast<int>( channels ), hostOutput.data(), static_cast<int>( width * channels ) )
        == 0 )
        throw std::runtime_error( "Failed to save output PNG: " + outputPath.string() );

    std::cout << "  Image saved: " << fs::absolute( outputPath ).string() << '\n';
}

class GpuTimer
{
  public:
    GpuTimer()
    {
        HIP_CHECK( hipEventCreate( &start_ ) );
        HIP_CHECK( hipEventCreate( &stop_ ) );
    }

    ~GpuTimer()
    {
        HIP_WARN( hipEventDestroy( start_ ) );
        HIP_WARN( hipEventDestroy( stop_ ) );
    }

    GpuTimer( const GpuTimer& )            = delete;
    GpuTimer& operator=( const GpuTimer& ) = delete;

    template <class Launch>
    float measure( hipStream_t stream, Launch&& launch )
    {
        HIP_CHECK( hipEventRecord( start_, stream ) );
        launch();
        HIP_CHECK( hipEventRecord( stop_, stream ) );
        HIP_CHECK( hipEventSynchronize( stop_ ) );

        float milliseconds = 0.0f;
        HIP_CHECK( hipEventElapsedTime( &milliseconds, start_, stop_ ) );
        return milliseconds;
    }

  private:
    hipEvent_t start_ = nullptr;
    hipEvent_t stop_  = nullptr;
};

class CompletionEvent
{
  public:
    CompletionEvent() { HIP_CHECK( hipEventCreateWithFlags( &event_, hipEventDisableTiming ) ); }
    ~CompletionEvent() { HIP_WARN( hipEventDestroy( event_ ) ); }

    CompletionEvent( const CompletionEvent& )            = delete;
    CompletionEvent& operator=( const CompletionEvent& ) = delete;

    void record( hipStream_t stream ) { HIP_CHECK( hipEventRecord( event_, stream ) ); }
    void wait() { HIP_CHECK( hipEventSynchronize( event_ ) ); }

  private:
    hipEvent_t event_ = nullptr;
};

class NativeMipmappedTexture
{
  public:
    NativeMipmappedTexture( hip_demand::ImageSource& image, const hip_demand::TextureInfo& info )
    {
        try
        {
            const hipChannelFormatDesc channelDesc = hipCreateChannelDesc<uchar4>();
            HIP_CHECK( hipMallocMipmappedArray( &mipmappedArray_, &channelDesc,
                                                make_hipExtent( info.width, info.height, 0 ), info.numMipLevels ) );

            uint32_t width  = info.width;
            uint32_t height = info.height;
            for( uint32_t level = 0; level < info.numMipLevels; ++level )
            {
                const size_t         pixelCount = static_cast<size_t>( width ) * height;
                std::vector<uint8_t> source( pixelCount * info.numChannels );
                if( !image.readMipLevel( reinterpret_cast<char*>( source.data() ), level, width, height ) )
                    throw std::runtime_error( "Could not read mip level " + std::to_string( level ) );

                std::vector<uchar4> rgba( pixelCount );
                for( size_t pixel = 0; pixel < pixelCount; ++pixel )
                {
                    const uint8_t* value = source.data() + pixel * info.numChannels;
                    rgba[pixel].x        = value[0];
                    rgba[pixel].y        = info.numChannels > 1 ? value[1] : value[0];
                    rgba[pixel].z        = info.numChannels > 2 ? value[2] : value[0];
                    rgba[pixel].w        = info.numChannels > 3 ? value[3] : 255;
                }

                hipArray_t levelArray = nullptr;
                HIP_CHECK( hipGetMipmappedArrayLevel( &levelArray, mipmappedArray_, level ) );
                const size_t rowBytes = static_cast<size_t>( width ) * sizeof( uchar4 );
                HIP_CHECK( hipMemcpy2DToArray( levelArray, 0, 0, rgba.data(), rowBytes, rowBytes, height, hipMemcpyHostToDevice ) );
                width  = std::max( 1u, width / 2 );
                height = std::max( 1u, height / 2 );
            }

            hipResourceDesc resourceDesc{};
            resourceDesc.resType           = hipResourceTypeMipmappedArray;
            resourceDesc.res.mipmap.mipmap = mipmappedArray_;

            hipTextureDesc textureDesc{};
            textureDesc.addressMode[0]      = hipAddressModeClamp;
            textureDesc.addressMode[1]      = hipAddressModeClamp;
            textureDesc.filterMode          = hipFilterModeLinear;
            textureDesc.readMode            = hipReadModeNormalizedFloat;
            textureDesc.normalizedCoords    = 1;
            textureDesc.mipmapFilterMode    = hipFilterModePoint;
            textureDesc.minMipmapLevelClamp = 0.0f;
            textureDesc.maxMipmapLevelClamp = static_cast<float>( info.numMipLevels - 1 );
            HIP_CHECK( hipCreateTextureObject( &texture_, &resourceDesc, &textureDesc, nullptr ) );
        }
        catch( ... )
        {
            reset();
            throw;
        }
    }

    ~NativeMipmappedTexture() { reset(); }

    NativeMipmappedTexture( const NativeMipmappedTexture& )            = delete;
    NativeMipmappedTexture& operator=( const NativeMipmappedTexture& ) = delete;

    hipTextureObject_t get() const { return texture_; }

  private:
    void reset()
    {
        if( texture_ != 0 )
            HIP_WARN( hipDestroyTextureObject( texture_ ) );
        if( mipmappedArray_ != nullptr )
            HIP_WARN( hipFreeMipmappedArray( mipmappedArray_ ) );
        texture_        = 0;
        mipmappedArray_ = nullptr;
    }

    hipMipmappedArray_t mipmappedArray_ = nullptr;
    hipTextureObject_t  texture_        = 0;
};

class GpuMemoryMonitor
{
  public:
    explicit GpuMemoryMonitor( int device )
        : device_( device )
    {
#ifdef _WIN32
        constexpr MempulseBackend backend = MEMPULSE_BACKEND_D3DKMT;
#else
        constexpr MempulseBackend backend = MEMPULSE_BACKEND_ANY;
#endif
        if( MempulseInitialize( &context_, backend ) != MEMPULSE_SUCCESS )
            throw std::runtime_error( "Could not initialize MemPulse" );

        int deviceCount = 0;
        if( MempulseGetAvailabeDeviceCount( context_, &deviceCount ) != MEMPULSE_SUCCESS || device_ >= deviceCount )
        {
            MempulseShutdown( context_ );
            context_ = nullptr;
            throw std::runtime_error( "MemPulse could not find the active HIP device" );
        }
    }

    ~GpuMemoryMonitor()
    {
        if( context_ != nullptr )
            MempulseShutdown( context_ );
    }

    GpuMemoryMonitor( const GpuMemoryMonitor& )            = delete;
    GpuMemoryMonitor& operator=( const GpuMemoryMonitor& ) = delete;

    MempulseDeviceMemoryInfo memoryInfo() const
    {
        MempulseDeviceMemoryInfo info{};
        if( MempulseGetDeviceMemoryInfo( context_, device_, &info ) != MEMPULSE_SUCCESS )
            throw std::runtime_error( "MemPulse could not read GPU memory usage" );
        return info;
    }

  private:
    MempulseContext context_ = nullptr;
    int             device_  = 0;
};

std::shared_ptr<hip_demand::ImageSource> readImage( const fs::path& path )
{
    auto image = std::make_shared<hip_demand::OIIOReader>( path.string(), HIP_AD_FORMAT_UNSIGNED_INT8 );

    hip_demand::TextureInfo info{};
    image->open( &info );
    if( !info.isValid || info.width == 0 || info.height == 0 )
        throw std::runtime_error( "Invalid image: " + path.string() );

    return image;
}

std::vector<fs::path> findImages( const fs::path& directory, bool includeOver4k )
{
    constexpr uint32_t max4kDimension = 4096;

    if( !fs::is_directory( directory ) )
        throw std::runtime_error( "Image directory not found: " + directory.string() );

    std::vector<fs::path> imagePaths;
    for( const fs::directory_entry& entry : fs::directory_iterator( directory ) )
    {
        if( !entry.is_regular_file() )
            continue;

        std::string extension = entry.path().extension().string();
        std::transform( extension.begin(), extension.end(), extension.begin(),
                        []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );
        if( extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp"
            || extension == ".tga" )
            imagePaths.push_back( entry.path() );
    }
    std::sort( imagePaths.begin(), imagePaths.end() );
    if( imagePaths.empty() )
        throw std::runtime_error( "No supported images found in: " + directory.string() );

    if( !includeOver4k )
    {
        std::vector<fs::path> selectedPaths;
        selectedPaths.reserve( imagePaths.size() );
        for( const fs::path& path : imagePaths )
        {
            const auto                     image = readImage( path );
            const hip_demand::TextureInfo& info  = image->getInfo();
            if( info.width > max4kDimension || info.height > max4kDimension )
            {
                std::cout << "  Skipped (>4K): " << path.filename().string() << " (" << info.width << 'x' << info.height << ")\n";
                continue;
            }
            selectedPaths.push_back( path );
        }
        imagePaths.swap( selectedPaths );
    }

    if( imagePaths.empty() )
        throw std::runtime_error( "No images up to 4K found in: " + directory.string() );

    return imagePaths;
}

fs::path findLargestBenchmarkImage( const fs::path& directory )
{
    hipDeviceProp_t deviceProperties{};
    HIP_CHECK( hipGetDeviceProperties( &deviceProperties, 0 ) );
    const uint32_t maxWidth =
        static_cast<uint32_t>( deviceProperties.maxTexture2DMipmap[0] > 0 ? deviceProperties.maxTexture2DMipmap[0] :
                                                                            deviceProperties.maxTexture2D[0] );
    const uint32_t maxHeight =
        static_cast<uint32_t>( deviceProperties.maxTexture2DMipmap[1] > 0 ? deviceProperties.maxTexture2DMipmap[1] :
                                                                            deviceProperties.maxTexture2D[1] );

    const std::vector<fs::path> imagePaths = findImages( directory, true );
    fs::path                    largestPath;
    uint64_t                    largestPixelCount = 0;
    for( const fs::path& imagePath : imagePaths )
    {
        const auto     image  = readImage( imagePath );
        const auto&    info   = image->getInfo();
        const uint64_t pixels = static_cast<uint64_t>( info.width ) * info.height;
        if( info.width > maxWidth || info.height > maxHeight )
        {
            std::cout << "  Benchmark skipped (exceeds native mipmapped texture limit " << maxWidth << 'x' << maxHeight
                      << "): " << imagePath.filename().string() << " (" << info.width << 'x' << info.height << ")\n";
            continue;
        }
        if( pixels > largestPixelCount )
        {
            largestPixelCount = pixels;
            largestPath       = imagePath;
        }
    }
    if( largestPath.empty() )
        throw std::runtime_error( "No test image fits the native mipmapped texture limit" );
    return largestPath;
}

template <class Launch>
void measureRenderCalls( const char* label, hipStream_t stream, uint32_t callCount, Launch&& launch )
{
    GpuTimer           timer;
    std::vector<float> times;
    times.reserve( callCount );

    std::cout << "\n  " << label << "\n" << std::fixed << std::setprecision( 3 );
    for( uint32_t call = 0; call < callCount; ++call )
    {
        const float milliseconds = timer.measure( stream, launch );
        times.push_back( milliseconds );
    }

    const auto [minimum, maximum] = std::minmax_element( times.begin(), times.end() );
    const double average          = std::accumulate( times.begin(), times.end(), 0.0 ) / times.size();
    std::cout << "    Min: " << *minimum << " ms | Max: " << *maximum << " ms | Average: " << average << " ms\n"
              << std::defaultfloat;
}

template <class Launch>
void measureContinuouslyQueuedRenderCalls( const char* label, hipStream_t stream, uint32_t callCount, uint32_t batchSize, Launch&& launch )
{
    if( batchSize == 0 )
        throw std::invalid_argument( "Batch size must be greater than zero" );

    constexpr uint32_t                              bufferedBatchCount = 2;
    std::array<CompletionEvent, bufferedBatchCount> batchFinished;
    GpuTimer                                        timer;

    const float milliseconds = timer.measure( stream, [&] {
        uint32_t submittedCalls   = 0;
        uint32_t submittedBatches = 0;
        while( submittedCalls < callCount )
        {
            const uint32_t slot = submittedBatches % bufferedBatchCount;
            // The other batch remains queued while the CPU waits to reuse this slot.
            if( submittedBatches >= bufferedBatchCount )
                batchFinished[slot].wait();

            const uint32_t callsInBatch = std::min( batchSize, callCount - submittedCalls );
            for( uint32_t call = 0; call < callsInBatch; ++call )
                launch();

            batchFinished[slot].record( stream );
            submittedCalls += callsInBatch;
            ++submittedBatches;
        }
    } );

    const uint32_t maxCallsInFlight = batchSize * bufferedBatchCount;
    std::cout << "\n  " << label << "\n"
              << "    Calls: " << callCount << " | Streams: 1 | Batch size: " << batchSize << '\n'
              << "    Buffered batches: " << bufferedBatchCount << " | Max calls in flight: " << maxCallsInFlight << '\n'
              << std::fixed << std::setprecision( 3 ) << "    Total rendering time: " << milliseconds << " ms\n"
              << std::defaultfloat;
}

std::unique_ptr<hip_demand::vmm::DemandTextureLoader> createLoaderForImages( const hip_demand::vmm::Options& options,
                                                                             const std::vector<fs::path>& imagePaths )
{
    using namespace hip_demand::vmm;

    auto              loader = createDemandTextureLoader( options );
    TextureDescriptor descriptor{};
    descriptor.addressMode[0]   = hipAddressModeClamp;
    descriptor.addressMode[1]   = hipAddressModeClamp;
    descriptor.filterMode       = hipFilterModeLinear;
    descriptor.mipmapFilterMode = hipFilterModePoint;
    descriptor.normalizedCoords = true;

    for( const fs::path& imagePath : imagePaths )
    {
        std::shared_ptr<hip_demand::ImageSource> image = readImage( imagePath );
        loader->createTexture( image, descriptor );
    }
    return loader;
}

void benchmarkTex2DLod( const fs::path& executableDir, const fs::path& imagePath )
{
    using namespace hip_demand::vmm;

    uint32_t           outputWidth     = 3840;
    uint32_t           outputHeight    = 2160;
    constexpr uint32_t blockWidth      = 16;
    constexpr uint32_t blockHeight     = 16;
    constexpr uint32_t channels        = 4;
    constexpr uint32_t maxWarmupPasses = 256;
    constexpr uint32_t renderCallCount = 1000;
    constexpr uint32_t batchSize       = 10;
    float              mipLevel        = 0.0f;

    const fs::path vmmOutputPath{ "vmm_texture_tiled_loading_tex2dlod_vmm.png" };
    const fs::path hipOutputPath{ "vmm_texture_tiled_loading_tex2dlod_hip_texture.png" };

    Options options{};
    options.maxTextures      = 1;
    options.maxVirtualPages  = 64 * 1024;
    options.maxPhysicalPages = 64 * 1024;
    options.maxRequests      = 8192;
    options.enableEviction   = false;
    options.maxEvictedPages  = 0;

    const fs::path kernelPath = executableDir / "vmm_texture_tiled_loading_kernel.co";
    if( !fs::exists( kernelPath ) )
        throw std::runtime_error( "HIP module not found: " + kernelPath.string() );

    HIP_CHECK( hipSetDevice( 0 ) );
    const auto                     image = readImage( imagePath );
    const hip_demand::TextureInfo& info  = image->getInfo();

    std::cout << "\ntex2DLod performance: VMM vs hipTextureObject_t\n"
              << "  Image: " << imagePath.filename().string() << " (" << info.width << 'x' << info.height << ")\n"
              << "  Available mip levels: " << info.numMipLevels << '\n'
              << "  Rendered mip level: " << mipLevel << '\n'
              << "  Output size: " << outputWidth << 'x' << outputHeight << " pixels\n"
              << "  Layout: full image\n"
              << "  Timing: GPU elapsed time after resources are resident\n";

    KernelModule   module( kernelPath );
    const size_t   byteCount = static_cast<size_t>( outputWidth ) * outputHeight * channels;
    GpuResources   gpu( byteCount );
    const uint32_t gridWidth  = ( outputWidth + blockWidth - 1 ) / blockWidth;
    const uint32_t gridHeight = ( outputHeight + blockHeight - 1 ) / blockHeight;

    {
        auto              loader = createDemandTextureLoader( options );
        TextureDescriptor descriptor{};
        descriptor.addressMode[0]   = hipAddressModeClamp;
        descriptor.addressMode[1]   = hipAddressModeClamp;
        descriptor.filterMode       = hipFilterModeLinear;
        descriptor.mipmapFilterMode = hipFilterModePoint;
        descriptor.normalizedCoords = true;
        loader->createTexture( image, descriptor );

        DeviceContext context{};
        bool          fullyResident = false;
        uint32_t      warmupPasses  = 0;
        for( ; warmupPasses < maxWarmupPasses; ++warmupPasses )
        {
            loader->launchPrepare( gpu.stream, context );
            DeviceContext mutableContext = context;
            void*         arguments[]    = { &mutableContext, &gpu.output, &outputWidth, &outputHeight, &mipLevel };
            HIP_CHECK( hipModuleLaunchKernel( module.vmmTex2DLodKernel(), gridWidth, gridHeight, 1, blockWidth,
                                              blockHeight, 1, 0, gpu.stream, arguments, nullptr ) );

            Ticket ticket = loader->processRequests( gpu.stream, context );
            ticket.wait();
            HIP_CHECK( hipStreamSynchronize( gpu.stream ) );
            if( ticket.numTasksTotal() == 0 )
            {
                fullyResident = true;
                ++warmupPasses;
                break;
            }
        }
        if( !fullyResident )
            throw std::runtime_error( "VMM benchmark texture did not become resident" );

        std::cout << "  VMM warm-up complete: all sampled mip pages resident after " << warmupPasses << " passes\n";
        const auto launchVmm = [&] {
            DeviceContext mutableContext = context;
            void*         arguments[]    = { &mutableContext, &gpu.output, &outputWidth, &outputHeight, &mipLevel };
            HIP_CHECK( hipModuleLaunchKernel( module.vmmTex2DLodKernel(), gridWidth, gridHeight, 1, blockWidth,
                                              blockHeight, 1, 0, gpu.stream, arguments, nullptr ) );
        };
        measureRenderCalls( "VMM tex2DLod", gpu.stream, renderCallCount, launchVmm );
        measureContinuouslyQueuedRenderCalls( "VMM tex2DLod continuously queued throughput", gpu.stream,
                                              renderCallCount, batchSize, launchVmm );
        saveGpuOutput( vmmOutputPath, gpu.stream, gpu.output, outputWidth, outputHeight, channels );
    }

    {
        std::cout << "\n  Uploading every mip level to hipTextureObject_t...\n";
        NativeMipmappedTexture texture( *image, info );
        hipTextureObject_t     textureObject = texture.get();
        const auto             launchHip     = [&] {
            void* arguments[] = { &textureObject, &gpu.output, &outputWidth, &outputHeight, &mipLevel };
            HIP_CHECK( hipModuleLaunchKernel( module.hipTex2DLodKernel(), gridWidth, gridHeight, 1, blockWidth,
                                              blockHeight, 1, 0, gpu.stream, arguments, nullptr ) );
        };

        launchHip();
        HIP_CHECK( hipStreamSynchronize( gpu.stream ) );
        std::cout << "  hipTextureObject_t warm-up complete\n";
        measureRenderCalls( "hipTextureObject_t tex2DLod", gpu.stream, renderCallCount, launchHip );
        measureContinuouslyQueuedRenderCalls( "hipTextureObject_t tex2DLod continuously queued throughput", gpu.stream,
                                              renderCallCount, batchSize, launchHip );
        saveGpuOutput( hipOutputPath, gpu.stream, gpu.output, outputWidth, outputHeight, channels );
    }
}

void renderTextureGrid( const fs::path& executableDir, const std::vector<fs::path>& imagePaths, bool enableEviction )
{
    using namespace hip_demand::vmm;

    uint32_t           outputWidth      = 3840;
    uint32_t           outputHeight     = 2160;
    constexpr uint32_t blockWidth       = 16;
    constexpr uint32_t blockHeight      = 16;
    constexpr uint32_t channels         = 4;
    constexpr uint32_t outputTileSize   = 512;
    constexpr uint32_t maxPassesPerTile = 64;
    constexpr double   bytesPerMiB      = 1024.0 * 1024.0;

    Options options{};
    options.maxVirtualPages  = 64 * 1024;
    options.maxPhysicalPages = 64 * 1024;
    options.maxRequests      = 1024;
    options.maxEvictedPages  = 1024;
    options.enableEviction   = enableEviction;

    const fs::path outputPath = enableEviction ? "vmm_texture_tiled_loading_eviction.png"
                                                : "vmm_texture_tiled_loading_no_eviction.png";
    const fs::path kernelPath = executableDir / "vmm_texture_tiled_loading_kernel.co";
    if( !fs::exists( kernelPath ) )
        throw std::runtime_error( "HIP module not found: " + kernelPath.string() );

    std::cout << "\nTiled render (eviction "
              << ( enableEviction ? "enabled" : "disabled" ) << ")\n"
              << "  Output size: " << outputWidth << 'x' << outputHeight << " pixels\n"
              << "  Output tile size: " << outputTileSize << 'x' << outputTileSize << " pixels\n";

    HIP_CHECK( hipSetDevice( 0 ) );
    // MemPulse resets the HIP device on shutdown, so keep it alive until the render resources are destroyed.
    GpuMemoryMonitor memoryMonitor( 0 );
    KernelModule module( kernelPath );

    auto loader = createLoaderForImages( options, imagePaths );

    uint32_t textureCount = static_cast<uint32_t>( imagePaths.size() );
    uint32_t columnCount  = static_cast<uint32_t>( std::ceil( std::sqrt( static_cast<double>( textureCount ) ) ) );
    uint32_t rowCount     = ( textureCount + columnCount - 1 ) / columnCount;

    const size_t byteCount = static_cast<size_t>( outputWidth ) * outputHeight * channels;
    GpuResources gpu( byteCount );

    std::vector<uint32_t> previousResidence;
    std::vector<uint32_t> currentResidence;
    uint32_t              evictedPages   = 0;
    uint32_t              completedTiles = 0;
    const uint32_t        tilesPerRow    = ( outputWidth + outputTileSize - 1 ) / outputTileSize;
    const uint32_t        tileRows       = ( outputHeight + outputTileSize - 1 ) / outputTileSize;
    const auto            renderStart    = std::chrono::steady_clock::now();

    for( uint32_t regionY = 0; regionY < outputHeight; regionY += outputTileSize )
    {
        for( uint32_t regionX = 0; regionX < outputWidth; regionX += outputTileSize )
        {
            uint32_t       regionWidth         = std::min( outputTileSize, outputWidth - regionX );
            uint32_t       regionHeight        = std::min( outputTileSize, outputHeight - regionY );
            const uint32_t gridWidth           = ( regionWidth + blockWidth - 1 ) / blockWidth;
            const uint32_t gridHeight          = ( regionHeight + blockHeight - 1 ) / blockHeight;
            bool           complete            = false;
            uint32_t       passesForTile       = 0;
            const uint32_t evictionsBeforeTile = evictedPages;

            for( uint32_t pass = 0; pass < maxPassesPerTile; ++pass )
            {
                DeviceContext context{};
                loader->launchPrepare( gpu.stream, context );

                DeviceContext mutableContext = context;
                void*         arguments[]    = { &mutableContext, &gpu.output,  &outputWidth, &outputHeight,
                                                 &textureCount,   &columnCount, &rowCount,    &regionX,
                                                 &regionY,        &regionWidth, &regionHeight };
                HIP_CHECK( hipModuleLaunchKernel( module.regionKernel(), gridWidth, gridHeight, 1, blockWidth,
                                                  blockHeight, 1, 0, gpu.stream, arguments, nullptr ) );

                Ticket ticket = loader->processRequests( gpu.stream, context );
                ticket.wait();
                ++passesForTile;

                if( enableEviction && currentResidence.empty() )
                {
                    currentResidence.resize( context.residenceBits.len );
                    previousResidence.resize( context.residenceBits.len, 0 );
                }

                if( enableEviction )
                    HIP_CHECK( hipMemcpyAsync( currentResidence.data(), context.residenceBits.ptr,
                                               context.residenceBits.sizeInBytes(), hipMemcpyDeviceToHost, gpu.stream ) );
                HIP_CHECK( hipStreamSynchronize( gpu.stream ) );

                if( enableEviction )
                {
                    // A resident tile bit changing from 1 to 0 proves eviction occurred.
                    for( uint32_t resourceId = context.resourceTable.textureTiles.start;
                         resourceId < context.resourceTable.textureTiles.end(); ++resourceId )
                    {
                        const uint32_t mask = 1u << ( resourceId & 31u );
                        const uint32_t word = resourceId >> 5;
                        if( ( previousResidence[word] & mask ) && !( currentResidence[word] & mask ) )
                            ++evictedPages;
                    }
                    previousResidence.swap( currentResidence );
                }

                // All resources sampled by this output tile were resident during the pass.
                if( ticket.numTasksTotal() == 0 )
                {
                    complete = true;
                    ++completedTiles;
                    break;
                }
            }

            if( !complete )
                throw std::runtime_error( "Output tile (" + std::to_string( regionX ) + ", " + std::to_string( regionY )
                                          + ") did not become resident" );

            std::cout << "  Tile " << completedTiles << '/' << tilesPerRow * tileRows << " at (" << regionX << ", "
                      << regionY << ") | passes: " << passesForTile;
            if( enableEviction )
                std::cout << " | evictions: +" << evictedPages - evictionsBeforeTile << " (total " << evictedPages
                          << ')';
            const MempulseDeviceMemoryInfo memory = memoryMonitor.memoryInfo();
            std::cout << " | GPU VRAM: " << memory.dedicatedUsed / bytesPerMiB << " / "
                      << memory.dedicatedTotal / bytesPerMiB << " MiB";
            std::cout << '\n';
        }
    }

    const auto renderEnd  = std::chrono::steady_clock::now();
    const auto renderTime = std::chrono::duration_cast<std::chrono::milliseconds>( renderEnd - renderStart );
    std::cout << "  Render complete: " << completedTiles << '/' << tilesPerRow * tileRows << " tiles in "
              << renderTime.count() << " ms";
    if( enableEviction )
        std::cout << "; VMM page evictions: " << evictedPages;
    std::cout << '\n';

    if( enableEviction && evictedPages == 0 )
        throw std::runtime_error( "Eviction test loaded every output tile without evicting a VMM page" );

    saveGpuOutput( outputPath, gpu.stream, gpu.output, outputWidth, outputHeight, channels );
}

}  // namespace

int main( int argc, char** argv )
{
    try
    {
        constexpr bool includeOver4kImages = true;
        const fs::path executableDir       = fs::absolute( fs::path{ argv[0] } ).parent_path();
        fs::path       imageDirectory      = fs::path{ TEST_IMAGES_DIR } / "png";
        imageDirectory.make_preferred();

        std::cout << "VMM texture loading tests\n"
                  << "  Image directory: " << imageDirectory.string() << '\n'
                  << "  Size filter: " << ( includeOver4kImages ? "all images" : "up to 4096 pixels per side" )
                  << "\n  GPU VRAM logging: enabled\n";
        const auto imagePaths = findImages( imageDirectory, includeOver4kImages );
        std::cout << "  Selected textures (" << imagePaths.size() << "):\n";
        for( const fs::path& imagePath : imagePaths )
        {
            const auto  image = readImage( imagePath );
            const auto& info  = image->getInfo();
            std::cout << "    " << imagePath.filename().string() << " (" << info.width << 'x' << info.height << ")\n";
        }
        std::cout << "  GPU VRAM readings cover the whole device, including other processes.\n";

        renderTextureGrid( executableDir, imagePaths, false );
        renderTextureGrid( executableDir, imagePaths, true );
        benchmarkTex2DLod( executableDir, findLargestBenchmarkImage( imageDirectory ) );
        std::cout << "\nResult: PASS (renders, eviction test, and tex2DLod benchmark completed)\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr << "\nResult: FAIL - " << error.what() << '\n';
        return 1;
    }
}
