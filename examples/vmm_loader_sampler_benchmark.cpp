#include <DemandLoading/VmmDemandTextureLoader.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vmm_loader_common.h"

#include <array>
#include <iomanip>

#ifndef TEST_IMAGES_DIR
#error "TEST_IMAGES_DIR must be defined by the build target"
#endif

namespace {

using namespace vmm_loader_example;

class KernelModule
{
  public:
    explicit KernelModule( const fs::path& path )
    {
        HIP_CHECK( hipModuleLoad( &module_, path.string().c_str() ) );
        HIP_CHECK( hipModuleGetFunction( &vmmTex2DLodKernel_, module_, "renderVmmTex2DLodBenchmark" ) );
        HIP_CHECK( hipModuleGetFunction( &hipTex2DLodKernel_, module_, "renderHipTex2DLodBenchmark" ) );
    }

    ~KernelModule()
    {
        if( module_ != nullptr )
            HIP_WARN( hipModuleUnload( module_ ) );
    }

    KernelModule( const KernelModule& )            = delete;
    KernelModule& operator=( const KernelModule& ) = delete;

    hipFunction_t vmmTex2DLodKernel() const { return vmmTex2DLodKernel_; }
    hipFunction_t hipTex2DLodKernel() const { return hipTex2DLodKernel_; }

  private:
    hipModule_t   module_            = nullptr;
    hipFunction_t vmmTex2DLodKernel_ = nullptr;
    hipFunction_t hipTex2DLodKernel_ = nullptr;
};

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

    void start( hipStream_t stream ) { HIP_CHECK( hipEventRecord( start_, stream ) ); }

    float stop( hipStream_t stream )
    {
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

class HipMipmappedTexture
{
  public:
    HipMipmappedTexture( hip_demand::ImageSource& image, const hip_demand::TextureInfo& info )
    {
        try
        {
            const hipChannelFormatDesc channelDesc = hipCreateChannelDesc<uchar4>();
            HIP_CHECK( hipMallocMipmappedArray( &mipmappedArray_, &channelDesc,
                                                make_hipExtent( info.width, info.height, 0 ), info.numMipLevels ) );
            uploadMipLevels( image, info );
            createTextureObject( info.numMipLevels );
        }
        catch( ... )
        {
            reset();
            throw;
        }
    }

    ~HipMipmappedTexture() { reset(); }

    HipMipmappedTexture( const HipMipmappedTexture& )            = delete;
    HipMipmappedTexture& operator=( const HipMipmappedTexture& ) = delete;

    hipTextureObject_t get() const { return texture_; }

  private:
    void uploadMipLevels( hip_demand::ImageSource& image, const hip_demand::TextureInfo& info )
    {
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
    }

    void createTextureObject( uint32_t mipLevelCount )
    {
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
        textureDesc.maxMipmapLevelClamp = static_cast<float>( mipLevelCount - 1 );
        HIP_CHECK( hipCreateTextureObject( &texture_, &resourceDesc, &textureDesc, nullptr ) );
    }

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

fs::path findLargestBenchmarkImage( const fs::path& directory, bool includeOver4k )
{
    hipDeviceProp_t deviceProperties{};
    HIP_CHECK( hipGetDeviceProperties( &deviceProperties, 0 ) );
    const uint32_t maxWidth =
        static_cast<uint32_t>( deviceProperties.maxTexture2DMipmap[0] > 0 ? deviceProperties.maxTexture2DMipmap[0] :
                                                                            deviceProperties.maxTexture2D[0] );
    const uint32_t maxHeight =
        static_cast<uint32_t>( deviceProperties.maxTexture2DMipmap[1] > 0 ? deviceProperties.maxTexture2DMipmap[1] :
                                                                            deviceProperties.maxTexture2D[1] );

    const std::vector<fs::path> imagePaths = findImages( directory, includeOver4k );
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
uint32_t makeVmmTextureResident( hip_demand::vmm::DemandTextureLoader& loader,
                                 hipStream_t                           stream,
                                 hip_demand::vmm::DeviceContext&       context,
                                 uint32_t                              maxPasses,
                                 Launch&&                              launch )
{
    using namespace hip_demand::vmm;

    for( uint32_t pass = 1; pass <= maxPasses; ++pass )
    {
        loader.launchPrepare( stream, context );
        launch( context );

        Ticket ticket = loader.processRequests( stream, context );
        ticket.wait();
        HIP_CHECK( hipStreamSynchronize( stream ) );
        if( ticket.numTasksTotal() == 0 )
            return pass;
    }
    throw std::runtime_error( "VMM benchmark texture did not become resident" );
}

template <class Launch>
void measureQueuedRenderCalls( const char* label, hipStream_t stream, uint32_t callCount, uint32_t batchSize, Launch&& launch )
{
    if( batchSize == 0 )
        throw std::invalid_argument( "Batch size must be greater than zero" );

    constexpr uint32_t                              bufferedBatchCount = 2;
    std::array<CompletionEvent, bufferedBatchCount> batchFinished;
    GpuTimer                                        timer;
    uint32_t                                        submittedCalls = 0;
    uint32_t                                        batchIndex     = 0;

    timer.start( stream );
    while( submittedCalls < callCount )
    {
        // Keep one batch queued while the CPU waits to reuse the other batch's event.
        const uint32_t eventIndex = batchIndex % bufferedBatchCount;
        if( batchIndex >= bufferedBatchCount )
            batchFinished[eventIndex].wait();

        const uint32_t callsInBatch = std::min( batchSize, callCount - submittedCalls );
        for( uint32_t call = 0; call < callsInBatch; ++call )
            launch();

        batchFinished[eventIndex].record( stream );
        submittedCalls += callsInBatch;
        ++batchIndex;
    }

    const float    milliseconds     = timer.stop( stream );
    const uint32_t maxCallsInFlight = batchSize * bufferedBatchCount;
    std::cout << "\n  " << label << "\n"
              << "    Calls: " << callCount << " | Batch size: " << batchSize << '\n'
              << "    Buffered batches: " << bufferedBatchCount << " | Max calls in flight: " << maxCallsInFlight << '\n'
              << std::fixed << std::setprecision( 3 ) << "    Total rendering time: " << milliseconds << " ms\n"
              << std::defaultfloat;
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
    constexpr uint32_t callCount       = 1000;
    constexpr uint32_t batchSize       = 10;
    float              mipLevel        = 0.0f;

    const fs::path vmmOutputPath{ "vmm_loader_sampler_benchmark_vmm.png" };
    const fs::path hipOutputPath{ "vmm_loader_sampler_benchmark_hip_texture.png" };

    Options options{};
    options.maxTextures      = 1;
    options.maxVirtualPages  = 64 * 1024;
    options.maxPhysicalPages = 64 * 1024;
    options.maxRequests      = 8192;
    options.enableEviction   = false;
    options.maxEvictedPages  = 0;

    const fs::path kernelPath = executableDir / "vmm_loader_sampler_benchmark_kernel.co";
    if( !fs::exists( kernelPath ) )
        throw std::runtime_error( "HIP module not found: " + kernelPath.string() );

    HIP_CHECK( hipSetDevice( 0 ) );
    const auto                     image = readImage( imagePath );
    const hip_demand::TextureInfo& info  = image->getInfo();

    std::cout << "\nTexture sampler performance: VMM vs hipTextureObject_t\n"
              << "  Image: " << imagePath.filename().string() << " (" << info.width << 'x' << info.height << ")\n"
              << "  Available mip levels: " << info.numMipLevels << '\n'
              << "  Rendered mip level: " << mipLevel << '\n'
              << "  Output size: " << outputWidth << 'x' << outputHeight << " pixels\n"
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
        const auto    launchVmmKernel = [&]( const DeviceContext& deviceContext ) {
            DeviceContext mutableContext = deviceContext;
            void*         arguments[]    = { &mutableContext, &gpu.output, &outputWidth, &outputHeight, &mipLevel };
            HIP_CHECK( hipModuleLaunchKernel( module.vmmTex2DLodKernel(), gridWidth, gridHeight, 1, blockWidth,
                                              blockHeight, 1, 0, gpu.stream, arguments, nullptr ) );
        };
        const uint32_t warmupPasses = makeVmmTextureResident( *loader, gpu.stream, context, maxWarmupPasses, launchVmmKernel );
        std::cout << "  VMM warm-up complete after " << warmupPasses << " passes\n";

        const auto launchVmm = [&] { launchVmmKernel( context ); };
        measureQueuedRenderCalls( "VMM tex2DLod continuously queued throughput", gpu.stream, callCount, batchSize, launchVmm );
        saveGpuOutput( vmmOutputPath, gpu.stream, gpu.output, outputWidth, outputHeight, channels );
    }

    {
        std::cout << "\n  Uploading every mip level to hipTextureObject_t...\n";
        HipMipmappedTexture texture( *image, info );
        hipTextureObject_t  textureObject = texture.get();
        const auto          launchHip     = [&] {
            void* arguments[] = { &textureObject, &gpu.output, &outputWidth, &outputHeight, &mipLevel };
            HIP_CHECK( hipModuleLaunchKernel( module.hipTex2DLodKernel(), gridWidth, gridHeight, 1, blockWidth,
                                              blockHeight, 1, 0, gpu.stream, arguments, nullptr ) );
        };

        launchHip();
        HIP_CHECK( hipStreamSynchronize( gpu.stream ) );
        std::cout << "  hipTextureObject_t warm-up complete\n";
        measureQueuedRenderCalls( "hipTextureObject_t tex2DLod continuously queued throughput", gpu.stream, callCount,
                                  batchSize, launchHip );
        saveGpuOutput( hipOutputPath, gpu.stream, gpu.output, outputWidth, outputHeight, channels );
    }
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

        std::cout << "VMM loader sampler benchmark\n"
                  << "  Image directory: " << imageDirectory.string() << '\n'
                  << "  Size filter: " << ( includeOver4kImages ? "all images" : "up to 4096 pixels per side" ) << '\n';

        benchmarkTex2DLod( executableDir, findLargestBenchmarkImage( imageDirectory, includeOver4kImages ) );
        std::cout << "\nResult: PASS (sampler benchmark completed)\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr << "\nResult: FAIL - " << error.what() << '\n';
        return 1;
    }
}
