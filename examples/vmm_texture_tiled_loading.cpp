#include <DemandLoading/VmmDemandTextureLoader.h>
#include <ImageSource/OIIOReader.h>
#include <ImageSource/TextureInfo.h>

#include "hip_check.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
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
        HIP_CHECK( hipModuleGetFunction( &gridKernel_, module_, "renderVmmTextureGrid" ) );
        HIP_CHECK( hipModuleGetFunction( &regionKernel_, module_, "renderVmmTextureRegion" ) );
    }

    ~KernelModule()
    {
        if( module_ != nullptr )
            HIP_WARN( hipModuleUnload( module_ ) );
    }

    hipFunction_t gridKernel() const { return gridKernel_; }
    hipFunction_t regionKernel() const { return regionKernel_; }

    KernelModule( const KernelModule& )            = delete;
    KernelModule& operator=( const KernelModule& ) = delete;

  private:
    hipModule_t   module_       = nullptr;
    hipFunction_t gridKernel_   = nullptr;
    hipFunction_t regionKernel_ = nullptr;
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
                std::cout << "  Skipped (>4K): " << path.filename().string() << " (" << info.width << 'x'
                          << info.height << ")\n";
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

void renderTextureGrid( const fs::path& executableDir, const std::vector<fs::path>& imagePaths )
{
    using namespace hip_demand::vmm;

    uint32_t           outputWidth        = 3840;
    uint32_t           outputHeight       = 2160;
    constexpr uint32_t blockWidth         = 16;
    constexpr uint32_t blockHeight        = 16;
    constexpr uint32_t channels           = 4;
    constexpr uint32_t maxVirtualPages    = 64 * 1024;
    constexpr uint32_t maxPhysicalPages   = 64 * 1024;
    constexpr uint32_t maxRequestsPerPass = 1024;
    constexpr double   bytesPerMiB        = 1024.0 * 1024.0;

    const fs::path outputPath{ "vmm_texture_tiled_loading_renderTextureGrid.png" };
    const fs::path kernelPath = executableDir / "vmm_texture_tiled_loading_kernel.co";
    if( !fs::exists( kernelPath ) )
        throw std::runtime_error( "HIP module not found: " + kernelPath.string() );

    std::cout << "\n[1/2] Texture grid (eviction disabled)\n"
              << "  Output size: " << outputWidth << 'x' << outputHeight << " pixels\n"
              << "  Physical page limit: " << maxPhysicalPages << '\n';

    HIP_CHECK( hipSetDevice( 0 ) );
    // MemPulse resets the HIP device on shutdown, so keep it alive until the render resources are destroyed.
    GpuMemoryMonitor memoryMonitor( 0 );
    KernelModule     module( kernelPath );

    Options options{};
    options.maxVirtualPages  = maxVirtualPages;
    options.maxPhysicalPages = maxPhysicalPages;
    options.maxRequests      = maxRequestsPerPass;

    auto loader = createLoaderForImages( options, imagePaths );

    uint32_t textureCount = static_cast<uint32_t>( imagePaths.size() );
    uint32_t columnCount  = static_cast<uint32_t>( std::ceil( std::sqrt( static_cast<double>( textureCount ) ) ) );
    uint32_t rowCount     = ( textureCount + columnCount - 1 ) / columnCount;

    const size_t         byteCount = static_cast<size_t>( outputWidth ) * outputHeight * channels;
    std::vector<uint8_t> hostOutput( byteCount );
    GpuResources         gpu( byteCount );

    const uint32_t gridWidth  = ( outputWidth + blockWidth - 1 ) / blockWidth;
    const uint32_t gridHeight = ( outputHeight + blockHeight - 1 ) / blockHeight;
    const auto     launchGrid = [&]( const DeviceContext& context ) {
        DeviceContext mutableContext = context;
        void*         arguments[]    = { &mutableContext, &gpu.output,  &outputWidth, &outputHeight,
                                         &textureCount,   &columnCount, &rowCount };

        HIP_CHECK( hipModuleLaunchKernel( module.gridKernel(), gridWidth, gridHeight, 1, blockWidth, blockHeight, 1, 0,
                                          gpu.stream, arguments, nullptr ) );
    };

    uint32_t   nextPass    = 0;
    const auto renderStart = std::chrono::steady_clock::now();
    // The first pass requests texture metadata; later passes request missing mip pages.
    while( true )
    {
        DeviceContext context{};
        loader->launchPrepare( gpu.stream, context );
        launchGrid( context );

        Ticket ticket = loader->processRequests( gpu.stream, context );

        ticket.wait();
        const int requestCount = ticket.numTasksTotal();
        HIP_CHECK( hipStreamSynchronize( gpu.stream ) );
        const MempulseDeviceMemoryInfo memory = memoryMonitor.memoryInfo();
        std::cout << "  Pass " << nextPass + 1 << " | requests: " << requestCount << " | GPU VRAM: "
                  << memory.dedicatedUsed / bytesPerMiB << " / " << memory.dedicatedTotal / bytesPerMiB << " MiB";
        if( requestCount == 0 )
            std::cout << " | complete";
        std::cout << '\n';
        if( requestCount == 0 )
            break;
        ++nextPass;
    }
    const auto renderEnd  = std::chrono::steady_clock::now();
    const auto renderTime = std::chrono::duration_cast<std::chrono::milliseconds>( renderEnd - renderStart );
    std::cout << "  Render complete: " << nextPass + 1 << " passes in " << renderTime.count() << " ms\n";

    // Ticket::wait() finishes CPU request processing. Synchronize before reading GPU output.
    HIP_CHECK( hipMemcpyAsync( hostOutput.data(), gpu.output, byteCount, hipMemcpyDeviceToHost, gpu.stream ) );
    HIP_CHECK( hipStreamSynchronize( gpu.stream ) );

    if( stbi_write_png( outputPath.string().c_str(), static_cast<int>( outputWidth ), static_cast<int>( outputHeight ),
                        channels, hostOutput.data(), static_cast<int>( outputWidth * channels ) )
        == 0 )
    {
        throw std::runtime_error( "Failed to save output PNG" );
    }

    std::cout << "  Image saved: " << fs::absolute( outputPath ).string() << '\n';
}

void renderTextureGridWithEviction( const fs::path& executableDir, const std::vector<fs::path>& imagePaths )
{
    using namespace hip_demand::vmm;

    uint32_t           outputWidth        = 3840;
    uint32_t           outputHeight       = 2160;
    constexpr uint32_t blockWidth         = 16;
    constexpr uint32_t blockHeight        = 16;
    constexpr uint32_t channels           = 4;
    constexpr uint32_t outputTileSize     = 512;
    constexpr uint32_t maxVirtualPages    = 64 * 1024;
    constexpr uint32_t maxPhysicalPages   = 64 * 1024;
    constexpr uint32_t maxEvictedPages    = 64 * 1024;
    constexpr uint32_t maxRequestsPerPass = 1024;
    constexpr uint32_t maxPassesPerTile   = 64;
    constexpr double   bytesPerMiB        = 1024.0 * 1024.0;

    const fs::path outputPath{ "vmm_texture_tiled_loading_eviction.png" };
    const fs::path kernelPath = executableDir / "vmm_texture_tiled_loading_kernel.co";
    if( !fs::exists( kernelPath ) )
        throw std::runtime_error( "HIP module not found: " + kernelPath.string() );

    std::cout << "\n[2/2] Tiled render (eviction enabled)\n"
              << "  Output size: " << outputWidth << 'x' << outputHeight << " pixels\n"
              << "  Output tile size: " << outputTileSize << 'x' << outputTileSize << " pixels\n"
              << "  Physical page limit: " << maxPhysicalPages << '\n';

    HIP_CHECK( hipSetDevice( 0 ) );
    // MemPulse resets the HIP device on shutdown, so keep it alive until the render resources are destroyed.
    GpuMemoryMonitor memoryMonitor( 0 );
    KernelModule     module( kernelPath );

    Options options{};
    options.maxTextures      = static_cast<uint32_t>( imagePaths.size() );
    options.maxVirtualPages  = maxVirtualPages;
    options.maxPhysicalPages = maxPhysicalPages;
    options.maxRequests      = maxRequestsPerPass;
    options.maxEvictedPages  = maxEvictedPages;
    options.enableEviction   = true;
    auto loader              = createLoaderForImages( options, imagePaths );

    uint32_t textureCount = static_cast<uint32_t>( imagePaths.size() );
    uint32_t columnCount  = static_cast<uint32_t>( std::ceil( std::sqrt( static_cast<double>( textureCount ) ) ) );
    uint32_t rowCount     = ( textureCount + columnCount - 1 ) / columnCount;

    const size_t         byteCount = static_cast<size_t>( outputWidth ) * outputHeight * channels;
    std::vector<uint8_t> hostOutput( byteCount );
    GpuResources         gpu( byteCount );

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
            uint32_t       regionWidth  = std::min( outputTileSize, outputWidth - regionX );
            uint32_t       regionHeight = std::min( outputTileSize, outputHeight - regionY );
            const uint32_t gridWidth    = ( regionWidth + blockWidth - 1 ) / blockWidth;
            const uint32_t gridHeight   = ( regionHeight + blockHeight - 1 ) / blockHeight;
            bool           complete     = false;
            uint32_t       passesForTile = 0;
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

                if( currentResidence.empty() )
                {
                    currentResidence.resize( context.residenceBits.len );
                    previousResidence.resize( context.residenceBits.len, 0 );
                }

                HIP_CHECK( hipMemcpyAsync( currentResidence.data(), context.residenceBits.ptr,
                                           context.residenceBits.sizeInBytes(), hipMemcpyDeviceToHost, gpu.stream ) );
                HIP_CHECK( hipStreamSynchronize( gpu.stream ) );

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

            const MempulseDeviceMemoryInfo memory = memoryMonitor.memoryInfo();
            std::cout << "  Tile " << completedTiles << '/' << tilesPerRow * tileRows << " at (" << regionX << ", "
                      << regionY << ") | passes: " << passesForTile << " | evictions: +"
                      << evictedPages - evictionsBeforeTile << " (total " << evictedPages << ") | GPU VRAM: "
                      << memory.dedicatedUsed / bytesPerMiB << " / " << memory.dedicatedTotal / bytesPerMiB
                      << " MiB\n";
        }
    }

    const auto renderEnd  = std::chrono::steady_clock::now();
    const auto renderTime = std::chrono::duration_cast<std::chrono::milliseconds>( renderEnd - renderStart );
    std::cout << "  Render complete: " << completedTiles << '/' << tilesPerRow * tileRows << " tiles in "
              << renderTime.count() << " ms; VMM page evictions: " << evictedPages << '\n';

    if( evictedPages == 0 )
        throw std::runtime_error( "Eviction test loaded every output tile without evicting a VMM page" );

    HIP_CHECK( hipMemcpyAsync( hostOutput.data(), gpu.output, byteCount, hipMemcpyDeviceToHost, gpu.stream ) );
    HIP_CHECK( hipStreamSynchronize( gpu.stream ) );
    if( stbi_write_png( outputPath.string().c_str(), static_cast<int>( outputWidth ), static_cast<int>( outputHeight ),
                        channels, hostOutput.data(), static_cast<int>( outputWidth * channels ) )
        == 0 )
        throw std::runtime_error( "Failed to save output PNG" );

    std::cout << "  Image saved: " << fs::absolute( outputPath ).string() << '\n';
}

}  // namespace

int main( int argc, char** argv )
{
    try
    {
        constexpr bool includeOver4kImages = true;
        const fs::path executableDir       = fs::absolute( fs::path{ argv[0] } ).parent_path();
        fs::path imageDirectory            = fs::path{ TEST_IMAGES_DIR } / "png";
        imageDirectory.make_preferred();

        std::cout << "VMM texture loading tests\n"
                  << "  Image directory: " << imageDirectory.string() << '\n'
                  << "  Size filter: " << ( includeOver4kImages ? "all images" : "up to 4096 pixels per side" )
                  << '\n';
        const auto imagePaths = findImages( imageDirectory, includeOver4kImages );
        std::cout << "  Selected textures (" << imagePaths.size() << "):\n";
        for( const fs::path& imagePath : imagePaths )
        {
            const auto image = readImage( imagePath );
            const auto& info = image->getInfo();
            std::cout << "    " << imagePath.filename().string() << " (" << info.width << 'x' << info.height
                      << ")\n";
        }
        std::cout << "  GPU VRAM readings cover the whole device, including other processes.\n";

        renderTextureGrid( executableDir, imagePaths );
        renderTextureGridWithEviction( executableDir, imagePaths );
        std::cout << "\nResult: PASS (both renders completed and eviction was observed)\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr << "\nResult: FAIL - " << error.what() << '\n';
        return 1;
    }
}
