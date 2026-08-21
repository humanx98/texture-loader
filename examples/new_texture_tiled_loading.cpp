#include <DemandLoading/VmmDemandTextureLoader.h>
#include <DemandLoading/VmmTextureSampling.h>
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
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#ifndef TEST_IMAGES_DIR
#error "TEST_IMAGES_DIR must be defined by the build target"
#endif

namespace fs = std::filesystem;

class KernelModule
{
  public:
    ~KernelModule()
    {
        if( module_ != nullptr )
            HIP_WARN( hipModuleUnload( module_ ) );
    }

    void load( const std::filesystem::path& path )
    {
        HIP_CHECK( hipModuleLoad( &module_, path.string().c_str() ) );
        HIP_CHECK( hipModuleGetFunction( &kernel_, module_, "renderVmmTexture" ) );
        HIP_CHECK( hipModuleGetFunction( &gridKernel_, module_, "renderVmmTextureGrid" ) );
        HIP_CHECK( hipModuleGetFunction( &mipGridKernel_, module_, "renderVmmTextureMipGrid" ) );
    }

    hipFunction_t kernel() const { return kernel_; }
    hipFunction_t gridKernel() const { return gridKernel_; }
    hipFunction_t mipGridKernel() const { return mipGridKernel_; }

  private:
    hipModule_t   module_        = nullptr;
    hipFunction_t kernel_        = nullptr;
    hipFunction_t gridKernel_    = nullptr;
    hipFunction_t mipGridKernel_ = nullptr;
};

std::shared_ptr<hip_demand::ImageSource> readImage( const fs::path& path )
{
#if !defined( USE_OIIO )
#error "This example requires a build configured with USE_OIIO=ON"
#endif

    if( !fs::exists( path ) )
        throw std::runtime_error( "Texture not found: " + path.string() );

    std::unique_ptr<hip_demand::ImageSource> image =
        std::make_unique<hip_demand::OIIOReader>( path.string(), HIP_AD_FORMAT_UNSIGNED_INT8 );
    if( !image )
        throw std::runtime_error( "Failed to create an image source for: " + path.string() );

    hip_demand::TextureInfo info{};
    image->open( &info );
    if( !info.isValid || info.width == 0 || info.height == 0 )
        throw std::runtime_error( "Invalid image: " + path.string() );

    return std::shared_ptr<hip_demand::ImageSource>( std::move( image ) );
}

template <typename T>
static void memcpyDtoH( std::vector<T>& dst, const hip_demand::DeviceSpan<T>& src )
{
    size_t bytes = count * sizeof( T );
    assert( bytes <= src.sizeInBytes() );
    assert( bytes <= dst.size() * sizeof( T ) );
    HIP_CHECK( hipMemcpy( dst.data(), src.ptr, bytes, hipMemcpyDeviceToHost ) );
}

void test( const fs::path& executableDir )
{
    using namespace hip_demand::vmm;

    const fs::path inputPath = fs::path{ TEST_IMAGES_DIR } / "png/hypno-cat.png";
    const fs::path outputPath{ "new_texture_tiled_loading_output.png" };
    const fs::path kernelPath = executableDir / "new_texture_tiled_loading_kernel.co";

    if( !fs::exists( inputPath ) )
    {
        std::cerr << "Texture not found: " << inputPath << '\n';
        return;
    }

    if( !fs::exists( kernelPath ) )
    {
        std::cerr << "Hip Module not found: " << kernelPath << '\n';
        return;
    }

    HIP_CHECK( hipSetDevice( 0 ) );

    KernelModule module;
    module.load( kernelPath );

    Options options{};
    options.maxRequests = 100u;

    std::unique_ptr<DemandTextureLoader> loader = createDemandTextureLoader( options );


    TextureDescriptor descriptor{};
    descriptor.addressMode[0]                            = hipAddressModeMirror;
    descriptor.addressMode[1]                            = hipAddressModeMirror;
    descriptor.filterMode                                = hipFilterModeLinear;
    descriptor.mipmapFilterMode                          = hipFilterModeLinear;
    descriptor.normalizedCoords                          = true;
    std::shared_ptr<hip_demand::ImageSource> imageSource = readImage( inputPath );
    assert( imageSource->isOpen() );
    const hip_demand::TextureInfo& imageInfo = imageSource->getInfo();
    uint32_t                       width     = imageInfo.width;
    uint32_t                       height    = imageInfo.height;


    const DemandTexture& texture   = loader->createTexture( imageSource, descriptor );
    uint32_t             textureId = texture.getId();
    const size_t         byteCount = width * height * 4;

    hipStream_t stream = nullptr;
    HIP_CHECK( hipStreamCreate( &stream ) );

    std::vector<uint8_t> hostOutput( byteCount );
    uint8_t*             deviceOutput = nullptr;
    HIP_CHECK( hipMalloc( reinterpret_cast<void**>( &deviceOutput ), byteCount ) );
    HIP_CHECK( hipMemsetAsync( deviceOutput, 0, byteCount, stream ) );

    constexpr uint32_t blockWidth  = 16;
    constexpr uint32_t blockHeight = 16;
    const uint32_t     gridWidth   = ( width + blockWidth - 1 ) / blockWidth;
    const uint32_t     gridHeight  = ( height + blockHeight - 1 ) / blockHeight;

    const auto launchKernel = [&]( const DeviceContext& context ) {
        DeviceContext mutableContext = context;
        void*         arguments[]    = {
            &mutableContext, &textureId, &deviceOutput, &width, &height,
        };

        HIP_CHECK( hipModuleLaunchKernel( module.kernel(), gridWidth, gridHeight, 1, blockWidth, blockHeight, 1, 0,
                                          stream, arguments, nullptr ) );
    };

    // First launch requests every missing mip-0 VMM tile.
    while( true )
    {
        DeviceContext context{};
        loader->launchPrepare( stream, context );
        launchKernel( context );
        Ticket ticket = loader->processRequests( stream, context );

        ticket.wait();
        if( ticket.numTasksTotal() == 0 )
            break;
    }

    HIP_CHECK( hipMemcpyAsync( hostOutput.data(), deviceOutput, byteCount, hipMemcpyDeviceToHost, stream ) );
    HIP_CHECK( hipStreamSynchronize( stream ) );
    //{
    //    std::vector<uint8_t>  pageMemory{};
    //    std::vector<uint32_t> requestedBits{};
    //    std::vector<uint32_t> residentBits{};
    //    std::vector<uint32_t> requestedResources{};
    //    std::vector<uint32_t> counters{};

    //    residentBits.resize( context.residentBits.len );
    //    requestedBits.resize( context.requestedBits.len );
    //    requestedResources.resize( context.requestedResources.len );
    //    counters.resize( context.counters.len );

    //    memcpyDtoH( residentBits, context.requestedBits );
    //    memcpyDtoH( requestedBits, context.requestedBits );
    //    memcpyDtoH( requestedResources, context.requestedResources );
    //    memcpyDtoH( counters, context.counters );

    //    uint32_t pageCount = ; // 510
    //    pageMemory.resize( pageCount * context.pageSize );
    //    HIP_CHECK( hipMemcpy( pageMemory.data(), context.pageMemory.ptr, pageMemory.size(), hipMemcpyDeviceToHost ) );

    //    DeviceContext hostContext{};
    //    hostContext.pageMemory    = hip_demand::DeviceSpan<uint8_t>( pageMemory.data(), pageMemory.size() );
    //    hostContext.requestedBits = hip_demand::DeviceSpan<uint32_t>( requestedBits.data(), requestedBits.size() );
    //    hostContext.residentBits  = hip_demand::DeviceSpan<uint32_t>( residentBits.data(), residentBits.size() );
    //    hostContext.requestedResources =
    //        hip_demand::DeviceSpan<uint32_t>( requestedResources.data(), requestedResources.size() );
    //    hostContext.counters     = hip_demand::DeviceSpan<uint32_t>( counters.data(), counters.size() );
    //    hostContext.pageSize     = context.pageSize;
    //    hostContext.textureCount = context.textureCount;
    //    hostContext.maxTextures  = context.maxTextures;

    //    for( int y = 0; y < height; y++ )
    //    {
    //        for( int x = 0; x < width; x++ )
    //        {
    //            auto toUnorm8 = []( float value ) {
    //                value = fminf( 1.0f, fmaxf( 0.0f, value ) );
    //                return static_cast<uint8_t>( value * 255.0f + 0.5f );
    //            };

    //            const size_t outputOffset = ( static_cast<size_t>( y ) * width + x ) * 4;
    //            bool         resident     = false;
    //            const float4 color = fetchTexel<float4>( hostContext, hostContext.textureInfos.ptr[textureId], 0,
    //                                                   static_cast<int>( x ), static_cast<int>( y ), resident );
    //            const float mipLevel = 0;
    //            const float4 color = hip_demand::vmm::tex2DLod<float4>( hostContext, textureId, x, y, mipLevel, resident );
    //            if( resident )
    //            {
    //                hostOutput[outputOffset + 0] = toUnorm8( color.x );
    //                hostOutput[outputOffset + 1] = toUnorm8( color.y );
    //                hostOutput[outputOffset + 2] = toUnorm8( color.z );
    //                hostOutput[outputOffset + 3] = toUnorm8( 1.0f );
    //            }
    //            else
    //            {
    //                hostOutput[outputOffset + 0] = toUnorm8( 1.0f );
    //                hostOutput[outputOffset + 1] = toUnorm8( 0.0f );
    //                hostOutput[outputOffset + 2] = toUnorm8( 1.0f );
    //                hostOutput[outputOffset + 3] = toUnorm8( 1.0f );
    //            }
    //        }
    //    }
    //}

    if( stbi_write_png( outputPath.string().c_str(), width, height, 4, hostOutput.data(), ( width * 4 ) ) == 0 )
        throw std::runtime_error( "Failed to save output PNG" );

    HIP_WARN( hipFree( deviceOutput ) );
    HIP_WARN( hipStreamDestroy( stream ) );

    std::cout << "Saved: " << fs::absolute( outputPath ) << '\n';
}

void renderGrid( const fs::path& executableDir, const fs::path& outputPath, bool renderMipmaps )
{
    using namespace hip_demand::vmm;

    uint32_t           outputWidth  = 3840;
    uint32_t           outputHeight = 2160;
    constexpr uint32_t blockWidth   = 16;
    constexpr uint32_t blockHeight  = 16;

    const fs::path inputDirectory = fs::path{ TEST_IMAGES_DIR } / "png";
    const fs::path kernelPath     = executableDir / "new_texture_tiled_loading_kernel.co";

    if( !fs::is_directory( inputDirectory ) )
        throw std::runtime_error( "Image directory not found: " + inputDirectory.string() );
    if( !fs::exists( kernelPath ) )
        throw std::runtime_error( "HIP module not found: " + kernelPath.string() );

    std::vector<fs::path> imagePaths;
    for( const fs::directory_entry& entry : fs::directory_iterator( inputDirectory ) )
    {
        if( !entry.is_regular_file() )
            continue;

        std::string extension = entry.path().extension().string();
        std::transform( extension.begin(), extension.end(), extension.begin(),
                        []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );
        if( extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp"
            || extension == ".tga" )
        {
            imagePaths.push_back( entry.path() );
        }
    }
    std::sort( imagePaths.begin(), imagePaths.end() );

    if( imagePaths.empty() )
        throw std::runtime_error( "No supported images found in: " + inputDirectory.string() );

    HIP_CHECK( hipSetDevice( 0 ) );

    KernelModule module;
    module.load( kernelPath );

    constexpr uint32_t IN_FLIGHT_COUNT = 5;

    Options options{};
    options.maxPhysicalPages = 4096;
    options.maxRequests      = 100;
    options.maxRequestQueue  = options.maxRequests * IN_FLIGHT_COUNT;

    std::unique_ptr<DemandTextureLoader> loader = createDemandTextureLoader( options );

    TextureDescriptor descriptor{};
    descriptor.addressMode[0]   = hipAddressModeClamp;
    descriptor.addressMode[1]   = hipAddressModeClamp;
    descriptor.filterMode       = hipFilterModeLinear;
    descriptor.mipmapFilterMode = renderMipmaps ? hipFilterModePoint : hipFilterModeLinear;
    descriptor.normalizedCoords = true;

    for( const fs::path& imagePath : imagePaths )
    {
        std::shared_ptr<hip_demand::ImageSource> image = readImage( imagePath );
        assert( image->isOpen() );
        loader->createTexture( image, descriptor );
        const hip_demand::TextureInfo& imageInfo = image->getInfo();
        std::cout << "Loaded: " << imagePath.filename() << " (" << imageInfo.width << 'x' << imageInfo.height << ")\n";
    }

    uint32_t textureCount = static_cast<uint32_t>( imagePaths.size() );
    uint32_t columnCount  = static_cast<uint32_t>( std::ceil( std::sqrt( static_cast<double>( textureCount ) ) ) );
    uint32_t rowCount     = ( textureCount + columnCount - 1 ) / columnCount;

    const size_t         byteCount = static_cast<size_t>( outputWidth ) * outputHeight * 4;
    std::vector<uint8_t> hostOutput( byteCount, 0 );
    for( size_t offset = 3; offset < hostOutput.size(); offset += 4 )
        hostOutput[offset] = 255;

    hipStream_t stream = nullptr;
    HIP_CHECK( hipStreamCreate( &stream ) );

    uint8_t* deviceOutput = nullptr;
    HIP_CHECK( hipMalloc( reinterpret_cast<void**>( &deviceOutput ), byteCount ) );
    HIP_CHECK( hipMemcpyAsync( deviceOutput, hostOutput.data(), byteCount, hipMemcpyHostToDevice, stream ) );

    const uint32_t gridWidth  = ( outputWidth + blockWidth - 1 ) / blockWidth;
    const uint32_t gridHeight = ( outputHeight + blockHeight - 1 ) / blockHeight;
    const auto     launchGrid = [&]( const DeviceContext& context ) {
        DeviceContext mutableContext = context;
        void*         arguments[]    = { &mutableContext, &deviceOutput, &outputWidth, &outputHeight,
                                         &textureCount,   &columnCount,  &rowCount };

        const hipFunction_t kernel = renderMipmaps ? module.mipGridKernel() : module.gridKernel();
        HIP_CHECK( hipModuleLaunchKernel( kernel, gridWidth, gridHeight, 1, blockWidth, blockHeight, 1, 0, stream, arguments, nullptr ) );
    };

    struct InFlightFrame
    {
        uint32_t pass = 0;
        Ticket   ticket;
        bool     active = false;
    };

    std::array<InFlightFrame, IN_FLIGHT_COUNT> inFlightFrames{};
    uint32_t                                   nextPass          = 0;
    bool                                       submitMoreFrames  = true;
    const auto renderStart = std::chrono::steady_clock::now();
    while (true)
    {
        uint32_t activeFrameCount = IN_FLIGHT_COUNT;
        for (auto& frame : inFlightFrames)
        {
            if( frame.active )
            {
                frame.ticket.wait();
                const int requestCount = frame.ticket.numTasksTotal();
                frame.active           = false;

                if (requestCount > 0)
                {
                    std::cout << "Pass " << frame.pass << ": " << requestCount << " resource requests\n";
                }
                else
                {
                    activeFrameCount--;
                }                 
            }

            DeviceContext context{};
            loader->launchPrepare( stream, context );
            launchGrid( context );

            frame.pass   = nextPass++;
            frame.ticket = loader->processRequests( stream, context );
            frame.active = true;
        }

        if( activeFrameCount == 0 )
            break;
    }
    const auto renderEnd = std::chrono::steady_clock::now();
    const auto renderTime = std::chrono::duration_cast<std::chrono::milliseconds>( renderEnd - renderStart );
    std::cout << "Rendering loop time: " << renderTime.count() << " ms\n";

    HIP_CHECK( hipMemcpyAsync( hostOutput.data(), deviceOutput, byteCount, hipMemcpyDeviceToHost, stream ) );
    HIP_CHECK( hipStreamSynchronize( stream ) );

    if( stbi_write_png( outputPath.string().c_str(), static_cast<int>( outputWidth ), static_cast<int>( outputHeight ),
                        4, hostOutput.data(), static_cast<int>( outputWidth * 4 ) )
        == 0 )
    {
        throw std::runtime_error( "Failed to save output PNG" );
    }

    HIP_WARN( hipFree( deviceOutput ) );
    HIP_WARN( hipStreamDestroy( stream ) );

    std::cout << "Saved 4K " << ( renderMipmaps ? "texture + mipmap" : "texture" )
              << " grid: " << fs::absolute( outputPath ) << '\n';
}

void test2( const fs::path& executableDir )
{
    renderGrid( executableDir, "new_texture_tiled_loading_test2_output.png", false );
}

void test3( const fs::path& executableDir )
{
    renderGrid( executableDir, "new_texture_tiled_loading_test3_output.png", true );
}

int main( int argc, char** argv )
{
    try
    {
        const fs::path executableDir = fs::absolute( fs::path{ argv[0] } ).parent_path();
        //test( executableDir );
        //test2( executableDir );
        test3( executableDir );
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
