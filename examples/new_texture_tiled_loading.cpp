#include <DemandLoading/VmmDemandTextureLoader.h>
#include <DemandLoading/VmmTextureSampling.h>

#include "hip_check.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "stb_image.h"
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
    }

    hipFunction_t kernel() const { return kernel_; }

  private:
    hipModule_t   module_ = nullptr;
    hipFunction_t kernel_ = nullptr;
};

std::shared_ptr<hip_demand::vmm::ImageSource> readImage( const fs::path& path )
{
    if( !fs::exists( path ) )
    {
        std::cerr << "Texture not found: " << path << '\n';
        std::abort();
    }

    int      width    = 0;
    int      height   = 0;
    int      channels = 0;
    stbi_uc* pixels   = stbi_load( path.string().c_str(), &width, &height, &channels, 4 );

    auto image = std::make_shared<hip_demand::vmm::ImageSource>();
    image->data.resize( width * height * 4 );
    image->width  = static_cast<uint32_t>( width );
    image->height = static_cast<uint32_t>( height );
    std::memcpy( image->data.data(), pixels, image->data.size() );
    stbi_image_free( pixels );
    return image;
}

template <typename T>
static void memcpyDtoH( std::vector<T>& dst, const hip_demand::DeviceSpan<T>& src )
{
    size_t bytes = count * sizeof( T );
    assert( bytes <= src.sizeInBytes() );
    assert( bytes <= sizeInBytes( dst ) );
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


    TextureDescriptor descriptor{ hip_demand::TextureFormat::RGBA8Unorm };
    descriptor.addressMode[0]                = hipAddressModeMirror;
    descriptor.addressMode[1]                = hipAddressModeMirror;
    descriptor.filterMode                    = hipFilterModeLinear;
    descriptor.mipmapFilterMode              = hipFilterModeLinear;
    descriptor.normalizedCoords              = true;
    std::shared_ptr<ImageSource> imageSource = readImage( inputPath );
    uint32_t                     width       = imageSource->width;
    uint32_t                     height      = imageSource->height;


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
    DeviceContext context{};
    loader->launchPrepare( stream, context );
    launchKernel( context );
    loader->processRequests( stream, context );

    // Second launch reads the now-resident VMM tiles into the RGBA8 output.
    loader->launchPrepare( stream, context );
    launchKernel( context );
    loader->processRequests( stream, context );

    loader->launchPrepare( stream, context );
    launchKernel( context );
    loader->processRequests( stream, context );

    loader->launchPrepare( stream, context );
    launchKernel( context );
    loader->processRequests( stream, context );

    loader->launchPrepare( stream, context );
    launchKernel( context );

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

int main( int, char** argv )
{
    try
    {
        const fs::path executableDir = fs::absolute( fs::path{ argv[0] } ).parent_path();
        test( executableDir );
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
