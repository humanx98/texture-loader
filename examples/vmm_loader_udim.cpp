// Renders a UDIM texture assembled from the test images with the VMM demand loader.
//
// The images fill a 3x3 grid with the center cell left empty, repeating when there are fewer images than cells. The
// smallest image is the base texture, so that it lacks the resolution of the near views and the subtextures are needed.
// Three views are rendered until every pixel is final and saved as PNG: the grid without a base texture, the grid with
// a base texture, and a distant view of the grid with a base texture, where the base texture alone has enough
// resolution. The passes printed for each view show the loading steps: entry-point metadata, subtexture metadata, tiles.

#include <DemandLoading/VmmDemandTextureLoader.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vmm_loader_common.h"

#include <array>
#include <cstring>

#ifndef TEST_IMAGES_DIR
#error "TEST_IMAGES_DIR must be defined by the build target"
#endif

namespace {

using namespace vmm_loader_example;

constexpr uint32_t outputWidth    = 1536;
constexpr uint32_t outputHeight   = 1536;
constexpr uint32_t outputChannels = 4;
constexpr uint32_t blockSize      = 16;
constexpr uint32_t maxPasses      = 16;
constexpr uint32_t udim           = 3;
constexpr uint32_t vdim           = 3;
constexpr uint32_t emptyCell      = 4;  // The center of the grid.

class KernelModule
{
  public:
    explicit KernelModule( const fs::path& path )
    {
        HIP_CHECK( hipModuleLoad( &module_, path.string().c_str() ) );
        HIP_CHECK( hipModuleGetFunction( &kernel_, module_, "renderUdimTexture" ) );
    }

    ~KernelModule()
    {
        if( module_ != nullptr )
            HIP_WARN( hipModuleUnload( module_ ) );
    }

    KernelModule( const KernelModule& )            = delete;
    KernelModule& operator=( const KernelModule& ) = delete;

    hipFunction_t kernel() const { return kernel_; }

  private:
    hipModule_t   module_ = nullptr;
    hipFunction_t kernel_ = nullptr;
};

struct View
{
    const char* name;
    bool        useBaseTexture;
    float       footprintScale;
};

hip_demand::vmm::TextureDescriptor makeDescriptor()
{
    hip_demand::vmm::TextureDescriptor descriptor{};
    descriptor.addressMode[0]   = hipAddressModeClamp;
    descriptor.addressMode[1]   = hipAddressModeClamp;
    descriptor.filterMode       = hipFilterModeLinear;
    descriptor.mipmapFilterMode = hipFilterModeLinear;
    descriptor.normalizedCoords = true;
    descriptor.maxAnisotropy    = 1;
    return descriptor;
}

uint32_t countPixels( const std::vector<uint8_t>& pixels, uint8_t alpha )
{
    uint32_t count = 0;
    for( size_t i = 3; i < pixels.size(); i += outputChannels )
        count += pixels[i] == alpha ? 1 : 0;
    return count;
}

void renderView( const fs::path&              kernelPath,
                       const std::vector<fs::path>& cellImages,
                       const fs::path&              baseImage,
                       const View&                  view,
                       const fs::path&              outputPath )
{
    using namespace hip_demand::vmm;

    Options options{};
    options.maxTextures      = 64;
    options.maxVirtualPages  = 64 * 1024;
    options.maxPhysicalPages = 64 * 1024;
    options.maxRequests      = 8192;
    options.enableEviction   = false;

    KernelModule module( kernelPath );
    GpuResources gpu( static_cast<size_t>( outputWidth ) * outputHeight * outputChannels );
    auto         loader = createDemandTextureLoader( options );

    uint32_t baseTextureId = INVALID_TEXTURE;
    if( view.useBaseTexture )
        baseTextureId = loader->createTexture( readImage( baseImage ), makeDescriptor() ).getId();

    std::vector<std::shared_ptr<hip_demand::ImageSource>> images( udim * vdim );
    std::vector<TextureDescriptor>                        descriptors( udim * vdim, makeDescriptor() );
    size_t                                                nextImage = 0;
    for( uint32_t cell = 0; cell < udim * vdim; ++cell )
    {
        if( cell != emptyCell )
            images[cell] = readImage( cellImages[nextImage++ % cellImages.size()] );
    }

    uint32_t textureId   = loader->createUdimTexture( images, descriptors, udim, vdim, baseTextureId ).getId();
    uint32_t renderWidth = outputWidth, renderHeight = outputHeight, gridU = udim, gridV = vdim;
    float    footprintScale = view.footprintScale;
    const uint32_t gridWidth  = ( outputWidth + blockSize - 1 ) / blockSize;
    const uint32_t gridHeight = ( outputHeight + blockSize - 1 ) / blockSize;

    std::vector<uint8_t> pixels( static_cast<size_t>( outputWidth ) * outputHeight * outputChannels );
    const auto           readPixels = [&] {
        HIP_CHECK( hipMemcpyAsync( pixels.data(), gpu.output, pixels.size(), hipMemcpyDeviceToHost, gpu.stream ) );
        HIP_CHECK( hipStreamSynchronize( gpu.stream ) );
    };

    for( uint32_t pass = 1; pass <= maxPasses; ++pass )
    {
        DeviceContext context{};
        loader->launchPrepare( gpu.stream, context );
        void* arguments[] = { &context, &gpu.output, &renderWidth, &renderHeight, &textureId, &gridU, &gridV, &footprintScale };
        HIP_CHECK( hipModuleLaunchKernel( module.kernel(), gridWidth, gridHeight, 1, blockSize, blockSize, 1, 0,
                                          gpu.stream, arguments, nullptr ) );

        Ticket ticket = loader->processRequests( gpu.stream, context );
        ticket.wait();
        readPixels();

        // Pixels whose sample was not final after this pass.
        const uint32_t provisionalPixels = countPixels( pixels, 0 );
        std::cout << "  Pass " << pass << ": " << ticket.numTasksTotal() << " requests, " << provisionalPixels
                  << " provisional pixels\n";
        if( ticket.numTasksTotal() == 0 && provisionalPixels == 0 )
            break;
    }

    // Alpha holds residency; write the image opaque.
    for( size_t i = 3; i < pixels.size(); i += outputChannels )
        pixels[i] = 255;
    if( stbi_write_png( outputPath.string().c_str(), static_cast<int>( outputWidth ), static_cast<int>( outputHeight ),
                        static_cast<int>( outputChannels ), pixels.data(), static_cast<int>( outputWidth * outputChannels ) )
        == 0 )
        throw std::runtime_error( "Failed to save output PNG: " + outputPath.string() );
    std::cout << "  Image saved: " << fs::absolute( outputPath ).string() << '\n';
}

}  // namespace

int main( int argc, char** argv )
{
    try
    {
        const fs::path executableDir = fs::absolute( fs::path{ argv[0] } ).parent_path();
        const fs::path kernelPath    = executableDir / "vmm_loader_udim_kernel.co";
        if( !fs::exists( kernelPath ) )
            throw std::runtime_error( "HIP module not found: " + kernelPath.string() );

        fs::path imageDirectory = fs::path{ TEST_IMAGES_DIR } / "png";
        imageDirectory.make_preferred();

        std::cout << "VMM loader UDIM texture\n  Image directory: " << imageDirectory.string() << '\n';
        std::vector<fs::path> imagePaths = findImages( imageDirectory, false );
        if( imagePaths.size() < 2 )
            throw std::runtime_error( "The UDIM example needs at least two images" );

        // The smallest image stands in for the whole grid as the base texture; the others fill the cells.
        const auto pixelCount = []( const fs::path& path ) {
            const hip_demand::TextureInfo& info = readImage( path )->getInfo();
            return static_cast<size_t>( info.width ) * info.height;
        };
        const auto smallest = std::min_element( imagePaths.begin(), imagePaths.end(), [&]( const fs::path& a, const fs::path& b ) {
            return pixelCount( a ) < pixelCount( b );
        } );
        const fs::path baseImage = *smallest;
        imagePaths.erase( smallest );
        std::cout << "  Grid: " << udim << 'x' << vdim << ", cell " << emptyCell << " empty\n  Base image: "
                  << baseImage.filename().string() << '\n';
        for( size_t i = 0; i < imagePaths.size(); ++i )
            std::cout << "  Cell image " << i << ": " << imagePaths[i].filename().string() << '\n';

        const std::array<View, 3> views{ { { "no_base", false, 1.0f }, { "base", true, 1.0f }, { "base_far", true, 64.0f } } };
        for( const View& view : views )
        {
            std::cout << "\nView " << view.name << " (base texture " << ( view.useBaseTexture ? "on" : "off" )
                      << ", footprint x" << view.footprintScale << ")\n";
            const fs::path outputPath = fs::path( "vmm_loader_udim_" ) += fs::path( view.name ).concat( ".png" );
            renderView( kernelPath, imagePaths, baseImage, view, outputPath );
        }

        std::cout << "\nResult: PASS\n";
        return 0;
    }
    catch( const std::exception& error )
    {
        std::cerr << "\nResult: FAIL - " << error.what() << '\n';
        return 1;
    }
}
