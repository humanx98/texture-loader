#pragma once

#include <cstdint>

namespace hip_demand::vmm {

struct ResourceTile
{
    uint32_t textureId = 0;
    uint32_t mipLevel  = 0;
    uint32_t tileX     = 0;
    uint32_t tileY     = 0;
    uint32_t pageId    = 0;
};

struct ResourceMipTail
{
    uint32_t textureId = 0;
    uint32_t pageId    = 0;
};

enum class ResourceType
{
    TextureInfo,
    TextureTile,
    MipTail
};

struct Resource
{
    ResourceType type;
    union
    {
        struct
        {
            uint32_t textureId;
        } textureInfo;
        ResourceTile    tile;
        ResourceMipTail mipTail;
    };

    static Resource TextureInfo( uint32_t textureId )
    {
        Resource resource{};
        resource.type                  = ResourceType::TextureInfo;
        resource.textureInfo.textureId = textureId;
        return resource;
    }

    static Resource TextureTile( uint32_t pageId, uint32_t textureId, uint32_t mipLevel, uint32_t tileX, uint32_t tileY )
    {
        Resource resource{};
        resource.type           = ResourceType::TextureTile;
        resource.tile.textureId = textureId;
        resource.tile.mipLevel  = mipLevel;
        resource.tile.tileX     = tileX;
        resource.tile.tileY     = tileY;
        resource.tile.pageId    = pageId;
        return resource;
    }

    static Resource MipTail( uint32_t pageId, uint32_t textureId )
    {
        Resource resource{};
        resource.type              = ResourceType::MipTail;
        resource.mipTail.textureId = textureId;
        resource.mipTail.pageId    = pageId;
        return resource;
    }

  private:
    Resource() {}
};

}  // namespace hip_demand::vmm