#pragma once

#include "../Internal/Utils.h"
#include <DemandLoading/DeviceContext.h>

#include <memory>
#include <mutex>
#include <vector>

namespace hip_demand::vmm {

class DemandTextureLoaderImpl;

struct InFlightContext
{
    DeviceContext deviceContext;

    HostSpan<uint32_t> requestedResources{};
    HostSpan<uint32_t> counters{};
    struct
    {
        HostSpan<EvictionCandidate> previous;
        uint32_t                    previousCount;
        HostSpan<EvictionCandidate> current;
        uint32_t                    currentCount;
    } evictionCandidates{};
};

class InFlightContextPool : NonCopyble
{
  public:
    ~InFlightContextPool();

    InFlightContext* alloc( hipStream_t stream, const DemandTextureLoaderImpl& loader );
    void             free( InFlightContext* context );
    InFlightContext* get( size_t poolIndex );

    DeviceSpan<uint32_t>           residenceBits() const { return residenceBits_; }
    DeviceSpan<DeviceTextureInfo*> textureInfos() const { return textureInfos_; }
    DeviceSpan<uint32_t>           lru() const { return lru_; }

  private:
    mutable std::mutex                            mutex_;
    std::vector<std::unique_ptr<InFlightContext>> contexts_{};
    std::vector<size_t>                           free_{};

    DeviceSpan<uint32_t>           residenceBits_{};
    DeviceSpan<DeviceTextureInfo*> textureInfos_{};
    DeviceSpan<uint32_t>           lru_{};
};

}  // namespace hip_demand::vmm
