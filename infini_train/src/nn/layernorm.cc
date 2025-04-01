#include "infini_train/include/nn/layernorm.h"

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/ops.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
namespace {
constexpr char kParamWeightName[] = "weight";
constexpr char kParamBiasName[] = "bias";
} // namespace

LayerNorm::LayerNorm(int64_t embed_dim, float eps) {
    AddNamedParameter(kParamWeightName, {embed_dim}, DataType::kFLOAT32);
    AddNamedParameter(kParamBiasName, {embed_dim}, DataType::kFLOAT32);
    layernorm_op_ = std::make_unique<ops::LayerNorm>(GetParameter(kParamWeightName), GetParameter(kParamBiasName), eps);
}

std::vector<std::shared_ptr<Tensor>> LayerNorm::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return layernorm_op_->Forward(input_tensors);
}

void LayerNorm::ToImpl(Device device) {
    if (device_ == device) {
        return;
    }

    auto *w = GetParameter(kParamWeightName);
    auto *b = GetParameter(kParamBiasName);
    switch (device.Type()) {
    case DeviceType::kCPU:
        layernorm_op_ = std::make_unique<ops::LayerNorm>(w, b);
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA:
        layernorm_op_ = std::make_unique<ops::CUDALayerNorm>(w, b);
        break;
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(device.Type());
    }
}
} // namespace infini_train::nn
