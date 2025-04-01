#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/network.h"
#include "infini_train/include/ops.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class LayerNorm : public Network {
public:
    LayerNorm(int64_t embed_dim, float eps = 1e-5f);
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ToImpl(Device device) override;

    std::unique_ptr<ops::LayerNorm> layernorm_op_ = nullptr;
};
} // namespace infini_train::nn
