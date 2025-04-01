#pragma once

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/nn/network.h"
#include "infini_train/include/ops.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
class Embedding : public Network {
public:
    Embedding(int64_t vocab_size, int64_t max_position, int64_t embed_dim);
    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;

private:
    void ToImpl(Device device) override;

    std::unique_ptr<ops::Embedding> embedding_op_ = nullptr;
};
} // namespace infini_train::nn
