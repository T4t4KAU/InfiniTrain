#include "infini_train/include/nn/container.h"

#include <memory>
#include <string>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train::nn {
Sequential::Sequential(std::vector<std::unique_ptr<Network>> &&layers) {
    int idx = 0;
    layers_.reserve(layers.size());
    for (auto &layer : layers) {
        layers_.push_back(AddNamedLayer(std::to_string(idx), std::move(layer)));
        ++idx;
    }
}

std::vector<std::shared_ptr<Tensor>> Sequential::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    auto &x = const_cast<std::vector<std::shared_ptr<Tensor>> &>(input_tensors);
    for (auto *layer : layers_) { x = layer->Forward(x); }
    return x;
}
} // namespace infini_train::nn
