#include "infini_train/include/nn/network.h"

#include <random>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train::nn {
Network *Network::AddNamedLayer(const std::string &name, std::unique_ptr<Network> &&op) {
    auto &&[iter, _] = named_layers_.emplace(name, std::move(op));
    return iter->second.get();
}

std::unique_ptr<Network> &Network::GetLayer(const std::string &name) {
    CHECK(named_layers_.find(name) != named_layers_.end());
    return named_layers_.at(name);
}

void Network::AddNamedParameter(const std::string &name, const std::vector<int64_t> &dims, const DataType dtype) {
    named_parameters_.emplace(name, std::make_unique<Tensor>(dims, dtype));
    named_parameters_.at(name)->UseGradient();

    // TODO(dcj): Initialize parameters outside later.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    named_parameters_.at(name)->Fill(dis(gen));
}

std::vector<Tensor *> Network::Parameters() const {
    std::vector<Tensor *> params;
    for (auto &[_, param] : named_parameters_) {
        params.push_back(param.get());
    }
    for (auto &[_, layer] : named_layers_) {
        for (auto &param : layer->Parameters()) {
            params.push_back(param);
        }
    }
    return params;
}

Tensor *Network::GetParameter(const std::string &name) const {
    return named_parameters_.at(name).get();
}
} // namespace infini_train::nn
