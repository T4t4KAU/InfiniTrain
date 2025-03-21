#include "infini_train/include/network.h"

#include <vector>

#include "infini_train/include/op.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
void Network::AddNamedLayer(const std::string &name, std::unique_ptr<Op> &&op) {
    name_to_layers_.emplace(name, std::move(op));
}

std::unique_ptr<Op> &Network::GetLayer(const std::string &name) {
    CHECK(name_to_layers_.find(name) != name_to_layers_.end());
    return name_to_layers_.at(name);
}

std::vector<Tensor *> Network::Parameters() {
    std::vector<Tensor *> params;
    for (auto &[_, layer] : name_to_layers_) {
        for (auto &weight : layer->Weights()) {
            params.push_back(&weight);
        }
    }
    return params;
}

namespace loss {
CrossEntropyLoss::CrossEntropyLoss() {
    AddNamedLayer("cross_entropy", std::make_unique<ops::CrossEntropy>());
}

std::vector<std::shared_ptr<Tensor>> CrossEntropyLoss::Forward(const std::vector<std::shared_ptr<Tensor>> &x) {
    CHECK_EQ(x.size(), 2);
    auto x1 = GetLayer("cross_entropy")->Forward(x);
    return x1;
}
} // namespace loss
} // namespace infini_train
