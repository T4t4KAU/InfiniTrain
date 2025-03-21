#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "infini_train/include/op.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
class Network {
public:
    virtual ~Network(){};

    void AddNamedLayer(const std::string &name, std::unique_ptr<Op> &&layer);
    std::unique_ptr<Op> &GetLayer(const std::string &name);

    std::vector<Tensor *> Parameters();

    virtual std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) = 0;

protected:
    std::unordered_map<std::string, std::unique_ptr<Op>> name_to_layers_;
};

namespace loss {
class CrossEntropyLoss : public Network {
public:
    CrossEntropyLoss();

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) override;
};
} // namespace loss
} // namespace infini_train
