#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/network.h"
#include "infini_train/include/tensor.h"

class MNIST : public infini_train::nn::Network {
public:
    MNIST();

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};
