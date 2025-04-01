#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/network.h"
#include "infini_train/include/tensor.h"

class GPT2 : public infini_train::nn::Network {
public:
    GPT2();

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};
