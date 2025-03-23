#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/network.h"
#include "infini_train/include/tensor.h"

class MNIST : public infini_train::Network {
public:
    MNIST();

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override;
};
