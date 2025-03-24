#pragma once

#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train {
class Optimizer {
public:
    explicit Optimizer(const std::vector<Tensor *> &params);

    void ZeroGrad();

    virtual void Step() = 0;

protected:
    std::vector<Tensor *> params_;
};

namespace optimizers {
class SGD : public Optimizer {
public:
    SGD(const std::vector<Tensor *> &params, float learning_rate);

    void Step() override;

private:
    const float learning_rate_ = 0.0;
};
} // namespace optimizers
} // namespace infini_train
