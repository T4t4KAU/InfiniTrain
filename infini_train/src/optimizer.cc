#include "infini_train/include/optimizer.h"

#include <cstddef>
#include <vector>

#include "infini_train/include/tensor.h"

namespace infini_train {
Optimizer::Optimizer(const std::vector<Tensor *> &params) : params_(params) {}

void Optimizer::ZeroGrad() {
    for (auto *param : params_) { param->ZeroGrad(); }
}

namespace optimizers {
SGD::SGD(const std::vector<Tensor *> &params, float learning_rate) : Optimizer(params), learning_rate_(learning_rate) {}

void SGD::Step() {
    for (auto *param : params_) {
        for (size_t i = 0; i < param->NumElements(); ++i) {
            reinterpret_cast<float *>(param->DataPtr())[i]
                -= learning_rate_ * reinterpret_cast<const float *>(param->Gradient()->DataPtr())[i];
        }
    }
}
} // namespace optimizers
} // namespace infini_train
