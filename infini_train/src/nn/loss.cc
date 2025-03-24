#include "infini_train/include/nn/loss.h"

#include <memory>
#include <vector>

#include "infini_train/include/ops.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
CrossEntropyLoss::CrossEntropyLoss() : cross_entropy_op_(std::make_unique<ops::CrossEntropy>()) {}

std::vector<std::shared_ptr<Tensor>>
CrossEntropyLoss::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return cross_entropy_op_->Forward(input_tensors);
}
} // namespace infini_train::nn
