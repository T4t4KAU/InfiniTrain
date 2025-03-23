#include "example/mnist/net.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/op.h"
#include "infini_train/include/tensor.h"

MNIST::MNIST() {
    AddNamedLayer("linear1", std::make_unique<infini_train::ops::Linear>(784, 30));
    AddNamedLayer("sigmoid1", std::make_unique<infini_train::ops::Sigmoid>());
    AddNamedLayer("linear2", std::make_unique<infini_train::ops::Linear>(30, 10));
}

std::vector<std::shared_ptr<infini_train::Tensor>>
MNIST::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    auto x1 = GetLayer("linear1")->Forward(x);
    auto x2 = GetLayer("sigmoid1")->Forward(x1);
    auto x3 = GetLayer("linear2")->Forward(x2);
    return x3;
}
