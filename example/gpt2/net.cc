#include "example/gpt2/net.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/nn/container.h"
#include "infini_train/include/nn/embedding.h"
#include "infini_train/include/nn/layernorm.h"
#include "infini_train/include/nn/network.h"
#include "infini_train/include/tensor.h"

namespace nn = infini_train::nn;

GPT2::GPT2() {
    AddNamedLayer("embedding", std::make_unique<nn::Embedding>(50257, 64, 768));
    AddNamedLayer("layernorm", std::make_unique<nn::LayerNorm>(768));
}

std::vector<std::shared_ptr<infini_train::Tensor>>
GPT2::Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) {
    CHECK_EQ(x.size(), 1);
    auto x1 = GetLayer("embedding")->Forward(x);
    auto x2 = GetLayer("layernorm")->Forward(x1);
    return x2;
}
