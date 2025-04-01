#include "infini_train/include/nn/embedding.h"

#include <memory>
#include <vector>

#include "infini_train/include/device.h"
#include "infini_train/include/ops.h"
#include "infini_train/include/tensor.h"

namespace infini_train::nn {
namespace {
constexpr char kParamTokenEmbeddingName[] = "token_emb";
constexpr char kParamPosEmbeddingName[] = "pos_emb";
} // namespace

Embedding::Embedding(int64_t vocab_size, int64_t max_position, int64_t embed_dim) {
    AddNamedParameter(kParamTokenEmbeddingName, {vocab_size, embed_dim}, DataType::kFLOAT32);
    AddNamedParameter(kParamPosEmbeddingName, {max_position, embed_dim}, DataType::kFLOAT32);
    embedding_op_ = std::make_unique<ops::Embedding>(GetParameter(kParamTokenEmbeddingName),
                                                     GetParameter(kParamPosEmbeddingName));
}

std::vector<std::shared_ptr<Tensor>> Embedding::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    return embedding_op_->Forward(input_tensors);
}

void Embedding::ToImpl(Device device) {
    if (device_ == device) {
        return;
    }

    auto *token_emb = GetParameter(kParamTokenEmbeddingName);
    auto *pos_emb = GetParameter(kParamPosEmbeddingName);
    switch (device.Type()) {
    case DeviceType::kCPU:
        embedding_op_ = std::make_unique<ops::Embedding>(token_emb, pos_emb);
        break;
#ifdef USE_CUDA
    case DeviceType::kCUDA:
        embedding_op_ = std::make_unique<ops::CUDAEmbedding>(token_emb, pos_emb);
        break;
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(device.Type());
    }
}
} // namespace infini_train::nn
