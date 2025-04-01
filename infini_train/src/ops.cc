#include "infini_train/include/ops.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train::ops {
std::vector<std::shared_ptr<Tensor>> Op::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    input_tensors_ = input_tensors;
    auto output_tensors = ForwardImpl();
    for (auto &tensor : output_tensors) {
        tensor->UseGradient();
        tensor->SetProducer(this);
    }
    output_tensors_ = output_tensors;
    return output_tensors;
}

void Op::Backward(const Tensor *output_tensor) {
    backward_reached_ += 1;
    if (backward_reached_ != output_tensors_.size()) {
        return;
    }
    BackwardImpl();
    for (const auto &tensor : input_tensors_) { tensor->Backward(); }
    backward_reached_ = 0;
}

Linear::Linear(Tensor *weight, Tensor *bias)
    : in_dim_(weight->Dims()[0]), out_dim_(weight->Dims()[1]), w_(weight), b_(bias) {}

std::vector<std::shared_ptr<Tensor>> Linear::ForwardImpl() {
    CHECK_EQ(input_tensors_.size(), 1);
    CHECK_EQ(input_tensors_[0]->Dims().size(), 2);
    CHECK_EQ(input_tensors_[0]->Dims()[1], in_dim_);

    const auto &input = input_tensors_[0];
    const int bs = input->Dims()[0];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{bs, out_dim_}, DataType::kFLOAT32);

    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < out_dim_; ++j) {
            float sum = 0.0f;
            for (int64_t k = 0; k < in_dim_; ++k) {
                sum += reinterpret_cast<const float *>(input->DataPtr())[i * in_dim_ + k]
                     * reinterpret_cast<const float *>(w_->DataPtr())[k * out_dim_ + j];
            }
            reinterpret_cast<float *>(output->DataPtr())[i * out_dim_ + j]
                = sum + reinterpret_cast<const float *>(b_->DataPtr())[j];
        }
    }

    return {output};
}

void Linear::BackwardImpl() {
    auto &output = output_tensors_[0];
    auto &input = input_tensors_[0];
    const int bs = input->Dims()[0];

    // Compute the gradient of the linear transformation
    if (input->Gradient()) {
        for (int64_t i = 0; i < bs; ++i) {
            for (int64_t j = 0; j < in_dim_; ++j) {
                float sum = 0.0f;
                for (int64_t k = 0; k < out_dim_; ++k) {
                    sum += reinterpret_cast<float *>(w_->DataPtr())[j * out_dim_ + k]
                         * reinterpret_cast<const float *>(output->Gradient()->DataPtr())[i * out_dim_ + k];
                }
                reinterpret_cast<float *>(input->Gradient()->DataPtr())[i * in_dim_ + j] += sum;
            }
        }
    }

    // Compute the gradient of the weights and biases
    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < in_dim_; ++j) {
            for (int64_t k = 0; k < out_dim_; ++k) {
                reinterpret_cast<float *>(w_->Gradient()->DataPtr())[j * out_dim_ + k]
                    += reinterpret_cast<float *>(input->DataPtr())[i * in_dim_ + j]
                     * reinterpret_cast<const float *>(output->Gradient()->DataPtr())[i * out_dim_ + k];
            }
        }
        for (int64_t k = 0; k < out_dim_; ++k) {
            reinterpret_cast<float *>(b_->Gradient()->DataPtr())[k]
                += reinterpret_cast<const float *>(output->Gradient()->DataPtr())[i * out_dim_ + k];
        }
    }
}

std::vector<std::shared_ptr<Tensor>> Sigmoid::ForwardImpl() {
    CHECK_EQ(input_tensors_.size(), 1);

    const auto &input = input_tensors_[0];
    const int bs = input->Dims()[0];
    const int out_dim = input->Dims()[1];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{bs, out_dim}, DataType::kFLOAT32);

    for (int64_t i = 0; i < bs; ++i) {
        for (int64_t j = 0; j < out_dim; ++j) {
            const float x = reinterpret_cast<const float *>(input->DataPtr())[i * out_dim + j];
            reinterpret_cast<float *>(output->DataPtr())[i * out_dim + j] = 1.0f / (1.0f + exp(-x));
        }
    }

    return {output};
}

void Sigmoid::BackwardImpl() {
    // Get the input tensor and output tensor
    const auto &output = output_tensors_[0];
    const auto &input = input_tensors_[0];
    const int bs = input->Dims()[0];
    const int out_dim = input->Dims()[1];

    // Compute the gradient of the sigmoid function
    if (input->Gradient()) {
        for (int64_t i = 0; i < bs; ++i) {
            for (int64_t j = 0; j < out_dim; ++j) {
                const float x = reinterpret_cast<const float *>(output->DataPtr())[i * out_dim + j];
                const float grad = reinterpret_cast<const float *>(output->Gradient()->DataPtr())[i * out_dim + j];
                reinterpret_cast<float *>(input->Gradient()->DataPtr())[i * out_dim + j] += grad * x * (1.0f - x);
            }
        }
    }
}

std::vector<std::shared_ptr<Tensor>> CrossEntropy::ForwardImpl() {
    CHECK_EQ(input_tensors_.size(), 2);
    CHECK_EQ(input_tensors_[0]->Dims().size(), 2);
    CHECK_EQ(input_tensors_[1]->Dims().size(), 2);
    CHECK_EQ(input_tensors_[0]->Dims()[0], input_tensors_[1]->Dims()[0]);
    CHECK_EQ(input_tensors_[1]->Dims()[1], 1);

    const auto &input = input_tensors_[0]; // Logits (before softmax)
    const int bs = input->Dims()[0];
    const int num_classes = input->Dims()[1];

    const Tensor &target = *input_tensors_[1]; // raw target index

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32);

    float loss = 0.0f;
    // compute log_softmax
    for (int64_t i = 0; i < bs; ++i) {
        // extract logits and compute softmax denominator
        float max_logit = -std::numeric_limits<float>::infinity();
        for (int64_t j = 0; j < num_classes; ++j) {
            max_logit = std::max(max_logit, reinterpret_cast<const float *>(input->DataPtr())[i * num_classes + j]);
        }

        // compute softmax denominator
        float sum_exp = 0.0f;
        for (int64_t j = 0; j < num_classes; ++j) {
            float exp_val = exp(reinterpret_cast<const float *>(input->DataPtr())[i * num_classes + j] - max_logit);
            sum_exp += exp_val;
        }

        // compute cross-entropy loss
        const auto target_idx = reinterpret_cast<const uint8_t *>(target.DataPtr())[i];
        loss -= log(exp(reinterpret_cast<const float *>(input->DataPtr())[i * num_classes + target_idx] - max_logit)
                    / sum_exp);
    }

    // compute average loss
    reinterpret_cast<float *>(output->DataPtr())[0] = loss / bs;
    return {output};
}

void CrossEntropy::BackwardImpl() {
    const auto &input = input_tensors_[0];
    const int bs = input->Dims()[0];
    const int num_classes = input->Dims()[1];

    const auto &target = input_tensors_[1];

    if (input->Gradient()) {
        std::vector<float> softmax_probs(bs * num_classes);

        // compute softmax
        for (int64_t i = 0; i < bs; ++i) {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j < num_classes; ++j) {
                max_logit = std::max(max_logit, reinterpret_cast<const float *>(input->DataPtr())[i * num_classes + j]);
            }

            float sum_exp = 0.0f;
            for (int64_t j = 0; j < num_classes; ++j) {
                float exp_val = exp(reinterpret_cast<const float *>(input->DataPtr())[i * num_classes + j] - max_logit);
                sum_exp += exp_val;
            }

            for (int64_t j = 0; j < num_classes; ++j) {
                float exp_val = exp(reinterpret_cast<const float *>(input->DataPtr())[i * num_classes + j] - max_logit);
                softmax_probs[i * num_classes + j] = exp_val / sum_exp;
            }
        }

        // compute dL/dx = softmax_probs - one_hot(target)
        for (int64_t i = 0; i < bs; ++i) {
            const auto target_idx = reinterpret_cast<uint8_t *>(target->DataPtr())[i];
            for (int j = 0; j < num_classes; ++j) {
                float grad = softmax_probs[i * num_classes + j] - (j == target_idx ? 1.0f : 0.0f);
                reinterpret_cast<float *>(input->Gradient()->DataPtr())[i * num_classes + j]
                    += grad / bs; // normalize by batch size
            }
        }
    }
}

Embedding::Embedding(Tensor *token_emb, Tensor *pos_emb) : wte_(token_emb), wpe_(pos_emb) {}

std::vector<std::shared_ptr<Tensor>> Embedding::ForwardImpl() {
    CHECK_EQ(input_tensors_.size(), 1);
    CHECK_EQ(input_tensors_[0]->Dims().size(), 2);
    CHECK_LE(input_tensors_[0]->Dims()[1], wpe_->Dims()[0]);

    const auto T = wpe_->Dims()[0];
    const auto C = wpe_->Dims()[1];
    const auto &input = input_tensors_[0];
    const auto B = input->Dims()[0];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{B, T, C}, DataType::kFLOAT32);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int ix = static_cast<int>(reinterpret_cast<const uint16_t *>(input->DataPtr())[b * T + t]);
            for (int i = 0; i < C; i++) {
                reinterpret_cast<float *>(output->DataPtr())[b * T * C + t * C + i]
                    = reinterpret_cast<float *>(wte_->DataPtr())[ix * C + i]
                    + reinterpret_cast<float *>(wpe_->DataPtr())[t * C + i];
            }
        }
    }

    return {output};
}

void Embedding::BackwardImpl() {
    const auto T = wpe_->Dims()[0];
    const auto C = wpe_->Dims()[1];

    auto &output = output_tensors_[0];
    auto &input = input_tensors_[0];
    const auto B = input->Dims()[0];

    if (input->Gradient()) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int ix = static_cast<int>(reinterpret_cast<const uint16_t *>(input->DataPtr())[b * T + t]);
                for (int i = 0; i < C; i++) {
                    float d = reinterpret_cast<float *>(output->Gradient()->DataPtr())[b * T * C + t * C + i];
                    reinterpret_cast<float *>(wte_->Gradient()->DataPtr())[ix * C + i] += d;
                    reinterpret_cast<float *>(wpe_->Gradient()->DataPtr())[t * C + i] += d;
                }
            }
        }
    }
}

LayerNorm::LayerNorm(Tensor *w, Tensor *b, float eps) : w_(w), b_(b), eps_(eps) {}

std::vector<std::shared_ptr<Tensor>> LayerNorm::ForwardImpl() {
    CHECK_EQ(input_tensors_.size(), 1);
    CHECK_EQ(input_tensors_[0]->Dims().size(), 3);
    CHECK_LE(input_tensors_[0]->Dims()[2], w_->Dims()[0]);

    const auto &input = input_tensors_[0];
    const auto B = input->Dims()[0];
    const auto T = input->Dims()[1];
    const auto C = w_->Dims()[0];

    auto output = std::make_shared<Tensor>(std::vector<int64_t>{B, T, C}, DataType::kFLOAT32);
    mean_ = std::make_unique<Tensor>(std::vector<int64_t>{B, T}, DataType::kFLOAT32);
    rstd_ = std::make_unique<Tensor>(std::vector<int64_t>{B, T}, DataType::kFLOAT32);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float m = 0.0f;
            for (int i = 0; i < C; i++) { m += reinterpret_cast<float *>(input->DataPtr())[b * T * C + t * C + i]; }
            m = m / C;

            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = reinterpret_cast<float *>(input->DataPtr())[b * T * C + t * C + i] - m;
                v += xshift * xshift;
            }
            v = v / C;

            float s = 1.0f / sqrtf(v + eps_);

            for (int i = 0; i < C; i++) {
                float n = (s * (reinterpret_cast<float *>(input->DataPtr())[b * T * C + t * C + i] - m)); // normalize
                float o = n * reinterpret_cast<float *>(w_->DataPtr())[i]
                        + reinterpret_cast<float *>(b_->DataPtr())[i];                   // scale and shift
                reinterpret_cast<float *>(output->DataPtr())[b * T * C + t * C + i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            reinterpret_cast<float *>(mean_->DataPtr())[b * T + t] = m;
            reinterpret_cast<float *>(rstd_->DataPtr())[b * T + t] = s;
        }
    }

    return {output};
}

void LayerNorm::BackwardImpl() {
    const auto &input = input_tensors_[0];
    const auto &output = output_tensors_[0];
    const auto B = input->Dims()[0];
    const auto T = input->Dims()[1];
    const auto C = w_->Dims()[0];

    if (input->Gradient()) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float mean_bt = reinterpret_cast<float *>(mean_->DataPtr())[b * T + t];
                float rstd_bt = reinterpret_cast<float *>(rstd_->DataPtr())[b * T + t];

                // first: two reduce operations
                float dnorm_mean = 0.0f;
                float dnorm_norm_mean = 0.0f;
                for (int i = 0; i < C; i++) {
                    float norm_bti
                        = (reinterpret_cast<float *>(input->DataPtr())[b * T * C + t * C + i] - mean_bt) * rstd_bt;
                    float dnorm_i = reinterpret_cast<float *>(w_->DataPtr())[i]
                                  * reinterpret_cast<float *>(output->Gradient()->DataPtr())[b * T * C + t * C + i];
                    dnorm_mean += dnorm_i;
                    dnorm_norm_mean += dnorm_i * norm_bti;
                }
                dnorm_mean = dnorm_mean / C;
                dnorm_norm_mean = dnorm_norm_mean / C;

                // now iterate again and accumulate all the gradients
                for (int i = 0; i < C; i++) {
                    float norm_bti
                        = (reinterpret_cast<float *>(input->DataPtr())[b * T * C + t * C + i] - mean_bt) * rstd_bt;
                    float dnorm_i = reinterpret_cast<float *>(w_->DataPtr())[i]
                                  * reinterpret_cast<float *>(output->Gradient()->DataPtr())[b * T * C + t * C + i];
                    // gradient contribution to bias
                    reinterpret_cast<float *>(b_->Gradient()->DataPtr())[i]
                        += reinterpret_cast<float *>(output->Gradient()->DataPtr())[b * T * C + t * C + i];
                    // gradient contribution to weight
                    reinterpret_cast<float *>(w_->Gradient()->DataPtr())[i]
                        += norm_bti * reinterpret_cast<float *>(output->Gradient()->DataPtr())[b * T * C + t * C + i];
                    // gradient contribution to input
                    float dval = 0.0f;
                    dval += dnorm_i;                    // term 1
                    dval -= dnorm_mean;                 // term 2
                    dval -= norm_bti * dnorm_norm_mean; // term 3
                    dval *= rstd_bt;                    // final scale
                    reinterpret_cast<float *>(input->Gradient()->DataPtr())[b * T * C + t * C + i] += dval;
                }
            }
        }
    }
}

} // namespace infini_train::ops
