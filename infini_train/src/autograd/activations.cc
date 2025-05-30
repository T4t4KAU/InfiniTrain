#include "infini_train/include/autograd/activations.h"

#include <memory>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/kernels/cpu/sigmoid.h"
#include "infini_train/include/tensor.h"
#ifdef USE_CUDA
#include "infini_train/include/kernels/cuda/sigmoid.h"
#endif

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Sigmoid::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::SigmoidForward(input);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::SigmoidForward(input);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

void Sigmoid::SetupContext(const std::vector<std::shared_ptr<Tensor>> &,
                           const std::vector<std::shared_ptr<Tensor>> &output_tensors) {
    const auto &output = output_tensors[0];
    saved_tensors_ = {output};
}

std::vector<std::shared_ptr<Tensor>> Sigmoid::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    CHECK_EQ(saved_tensors_.size(), 1);
    const auto &output = saved_tensors_[0];
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    switch (output->GetDevice().Type()) {
    case DeviceType::kCPU: {
        auto grad_input = kernels::cpu::SigmoidBackward(output, grad_output);
        return {grad_input};
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        auto grad_input = kernels::cuda::SigmoidBackward(output, grad_output);
        return {grad_input};
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(output->GetDevice().Type());
        break;
    }
    return {};
}
} // namespace infini_train::autograd
