#include "infini_train/include/autograd/transform.h"

#include "infini_train/include/kernels/cpu/transform.h"
#ifdef USE_CUDA
#include "infini_train/include/kernels/cuda/transform.h"
#endif

#include <vector>

namespace infini_train::autograd {
std::vector<std::shared_ptr<Tensor>> Tril::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::TrilForward(input, diagonal_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::TrilForward(input, diagonal_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

std::vector<std::shared_ptr<Tensor>> Tril::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];
    std::shared_ptr<Tensor> grad_input = nullptr;

    switch (grad_output->GetDevice().Type()) {
    case DeviceType::kCPU: {
        grad_input = kernels::cpu::TrilBackward(grad_output, diagonal_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        grad_input = kernels::cuda::TrilBackward(grad_output, diagonal_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(grad_output->GetDevice().Type());
        break;
    }
    return {grad_input};
}

std::vector<std::shared_ptr<Tensor>> Transpose::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::TransposeForward(input, dim0_, dim1_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::TransposeForward(input, dim0_, dim1_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

std::vector<std::shared_ptr<Tensor>> Transpose::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];
    std::shared_ptr<Tensor> grad_input = nullptr;

    switch (grad_output->GetDevice().Type()) {
    case DeviceType::kCPU: {
        grad_input = kernels::cpu::TransposeBackward(grad_output, dim0_, dim1_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        grad_input = kernels::cuda::TransposeBackward(grad_output, dim0_, dim1_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(grad_output->GetDevice().Type());
        break;
    }
    return {grad_input};
}

std::vector<std::shared_ptr<Tensor>> Mask::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    std::shared_ptr<Tensor> output = nullptr;
    switch (input->GetDevice().Type()) {
    case DeviceType::kCPU: {
        output = kernels::cpu::MaskForward(input, mask_, value_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        output = kernels::cuda::MaskForward(input, mask_, value_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(input->GetDevice().Type());
        break;
    }
    return {output};
}

std::vector<std::shared_ptr<Tensor>> Mask::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    const auto &grad_output = grad_outputs[0];
    std::shared_ptr<Tensor> grad_input = nullptr;

    switch (grad_output->GetDevice().Type()) {
    case DeviceType::kCPU: {
        grad_input = kernels::cpu::MaskBackward(grad_output, mask_);
        break;
    }
#ifdef USE_CUDA
    case DeviceType::kCUDA: {
        grad_input = kernels::cuda::MaskBackward(grad_output, mask_);
        break;
    }
#endif
    default:
        LOG(FATAL) << "Unsupported device type: " << static_cast<int>(grad_output->GetDevice().Type());
        break;
    }
    return {grad_input};
}
} // namespace infini_train::autograd
