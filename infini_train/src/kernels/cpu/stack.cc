#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> StackForward(const std::vector<std::shared_ptr<Tensor>> &inputs, int64_t dim) {
    CHECK(!inputs.empty());

    const auto &base_dims = inputs[0]->Dims();
    if (dim < 0) {
        dim += base_dims.size() + 1;
    }
    CHECK_GE(dim, 0);
    CHECK_LE(dim, base_dims.size());

    for (const auto &input : inputs) { CHECK(input->Dims() == base_dims); }

    std::vector<int64_t> output_dims = base_dims;
    output_dims.insert(output_dims.begin() + dim, inputs.size());

    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    const int64_t outer_size
        = std::accumulate(output_dims.begin(), output_dims.begin() + dim, 1, std::multiplies<int64_t>());
    const int64_t inner_size
        = std::accumulate(output_dims.begin() + dim + 1, output_dims.end(), 1, std::multiplies<int64_t>());
    const int64_t slice_size = inputs[0]->NumElements();

    for (size_t i = 0; i < inputs.size(); ++i) {
        const float *src_ptr = static_cast<const float *>(inputs[i]->DataPtr());
        float *dst_ptr = static_cast<float *>(output->DataPtr()) + i * inner_size;
        for (int64_t n = 0; n < outer_size; ++n) {
            memcpy(dst_ptr + n * inputs.size() * inner_size, src_ptr + n * inner_size, inner_size * sizeof(float));
        }
    }

    return output;
}

std::vector<std::shared_ptr<Tensor>> StackBackward(const std::vector<int64_t> &input_dims, int64_t dim,
                                                   const std::shared_ptr<Tensor> &grad_output) {
    const int64_t num_inputs = grad_output->Dims()[dim];
    std::vector<std::shared_ptr<Tensor>> grads;
    std::vector<int64_t> out_dims = grad_output->Dims();
    out_dims.erase(out_dims.begin() + dim); // remove the stack dim

    const int64_t outer_size = std::accumulate(out_dims.begin(), out_dims.begin() + dim, 1, std::multiplies<int64_t>());
    const int64_t inner_size = std::accumulate(out_dims.begin() + dim, out_dims.end(), 1, std::multiplies<int64_t>());

    for (int i = 0; i < num_inputs; ++i) {
        auto grad = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
        float *dst_ptr = static_cast<float *>(grad->DataPtr());
        const float *src_ptr = static_cast<const float *>(grad_output->DataPtr()) + i * inner_size;
        for (int64_t n = 0; n < outer_size; ++n) {
            memcpy(dst_ptr + n * inner_size, src_ptr + n * num_inputs * inner_size, inner_size * sizeof(float));
        }
        grads.push_back(grad);
    }

    return grads;
}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_STACK_KERNEL(kernel_name)                                                                         \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_STACK_KERNEL(StackForward)
REGISTER_CPU_STACK_KERNEL(StackBackward)

#undef REGISTER_CPU_STACK_KERNEL