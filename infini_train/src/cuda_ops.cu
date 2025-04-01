#include "infini_train/include/ops.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

#include "cublas_v2.h"
#include "glog/logging.h"

#include "infini_train/include/device.h"
#include "infini_train/include/tensor.h"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

namespace infini_train::ops {
namespace {
constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();
}

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "CUBLAS Error: " << cublasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

CUDALinear::CUDALinear(Tensor *weight, Tensor *bias) : Linear(weight, bias) {}

std::vector<std::shared_ptr<Tensor>> CUDALinear::ForwardImpl() {
    CHECK_EQ(input_tensors_.size(), 1);

    auto &x = input_tensors_[0];
    CHECK_EQ(x->Dims().size(), 2);
    CHECK_EQ(x->Dims()[1], in_dim_);
    const int bs = x->Dims()[0];

    auto y = std::make_shared<Tensor>(std::vector<int64_t>{bs, out_dim_}, DataType::kFLOAT32,
                                      Device(DeviceType::kCUDA, 0));
    for (int idx = 0; idx < bs; ++idx) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<float *>(y->DataPtr()) + idx * out_dim_, b_->DataPtr(),
                              out_dim_ * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Y = X * W + B
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_dim_, bs, in_dim_, &alpha,
                             reinterpret_cast<const float *>(w_->DataPtr()), out_dim_,
                             reinterpret_cast<const float *>(x->DataPtr()), in_dim_, &beta,
                             reinterpret_cast<float *>(y->DataPtr()), out_dim_));

    CUBLAS_CHECK(cublasDestroy(handle));

    return {y};
}

__global__ void set_ones(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f;
    }
}

void CUDALinear::BackwardImpl() {

    auto &y = output_tensors_[0];
    auto &x = input_tensors_[0];
    const int bs = x->Dims()[0];

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // dX = dY * W^T
    if (x->Gradient()) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_dim_, bs, out_dim_, &alpha,
                                 reinterpret_cast<const float *>(w_->DataPtr()), out_dim_,
                                 reinterpret_cast<const float *>(y->Gradient()->DataPtr()), out_dim_, &beta,
                                 reinterpret_cast<float *>(x->Gradient()->DataPtr()), in_dim_));
    }

    // dW = X^T * dY
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_dim_, in_dim_, bs, &alpha,
                             reinterpret_cast<const float *>(y->Gradient()->DataPtr()), out_dim_,
                             reinterpret_cast<const float *>(x->DataPtr()), in_dim_, &beta,
                             reinterpret_cast<float *>(w_->Gradient()->DataPtr()), out_dim_));
    // FIXME(dcj): remove this sync
    CUDA_CHECK(cudaDeviceSynchronize());

    // dB = \sum_i(i=0, bs-1) dY_i
    // TODO(dcj): use thrust::fill or reduce kernel do this
    auto ones = std::make_shared<Tensor>(std::vector<int64_t>{bs}, DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    float *d_ptr = reinterpret_cast<float *>(ones->DataPtr());

    // TODO(dcj): use const variable for threads_per_block and num_blocks
    int threads_per_block = 256;
    int num_blocks = (bs + threads_per_block - 1) / threads_per_block;
    set_ones<<<num_blocks, threads_per_block>>>(d_ptr, bs);

    CUBLAS_CHECK(cublasSgemv(handle, CUBLAS_OP_T, out_dim_, bs, &alpha,
                             reinterpret_cast<const float *>(y->Gradient()->DataPtr()), out_dim_,
                             reinterpret_cast<const float *>(ones->DataPtr()), 1, &beta,
                             reinterpret_cast<float *>(b_->Gradient()->DataPtr()), 1));

    CUBLAS_CHECK(cublasDestroy(handle));
}

// Sigmoid CUDA Kernel
__global__ void SigmoidKernel(const float *input, float *output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

// Sigmoid backward CUDA Kernel
__global__ void SigmoidBackwardKernel(const float *output, const float *grad_output, float *grad_input, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        grad_input[i] = grad_output[i] * output[i] * (1 - output[i]);
    }
}

// Sigmoid forward
std::vector<std::shared_ptr<Tensor>> CUDASigmoid::ForwardImpl() {
    auto &input = input_tensors_[0];
    int n = input->NumElements();

    auto output = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));

    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    SigmoidKernel<<<num_blocks, threads_per_block>>>(reinterpret_cast<const float *>(input->DataPtr()),
                                                     reinterpret_cast<float *>(output->DataPtr()), n);

    return {output};
}

// Sigmoid backward
void CUDASigmoid::BackwardImpl() {
    auto &output = output_tensors_[0];
    auto &input = input_tensors_[0];
    int n = input->NumElements();

    if (input->Gradient()) {
        int threads_per_block = 256;
        int num_blocks = (n + threads_per_block - 1) / threads_per_block;

        SigmoidBackwardKernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const float *>(output->DataPtr()),
            reinterpret_cast<const float *>(output->Gradient()->DataPtr()),
            reinterpret_cast<float *>(input->Gradient()->DataPtr()), n);
    }
}

// CrossEntropy CUDA Kernel
__global__ void CrossEntropyKernel(const float *y_pred, const uint8_t *y_target, float *loss, int batch_size,
                                   int num_classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch_size) {
        float max_logit = kNegativeInfinity;
        for (int j = 0; j < num_classes; j++) { max_logit = max(max_logit, y_pred[i * num_classes + j]); }
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) { sum_exp += expf(y_pred[i * num_classes + j] - max_logit); }
        loss[i] = -logf(expf(y_pred[i * num_classes + y_target[i]] - max_logit) / sum_exp);
    }
}

// CrossEntropy backward CUDA Kernel
__global__ void CrossEntropyBackwardKernel(float *input, float *input_grad, uint8_t *target, int bs, int num_classes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bs) {
        float max_logit = kNegativeInfinity;
        for (int j = 0; j < num_classes; j++) { max_logit = max(max_logit, input[i * num_classes + j]); }
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) { sum_exp += expf(input[i * num_classes + j] - max_logit); }
        for (int j = 0; j < num_classes; j++) {
            int idx = i * num_classes + j;
            input_grad[idx] += (expf(input[idx] - max_logit) / sum_exp - (j == target[i] ? 1.0f : 0.0f)) / bs;
        }
    }
}

// CrossEntropy forward
std::vector<std::shared_ptr<Tensor>> CUDACrossEntropy::ForwardImpl() {
    auto &y_pred = input_tensors_[0];
    auto &y_target = input_tensors_[1];

    int batch_size = y_pred->Dims()[0];
    int num_classes = y_pred->Dims()[1];

    auto batched_loss
        = std::make_shared<Tensor>(std::vector<int64_t>{batch_size}, DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));

    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    CrossEntropyKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const float *>(y_pred->DataPtr()), reinterpret_cast<const uint8_t *>(y_target->DataPtr()),
        reinterpret_cast<float *>(batched_loss->DataPtr()), batch_size, num_classes);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto loss_cpu = batched_loss->To(Device());
    auto loss = std::make_shared<Tensor>(std::vector<int64_t>{}, DataType::kFLOAT32, Device());
    reinterpret_cast<float *>(loss->DataPtr())[0]
        = std::accumulate(reinterpret_cast<const float *>(loss_cpu.DataPtr()),
                          reinterpret_cast<const float *>(loss_cpu.DataPtr()) + batch_size, 0.0f)
        / batch_size;

    return {std::make_shared<Tensor>(loss->To(Device(DeviceType::kCUDA, 0)))};
}

// CrossEntropy backward
void CUDACrossEntropy::BackwardImpl() {
    auto &input = input_tensors_[0];
    auto &target = input_tensors_[1];

    int bs = input->Dims()[0];
    int num_classes = input->Dims()[1];

    if (input->Gradient()) {
        int threads_per_block = 256;
        int num_blocks = (bs + threads_per_block - 1) / threads_per_block;

        CrossEntropyBackwardKernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<float *>(input->DataPtr()), reinterpret_cast<float *>(input->Gradient()->DataPtr()),
            reinterpret_cast<uint8_t *>(target->DataPtr()), bs, num_classes);
    }
}

CUDAEmbedding::CUDAEmbedding(Tensor *token_emb, Tensor *pos_emb) : Embedding(token_emb, pos_emb) {}

__global__ void EmbeddingForwardKernel(const uint16_t *input, float *output, const float *wte, const float *wpe, int B,
                                       int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x);
    int N = B * T * C;
    if (idx >= N) {
        return;
    }

    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;

    int ix = static_cast<int>(input[b * T + t]);

    output[b * T * C + t * C + c] = wte[ix * C + c] + wpe[t * C + c];
}

// Embedding forward
std::vector<std::shared_ptr<Tensor>> CUDAEmbedding::ForwardImpl() {
    CHECK_EQ(input_tensors_.size(), 1);
    CHECK_EQ(input_tensors_[0]->Dims().size(), 2);
    CHECK_LE(input_tensors_[0]->Dims()[1], wpe_->Dims()[0]);

    const int T = wpe_->Dims()[0];
    const int C = wpe_->Dims()[1];
    const auto &input = input_tensors_[0];
    const int B = input->Dims()[0];

    auto output
        = std::make_shared<Tensor>(std::vector<int64_t>{B, T, C}, DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));

    int threads_per_block = 256;
    int num_blocks = (B + threads_per_block - 1) / threads_per_block;
    EmbeddingForwardKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const uint16_t *>(input->DataPtr()), reinterpret_cast<float *>(output->DataPtr()),
        reinterpret_cast<const float *>(wte_->DataPtr()), reinterpret_cast<const float *>(wpe_->DataPtr()), B, T, C);

    return {output};
}

template <int BLOCK_SIZE = 256>
__global__ void WTEBackwardKernel(float *dwte, const float *doutput, const uint16_t *input, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) {
        return;
    }

    int token = static_cast<int>(input[idx]);
    if (token < 0) {
        return;
    }

    int c = threadIdx.x % C;
    float grad = doutput[idx * C + c];

    atomicAdd(&dwte[token * C + c], grad);
}

__global__ void WPEBackwardKernel(float *dwpe, const float *doutput, const uint16_t *input, int B, int T, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * C) {
        return;
    }

    int t = idx / C;
    int c = idx % C;
    float accum = 0.0f;

    for (int b = 0; b < B; b++) { accum += doutput[b * T * C + t * C + c]; }

    atomicAdd(&dwpe[t * C + c], accum);
}

// Embedding backward
void CUDAEmbedding::BackwardImpl() {
    const int T = wpe_->Dims()[0];
    const int C = wpe_->Dims()[1];
    const auto &input = input_tensors_[0];
    const auto &output = output_tensors_[0];
    const int B = input->Dims()[0];

    if (input->Gradient()) {
        // TODO(zbl): check correctness
        int threads_per_block = 256;
        int num_blocks = ((T * C) + threads_per_block - 1) / threads_per_block;
        WPEBackwardKernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<float *>(wpe_->Gradient()->DataPtr()),
            reinterpret_cast<const float *>(output->Gradient()->DataPtr()),
            reinterpret_cast<const uint16_t *>(input->DataPtr()), B, T, C);

        num_blocks = ((B * T) + threads_per_block - 1) / threads_per_block;
        WTEBackwardKernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<float *>(wte_->Gradient()->DataPtr()),
            reinterpret_cast<const float *>(output->Gradient()->DataPtr()),
            reinterpret_cast<const uint16_t *>(input->DataPtr()), B, T, C);
    }
}

CUDALayerNorm::CUDALayerNorm(Tensor *w, Tensor *b, float eps) : LayerNorm(w, b, eps) {}

__forceinline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) { val += __shfl_down_sync(0xffffffff, val, offset); }
    return val;
}

__global__ void LayerNormForwardKernel(const float *input, float *output, float *mean, float *rstd, const float *weight,
                                       const float *bias, int N, int C) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int idx = blockIdx.x * (blockDim.x / WARP_SIZE) + warp_id;
    if (idx >= N) {
        return;
    }

    float sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) { sum += (float)input[idx * C + i]; }
    float m = warpReduceSum(sum) / C;
    if (lane_id == 0 && mean) {
        mean[idx] = m;
    }

    sum = 0.0f;
    for (int i = lane_id; i < C; i += WARP_SIZE) {
        float diff = (float)input[idx * C + i] - m;
        sum += diff * diff;
    }
    float s = rsqrtf(warpReduceSum(sum) / C + 1e-5f);
    if (lane_id == 0 && rstd) {
        rstd[idx] = s;
    }

    for (int c = lane_id; c < C; c += WARP_SIZE) {
        float n = s * ((float)input[idx * C + c] - m);
        output[idx * C + c] = (float)(n * (float)weight[c] + (float)bias[c]);
    }
}

// LayerNorm forward
std::vector<std::shared_ptr<Tensor>> CUDALayerNorm::ForwardImpl() {
    CHECK_EQ(input_tensors_.size(), 1);
    CHECK_EQ(input_tensors_[0]->Dims().size(), 3);
    CHECK_LE(input_tensors_[0]->Dims()[2], w_->Dims()[0]);

    const auto &input = input_tensors_[0];
    const auto B = input->Dims()[0];
    const auto T = input->Dims()[1];
    const auto C = w_->Dims()[0];

    auto output
        = std::make_shared<Tensor>(std::vector<int64_t>{B, T, C}, DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    mean_ = std::make_unique<Tensor>(std::vector<int64_t>{B, T}, DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));
    rstd_ = std::make_unique<Tensor>(std::vector<int64_t>{B, T}, DataType::kFLOAT32, Device(DeviceType::kCUDA, 0));

    int threads_per_block = 256;
    int block_y = threads_per_block / WARP_SIZE;
    int N = B * T;
    int num_blocks = (N + block_y - 1) / block_y;

    LayerNormForwardKernel<<<num_blocks, threads_per_block>>>(
        reinterpret_cast<const float *>(input->DataPtr()), reinterpret_cast<float *>(output->DataPtr()),
        reinterpret_cast<float *>(mean_->DataPtr()), reinterpret_cast<float *>(rstd_->DataPtr()),
        reinterpret_cast<const float *>(w_->DataPtr()), reinterpret_cast<const float *>(b_->DataPtr()), N, C);

    return {output};
}

__global__ void LayerNormBackwardKernel(float *dinput, float *dweight, float *dbias, const float *doutput,
                                        const float *input, const float *weight, const float *mean, const float *rstd,
                                        int N, int C) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane = tid % warpSize;
    const int warp_id = tid / warpSize;
    const int warps_per_block = blockDim.x / warpSize;

    __shared__ float shared_dweight[32];
    __shared__ float shared_dbias[32];

    float dweight_sum = 0.0f;
    float dbias_sum = 0.0f;
    float dinput_val = 0.0f;

    for (int i = bid * C + tid; i < N * C; i += gridDim.x * C) {
        int idx = i % C;
        float val_x = input[i];
        float val_doutput = doutput[i];
        float norm_x = (val_x - mean[i / C]) * rstd[i / C];

        dweight_sum += val_doutput * norm_x;
        dbias_sum += val_doutput;

        // Compute dinput using doutput
        dinput_val = val_doutput * weight[idx] * rstd[i / C];
        dinput[i] = dinput_val;
    }

    dweight_sum = warpReduceSum(dweight_sum);
    dbias_sum = warpReduceSum(dbias_sum);

    if (lane == 0) {
        shared_dweight[warp_id] = dweight_sum;
        shared_dbias[warp_id] = dbias_sum;
    }
    __syncthreads();

    if (warp_id == 0 && lane < warps_per_block) {
        dweight_sum = shared_dweight[lane];
        dbias_sum = shared_dbias[lane];
        dweight_sum = warpReduceSum(dweight_sum);
        dbias_sum = warpReduceSum(dbias_sum);

        if (lane == 0) {
            atomicAdd(&dweight[bid], dweight_sum);
            atomicAdd(&dbias[bid], dbias_sum);
        }
    }
}

// LayerNorm backward
void CUDALayerNorm::BackwardImpl() {
    const auto &input = input_tensors_[0];
    const auto &output = output_tensors_[0];
    const auto B = input->Dims()[0];
    const auto T = input->Dims()[1];
    const auto C = w_->Dims()[0];

    int threads_per_block = 256;
    int num_blocks = (B * T * C + threads_per_block - 1) / threads_per_block;

    if (input->Gradient()) {
        // TODO(zbl): check correctness
        LayerNormBackwardKernel<<<num_blocks, threads_per_block>>>(
            reinterpret_cast<float *>(input->Gradient()->DataPtr()),
            reinterpret_cast<float *>(w_->Gradient()->DataPtr()), reinterpret_cast<float *>(b_->Gradient()->DataPtr()),
            reinterpret_cast<const float *>(output->Gradient()->DataPtr()),
            reinterpret_cast<const float *>(input->DataPtr()), reinterpret_cast<const float *>(w_->DataPtr()),
            reinterpret_cast<const float *>(mean_->DataPtr()), reinterpret_cast<const float *>(rstd_->DataPtr()), B * T,
            C);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace infini_train::ops
