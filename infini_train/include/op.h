#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace infini_train {
class Op {
public:
    virtual ~Op() = default;

    std::vector<std::shared_ptr<Tensor>> Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors);
    // TODO(dcj): Support multiple output tensors later.
    void Backward(const Tensor *output_tensor);

    virtual std::vector<std::shared_ptr<Tensor>> ForwardImpl() = 0;
    virtual void BackwardImpl(const Tensor *output_tensor) = 0;

    void AddWeight(const std::vector<int64_t> &dims, const DataType dtype);

    std::vector<Tensor> &Weights();
    const std::vector<Tensor> &Weights() const;

protected:
    std::vector<std::shared_ptr<Tensor>> input_tensors_;
    std::vector<Tensor> weights_;
};

namespace ops {
/*
  x: [bs, in_dim]
  -> Linear (w: [in_dim, out_dim], b: [out_dim])
  -> o: [bs, out_dim]
 */
class Linear : public Op {
public:
    Linear(int64_t in_dim, int64_t out_dim);

    std::vector<std::shared_ptr<Tensor>> ForwardImpl() override;
    void BackwardImpl(const Tensor *output_tensor) override;

private:
    int64_t in_dim_ = 0;
    int64_t out_dim_ = 0;
};

/*
  -> Sigmoid
  -> input: [bs, out_dim]
  -> output: [bs, out_dim]
 */
class Sigmoid : public Op {
public:
    Sigmoid() = default;

    std::vector<std::shared_ptr<Tensor>> ForwardImpl() override;
    void BackwardImpl(const Tensor *output_tensor) override;
};

/*
  -> CrossEntropy
  -> input: [bs, out_dim]
  -> target: [bs,]
  -> output: scalar
*/
class CrossEntropy : public Op {
public:
    CrossEntropy() = default;

    std::vector<std::shared_ptr<Tensor>> ForwardImpl() override;
    void BackwardImpl(const Tensor *output_tensor) override;
};
} // namespace ops
} // namespace infini_train
