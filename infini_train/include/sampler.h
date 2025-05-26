#pragma once

#include <vector>

#include "infini_train/include/dataset.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
class Sampler {
public:
    virtual ~Sampler() = default;
    virtual std::vector<size_t> GetIndices(size_t dataset_size) = 0;
};

class RandomSampler : public Sampler {
public:
    RandomSampler(bool repalcement = false, size_t num_samples = 0)
        : repalcement_(repalcement), num_samples_(num_samples) {}

    std::vector<size_t> GetIndices(size_t dataset_size) override;

private:
    bool repalcement_;
    size_t num_samples_;
};

class SequentialSampler : public Sampler {
public:
    SequentialSampler() {}
    std::vector<size_t> GetIndices(size_t dataset_size) override;
};

class SubsetRandomSampler : public Sampler {
public:
    SubsetRandomSampler(std::vector<size_t> &indices) : indices_(&indices) {}
    std::vector<size_t> GetIndices(size_t dataset_size) override;

private:
    std::shared_ptr<std::vector<size_t>> indices_;
};
} // namespace infini_train