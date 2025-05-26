#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "infini_train/include/dataset.h"
#include "infini_train/include/sampler.h"
#include "infini_train/include/tensor.h"

namespace infini_train {
class DataLoaderIterator {
public:
    DataLoaderIterator(const Dataset &dataset, size_t batch_size, size_t batch_idx, size_t max_batch_idx,
                       const std::vector<size_t> &indices);

    std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> operator*() const;

    DataLoaderIterator &operator++();
    DataLoaderIterator operator++(int);

    friend bool operator<(const DataLoaderIterator &lhs, const DataLoaderIterator &rhs);
    friend bool operator!=(const DataLoaderIterator &lhs, const DataLoaderIterator &rhs);
    friend bool operator==(const DataLoaderIterator &lhs, const DataLoaderIterator &rhs);

private:
    const Dataset *dataset_ = nullptr; // not owned
    size_t batch_size_ = 0;
    size_t batch_idx_ = 0;
    size_t max_batch_idx_ = 0;
    const std::vector<size_t> *indices_ = nullptr;
};

class DataLoader {
public:
    DataLoader(const std::shared_ptr<Dataset> &dataset, size_t batch_size, bool shuffle = false,
               std::unique_ptr<Sampler> sampler = nullptr);

    DataLoaderIterator begin() const;
    DataLoaderIterator end() const;

private:
    std::shared_ptr<Dataset> dataset_;
    size_t batch_size_ = 0;
    size_t max_batch_idx_ = 0;
    std::unique_ptr<Sampler> sampler_;
    std::shared_ptr<std::vector<size_t>> indices_;
};
} // namespace infini_train
