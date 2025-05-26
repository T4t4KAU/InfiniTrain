#include "glog/logging.h"

#include "infini_train/include/sampler.h"

namespace infini_train {
std::vector<size_t> RandomSampler::GetIndices(size_t dataset_size) {
    std::vector<size_t> indices;
    std::random_device rd;
    std::mt19937 gen(rd());

    const size_t n = num_samples_ > 0 ? num_samples_ : dataset_size;

    if (repalcement_) {
        indices.resize(n);
        std::uniform_int_distribution<size_t> dist(0, dataset_size - 1);
        for (size_t i = 0; i < n; ++i) { indices[i] = dist(gen); }
    } else {
        indices.resize(dataset_size);
        for (size_t i = 0; i < dataset_size; ++i) { indices[i] = i; }
        std::shuffle(indices.begin(), indices.end(), gen);
        if (n < dataset_size) {
            indices.resize(n);
        }
    }

    return indices;
}

std::vector<size_t> SequentialSampler::GetIndices(size_t dataset_size) {
    std::vector<size_t> indices(dataset_size);

    for (int i = 0; i < dataset_size; i++) { indices[i] = i; }

    return indices;
}

std::vector<size_t> SubsetRandomSampler::GetIndices(size_t dataset_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<size_t> indices(*indices_);

    std::shuffle(indices.begin(), indices.end(), gen);

    return indices;
};
} // namespace infini_train