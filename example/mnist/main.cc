#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/network.h"
#include "infini_train/include/optimizer.h"

#include "example/mnist/dataset.h"
#include "example/mnist/net.h"

DEFINE_string(dataset, "", "mnist dataset path");
DEFINE_int32(bs, 64, "batch size");
DEFINE_int32(num_epoch, 1, "num epochs");
DEFINE_double(lr, 0.01, "learning rate");

using namespace infini_train;

namespace {
constexpr int kNumItersOfOutputDuration = 10;
constexpr int kNumClasses = 10;
}; // namespace

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    auto train_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, true);
    DataLoader train_dataloader(train_dataset, FLAGS_bs);

    // TODO(dcj): Add sampler & eval dataloader later.
    auto test_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset, false);
    DataLoader test_dataloader(test_dataset, FLAGS_bs);

    auto network = MNIST();
    // network.To(device);
    auto loss_fn = loss::CrossEntropyLoss();
    auto optimizer = optimizers::SGD(network.Parameters(), FLAGS_lr);

    for (int epoch = 0; epoch < FLAGS_num_epoch; ++epoch) {
        int train_idx = 0;
        for (const auto &[image, label] : train_dataloader) {
            // image.to(device);
            // label.to(device);
            auto outputs = network.Forward({image});
            optimizer.ZeroGrad();
            auto loss = loss_fn.Forward({outputs[0], label});
            if (train_idx % kNumItersOfOutputDuration == 0) {
                LOG(ERROR) << "epoch: " << epoch << ", [" << train_idx * FLAGS_bs << "/" << train_dataset->Size() << "] "
                           << " loss: " << reinterpret_cast<float *>(loss[0]->DataPtr())[0];
            }
            loss[0]->Backward();
            optimizer.Step();
            train_idx += 1;
        }
    }

    // TODO(dcj): Add no_grad() context manager later.
    std::vector<float> test_losses;
    int correct = 0;
    int total = 0;
    for (const auto &[image, label] : test_dataloader) {
        auto outputs = network.Forward({image});
        auto loss = loss_fn.Forward({outputs[0], label});
        const int batch_size = outputs[0]->Dims()[0];
        for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
            auto label_index = reinterpret_cast<uint8_t *>(label->DataPtr())[batch_idx];
            const auto *output_values = reinterpret_cast<float *>(outputs[0]->DataPtr()) + batch_idx * kNumClasses;
            const int output_index = std::max_element(output_values, output_values + kNumClasses) - output_values;
            if (output_index == label_index) {
                ++correct;
            }
        }
        total += batch_size;
        test_losses.push_back(reinterpret_cast<float *>(loss[0]->DataPtr())[0]);
    }
    const auto avg_loss = std::accumulate(test_losses.begin(), test_losses.end(), 0.0) / test_losses.size();
    LOG(ERROR) << "Total: " << total << ", Correct: " << correct << ", Accuracy: " << static_cast<float>(correct) / total << ", AverateLoss: " << avg_loss;

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}
