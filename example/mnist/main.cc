#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "infini_train/include/dataloader.h"
#include "infini_train/include/dataset.h"
#include "infini_train/include/network.h"
#include "infini_train/include/op.h"
#include "infini_train/include/optimizer.h"
#include "infini_train/include/tensor.h"

DEFINE_string(dataset, "", "mnist dataset path");
DEFINE_int32(bs, 64, "batch size");
DEFINE_int32(num_epoch, 1, "num epochs");
DEFINE_double(lr, 0.01, "learning rate");

namespace {
using DataType = infini_train::DataType;

enum class SN3PascalVincentType : int {
    kUINT8,
    kINT8,
    kINT16,
    kINT32,
    kFLOAT32,
    kFLOAT64,
    kINVALID,
};

const std::unordered_map<int, SN3PascalVincentType> kTypeMap = {
    {8, SN3PascalVincentType::kUINT8},
    {9, SN3PascalVincentType::kINT8},
    {11, SN3PascalVincentType::kINT16},
    {12, SN3PascalVincentType::kINT32},
    {13, SN3PascalVincentType::kFLOAT32},
    {14, SN3PascalVincentType::kFLOAT64},
};

const std::unordered_map<SN3PascalVincentType, size_t> kSN3TypeToSize = {
    {SN3PascalVincentType::kUINT8, 1},
    {SN3PascalVincentType::kINT8, 1},
    {SN3PascalVincentType::kINT16, 2},
    {SN3PascalVincentType::kINT32, 4},
    {SN3PascalVincentType::kFLOAT32, 4},
    {SN3PascalVincentType::kFLOAT64, 8},
};

const std::unordered_map<SN3PascalVincentType, DataType> kSN3TypeToDataType = {
    {SN3PascalVincentType::kUINT8, DataType::kUINT8},
    {SN3PascalVincentType::kINT8, DataType::kINT8},
    {SN3PascalVincentType::kINT16, DataType::kINT16},
    {SN3PascalVincentType::kINT32, DataType::kINT32},
    {SN3PascalVincentType::kFLOAT32, DataType::kFLOAT32},
    {SN3PascalVincentType::kFLOAT64, DataType::kFLOAT64},
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes,
                                                  std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

struct SN3PascalVincentFile {
    SN3PascalVincentType type = SN3PascalVincentType::kINVALID;
    std::vector<int64_t> dims;
    infini_train::Tensor tensor;
};

SN3PascalVincentFile ReadSN3PascalVincentFile(const std::string &path) {
    /*
      magic                                     | dims               | data    |
      reserved | reserved | type_int | num_dims |
      1 byte   | 1 byte   | 1 byte   | 1 byte   | 4*{num_dims} bytes | # bytes |
    */
    SN3PascalVincentFile sn3_file;
    std::ifstream ifs(path, std::ios::binary);
    const auto magic = ReadSeveralBytesFromIfstream(4, &ifs);
    const int num_dims = magic[3];
    const int type_int = magic[2];
    sn3_file.type = kTypeMap.at(type_int);

    auto &dims = sn3_file.dims;
    dims.resize(num_dims, 0);
    for (int dim_idx = 0; dim_idx < num_dims; ++dim_idx) {
        for (const auto &v : ReadSeveralBytesFromIfstream(4, &ifs)) {
            dims[dim_idx] <<= 8;
            dims[dim_idx] += v;
        }
    }
    const int data_size_in_bytes = kSN3TypeToSize.at(sn3_file.type) * std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>());
    sn3_file.tensor = infini_train::Tensor(dims, kSN3TypeToDataType.at(sn3_file.type));
    ifs.read(reinterpret_cast<char *>(sn3_file.tensor.DataPtr()),
             data_size_in_bytes);
    return sn3_file;
}

class MNISTDataset : public infini_train::Dataset {
public:
    MNISTDataset(const std::string &prefix)
        : image_file_(ReadSN3PascalVincentFile(prefix + "-images-idx3-ubyte")),
          label_file_(ReadSN3PascalVincentFile(prefix + "-labels-idx1-ubyte")),
          image_dims_(image_file_.dims.begin() + 1, image_file_.dims.end()),
          label_dims_(label_file_.dims.begin() + 1, label_file_.dims.end()),
          image_size_in_bytes_(kSN3TypeToSize.at(image_file_.type) * std::accumulate(image_dims_.begin(), image_dims_.end(), 1, std::multiplies<int>())),
          label_size_in_bytes_(kSN3TypeToSize.at(label_file_.type) * std::accumulate(label_dims_.begin(), label_dims_.end(), 1, std::multiplies<int>())) {
        CHECK_EQ(image_file_.dims[0], label_file_.dims[0]);
        CHECK_EQ(static_cast<int>(image_file_.tensor.Dtype()),
                 static_cast<int>(DataType::kUINT8));
        const auto &image_dims = image_file_.tensor.Dims();
        CHECK_EQ(image_dims.size(), 3);
        CHECK_EQ(image_dims[1], 28);
        CHECK_EQ(image_dims[2], 28);

        const auto bs = image_dims[0];
        infini_train::Tensor transposed_tensor(image_dims, DataType::kFLOAT32);
        for (int idx = 0; idx < bs; ++idx) {
            const auto *image_data = reinterpret_cast<uint8_t *>(image_file_.tensor.DataPtr()) + idx * 28 * 28;
            auto *transposed_data = reinterpret_cast<float *>(transposed_tensor.DataPtr()) + idx * 28 * 28;
            for (int i = 0; i < 28; ++i) {
                for (int j = 0; j < 28; ++j) {
                    transposed_data[i * 28 + j] = image_data[i * 28 + j] / 255.0f;
                }
            }
        }
        image_file_.tensor = std::move(transposed_tensor);
    }

    std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
    operator[](size_t idx) const override {
        CHECK_LT(idx, image_file_.dims[0]);
        return {std::make_shared<infini_train::Tensor>(image_file_.tensor, idx * image_size_in_bytes_, image_dims_),
                std::make_shared<infini_train::Tensor>(label_file_.tensor, idx * label_size_in_bytes_, label_dims_)};
    }

    size_t Size() const override { return image_file_.dims[0]; }

private:
    SN3PascalVincentFile image_file_;
    SN3PascalVincentFile label_file_;
    std::vector<int64_t> image_dims_;
    std::vector<int64_t> label_dims_;
    const size_t image_size_in_bytes_ = 0;
    const size_t label_size_in_bytes_ = 0;
};

class MNIST : public infini_train::Network {
public:
    MNIST() {
        AddNamedLayer("linear1", std::make_unique<infini_train::ops::Linear>(784, 30));
        AddNamedLayer("sigmoid1", std::make_unique<infini_train::ops::Sigmoid>());
        AddNamedLayer("linear2", std::make_unique<infini_train::ops::Linear>(30, 10));
    }

    std::vector<std::shared_ptr<infini_train::Tensor>>
    Forward(const std::vector<std::shared_ptr<infini_train::Tensor>> &x) override {
        CHECK_EQ(x.size(), 1);
        auto x1 = GetLayer("linear1")->Forward(x);
        auto x2 = GetLayer("sigmoid1")->Forward(x1);
        auto x3 = GetLayer("linear2")->Forward(x2);
        return x3;
    }
};
} // namespace

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    auto train_dataset = std::make_shared<MNISTDataset>(FLAGS_dataset);
    infini_train::DataLoader train_dataloader(train_dataset, FLAGS_bs);

    auto network = MNIST();
    // network.To(device);
    auto loss_fn = infini_train::loss::CrossEntropyLoss();
    auto optimizer = infini_train::optimizers::SGD(network.Parameters(), FLAGS_lr);

    int idx = 0;
    for (int epoch = 0; epoch < FLAGS_num_epoch; ++epoch) {
        idx = 0;
        for (const auto &[image, label] : train_dataloader) {
            // image.to(device);
            // label.to(device);
            auto outputs = network.Forward({image});
            optimizer.ZeroGrad();
            auto loss = loss_fn.Forward({outputs[0], label});
            if (idx % 10 == 0) {
                LOG(ERROR) << "epoch: " << epoch << ", [" << idx * FLAGS_bs << "/" << train_dataset->Size() << "] "
                           << " loss: " << reinterpret_cast<float *>(loss[0]->DataPtr())[0];
            }
            loss[0]->Backward();
            optimizer.Step();
            idx += 1;
        }
    }

    gflags::ShutDownCommandLineFlags();
    google::ShutdownGoogleLogging();

    return 0;
}
