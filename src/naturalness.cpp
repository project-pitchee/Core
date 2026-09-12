#include "internal.hpp"
#include "ort_runtime.hpp"

#include <algorithm>
#include <stdexcept>

namespace pitchee {

NaturalnessModel::NaturalnessModel(
    const std::filesystem::path& path,
    int intra_op_threads
) : model_(std::make_unique<OrtModel>(path, intra_op_threads, false)) {}

NaturalnessModel::~NaturalnessModel() = default;

double NaturalnessModel::score(const std::vector<float>& features) const {
    if (features.size() != 384) {
        throw std::invalid_argument("invalid naturalness feature vector");
    }
    Tensor input;
    input.shape = {1, 384};
    input.values = features;
    auto output = model_->run({{"features", std::move(input)}});
    if (output.values.empty()) {
        throw std::runtime_error("Naturalness model returned no score");
    }
    return std::max(0.0, std::min(100.0, static_cast<double>(output.values[0])));
}

}  // namespace pitchee
