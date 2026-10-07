#ifndef PITCHEE_VAD_HPP
#define PITCHEE_VAD_HPP

#include "internal.hpp"
#include "model_runtime.hpp"

#include <filesystem>
#include <functional>
#include <memory>

namespace pitchee {

class VadDetector {
public:
    VadDetector(
        const std::filesystem::path& model_path,
        int intra_op_threads
    );

    VadResult detect(
        const std::vector<float>& samples,
        const std::function<void(size_t, size_t)>& progress = {}
    ) const;

private:
    VadResult detect_light(
        const std::vector<float>& samples,
        const std::function<void(size_t, size_t)>& progress
    ) const;

    bool lightweight_ = false;
    std::unique_ptr<ModelRuntime> model_;
};

}  // namespace pitchee

#endif
