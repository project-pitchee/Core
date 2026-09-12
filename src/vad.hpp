#ifndef PITCHEE_VAD_HPP
#define PITCHEE_VAD_HPP

#include "internal.hpp"
#include "ort_runtime.hpp"

#include <filesystem>
#include <memory>

namespace pitchee {

class VadDetector {
public:
    VadDetector(
        const std::filesystem::path& model_path,
        int intra_op_threads
    );

    VadResult detect(const std::vector<float>& samples) const;

private:
    std::unique_ptr<OrtModel> model_;
};

}  // namespace pitchee

#endif
