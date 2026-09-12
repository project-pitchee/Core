#include "ort_runtime.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

#ifdef PITCHEE_ENABLE_ORT
#include <onnxruntime_cxx_api.h>
#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif
#endif

namespace pitchee {

bool ort_available() {
#ifdef PITCHEE_ENABLE_ORT
    return true;
#else
    return false;
#endif
}

#ifdef PITCHEE_ENABLE_ORT

struct OrtModel::Impl {
    explicit Impl(
        const std::filesystem::path& path,
        int intra_op_threads,
        bool use_coreml
    )
        : environment(ORT_LOGGING_LEVEL_WARNING, "PitcheeCore"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(intra_op_threads);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if defined(__APPLE__)
        if (use_coreml) {
            try {
                Ort::ThrowOnError(
                    OrtSessionOptionsAppendExecutionProvider_CoreML(options, 0)
                );
            } catch (...) {
                // Core ML is an acceleration hint. CPU remains a valid fallback.
            }
        }
#else
        (void)use_coreml;
#endif
#if defined(_WIN32)
        const std::wstring model_path = path.wstring();
        session = std::make_unique<Ort::Session>(
            environment,
            model_path.c_str(),
            options
        );
#else
        const std::string model_path = path.string();
        session = std::make_unique<Ort::Session>(
            environment,
            model_path.c_str(),
            options
        );
#endif
    }

    Ort::Env environment;
    Ort::MemoryInfo memory_info;
    std::unique_ptr<Ort::Session> session;
};

OrtModel::OrtModel(
    const std::filesystem::path& path,
    int intra_op_threads,
    bool use_coreml
) : impl_(new Impl(path, intra_op_threads, use_coreml)) {}

std::vector<Tensor> OrtModel::run_all(
    const std::unordered_map<std::string, Tensor>& inputs
) const {
    if (!impl_ || !impl_->session) {
        throw std::runtime_error("ONNX Runtime session is unavailable");
    }

    std::vector<std::string> input_names_storage;
    std::vector<std::string> output_names_storage;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<Ort::Value> input_values;
    input_names_storage.reserve(inputs.size());
    input_values.reserve(inputs.size());

    for (const auto& [name, tensor] : inputs) {
        input_names_storage.push_back(name);
        input_names.push_back(input_names_storage.back().c_str());
        if (tensor.data_type == Tensor::DataType::Int64) {
            input_values.push_back(Ort::Value::CreateTensor<int64_t>(
                impl_->memory_info,
                const_cast<int64_t*>(tensor.int64_values.data()),
                tensor.int64_values.size(),
                tensor.shape.data(),
                tensor.shape.size()
            ));
        } else {
            input_values.push_back(Ort::Value::CreateTensor<float>(
                impl_->memory_info,
                const_cast<float*>(tensor.values.data()),
                tensor.values.size(),
                tensor.shape.data(),
                tensor.shape.size()
            ));
        }
    }

    const auto output_count = impl_->session->GetOutputCount();
    output_names_storage.reserve(output_count);
    for (size_t index = 0; index < output_count; ++index) {
        Ort::AllocatedStringPtr name = impl_->session->GetOutputNameAllocated(
            index,
            Ort::AllocatorWithDefaultOptions()
        );
        output_names_storage.emplace_back(name.get());
    }
    for (const auto& name : output_names_storage) output_names.push_back(name.c_str());

    auto outputs = impl_->session->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_values.data(),
        input_values.size(),
        output_names.data(),
        output_names.size()
    );
    std::vector<Tensor> result;
    result.reserve(outputs.size());
    for (const auto& value : outputs) {
        const auto info = value.GetTensorTypeAndShapeInfo();
        Tensor tensor;
        tensor.shape = info.GetShape();
        const size_t element_count = info.GetElementCount();
        const float* data = value.GetTensorData<float>();
        tensor.values.assign(data, data + element_count);
        result.push_back(std::move(tensor));
    }
    return result;
}

Tensor OrtModel::run(
    const std::unordered_map<std::string, Tensor>& inputs
) const {
    auto outputs = run_all(inputs);
    if (outputs.empty()) throw std::runtime_error("ONNX model returned no outputs");
    return std::move(outputs.front());
}

OrtModel::~OrtModel() {
    delete impl_;
}

#else

struct OrtModel::Impl {};

OrtModel::OrtModel(
    const std::filesystem::path&,
    int,
    bool
) {
    impl_ = nullptr;
    throw std::runtime_error("PitcheeCore was built without ONNX Runtime");
}

Tensor OrtModel::run(
    const std::unordered_map<std::string, Tensor>&
) const {
    throw std::runtime_error("PitcheeCore was built without ONNX Runtime");
}

std::vector<Tensor> OrtModel::run_all(
    const std::unordered_map<std::string, Tensor>&
) const {
    throw std::runtime_error("PitcheeCore was built without ONNX Runtime");
}

OrtModel::~OrtModel() = default;

#endif

}  // namespace pitchee
