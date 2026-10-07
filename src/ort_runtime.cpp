#include "ort_runtime.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

#ifdef PITCHEE_ENABLE_ORT
#include <onnxruntime_cxx_api.h>
#if defined(__APPLE__)
#include <coreml_provider_factory.h>
#endif
#endif

#ifdef PITCHEE_ENABLE_NCNN
#include "ncnn_pitchee_layers.hpp"

#include <ncnn/mat.h>
#include <ncnn/net.h>
#ifdef PITCHEE_NCNN_VULKAN
#include <ncnn/gpu.h>
#endif
#ifdef __ANDROID__
#include <android/log.h>
#endif
#endif

namespace pitchee {

#ifdef PITCHEE_ENABLE_NCNN
namespace {

struct NcnnModelSpec {
    std::filesystem::path param_path;
    std::filesystem::path bin_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

void ncnn_log(const char* message) {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "PitcheeCore", "%s", message);
#endif
}

std::optional<NcnnModelSpec> ncnn_spec_for(const std::filesystem::path& path) {
    const std::string filename = path.filename().string();
    NcnnModelSpec spec;
    if (filename == "ECAPAFrontend.onnx") {
        spec.input_names = {"waveforms"};
        spec.output_names = {"features"};
    } else if (filename == "ECAPA.onnx") {
        spec.input_names = {"in0"};
        spec.output_names = {"out0"};
    } else if (filename == "VFPHead.onnx") {
        spec.input_names = {"embeddings"};
        spec.output_names = {"probabilities"};
    } else if (filename == "Naturalness.onnx") {
        spec.input_names = {"features"};
        spec.output_names = {"naturalness_score"};
    } else if (filename == "SwiftF0.onnx") {
        spec.input_names = {"input_audio"};
        spec.output_names = {"pitch_hz", "confidence"};
    } else if (filename == "SileroVAD.onnx") {
        spec.input_names = {"input", "state"};
        spec.output_names = {"output", "stateN"};
    } else {
        return std::nullopt;
    }

    spec.param_path = path.parent_path() / (path.stem().string() + ".ncnn.param");
    spec.bin_path = path.parent_path() / (path.stem().string() + ".ncnn.bin");
    if (!std::filesystem::exists(spec.param_path)
        || !std::filesystem::exists(spec.bin_path)) {
        return std::nullopt;
    }
    return spec;
}

#ifdef PITCHEE_NCNN_VULKAN
bool ncnn_gpu_requested() {
    const char* gpu = std::getenv("PITCHEE_NCNN_GPU");
    if (!gpu) return false;
    return std::string_view(gpu) != "0";
}
#endif

bool ncnn_requested_for(const std::filesystem::path& path) {
#ifdef __ANDROID__
    (void)path;
    return true;
#else
    if (path.filename().string() == "SileroVAD.onnx") {
        const char* vad_backend = std::getenv("PITCHEE_VAD_BACKEND");
        if (vad_backend && *vad_backend) {
            return std::string_view(vad_backend) == "ncnn"
                || std::string_view(vad_backend) == "hybrid";
        }
    }
    const char* backend = std::getenv("PITCHEE_BACKEND");
    if (backend && *backend) {
        return std::string_view(backend) == "ncnn"
            || std::string_view(backend) == "hybrid";
    }
    return false;
#endif
}

#ifdef PITCHEE_NCNN_VULKAN
bool initialize_ncnn_gpu() {
    static std::once_flag once;
    static bool available = false;
    std::call_once(once, [] {
        const int status = ncnn::create_gpu_instance();
        available = status == 0 && ncnn::get_gpu_count() > 0;
        const std::string message = available
            ? "ncnn Vulkan GPU instance ready"
            : "ncnn Vulkan GPU initialization failed";
        ncnn_log(message.c_str());
    });
    return available;
}
#endif

ncnn::Mat tensor_to_ncnn(const Tensor& tensor) {
    if (tensor.shape.empty()) {
        throw std::runtime_error("ncnn input has no shape");
    }

    ncnn::Mat input;
    if (tensor.shape.size() == 1) {
        input = ncnn::Mat(static_cast<int>(tensor.shape[0]));
    } else if (tensor.shape.size() == 2) {
        input = ncnn::Mat(
            static_cast<int>(tensor.shape[1]),
            1,
            static_cast<int>(tensor.shape[0])
        );
    } else if (tensor.shape.size() == 3) {
        input = ncnn::Mat(
            static_cast<int>(tensor.shape[2]),
            static_cast<int>(tensor.shape[1]),
            static_cast<int>(tensor.shape[0])
        );
    } else {
        throw std::runtime_error("ncnn input rank is unsupported");
    }

    const size_t input_elements = static_cast<size_t>(input.w)
        * static_cast<size_t>(input.h)
        * static_cast<size_t>(input.c)
        * static_cast<size_t>(input.elempack);
    if (!input.data || input_elements != tensor.values.size()) {
        throw std::runtime_error("ncnn input shape does not match values");
    }
    std::memcpy(input.data, tensor.values.data(), tensor.values.size() * sizeof(float));
    return input;
}

Tensor ncnn_to_tensor(const ncnn::Mat& output) {
    ncnn::Mat unpacked = output;
    if (unpacked.elempack != 1) {
        ncnn::Mat converted;
        ncnn::convert_packing(unpacked, converted, 1);
        unpacked = converted;
    }

    ncnn::Mat float_output = unpacked;
    if (unpacked.elembits() == 16) {
        ncnn::Mat converted;
        ncnn::cast_float16_to_float32(unpacked, converted);
        float_output = converted;
    }

    Tensor tensor;
    if (float_output.c == 1 && float_output.h == 1) {
        tensor.shape = {float_output.w};
    } else if (float_output.c == 1) {
        tensor.shape = {float_output.h, float_output.w};
    } else {
        tensor.shape = {float_output.c, float_output.h, float_output.w};
    }
    const float* data = static_cast<const float*>(float_output.data);
    const size_t output_elements = static_cast<size_t>(float_output.w)
        * static_cast<size_t>(float_output.h)
        * static_cast<size_t>(float_output.c)
        * static_cast<size_t>(float_output.elempack);
    tensor.values.assign(data, data + output_elements);
    return tensor;
}

}  // namespace
#endif

bool ort_available() {
#ifdef PITCHEE_ENABLE_ORT
    return true;
#else
    return false;
#endif
}

enum class ModelBackend { Ort, Ncnn };

struct OrtModel::Impl {
    explicit Impl(
        const std::filesystem::path& path,
        int intra_op_threads,
        bool use_coreml
    )
#ifdef PITCHEE_ENABLE_ORT
        : environment(ORT_LOGGING_LEVEL_WARNING, "PitcheeCore"),
          memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
    {
        (void)use_coreml;
#ifdef PITCHEE_ENABLE_NCNN
        if (ncnn_requested_for(path)) {
            const auto spec = ncnn_spec_for(path);
            if (spec) {
                option.num_threads = intra_op_threads;
                option.use_packing_layout = true;
                option.use_winograd_convolution = true;
                option.use_sgemm_convolution = true;
                option.use_fp16_packed = false;
                option.use_fp16_storage = false;
                option.use_fp16_arithmetic = false;
                option.use_bf16_storage = false;
                option.use_vulkan_compute = false;
#ifdef PITCHEE_NCNN_VULKAN
                if (ncnn_gpu_requested() && initialize_ncnn_gpu()) {
                    option.use_vulkan_compute = true;
                    option.vulkan_device_index = 0;
                    option.use_fp16_packed = true;
                    option.use_fp16_storage = true;
                    option.use_fp16_arithmetic = true;
                    option.use_cooperative_matrix = true;
                    option.use_subgroup_ops = true;
                    option.use_tensor_storage = true;
                }
#endif
                ncnn_net = std::make_unique<ncnn::Net>();
                ncnn_net->opt = option;
                register_ncnn_layers(*ncnn_net);
                if (ncnn_net->load_param(spec->param_path.string().c_str()) == 0
                    && ncnn_net->load_model(spec->bin_path.string().c_str()) == 0) {
                    ncnn_input_names = spec->input_names;
                    ncnn_output_names = spec->output_names;
                    backend = ModelBackend::Ncnn;
                    const std::string message = "ncnn backend: "
                        + path.filename().string();
                    ncnn_log(message.c_str());
                } else {
                    ncnn_net.reset();
                }
            }
        }
#endif
        if (backend == ModelBackend::Ncnn) return;

#ifdef PITCHEE_ENABLE_ORT
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
#else
        (void)path;
        (void)intra_op_threads;
        throw std::runtime_error("PitcheeCore was built without ONNX Runtime");
#endif
    }

#ifdef PITCHEE_ENABLE_ORT
    Ort::Env environment;
    Ort::MemoryInfo memory_info;
    std::unique_ptr<Ort::Session> session;
#endif
    ModelBackend backend = ModelBackend::Ort;
#ifdef PITCHEE_ENABLE_NCNN
    ncnn::Option option;
    std::unique_ptr<ncnn::Net> ncnn_net;
    std::vector<std::string> ncnn_input_names;
    std::vector<std::string> ncnn_output_names;
#endif
};

OrtModel::OrtModel(
    const std::filesystem::path& path,
    int intra_op_threads,
    bool use_coreml
) : impl_(new Impl(path, intra_op_threads, use_coreml)) {}

std::vector<Tensor> OrtModel::run_all(
    const std::unordered_map<std::string, Tensor>& inputs
) const {
    if (!impl_) {
        throw std::runtime_error("model runtime is unavailable");
    }
#ifdef PITCHEE_ENABLE_NCNN
    if (impl_->backend == ModelBackend::Ncnn) {
        if (!impl_->ncnn_net || impl_->ncnn_input_names.empty()
            || impl_->ncnn_output_names.empty()) {
            throw std::runtime_error("ncnn model input is invalid");
        }
        auto run_ncnn = [&](const std::unordered_map<std::string, Tensor>& model_inputs) {
            ncnn::Extractor extractor = impl_->ncnn_net->create_extractor();
            for (const std::string& input_name : impl_->ncnn_input_names) {
                auto input_it = model_inputs.find(input_name);
                if (input_it == model_inputs.end()
                    && impl_->ncnn_input_names.size() == 1) {
                    input_it = model_inputs.begin();
                }
                if (input_it == model_inputs.end()) {
                    throw std::runtime_error("ncnn model input is missing: " + input_name);
                }
                ncnn::Mat input = tensor_to_ncnn(input_it->second);
                if (extractor.input(input_name.c_str(), input) != 0) {
                    throw std::runtime_error("ncnn model rejected input: " + input_name);
                }
            }
            std::vector<Tensor> outputs;
            outputs.reserve(impl_->ncnn_output_names.size());
            for (const std::string& output_name : impl_->ncnn_output_names) {
                ncnn::Mat output;
                if (extractor.extract(output_name.c_str(), output) != 0) {
                    throw std::runtime_error("ncnn model returned no output: " + output_name);
                }
                outputs.push_back(ncnn_to_tensor(output));
            }
            return outputs;
        };

        if (inputs.size() == 1
            && inputs.begin()->second.shape.size() >= 2
            && inputs.begin()->second.shape[0] > 1) {
            const std::string input_name = inputs.begin()->first;
            const Tensor& tensor = inputs.begin()->second;
            const int64_t batch = tensor.shape[0];
            const size_t row_size = tensor.values.size()
                / static_cast<size_t>(batch);
            std::vector<int64_t> row_shape = {1};
            row_shape.insert(
                row_shape.end(),
                tensor.shape.begin() + 1,
                tensor.shape.end()
            );
            Tensor batched;
            batched.shape = {batch};
            batched.values.reserve(tensor.values.size());
            std::vector<Tensor> row_outputs(static_cast<size_t>(batch));
            auto run_index = [&](int64_t index) {
                Tensor row;
                row.shape = row_shape;
                row.values.assign(
                    tensor.values.begin()
                        + static_cast<std::ptrdiff_t>(index * row_size),
                    tensor.values.begin()
                        + static_cast<std::ptrdiff_t>((index + 1) * row_size)
                );
                std::unordered_map<std::string, Tensor> single_inputs;
                single_inputs.emplace(input_name, std::move(row));
                auto single_outputs = run_ncnn(single_inputs);
                if (single_outputs.empty()) {
                    throw std::runtime_error("ncnn model returned no outputs");
                }
                row_outputs[static_cast<size_t>(index)] =
                    std::move(single_outputs.front());
            };

            int concurrency = 1;
#ifdef PITCHEE_NCNN_VULKAN
            if (impl_->ncnn_net->opt.use_vulkan_compute) {
                ncnn::VulkanDevice* gpu_device = ncnn::get_gpu_device(0);
                if (gpu_device) {
                    const char* override = std::getenv(
                        "PITCHEE_NCNN_GPU_CONCURRENCY"
                    );
                    const int requested = override
                        ? std::atoi(override)
                        : static_cast<int>(gpu_device->info.compute_queue_count());
                    concurrency = std::max(1, std::min<int>(
                        static_cast<int>(batch),
                        requested
                    ));
                }
            }
#endif
            if (concurrency == 1) {
                for (int64_t index = 0; index < batch; ++index) {
                    run_index(index);
                }
            } else {
                std::atomic<int64_t> next{0};
                const int worker_count = std::min<int64_t>(
                    concurrency,
                    batch
                );
                std::vector<std::thread> workers;
                workers.reserve(static_cast<size_t>(worker_count));
                for (int worker = 0; worker < worker_count; ++worker) {
                    workers.emplace_back([&]() {
                        for (;;) {
                            const int64_t index = next.fetch_add(1);
                            if (index >= batch) break;
                            run_index(index);
                        }
                    });
                }
                for (auto& worker : workers) worker.join();
            }

            for (int64_t index = 0; index < batch; ++index) {
                const Tensor& row_output = row_outputs[static_cast<size_t>(index)];
                if (index == 0) {
                    if (row_output.shape.size() == 1
                        && row_output.shape[0] == 1) {
                        // Scalar outputs are represented as a vector of batch
                        // probabilities rather than [batch, 1].
                    } else {
                        batched.shape.insert(
                            batched.shape.end(),
                            row_output.shape.begin(),
                            row_output.shape.end()
                        );
                    }
                }
                batched.values.insert(
                    batched.values.end(),
                    row_output.values.begin(),
                    row_output.values.end()
                );
            }
            return {std::move(batched)};
        }
        return run_ncnn(inputs);
    }
#endif
#ifdef PITCHEE_ENABLE_ORT
    if (!impl_->session) {
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
#else
    (void)inputs;
    throw std::runtime_error("PitcheeCore was built without ONNX Runtime");
#endif
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

}  // namespace pitchee
