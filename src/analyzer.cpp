#include "internal.hpp"
#include "ort_runtime.hpp"
#include "vad.hpp"
#include "wav_reader.hpp"

#include "pitchee/pitchee.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

struct pitchee_analyzer_t {
    std::filesystem::path model_directory;
    int intra_op_threads = 2;
    bool use_coreml = true;
    std::unique_ptr<pitchee::OrtModel> ecapa_frontend;
    std::unique_ptr<pitchee::OrtModel> ecapa;
    std::unique_ptr<pitchee::OrtModel> vfp_head;
    std::unique_ptr<pitchee::OrtModel> swift_f0;
    std::unique_ptr<pitchee::VadDetector> vad;
    std::unique_ptr<pitchee::NaturalnessModel> naturalness;
};

namespace {

constexpr double kF0WindowSeconds = 0.1;

struct ProgressReporter {
    pitchee_phase_callback_t phase_callback = nullptr;
    void* phase_user_data = nullptr;
    pitchee_progress_callback_t progress_callback = nullptr;
    void* progress_user_data = nullptr;

    void phase(pitchee_analysis_phase_t value) const {
        if (phase_callback) phase_callback(value, phase_user_data);
    }

    void progress(
        pitchee_progress_stage_t stage,
        uint64_t completed,
        uint64_t total
    ) const {
        if (!progress_callback) return;
        pitchee_progress_t progress{};
        progress.stage = stage;
        progress.completed = std::min(completed, total);
        progress.total = total;
        progress.fraction = total > 0
            ? static_cast<double>(progress.completed) / total
            : (stage == PITCHEE_PROGRESS_STAGE_COMPLETED ? 1.0 : 0.0);
        progress_callback(&progress, progress_user_data);
    }
};

void set_error(char* target, size_t capacity, const std::string& message) {
    if (!target || capacity == 0) return;
    const size_t count = std::min(capacity - 1, message.size());
    std::memcpy(target, message.data(), count);
    target[count] = '\0';
}

pitchee_status_t status_for_exception(const std::exception& error) {
    const std::string message = error.what();
    if (message.find("without ONNX Runtime") != std::string::npos) {
        return PITCHEE_ERROR_ORT_UNAVAILABLE;
    }
    if (message.find("Silero VAD detected no speech") != std::string::npos) {
        return PITCHEE_ERROR_NO_SPEECH;
    }
    if (message.find("unsupported WAV") != std::string::npos) {
        return PITCHEE_ERROR_UNSUPPORTED_FORMAT;
    }
    if (message.find("WAV") != std::string::npos
        || message.find("unable to open") != std::string::npos) {
        return PITCHEE_ERROR_IO;
    }
    if (message.find("model") != std::string::npos
        || message.find("ONNX") != std::string::npos) {
        return PITCHEE_ERROR_MODEL;
    }
    return PITCHEE_ERROR_INTERNAL;
}

pitchee::PitchResult analyze_pitch(
    pitchee::OrtModel& model,
    const std::vector<float>& samples,
    const ProgressReporter& reporter
) {
    reporter.progress(PITCHEE_PROGRESS_STAGE_ANALYZING_F0, 0, 1);
    std::vector<float> input = samples;
    if (input.size() < 256) input.resize(256, 0.0f);
    pitchee::Tensor tensor;
    tensor.shape = {1, static_cast<int64_t>(input.size())};
    tensor.values = std::move(input);
    auto outputs = model.run_all({{"input_audio", std::move(tensor)}});
    if (outputs.size() < 2 || outputs[0].values.size() != outputs[1].values.size()) {
        throw std::runtime_error("SwiftF0 returned invalid output");
    }

    pitchee::PitchResult result;
    result.pitch_hz = std::move(outputs[0].values);
    result.confidence = std::move(outputs[1].values);
    result.timestamps.resize(result.pitch_hz.size());
    result.voicing.resize(result.pitch_hz.size(), 0);
    std::vector<float> voiced_pitch;
    const size_t f0_window_samples = static_cast<size_t>(
        kF0WindowSeconds * pitchee::kSampleRate
    );
    const size_t f0_window_count = std::max<size_t>(
        1,
        (samples.size() + f0_window_samples - 1) / f0_window_samples
    );
    std::vector<double> f0_window_sums(f0_window_count, 0.0);
    std::vector<size_t> f0_window_counts(f0_window_count, 0);
    for (size_t index = 0; index < result.pitch_hz.size(); ++index) {
        result.timestamps[index] = static_cast<float>(
            (static_cast<double>(index * 256) + 127.5) / 16000.0
        );
        const bool voiced =
            result.confidence[index] > 0.9f
            && result.pitch_hz[index] >= 75.0f
            && result.pitch_hz[index] <= 600.0f;
        result.voicing[index] = voiced ? 1 : 0;
        if (voiced) {
            voiced_pitch.push_back(result.pitch_hz[index]);
            const size_t window_index = std::min(
                f0_window_count - 1,
                static_cast<size_t>(
                    result.timestamps[index] / kF0WindowSeconds
                )
            );
            f0_window_sums[window_index] += result.pitch_hz[index];
            ++f0_window_counts[window_index];
        }
    }

    const double total_seconds = static_cast<double>(samples.size())
        / pitchee::kSampleRate;
    result.windows.reserve(f0_window_count);
    for (size_t index = 0; index < f0_window_count; ++index) {
        pitchee::F0Window window;
        window.start_seconds = index * kF0WindowSeconds;
        window.end_seconds = std::min(
            total_seconds,
            (index + 1) * kF0WindowSeconds
        );
        if (f0_window_counts[index] > 0) {
            window.has_f0 = true;
            window.f0_hz = f0_window_sums[index]
                / static_cast<double>(f0_window_counts[index]);
            ++result.voiced_window_count;
        }
        result.windows.push_back(window);
    }

    if (!voiced_pitch.empty()) {
        const double mean = std::accumulate(
            voiced_pitch.begin(),
            voiced_pitch.end(),
            0.0
        ) / voiced_pitch.size();
        double variance = 0.0;
        for (const float value : voiced_pitch) {
            const double delta = value - mean;
            variance += delta * delta;
        }
        result.has_mean_f0 = true;
        result.mean_f0_hz = mean;
        result.standard_deviation_f0_hz = std::sqrt(variance / voiced_pitch.size());
        result.voiced_frame_count = static_cast<int>(voiced_pitch.size());
    }
    reporter.progress(PITCHEE_PROGRESS_STAGE_ANALYZING_F0, 1, 1);
    return result;
}

std::vector<std::vector<float>> embed_waveforms(
    pitchee::OrtModel& frontend,
    pitchee::OrtModel& encoder,
    const std::vector<std::vector<float>>& waveforms,
    const ProgressReporter& reporter,
    pitchee_progress_stage_t stage
) {
    std::vector<std::vector<float>> embeddings(waveforms.size());
    std::map<size_t, std::vector<size_t>> groups;
    for (size_t index = 0; index < waveforms.size(); ++index) {
        if (waveforms[index].empty()) continue;
        groups[waveforms[index].size()].push_back(index);
    }
    if (groups.empty()) throw std::runtime_error("no waveform windows");

    reporter.progress(stage, 0, waveforms.size());
    size_t completed = 0;
    for (const auto& [window_samples, indices] : groups) {
        for (size_t batch_start = 0;
             batch_start < indices.size();
             batch_start += pitchee::kEmbeddingBatchSize) {
            const size_t real_count = std::min<size_t>(
                pitchee::kEmbeddingBatchSize,
                indices.size() - batch_start
            );
            pitchee::Tensor input;
            input.shape = {
                pitchee::kEmbeddingBatchSize,
                static_cast<int64_t>(window_samples),
            };
            input.values.reserve(
                static_cast<size_t>(pitchee::kEmbeddingBatchSize)
                * window_samples
            );
            for (size_t index = 0; index < pitchee::kEmbeddingBatchSize; ++index) {
                if (index < real_count) {
                    const auto& waveform = waveforms[indices[batch_start + index]];
                    input.values.insert(
                        input.values.end(),
                        waveform.begin(),
                        waveform.end()
                    );
                } else {
                    input.values.insert(input.values.end(), window_samples, 0.0f);
                }
            }

            auto features = frontend.run({{"waveforms", std::move(input)}});
            auto batch_embeddings = encoder.run(
                {{"features", std::move(features)}}
            );
            if (batch_embeddings.values.size()
                < static_cast<size_t>(pitchee::kEmbeddingBatchSize)
                    * pitchee::kEmbeddingDimensions) {
                throw std::runtime_error("ECAPA returned invalid embeddings");
            }
            for (size_t index = 0; index < real_count; ++index) {
                const auto start = batch_embeddings.values.begin()
                    + static_cast<std::ptrdiff_t>(
                        index * pitchee::kEmbeddingDimensions
                    );
                embeddings[indices[batch_start + index]].assign(
                    start,
                    start + pitchee::kEmbeddingDimensions
                );
            }
            completed += real_count;
            reporter.progress(stage, completed, waveforms.size());
        }
    }
    return embeddings;
}

std::vector<double> classify_embeddings(
    pitchee::OrtModel& model,
    const std::vector<std::vector<float>>& embeddings,
    const ProgressReporter& reporter
) {
    std::vector<double> probabilities;
    probabilities.reserve(embeddings.size());
    reporter.progress(
        PITCHEE_PROGRESS_STAGE_CLASSIFYING_VFP_WINDOWS,
        0,
        embeddings.size()
    );
    for (size_t batch_start = 0;
         batch_start < embeddings.size();
         batch_start += pitchee::kEmbeddingBatchSize) {
        const size_t real_count = std::min<size_t>(
            pitchee::kEmbeddingBatchSize,
            embeddings.size() - batch_start
        );
        pitchee::Tensor input;
        input.shape = {
            pitchee::kEmbeddingBatchSize,
            pitchee::kEmbeddingDimensions,
        };
        input.values.reserve(
            static_cast<size_t>(pitchee::kEmbeddingBatchSize)
            * pitchee::kEmbeddingDimensions
        );
        for (size_t index = 0; index < pitchee::kEmbeddingBatchSize; ++index) {
            if (index < real_count) {
                input.values.insert(
                    input.values.end(),
                    embeddings[batch_start + index].begin(),
                    embeddings[batch_start + index].end()
                );
            } else {
                input.values.insert(
                    input.values.end(),
                    pitchee::kEmbeddingDimensions,
                    0.0f
                );
            }
        }
        auto output = model.run({{"embeddings", std::move(input)}});
        if (output.values.size() < real_count) {
            throw std::runtime_error("VFP classifier returned invalid output");
        }
        for (size_t index = 0; index < real_count; ++index) {
            probabilities.push_back(output.values[index]);
        }
        reporter.progress(
            PITCHEE_PROGRESS_STAGE_CLASSIFYING_VFP_WINDOWS,
            std::min(
                embeddings.size(),
                batch_start + pitchee::kEmbeddingBatchSize
            ),
            embeddings.size()
        );
    }
    return probabilities;
}

std::vector<float> naturalness_features(
    const std::vector<std::vector<float>>& embeddings
) {
    if (embeddings.empty()) throw std::invalid_argument("no naturalness embeddings");
    std::vector<std::vector<float>> normalized = embeddings;
    for (auto& embedding : normalized) {
        pitchee::normalize_l2(embedding.data(), embedding.size());
    }

    std::vector<float> raw_mean(pitchee::kEmbeddingDimensions, 0.0f);
    for (const auto& embedding : normalized) {
        for (size_t index = 0; index < raw_mean.size(); ++index) {
            raw_mean[index] += embedding[index];
        }
    }
    for (float& value : raw_mean) value /= normalized.size();

    std::vector<float> mean = raw_mean;
    pitchee::normalize_l2(mean.data(), mean.size());
    std::vector<double> variance(pitchee::kEmbeddingDimensions, 0.0);
    for (const auto& embedding : normalized) {
        for (size_t index = 0; index < variance.size(); ++index) {
            const double delta = static_cast<double>(embedding[index]) - raw_mean[index];
            variance[index] += delta * delta;
        }
    }
    std::vector<float> features = mean;
    features.reserve(pitchee::kEmbeddingDimensions * 2);
    for (double value : variance) {
        features.push_back(static_cast<float>(
            std::sqrt(value / normalized.size())
        ));
    }
    return features;
}

}  // namespace

extern "C" {

const char* pitchee_core_version(void) {
    return "0.1.0";
}

pitchee_status_t pitchee_analyzer_create(
    const char* model_directory,
    const pitchee_analyzer_options_t* options,
    pitchee_analyzer_t** out_analyzer,
    char* error_message,
    size_t error_message_capacity
) {
    if (!model_directory || !out_analyzer) {
        set_error(error_message, error_message_capacity, "invalid argument");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto analyzer = std::make_unique<pitchee_analyzer_t>();
        analyzer->model_directory = model_directory;
        analyzer->intra_op_threads = options && options->intra_op_threads > 0
            ? options->intra_op_threads
            : 2;
        analyzer->use_coreml = !options || options->use_coreml != 0;
        const auto directory = analyzer->model_directory;
        analyzer->ecapa_frontend = std::make_unique<pitchee::OrtModel>(
            directory / "ECAPAFrontend.onnx",
            analyzer->intra_op_threads,
            false
        );
        analyzer->ecapa = std::make_unique<pitchee::OrtModel>(
            directory / "ECAPA.onnx",
            analyzer->intra_op_threads,
            analyzer->use_coreml
        );
        analyzer->vfp_head = std::make_unique<pitchee::OrtModel>(
            directory / "VFPHead.onnx",
            analyzer->intra_op_threads,
            false
        );
        analyzer->swift_f0 = std::make_unique<pitchee::OrtModel>(
            directory / "SwiftF0.onnx",
            analyzer->intra_op_threads,
            false
        );
        analyzer->vad = std::make_unique<pitchee::VadDetector>(
            directory / "SileroVAD.onnx",
            analyzer->intra_op_threads
        );
        analyzer->naturalness = std::make_unique<pitchee::NaturalnessModel>(
            directory / "Naturalness.onnx",
            analyzer->intra_op_threads
        );
        *out_analyzer = analyzer.release();
        return PITCHEE_SUCCESS;
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return status_for_exception(error);
    }
}

void pitchee_analyzer_destroy(pitchee_analyzer_t* analyzer) {
    delete analyzer;
}

pitchee_status_t analyze_pcm_impl(
    pitchee_analyzer_t* analyzer,
    const float* samples,
    size_t sample_count,
    int32_t sample_rate,
    int32_t channels,
    const ProgressReporter& reporter,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
) {
    if (!analyzer || !samples || sample_count == 0 || sample_rate <= 0
        || channels <= 0 || !out_json) {
        set_error(error_message, error_message_capacity, "invalid PCM argument");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    *out_json = nullptr;
    try {
        reporter.phase(PITCHEE_PHASE_LOADING_AUDIO);
        reporter.progress(PITCHEE_PROGRESS_STAGE_LOADING_AUDIO, 1, 1);
        reporter.progress(PITCHEE_PROGRESS_STAGE_RESAMPLING_AUDIO, 0, 1);
        std::vector<float> signal = pitchee::resample_mono(
            samples,
            sample_count,
            channels,
            sample_rate,
            pitchee::kSampleRate
        );
        const double source_seconds =
            static_cast<double>(signal.size()) / pitchee::kSampleRate;
        if (std::isfinite(pitchee::kMaximumSeconds)) {
            const size_t maximum_samples = static_cast<size_t>(
                pitchee::kMaximumSeconds * pitchee::kSampleRate
            );
            if (signal.size() > maximum_samples) signal.resize(maximum_samples);
        }
        if (signal.empty()) throw std::invalid_argument("empty audio");
        reporter.progress(PITCHEE_PROGRESS_STAGE_RESAMPLING_AUDIO, 1, 1);

        reporter.phase(PITCHEE_PHASE_ANALYZING);
        const auto pitch = analyze_pitch(*analyzer->swift_f0, signal, reporter);
        auto vad = analyzer->vad->detect(
            signal,
            [&reporter](size_t completed, size_t total) {
                reporter.progress(
                    PITCHEE_PROGRESS_STAGE_DETECTING_SPEECH,
                    completed,
                    total
                );
            }
        );
        if (vad.segments.empty()) {
            throw std::runtime_error("Silero VAD detected no speech");
        }
        reporter.progress(PITCHEE_PROGRESS_STAGE_PREPARING_VFP_WINDOWS, 0, 1);
        const auto source_windows = pitchee::native_speech_windows(
            signal.size(),
            vad.segments
        );
        if (source_windows.empty()) {
            throw std::runtime_error("Silero VAD detected no speech");
        }
        reporter.progress(PITCHEE_PROGRESS_STAGE_PREPARING_VFP_WINDOWS, 1, 1);

        std::vector<std::vector<float>> vfp_patches;
        vfp_patches.reserve(source_windows.size());
        for (const auto& window : source_windows) {
            vfp_patches.push_back(pitchee::crop_window(signal, window));
        }
        const auto speech_embeddings = embed_waveforms(
            *analyzer->ecapa_frontend,
            *analyzer->ecapa,
            vfp_patches,
            reporter,
            PITCHEE_PROGRESS_STAGE_EXTRACTING_VFP_EMBEDDINGS
        );
        const auto probabilities = classify_embeddings(
            *analyzer->vfp_head,
            speech_embeddings,
            reporter
        );
        if (probabilities.empty()) throw std::runtime_error("no VFP windows");

        reporter.progress(
            PITCHEE_PROGRESS_STAGE_PREPARING_NATURALNESS_WINDOWS,
            0,
            1
        );
        // Naturalness uses the exact VFP window set on the source timeline.
        const auto& natural_windows = source_windows;
        std::vector<std::vector<float>> natural_patches;
        natural_patches.reserve(natural_windows.size());
        for (const auto& window : natural_windows) {
            natural_patches.push_back(pitchee::crop_window(signal, window));
        }
        const auto natural_embeddings = embed_waveforms(
            *analyzer->ecapa_frontend,
            *analyzer->ecapa,
            natural_patches,
            reporter,
            PITCHEE_PROGRESS_STAGE_EXTRACTING_NATURALNESS_EMBEDDINGS
        );
        reporter.progress(
            PITCHEE_PROGRESS_STAGE_PREPARING_NATURALNESS_WINDOWS,
            1,
            1
        );
        const auto features = naturalness_features(natural_embeddings);

        pitchee::AnalysisResult result;
        const double mean_probability = std::accumulate(
            probabilities.begin(),
            probabilities.end(),
            0.0
        ) / probabilities.size();
        result.vfp_standard_score = std::max(
            0.0,
            std::min(100.0, mean_probability * 100.0)
        );
        result.window_count = static_cast<int>(probabilities.size());
        result.window_duration_seconds = static_cast<double>(
            pitchee::kPatchSamples
        ) / pitchee::kSampleRate;
        result.vfp_windows.reserve(source_windows.size());
        for (size_t index = 0; index < source_windows.size(); ++index) {
            const double start_seconds = static_cast<double>(
                source_windows[index].start
            ) / pitchee::kSampleRate;
            const double end_seconds = static_cast<double>(
                source_windows[index].start + source_windows[index].length
            ) / pitchee::kSampleRate;
            pitchee::VfpWindow window;
            window.start_seconds = start_seconds;
            window.end_seconds = end_seconds;
            window.vfp_standard_score = std::max(
                0.0,
                std::min(100.0, probabilities[index] * 100.0)
            );
            result.vfp_windows.push_back(window);
        }

        result.naturalness_window_duration_seconds = static_cast<double>(
            pitchee::kPatchSamples
        ) / pitchee::kSampleRate;
        const std::vector<float> global_naturalness_std(
            features.begin() + pitchee::kEmbeddingDimensions,
            features.end()
        );
        result.naturalness_windows.reserve(natural_windows.size());
        reporter.progress(
            PITCHEE_PROGRESS_STAGE_SCORING_NATURALNESS_WINDOWS,
            0,
            natural_windows.size()
        );
        for (size_t index = 0; index < natural_windows.size(); ++index) {
            const double start_seconds = static_cast<double>(
                natural_windows[index].start
            ) / pitchee::kSampleRate;
            const double end_seconds = static_cast<double>(
                natural_windows[index].start + natural_windows[index].length
            ) / pitchee::kSampleRate;
            std::vector<float> window_features = natural_embeddings[index];
            pitchee::normalize_l2(
                window_features.data(),
                window_features.size()
            );
            window_features.insert(
                window_features.end(),
                global_naturalness_std.begin(),
                global_naturalness_std.end()
            );
            pitchee::NaturalnessWindow window;
            window.start_seconds = start_seconds;
            window.end_seconds = end_seconds;
            window.score = analyzer->naturalness->score(window_features);
            result.naturalness_windows.push_back(window);
            reporter.progress(
                PITCHEE_PROGRESS_STAGE_SCORING_NATURALNESS_WINDOWS,
                index + 1,
                natural_windows.size()
            );
        }

        reporter.progress(PITCHEE_PROGRESS_STAGE_CALCULATING_SCORES, 0, 1);
        result.source_sample_rate = sample_rate;
        result.source_channels = channels;
        result.source_seconds = source_seconds;
        result.analyzed_seconds = static_cast<double>(signal.size()) / pitchee::kSampleRate;
        result.has_f0 = pitch.has_mean_f0;
        result.f0_mean_hz = pitch.mean_f0_hz;
        result.f0_standard_deviation_hz = pitch.standard_deviation_f0_hz;
        result.voiced_frame_count = pitch.voiced_frame_count;
        result.voiced_window_count = pitch.voiced_window_count;
        result.f0_windows = pitch.windows;
        result.naturalness_score = analyzer->naturalness->score(features);
        result.score = pitchee::calculate_composite_score(
            result.vfp_standard_score,
            result.naturalness_score,
            result.has_f0,
            result.f0_mean_hz
        );
        result.vad = vad;
        reporter.progress(PITCHEE_PROGRESS_STAGE_CALCULATING_SCORES, 1, 1);

        reporter.progress(PITCHEE_PROGRESS_STAGE_SERIALIZING_RESULT, 0, 1);
        const std::string json = pitchee::result_to_json(result);
        auto* output = new char[json.size() + 1];
        std::memcpy(output, json.c_str(), json.size() + 1);
        *out_json = output;
        reporter.progress(PITCHEE_PROGRESS_STAGE_SERIALIZING_RESULT, 1, 1);
        reporter.phase(PITCHEE_PHASE_COMPLETED);
        reporter.progress(PITCHEE_PROGRESS_STAGE_COMPLETED, 1, 1);
        return PITCHEE_SUCCESS;
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return status_for_exception(error);
    }
}

pitchee_status_t pitchee_analyzer_analyze_pcm(
    pitchee_analyzer_t* analyzer,
    const float* samples,
    size_t sample_count,
    int32_t sample_rate,
    int32_t channels,
    pitchee_phase_callback_t phase_callback,
    void* user_data,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
) {
    const ProgressReporter reporter{
        phase_callback,
        user_data,
        nullptr,
        nullptr,
    };
    return analyze_pcm_impl(
        analyzer,
        samples,
        sample_count,
        sample_rate,
        channels,
        reporter,
        out_json,
        error_message,
        error_message_capacity
    );
}

pitchee_status_t pitchee_analyzer_analyze_pcm_with_progress(
    pitchee_analyzer_t* analyzer,
    const float* samples,
    size_t sample_count,
    int32_t sample_rate,
    int32_t channels,
    pitchee_progress_callback_t progress_callback,
    void* user_data,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
) {
    const ProgressReporter reporter{
        nullptr,
        nullptr,
        progress_callback,
        user_data,
    };
    return analyze_pcm_impl(
        analyzer,
        samples,
        sample_count,
        sample_rate,
        channels,
        reporter,
        out_json,
        error_message,
        error_message_capacity
    );
}

pitchee_status_t analyze_wav_impl(
    pitchee_analyzer_t* analyzer,
    const char* wav_path,
    const ProgressReporter& reporter,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
) {
    if (!analyzer || !wav_path || !out_json) {
        set_error(error_message, error_message_capacity, "invalid WAV argument");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    *out_json = nullptr;
    try {
        reporter.progress(PITCHEE_PROGRESS_STAGE_LOADING_AUDIO, 0, 1);
        const pitchee::WavData wav = pitchee::read_wav(wav_path);
        return analyze_pcm_impl(
            analyzer,
            wav.samples.data(),
            wav.samples.size(),
            wav.sample_rate,
            wav.channels,
            reporter,
            out_json,
            error_message,
            error_message_capacity
        );
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return status_for_exception(error);
    }
}

pitchee_status_t pitchee_analyzer_analyze_wav_file(
    pitchee_analyzer_t* analyzer,
    const char* wav_path,
    pitchee_phase_callback_t phase_callback,
    void* user_data,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
) {
    const ProgressReporter reporter{
        phase_callback,
        user_data,
        nullptr,
        nullptr,
    };
    return analyze_wav_impl(
        analyzer,
        wav_path,
        reporter,
        out_json,
        error_message,
        error_message_capacity
    );
}

pitchee_status_t pitchee_analyzer_analyze_wav_file_with_progress(
    pitchee_analyzer_t* analyzer,
    const char* wav_path,
    pitchee_progress_callback_t progress_callback,
    void* user_data,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
) {
    const ProgressReporter reporter{
        nullptr,
        nullptr,
        progress_callback,
        user_data,
    };
    return analyze_wav_impl(
        analyzer,
        wav_path,
        reporter,
        out_json,
        error_message,
        error_message_capacity
    );
}

pitchee_status_t pitchee_composite_score(
    double vfp_standard_score,
    double naturalness_score,
    double f0_hz,
    int32_t has_f0,
    pitchee_composite_score_t* out_score
) {
    if (!out_score) return PITCHEE_ERROR_INVALID_ARGUMENT;
    const auto score = pitchee::calculate_composite_score(
        vfp_standard_score,
        naturalness_score,
        has_f0 != 0,
        f0_hz
    );
    out_score->base_score = score.base_score;
    out_score->final_score = score.final_score;
    out_score->has_score_cap = score.has_score_cap ? 1 : 0;
    out_score->score_cap = score.score_cap;
    out_score->score_limited = score.score_limited ? 1 : 0;
    out_score->score_boosted = score.score_boosted ? 1 : 0;
    std::memset(out_score->score_rule, 0, sizeof(out_score->score_rule));
    std::memcpy(
        out_score->score_rule,
        score.score_rule.c_str(),
        std::min(
            sizeof(out_score->score_rule) - 1,
            score.score_rule.size()
        )
    );
    return PITCHEE_SUCCESS;
}

void pitchee_string_free(char* value) {
    delete[] value;
}

}  // extern "C"
