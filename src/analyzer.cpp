#include "internal.hpp"
#include "ort_runtime.hpp"
#include "vad.hpp"
#include "wav_reader.hpp"

#include "pitchee/pitchee.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
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

void report_phase(
    pitchee_phase_callback_t callback,
    void* user_data,
    pitchee_analysis_phase_t phase
) {
    if (callback) callback(phase, user_data);
}

pitchee::PitchResult analyze_pitch(
    pitchee::OrtModel& model,
    const std::vector<float>& samples
) {
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
    return result;
}

std::vector<std::vector<float>> embed_waveforms(
    pitchee::OrtModel& frontend,
    pitchee::OrtModel& encoder,
    const std::vector<std::vector<float>>& waveforms
) {
    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(waveforms.size());
    for (size_t batch_start = 0;
         batch_start < waveforms.size();
         batch_start += pitchee::kEmbeddingBatchSize) {
        const size_t real_count = std::min<size_t>(
            pitchee::kEmbeddingBatchSize,
            waveforms.size() - batch_start
        );
        pitchee::Tensor input;
        input.shape = {
            pitchee::kEmbeddingBatchSize,
            pitchee::kPatchSamples,
        };
        input.values.reserve(
            static_cast<size_t>(pitchee::kEmbeddingBatchSize)
            * pitchee::kPatchSamples
        );
        for (size_t index = 0; index < pitchee::kEmbeddingBatchSize; ++index) {
            if (index < real_count) {
                input.values.insert(
                    input.values.end(),
                    waveforms[batch_start + index].begin(),
                    waveforms[batch_start + index].end()
                );
            } else {
                input.values.insert(
                    input.values.end(),
                    pitchee::kPatchSamples,
                    0.0f
                );
            }
        }

        auto features = frontend.run({{"waveforms", std::move(input)}});
        features.shape = {pitchee::kEmbeddingBatchSize, 152, 80};
        auto batch_embeddings = encoder.run({{"features", std::move(features)}});
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
            embeddings.emplace_back(
                start,
                start + pitchee::kEmbeddingDimensions
            );
        }
    }
    return embeddings;
}

std::vector<double> classify_embeddings(
    pitchee::OrtModel& model,
    const std::vector<std::vector<float>>& embeddings
) {
    std::vector<double> probabilities;
    probabilities.reserve(embeddings.size());
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
    if (!analyzer || !samples || sample_count == 0 || sample_rate <= 0
        || channels <= 0 || !out_json) {
        set_error(error_message, error_message_capacity, "invalid PCM argument");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    *out_json = nullptr;
    try {
        report_phase(phase_callback, user_data, PITCHEE_PHASE_LOADING_AUDIO);
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

        report_phase(phase_callback, user_data, PITCHEE_PHASE_ANALYZING);
        const auto pitch = analyze_pitch(*analyzer->swift_f0, signal);
        auto vad = analyzer->vad->detect(signal);
        if (vad.segments.empty()) {
            throw std::runtime_error("Silero VAD detected no speech");
        }
        const auto speech = pitchee::concatenate_speech(signal, vad.segments);
        if (speech.empty()) throw std::runtime_error("Silero VAD detected no speech");

        const auto starts = pitchee::sliding_patch_starts(speech.size());
        std::vector<std::vector<float>> speech_patches;
        speech_patches.reserve(starts.size());
        for (const size_t start : starts) {
            speech_patches.push_back(pitchee::crop_patch(speech, start));
        }
        const auto speech_embeddings = embed_waveforms(
            *analyzer->ecapa_frontend,
            *analyzer->ecapa,
            speech_patches
        );
        const auto probabilities = classify_embeddings(
            *analyzer->vfp_head,
            speech_embeddings
        );
        if (probabilities.empty()) throw std::runtime_error("no VFP windows");

        const auto natural_starts = pitchee::naturalness_patch_starts(signal.size());
        std::vector<std::vector<float>> natural_patches;
        natural_patches.reserve(natural_starts.size());
        for (const size_t start : natural_starts) {
            natural_patches.push_back(pitchee::crop_patch(signal, start));
        }
        const auto natural_embeddings = embed_waveforms(
            *analyzer->ecapa_frontend,
            *analyzer->ecapa,
            natural_patches
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
            std::min<size_t>(pitchee::kPatchSamples, speech.size())
        ) / pitchee::kSampleRate;
        const double speech_seconds = static_cast<double>(speech.size())
            / pitchee::kSampleRate;
        result.vfp_windows.reserve(starts.size());
        for (size_t index = 0; index < starts.size(); ++index) {
            const double speech_start_seconds = static_cast<double>(starts[index])
                / pitchee::kSampleRate;
            const double speech_end_seconds = std::min(
                speech_seconds,
                speech_start_seconds + result.window_duration_seconds
            );
            const auto [start_seconds, end_seconds] =
                pitchee::map_speech_range_to_source(
                    vad.segments,
                    speech_start_seconds,
                    speech_end_seconds
                );
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
            std::min<size_t>(pitchee::kPatchSamples, signal.size())
        ) / pitchee::kSampleRate;
        const std::vector<float> global_naturalness_std(
            features.begin() + pitchee::kEmbeddingDimensions,
            features.end()
        );
        result.naturalness_windows.reserve(natural_starts.size());
        for (size_t index = 0; index < natural_starts.size(); ++index) {
            const double start_seconds = static_cast<double>(natural_starts[index])
                / pitchee::kSampleRate;
            const double window_duration = static_cast<double>(
                std::min(
                    static_cast<size_t>(pitchee::kPatchSamples),
                    signal.size() - natural_starts[index]
                )
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
            window.end_seconds = std::min(
                source_seconds,
                start_seconds + window_duration
            );
            window.score = analyzer->naturalness->score(window_features);
            result.naturalness_windows.push_back(window);
        }

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

        const std::string json = pitchee::result_to_json(result);
        auto* output = new char[json.size() + 1];
        std::memcpy(output, json.c_str(), json.size() + 1);
        *out_json = output;
        report_phase(phase_callback, user_data, PITCHEE_PHASE_COMPLETED);
        return PITCHEE_SUCCESS;
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
    if (!analyzer || !wav_path || !out_json) {
        set_error(error_message, error_message_capacity, "invalid WAV argument");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    *out_json = nullptr;
    try {
        report_phase(phase_callback, user_data, PITCHEE_PHASE_LOADING_AUDIO);
        const pitchee::WavData wav = pitchee::read_wav(wav_path);
        return pitchee_analyzer_analyze_pcm(
            analyzer,
            wav.samples.data(),
            wav.samples.size(),
            wav.sample_rate,
            wav.channels,
            phase_callback,
            user_data,
            out_json,
            error_message,
            error_message_capacity
        );
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return status_for_exception(error);
    }
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
