#include "vad.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pitchee {
namespace {

constexpr int kWindowSize = 512;
constexpr int kContextSize = 64;
constexpr int kStateSize = 2 * 128;
constexpr double kThreshold = 0.30;
constexpr double kNegativeThreshold = 0.60;
constexpr double kMinimumPeriodicity = 0.50;
constexpr double kMinimumVoicedRatio = 0.12;
constexpr int kPeriodicityFrameSamples = 400;
constexpr int kPeriodicityHopSamples = 160;

struct RawSegment {
    int start_sample = 0;
    int end_sample = 0;
};

double frame_periodicity(const float* frame, size_t size) {
    if (size != kPeriodicityFrameSamples) return 0.0;
    double mean = 0.0;
    for (size_t index = 0; index < size; ++index) mean += frame[index];
    mean /= size;

    std::vector<double> centered(size);
    double energy = 0.0;
    for (size_t index = 0; index < size; ++index) {
        centered[index] = frame[index] - mean;
        energy += centered[index] * centered[index];
    }
    if (std::sqrt(energy / size) < 1e-4) return 0.0;

    constexpr double pi = 3.14159265358979323846;
    for (size_t index = 0; index < size; ++index) {
        const double hamming = 0.54 - 0.46 * std::cos(
            2.0 * pi * static_cast<double>(index)
            / static_cast<double>(size - 1)
        );
        centered[index] *= hamming;
    }
    energy = 0.0;
    for (const double value : centered) energy += value * value;
    if (energy <= 1e-10) return 0.0;

    double maximum = -1e30;
    const int first_lag = 32;
    const int last_lag = 213;
    for (int lag = first_lag; lag < last_lag; ++lag) {
        double correlation = 0.0;
        for (size_t index = 0; index + lag < centered.size(); ++index) {
            correlation += centered[index] * centered[index + lag];
        }
        maximum = std::max(maximum, correlation / energy);
    }
    return std::isfinite(maximum) ? maximum : 0.0;
}

}  // namespace

VadDetector::VadDetector(
    const std::filesystem::path& model_path,
    int intra_op_threads
) : model_(std::make_unique<OrtModel>(model_path, intra_op_threads, false)) {}

VadResult VadDetector::detect(const std::vector<float>& samples) const {
    if (samples.empty()) throw std::invalid_argument("empty audio");
    std::vector<float> state(kStateSize, 0.0f);
    std::vector<float> context(kContextSize, 0.0f);
    std::vector<double> probabilities;
    probabilities.reserve((samples.size() + kWindowSize - 1) / kWindowSize);

    for (size_t start = 0; start < samples.size(); start += kWindowSize) {
        Tensor input;
        input.shape = {1, kWindowSize + kContextSize};
        input.values.reserve(kWindowSize + kContextSize);
        input.values.insert(input.values.end(), context.begin(), context.end());
        for (int index = 0; index < kWindowSize; ++index) {
            const size_t source = start + static_cast<size_t>(index);
            input.values.push_back(source < samples.size() ? samples[source] : 0.0f);
        }
        Tensor state_tensor;
        state_tensor.shape = {2, 1, 128};
        state_tensor.values = state;
        Tensor sample_rate;
        sample_rate.shape = {};
        sample_rate.data_type = Tensor::DataType::Int64;
        sample_rate.int64_values = {16000};
        std::vector<float> next_context(
            input.values.end() - kContextSize,
            input.values.end()
        );

        auto outputs = model_->run_all({
            {"input", std::move(input)},
            {"state", std::move(state_tensor)},
            {"sr", std::move(sample_rate)},
        });
        if (outputs.size() < 2 || outputs[0].values.empty()) {
            throw std::runtime_error("Silero VAD returned invalid output");
        }
        probabilities.push_back(outputs[0].values[0]);
        state = std::move(outputs[1].values);
        if (state.size() != kStateSize) {
            throw std::runtime_error("Silero VAD returned invalid state");
        }
        context = std::move(next_context);
    }

    const int minimum_speech_samples = kSampleRate * 80 / 1000;
    const int minimum_silence_samples = kSampleRate * 80 / 1000;
    const int speech_padding_samples = kSampleRate * 20 / 1000;
    bool triggered = false;
    int current_start = 0;
    int temporary_end = 0;
    std::vector<RawSegment> raw_segments;

    for (size_t index = 0; index < probabilities.size(); ++index) {
        const double probability = probabilities[index];
        const int current_sample = static_cast<int>(index) * kWindowSize;
        if (probability >= kThreshold && temporary_end != 0) temporary_end = 0;
        if (probability >= kThreshold && !triggered) {
            triggered = true;
            current_start = current_sample;
            continue;
        }
        if (probability < kNegativeThreshold && triggered) {
            if (temporary_end == 0) temporary_end = current_sample;
            const int silence_duration = current_sample - temporary_end;
            if (silence_duration < minimum_silence_samples) continue;
            if (temporary_end - current_start > minimum_speech_samples) {
                raw_segments.push_back({current_start, temporary_end});
            }
            current_start = 0;
            temporary_end = 0;
            triggered = false;
        }
    }
    if (triggered && static_cast<int>(samples.size()) - current_start > minimum_speech_samples) {
        raw_segments.push_back({current_start, static_cast<int>(samples.size())});
    }

    for (size_t index = 0; index < raw_segments.size(); ++index) {
        if (index == 0) {
            raw_segments[index].start_sample = std::max(
                0,
                raw_segments[index].start_sample - speech_padding_samples
            );
        }
        if (index + 1 < raw_segments.size()) {
            const int silence_duration = raw_segments[index + 1].start_sample
                - raw_segments[index].end_sample;
            if (silence_duration < 2 * speech_padding_samples) {
                raw_segments[index].end_sample += silence_duration / 2;
                raw_segments[index + 1].start_sample = std::max(
                    0,
                    raw_segments[index + 1].start_sample - silence_duration / 2
                );
            } else {
                raw_segments[index].end_sample = std::min(
                    static_cast<int>(samples.size()),
                    raw_segments[index].end_sample + speech_padding_samples
                );
                raw_segments[index + 1].start_sample = std::max(
                    0,
                    raw_segments[index + 1].start_sample - speech_padding_samples
                );
            }
        } else {
            raw_segments[index].end_sample = std::min(
                static_cast<int>(samples.size()),
                raw_segments[index].end_sample + speech_padding_samples
            );
        }
    }

    VadResult result;
    result.input_seconds = static_cast<double>(samples.size()) / kSampleRate;
    result.silero_segment_count = static_cast<int>(raw_segments.size());
    double speech_offset = 0.0;
    for (const auto& raw : raw_segments) {
        const int first = std::max(0, raw.start_sample);
        const int last = std::min(static_cast<int>(samples.size()), raw.end_sample);
        if (last <= first) {
            ++result.discarded_breath_like_count;
            continue;
        }

        std::vector<double> periodicities;
        for (int offset = first; offset + kPeriodicityFrameSamples <= last;
             offset += kPeriodicityHopSamples) {
            periodicities.push_back(
                frame_periodicity(
                    samples.data() + offset,
                    kPeriodicityFrameSamples
                )
            );
        }
        std::vector<int> voiced;
        for (size_t index = 0; index < periodicities.size(); ++index) {
            if (periodicities[index] >= kMinimumPeriodicity) {
                voiced.push_back(static_cast<int>(index));
            }
        }
        const double voiced_ratio = static_cast<double>(voiced.size())
            / std::max<size_t>(1, periodicities.size());
        if (voiced.empty() || voiced_ratio < kMinimumVoicedRatio) {
            ++result.discarded_breath_like_count;
            continue;
        }

        const int trimmed_first = first + std::max(
            0,
            voiced.front() * kPeriodicityHopSamples - 80
        );
        const int trimmed_last = first + std::min(
            last - first,
            (voiced.back() + 1) * kPeriodicityHopSamples + 80
        );
        if (trimmed_first > first || trimmed_last < last) {
            ++result.trimmed_segment_count;
        }
        const double duration = static_cast<double>(trimmed_last - trimmed_first) / kSampleRate;
        result.segments.push_back({
            static_cast<double>(trimmed_first) / kSampleRate,
            static_cast<double>(trimmed_last) / kSampleRate,
            speech_offset,
            speech_offset + duration,
        });
        speech_offset += duration;
    }
    result.speech_seconds = speech_offset;
    return result;
}

}  // namespace pitchee
