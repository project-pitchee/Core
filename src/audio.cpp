#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pitchee {
namespace {

constexpr int kResampleTaps = 32;
constexpr double kPi = 3.14159265358979323846;

double sinc(double value) {
    if (std::abs(value) < 1e-12) return 1.0;
    return std::sin(kPi * value) / (kPi * value);
}

double blackman_window(int index, int count) {
    const double ratio = static_cast<double>(index) / static_cast<double>(count - 1);
    return 0.42 - 0.5 * std::cos(2.0 * kPi * ratio)
        + 0.08 * std::cos(4.0 * kPi * ratio);
}

}  // namespace

std::vector<float> resample_mono(
    const float* interleaved,
    size_t sample_count,
    int channels,
    int source_rate,
    int target_rate
) {
    if (!interleaved || sample_count == 0 || channels <= 0 || source_rate <= 0
        || target_rate <= 0) {
        throw std::invalid_argument("invalid PCM buffer");
    }
    const size_t frame_count = sample_count / static_cast<size_t>(channels);
    if (frame_count == 0) return {};

    std::vector<double> mono(frame_count, 0.0);
    for (size_t frame = 0; frame < frame_count; ++frame) {
        double sum = 0.0;
        for (int channel = 0; channel < channels; ++channel) {
            sum += interleaved[frame * static_cast<size_t>(channels) + channel];
        }
        mono[frame] = sum / channels;
    }
    if (source_rate == target_rate) {
        std::vector<float> output(frame_count);
        std::transform(mono.begin(), mono.end(), output.begin(), [](double value) {
            return static_cast<float>(value);
        });
        return output;
    }

    const double ratio = static_cast<double>(target_rate) / source_rate;
    const size_t output_count = static_cast<size_t>(
        std::floor((static_cast<double>(frame_count - 1) / ratio)) + 1.0
    );
    std::vector<float> output(output_count);
    const double cutoff = 0.5 * std::min(1.0, ratio);

    for (size_t output_index = 0; output_index < output_count; ++output_index) {
        const double source_position = static_cast<double>(output_index) / ratio;
        const long center = static_cast<long>(std::floor(source_position));
        const int left = center - kResampleTaps / 2 + 1;
        const int right = left + kResampleTaps - 1;
        double weighted = 0.0;
        double weight_sum = 0.0;

        for (int source_index = left; source_index <= right; ++source_index) {
            if (source_index < 0 || source_index >= static_cast<int>(frame_count)) {
                continue;
            }
            const double distance = source_position - source_index;
            const double filter = 2.0 * cutoff * sinc(2.0 * cutoff * distance);
            const double window = blackman_window(
                source_index - left,
                kResampleTaps
            );
            const double weight = filter * window;
            weighted += mono[static_cast<size_t>(source_index)] * weight;
            weight_sum += weight;
        }
        output[output_index] = static_cast<float>(
            weight_sum > 1e-12 ? weighted / weight_sum : 0.0
        );
    }
    return output;
}

std::vector<float> concatenate_speech(
    const std::vector<float>& samples,
    const std::vector<VadSegment>& segments
) {
    std::vector<float> speech;
    for (const auto& segment : segments) {
        const size_t start = std::max(
            0,
            static_cast<int>(segment.source_start_seconds * kSampleRate)
        );
        const size_t end = std::min(
            samples.size(),
            static_cast<size_t>(segment.source_end_seconds * kSampleRate)
        );
        if (end > start) {
            speech.insert(speech.end(), samples.begin() + start, samples.begin() + end);
        }
    }
    return speech;
}

std::vector<float> crop_patch(const std::vector<float>& signal, size_t start) {
    std::vector<float> patch(kPatchSamples, 0.0f);
    if (start >= signal.size()) return patch;
    const size_t count = std::min(
        static_cast<size_t>(kPatchSamples),
        signal.size() - start
    );
    std::copy_n(signal.begin() + static_cast<std::ptrdiff_t>(start), count, patch.begin());
    return patch;
}

std::vector<size_t> sliding_patch_starts(size_t sample_count) {
    if (sample_count <= static_cast<size_t>(kPatchSamples)) return {0};
    std::vector<size_t> starts;
    for (size_t start = 0; start <= sample_count - kPatchSamples; start += kStrideSamples) {
        starts.push_back(start);
    }
    const size_t final_start = sample_count - kPatchSamples;
    if (starts.empty() || starts.back() != final_start) starts.push_back(final_start);
    return starts;
}

std::vector<size_t> naturalness_patch_starts(size_t sample_count) {
    const size_t patch_count = std::min<size_t>(
        24,
        std::max<size_t>(1, sample_count / kPatchSamples)
    );
    if (patch_count == 1) return {0};
    const size_t maximum_start = sample_count > kPatchSamples
        ? sample_count - kPatchSamples
        : 0;
    std::vector<size_t> starts;
    starts.reserve(patch_count);
    for (size_t index = 0; index < patch_count; ++index) {
        starts.push_back(
            static_cast<size_t>(
                static_cast<double>(index) * maximum_start
                / static_cast<double>(patch_count - 1)
            )
        );
    }
    return starts;
}

void normalize_l2(float* values, size_t size) {
    double sum = 0.0;
    for (size_t index = 0; index < size; ++index) {
        sum += static_cast<double>(values[index]) * values[index];
    }
    const double norm = std::sqrt(sum);
    if (norm <= 1e-12) return;
    for (size_t index = 0; index < size; ++index) {
        values[index] = static_cast<float>(values[index] / norm);
    }
}

}  // namespace pitchee
