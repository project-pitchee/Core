#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace pitchee {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kKaiserBeta = 5.0;

double sinc(double value) {
    if (std::abs(value) < 1e-12) return 1.0;
    return std::sin(kPi * value) / (kPi * value);
}

double bessel_i0(double value) {
    double sum = 1.0;
    double term = 1.0;
    const double half = value / 2.0;
    for (int order = 1; order < 64; ++order) {
        term *= (half / order) * (half / order);
        sum += term;
        if (term < sum * 1e-16) break;
    }
    return sum;
}

std::vector<double> kaiser_window(int count, double beta) {
    std::vector<double> window(static_cast<size_t>(count));
    const double denominator = bessel_i0(beta);
    for (int index = 0; index < count; ++index) {
        const double ratio = (
            2.0 * index / static_cast<double>(count - 1)
        ) - 1.0;
        window[static_cast<size_t>(index)] = bessel_i0(
            beta * std::sqrt(std::max(0.0, 1.0 - ratio * ratio))
        ) / denominator;
    }
    return window;
}

std::vector<double> prototype_filter(int up, int down) {
    const int maximum_rate = std::max(up, down);
    const int half_length = 10 * maximum_rate;
    const int tap_count = 2 * half_length + 1;
    const double cutoff = 1.0 / maximum_rate;
    const double center = half_length;
    std::vector<double> coefficients(static_cast<size_t>(tap_count));
    const auto window = kaiser_window(tap_count, kKaiserBeta);
    for (int index = 0; index < tap_count; ++index) {
        coefficients[static_cast<size_t>(index)] = cutoff * sinc(
            cutoff * (index - center)
        ) * window[static_cast<size_t>(index)];
    }

    const double sum = std::accumulate(
        coefficients.begin(),
        coefficients.end(),
        0.0
    );
    if (std::abs(sum) > 1e-15) {
        for (double& coefficient : coefficients) coefficient /= sum;
    }
    for (double& coefficient : coefficients) {
        coefficient *= static_cast<double>(up);
    }
    return coefficients;
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

    const int divisor = std::gcd(source_rate, target_rate);
    const int up = target_rate / divisor;
    const int down = source_rate / divisor;
    const std::vector<double> filter = prototype_filter(up, down);
    const int half_length = static_cast<int>((filter.size() - 1) / 2);
    const size_t upsampled_count = frame_count * static_cast<size_t>(up);
    const size_t output_count = (
        upsampled_count + static_cast<size_t>(down) - 1
    ) / static_cast<size_t>(down);
    std::vector<float> output(output_count);

    for (size_t output_index = 0; output_index < output_count; ++output_index) {
        double value = 0.0;
        for (size_t tap = 0; tap < filter.size(); ++tap) {
            const int source_position = static_cast<int>(output_index)
                * down + half_length - static_cast<int>(tap);
            if (source_position < 0 || source_position % up != 0) continue;
            const int source_index = source_position / up;
            if (source_index < 0 || source_index >= static_cast<int>(frame_count)) {
                continue;
            }
            value += mono[static_cast<size_t>(source_index)] * filter[tap];
        }
        output[output_index] = static_cast<float>(value);
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
