#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace pitchee {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSwrCutoff = 0.97;
constexpr int kSwrBaseFilterSize = 32;
constexpr int kSwrMaximumPhases = 1024;
constexpr double kSwrKaiserBeta = 9.0;

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

std::vector<double> swr_phase_filter(
    int tap_count,
    int center,
    int phase,
    int phase_count,
    double factor,
    double normalization
) {
    std::vector<double> coefficients(static_cast<size_t>(tap_count));
    for (int index = 0; index < tap_count; ++index) {
        const double distance = static_cast<double>(index - center)
            - static_cast<double>(phase) / phase_count;
        const double x = kPi * distance * factor;
        double coefficient = std::abs(x) < 1e-12
            ? 1.0
            : std::sin(x) / x;
        const double window_position = 2.0 * distance / tap_count;
        const double window_argument = std::max(
            0.0,
            1.0 - window_position * window_position
        );
        coefficient *= bessel_i0(
            kSwrKaiserBeta * std::sqrt(window_argument)
        );
        coefficients[static_cast<size_t>(index)] = coefficient / normalization;
    }
    return coefficients;
}

float quantize_pcm16(double value) {
    // The site and training pipeline use PCM16 WAV; naturalness features are
    // sensitive enough that even sub-LSB float differences matter.
    const double scaled = std::max(
        -32768.0,
        std::min(32767.0, std::round(value * 32768.0))
    );
    return static_cast<float>(scaled / 32768.0);
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
            return quantize_pcm16(value);
        });
        return output;
    }

    const double factor = std::min(
        1.0,
        static_cast<double>(target_rate) * kSwrCutoff / source_rate
    );
    int tap_count = static_cast<int>(std::ceil(kSwrBaseFilterSize / factor));
    if (tap_count > 1 && tap_count % 2 != 0) ++tap_count;
    const int center = (tap_count - 1) / 2;
    const int divisor = std::gcd(source_rate, target_rate);
    const int exact_phase_count = target_rate / divisor;
    const int phase_count = std::min(exact_phase_count, kSwrMaximumPhases);

    std::vector<std::vector<double>> phase_filters(
        static_cast<size_t>(phase_count + 1)
    );
    const auto first_filter = swr_phase_filter(
        tap_count,
        center,
        0,
        phase_count,
        factor,
        1.0
    );
    const double normalization = std::accumulate(
        first_filter.begin(),
        first_filter.end(),
        0.0
    );
    for (int phase = 0; phase <= phase_count; ++phase) {
        const int wrapped_phase = phase == phase_count ? 0 : phase;
        phase_filters[static_cast<size_t>(phase)] = swr_phase_filter(
            tap_count,
            center,
            wrapped_phase,
            phase_count,
            factor,
            normalization
        );
    }

    const size_t output_count = static_cast<size_t>(std::ceil(
        static_cast<long double>(frame_count) * target_rate / source_rate
    ));
    std::vector<float> output(output_count);
    for (size_t output_index = 0; output_index < output_count; ++output_index) {
        const double source_position = static_cast<double>(output_index)
            * source_rate / target_rate;
        const long base_index = static_cast<long>(std::floor(source_position));
        const double phase_position = (
            source_position - static_cast<double>(base_index)
        ) * phase_count;
        const int phase = std::min(
            static_cast<int>(std::floor(phase_position)),
            phase_count - 1
        );
        const double interpolation = phase_position - phase;
        const auto& left = phase_filters[static_cast<size_t>(phase)];
        const auto& right = phase_filters[static_cast<size_t>(phase + 1)];
        double value = 0.0;
        for (int tap = 0; tap < tap_count; ++tap) {
            const long source_index = base_index - center + tap;
            if (source_index < 0
                || source_index >= static_cast<long>(frame_count)) {
                continue;
            }
            const double coefficient = left[static_cast<size_t>(tap)] * (
                1.0 - interpolation
            ) + right[static_cast<size_t>(tap)] * interpolation;
            value += mono[static_cast<size_t>(source_index)] * coefficient;
        }
        output[output_index] = quantize_pcm16(value);
    }
    return output;
}

std::vector<SampleWindow> native_speech_windows(
    size_t sample_count,
    const std::vector<VadSegment>& segments,
    size_t stride_samples
) {
    std::vector<SampleWindow> windows;
    if (stride_samples == 0) stride_samples = 1;
    for (const auto& segment : segments) {
        const size_t first = std::min(
            sample_count,
            static_cast<size_t>(segment.source_start_seconds * kSampleRate)
        );
        const size_t last = std::min(
            sample_count,
            static_cast<size_t>(segment.source_end_seconds * kSampleRate)
        );
        if (last <= first) continue;
        const size_t length = last - first;
        if (length <= static_cast<size_t>(kPatchSamples)) {
            windows.push_back({first, length});
            continue;
        }

        const size_t maximum_start = length - kPatchSamples;
        size_t last_start = 0;
        bool added = false;
        for (size_t start = 0; start <= maximum_start; start += stride_samples) {
            windows.push_back({first + start, kPatchSamples});
            last_start = start;
            added = true;
        }
        if (!added || last_start != maximum_start) {
            windows.push_back({first + maximum_start, kPatchSamples});
        }
    }
    return windows;
}

std::vector<float> crop_window(
    const std::vector<float>& signal,
    const SampleWindow& window
) {
    if (window.start >= signal.size() || window.length == 0) return {};
    const size_t count = std::min(window.length, signal.size() - window.start);
    return std::vector<float>(
        signal.begin() + static_cast<std::ptrdiff_t>(window.start),
        signal.begin() + static_cast<std::ptrdiff_t>(window.start + count)
    );
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
