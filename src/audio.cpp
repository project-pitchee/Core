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

std::pair<double, double> map_speech_range_to_source(
    const std::vector<VadSegment>& segments,
    double speech_start_seconds,
    double speech_end_seconds
) {
    if (segments.empty()) return {0.0, 0.0};

    const double total_speech_seconds = segments.back().speech_end_seconds;
    const double start = std::max(
        0.0,
        std::min(total_speech_seconds, speech_start_seconds)
    );
    const double end = std::max(
        start,
        std::min(total_speech_seconds, speech_end_seconds)
    );
    constexpr double kBoundaryEpsilon = 1e-9;
    double source_start = segments.back().source_start_seconds;
    double source_end = segments.back().source_end_seconds;

    // At a boundary, a window start belongs to the following speech segment;
    // a window end belongs to the preceding one. This preserves the full
    // original-time span even when the window crosses removed silence.
    for (const auto& segment : segments) {
        if (start < segment.speech_end_seconds - kBoundaryEpsilon) {
            const double offset = std::max(
                0.0,
                start - segment.speech_start_seconds
            );
            source_start = segment.source_start_seconds + offset;
            break;
        }
    }
    for (const auto& segment : segments) {
        if (end <= segment.speech_end_seconds + kBoundaryEpsilon) {
            const double offset = std::max(
                0.0,
                std::min(
                    segment.speech_end_seconds - segment.speech_start_seconds,
                    end - segment.speech_start_seconds
                )
            );
            source_end = segment.source_start_seconds + offset;
            break;
        }
    }
    return {source_start, std::max(source_start, source_end)};
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
