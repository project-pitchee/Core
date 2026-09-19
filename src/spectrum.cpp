#include "pitchee/pitchee.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSampleRate = 16000.0;
constexpr double kMinimumDb = -120.0;
constexpr float kEpsilon = 1e-12f;

void set_error(char* target, size_t capacity, const char* message) {
    if (!target || capacity == 0) return;
    const size_t count = std::min(capacity - 1, std::strlen(message));
    std::memcpy(target, message, count);
    target[count] = '\0';
}

bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

void fft(std::vector<double>& real, std::vector<double>& imaginary) {
    const size_t size = real.size();
    for (size_t index = 1, reversed = 0; index < size; ++index) {
        size_t bit = size >> 1;
        for (; reversed & bit; bit >>= 1) reversed ^= bit;
        reversed ^= bit;
        if (index < reversed) {
            std::swap(real[index], real[reversed]);
            std::swap(imaginary[index], imaginary[reversed]);
        }
    }

    for (size_t length = 2; length <= size; length <<= 1) {
        const double angle = -2.0 * kPi / static_cast<double>(length);
        const double step_real = std::cos(angle);
        const double step_imaginary = std::sin(angle);
        for (size_t start = 0; start < size; start += length) {
            double phase_real = 1.0;
            double phase_imaginary = 0.0;
            const size_t half = length >> 1;
            for (size_t offset = 0; offset < half; ++offset) {
                const size_t even = start + offset;
                const size_t odd = even + half;
                const double odd_real = real[odd] * phase_real
                    - imaginary[odd] * phase_imaginary;
                const double odd_imaginary = real[odd] * phase_imaginary
                    + imaginary[odd] * phase_real;
                real[odd] = real[even] - odd_real;
                imaginary[odd] = imaginary[even] - odd_imaginary;
                real[even] += odd_real;
                imaginary[even] += odd_imaginary;
                const double next_real = phase_real * step_real
                    - phase_imaginary * step_imaginary;
                phase_imaginary = phase_real * step_imaginary
                    + phase_imaginary * step_real;
                phase_real = next_real;
            }
        }
    }
}

std::vector<float> hann_window(int size) {
    std::vector<float> window(static_cast<size_t>(size));
    for (int index = 0; index < size; ++index) {
        window[static_cast<size_t>(index)] = static_cast<float>(
            0.5 - 0.5 * std::cos(
                2.0 * kPi * static_cast<double>(index)
                / static_cast<double>(size - 1)
            )
        );
    }
    return window;
}

}  // namespace

struct pitchee_spectrum_t {
    int fft_size = 2048;
    int hop_samples = 256;
    int min_hz = 40;
    int max_hz = 8000;
    size_t first_bin = 0;
    size_t last_bin = 0;
    pitchee_spectrum_value_t value_type = PITCHEE_SPECTRUM_DBFS;
    float smoothing = 0.65f;
    float bin_hz = 0.0f;
    double window_sum = 1.0;
    std::vector<float> window;
    std::vector<float> buffer;
    std::vector<float> previous_linear;
    std::vector<float> linear_magnitudes;
    std::vector<float> output_values;
    std::vector<double> real;
    std::vector<double> imaginary;
    size_t buffer_start_sample = 0;
    size_t total_samples = 0;
    size_t next_frame_end_sample = 0;
    bool has_previous = false;
};

namespace {

float convert_value(float linear, pitchee_spectrum_value_t value_type) {
    if (value_type == PITCHEE_SPECTRUM_POWER) return linear * linear;
    if (value_type == PITCHEE_SPECTRUM_DBFS) {
        return static_cast<float>(std::max(
            kMinimumDb,
            20.0 * std::log10(std::max(linear, kEpsilon))
        ));
    }
    return linear;
}

bool process_frame(
    pitchee_spectrum_t& spectrum,
    size_t window_start,
    pitchee_spectrum_callback_t callback,
    void* user_data
) {
    if (window_start < spectrum.buffer_start_sample) {
        throw std::runtime_error("spectrum buffer underflow");
    }
    const size_t buffer_offset = window_start - spectrum.buffer_start_sample;
    if (buffer_offset + static_cast<size_t>(spectrum.fft_size)
        > spectrum.buffer.size()) {
        return false;
    }

    std::fill(spectrum.imaginary.begin(), spectrum.imaginary.end(), 0.0);
    for (int index = 0; index < spectrum.fft_size; ++index) {
        spectrum.real[static_cast<size_t>(index)] = static_cast<double>(
            spectrum.buffer[buffer_offset + static_cast<size_t>(index)]
            * spectrum.window[static_cast<size_t>(index)]
        );
    }
    fft(spectrum.real, spectrum.imaginary);

    const size_t bin_count = spectrum.last_bin - spectrum.first_bin + 1;
    spectrum.linear_magnitudes.resize(bin_count);
    spectrum.output_values.resize(bin_count);
    if (!spectrum.has_previous) {
        spectrum.previous_linear.assign(bin_count, 0.0f);
    }

    double weighted_frequency = 0.0;
    double magnitude_sum = 0.0;
    double log_sum = 0.0;
    double energy_sum = 0.0;
    size_t peak_index = 0;
    float peak_magnitude = -1.0f;
    std::vector<double> energies(bin_count, 0.0);
    for (size_t offset = 0; offset < bin_count; ++offset) {
        const size_t bin = spectrum.first_bin + offset;
        double magnitude = std::sqrt(
            spectrum.real[bin] * spectrum.real[bin]
            + spectrum.imaginary[bin] * spectrum.imaginary[bin]
        ) / spectrum.window_sum;
        if (bin > 0 && bin < static_cast<size_t>(spectrum.fft_size / 2)) {
            magnitude *= 2.0;
        }
        const float current = static_cast<float>(magnitude);
        const float smoothed = spectrum.has_previous
            ? spectrum.smoothing * spectrum.previous_linear[offset]
                + (1.0f - spectrum.smoothing) * current
            : current;
        spectrum.previous_linear[offset] = smoothed;
        spectrum.linear_magnitudes[offset] = smoothed;
        spectrum.output_values[offset] = convert_value(
            smoothed,
            spectrum.value_type
        );

        const double frequency = static_cast<double>(bin) * spectrum.bin_hz;
        weighted_frequency += frequency * smoothed;
        magnitude_sum += smoothed;
        log_sum += std::log(std::max(
            static_cast<double>(smoothed),
            static_cast<double>(kEpsilon)
        ));
        energies[offset] = static_cast<double>(smoothed) * smoothed;
        energy_sum += energies[offset];
        if (smoothed > peak_magnitude) {
            peak_magnitude = smoothed;
            peak_index = offset;
        }
    }
    spectrum.has_previous = true;

    const double centroid_hz = magnitude_sum > kEpsilon
        ? weighted_frequency / magnitude_sum
        : 0.0;
    const double geometric_mean = std::exp(
        log_sum / static_cast<double>(bin_count)
    );
    const double arithmetic_mean = magnitude_sum / static_cast<double>(bin_count);
    const double flatness = arithmetic_mean > kEpsilon
        ? geometric_mean / arithmetic_mean
        : 0.0;
    double cumulative = 0.0;
    double rolloff_hz = spectrum.first_bin * spectrum.bin_hz;
    const double rolloff_target = energy_sum * 0.85;
    for (size_t offset = 0; offset < bin_count; ++offset) {
        cumulative += energies[offset];
        if (cumulative >= rolloff_target) {
            rolloff_hz = static_cast<double>(spectrum.first_bin + offset)
                * spectrum.bin_hz;
            break;
        }
    }

    if (callback) {
        pitchee_spectrum_frame_t frame{};
        frame.timestamp_seconds = (
            static_cast<double>(window_start)
            + static_cast<double>(spectrum.fft_size - 1) / 2.0
        ) / kSampleRate;
        frame.magnitudes = spectrum.output_values.data();
        frame.bin_count = bin_count;
        frame.first_bin_index = spectrum.first_bin;
        frame.bin_hz = spectrum.bin_hz;
        frame.peak_hz = static_cast<float>(
            static_cast<double>(spectrum.first_bin + peak_index)
            * spectrum.bin_hz
        );
        frame.centroid_hz = static_cast<float>(centroid_hz);
        frame.rolloff_hz = static_cast<float>(rolloff_hz);
        frame.flatness = static_cast<float>(flatness);
        callback(&frame, user_data);
    }
    return true;
}

}  // namespace

extern "C" {

pitchee_status_t pitchee_spectrum_create(
    const pitchee_spectrum_options_t* options,
    pitchee_spectrum_t** out_spectrum,
    char* error_message,
    size_t error_message_capacity
) {
    if (!out_spectrum) {
        set_error(error_message, error_message_capacity, "invalid spectrum argument");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    *out_spectrum = nullptr;
    try {
        auto spectrum = std::make_unique<pitchee_spectrum_t>();
        if (options) {
            if (options->fft_size > 0) spectrum->fft_size = options->fft_size;
            if (options->hop_samples > 0) spectrum->hop_samples = options->hop_samples;
            if (options->min_hz > 0) spectrum->min_hz = options->min_hz;
            if (options->max_hz > 0) spectrum->max_hz = options->max_hz;
            spectrum->value_type = options->value_type;
            if (std::isfinite(options->smoothing) && options->smoothing >= 0.0f) {
                spectrum->smoothing = std::min(0.99f, options->smoothing);
            }
        }
        if (!is_power_of_two(spectrum->fft_size)
            || spectrum->fft_size < 256 || spectrum->fft_size > 8192
            || spectrum->hop_samples < 1
            || spectrum->hop_samples > spectrum->fft_size
            || spectrum->min_hz < 0 || spectrum->max_hz > 8000
            || spectrum->min_hz >= spectrum->max_hz
            || spectrum->value_type < PITCHEE_SPECTRUM_AMPLITUDE
            || spectrum->value_type > PITCHEE_SPECTRUM_DBFS) {
            throw std::invalid_argument("invalid spectrum options");
        }
        spectrum->bin_hz = static_cast<float>(
            kSampleRate / spectrum->fft_size
        );
        spectrum->first_bin = static_cast<size_t>(std::ceil(
            spectrum->min_hz / spectrum->bin_hz
        ));
        spectrum->last_bin = std::min<size_t>(
            static_cast<size_t>(spectrum->fft_size / 2),
            static_cast<size_t>(std::floor(
                spectrum->max_hz / spectrum->bin_hz
            ))
        );
        if (spectrum->first_bin > spectrum->last_bin) {
            throw std::invalid_argument("spectrum range contains no FFT bins");
        }
        spectrum->window = hann_window(spectrum->fft_size);
        spectrum->window_sum = 0.0;
        for (float value : spectrum->window) spectrum->window_sum += value;
        spectrum->real.resize(static_cast<size_t>(spectrum->fft_size));
        spectrum->imaginary.resize(static_cast<size_t>(spectrum->fft_size));
        spectrum->next_frame_end_sample = static_cast<size_t>(spectrum->fft_size);
        *out_spectrum = spectrum.release();
        return PITCHEE_SUCCESS;
    } catch (const std::invalid_argument&) {
        set_error(error_message, error_message_capacity, "invalid spectrum options");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return PITCHEE_ERROR_INTERNAL;
    }
}

pitchee_status_t pitchee_spectrum_process(
    pitchee_spectrum_t* spectrum,
    const float* samples,
    size_t sample_count,
    pitchee_spectrum_callback_t frame_callback,
    void* user_data,
    size_t* out_frame_count,
    char* error_message,
    size_t error_message_capacity
) {
    if (out_frame_count) *out_frame_count = 0;
    if (!spectrum || (!samples && sample_count > 0)) {
        set_error(error_message, error_message_capacity, "invalid spectrum argument");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    if (sample_count == 0) return PITCHEE_SUCCESS;

    try {
        size_t read_offset = 0;
        size_t frame_count = 0;
        while (read_offset < sample_count) {
            const size_t block = std::min(
                sample_count - read_offset,
                static_cast<size_t>(spectrum->fft_size)
            );
            spectrum->buffer.insert(
                spectrum->buffer.end(),
                samples + read_offset,
                samples + read_offset + block
            );
            spectrum->total_samples += block;
            read_offset += block;

            while (spectrum->next_frame_end_sample <= spectrum->total_samples) {
                const size_t window_start = spectrum->next_frame_end_sample
                    - static_cast<size_t>(spectrum->fft_size);
                if (!process_frame(
                        *spectrum,
                        window_start,
                        frame_callback,
                        user_data
                    )) {
                    break;
                }
                spectrum->next_frame_end_sample += static_cast<size_t>(
                    spectrum->hop_samples
                );
                ++frame_count;
            }

            if (spectrum->buffer.size() > static_cast<size_t>(spectrum->fft_size)) {
                const size_t drop = spectrum->buffer.size()
                    - static_cast<size_t>(spectrum->fft_size);
                spectrum->buffer.erase(
                    spectrum->buffer.begin(),
                    spectrum->buffer.begin() + static_cast<std::ptrdiff_t>(drop)
                );
                spectrum->buffer_start_sample += drop;
            }
        }
        if (out_frame_count) *out_frame_count = frame_count;
        return PITCHEE_SUCCESS;
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return PITCHEE_ERROR_INTERNAL;
    }
}

void pitchee_spectrum_reset(pitchee_spectrum_t* spectrum) {
    if (!spectrum) return;
    spectrum->buffer.clear();
    spectrum->previous_linear.clear();
    spectrum->linear_magnitudes.clear();
    spectrum->output_values.clear();
    spectrum->buffer_start_sample = 0;
    spectrum->total_samples = 0;
    spectrum->next_frame_end_sample = static_cast<size_t>(spectrum->fft_size);
    spectrum->has_previous = false;
}

void pitchee_spectrum_destroy(pitchee_spectrum_t* spectrum) {
    delete spectrum;
}

}  // namespace
