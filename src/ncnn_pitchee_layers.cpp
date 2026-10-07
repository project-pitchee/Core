#include "ncnn_pitchee_layers.hpp"

#include <ncnn/layer.h>
#include <ncnn/net.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <vector>

namespace pitchee {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::size_t mat_element_count(const ncnn::Mat& mat) {
    return static_cast<std::size_t>(mat.w)
        * static_cast<std::size_t>(mat.h)
        * static_cast<std::size_t>(mat.c)
        * static_cast<std::size_t>(mat.d);
}

void fft(std::vector<std::complex<double>>& values) {
    const int size = static_cast<int>(values.size());
    for (int index = 1, reverse = 0; index < size; ++index) {
        int bit = size >> 1;
        for (; reverse & bit; bit >>= 1) reverse ^= bit;
        reverse ^= bit;
        if (index < reverse) std::swap(values[index], values[reverse]);
    }

    for (int length = 2; length <= size; length <<= 1) {
        const double angle = -2.0 * kPi / static_cast<double>(length);
        const std::complex<double> root(std::cos(angle), std::sin(angle));
        for (int start = 0; start < size; start += length) {
            std::complex<double> factor(1.0, 0.0);
            for (int offset = 0; offset < length / 2; ++offset) {
                const std::complex<double> even = values[start + offset];
                const std::complex<double> odd = values[start + offset + length / 2]
                    * factor;
                values[start + offset] = even + odd;
                values[start + offset + length / 2] = even - odd;
                factor *= root;
            }
        }
    }
}

class PitcheeMagnitudeStft : public ncnn::Layer {
public:
    PitcheeMagnitudeStft() {
        one_blob_only = true;
        support_inplace = false;
        support_packing = false;
    }

    int forward(
        const ncnn::Mat& bottom_blob,
        ncnn::Mat& top_blob,
        const ncnn::Option& opt
    ) const override {
        constexpr int window_size = 1024;
        constexpr int hop_size = 256;
        const int input_size = static_cast<int>(mat_element_count(bottom_blob));
        if (input_size < window_size) return -1;
        const int frame_count = 1 + (input_size - window_size) / hop_size;
        const int bin_count = window_size / 2 + 1;
        top_blob.create(
            frame_count,
            bin_count,
            1,
            static_cast<std::size_t>(4u),
            opt.blob_allocator
        );
        if (top_blob.empty()) return -100;

        std::vector<double> window(window_size);
        for (int index = 0; index < window_size; ++index) {
            window[index] = 0.5 - 0.5 * std::cos(
                2.0 * kPi * static_cast<double>(index)
                / static_cast<double>(window_size)
            );
        }

        const float* input_data = static_cast<const float*>(bottom_blob.data);
        std::vector<std::complex<double>> frame(window_size);
        for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
            const int start = frame_index * hop_size;
            for (int sample = 0; sample < window_size; ++sample) {
                frame[sample] = std::complex<double>(
                    static_cast<double>(input_data[start + sample]) * window[sample],
                    0.0
                );
            }
            fft(frame);
            for (int bin = 0; bin < bin_count; ++bin) {
                top_blob.channel(0).row(bin)[frame_index] = static_cast<float>(
                    std::abs(frame[bin])
                );
            }
        }
        return 0;
    }
};

DEFINE_LAYER_CREATOR(PitcheeMagnitudeStft)

class PitcheeSquare : public ncnn::Layer {
public:
    PitcheeSquare() {
        one_blob_only = true;
        support_inplace = false;
        support_packing = false;
    }

    int forward(
        const ncnn::Mat& bottom_blob,
        ncnn::Mat& top_blob,
        const ncnn::Option& opt
    ) const override {
        top_blob = bottom_blob.clone(opt.blob_allocator);
        if (top_blob.empty()) return -100;
        const float* input = static_cast<const float*>(bottom_blob.data);
        float* output = static_cast<float*>(top_blob.data);
        const size_t count = mat_element_count(bottom_blob)
            * static_cast<size_t>(bottom_blob.elempack);
        for (size_t index = 0; index < count; ++index) {
            output[index] = input[index] * input[index];
        }
        return 0;
    }
};

DEFINE_LAYER_CREATOR(PitcheeSquare)

class PitcheePitchDecode : public ncnn::Layer {
public:
    PitcheePitchDecode() {
        one_blob_only = false;
        support_inplace = false;
        support_packing = false;
    }

    int forward(
        const std::vector<ncnn::Mat>& bottom_blobs,
        std::vector<ncnn::Mat>& top_blobs,
        const ncnn::Option& opt
    ) const override {
        if (bottom_blobs.size() < 2 || top_blobs.size() < 2) return -1;
        const ncnn::Mat& probabilities = bottom_blobs[0];
        const ncnn::Mat& centers = bottom_blobs[1];
        const int bin_count = static_cast<int>(mat_element_count(centers));
        int time_count = 1;
        if (probabilities.dims == 1) {
            time_count = 1;
        } else {
            time_count = probabilities.h;
        }
        if (static_cast<int>(mat_element_count(probabilities))
            != time_count * bin_count) {
            return -1;
        }

        ncnn::Mat& pitch = top_blobs[0];
        ncnn::Mat& confidence = top_blobs[1];
        pitch.create(time_count, static_cast<std::size_t>(4u), opt.blob_allocator);
        confidence.create(time_count, static_cast<std::size_t>(4u), opt.blob_allocator);
        if (pitch.empty() || confidence.empty()) return -100;

        const float* center_data = static_cast<const float*>(centers.data);
        const float* probability_data = static_cast<const float*>(probabilities.data);
        for (int time_index = 0; time_index < time_count; ++time_index) {
            const float* row = probability_data + time_index * bin_count;
            int best_bin = 0;
            for (int bin = 1; bin < bin_count; ++bin) {
                if (row[bin] > row[best_bin]) best_bin = bin;
            }
            double probability_sum = 0.0;
            double weighted_pitch = 0.0;
            for (int bin = 0; bin < bin_count; ++bin) {
                if (std::abs(bin - best_bin) > 9) continue;
                probability_sum += row[bin];
                weighted_pitch += static_cast<double>(row[bin]) * center_data[bin];
            }
            confidence[time_index] = static_cast<float>(probability_sum);
            pitch[time_index] = static_cast<float>(
                weighted_pitch / (probability_sum + 1.0e-7)
            );
        }
        return 0;
    }
};

DEFINE_LAYER_CREATOR(PitcheePitchDecode)

class TensorMaskedFill : public ncnn::Layer {
public:
    TensorMaskedFill() {
        one_blob_only = false;
        support_inplace = false;
        support_packing = false;
    }

    int forward(
        const std::vector<ncnn::Mat>& bottom_blobs,
        std::vector<ncnn::Mat>& top_blobs,
        const ncnn::Option& opt
    ) const override {
        if (bottom_blobs.empty()) return -1;
        top_blobs[0] = bottom_blobs[0].clone(opt.blob_allocator);
        if (top_blobs[0].empty()) return -100;
        // The exported ECAPA graph is fixed at 150 frames, where the mask is
        // all false. Keeping the layer as an identity avoids reading bool data
        // through ncnn's float MemoryData path.
        return 0;
    }
};

DEFINE_LAYER_CREATOR(TensorMaskedFill)

class PitcheeIdentity : public ncnn::Layer {
public:
    PitcheeIdentity() {
        one_blob_only = true;
        support_inplace = false;
        support_packing = false;
    }

    int forward(
        const ncnn::Mat& bottom_blob,
        ncnn::Mat& top_blob,
        const ncnn::Option& opt
    ) const override {
        top_blob = bottom_blob.clone(opt.blob_allocator);
        return top_blob.empty() ? -100 : 0;
    }
};

DEFINE_LAYER_CREATOR(PitcheeIdentity)

}  // namespace

void register_ncnn_layers(ncnn::Net& net) {
    net.register_custom_layer("PitcheeMagnitudeSTFT", PitcheeMagnitudeStft_layer_creator);
    net.register_custom_layer("PitcheeSquare", PitcheeSquare_layer_creator);
    net.register_custom_layer("PitcheePitchDecode", PitcheePitchDecode_layer_creator);
    net.register_custom_layer("Tensor.masked_fill", TensorMaskedFill_layer_creator);
    net.register_custom_layer("PitcheeIdentity", PitcheeIdentity_layer_creator);
}

}  // namespace pitchee
