#include "ncnn_pitchee_layers.hpp"

#include <ncnn/layer.h>
#include <ncnn/net.h>

#include <algorithm>
#include <cmath>
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

class PitcheeMagnitudeStft : public ncnn::Layer {
public:
    PitcheeMagnitudeStft() {
        one_blob_only = true;
        support_inplace = false;
        support_packing = false;

        window.resize(kWindowSize);
        cos_table.resize(kWindowSize / 2);
        sin_table.resize(kWindowSize / 2);
        bit_reverse.resize(kHalfSize);
        for (int index = 0; index < kWindowSize; ++index) {
            window[index] = static_cast<float>(
                0.5 - 0.5 * std::cos(2.0 * kPi * index / kWindowSize)
            );
        }
        for (int index = 0; index < kWindowSize / 2; ++index) {
            const double angle = 2.0 * kPi * index / kWindowSize;
            cos_table[index] = static_cast<float>(std::cos(angle));
            sin_table[index] = static_cast<float>(std::sin(angle));
        }
        for (int index = 1, reverse = 0; index < kHalfSize; ++index) {
            int bit = kHalfSize >> 1;
            for (; reverse & bit; bit >>= 1) reverse ^= bit;
            reverse ^= bit;
            bit_reverse[index] = reverse;
        }
    }

    int forward(
        const ncnn::Mat& bottom_blob,
        ncnn::Mat& top_blob,
        const ncnn::Option& opt
    ) const override {
        constexpr int window_size = kWindowSize;
        constexpr int hop_size = kHopSize;
        const int input_size = static_cast<int>(mat_element_count(bottom_blob));
        if (input_size < window_size) return -1;
        const int frame_count = 1 + (input_size - window_size) / hop_size;
        const int bin_count = kBinCount;
        top_blob.create(
            frame_count,
            bin_count,
            1,
            static_cast<std::size_t>(4u),
            opt.blob_allocator
        );
        if (top_blob.empty()) return -100;

        const float* input_data = static_cast<const float*>(bottom_blob.data);
        std::vector<float> real(kHalfSize);
        std::vector<float> imag(kHalfSize);
        for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
            const int start = frame_index * hop_size;
            // Pack even/odd real samples into a complex sequence, then recover
            // the real-signal spectrum from its conjugate symmetry.
            for (int sample = 0; sample < kHalfSize; ++sample) {
                real[sample] = input_data[start + sample * 2] * window[sample * 2];
                imag[sample] = input_data[start + sample * 2 + 1]
                    * window[sample * 2 + 1];
            }
            fft_half(real, imag);
            for (int bin = 0; bin < kHalfSize; ++bin) {
                const int mirrored = (kHalfSize - bin) & (kHalfSize - 1);
                const float zr = real[bin];
                const float zi = imag[bin];
                const float br = real[mirrored];
                const float bi = -imag[mirrored];
                const float er = 0.5f * (zr + br);
                const float ei = 0.5f * (zi + bi);
                const float dr = zr - br;
                const float di = zi - bi;
                const float or_value = 0.5f * di;
                const float oi_value = -0.5f * dr;
                const float wr = cos_table[bin];
                const float wi = -sin_table[bin];
                const float xr = er + wr * or_value - wi * oi_value;
                const float xi = ei + wr * oi_value + wi * or_value;
                top_blob.channel(0).row(bin)[frame_index] = std::sqrt(
                    xr * xr + xi * xi
                );
            }
            const float nyquist = std::abs(real[0] - imag[0]);
            top_blob.channel(0).row(kHalfSize)[frame_index] = nyquist;
        }
        return 0;
    }

private:
    static constexpr int kWindowSize = 1024;
    static constexpr int kHopSize = 256;
    static constexpr int kBinCount = kWindowSize / 2 + 1;
    static constexpr int kHalfSize = kWindowSize / 2;

    void fft_half(
        std::vector<float>& real,
        std::vector<float>& imag
    ) const {
        for (int index = 1; index < kHalfSize; ++index) {
            const int reverse = bit_reverse[index];
            if (index < reverse) {
                std::swap(real[index], real[reverse]);
                std::swap(imag[index], imag[reverse]);
            }
        }

        for (int length = 2; length <= kHalfSize; length <<= 1) {
            const int half = length >> 1;
            const int table_stride = kWindowSize / length;
            for (int start = 0; start < kHalfSize; start += length) {
                for (int offset = 0; offset < half; ++offset) {
                    const int table_index = offset * table_stride;
                    const float wr = cos_table[table_index];
                    const float wi = -sin_table[table_index];
                    const int even = start + offset;
                    const int odd = even + half;
                    const float tr = wr * real[odd] - wi * imag[odd];
                    const float ti = wr * imag[odd] + wi * real[odd];
                    const float er = real[even];
                    const float ei = imag[even];
                    real[even] = er + tr;
                    imag[even] = ei + ti;
                    real[odd] = er - tr;
                    imag[odd] = ei - ti;
                }
            }
        }
    }

    std::vector<float> window;
    std::vector<float> cos_table;
    std::vector<float> sin_table;
    std::vector<int> bit_reverse;
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
