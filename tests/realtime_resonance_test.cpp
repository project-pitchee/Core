#include "pitchee/pitchee.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Capture {
    size_t frames = 0;
    float last_f0 = 0.0f;
    float last_score = 0.0f;
    float last_formants[4] = {};
};

void capture_frame(const pitchee_resonance_frame_t* frame, void* user_data) {
    auto& capture = *static_cast<Capture*>(user_data);
    capture.frames += 1;
    capture.last_f0 = frame->f0_hz;
    capture.last_score = frame->resonance_score;
    capture.last_formants[0] = frame->f1_hz;
    capture.last_formants[1] = frame->f2_hz;
    capture.last_formants[2] = frame->f3_hz;
    capture.last_formants[3] = frame->f4_hz;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "model directory is required\n";
        return 1;
    }
    pitchee_analyzer_options_t analyzer_options{1, 0, 0};
    pitchee_analyzer_t* analyzer = nullptr;
    char error[1024] = {};
    const auto analyzer_status = pitchee_analyzer_create(
        argv[1],
        &analyzer_options,
        &analyzer,
        error,
        sizeof(error)
    );
    if (analyzer_status == PITCHEE_ERROR_ORT_UNAVAILABLE) {
        std::cout << "PitcheeCore realtime resonance test skipped\n";
        return 0;
    }
    if (analyzer_status != PITCHEE_SUCCESS) {
        std::cerr << "analyzer create failed: " << error << "\n";
        return 1;
    }

    pitchee_realtime_resonance_options_t options{
        5120,
        256,
        "\xC3\xA6",
        3200,
        0,
    };
    pitchee_realtime_resonance_t* stream = nullptr;
    if (pitchee_realtime_resonance_create(
            analyzer,
            &options,
            &stream,
            error,
            sizeof(error)
        ) != PITCHEE_SUCCESS) {
        std::cerr << "resonance create failed: " << error << "\n";
        pitchee_analyzer_destroy(analyzer);
        return 1;
    }

    constexpr int sample_rate = 16000;
    constexpr int duration_samples = sample_rate * 2;
    constexpr double f0 = 200.0;
    std::vector<float> samples(duration_samples);
    for (int index = 0; index < duration_samples; ++index) {
        const double time = static_cast<double>(index) / sample_rate;
        double value = 0.0;
        for (int harmonic = 1; harmonic <= 20; ++harmonic) {
            const double frequency = harmonic * f0;
            if (frequency >= 7000.0) break;
            const double formant_shape =
                std::exp(-std::pow((frequency - 700.0) / 350.0, 2.0))
                + 0.8 * std::exp(-std::pow((frequency - 2200.0) / 500.0, 2.0))
                + 0.5 * std::exp(-std::pow((frequency - 3500.0) / 600.0, 2.0));
            value += (0.05 + formant_shape)
                * std::sin(2.0 * kPi * frequency * time)
                / harmonic;
        }
        samples[static_cast<size_t>(index)] = static_cast<float>(0.25 * value);
    }

    Capture capture;
    for (size_t start = 0; start < samples.size(); start += 256) {
        const size_t count = std::min<size_t>(256, samples.size() - start);
        size_t emitted = 0;
        if (pitchee_realtime_resonance_process(
                stream,
                samples.data() + start,
                count,
                capture_frame,
                &capture,
                &emitted,
                error,
                sizeof(error)
            ) != PITCHEE_SUCCESS) {
            std::cerr << "resonance process failed: " << error << "\n";
            pitchee_realtime_resonance_destroy(stream);
            pitchee_analyzer_destroy(analyzer);
            return 1;
        }
    }

    if (capture.frames == 0) {
        std::cerr << "no realtime resonance frames\n";
        return 1;
    }
    if (capture.last_f0 < 150.0f || capture.last_f0 > 250.0f) {
        std::cerr << "unexpected F0: " << capture.last_f0 << "\n";
        return 1;
    }
    if (capture.last_score < 0.0f || capture.last_score > 100.0f) {
        std::cerr << "unexpected resonance score: " << capture.last_score << "\n";
        return 1;
    }
    for (const float formant : capture.last_formants) {
        if (!std::isfinite(formant) || formant <= 0.0f || formant >= 8000.0f) {
            std::cerr << "invalid formant\n";
            return 1;
        }
    }

    pitchee_realtime_resonance_reset(stream);
    pitchee_realtime_resonance_destroy(stream);
    pitchee_analyzer_destroy(analyzer);
    std::cout << "PitcheeCore realtime resonance test passed\n";
    return 0;
}
