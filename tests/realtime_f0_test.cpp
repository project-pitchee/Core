#include "pitchee/pitchee.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Capture {
    size_t frames = 0;
    size_t voiced_frames = 0;
};

void capture_frame(
    const pitchee_f0_frame_t* frame,
    void* user_data
) {
    auto& capture = *static_cast<Capture*>(user_data);
    capture.frames += 1;
    if (frame->voiced) capture.voiced_frames += 1;
}

pitchee_analyzer_t* create_analyzer(
    const char* model_directory,
    float min_f0_hz,
    char* error,
    size_t error_capacity
) {
    pitchee_analyzer_t* analyzer = nullptr;
    const auto status = pitchee_analyzer_create(
        model_directory,
        1,
        1,
        min_f0_hz,
        600.0f,
        0.9f,
        &analyzer,
        error,
        error_capacity
    );
    if (status == PITCHEE_ERROR_ORT_UNAVAILABLE) return nullptr;
    if (status != PITCHEE_SUCCESS) {
        std::cerr << "analyzer create failed: " << error << "\n";
        std::exit(1);
    }
    return analyzer;
}

Capture run_f0(
    pitchee_analyzer_t* analyzer,
    const std::vector<float>& samples,
    char* error,
    size_t error_capacity
) {
    pitchee_realtime_f0_t* stream = nullptr;
    if (pitchee_realtime_f0_create(
            analyzer,
            5120,
            256,
            &stream,
            error,
            error_capacity
        ) != PITCHEE_SUCCESS) {
        std::cerr << "F0 stream create failed: " << error << "\n";
        std::exit(1);
    }

    Capture capture;
    for (size_t start = 0; start < samples.size(); start += 256) {
        const size_t count = std::min<size_t>(256, samples.size() - start);
        if (pitchee_realtime_f0_process(
                stream,
                samples.data() + start,
                count,
                capture_frame,
                &capture,
                error,
                error_capacity
            ) != PITCHEE_SUCCESS) {
            std::cerr << "F0 stream process failed: " << error << "\n";
            pitchee_realtime_f0_destroy(stream);
            std::exit(1);
        }
    }
    pitchee_realtime_f0_destroy(stream);
    return capture;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "model directory is required\n";
        return 1;
    }

    char error[1024] = {};
    pitchee_analyzer_t* analyzer = create_analyzer(
        argv[1],
        75.0f,
        error,
        sizeof(error)
    );
    if (!analyzer) {
        std::cout << "PitcheeCore realtime F0 test skipped\n";
        return 0;
    }

    constexpr int sample_rate = 16000;
    constexpr int duration_samples = sample_rate * 2;
    std::vector<float> samples(duration_samples);
    for (int index = 0; index < duration_samples; ++index) {
        const double time = static_cast<double>(index) / sample_rate;
        samples[static_cast<size_t>(index)] = static_cast<float>(
            0.15 * std::sin(2.0 * kPi * 180.0 * time)
            + 0.05 * std::sin(2.0 * kPi * 360.0 * time)
        );
    }

    const Capture accepted = run_f0(
        analyzer,
        samples,
        error,
        sizeof(error)
    );
    pitchee_analyzer_destroy(analyzer);
    if (accepted.frames == 0 || accepted.voiced_frames == 0) {
        std::cerr << "default SwiftF0 thresholds rejected the test signal\n";
        return 1;
    }

    analyzer = create_analyzer(argv[1], 200.0f, error, sizeof(error));
    if (!analyzer) return 0;
    const Capture rejected = run_f0(
        analyzer,
        samples,
        error,
        sizeof(error)
    );
    pitchee_analyzer_destroy(analyzer);
    if (rejected.frames == 0 || rejected.voiced_frames != 0) {
        std::cerr << "analyzer SwiftF0 thresholds were not applied\n";
        return 1;
    }

    std::cout << "PitcheeCore realtime F0 test passed\n";
    return 0;
}
