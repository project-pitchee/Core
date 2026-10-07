#include "pitchee/pitchee.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Capture {
    size_t frames = 0;
    float last_vfp = 0.0f;
    float last_naturalness = 0.0f;
};

struct F0Capture {
    size_t frames = 0;
    size_t voiced_frames = 0;
};

void capture_frame(
    const pitchee_realtime_scores_frame_t* frame,
    void* user_data
) {
    auto& capture = *static_cast<Capture*>(user_data);
    capture.frames += 1;
    capture.last_vfp = frame->vfp_standard_score;
    capture.last_naturalness = frame->naturalness_score;
}

void capture_f0_frame(
    const pitchee_f0_frame_t* frame,
    void* user_data
) {
    auto& capture = *static_cast<F0Capture*>(user_data);
    capture.frames += 1;
    if (frame->voiced) capture.voiced_frames += 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "model directory is required\n";
        return 1;
    }
    pitchee_analyzer_options_t analyzer_options{
        1,
        1,
        {75.0f, 600.0f, 0.9f, 0},
    };
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
        std::cout << "PitcheeCore realtime scores test skipped\n";
        return 0;
    }
    if (analyzer_status != PITCHEE_SUCCESS) {
        std::cerr << "analyzer create failed: " << error << "\n";
        return 1;
    }

    pitchee_realtime_scores_options_t options{
        24240,
        1600,
        {75.0f, 600.0f, 0.9f, 0},
    };
    pitchee_realtime_scores_t* stream = nullptr;
    if (pitchee_realtime_scores_create(
            analyzer,
            &options,
            &stream,
            error,
            sizeof(error)
        ) != PITCHEE_SUCCESS) {
        std::cerr << "scores create failed: " << error << "\n";
        pitchee_analyzer_destroy(analyzer);
        return 1;
    }

    constexpr int sample_rate = 16000;
    constexpr int duration_samples = sample_rate * 4;
    std::vector<float> samples(duration_samples);
    for (int index = 0; index < duration_samples; ++index) {
        const double time = static_cast<double>(index) / sample_rate;
        samples[static_cast<size_t>(index)] = static_cast<float>(
            0.15 * std::sin(2.0 * kPi * 180.0 * time)
            + 0.05 * std::sin(2.0 * kPi * 360.0 * time)
        );
    }

    Capture capture;
    for (size_t start = 0; start < samples.size(); start += 256) {
        const size_t count = std::min<size_t>(256, samples.size() - start);
        size_t emitted = 0;
        if (pitchee_realtime_scores_process(
                stream,
                samples.data() + start,
                count,
                capture_frame,
                &capture,
                &emitted,
                error,
                sizeof(error)
            ) != PITCHEE_SUCCESS) {
            std::cerr << "scores process failed: " << error << "\n";
            pitchee_realtime_scores_destroy(stream);
            pitchee_analyzer_destroy(analyzer);
            return 1;
        }
    }

    if (capture.frames == 0) {
        std::cerr << "no realtime scores frames\n";
        return 1;
    }
    if (capture.last_vfp < 0.0f || capture.last_vfp > 100.0f
        || capture.last_naturalness < 0.0f
        || capture.last_naturalness > 100.0f) {
        std::cerr << "unexpected scores\n";
        return 1;
    }

    pitchee_realtime_scores_destroy(stream);

    pitchee_realtime_f0_options_t f0_options{
        5120,
        256,
        {75.0f, 600.0f, 0.9f, 0},
    };
    pitchee_realtime_f0_t* f0_stream = nullptr;
    if (pitchee_realtime_f0_create(
            analyzer,
            &f0_options,
            &f0_stream,
            error,
            sizeof(error)
        ) != PITCHEE_SUCCESS) {
        std::cerr << "F0 stream create failed: " << error << "\n";
        pitchee_analyzer_destroy(analyzer);
        return 1;
    }
    F0Capture accepted_f0;
    for (size_t start = 0; start < samples.size(); start += 256) {
        const size_t count = std::min<size_t>(256, samples.size() - start);
        size_t emitted = 0;
        if (pitchee_realtime_f0_process(
                f0_stream,
                samples.data() + start,
                count,
                capture_f0_frame,
                &accepted_f0,
                &emitted,
                error,
                sizeof(error)
            ) != PITCHEE_SUCCESS) {
            std::cerr << "F0 stream process failed: " << error << "\n";
            pitchee_realtime_f0_destroy(f0_stream);
            pitchee_analyzer_destroy(analyzer);
            return 1;
        }
    }
    pitchee_realtime_f0_destroy(f0_stream);
    if (accepted_f0.frames == 0 || accepted_f0.voiced_frames == 0) {
        std::cerr << "default SwiftF0 thresholds rejected the test signal\n";
        pitchee_analyzer_destroy(analyzer);
        return 1;
    }

    f0_options.thresholds.min_f0_hz = 200.0f;
    if (pitchee_realtime_f0_create(
            analyzer,
            &f0_options,
            &f0_stream,
            error,
            sizeof(error)
        ) != PITCHEE_SUCCESS) {
        std::cerr << "restricted F0 stream create failed: " << error << "\n";
        pitchee_analyzer_destroy(analyzer);
        return 1;
    }
    F0Capture rejected_f0;
    for (size_t start = 0; start < samples.size(); start += 256) {
        const size_t count = std::min<size_t>(256, samples.size() - start);
        size_t emitted = 0;
        if (pitchee_realtime_f0_process(
                f0_stream,
                samples.data() + start,
                count,
                capture_f0_frame,
                &rejected_f0,
                &emitted,
                error,
                sizeof(error)
            ) != PITCHEE_SUCCESS) {
            std::cerr << "restricted F0 stream process failed: " << error << "\n";
            pitchee_realtime_f0_destroy(f0_stream);
            pitchee_analyzer_destroy(analyzer);
            return 1;
        }
    }
    pitchee_realtime_f0_destroy(f0_stream);
    if (rejected_f0.frames == 0 || rejected_f0.voiced_frames != 0) {
        std::cerr << "realtime F0 thresholds were not applied\n";
        pitchee_analyzer_destroy(analyzer);
        return 1;
    }

    options.thresholds.min_f0_hz = 75.0f;
    options.thresholds.max_f0_hz = 600.0f;
    options.thresholds.min_confidence = 1.0f;
    Capture rejected;
    if (pitchee_realtime_scores_create(
            analyzer,
            &options,
            &stream,
            error,
            sizeof(error)
        ) != PITCHEE_SUCCESS) {
        std::cerr << "scores thresholds create failed: " << error << "\n";
        pitchee_analyzer_destroy(analyzer);
        return 1;
    }
    for (size_t start = 0; start < samples.size(); start += 256) {
        const size_t count = std::min<size_t>(256, samples.size() - start);
        size_t emitted = 0;
        if (pitchee_realtime_scores_process(
                stream,
                samples.data() + start,
                count,
                capture_frame,
                &rejected,
                &emitted,
                error,
                sizeof(error)
            ) != PITCHEE_SUCCESS) {
            std::cerr << "scores thresholds process failed: " << error << "\n";
            pitchee_realtime_scores_destroy(stream);
            pitchee_analyzer_destroy(analyzer);
            return 1;
        }
    }
    if (rejected.frames != 0) {
        std::cerr << "SwiftF0 thresholds were not applied\n";
        pitchee_realtime_scores_destroy(stream);
        pitchee_analyzer_destroy(analyzer);
        return 1;
    }
    pitchee_realtime_scores_destroy(stream);
    pitchee_analyzer_destroy(analyzer);
    std::cout << "PitcheeCore realtime scores test passed\n";
    return 0;
}
