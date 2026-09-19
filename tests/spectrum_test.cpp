#include "pitchee/pitchee.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

struct SpectrumCapture {
    size_t frames = 0;
    float first_peak_hz = 0.0f;
    size_t first_bin_count = 0;
};

void capture_frame(const pitchee_spectrum_frame_t* frame, void* user_data) {
    auto& capture = *static_cast<SpectrumCapture*>(user_data);
    if (capture.frames == 0) {
        capture.first_peak_hz = frame->peak_hz;
        capture.first_bin_count = frame->bin_count;
    }
    ++capture.frames;
}

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    pitchee_spectrum_options_t options{
        2048,
        256,
        40,
        8000,
        PITCHEE_SPECTRUM_DBFS,
        0.0f,
        0
    };
    pitchee_spectrum_t* spectrum = nullptr;
    char error[1024] = {};
    require(
        pitchee_spectrum_create(
            &options,
            &spectrum,
            error,
            sizeof(error)
        ) == PITCHEE_SUCCESS,
        "spectrum create"
    );

    constexpr size_t sample_count = 16000;
    std::vector<float> samples(sample_count);
    for (size_t index = 0; index < sample_count; ++index) {
        samples[index] = static_cast<float>(
            0.5 * std::sin(2.0 * 3.14159265358979323846 * 1000.0
                * static_cast<double>(index) / 16000.0)
        );
    }

    SpectrumCapture capture;
    size_t emitted = 0;
    for (size_t start = 0; start < sample_count; start += 1024) {
        const size_t count = std::min<size_t>(1024, sample_count - start);
        require(
            pitchee_spectrum_process(
                spectrum,
                samples.data() + start,
                count,
                capture_frame,
                &capture,
                &emitted,
                error,
                sizeof(error)
            ) == PITCHEE_SUCCESS,
            "spectrum process"
        );
        (void)emitted;
    }
    require(capture.frames == 55, "spectrum frame count");
    require(capture.first_bin_count == 1019, "spectrum bin count");
    require(
        std::abs(capture.first_peak_hz - 1000.0f) <= 8.0f,
        "spectrum peak frequency"
    );

    pitchee_spectrum_reset(spectrum);
    pitchee_spectrum_destroy(spectrum);
    std::cout << "PitcheeCore spectrum test passed\n";
    return 0;
}
