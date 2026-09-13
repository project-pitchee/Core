#include "internal.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    const std::vector<float> source{
        0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
        0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f
    };
    const auto resampled = pitchee::resample_mono(
        source.data(),
        source.size(),
        1,
        48000,
        16000
    );
    const std::vector<double> expected{
        0.01287841796875,
        0.321319580078125,
        0.55743408203125,
        0.991058349609375
    };
    require(resampled.size() == expected.size(), "output length");
    for (size_t index = 0; index < expected.size(); ++index) {
        require(
            std::abs(resampled[index] - expected[index]) <= 1e-6,
            "polyphase sample"
        );
    }

    const auto short_windows = pitchee::native_speech_windows(
        32000,
        {{1.0, 1.0625, 0.0, 0.0625}},
        1600
    );
    require(
        short_windows.size() == 1
            && short_windows[0].start == 16000
            && short_windows[0].length == 1000,
        "short speech window is not padded"
    );

    const size_t first = 1000;
    const size_t second = first + pitchee::kPatchSamples + 4000;
    const auto full_windows = pitchee::native_speech_windows(
        second + pitchee::kPatchSamples + 3200,
        {
            {
                static_cast<double>(first) / pitchee::kSampleRate,
                static_cast<double>(first + pitchee::kPatchSamples)
                    / pitchee::kSampleRate,
                0.0,
                0.0,
            },
            {
                static_cast<double>(second) / pitchee::kSampleRate,
                static_cast<double>(second + pitchee::kPatchSamples + 3200)
                    / pitchee::kSampleRate,
                0.0,
                0.0,
            },
        },
        1600
    );
    require(
        full_windows.size() == 4
            && full_windows[0].start == first
            && full_windows[1].start == second
            && full_windows[2].start == second + 1600
            && full_windows[3].start == second + 3200,
        "long speech windows remain 1515 ms and do not cross segments"
    );
    std::cout << "PitcheeCore resampling test passed\n";
    return 0;
}
