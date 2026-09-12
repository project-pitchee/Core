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

    const std::vector<pitchee::VadSegment> segments{
        {1.0, 2.0, 0.0, 1.0},
        {3.0, 4.0, 1.0, 2.0},
    };
    const auto within_segment = pitchee::map_speech_range_to_source(
        segments,
        0.1,
        0.2
    );
    require(
        std::abs(within_segment.first - 1.1) <= 1e-9
            && std::abs(within_segment.second - 1.2) <= 1e-9,
        "source mapping within segment"
    );
    const auto across_segments = pitchee::map_speech_range_to_source(
        segments,
        0.9,
        1.1
    );
    require(
        std::abs(across_segments.first - 1.9) <= 1e-9
            && std::abs(across_segments.second - 3.1) <= 1e-9,
        "source mapping across segments"
    );
    std::cout << "PitcheeCore resampling test passed\n";
    return 0;
}
