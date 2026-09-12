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
    std::cout << "PitcheeCore resampling test passed\n";
    return 0;
}
