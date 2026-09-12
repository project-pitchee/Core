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
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    };
    const auto resampled = pitchee::resample_mono(
        source.data(),
        source.size(),
        1,
        48000,
        16000
    );
    const std::vector<double> expected{
        0.08643475,
        3.2635032,
        5.52451563,
        9.97281256
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
