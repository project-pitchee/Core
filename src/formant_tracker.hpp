#ifndef PITCHEE_FORMANT_TRACKER_HPP
#define PITCHEE_FORMANT_TRACKER_HPP

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace pitchee {

constexpr size_t kHarmonicFrameSamples = 640;

bool parse_resonance_vowel(const std::string& vowel, int* index);

float resonance_score(
    int vowel_index,
    const std::array<float, 4>& formants_hz
);

class HarmonicFormantTracker {
public:
    HarmonicFormantTracker(int vowel_index, size_t hop_samples);

    std::array<float, 4> process(
        const std::vector<float>& samples,
        double f0_hz
    );

    void reset();

private:
    int vowel_index_ = 0;
    size_t history_limit_ = 1;
    bool initialized_ = false;
    std::array<double, 4> previous_{};
    std::array<double, 4> velocity_{};
    std::array<double, 4> output_{};
    std::array<double, 4> pending_{};
    std::array<int, 4> pending_count_{};
    std::vector<std::array<double, 4>> history_;
};

}  // namespace pitchee

#endif
