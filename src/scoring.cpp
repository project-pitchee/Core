#include "internal.hpp"

#include <algorithm>
#include <cmath>

namespace pitchee {

CompositeScore calculate_composite_score(
    double vfp_standard_score,
    double naturalness_score,
    bool has_f0,
    double f0_hz
) {
    const double standard = std::max(
        0.0,
        std::min(100.0, vfp_standard_score)
    );
    const double naturalness = std::max(0.0, std::min(100.0, naturalness_score));
    CompositeScore output;
    if (!has_f0 || f0_hz <= 0.0) {
        output.base_score = standard;
        output.final_score = standard;
        output.score_rule = "f0_unavailable";
        return output;
    }

    const double standard_ratio = standard / 100.0;
    const double naturalness_ratio = std::max(
        0.0,
        std::min(1.0, (naturalness - 40.0) / 50.0)
    );
    const double f0_ratio = std::max(
        0.0,
        std::min(1.0, (f0_hz - 110.0) / 90.0)
    );
    const double base = 100.0 * (
        0.50 * standard_ratio
        + 0.20 * naturalness_ratio
        + 0.15 * f0_ratio
        + 0.15 * standard_ratio * naturalness_ratio * f0_ratio
    );

    double score = base;
    output.score_rule = "continuous";
    if (f0_hz > 165.0 && naturalness > 80.0 && standard > 50.0) {
        const double strength = std::min({
            (f0_hz - 165.0) / 25.0,
            (naturalness - 80.0) / 20.0,
            (standard - 50.0) / 30.0,
            1.0
        });
        const double promoted = 60.0 + 40.0 * strength;
        if (promoted > score) {
            score = promoted;
            output.score_boosted = true;
        }
        output.score_rule = "pass_boost";
    } else if (f0_hz > 165.0 && naturalness < 50.0 && standard > 50.0) {
        output.score_rule = "high_f0_stylized_cap";
        output.has_score_cap = true;
        output.score_cap = 45.0;
    } else if (f0_hz <= 165.0 && naturalness >= 50.0) {
        output.score_rule = "low_f0_natural_cap";
        output.has_score_cap = true;
        output.score_cap = 59.0;
    } else if (f0_hz <= 165.0 && naturalness < 50.0) {
        output.score_rule = "low_f0_stylized_cap";
        output.has_score_cap = true;
        output.score_cap = 20.0;
    } else if (f0_hz > 165.0 && naturalness >= 50.0 && standard < 50.0) {
        output.score_rule = "high_f0_male_cap";
        output.has_score_cap = true;
        output.score_cap = 59.0;
    }

    output.base_score = base;
    output.final_score = output.has_score_cap
        ? std::min(score, output.score_cap)
        : score;
    output.final_score = std::max(0.0, std::min(100.0, output.final_score));
    output.score_limited = output.has_score_cap && output.final_score < score;
    return output;
}

}  // namespace pitchee
