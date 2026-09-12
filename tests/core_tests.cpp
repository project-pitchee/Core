#include "pitchee/pitchee.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

bool close(double left, double right, double tolerance = 1e-9) {
    return std::abs(left - right) <= tolerance;
}

}  // namespace

int main() {
    require(std::string(pitchee_core_version()) == "0.1.0", "version");

    pitchee_composite_score_t score{};
    require(
        pitchee_composite_score(90.0, 100.0, 200.0, 1, &score)
            == PITCHEE_SUCCESS,
        "composite call"
    );
    require(close(score.final_score, 100.0), "pass boost score");
    require(score.score_boosted == 1, "pass boost flag");
    require(std::string(score.score_rule) == "pass_boost", "pass boost rule");

    require(
        pitchee_composite_score(70.0, 20.0, 180.0, 1, &score)
            == PITCHEE_SUCCESS,
        "cap call"
    );
    require(close(score.final_score, 45.0), "stylized cap");
    require(score.score_limited == 1, "cap flag");
    require(
        std::string(score.score_rule) == "high_f0_stylized_cap",
        "cap rule"
    );

    require(
        pitchee_composite_score(63.0, 90.0, 0.0, 0, &score)
            == PITCHEE_SUCCESS,
        "no f0 call"
    );
    require(close(score.final_score, 63.0), "no f0 fallback");
    require(std::string(score.score_rule) == "f0_unavailable", "no f0 rule");

    std::cout << "PitcheeCore C++ tests passed\n";
    return 0;
}
