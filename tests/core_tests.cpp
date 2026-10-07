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
    require(std::string(pitchee_version()) == "0.5.0", "version");

    pitchee_score_result_t score{};
    require(
        pitchee_score(
            PITCHEE_SCORE_PROFILE_FEMINIZATION,
            90.0,
            100.0,
            200.0,
            &score
        )
            == PITCHEE_SUCCESS,
        "composite call"
    );
    require(close(score.final_score, 100.0), "pass boost score");
    require(score.score_boosted == 1, "pass boost flag");
    require(std::string(score.score_rule) == "pass_boost", "pass boost rule");

    require(
        pitchee_score(
            PITCHEE_SCORE_PROFILE_FEMINIZATION,
            70.0,
            20.0,
            180.0,
            &score
        )
            == PITCHEE_SUCCESS,
        "cap call"
    );
    require(close(score.final_score, 30.0), "stylized cap");
    require(score.score_limited == 1, "cap flag");
    require(
        std::string(score.score_rule) == "high_f0_stylized_cap",
        "cap rule"
    );

    require(
        pitchee_score(
            PITCHEE_SCORE_PROFILE_FEMINIZATION,
            20.0,
            20.0,
            180.0,
            &score
        )
            == PITCHEE_SUCCESS,
        "low standard cap call"
    );
    require(
        score.final_score <= 30.0,
        "low standard remains within cap"
    );
    require(
        std::string(score.score_rule) == "high_f0_stylized_cap",
        "low standard cap rule"
    );

    require(
        pitchee_score(
            PITCHEE_SCORE_PROFILE_FEMINIZATION,
            63.0,
            90.0,
            0.0,
            &score
        )
            == PITCHEE_SUCCESS,
        "no f0 call"
    );
    require(close(score.final_score, 63.0), "no f0 fallback");
    require(std::string(score.score_rule) == "f0_unavailable", "no f0 rule");

    require(
        pitchee_score(
            PITCHEE_SCORE_PROFILE_MASCULINIZATION,
            50.0,
            0.0,
            165.0,
            &score
        ) == PITCHEE_SUCCESS,
        "masculinization pivot call"
    );
    require(close(score.base_score, 60.0), "masculinization pivot base");
    require(close(score.final_score, 60.0), "masculinization pivot final");
    require(score.has_score_cap == 0, "masculinization no cap");
    require(score.score_limited == 0, "masculinization not limited");
    require(score.score_boosted == 0, "masculinization not boosted");
    require(
        std::string(score.score_rule) == "continuous",
        "masculinization continuous rule"
    );

    require(
        pitchee_score(
            PITCHEE_SCORE_PROFILE_MASCULINIZATION,
            30.0,
            100.0,
            120.0,
            &score
        ) == PITCHEE_SUCCESS,
        "masculinization score call"
    );
    require(close(score.final_score, 81.0), "masculinization score");
    require(close(score.base_score, score.final_score), "base equals final");

    pitchee_score_result_t naturalness_independent{};
    require(
        pitchee_score(
            PITCHEE_SCORE_PROFILE_MASCULINIZATION,
            30.0,
            0.0,
            120.0,
            &naturalness_independent
        ) == PITCHEE_SUCCESS,
        "masculinization naturalness independent call"
    );
    require(
        close(naturalness_independent.final_score, score.final_score),
        "masculinization ignores naturalness"
    );
    require(
        pitchee_score(
            static_cast<pitchee_score_profile_t>(99),
            30.0,
            0.0,
            120.0,
            &score
        ) == PITCHEE_ERROR_INVALID_ARGUMENT,
        "invalid score profile"
    );

    std::cout << "PitcheeCore C++ tests passed\n";
    return 0;
}
