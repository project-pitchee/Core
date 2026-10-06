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
    require(std::string(pitchee_core_version()) == "0.3.0", "version");

    pitchee_composite_score_t score{};
    require(
        pitchee_composite_score(
            PITCHEE_SCORE_PROFILE_FEMINIZATION,
            90.0,
            100.0,
            200.0,
            1,
            &score
        )
            == PITCHEE_SUCCESS,
        "composite call"
    );
    require(close(score.final_score, 100.0), "pass boost score");
    require(score.score_boosted == 1, "pass boost flag");
    require(std::string(score.score_rule) == "pass_boost", "pass boost rule");

    require(
        pitchee_composite_score(
            PITCHEE_SCORE_PROFILE_FEMINIZATION,
            70.0,
            20.0,
            180.0,
            1,
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
        pitchee_composite_score(
            PITCHEE_SCORE_PROFILE_FEMINIZATION,
            20.0,
            20.0,
            180.0,
            1,
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
        pitchee_composite_score(
            PITCHEE_SCORE_PROFILE_FEMINIZATION,
            63.0,
            90.0,
            0.0,
            0,
            &score
        )
            == PITCHEE_SUCCESS,
        "no f0 call"
    );
    require(close(score.final_score, 63.0), "no f0 fallback");
    require(std::string(score.score_rule) == "f0_unavailable", "no f0 rule");

    require(
        close(
            pitchee_composite_score_value(
                PITCHEE_SCORE_PROFILE_FEMINIZATION,
                90.0,
                100.0,
                200.0
            ),
            100.0
        ),
        "three-metric composite score"
    );
    require(
        close(
            pitchee_composite_score_value(
                PITCHEE_SCORE_PROFILE_FEMINIZATION,
                63.0,
                90.0,
                0.0
            ),
            63.0
        ),
        "three-metric f0 fallback"
    );

    require(
        pitchee_composite_score(
            PITCHEE_SCORE_PROFILE_MASCULINIZATION,
            50.0,
            0.0,
            165.0,
            1,
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
        pitchee_composite_score(
            PITCHEE_SCORE_PROFILE_MASCULINIZATION,
            30.0,
            100.0,
            120.0,
            1,
            &score
        ) == PITCHEE_SUCCESS,
        "masculinization score call"
    );
    require(close(score.final_score, 81.0), "masculinization score");
    require(close(score.base_score, score.final_score), "base equals final");

    const double naturalness_independent = pitchee_composite_score_value(
        PITCHEE_SCORE_PROFILE_MASCULINIZATION,
        30.0,
        0.0,
        120.0
    );
    require(
        close(naturalness_independent, score.final_score),
        "masculinization ignores naturalness"
    );
    require(
        std::isnan(pitchee_composite_score_value(
            static_cast<pitchee_score_profile_t>(99),
            30.0,
            0.0,
            120.0
        )),
        "invalid score profile"
    );

    std::cout << "PitcheeCore C++ tests passed\n";
    return 0;
}
