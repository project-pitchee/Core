#include "pitchee/pitchee.h"

#include <stdio.h>

int main(void) {
    pitchee_score_result_t score;
    if (pitchee_score(
            PITCHEE_SCORE_PROFILE_FEMINIZATION,
            90.0,
            100.0,
            200.0,
            &score
        )
        != PITCHEE_SUCCESS) {
        return 1;
    }
    printf("PitcheeCore %s: %.2f (%s)\n",
           pitchee_version(),
           score.final_score,
           score.score_rule);
    return 0;
}
