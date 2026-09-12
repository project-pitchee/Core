#include "pitchee/pitchee.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    if (strcmp(pitchee_core_version(), "0.1.0") != 0) {
        fprintf(stderr, "unexpected version\n");
        return 1;
    }

    pitchee_composite_score_t score;
    if (pitchee_composite_score(63.0, 90.0, 0.0, 0, &score)
        != PITCHEE_SUCCESS) {
        fprintf(stderr, "score call failed\n");
        return 1;
    }
    if (score.final_score < 62.999 || score.final_score > 63.001) {
        fprintf(stderr, "unexpected fallback score\n");
        return 1;
    }
    if (strcmp(score.score_rule, "f0_unavailable") != 0) {
        fprintf(stderr, "unexpected score rule\n");
        return 1;
    }
    return 0;
}
