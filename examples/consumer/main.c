#include "pitchee/pitchee.h"

#include <stdio.h>

int main(void) {
    pitchee_composite_score_t score;
    if (pitchee_composite_score(90.0, 100.0, 200.0, 1, &score)
        != PITCHEE_SUCCESS) {
        return 1;
    }
    printf("PitcheeCore %s: %.2f (%s)\n",
           pitchee_core_version(),
           score.final_score,
           score.score_rule);
    return 0;
}
