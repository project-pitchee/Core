#include "pitchee/pitchee.h"

#include <stdio.h>
#include <string.h>

static void progress_callback(const pitchee_progress_t* progress, void* user_data) {
    (void)progress;
    (void)user_data;
}

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

    if (pitchee_analyzer_analyze_wav_file_with_progress(
            NULL,
            "missing.wav",
            progress_callback,
            NULL,
            NULL,
            NULL,
            0
        ) != PITCHEE_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "unexpected progress API status\n");
        return 1;
    }

    if (pitchee_realtime_f0_create(
            NULL,
            NULL,
            NULL,
            NULL,
            0
        ) != PITCHEE_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "unexpected realtime F0 create status\n");
        return 1;
    }
    if (pitchee_realtime_f0_process(
            NULL,
            NULL,
            0,
            NULL,
            NULL,
            NULL,
            NULL,
            0
        ) != PITCHEE_ERROR_INVALID_ARGUMENT) {
        fprintf(stderr, "unexpected realtime F0 process status\n");
        return 1;
    }
    return 0;
}
