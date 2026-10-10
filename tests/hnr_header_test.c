#include "pitchee/hnr.h"

typedef struct callback_state {
    int window_count;
    int available_count;
} callback_state;

static void receive_window(const pitchee_hnr_window_t* window, void* user_data) {
    callback_state* state = (callback_state*)user_data;
    ++state->window_count;
    if (window->has_hnr) ++state->available_count;
}

int main(void) {
    float samples[640];
    size_t index;
    for (index = 0; index < 640; ++index) {
        samples[index] = index % 80 < 40 ? 0.3f : -0.3f;
    }
    const pitchee_hnr_vad_segment_t vad = {0.02, 0.03};
    callback_state state = {0, 0};
    pitchee_hnr_summary_t summary;
    /* Only hnr.h is included; no analyzer header, handle, or model is needed. */
    if (pitchee_hnr_analyze(samples, 640, &vad, 1, receive_window, &state, &summary)
        != PITCHEE_SUCCESS) return 1;
    if (state.window_count != 4 || state.available_count != 2) return 2;
    if (summary.window_seconds != 0.04 || summary.hnr_window_count != 2
        || !summary.has_hnr) return 3;
    if (pitchee_hnr_analyze(samples, 640, 0, 0, 0, 0, &summary) != PITCHEE_SUCCESS)
        return 4;
    if (summary.has_hnr || summary.hnr_window_count) return 5;
    return 0;
}
