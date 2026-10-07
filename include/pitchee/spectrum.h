#ifndef PITCHEE_SPECTRUM_H
#define PITCHEE_SPECTRUM_H

#include "pitchee/pitchee.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pitchee_spectrum_t pitchee_spectrum_t;

typedef enum pitchee_spectrum_value_t {
    PITCHEE_SPECTRUM_AMPLITUDE = 0,
    PITCHEE_SPECTRUM_POWER = 1,
    PITCHEE_SPECTRUM_DBFS = 2
} pitchee_spectrum_value_t;

typedef struct pitchee_spectrum_options_t {
    int32_t fft_size;
    int32_t hop_samples;
    int32_t min_hz;
    int32_t max_hz;
    pitchee_spectrum_value_t value_type;
    float smoothing;
} pitchee_spectrum_options_t;

typedef struct pitchee_spectrum_frame_t {
    double timestamp_seconds;
    const float* magnitudes;
    size_t bin_count;
    size_t first_bin_index;
    float bin_hz;
    float peak_hz;
    float centroid_hz;
    float rolloff_hz;
    float flatness;
} pitchee_spectrum_frame_t;

typedef void (*pitchee_spectrum_callback_t)(
    const pitchee_spectrum_frame_t* frame,
    void* user_data
);

PITCHEE_API pitchee_status_t pitchee_spectrum_create(
    const pitchee_spectrum_options_t* options,
    pitchee_spectrum_t** out_spectrum,
    char* error_message,
    size_t error_message_capacity
);

PITCHEE_API pitchee_status_t pitchee_spectrum_process(
    pitchee_spectrum_t* spectrum,
    const float* samples,
    size_t sample_count,
    pitchee_spectrum_callback_t frame_callback,
    void* user_data,
    char* error_message,
    size_t error_message_capacity
);

PITCHEE_API void pitchee_spectrum_reset(pitchee_spectrum_t* spectrum);

PITCHEE_API void pitchee_spectrum_destroy(pitchee_spectrum_t* spectrum);

#ifdef __cplusplus
}
#endif

#endif
