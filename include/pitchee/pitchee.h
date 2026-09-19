#ifndef PITCHEE_PITCHEE_H
#define PITCHEE_PITCHEE_H

#include <stddef.h>
#include <stdint.h>

#if defined(PITCHEE_STATIC)
#  define PITCHEE_API
#elif defined(_WIN32)
#  if defined(PITCHEE_CORE_BUILD)
#    define PITCHEE_API __declspec(dllexport)
#  else
#    define PITCHEE_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define PITCHEE_API __attribute__((visibility("default")))
#else
#  define PITCHEE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pitchee_analyzer_t pitchee_analyzer_t;
typedef struct pitchee_realtime_f0_t pitchee_realtime_f0_t;
typedef struct pitchee_spectrum_t pitchee_spectrum_t;

typedef enum pitchee_status_t {
    PITCHEE_SUCCESS = 0,
    PITCHEE_ERROR_INVALID_ARGUMENT = 1,
    PITCHEE_ERROR_IO = 2,
    PITCHEE_ERROR_ORT_UNAVAILABLE = 3,
    PITCHEE_ERROR_MODEL = 4,
    PITCHEE_ERROR_NO_SPEECH = 5,
    PITCHEE_ERROR_UNSUPPORTED_FORMAT = 6,
    PITCHEE_ERROR_INTERNAL = 7
} pitchee_status_t;

typedef enum pitchee_analysis_phase_t {
    PITCHEE_PHASE_PREPARING_MODELS = 0,
    PITCHEE_PHASE_LOADING_AUDIO = 1,
    PITCHEE_PHASE_ANALYZING = 2,
    PITCHEE_PHASE_COMPLETED = 3
} pitchee_analysis_phase_t;

typedef void (*pitchee_phase_callback_t)(
    pitchee_analysis_phase_t phase,
    void* user_data
);

typedef enum pitchee_progress_stage_t {
    PITCHEE_PROGRESS_STAGE_LOADING_AUDIO = 0,
    PITCHEE_PROGRESS_STAGE_RESAMPLING_AUDIO = 1,
    PITCHEE_PROGRESS_STAGE_ANALYZING_F0 = 2,
    PITCHEE_PROGRESS_STAGE_DETECTING_SPEECH = 3,
    PITCHEE_PROGRESS_STAGE_PREPARING_VFP_WINDOWS = 4,
    PITCHEE_PROGRESS_STAGE_EXTRACTING_VFP_EMBEDDINGS = 5,
    PITCHEE_PROGRESS_STAGE_CLASSIFYING_VFP_WINDOWS = 6,
    PITCHEE_PROGRESS_STAGE_PREPARING_NATURALNESS_WINDOWS = 7,
    PITCHEE_PROGRESS_STAGE_EXTRACTING_NATURALNESS_EMBEDDINGS = 8,
    PITCHEE_PROGRESS_STAGE_SCORING_NATURALNESS_WINDOWS = 9,
    PITCHEE_PROGRESS_STAGE_CALCULATING_SCORES = 10,
    PITCHEE_PROGRESS_STAGE_SERIALIZING_RESULT = 11,
    PITCHEE_PROGRESS_STAGE_COMPLETED = 12,
    PITCHEE_PROGRESS_STAGE_COUNT = 13
} pitchee_progress_stage_t;

typedef struct pitchee_progress_t {
    pitchee_progress_stage_t stage;
    int32_t reserved;
    uint64_t completed;
    uint64_t total;
    double fraction;
} pitchee_progress_t;

typedef void (*pitchee_progress_callback_t)(
    const pitchee_progress_t* progress,
    void* user_data
);

typedef struct pitchee_realtime_f0_options_t {
    int32_t context_samples;
    int32_t hop_samples;
    int32_t reserved;
} pitchee_realtime_f0_options_t;

typedef struct pitchee_f0_frame_t {
    double timestamp_seconds;
    float f0_hz;
    float confidence;
    int32_t voiced;
    int32_t reserved;
} pitchee_f0_frame_t;

typedef void (*pitchee_f0_frame_callback_t)(
    const pitchee_f0_frame_t* frame,
    void* user_data
);

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
    int32_t reserved;
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
    int32_t reserved;
} pitchee_spectrum_frame_t;

typedef void (*pitchee_spectrum_callback_t)(
    const pitchee_spectrum_frame_t* frame,
    void* user_data
);

typedef struct pitchee_analyzer_options_t {
    int32_t intra_op_threads;
    int32_t use_coreml;
    int32_t reserved;
} pitchee_analyzer_options_t;

typedef struct pitchee_composite_score_t {
    double base_score;
    double final_score;
    double score_cap;
    int32_t has_score_cap;
    int32_t score_limited;
    int32_t score_boosted;
    char score_rule[32];
} pitchee_composite_score_t;

PITCHEE_API const char* pitchee_core_version(void);

PITCHEE_API pitchee_status_t pitchee_analyzer_create(
    const char* model_directory,
    const pitchee_analyzer_options_t* options,
    pitchee_analyzer_t** out_analyzer,
    char* error_message,
    size_t error_message_capacity
);

PITCHEE_API void pitchee_analyzer_destroy(pitchee_analyzer_t* analyzer);

/*
 * Samples must be mono or interleaved float32 PCM in [-1, 1].
 * The implementation resamples to 16 kHz, so any positive source rate is valid.
 * The returned JSON string is allocated by PitcheeCore and must be released with
 * pitchee_string_free().
 */
PITCHEE_API pitchee_status_t pitchee_analyzer_analyze_pcm(
    pitchee_analyzer_t* analyzer,
    const float* samples,
    size_t sample_count,
    int32_t sample_rate,
    int32_t channels,
    pitchee_phase_callback_t phase_callback,
    void* user_data,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
);

PITCHEE_API pitchee_status_t pitchee_analyzer_analyze_pcm_with_progress(
    pitchee_analyzer_t* analyzer,
    const float* samples,
    size_t sample_count,
    int32_t sample_rate,
    int32_t channels,
    pitchee_progress_callback_t progress_callback,
    void* user_data,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
);

PITCHEE_API pitchee_status_t pitchee_analyzer_analyze_wav_file(
    pitchee_analyzer_t* analyzer,
    const char* wav_path,
    pitchee_phase_callback_t phase_callback,
    void* user_data,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
);

PITCHEE_API pitchee_status_t pitchee_analyzer_analyze_wav_file_with_progress(
    pitchee_analyzer_t* analyzer,
    const char* wav_path,
    pitchee_progress_callback_t progress_callback,
    void* user_data,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
);

PITCHEE_API pitchee_status_t pitchee_realtime_f0_create(
    pitchee_analyzer_t* analyzer,
    const pitchee_realtime_f0_options_t* options,
    pitchee_realtime_f0_t** out_stream,
    char* error_message,
    size_t error_message_capacity
);

PITCHEE_API pitchee_status_t pitchee_realtime_f0_process(
    pitchee_realtime_f0_t* stream,
    const float* samples,
    size_t sample_count,
    pitchee_f0_frame_callback_t frame_callback,
    void* user_data,
    size_t* out_frame_count,
    char* error_message,
    size_t error_message_capacity
);

PITCHEE_API void pitchee_realtime_f0_reset(pitchee_realtime_f0_t* stream);

PITCHEE_API void pitchee_realtime_f0_destroy(pitchee_realtime_f0_t* stream);

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
    size_t* out_frame_count,
    char* error_message,
    size_t error_message_capacity
);

PITCHEE_API void pitchee_spectrum_reset(pitchee_spectrum_t* spectrum);

PITCHEE_API void pitchee_spectrum_destroy(pitchee_spectrum_t* spectrum);

PITCHEE_API pitchee_status_t pitchee_composite_score(
    double vfp_standard_score,
    double naturalness_score,
    double f0_hz,
    int32_t has_f0,
    pitchee_composite_score_t* out_score
);

/*
 * Convenience form using exactly three metrics. Pass NAN or a non-positive
 * f0_hz when no valid F0 is available. The returned value is clamped to 0-100.
 */
PITCHEE_API double pitchee_composite_score_value(
    double vfp_standard_score,
    double naturalness_score,
    double f0_hz
);

PITCHEE_API void pitchee_string_free(char* value);

#ifdef __cplusplus
}
#endif

#endif
