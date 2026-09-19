# C ABI

## Analyzer lifetime

```c
pitchee_analyzer_options_t options = {
    .intra_op_threads = 2,
    .use_coreml = 0,
    .reserved = 0
};

pitchee_analyzer_t* analyzer = NULL;
char error[1024] = {0};

pitchee_status_t status = pitchee_analyzer_create(
    model_directory,
    &options,
    &analyzer,
    error,
    sizeof(error)
);
```

The analyzer owns every ONNX Runtime session. It is reusable across multiple
audio buffers. Serialize calls on one analyzer, or create one analyzer per
worker.

Destroy it with `pitchee_analyzer_destroy(analyzer)`.

## Raw PCM input

```c
char* json = NULL;
status = pitchee_analyzer_analyze_pcm(
    analyzer,
    samples,
    sample_count,
    sample_rate,
    channels,
    phase_callback,
    user_data,
    &json,
    error,
    sizeof(error)
);
```

- `samples` is Float32 PCM in `[-1, 1]`.
- `sample_count` is the total number of float values across all channels.
- `channels == 1` means mono; greater than one means interleaved.
- The library resamples to 16 kHz and does not impose a maximum analysis duration.
- Resampled audio is quantized to PCM16 to match the model training and server pipeline.
- The result is UTF-8 JSON.
- Release it with `pitchee_string_free(json)`.

## Detailed progress

Use the `_with_progress` variants when the caller needs more detail than the
legacy four-value phase callback:

```c
void on_progress(const pitchee_progress_t* progress, void* user_data) {
    if (progress->stage == PITCHEE_PROGRESS_STAGE_DETECTING_SPEECH) {
        update_progress(progress->fraction);
    }
}

status = pitchee_analyzer_analyze_pcm_with_progress(
    analyzer,
    samples,
    sample_count,
    sample_rate,
    channels,
    on_progress,
    user_data,
    &json,
    error,
    sizeof(error)
);
```

`pitchee_progress_t` contains:

| Field | Meaning |
| --- | --- |
| `stage` | A `pitchee_progress_stage_t` enum constant. No stage strings are returned. |
| `completed` | Completed units in the current stage. |
| `total` | Total units in the current stage. |
| `fraction` | Stage-local value in `[0, 1]`. |

The stage constants are `LOADING_AUDIO`, `RESAMPLING_AUDIO`, `ANALYZING_F0`,
`DETECTING_SPEECH`, `PREPARING_VFP_WINDOWS`, `EXTRACTING_VFP_EMBEDDINGS`,
`CLASSIFYING_VFP_WINDOWS`, `PREPARING_NATURALNESS_WINDOWS`,
`EXTRACTING_NATURALNESS_EMBEDDINGS`, `SCORING_NATURALNESS_WINDOWS`,
`CALCULATING_SCORES`, `SERIALIZING_RESULT`, and `COMPLETED`. All constants use
the `PITCHEE_PROGRESS_STAGE_` prefix.

Callbacks run synchronously on the thread performing analysis. The same
analyzer must not be used concurrently.

## Realtime F0

Create one stream per microphone session:

```c
pitchee_realtime_f0_options_t options = {5120, 256, 0};
pitchee_realtime_f0_t* stream = NULL;

pitchee_realtime_f0_create(
    analyzer,
    &options,
    &stream,
    error,
    sizeof(error)
);
```

`pitchee_realtime_f0_process()` accepts any number of 16 kHz mono Float32
samples. It keeps the latest 320 ms context, runs SwiftF0 every 16 ms of new
audio, and invokes the callback once per new F0 frame:

```c
void on_f0_frame(const pitchee_f0_frame_t* frame, void* user_data);
```

Each frame contains:

| Field | Meaning |
| --- | --- |
| `timestamp_seconds` | Monotonic timestamp since the stream was reset. |
| `f0_hz` | Raw SwiftF0 estimate. |
| `confidence` | SwiftF0 confidence in `[0, 1]`. |
| `voiced` | `1` when confidence is above `0.9` and F0 is in `75–600 Hz`. |

The first callback occurs after the full context is available. Call
`pitchee_realtime_f0_reset()` when starting a new recording without recreating
the stream. The analyzer must outlive every stream created from it.

## Spectrum

Spectrum is independent of model loading and ONNX Runtime:

```c
pitchee_spectrum_options_t options = {
    2048, 256, 40, 8000,
    PITCHEE_SPECTRUM_DBFS, 0.65f, 0
};
pitchee_spectrum_t* spectrum = NULL;

pitchee_spectrum_create(
    &options,
    &spectrum,
    error,
    sizeof(error)
);
```

`pitchee_spectrum_process()` accepts any number of 16 kHz mono Float32 samples
and emits one frame every 256 samples after the first full FFT window:

```c
void on_spectrum_frame(
    const pitchee_spectrum_frame_t* frame,
    void* user_data
);
```

Each frame contains a pointer to float magnitudes, the first FFT bin index,
frequency spacing, and summary values `peak_hz`, `centroid_hz`, `rolloff_hz`,
and `flatness`. The magnitude pointer is valid only during the callback.
Smoothing is applied to linear magnitudes before conversion to amplitude, power,
or dBFS.

## WAV file input

```c
status = pitchee_analyzer_analyze_wav_file(
    analyzer,
    wav_path,
    phase_callback,
    user_data,
    &json,
    error,
    sizeof(error)
);
```

PCM16, PCM32, and Float32 WAV files are supported. Compressed formats should
be decoded by the platform audio API and passed to
`pitchee_analyzer_analyze_pcm()`.

## Result schema

The result is deliberately data-only:

```json
{
  "schema_version": 2,
  "model_version": "2026-09",
  "audio": {
    "source_sample_rate": 48000,
    "source_channels": 1,
    "input_seconds": 5.2,
    "analyzed_seconds": 5.2
  },
  "vad": {
    "segment_count": 4,
    "speech_seconds": 1.5,
    "silero_segment_count": 4,
    "discarded_breath_like_count": 0,
    "trimmed_segment_count": 4,
    "segments": []
  },
  "f0": {
    "window_seconds": 0.05,
    "mean_hz": null,
    "standard_deviation_hz": null,
    "voiced_frame_count": 0,
    "voiced_window_count": 0,
    "windows": []
  },
  "vfp": {
    "vfp_standard_score": 0.0,
    "window_count": 0,
    "window_duration_seconds": 0.0,
    "windows": []
  },
  "naturalness": {
    "score": 0.0,
    "window_count": 0,
    "window_duration_seconds": 0.0,
    "windows": []
  },
  "composite": {
    "base_score": 0.0,
    "final_score": 0.0,
    "cap": null,
    "rule": "continuous",
    "limited": false,
    "boosted": false
  }
}
```

`f0.windows`, `naturalness.windows`, and `vfp.windows` all expose the original
analyzed-audio timeline. VFP and naturalness windows are generated only inside
retained VAD speech segments and never cross removed silence. A speech segment
shorter than the model patch is processed at its exact length without padding.
There is no timeline geometry, color band, label, player state, or other UI
concept in the result.

## Composite score helper

```c
pitchee_composite_score_t score;
pitchee_composite_score(
    vfp_standard_score,
    naturalness_score,
    f0_hz,
    has_f0,
    &score
);
```

This function has no ONNX Runtime dependency and can be reused independently.

For callers that only need the final number, use the three-metric convenience
function:

```c
double final_score = pitchee_composite_score_value(
    vfp_standard_score,
    naturalness_score,
    f0_hz
);
```

Pass `NAN` or a non-positive `f0_hz` when F0 is unavailable.
