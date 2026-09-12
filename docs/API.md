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
- The library resamples to 16 kHz and analyzes the entire supplied audio.
- The result is UTF-8 JSON.
- Release it with `pitchee_string_free(json)`.

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
  "schema_version": 1,
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
  "pitch": {
    "mean_hz": null,
    "standard_deviation_hz": null,
    "voiced_frame_count": 0
  },
  "models": {
    "raw_female_score": 0.0,
    "standard_score": 0.0,
    "naturalness_score": 0.0,
    "vfp_window_count": 0,
    "vfp_window_duration_seconds": 0.0,
    "vfp_window_starts_seconds": [],
    "vfp_window_raw_scores": [],
    "naturalness_patch_count": 0
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

There is no timeline geometry, color band, label, player state, or other UI
concept in the result. A UI can construct those from the VAD segments and VFP
window arrays.

## Composite score helper

```c
pitchee_composite_score_t score;
pitchee_composite_score(
    standard_score,
    naturalness_score,
    f0_hz,
    has_f0,
    &score
);
```

This function has no ONNX Runtime dependency and can be reused independently.
