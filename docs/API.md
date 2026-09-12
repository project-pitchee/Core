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
    "window_seconds": 0.1,
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
analyzed-audio timeline. VFP inference internally uses concatenated speech, and
each window's source start/end is mapped through `vad.segments`. A window that
crosses multiple retained speech segments therefore spans the removed silence
between them. There is no timeline geometry, color band, label, player state,
or other UI concept in the result.

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
