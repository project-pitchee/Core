# C ABI

PitcheeCore `0.5.0` keeps the main ABI focused on four operations:

1. Create one reusable analyzer.
2. Analyze an audio buffer or WAV file.
3. Run a realtime F0 stream.
4. Calculate a composite score from three metrics.

Spectrum has its own header and is independent of model loading.

```c
#include <pitchee/pitchee.h>
#include <pitchee/spectrum.h>
```

## Analyzer lifetime

Analyzer creation owns all common runtime settings and all ONNX Runtime sessions:

```c
pitchee_analyzer_t* analyzer = NULL;
char error[1024] = {0};

pitchee_status_t status = pitchee_analyzer_create(
    model_directory,
    2,       /* intra_op_threads */
    0,       /* use_coreml */
    75.0f,   /* min_f0_hz */
    600.0f,  /* max_f0_hz */
    0.9f,    /* min SwiftF0 confidence */
    &analyzer,
    error,
    sizeof(error)
);
```

Parameter rules:

- `intra_op_threads <= 0` selects `2`.
- `use_coreml == 0` disables CoreML; non-zero enables it.
- `min_f0_hz <= 0` selects `75`.
- `max_f0_hz <= 0` selects `600`.
- `min_confidence <= 0` selects `0.9`.
- Realtime F0 streams inherit these F0 thresholds.

The analyzer is reusable across audio buffers. Calls on one analyzer must be
serialized, or each worker should create its own analyzer.

Destroy it with `pitchee_analyzer_destroy(analyzer)`.

## Score profile

Every offline analysis and score call requires:

```c
PITCHEE_SCORE_PROFILE_FEMINIZATION
PITCHEE_SCORE_PROFILE_MASCULINIZATION
```

The profile selects the composite scoring rule. It does not change the raw F0,
VFP, or Naturalness model outputs.

## Analyze PCM

```c
char* json = NULL;
status = pitchee_analyze_pcm(
    analyzer,
    samples,
    sample_count,
    sample_rate,
    channels,
    PITCHEE_SCORE_PROFILE_FEMINIZATION,
    progress_callback,
    user_data,
    &json,
    error,
    sizeof(error)
);
```

- `samples` is Float32 PCM in `[-1, 1]`.
- `sample_count` is the total number of float values across all channels.
- `channels == 1` means mono; greater than one means interleaved.
- The library resamples to 16 kHz and does not impose a maximum duration.
- Resampled audio is quantized to PCM16 to match model training.
- `progress_callback` can be `NULL`.
- Release `json` with `pitchee_string_free(json)`.

`pitchee_progress_t` contains:

| Field | Meaning |
| --- | --- |
| `stage` | A `pitchee_progress_stage_t` constant. |
| `completed` | Completed units in the current stage. |
| `total` | Total units in the current stage. |
| `fraction` | Stage-local value in `[0, 1]`. |

Stages cover loading, resampling, F0, VAD, VFP, Naturalness, scoring,
serialization, and completion. Callbacks run synchronously.

## Analyze WAV

```c
status = pitchee_analyze_wav_file(
    analyzer,
    wav_path,
    PITCHEE_SCORE_PROFILE_FEMINIZATION,
    progress_callback,
    user_data,
    &json,
    error,
    sizeof(error)
);
```

PCM16, PCM32, and Float32 WAV files are supported. Compressed formats should be
decoded by the platform and passed to `pitchee_analyze_pcm()`.

## Realtime F0

```c
pitchee_realtime_f0_t* stream = NULL;
pitchee_realtime_f0_create(
    analyzer,
    5120,  /* context_samples; <= 0 uses 320 ms */
    256,   /* hop_samples; <= 0 uses 16 ms */
    &stream,
    error,
    sizeof(error)
);

void on_f0_frame(const pitchee_f0_frame_t* frame, void* user_data);

pitchee_realtime_f0_process(
    stream,
    samples,
    sample_count,
    on_f0_frame,
    user_data,
    error,
    sizeof(error)
);

pitchee_realtime_f0_reset(stream);
pitchee_realtime_f0_destroy(stream);
```

Input is 16 kHz mono Float32 PCM. Each callback frame contains:

| Field | Meaning |
| --- | --- |
| `timestamp_seconds` | Monotonic timestamp since the stream was reset. |
| `f0_hz` | Raw SwiftF0 estimate. |
| `confidence` | SwiftF0 confidence in `[0, 1]`. |
| `voiced` | `1` when the analyzer's F0 thresholds are satisfied. |

The analyzer must outlive every stream created from it.

## Composite score

```c
pitchee_score_result_t score;
pitchee_score(
    PITCHEE_SCORE_PROFILE_FEMINIZATION,
    vfp_standard_score,
    naturalness_score,
    f0_hz,
    &score
);
```

Pass `NAN` or a non-positive `f0_hz` when F0 is unavailable. The final value is
`score.final_score`; the other fields describe the selected rule, cap, and
whether a boost or limit was applied.

This function has no ONNX Runtime dependency.

## Spectrum

Spectrum is independent of analyzer creation and ONNX Runtime:

```c
#include <pitchee/spectrum.h>

pitchee_spectrum_options_t options = {
    2048, 256, 40, 8000,
    PITCHEE_SPECTRUM_DBFS, 0.65f
};

pitchee_spectrum_t* spectrum = NULL;
pitchee_spectrum_create(&options, &spectrum, error, sizeof(error));

void on_spectrum_frame(
    const pitchee_spectrum_frame_t* frame,
    void* user_data
);

pitchee_spectrum_process(
    spectrum,
    samples,
    sample_count,
    on_spectrum_frame,
    user_data,
    error,
    sizeof(error)
);

pitchee_spectrum_reset(spectrum);
pitchee_spectrum_destroy(spectrum);
```

The magnitude pointer is valid only during the callback. Input is 16 kHz mono
Float32 PCM.

## Result schema

The batch result is UTF-8 JSON with these top-level sections:

```json
{
  "schema_version": 4,
  "model_version": "2026-10-f1",
  "score_profile": "feminization",
  "audio": {},
  "vad": {},
  "f0": {},
  "vfp": {},
  "naturalness": {},
  "composite": {}
}
```

`f0.windows`, `vfp.windows`, and `naturalness.windows` use the original analyzed
audio timeline. The result contains data only: no UI geometry, labels, player
state, or color information.
