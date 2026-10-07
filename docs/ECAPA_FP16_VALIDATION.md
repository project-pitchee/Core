# ECAPA FP16 storage validation

`ECAPA.ncnn.bin` is stored with ncnn FP16 weights. The `.param` graph is
unchanged, and ncnn expands the weights to FP32 for inference. Only ECAPA was
converted; the other ncnn model assets remain FP32.

## Artifact sizes

```text
ECAPA.ncnn.bin FP32   83,224,560 bytes (79.4 MiB)
ECAPA.ncnn.bin FP16   41,805,808 bytes (39.9 MiB)
reduction             49.8%
```

The Android benchmark debug APK, including all six models and ten test WAVs,
changed from `88,959,197` bytes to `50,102,901` bytes in a clean build. The
reduction is `38,856,296` bytes (`37.1 MiB`) or `43.7%`.

## ANA-AN00 arm64 validation

The same 10 WAV files were analyzed twice on a Huawei `ANA-AN00` running
Android 12. Only `ECAPA.ncnn.bin` was swapped between runs.

| Metric | Mean absolute delta | Maximum absolute delta |
| --- | ---: | ---: |
| VFP standard score | 0.0155 | 0.0925 |
| Naturalness score | 0.0049 | 0.0132 |
| Mean F0 | 0.0000 Hz | 0.0000 Hz |
| Composite score, current rules | 0.0082 | 0.0579 |

FP16 is a storage optimization only. ncnn expands the weights to FP32 for
computation, so the benchmark does not treat the compressed model as a speed
optimization.

## Reproducing the FP16 bin

Use the matching ncnn `ncnnoptimize` tool. The resulting FP16 bin can be paired
with the original `.param`; the optimized `.param` is not required.

```bash
ncnnoptimize \
  ECAPA.ncnn.param \
  ECAPA.ncnn.bin \
  /tmp/ECAPA.ncnn.param \
  /tmp/ECAPA.ncnn.fp16.bin \
  1
```

Only replace `models/ECAPA.ncnn.bin` with `/tmp/ECAPA.ncnn.fp16.bin` when
isolating the weight-storage change.
