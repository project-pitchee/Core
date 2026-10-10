# Android ncnn benchmark

This application builds PitcheeCore with Android ncnn-only code and reports
the VFP, naturalness, F0 and total inference time for every WAV in
`assets/test_audio`. The Android APK does not package or link
`libonnxruntime.so`.

## Build

Set the ncnn Android arm64 distribution paths, then build the debug APK:

```bash
export PITCHEE_NCNN_INCLUDE_DIR=/path/to/ncnn-android/arm64-v8a/include
export PITCHEE_NCNN_LIBRARY=/path/to/ncnn-android/arm64-v8a/lib/libncnn.a
gradle -p platform/android/benchmark assembleDebug
```

The `syncPitcheeModels` Gradle task copies `models/` into the APK assets before
packaging. Model copies and build outputs are intentionally not tracked.

OpenMP is linked statically into `libpitchee_core.so`; the APK does not need a
separate `libomp.so`. Android native libraries are linked with a 16 KB maximum
page size for modern Android devices.
