# Building PitcheeCore

PitcheeCore is built with CMake and links to ONNX Runtime. ONNX Runtime is not
vendored in this repository. The validated version is `1.24.2`.

Official ONNX Runtime installation documentation:

<https://onnxruntime.ai/docs/install/>

Official `1.24.2` release:

<https://github.com/microsoft/onnxruntime/releases/tag/v1.24.2>

## Where to put ONNX Runtime

The examples below use this layout:

```text
third_party/
  onnxruntime/
    include/
    lib/
```

For Windows, also keep the runtime DLL available:

```text
third_party/
  onnxruntime/
    bin/onnxruntime.dll
    include/
    lib/onnxruntime.lib
```

CMake accepts either explicit paths:

```text
PITCHEE_ORT_INCLUDE_DIR
PITCHEE_ORT_LIBRARY
```

or a root directory:

```text
PITCHEE_ORT_ROOT
```

## macOS arm64

Download:

<https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-osx-arm64-1.24.2.tgz>

```bash
mkdir -p third_party
curl -L -o /tmp/onnxruntime-osx.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-osx-arm64-1.24.2.tgz
tar -xzf /tmp/onnxruntime-osx.tgz -C third_party
mv third_party/onnxruntime-osx-arm64-1.24.2 third_party/onnxruntime

cmake -S . -B build-macos -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPITCHEE_ORT_INCLUDE_DIR="$PWD/third_party/onnxruntime/include" \
  -DPITCHEE_ORT_LIBRARY="$PWD/third_party/onnxruntime/lib/libonnxruntime.dylib"

cmake --build build-macos -j 8
ctest --test-dir build-macos --output-on-failure
```

For packaging, copy `libonnxruntime.dylib` into the application frameworks
directory and set an `@rpath`, or place it next to the executable and configure
the runtime search path.

Intel Macs require an x86_64 ONNX Runtime build. The official `1.24.2` macOS
archive shown above is Apple Silicon only.

## Windows x64

Download:

<https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-win-x64-1.24.2.zip>

PowerShell:

```powershell
New-Item -ItemType Directory -Force third_party
Invoke-WebRequest `
  -Uri "https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-win-x64-1.24.2.zip" `
  -OutFile "$env:TEMP\onnxruntime-win.zip"
Expand-Archive "$env:TEMP\onnxruntime-win.zip" -DestinationPath third_party -Force
Rename-Item third_party\onnxruntime-win-x64-1.24.2 third_party\onnxruntime

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -DPITCHEE_ORT_INCLUDE_DIR="$PWD\third_party\onnxruntime\include" `
  -DPITCHEE_ORT_LIBRARY="$PWD\third_party\onnxruntime\lib\onnxruntime.lib"

cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

At runtime, either copy:

```text
third_party\onnxruntime\bin\onnxruntime.dll
```

next to the executable, or add its directory to `PATH`.

Windows arm64:

<https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-win-arm64-1.24.2.zip>

## Linux x64

Download:

<https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz>

```bash
mkdir -p third_party
curl -L -o /tmp/onnxruntime-linux.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz
tar -xzf /tmp/onnxruntime-linux.tgz -C third_party
mv third_party/onnxruntime-linux-x64-1.24.2 third_party/onnxruntime

cmake -S . -B build-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPITCHEE_ORT_INCLUDE_DIR="$PWD/third_party/onnxruntime/include" \
  -DPITCHEE_ORT_LIBRARY="$PWD/third_party/onnxruntime/lib/libonnxruntime.so"

cmake --build build-linux -j 8
ctest --test-dir build-linux --output-on-failure
```

Install the shared library in a standard loader path or set an rpath:

```bash
export LD_LIBRARY_PATH="$PWD/third_party/onnxruntime/lib:$LD_LIBRARY_PATH"
```

Linux aarch64:

<https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-aarch64-1.24.2.tgz>

## Android

The Maven AAR contains C/C++ headers and native libraries for:

- `arm64-v8a`
- `armeabi-v7a`
- `x86`
- `x86_64`

Download:

<https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.24.2/onnxruntime-android-1.24.2.aar>

Extract it:

```bash
curl -L -o /tmp/onnxruntime-android.aar \
  https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.24.2/onnxruntime-android-1.24.2.aar
unzip /tmp/onnxruntime-android.aar -d third_party/onnxruntime-android
```

Build for Android arm64:

```bash
cmake -S . -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DPITCHEE_BUILD_SHARED=ON \
  -DPITCHEE_BUILD_CLI=OFF \
  -DPITCHEE_ORT_INCLUDE_DIR="$PWD/third_party/onnxruntime-android/headers" \
  -DPITCHEE_ORT_LIBRARY="$PWD/third_party/onnxruntime-android/jni/arm64-v8a/libonnxruntime.so"

cmake --build build-android -j 8
```

The Gradle application must package both:

```text
libpitchee_core.so
libonnxruntime.so
```

The Kotlin and JNI bridge examples are in `platform/android`.

## iOS

Download the C/C++ pod archive:

<https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.24.2.zip>

```bash
curl -L -o /tmp/onnxruntime-ios.zip \
  https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.24.2.zip
unzip /tmp/onnxruntime-ios.zip -d third_party/onnxruntime-ios
```

Extract or locate `onnxruntime.xcframework`. Use the device or simulator slice
and pass the framework headers and binary to CMake:

```bash
cmake -S . -B build-ios -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=15.1 \
  -DPITCHEE_BUILD_SHARED=OFF \
  -DPITCHEE_BUILD_CLI=OFF \
  -DPITCHEE_ORT_INCLUDE_DIR="$PWD/third_party/onnxruntime-ios/onnxruntime.framework/Headers" \
  -DPITCHEE_ORT_LIBRARY="$PWD/third_party/onnxruntime-ios/onnxruntime.framework/onnxruntime"

cmake --build build-ios -j 8
```

For Xcode integration:

1. Add `onnxruntime.xcframework` to the app target.
2. Set **Embed & Sign** or **Do Not Embed** according to the archive type and
   signing setup.
3. Link `libpitchee_core.a` or your packaged `PitcheeCore.xcframework`.
4. Keep the `models/` directory in the app bundle or another readable folder.

The Swift bridge example is in `platform/ios/PitcheeAnalyzer.swift`.

## Model placement

The application must make this directory readable at runtime:

```text
models/
  SileroVAD.onnx
  ECAPAFrontend.onnx
  ECAPA.onnx
  VFPHead.onnx
  SwiftF0.onnx
  Naturalness.onnx
  manifest.json
```

Pass that directory to `pitchee_analyzer_create()`. The library does not search
an app bundle and does not download models automatically.

## Testing

```bash
ctest --test-dir build --output-on-failure
./build/pitchee_cli ./models /path/to/audio.wav
```

The static C ABI consumer example is in `examples/consumer`.
