# PitcheeCore C++

PitcheeCore 是一个独立、跨平台的 C++17 语音预测库。输入一段音频，其返回该音频的女性化分数及各指标（如基频F0、语音自然度、语音顺性别女性概率以及最后的综合分）。它只负责音频分析、模型推理和数值计算，不依赖任何 UI 框架，也不依赖 Pitchee。

## 项目用途

本项目主要用于：

- 辅助跨性别女性进行声音训练，帮助她们观察长期变化和练习效果。
- 给语音训练师、言语治疗师和教学人员提供可量化的课堂反馈。
- 研究声音女性化感知、音高、自然度和模型评分之间的关系。

分数用于训练反馈和趋势观察，不应作为性别身份鉴定、医疗诊断、就业、
服务准入或其他高风险决策的唯一依据。

## 功能

- VFP 语音顺性别女性概率分析
- Naturalness 语音自然度分析
- SwiftF0 基频分析
- 自定义综合分计算方法
- UTF-8 JSON 数据输出

## Quick Start

### 1. 获取 ONNX Runtime

本仓库当前使用并验证的是 ONNX Runtime `1.24.2`。请尽量在同一个项目中
统一使用相同版本。官方发布页：

<https://github.com/microsoft/onnxruntime/releases/tag/v1.24.2>

直接下载链接：

| 平台 | 架构 | 下载 |
| --- | --- | --- |
| macOS | Apple Silicon arm64 | [onnxruntime-osx-arm64-1.24.2.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-osx-arm64-1.24.2.tgz) |
| Windows | x64 | [onnxruntime-win-x64-1.24.2.zip](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-win-x64-1.24.2.zip) |
| Windows | arm64 | [onnxruntime-win-arm64-1.24.2.zip](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-win-arm64-1.24.2.zip) |
| Linux | x64 | [onnxruntime-linux-x64-1.24.2.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz) |
| Linux | aarch64 | [onnxruntime-linux-aarch64-1.24.2.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-aarch64-1.24.2.tgz) |
| Android | AAR，多 ABI | [onnxruntime-android-1.24.2.aar](https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.24.2/onnxruntime-android-1.24.2.aar) |
| iOS | C/C++ pod archive | [pod-archive-onnxruntime-c-1.24.2.zip](https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.24.2.zip) |

官方安装文档：
<https://onnxruntime.ai/docs/install/>

### 2. macOS 快速构建

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

Intel Mac 需要使用对应 x86_64 的 ONNX Runtime 构建；`1.24.2` 官方发布包
提供的是 Apple Silicon arm64 包。

### 3. Windows 快速构建

PowerShell：

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

运行时需要把 `third_party\onnxruntime\bin\onnxruntime.dll` 放到可执行文件
旁边，或者加入 `PATH`。

### 4. Linux 快速构建

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

Linux 运行时需要设置 rpath，或把 `libonnxruntime.so` 放入标准动态库搜索路径。

### 5. Android 和 iOS

Android AAR 同时包含头文件和 `arm64-v8a`、`armeabi-v7a`、`x86`、`x86_64`
的 `libonnxruntime.so`。解压后把 `headers` 和一个 ABI 的 `.so` 传给 CMake：

```bash
curl -L -o /tmp/onnxruntime-android.aar \
  https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.24.2/onnxruntime-android-1.24.2.aar
unzip /tmp/onnxruntime-android.aar -d third_party/onnxruntime-android

cmake -S . -B build-android -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  -DPITCHEE_BUILD_CLI=OFF \
  -DPITCHEE_ORT_INCLUDE_DIR="$PWD/third_party/onnxruntime-android/headers" \
  -DPITCHEE_ORT_LIBRARY="$PWD/third_party/onnxruntime-android/jni/arm64-v8a/libonnxruntime.so"
```

iOS：

```bash
curl -L -o /tmp/onnxruntime-ios.zip \
  https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.24.2.zip
unzip /tmp/onnxruntime-ios.zip -d third_party/onnxruntime-ios
```

然后把 `onnxruntime.framework` 链接并嵌入 Xcode 工程。详细路径和 CMake 命令见
[docs/BUILDING.md](docs/BUILDING.md)。

### 6. 运行预测

```bash
./build-macos/pitchee_cli ./models /path/to/audio.wav
```

CLI 支持 PCM16、PCM32 和 Float32 WAV。M4A、MP3、AAC 等压缩格式应通过
AVFoundation、MediaCodec 或桌面解码器转换成 Float32 PCM。

Core 的 16 kHz 重采样与 FFmpeg `swresample` 的默认参数对齐，并在推理前量化
为 PCM16。这样桌面站点、服务器和移动端会让模型接收到同一种数值表示，避免
自然度模型对微小浮点差异过度敏感。

## 最小 C 调用

```c
#include <pitchee/pitchee.h>
#include <stdio.h>

int main(void) {
    pitchee_analyzer_options_t options = {2, 0, 0};
    pitchee_analyzer_t* analyzer = NULL;
    char error[1024] = {0};

    pitchee_status_t status = pitchee_analyzer_create(
        "./models",
        &options,
        &analyzer,
        error,
        sizeof(error)
    );
    if (status != PITCHEE_SUCCESS) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }

    char* json = NULL;
    status = pitchee_analyzer_analyze_wav_file(
        analyzer,
        "/path/to/audio.wav",
        NULL,
        NULL,
        &json,
        error,
        sizeof(error)
    );

    if (status == PITCHEE_SUCCESS) {
        puts(json);
        pitchee_string_free(json);
    } else {
        fprintf(stderr, "%s\n", error);
    }

    pitchee_analyzer_destroy(analyzer);
    return status == PITCHEE_SUCCESS ? 0 : 1;
}
```

### 详细进度回调

旧版 `pitchee_phase_callback_t` 继续保留。需要细粒度进度时，使用：

```c
void progress_callback(const pitchee_progress_t* progress, void* user_data) {
    switch (progress->stage) {
        case PITCHEE_PROGRESS_STAGE_LOADING_AUDIO:
        case PITCHEE_PROGRESS_STAGE_RESAMPLING_AUDIO:
        case PITCHEE_PROGRESS_STAGE_ANALYZING_F0:
        case PITCHEE_PROGRESS_STAGE_DETECTING_SPEECH:
        case PITCHEE_PROGRESS_STAGE_PREPARING_VFP_WINDOWS:
        case PITCHEE_PROGRESS_STAGE_EXTRACTING_VFP_EMBEDDINGS:
        case PITCHEE_PROGRESS_STAGE_CLASSIFYING_VFP_WINDOWS:
        case PITCHEE_PROGRESS_STAGE_PREPARING_NATURALNESS_WINDOWS:
        case PITCHEE_PROGRESS_STAGE_EXTRACTING_NATURALNESS_EMBEDDINGS:
        case PITCHEE_PROGRESS_STAGE_SCORING_NATURALNESS_WINDOWS:
        case PITCHEE_PROGRESS_STAGE_CALCULATING_SCORES:
        case PITCHEE_PROGRESS_STAGE_SERIALIZING_RESULT:
        case PITCHEE_PROGRESS_STAGE_COMPLETED:
            break;
        default:
            return;
    }

    /* progress->completed / progress->total 是当前阶段内的计数。 */
    /* progress->fraction 是当前阶段内的 0-1 比例。 */
    (void)user_data;
}

status = pitchee_analyzer_analyze_wav_file_with_progress(
    analyzer,
    "/path/to/audio.wav",
    progress_callback,
    NULL,
    &json,
    error,
    sizeof(error)
);
```

详细进度阶段均使用 `pitchee_progress_stage_t` 枚举常量，不返回阶段字符串。
`DETECTING_SPEECH`、VFP 批处理、自然度窗口评分会持续更新实际完成数；
单项任务阶段使用 `0/1` 和 `1/1`。

### 实时 F0

实时 F0 接口复用同一个 analyzer 中已加载的 SwiftF0 session。输入必须是
16 kHz mono Float32 PCM；Core 不负责麦克风采集或重采样。

```c
pitchee_realtime_f0_t* f0_stream = NULL;
pitchee_realtime_f0_options_t f0_options = {5120, 256, 0};

pitchee_realtime_f0_create(
    analyzer,
    &f0_options,
    &f0_stream,
    error,
    sizeof(error)
);

void f0_callback(const pitchee_f0_frame_t* frame, void* user_data) {
    if (frame->voiced) {
        /* frame->timestamp_seconds */
        /* frame->f0_hz */
        /* frame->confidence */
    }
}

pitchee_realtime_f0_process(
    f0_stream,
    samples,
    sample_count,
    f0_callback,
    NULL,
    NULL,
    error,
    sizeof(error)
);

pitchee_realtime_f0_reset(f0_stream);
pitchee_realtime_f0_destroy(f0_stream);
```

默认参数：

| 字段 | 默认值 | 含义 |
| --- | ---: | --- |
| `context_samples` | `5120` | 每次推理使用的最近 320 ms 音频 |
| `hop_samples` | `256` | 每收到 16 ms 新音频更新一次 |

SwiftF0 每帧时间步长是 256 samples。Core 使用重叠上下文维持低频稳定性，
并对重复帧按整数 sample index 去重。`voiced` 为 Core 的显示判定，当前要求
置信度大于 `0.9` 且 F0 位于 `75–600 Hz`。

同一个实时流必须串行调用；不同流可以使用不同 analyzer。analyzer 的生命周期
必须长于其创建的实时流。

## Output

输入一段有效音频后，会返回一个 UTF-8 JSON。下面是示例，数组内容省略了一部分：

```json
{
  "schema_version": 2,
  "model_version": "2026-09",
  "audio": {
    "source_sample_rate": 16000,
    "source_channels": 1,
    "input_seconds": 5.248,
    "analyzed_seconds": 5.248
  },
  "vad": {
    "segment_count": 4,
    "speech_seconds": 1.505,
    "silero_segment_count": 4,
    "discarded_breath_like_count": 0,
    "trimmed_segment_count": 4,
    "segments": [
      {
        "start_seconds": 0.753,
        "end_seconds": 1.173,
        "speech_start_seconds": 0.0,
        "speech_end_seconds": 0.42
      }
    ]
  },
  "f0": {
    "window_seconds": 0.1,
    "mean_hz": 169.1802,
    "standard_deviation_hz": 10.8073,
    "voiced_frame_count": 88,
    "voiced_window_count": 17,
    "windows": [
      {"start_seconds": 0.0, "end_seconds": 0.1, "f0_hz": null},
      {"start_seconds": 0.7, "end_seconds": 0.8, "f0_hz": 187.3043}
    ]
  },
  "vfp": {
    "vfp_standard_score": 3.47584,
    "window_count": 1,
    "window_duration_seconds": 1.505,
    "windows": [
      {"start_seconds": 0.753, "end_seconds": 4.625, "vfp_standard_score": 3.47584}
    ]
  },
  "naturalness": {
    "score": 41.2709,
    "window_count": 3,
    "window_duration_seconds": 1.515,
    "windows": [
      {"start_seconds": 0.0, "end_seconds": 1.515, "score": 31.2666},
      {"start_seconds": 1.8665, "end_seconds": 3.3815, "score": 22.9298},
      {"start_seconds": 3.733, "end_seconds": 5.248, "score": 80.8774}
    ]
  },
  "composite": {
    "base_score": 12.1184,
    "final_score": 12.1184,
    "cap": null,
    "rule": "continuous",
    "limited": false,
    "boosted": false
  }
}
```

### 顶层字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `schema_version` | 整数 | JSON 数据契约版本。结构不兼容变化时递增。当前为 `2`。 |
| `model_version` | 字符串 | 模型组合版本。当前为 `2026-09`。模型更新后应同步更新。 |

### `audio`

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `source_sample_rate` | Hz | 调用方传入的原始采样率。 |
| `source_channels` | 数量 | 原始声道数。`1` 为 mono；大于 1 为 interleaved。 |
| `input_seconds` | 秒 | 重采样到 16 kHz 后的输入时长。 |
| `analyzed_seconds` | 秒 | 实际分析的时长；当前不设最大时长限制。 |

`input_seconds` 是 Core 实际接收到的重采样 PCM 时长，不一定与压缩音频文件
容器声明的时长完全相同。Core 不再截断音频，因此 `analyzed_seconds` 当前等于
`input_seconds`。

### `vad`

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `segment_count` | 数量 | 经过 VAD 和呼吸过滤后保留的语音段数量。 |
| `speech_seconds` | 秒 | 所有保留语音段的时长总和。 |
| `silero_segment_count` | 数量 | Silero VAD 在过滤前的原始段数。 |
| `discarded_breath_like_count` | 数量 | 因周期性不足而删除的段数。 |
| `trimmed_segment_count` | 数量 | 首尾被裁剪过的语音段数量。 |
| `segments` | 数组 | VAD 保留的最终语音区间，VFP 和自然度都直接使用这些原始时间轴区间。 |

#### `vad.segments[]`

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `start_seconds` | 秒 | 语音段在原始音频时间轴上的起点。 |
| `end_seconds` | 秒 | 语音段在原始音频时间轴上的终点。 |
| `speech_start_seconds` | 秒 | 该段在拼接语音缓冲区中的起点。 |
| `speech_end_seconds` | 秒 | 该段在拼接语音缓冲区中的终点。 |

映射关系：

```text
end_seconds - start_seconds
= speech_end_seconds - speech_start_seconds
```

### `f0`

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `window_seconds` | 秒 | F0 时间轴窗口长度，固定为 `0.1`。 |
| `mean_hz` | Hz | 有效浊音帧的平均 F0。没有有效帧时为 `null`。 |
| `standard_deviation_hz` | Hz | 浊音 F0 的总体标准差，分母为帧数 `N`。 |
| `voiced_frame_count` | 数量 | 置信度大于 `0.9` 且 F0 在 `75–600 Hz` 的帧数。 |
| `voiced_window_count` | 数量 | 至少包含一个有效浊音帧的 0.1 秒窗口数。 |
| `windows` | 数组 | 原始分析音频时间轴上的 F0 时间序列。 |

#### `f0.windows[]`

| 字段 | 单位 | 含义 |
| --- | --- | --- |
| `start_seconds` | 秒 | 窗口在原始分析音频时间轴上的起点。 |
| `end_seconds` | 秒 | 窗口在原始分析音频时间轴上的终点。 |
| `f0_hz` | Hz 或 `null` | 窗口内有效浊音帧的平均 F0；没有有效帧时为 `null`。 |

### `vfp`

| 字段 | 类型/范围 | 含义 |
| --- | --- | --- |
| `vfp_standard_score` | `0–100` | 所有 VFP 窗口概率平均值乘以 100，不再返回冗余的 0–1 原始分。 |
| `window_count` | 数量 | VFP 窗口数量。 |
| `window_duration_seconds` | 秒 | 标称完整窗口长度，即 1.515 秒；短语音窗口的实际长度看各窗口的 start/end。 |
| `windows` | 数组 | 每个 VFP 窗口的时间和标准分。 |

#### `vfp.windows[]`

| 字段 | 单位/范围 | 含义 |
| --- | --- | --- |
| `start_seconds` | 秒 | 窗口映射到原始分析音频时间轴后的起点。 |
| `end_seconds` | 秒 | 窗口映射到原始分析音频时间轴后的终点。 |
| `vfp_standard_score` | `0–100` | 当前窗口概率乘以 100。 |

Core 不对语音段做拼接。长语音段按 1.515 秒窗口和 0.1 秒步长滑动；短于
1.515 秒的语音段直接按原长度送入模型，不补零。最后一个完整窗口对齐语音段
尾部，窗口不会跨越静音或相邻语音段。

### `naturalness`

| 字段 | 类型/范围 | 含义 |
| --- | --- | --- |
| `score` | `0–100` | 整段音频的聚合自然度分；50 是加减分分界。 |
| `window_count` | 数量 | 自然度窗口数量，与 VFP 窗口数量一致。 |
| `window_duration_seconds` | 秒 | 标称完整窗口长度，即 1.515 秒；短语音窗口的实际长度看各窗口的 start/end。 |
| `windows` | 数组 | 每个自然度窗口的时间轴位置和分数。 |

#### `naturalness.windows[]`

| 字段 | 单位/范围 | 含义 |
| --- | --- | --- |
| `start_seconds` | 秒 | 窗口在原始分析音频时间轴上的起点。 |
| `end_seconds` | 秒 | 窗口在原始分析音频时间轴上的终点。 |
| `score` | `0–100` | 使用当前窗口嵌入特征和整段音频共享标准差计算的局部自然度分。 |

聚合 `naturalness.score` 使用所选自然度语音窗口的均值和标准差组成模型输入；
`naturalness.windows[].score` 用于时间轴展示，不改变综合分计算。

自然度使用与 VFP 完全相同的窗口集合，窗口数量、起点和终点一一对应。长语音段
使用 1.515 秒完整窗口；短语音段使用原长窗口，不补零。

### `composite`

| 字段 | 类型/范围 | 含义 |
| --- | --- | --- |
| `base_score` | `0–100` | 规则提升或封顶之前的连续综合分。 |
| `final_score` | `0–100` | 应用规则后的最终综合分。 |
| `cap` | 数值或 `null` | 本次选择的封顶值；无 cap 时为 `null`。 |
| `rule` | 字符串 | 本次命中的评分规则。 |
| `limited` | 布尔值 | 封顶是否实际降低了分数。 |
| `boosted` | 布尔值 | 及格提升是否实际提高了分数。 |

## 综合分规则

令：

```text
S  = vfp.vfp_standard_score / 100
N  = naturalness.score
F0 = f0.mean_hz

Sr = S
Nr = clamp((N - 40) / 50, 0, 1)
Fr = clamp((F0 - 110) / 90, 0, 1)

base_score = 100 × (
    0.50 × Sr
  + 0.20 × Nr
  + 0.15 × Fr
  + 0.15 × Sr × Nr × Fr
)
```

规则按以下顺序判断：

| `rule` | 条件 | 处理 |
| --- | --- | --- |
| `pass_boost` | `F0 > 165`，`N > 80`，`S > 50` | 平滑提升，最高到 100 |
| `high_f0_stylized_cap` | `F0 > 165`，`N < 50`，`S` 任意 | 最高 30 |
| `low_f0_natural_cap` | `F0 <= 165`，`N >= 50` | 最高 59 |
| `low_f0_stylized_cap` | `F0 <= 165`，`N < 50` | 最高 20 |
| `high_f0_male_cap` | `F0 > 165`，`N >= 50`，`S < 50` | 最高 59 |
| `continuous` | 未命中以上规则 | 直接使用 `base_score` |
| `f0_unavailable` | 没有有效 F0 | 直接使用 `vfp.vfp_standard_score` |

及格提升的公式：

```text
strength = min(
    (F0 - 165) / 25,
    (N - 80) / 20,
    (S - 50) / 30,
    1
)

promoted = 60 + 40 × strength

如果 promoted > base_score，则 final_score = promoted。
```

最终分数始终限制在 `0–100`。只有当 cap 真正降低分数时 `limited` 才为
`true`；只有及格提升真正提高分数时 `boosted` 才为 `true`。

## 模型目录

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

调用 `pitchee_analyzer_create(model_directory, ...)` 时，目录中必须存在这些
文件。模型只加载一次，Core 不需要网络，也不会自动下载模型。

`VFPHead.onnx`、`Naturalness.onnx` 和官方 `SwiftF0.onnx` 均使用未压缩的
FP32 权重。`ECAPA.onnx` 当前仍为 FP16 权重，用于控制移动端包体和内存占用。
`ECAPAFrontend.onnx` 和 `ECAPA.onnx` 的时间维是动态的，因此 VAD 短语音段
可以按原长度直接推理，不需要补齐到 1.515 秒。

## 线程与生命周期

- 模型在 `pitchee_analyzer_create()` 中加载。
- 同一个 analyzer 可以重复分析多段音频。
- 单个 analyzer 的调用应串行；或者每个 worker 创建自己的 analyzer。
- 用 `pitchee_analyzer_destroy()` 释放 analyzer。
- 用 `pitchee_string_free()` 释放返回的 JSON。

## 论文、成果与上游项目

### Voice Passing / VFP

本项目综合分所参考的 Voice Femininity Percentage 方法来自：

```bibtex
@inproceedings{doukhan2023voicepassing,
  title={Voice Passing: a Non-Binary Voice Gender Prediction System for evaluating Transgender voice transition},
  author={David Doukhan and Simon Devauchelle and Lucile Girard-Monneron and Mía Chávez Ruz and V. Chaddouk and Isabelle Wagner and Albert Rilliard},
  booktitle={Proc. INTERSPEECH 2023},
  pages={5207--5211},
  year={2023},
  doi={10.21437/Interspeech.2023-1835},
  url={https://arxiv.org/abs/2404.15176}
}
```

论文链接：<https://arxiv.org/abs/2404.15176>

### SwiftF0

音高检测使用 SwiftF0 ONNX 模型：

```bibtex
@misc{nieradzik2025swiftf0,
  title={SwiftF0: Fast and Accurate Monophonic Pitch Detection},
  author={Lars Nieradzik},
  year={2025},
  eprint={2508.18440},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2508.18440}
}
```

项目地址：<https://github.com/lars76/swift-f0>

### ECAPA-TDNN

ECAPA speaker embedding 来自：

```bibtex
@inproceedings{desplanques2020ecapa,
  title={ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification},
  author={Brecht Desplanques and Jenthe Thienpondt and Kris Demuynck},
  booktitle={Proc. INTERSPEECH 2020},
  year={2020},
  url={https://arxiv.org/abs/2005.07143}
}
```

### SpeechBrain

ECAPA 前端和预训练 speaker encoder 使用 SpeechBrain：

```bibtex
@article{ravanelli2024speechbrain,
  title={Open-Source Conversational AI with SpeechBrain 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and others},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={333},
  year={2024},
  url={http://jmlr.org/papers/v25/24-0991.html}
}
```

项目地址：<https://github.com/speechbrain/speechbrain>

### Silero VAD

```bibtex
@misc{silerovad2024,
  author={Silero Team},
  title={Silero VAD: pre-trained enterprise-grade Voice Activity Detector},
  year={2024},
  howpublished={GitHub},
  url={https://github.com/snakers4/silero-vad}
}
```

### ONNX Runtime

ONNX Runtime 用于所有平台的模型执行：

<https://github.com/microsoft/onnxruntime>

## 许可证

| 组件 | 许可证 |
| --- | --- |
| SwiftF0 模型与算法 | MIT，许可证位于 `models/SWIFT_F0_LICENSE` |
| Silero VAD | MIT |
| SpeechBrain | Apache License 2.0 |
| ONNX Runtime | MIT |
| PitcheeCore 和项目训练的模型 | **GNU GPL v3.0（GPL-3.0-only）** |

完整许可证正文见 [LICENSE](LICENSE)。第三方模型、运行库和上游项目仍分别适用
其原始许可证；GPL v3 不会替换这些第三方许可证。

更完整的构建说明见 [docs/BUILDING.md](docs/BUILDING.md)，C API 细节见
[docs/API.md](docs/API.md)。
