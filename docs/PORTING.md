# Platform integration

## What the shared core owns

- FFmpeg-compatible PCM resampling to 16 kHz
- Silero VAD and breath filtering
- ECAPA log-Mel frontend
- ECAPA embeddings
- VFP probabilities
- Naturalness aggregate and per-window scores
- 0.05-second SwiftF0 pitch timeline
- Realtime chunk-buffered SwiftF0 frames
- Streaming STFT spectrum frames and spectral summary metrics
- Composite score

## What each platform owns

### iOS

- AVAudioEngine or AVAudioRecorder for capture
- AVAudioFile or AVAssetReader for file decoding
- AVAudioPlayer for playback
- SwiftUI or another UI framework for presentation

### Android

- AudioRecord or MediaRecorder for capture
- MediaExtractor/MediaCodec or ExoPlayer for decoding
- AudioTrack/ExoPlayer for playback
- Jetpack Compose or another UI framework for presentation

### Desktop

- Platform audio APIs or a small WAV decoder
- Desktop UI toolkit of choice

Every platform should pass Float32 PCM to PitcheeCore. Do not reimplement the
mel frontend, VAD hysteresis, model windowing, or score rules on the platform
side. Timeline geometry and visual colors belong to the UI layer, not the core.

## ABI rules

- Keep `pitchee.h` C-compatible.
- Never expose `std::string`, `std::vector`, or exceptions across the ABI.
- Return errors through `pitchee_status_t` and a fixed caller buffer.
- Return dynamic results as UTF-8 JSON or opaque handles.
- Keep model paths and file decoding outside the shared ABI.
- Increment the model version whenever model assets change.
