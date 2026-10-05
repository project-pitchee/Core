#include "internal.hpp"
#include "ort_runtime.hpp"
#include "vad.hpp"
#include "wav_reader.hpp"

#include "pitchee/pitchee.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

struct pitchee_analyzer_t {
    std::filesystem::path model_directory;
    std::string model_version = "unknown";
    int intra_op_threads = 2;
    bool use_coreml = true;
    std::unique_ptr<pitchee::OrtModel> ecapa_frontend;
    std::unique_ptr<pitchee::OrtModel> ecapa;
    std::unique_ptr<pitchee::OrtModel> vfp_head;
    std::unique_ptr<pitchee::OrtModel> swift_f0;
    std::unique_ptr<pitchee::VadDetector> vad;
    std::unique_ptr<pitchee::NaturalnessModel> naturalness;
};

struct pitchee_realtime_f0_t {
    pitchee::OrtModel* model = nullptr;
    size_t context_samples = 5120;
    size_t hop_samples = 256;
    size_t buffer_start_sample = 0;
    size_t total_samples = 0;
    size_t samples_since_inference = 0;
    bool has_emitted = false;
    int64_t last_emitted_center_sample = -1;
    std::vector<float> buffer;
};

struct pitchee_realtime_resonance_t {
    pitchee_realtime_f0_t* f0_stream = nullptr;
    std::unique_ptr<pitchee::OrtModel> formant_model;
    std::string vowel;
    int vowel_index = 3;
    size_t formant_window_samples = 3200;
    size_t context_samples = 5120;
    size_t buffer_start_sample = 0;
    size_t total_samples = 0;
    size_t emitted_frames = 0;
    std::vector<float> buffer;
    std::string callback_error;
    pitchee_resonance_frame_callback_t frame_callback = nullptr;
    void* user_data = nullptr;
};

namespace {

constexpr double kF0WindowSeconds = 0.05;

bool valid_score_profile(pitchee_score_profile_t score_profile) {
    return score_profile == PITCHEE_SCORE_PROFILE_FEMINIZATION
        || score_profile == PITCHEE_SCORE_PROFILE_MASCULINIZATION;
}

struct ProgressReporter {
    pitchee_phase_callback_t phase_callback = nullptr;
    void* phase_user_data = nullptr;
    pitchee_progress_callback_t progress_callback = nullptr;
    void* progress_user_data = nullptr;

    void phase(pitchee_analysis_phase_t value) const {
        if (phase_callback) phase_callback(value, phase_user_data);
    }

    void progress(
        pitchee_progress_stage_t stage,
        uint64_t completed,
        uint64_t total
    ) const {
        if (!progress_callback) return;
        pitchee_progress_t progress{};
        progress.stage = stage;
        progress.completed = std::min(completed, total);
        progress.total = total;
        progress.fraction = total > 0
            ? static_cast<double>(progress.completed) / total
            : (stage == PITCHEE_PROGRESS_STAGE_COMPLETED ? 1.0 : 0.0);
        progress_callback(&progress, progress_user_data);
    }
};

void set_error(char* target, size_t capacity, const std::string& message) {
    if (!target || capacity == 0) return;
    const size_t count = std::min(capacity - 1, message.size());
    std::memcpy(target, message.data(), count);
    target[count] = '\0';
}

pitchee_status_t status_for_exception(const std::exception& error) {
    const std::string message = error.what();
    if (message.find("without ONNX Runtime") != std::string::npos) {
        return PITCHEE_ERROR_ORT_UNAVAILABLE;
    }
    if (message.find("Silero VAD detected no speech") != std::string::npos) {
        return PITCHEE_ERROR_NO_SPEECH;
    }
    if (message.find("unsupported WAV") != std::string::npos) {
        return PITCHEE_ERROR_UNSUPPORTED_FORMAT;
    }
    if (message.find("WAV") != std::string::npos
        || message.find("unable to open") != std::string::npos) {
        return PITCHEE_ERROR_IO;
    }
    if (message.find("model") != std::string::npos
        || message.find("ONNX") != std::string::npos) {
        return PITCHEE_ERROR_MODEL;
    }
    return PITCHEE_ERROR_INTERNAL;
}

pitchee::PitchResult analyze_pitch(
    pitchee::OrtModel& model,
    const std::vector<float>& samples,
    const ProgressReporter& reporter
) {
    reporter.progress(PITCHEE_PROGRESS_STAGE_ANALYZING_F0, 0, 1);
    std::vector<float> input = samples;
    if (input.size() < 256) input.resize(256, 0.0f);
    pitchee::Tensor tensor;
    tensor.shape = {1, static_cast<int64_t>(input.size())};
    tensor.values = std::move(input);
    auto outputs = model.run_all({{"input_audio", std::move(tensor)}});
    if (outputs.size() < 2 || outputs[0].values.size() != outputs[1].values.size()) {
        throw std::runtime_error("SwiftF0 returned invalid output");
    }

    pitchee::PitchResult result;
    result.pitch_hz = std::move(outputs[0].values);
    result.confidence = std::move(outputs[1].values);
    result.timestamps.resize(result.pitch_hz.size());
    result.voicing.resize(result.pitch_hz.size(), 0);
    std::vector<float> voiced_pitch;
    const size_t f0_window_samples = static_cast<size_t>(
        kF0WindowSeconds * pitchee::kSampleRate
    );
    const size_t f0_window_count = std::max<size_t>(
        1,
        (samples.size() + f0_window_samples - 1) / f0_window_samples
    );
    std::vector<double> f0_window_sums(f0_window_count, 0.0);
    std::vector<size_t> f0_window_counts(f0_window_count, 0);
    for (size_t index = 0; index < result.pitch_hz.size(); ++index) {
        result.timestamps[index] = static_cast<float>(
            (static_cast<double>(index * 256) + 127.5) / 16000.0
        );
        const bool voiced =
            result.confidence[index] > 0.9f
            && result.pitch_hz[index] >= 75.0f
            && result.pitch_hz[index] <= 600.0f;
        result.voicing[index] = voiced ? 1 : 0;
        if (voiced) {
            voiced_pitch.push_back(result.pitch_hz[index]);
            const size_t window_index = std::min(
                f0_window_count - 1,
                static_cast<size_t>(
                    result.timestamps[index] / kF0WindowSeconds
                )
            );
            f0_window_sums[window_index] += result.pitch_hz[index];
            ++f0_window_counts[window_index];
        }
    }

    const double total_seconds = static_cast<double>(samples.size())
        / pitchee::kSampleRate;
    result.windows.reserve(f0_window_count);
    for (size_t index = 0; index < f0_window_count; ++index) {
        pitchee::F0Window window;
        window.start_seconds = index * kF0WindowSeconds;
        window.end_seconds = std::min(
            total_seconds,
            (index + 1) * kF0WindowSeconds
        );
        if (f0_window_counts[index] > 0) {
            window.has_f0 = true;
            window.f0_hz = f0_window_sums[index]
                / static_cast<double>(f0_window_counts[index]);
            ++result.voiced_window_count;
        }
        result.windows.push_back(window);
    }

    if (!voiced_pitch.empty()) {
        const double mean = std::accumulate(
            voiced_pitch.begin(),
            voiced_pitch.end(),
            0.0
        ) / voiced_pitch.size();
        double variance = 0.0;
        for (const float value : voiced_pitch) {
            const double delta = value - mean;
            variance += delta * delta;
        }
        result.has_mean_f0 = true;
        result.mean_f0_hz = mean;
        result.standard_deviation_f0_hz = std::sqrt(variance / voiced_pitch.size());
        result.voiced_frame_count = static_cast<int>(voiced_pitch.size());
    }
    reporter.progress(PITCHEE_PROGRESS_STAGE_ANALYZING_F0, 1, 1);
    return result;
}

constexpr int kFormantWindowSamples = 512;
constexpr int kFormantHopSamples = 80;
constexpr int kFormantBins = 257;
constexpr float kFormantPreemphasis = 0.98f;
constexpr float kFormantFloor = 0.001f;

void fft_radix2(std::vector<std::complex<double>>& values) {
    const size_t size = values.size();
    for (size_t index = 1, reversed = 0; index < size; ++index) {
        size_t bit = size >> 1;
        for (; reversed & bit; bit >>= 1) reversed ^= bit;
        reversed ^= bit;
        if (index < reversed) std::swap(values[index], values[reversed]);
    }
    constexpr double kPi = 3.14159265358979323846;
    for (size_t length = 2; length <= size; length <<= 1) {
        const double angle = -2.0 * kPi / static_cast<double>(length);
        const std::complex<double> step(std::cos(angle), std::sin(angle));
        for (size_t start = 0; start < size; start += length) {
            std::complex<double> phase(1.0, 0.0);
            const size_t half = length >> 1;
            for (size_t offset = 0; offset < half; ++offset) {
                const size_t even = start + offset;
                const size_t odd = even + half;
                const std::complex<double> product = values[odd] * phase;
                values[odd] = values[even] - product;
                values[even] += product;
                phase *= step;
            }
        }
    }
}

void smooth_formant_envelope(std::vector<float>& values) {
    for (int pass = 0; pass < 6; ++pass) {
        for (size_t index = 1; index + 1 < values.size(); ++index) {
            if (values[index] < values[index - 1]
                && values[index] < values[index + 1]) {
                values[index] = 0.5f * (values[index - 1] + values[index + 1]);
            }
        }
        for (size_t index = 1; index + 1 < values.size(); ++index) {
            if (values[index] <= values[index - 1]
                || values[index] <= values[index + 1]) {
                values[index] = 0.25f * values[index - 1]
                    + 0.5f * values[index]
                    + 0.25f * values[index + 1];
            }
        }
    }
}

std::array<float, 4> formantnet_predict(
    pitchee::OrtModel& model,
    const std::vector<float>& samples
) {
    if (samples.empty()) throw std::runtime_error("empty FormantNet input");
    std::vector<float> audio = samples;
    if (audio.size() < kFormantWindowSamples) {
        audio.insert(audio.begin(), kFormantWindowSamples - audio.size(), 0.0f);
    }
    std::vector<float> preemphasized = audio;
    for (size_t index = 1; index < preemphasized.size(); ++index) {
        preemphasized[index] -= kFormantPreemphasis * audio[index - 1];
    }
    std::vector<float> padded(
        kFormantWindowSamples / 2 + preemphasized.size() + kFormantHopSamples - 1,
        0.0f
    );
    std::copy(
        preemphasized.begin(),
        preemphasized.end(),
        padded.begin() + kFormantWindowSamples / 2
    );
    const size_t frame_count = padded.size() >= kFormantWindowSamples
        ? (padded.size() - kFormantWindowSamples) / kFormantHopSamples + 1
        : 0;
    if (frame_count == 0) throw std::runtime_error("FormantNet has no frames");

    std::vector<float> envelope(frame_count * kFormantBins, 0.0f);
    std::vector<double> window(kFormantWindowSamples, 0.0);
    double window_sum = 0.0;
    constexpr double kPi = 3.14159265358979323846;
    for (int index = 0; index < kFormantWindowSamples; ++index) {
        window[static_cast<size_t>(index)] = 0.5 - 0.5 * std::cos(
            2.0 * kPi * index / static_cast<double>(kFormantWindowSamples - 1)
        );
        window_sum += window[static_cast<size_t>(index)];
    }
    const float scaling = static_cast<float>(2.0 / window_sum);
    std::vector<std::complex<double>> fft(kFormantWindowSamples);
    std::vector<float> frame_values(kFormantBins);
    for (size_t frame = 0; frame < frame_count; ++frame) {
        const size_t start = frame * kFormantHopSamples;
        for (int index = 0; index < kFormantWindowSamples; ++index) {
            fft[static_cast<size_t>(index)] = std::complex<double>(
                padded[start + static_cast<size_t>(index)]
                    * window[static_cast<size_t>(index)],
                0.0
            );
        }
        fft_radix2(fft);
        for (int bin = 0; bin < kFormantBins; ++bin) {
            frame_values[static_cast<size_t>(bin)] = static_cast<float>(
                std::abs(fft[static_cast<size_t>(bin)])
            );
        }
        smooth_formant_envelope(frame_values);
        for (int bin = 0; bin < kFormantBins; ++bin) {
            envelope[frame * kFormantBins + static_cast<size_t>(bin)] =
                20.0f * std::log10(
                    scaling * frame_values[static_cast<size_t>(bin)] + kFormantFloor
                );
        }
    }
    const double mean = std::accumulate(
        envelope.begin(),
        envelope.end(),
        0.0
    ) / envelope.size();
    double variance = 0.0;
    for (const float value : envelope) {
        const double delta = value - mean;
        variance += delta * delta;
    }
    const double standard_deviation = std::sqrt(variance / envelope.size());
    for (float& value : envelope) {
        value = static_cast<float>(
            (value - mean) / std::max(standard_deviation, 1e-6)
        );
    }

    pitchee::Tensor input;
    input.shape = {
        1,
        static_cast<int64_t>(frame_count),
        kFormantBins,
    };
    input.values = std::move(envelope);
    const auto output = model.run({{"input", std::move(input)}});
    if (output.values.size() < frame_count * 20) {
        throw std::runtime_error("FormantNet returned invalid output");
    }

    std::array<int, 6> order{};
    std::array<double, 6> mean_frequency{};
    for (int pole = 0; pole < 6; ++pole) {
        order[static_cast<size_t>(pole)] = pole;
        double sum = 0.0;
        for (size_t frame = 0; frame < frame_count; ++frame) {
            sum += output.values[frame * 20 + static_cast<size_t>(pole)];
        }
        mean_frequency[static_cast<size_t>(pole)] = sum / frame_count;
    }
    std::sort(
        order.begin(),
        order.end(),
        [&](int left, int right) {
            return mean_frequency[static_cast<size_t>(left)]
                < mean_frequency[static_cast<size_t>(right)];
        }
    );
    std::vector<double> frequencies(frame_count * 6, 0.0);
    for (size_t frame = 0; frame < frame_count; ++frame) {
        for (int pole = 0; pole < 6; ++pole) {
            const int source = order[static_cast<size_t>(pole)];
            frequencies[frame * 6 + static_cast<size_t>(pole)] =
                output.values[frame * 20 + static_cast<size_t>(source)] * 8000.0;
        }
    }
    for (int pass = 0; pass < 10; ++pass) {
        std::vector<double> smoothed = frequencies;
        if (frame_count > 1) {
            for (int pole = 0; pole < 6; ++pole) {
                smoothed[static_cast<size_t>(pole)] =
                    0.75 * frequencies[static_cast<size_t>(pole)]
                    + 0.25 * frequencies[6 + static_cast<size_t>(pole)];
                smoothed[(frame_count - 1) * 6 + static_cast<size_t>(pole)] =
                    0.75 * frequencies[(frame_count - 1) * 6 + static_cast<size_t>(pole)]
                    + 0.25 * frequencies[(frame_count - 2) * 6 + static_cast<size_t>(pole)];
            }
        }
        for (size_t frame = 1; frame + 1 < frame_count; ++frame) {
            for (int pole = 0; pole < 6; ++pole) {
                smoothed[frame * 6 + static_cast<size_t>(pole)] =
                    0.5 * frequencies[frame * 6 + static_cast<size_t>(pole)]
                    + 0.25 * frequencies[(frame - 1) * 6 + static_cast<size_t>(pole)]
                    + 0.25 * frequencies[(frame + 1) * 6 + static_cast<size_t>(pole)];
            }
        }
        frequencies.swap(smoothed);
    }

    std::array<float, 4> result{};
    const size_t average_frames = std::min<size_t>(5, frame_count);
    for (int pole = 0; pole < 4; ++pole) {
        double sum = 0.0;
        for (size_t frame = frame_count - average_frames; frame < frame_count; ++frame) {
            sum += frequencies[frame * 6 + static_cast<size_t>(pole)];
        }
        result[static_cast<size_t>(pole)] = static_cast<float>(sum / average_frames);
    }
    return result;
}

struct ResonanceReference {
    std::array<double, 4> male_median_log;
    std::array<double, 4> female_median_log;
    std::array<double, 4> weights;
};

constexpr std::array<ResonanceReference, 4> kResonanceReferences{{
    {{{5.916853427886963, 7.757352828979492, 7.969632148742676, 8.158721923828125}},
     {{6.174807548522949, 7.924232482910156, 8.048027038574219, 8.274372100830078}},
     {{0.27419710572769995, 0.38775370041711105, 0.1467223412578838, 0.19132685259730509}}},
    {{{6.022394180297852, 6.91353178024292, 7.786130428314209, 8.179064750671387}},
     {{6.212732315063477, 6.970745086669922, 7.920019149780273, 8.295587539672852}},
     {{0.27937603047131476, 0.17954452622558248, 0.24746179272875632, 0.2936176505743465}}},
    {{{6.437050819396973, 7.464291572570801, 7.837002277374268, 8.186055183410645}},
     {{6.629748821258545, 7.600702285766602, 7.937709331512451, 8.309038162231445}},
     {{0.2928590143908253, 0.32897667116064094, 0.17179146310309187, 0.2063728513454419}}},
    {{{6.627399921417236, 7.250248432159424, 7.8247971534729, 8.222831726074219}},
     {{6.8298187255859375, 7.397768020629883, 7.941613674163818, 8.295548439025879}},
     {{0.32187276609775717, 0.28767021980082047, 0.19948711527053295, 0.19096989883088925}}},
}};

bool parse_resonance_vowel(const std::string& vowel, int* index) {
    if (!index) return false;
    if (vowel == "i") *index = 0;
    else if (vowel == "u") *index = 1;
    else if (vowel == "\xC3\xA6" || vowel == "ae") *index = 2;
    else if (vowel == "\xC9\x91" || vowel == "a" || vowel == "A") *index = 3;
    else return false;
    return true;
}

float resonance_score(
    int vowel_index,
    const std::array<float, 4>& formants_hz
) {
    const auto index = static_cast<size_t>(vowel_index);
    if (index >= kResonanceReferences.size()) return -1.0f;
    const auto& reference = kResonanceReferences[index];
    double position = 0.0;
    for (size_t formant = 0; formant < 4; ++formant) {
        const double value = std::log(std::max(
            static_cast<double>(formants_hz[formant]),
            1.0
        ));
        const double normalized = (
            value - reference.male_median_log[formant]
        ) / (
            reference.female_median_log[formant]
            - reference.male_median_log[formant]
        );
        position += reference.weights[formant] * normalized;
    }
    const double logit = -2.0 * std::log(1.0 / 3.0) * (position - 0.5);
    return static_cast<float>(100.0 / (1.0 + std::exp(-logit)));
}

void realtime_resonance_f0_callback(
    const pitchee_f0_frame_t* frame,
    void* user_data
) {
    auto& stream = *static_cast<pitchee_realtime_resonance_t*>(user_data);
    if (!frame || frame->voiced == 0 || !stream.callback_error.empty()) return;
    try {
        const int64_t center_sample = static_cast<int64_t>(std::llround(
            frame->timestamp_seconds * pitchee::kSampleRate
        ));
        const int64_t start_sample = center_sample
            - static_cast<int64_t>(stream.formant_window_samples);
        const int64_t end_sample = center_sample;
        const int64_t buffer_end = static_cast<int64_t>(
            stream.buffer_start_sample + stream.buffer.size()
        );
        if (start_sample < static_cast<int64_t>(stream.buffer_start_sample)
            || end_sample > buffer_end) {
            return;
        }
        const size_t local_start = static_cast<size_t>(
            start_sample - static_cast<int64_t>(stream.buffer_start_sample)
        );
        std::vector<float> window(
            stream.buffer.begin() + static_cast<std::ptrdiff_t>(local_start),
            stream.buffer.begin() + static_cast<std::ptrdiff_t>(local_start)
                + static_cast<std::ptrdiff_t>(stream.formant_window_samples)
        );
        const auto formants = formantnet_predict(*stream.formant_model, window);
        pitchee_resonance_frame_t output{};
        output.timestamp_seconds = frame->timestamp_seconds;
        output.f0_hz = frame->f0_hz;
        output.f0_confidence = frame->confidence;
        output.f1_hz = formants[0];
        output.f2_hz = formants[1];
        output.f3_hz = formants[2];
        output.f4_hz = formants[3];
        output.resonance_score = resonance_score(stream.vowel_index, formants);
        output.vowel = stream.vowel.c_str();
        output.voiced = 1;
        ++stream.emitted_frames;
        stream.frame_callback(&output, stream.user_data);
    } catch (const std::exception& error) {
        stream.callback_error = error.what();
    }
}

std::vector<std::vector<float>> embed_waveforms(
    pitchee::OrtModel& frontend,
    pitchee::OrtModel& encoder,
    const std::vector<std::vector<float>>& waveforms,
    const ProgressReporter& reporter,
    pitchee_progress_stage_t stage
) {
    std::vector<std::vector<float>> embeddings(waveforms.size());
    std::map<size_t, std::vector<size_t>> groups;
    for (size_t index = 0; index < waveforms.size(); ++index) {
        if (waveforms[index].empty()) continue;
        groups[waveforms[index].size()].push_back(index);
    }
    if (groups.empty()) throw std::runtime_error("no waveform windows");

    reporter.progress(stage, 0, waveforms.size());
    size_t completed = 0;
    for (const auto& [window_samples, indices] : groups) {
        for (size_t batch_start = 0;
             batch_start < indices.size();
             batch_start += pitchee::kEmbeddingBatchSize) {
            const size_t real_count = std::min<size_t>(
                pitchee::kEmbeddingBatchSize,
                indices.size() - batch_start
            );
            pitchee::Tensor input;
            input.shape = {
                pitchee::kEmbeddingBatchSize,
                static_cast<int64_t>(window_samples),
            };
            input.values.reserve(
                static_cast<size_t>(pitchee::kEmbeddingBatchSize)
                * window_samples
            );
            for (size_t index = 0; index < pitchee::kEmbeddingBatchSize; ++index) {
                if (index < real_count) {
                    const auto& waveform = waveforms[indices[batch_start + index]];
                    input.values.insert(
                        input.values.end(),
                        waveform.begin(),
                        waveform.end()
                    );
                } else {
                    input.values.insert(input.values.end(), window_samples, 0.0f);
                }
            }

            auto features = frontend.run({{"waveforms", std::move(input)}});
            auto batch_embeddings = encoder.run(
                {{"features", std::move(features)}}
            );
            if (batch_embeddings.values.size()
                < static_cast<size_t>(pitchee::kEmbeddingBatchSize)
                    * pitchee::kEmbeddingDimensions) {
                throw std::runtime_error("ECAPA returned invalid embeddings");
            }
            for (size_t index = 0; index < real_count; ++index) {
                const auto start = batch_embeddings.values.begin()
                    + static_cast<std::ptrdiff_t>(
                        index * pitchee::kEmbeddingDimensions
                    );
                embeddings[indices[batch_start + index]].assign(
                    start,
                    start + pitchee::kEmbeddingDimensions
                );
            }
            completed += real_count;
            reporter.progress(stage, completed, waveforms.size());
        }
    }
    return embeddings;
}

std::vector<double> classify_embeddings(
    pitchee::OrtModel& model,
    const std::vector<std::vector<float>>& embeddings,
    const ProgressReporter& reporter
) {
    std::vector<double> probabilities;
    probabilities.reserve(embeddings.size());
    reporter.progress(
        PITCHEE_PROGRESS_STAGE_CLASSIFYING_VFP_WINDOWS,
        0,
        embeddings.size()
    );
    for (size_t batch_start = 0;
         batch_start < embeddings.size();
         batch_start += pitchee::kEmbeddingBatchSize) {
        const size_t real_count = std::min<size_t>(
            pitchee::kEmbeddingBatchSize,
            embeddings.size() - batch_start
        );
        pitchee::Tensor input;
        input.shape = {
            pitchee::kEmbeddingBatchSize,
            pitchee::kEmbeddingDimensions,
        };
        input.values.reserve(
            static_cast<size_t>(pitchee::kEmbeddingBatchSize)
            * pitchee::kEmbeddingDimensions
        );
        for (size_t index = 0; index < pitchee::kEmbeddingBatchSize; ++index) {
            if (index < real_count) {
                input.values.insert(
                    input.values.end(),
                    embeddings[batch_start + index].begin(),
                    embeddings[batch_start + index].end()
                );
            } else {
                input.values.insert(
                    input.values.end(),
                    pitchee::kEmbeddingDimensions,
                    0.0f
                );
            }
        }
        auto output = model.run({{"embeddings", std::move(input)}});
        if (output.values.size() < real_count) {
            throw std::runtime_error("VFP classifier returned invalid output");
        }
        for (size_t index = 0; index < real_count; ++index) {
            probabilities.push_back(output.values[index]);
        }
        reporter.progress(
            PITCHEE_PROGRESS_STAGE_CLASSIFYING_VFP_WINDOWS,
            std::min(
                embeddings.size(),
                batch_start + pitchee::kEmbeddingBatchSize
            ),
            embeddings.size()
        );
    }
    return probabilities;
}

std::vector<float> naturalness_features(
    const std::vector<std::vector<float>>& embeddings
) {
    if (embeddings.empty()) throw std::invalid_argument("no naturalness embeddings");
    std::vector<std::vector<float>> normalized = embeddings;
    for (auto& embedding : normalized) {
        pitchee::normalize_l2(embedding.data(), embedding.size());
    }

    std::vector<float> raw_mean(pitchee::kEmbeddingDimensions, 0.0f);
    for (const auto& embedding : normalized) {
        for (size_t index = 0; index < raw_mean.size(); ++index) {
            raw_mean[index] += embedding[index];
        }
    }
    for (float& value : raw_mean) value /= normalized.size();

    std::vector<float> mean = raw_mean;
    pitchee::normalize_l2(mean.data(), mean.size());
    std::vector<double> variance(pitchee::kEmbeddingDimensions, 0.0);
    for (const auto& embedding : normalized) {
        for (size_t index = 0; index < variance.size(); ++index) {
            const double delta = static_cast<double>(embedding[index]) - raw_mean[index];
            variance[index] += delta * delta;
        }
    }
    std::vector<float> features = mean;
    features.reserve(pitchee::kEmbeddingDimensions * 2);
    for (double value : variance) {
        features.push_back(static_cast<float>(
            std::sqrt(value / normalized.size())
        ));
    }
    return features;
}

}  // namespace

extern "C" {

const char* pitchee_core_version(void) {
    return "0.2.0";
}

pitchee_status_t pitchee_analyzer_create(
    const char* model_directory,
    const pitchee_analyzer_options_t* options,
    pitchee_analyzer_t** out_analyzer,
    char* error_message,
    size_t error_message_capacity
) {
    if (!model_directory || !out_analyzer) {
        set_error(error_message, error_message_capacity, "invalid argument");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto analyzer = std::make_unique<pitchee_analyzer_t>();
        analyzer->model_directory = model_directory;
        analyzer->model_version = pitchee::model_version_from_directory(
            analyzer->model_directory
        );
        analyzer->intra_op_threads = options && options->intra_op_threads > 0
            ? options->intra_op_threads
            : 2;
        analyzer->use_coreml = !options || options->use_coreml != 0;
        const auto directory = analyzer->model_directory;
        analyzer->ecapa_frontend = std::make_unique<pitchee::OrtModel>(
            directory / "ECAPAFrontend.onnx",
            analyzer->intra_op_threads,
            false
        );
        analyzer->ecapa = std::make_unique<pitchee::OrtModel>(
            directory / "ECAPA.onnx",
            analyzer->intra_op_threads,
            analyzer->use_coreml
        );
        analyzer->vfp_head = std::make_unique<pitchee::OrtModel>(
            directory / "VFPHead.onnx",
            analyzer->intra_op_threads,
            false
        );
        analyzer->swift_f0 = std::make_unique<pitchee::OrtModel>(
            directory / "SwiftF0.onnx",
            analyzer->intra_op_threads,
            false
        );
        analyzer->vad = std::make_unique<pitchee::VadDetector>(
            directory / "SileroVAD.onnx",
            analyzer->intra_op_threads
        );
        analyzer->naturalness = std::make_unique<pitchee::NaturalnessModel>(
            directory / "Naturalness.onnx",
            analyzer->intra_op_threads
        );
        *out_analyzer = analyzer.release();
        return PITCHEE_SUCCESS;
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return status_for_exception(error);
    }
}

void pitchee_analyzer_destroy(pitchee_analyzer_t* analyzer) {
    delete analyzer;
}

pitchee_status_t analyze_pcm_impl(
    pitchee_analyzer_t* analyzer,
    const float* samples,
    size_t sample_count,
    int32_t sample_rate,
    int32_t channels,
    pitchee_score_profile_t score_profile,
    const ProgressReporter& reporter,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
) {
    if (!analyzer || !samples || sample_count == 0 || sample_rate <= 0
        || channels <= 0 || !valid_score_profile(score_profile) || !out_json) {
        set_error(error_message, error_message_capacity, "invalid PCM argument");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    *out_json = nullptr;
    try {
        reporter.phase(PITCHEE_PHASE_LOADING_AUDIO);
        reporter.progress(PITCHEE_PROGRESS_STAGE_LOADING_AUDIO, 1, 1);
        reporter.progress(PITCHEE_PROGRESS_STAGE_RESAMPLING_AUDIO, 0, 1);
        std::vector<float> signal = pitchee::resample_mono(
            samples,
            sample_count,
            channels,
            sample_rate,
            pitchee::kSampleRate
        );
        if (std::isfinite(pitchee::kMaximumSeconds)) {
            const size_t maximum_samples = static_cast<size_t>(
                pitchee::kMaximumSeconds * pitchee::kSampleRate
            );
            if (signal.size() > maximum_samples) signal.resize(maximum_samples);
        }
        if (signal.empty()) throw std::invalid_argument("empty audio");
        const double source_seconds =
            static_cast<double>(signal.size()) / pitchee::kSampleRate;
        reporter.progress(PITCHEE_PROGRESS_STAGE_RESAMPLING_AUDIO, 1, 1);

        reporter.phase(PITCHEE_PHASE_ANALYZING);
        const auto pitch = analyze_pitch(*analyzer->swift_f0, signal, reporter);
        auto vad = analyzer->vad->detect(
            signal,
            [&reporter](size_t completed, size_t total) {
                reporter.progress(
                    PITCHEE_PROGRESS_STAGE_DETECTING_SPEECH,
                    completed,
                    total
                );
            }
        );
        if (vad.segments.empty()) {
            throw std::runtime_error("Silero VAD detected no speech");
        }
        reporter.progress(PITCHEE_PROGRESS_STAGE_PREPARING_VFP_WINDOWS, 0, 1);
        const auto source_windows = pitchee::native_speech_windows(
            signal.size(),
            vad.segments
        );
        if (source_windows.empty()) {
            throw std::runtime_error("Silero VAD detected no speech");
        }
        reporter.progress(PITCHEE_PROGRESS_STAGE_PREPARING_VFP_WINDOWS, 1, 1);

        std::vector<std::vector<float>> vfp_patches;
        vfp_patches.reserve(source_windows.size());
        for (const auto& window : source_windows) {
            vfp_patches.push_back(pitchee::crop_window(signal, window));
        }
        const auto speech_embeddings = embed_waveforms(
            *analyzer->ecapa_frontend,
            *analyzer->ecapa,
            vfp_patches,
            reporter,
            PITCHEE_PROGRESS_STAGE_EXTRACTING_VFP_EMBEDDINGS
        );
        const auto probabilities = classify_embeddings(
            *analyzer->vfp_head,
            speech_embeddings,
            reporter
        );
        if (probabilities.empty()) throw std::runtime_error("no VFP windows");

        reporter.progress(
            PITCHEE_PROGRESS_STAGE_PREPARING_NATURALNESS_WINDOWS,
            0,
            1
        );
        // Naturalness uses the exact VFP window set on the source timeline.
        const auto& natural_windows = source_windows;
        reporter.progress(
            PITCHEE_PROGRESS_STAGE_PREPARING_NATURALNESS_WINDOWS,
            1,
            1
        );
        reporter.progress(
            PITCHEE_PROGRESS_STAGE_EXTRACTING_NATURALNESS_EMBEDDINGS,
            0,
            speech_embeddings.size()
        );
        const auto& natural_embeddings = speech_embeddings;
        reporter.progress(
            PITCHEE_PROGRESS_STAGE_EXTRACTING_NATURALNESS_EMBEDDINGS,
            natural_embeddings.size(),
            natural_embeddings.size()
        );
        const auto features = naturalness_features(natural_embeddings);

        pitchee::AnalysisResult result;
        result.model_version = analyzer->model_version;
        result.score_profile = score_profile;
        const double mean_probability = std::accumulate(
            probabilities.begin(),
            probabilities.end(),
            0.0
        ) / probabilities.size();
        result.vfp_standard_score = std::max(
            0.0,
            std::min(100.0, mean_probability * 100.0)
        );
        result.window_count = static_cast<int>(probabilities.size());
        result.window_duration_seconds = static_cast<double>(
            pitchee::kPatchSamples
        ) / pitchee::kSampleRate;
        result.vfp_windows.reserve(source_windows.size());
        for (size_t index = 0; index < source_windows.size(); ++index) {
            const double start_seconds = static_cast<double>(
                source_windows[index].start
            ) / pitchee::kSampleRate;
            const double end_seconds = static_cast<double>(
                source_windows[index].start + source_windows[index].length
            ) / pitchee::kSampleRate;
            pitchee::VfpWindow window;
            window.start_seconds = start_seconds;
            window.end_seconds = end_seconds;
            window.vfp_standard_score = std::max(
                0.0,
                std::min(100.0, probabilities[index] * 100.0)
            );
            result.vfp_windows.push_back(window);
        }

        result.naturalness_window_duration_seconds = static_cast<double>(
            pitchee::kPatchSamples
        ) / pitchee::kSampleRate;
        const std::vector<float> global_naturalness_std(
            features.begin() + pitchee::kEmbeddingDimensions,
            features.end()
        );
        result.naturalness_windows.reserve(natural_windows.size());
        reporter.progress(
            PITCHEE_PROGRESS_STAGE_SCORING_NATURALNESS_WINDOWS,
            0,
            natural_windows.size()
        );
        for (size_t index = 0; index < natural_windows.size(); ++index) {
            const double start_seconds = static_cast<double>(
                natural_windows[index].start
            ) / pitchee::kSampleRate;
            const double end_seconds = static_cast<double>(
                natural_windows[index].start + natural_windows[index].length
            ) / pitchee::kSampleRate;
            std::vector<float> window_features = natural_embeddings[index];
            pitchee::normalize_l2(
                window_features.data(),
                window_features.size()
            );
            window_features.insert(
                window_features.end(),
                global_naturalness_std.begin(),
                global_naturalness_std.end()
            );
            pitchee::NaturalnessWindow window;
            window.start_seconds = start_seconds;
            window.end_seconds = end_seconds;
            window.score = analyzer->naturalness->score(window_features);
            result.naturalness_windows.push_back(window);
            reporter.progress(
                PITCHEE_PROGRESS_STAGE_SCORING_NATURALNESS_WINDOWS,
                index + 1,
                natural_windows.size()
            );
        }

        reporter.progress(PITCHEE_PROGRESS_STAGE_CALCULATING_SCORES, 0, 1);
        result.source_sample_rate = sample_rate;
        result.source_channels = channels;
        result.source_seconds = source_seconds;
        result.analyzed_seconds = static_cast<double>(signal.size()) / pitchee::kSampleRate;
        result.has_f0 = pitch.has_mean_f0;
        result.f0_mean_hz = pitch.mean_f0_hz;
        result.f0_standard_deviation_hz = pitch.standard_deviation_f0_hz;
        result.voiced_frame_count = pitch.voiced_frame_count;
        result.voiced_window_count = pitch.voiced_window_count;
        result.f0_windows = pitch.windows;
        result.naturalness_score = analyzer->naturalness->score(features);
        result.score = pitchee::calculate_composite_score(
            score_profile,
            result.vfp_standard_score,
            result.naturalness_score,
            result.has_f0,
            result.f0_mean_hz
        );
        result.vad = vad;
        reporter.progress(PITCHEE_PROGRESS_STAGE_CALCULATING_SCORES, 1, 1);

        reporter.progress(PITCHEE_PROGRESS_STAGE_SERIALIZING_RESULT, 0, 1);
        const std::string json = pitchee::result_to_json(result);
        auto* output = new char[json.size() + 1];
        std::memcpy(output, json.c_str(), json.size() + 1);
        *out_json = output;
        reporter.progress(PITCHEE_PROGRESS_STAGE_SERIALIZING_RESULT, 1, 1);
        reporter.phase(PITCHEE_PHASE_COMPLETED);
        reporter.progress(PITCHEE_PROGRESS_STAGE_COMPLETED, 1, 1);
        return PITCHEE_SUCCESS;
    } catch (const std::invalid_argument& error) {
        set_error(error_message, error_message_capacity, error.what());
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return status_for_exception(error);
    }
}

pitchee_status_t pitchee_analyzer_analyze_pcm(
    pitchee_analyzer_t* analyzer,
    const float* samples,
    size_t sample_count,
    int32_t sample_rate,
    int32_t channels,
    pitchee_score_profile_t score_profile,
    pitchee_phase_callback_t phase_callback,
    void* user_data,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
) {
    const ProgressReporter reporter{
        phase_callback,
        user_data,
        nullptr,
        nullptr,
    };
    return analyze_pcm_impl(
        analyzer,
        samples,
        sample_count,
        sample_rate,
        channels,
        score_profile,
        reporter,
        out_json,
        error_message,
        error_message_capacity
    );
}

pitchee_status_t pitchee_analyzer_analyze_pcm_with_progress(
    pitchee_analyzer_t* analyzer,
    const float* samples,
    size_t sample_count,
    int32_t sample_rate,
    int32_t channels,
    pitchee_score_profile_t score_profile,
    pitchee_progress_callback_t progress_callback,
    void* user_data,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
) {
    const ProgressReporter reporter{
        nullptr,
        nullptr,
        progress_callback,
        user_data,
    };
    return analyze_pcm_impl(
        analyzer,
        samples,
        sample_count,
        sample_rate,
        channels,
        score_profile,
        reporter,
        out_json,
        error_message,
        error_message_capacity
    );
}

pitchee_status_t analyze_wav_impl(
    pitchee_analyzer_t* analyzer,
    const char* wav_path,
    pitchee_score_profile_t score_profile,
    const ProgressReporter& reporter,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
) {
    if (!analyzer || !wav_path || !valid_score_profile(score_profile) || !out_json) {
        set_error(error_message, error_message_capacity, "invalid WAV argument");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    *out_json = nullptr;
    try {
        reporter.progress(PITCHEE_PROGRESS_STAGE_LOADING_AUDIO, 0, 1);
        const pitchee::WavData wav = pitchee::read_wav(wav_path);
        return analyze_pcm_impl(
            analyzer,
            wav.samples.data(),
            wav.samples.size(),
            wav.sample_rate,
            wav.channels,
            score_profile,
            reporter,
            out_json,
            error_message,
            error_message_capacity
        );
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return status_for_exception(error);
    }
}

pitchee_status_t pitchee_analyzer_analyze_wav_file(
    pitchee_analyzer_t* analyzer,
    const char* wav_path,
    pitchee_score_profile_t score_profile,
    pitchee_phase_callback_t phase_callback,
    void* user_data,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
) {
    const ProgressReporter reporter{
        phase_callback,
        user_data,
        nullptr,
        nullptr,
    };
    return analyze_wav_impl(
        analyzer,
        wav_path,
        score_profile,
        reporter,
        out_json,
        error_message,
        error_message_capacity
    );
}

pitchee_status_t pitchee_analyzer_analyze_wav_file_with_progress(
    pitchee_analyzer_t* analyzer,
    const char* wav_path,
    pitchee_score_profile_t score_profile,
    pitchee_progress_callback_t progress_callback,
    void* user_data,
    char** out_json,
    char* error_message,
    size_t error_message_capacity
) {
    const ProgressReporter reporter{
        nullptr,
        nullptr,
        progress_callback,
        user_data,
    };
    return analyze_wav_impl(
        analyzer,
        wav_path,
        score_profile,
        reporter,
        out_json,
        error_message,
        error_message_capacity
    );
}

pitchee_status_t pitchee_realtime_f0_create(
    pitchee_analyzer_t* analyzer,
    const pitchee_realtime_f0_options_t* options,
    pitchee_realtime_f0_t** out_stream,
    char* error_message,
    size_t error_message_capacity
) {
    if (!analyzer || !out_stream) {
        set_error(error_message, error_message_capacity, "invalid realtime F0 argument");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    *out_stream = nullptr;
    const size_t context_samples = options && options->context_samples > 0
        ? static_cast<size_t>(options->context_samples)
        : 5120;
    const size_t hop_samples = options && options->hop_samples > 0
        ? static_cast<size_t>(options->hop_samples)
        : 256;
    if (context_samples < 256 || hop_samples < 1
        || hop_samples > context_samples) {
        set_error(error_message, error_message_capacity, "invalid realtime F0 window options");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto stream = std::make_unique<pitchee_realtime_f0_t>();
        stream->model = analyzer->swift_f0.get();
        stream->context_samples = context_samples;
        stream->hop_samples = hop_samples;
        *out_stream = stream.release();
        return PITCHEE_SUCCESS;
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return status_for_exception(error);
    }
}

pitchee_status_t pitchee_realtime_f0_process(
    pitchee_realtime_f0_t* stream,
    const float* samples,
    size_t sample_count,
    pitchee_f0_frame_callback_t frame_callback,
    void* user_data,
    size_t* out_frame_count,
    char* error_message,
    size_t error_message_capacity
) {
    if (out_frame_count) *out_frame_count = 0;
    if (!stream || !stream->model || (!samples && sample_count > 0)) {
        set_error(error_message, error_message_capacity, "invalid realtime F0 argument");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    if (sample_count == 0) return PITCHEE_SUCCESS;

    try {
        stream->buffer.insert(
            stream->buffer.end(),
            samples,
            samples + sample_count
        );
        stream->total_samples += sample_count;
        stream->samples_since_inference += sample_count;
        if (stream->buffer.size() > stream->context_samples) {
            const size_t drop = stream->buffer.size() - stream->context_samples;
            stream->buffer.erase(
                stream->buffer.begin(),
                stream->buffer.begin() + static_cast<std::ptrdiff_t>(drop)
            );
            stream->buffer_start_sample += drop;
        }

        if (!stream->has_emitted
            && stream->buffer.size() < stream->context_samples) {
            return PITCHEE_SUCCESS;
        }
        if (stream->has_emitted
            && stream->samples_since_inference < stream->hop_samples) {
            return PITCHEE_SUCCESS;
        }

        const ProgressReporter reporter{};
        const auto pitch = analyze_pitch(*stream->model, stream->buffer, reporter);
        size_t emitted = 0;
        for (size_t index = 0; index < pitch.pitch_hz.size(); ++index) {
            const int64_t center_sample = static_cast<int64_t>(
                stream->buffer_start_sample
            ) + static_cast<int64_t>(std::llround(
                pitch.timestamps[index] * pitchee::kSampleRate
            ));
            if (stream->has_emitted
                && center_sample <= stream->last_emitted_center_sample) {
                continue;
            }
            const double timestamp = static_cast<double>(center_sample)
                / pitchee::kSampleRate;
            const float f0 = pitch.pitch_hz[index];
            const float confidence = pitch.confidence[index];
            const bool voiced = confidence > 0.9f
                && f0 >= 75.0f
                && f0 <= 600.0f;
            if (frame_callback) {
                pitchee_f0_frame_t frame{};
                frame.timestamp_seconds = timestamp;
                frame.f0_hz = f0;
                frame.confidence = confidence;
                frame.voiced = voiced ? 1 : 0;
                frame_callback(&frame, user_data);
            }
            stream->last_emitted_center_sample = center_sample;
            stream->has_emitted = true;
            ++emitted;
        }
        stream->samples_since_inference = 0;
        if (out_frame_count) *out_frame_count = emitted;
        return PITCHEE_SUCCESS;
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return status_for_exception(error);
    }
}

void pitchee_realtime_f0_reset(pitchee_realtime_f0_t* stream) {
    if (!stream) return;
    stream->buffer.clear();
    stream->buffer_start_sample = 0;
    stream->total_samples = 0;
    stream->samples_since_inference = 0;
    stream->has_emitted = false;
    stream->last_emitted_center_sample = -1;
}

void pitchee_realtime_f0_destroy(pitchee_realtime_f0_t* stream) {
    delete stream;
}

pitchee_status_t pitchee_realtime_resonance_create(
    pitchee_analyzer_t* analyzer,
    const pitchee_realtime_resonance_options_t* options,
    pitchee_realtime_resonance_t** out_stream,
    char* error_message,
    size_t error_message_capacity
) {
    if (!analyzer || !options || !out_stream) {
        set_error(
            error_message,
            error_message_capacity,
            "invalid realtime resonance argument"
        );
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    *out_stream = nullptr;
    if (!options->vowel) {
        set_error(error_message, error_message_capacity, "invalid corner vowel");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    const std::string vowel(options->vowel);
    int vowel_index = 0;
    if (!parse_resonance_vowel(vowel, &vowel_index)) {
        set_error(error_message, error_message_capacity, "unsupported corner vowel");
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    const size_t context_samples = options->context_samples > 0
        ? static_cast<size_t>(options->context_samples)
        : 5120;
    const size_t hop_samples = options->hop_samples > 0
        ? static_cast<size_t>(options->hop_samples)
        : 256;
    const size_t formant_window_samples = options->formant_window_samples > 0
        ? static_cast<size_t>(options->formant_window_samples)
        : 3200;
    if (context_samples < 256 || hop_samples < 1
        || hop_samples > context_samples
        || formant_window_samples < 512) {
        set_error(
            error_message,
            error_message_capacity,
            "invalid realtime resonance window options"
        );
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto stream = std::make_unique<pitchee_realtime_resonance_t>();
        stream->f0_stream = new pitchee_realtime_f0_t();
        stream->f0_stream->model = analyzer->swift_f0.get();
        stream->f0_stream->context_samples = context_samples;
        stream->f0_stream->hop_samples = hop_samples;
        stream->formant_model = std::make_unique<pitchee::OrtModel>(
            analyzer->model_directory / "FormantNet.onnx",
            analyzer->intra_op_threads,
            false
        );
        stream->vowel = vowel;
        stream->vowel_index = vowel_index;
        stream->formant_window_samples = formant_window_samples;
        stream->context_samples = context_samples;
        *out_stream = stream.release();
        return PITCHEE_SUCCESS;
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return status_for_exception(error);
    }
}

pitchee_status_t pitchee_realtime_resonance_process(
    pitchee_realtime_resonance_t* stream,
    const float* samples,
    size_t sample_count,
    pitchee_resonance_frame_callback_t frame_callback,
    void* user_data,
    size_t* out_frame_count,
    char* error_message,
    size_t error_message_capacity
) {
    if (out_frame_count) *out_frame_count = 0;
    if (!stream || !stream->f0_stream || !stream->formant_model
        || (!samples && sample_count > 0)) {
        set_error(
            error_message,
            error_message_capacity,
            "invalid realtime resonance stream"
        );
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    if (sample_count == 0) return PITCHEE_SUCCESS;
    try {
        stream->buffer.insert(
            stream->buffer.end(),
            samples,
            samples + sample_count
        );
        stream->total_samples += sample_count;
        const size_t maximum_buffer = stream->context_samples
            + stream->formant_window_samples;
        if (stream->buffer.size() > maximum_buffer) {
            const size_t drop = stream->buffer.size() - maximum_buffer;
            stream->buffer.erase(
                stream->buffer.begin(),
                stream->buffer.begin() + static_cast<std::ptrdiff_t>(drop)
            );
            stream->buffer_start_sample += drop;
        }
        stream->emitted_frames = 0;
        stream->callback_error.clear();
        stream->frame_callback = frame_callback;
        stream->user_data = user_data;
        size_t f0_frame_count = 0;
        const auto status = pitchee_realtime_f0_process(
            stream->f0_stream,
            samples,
            sample_count,
            realtime_resonance_f0_callback,
            stream,
            &f0_frame_count,
            error_message,
            error_message_capacity
        );
        if (status != PITCHEE_SUCCESS) return status;
        if (!stream->callback_error.empty()) {
            set_error(
                error_message,
                error_message_capacity,
                stream->callback_error
            );
            return PITCHEE_ERROR_INTERNAL;
        }
        if (out_frame_count) *out_frame_count = stream->emitted_frames;
        return PITCHEE_SUCCESS;
    } catch (const std::exception& error) {
        set_error(error_message, error_message_capacity, error.what());
        return status_for_exception(error);
    }
}

void pitchee_realtime_resonance_reset(pitchee_realtime_resonance_t* stream) {
    if (!stream) return;
    if (stream->f0_stream) pitchee_realtime_f0_reset(stream->f0_stream);
    stream->buffer.clear();
    stream->buffer_start_sample = 0;
    stream->total_samples = 0;
    stream->emitted_frames = 0;
    stream->callback_error.clear();
}

void pitchee_realtime_resonance_destroy(pitchee_realtime_resonance_t* stream) {
    if (!stream) return;
    if (stream->f0_stream) {
        pitchee_realtime_f0_destroy(stream->f0_stream);
        stream->f0_stream = nullptr;
    }
    delete stream;
}

double pitchee_composite_score_value(
    pitchee_score_profile_t score_profile,
    double vfp_standard_score,
    double naturalness_score,
    double f0_hz
) {
    if (!valid_score_profile(score_profile)) return std::numeric_limits<double>::quiet_NaN();
    const bool has_f0 = std::isfinite(f0_hz) && f0_hz > 0.0;
    return pitchee::calculate_composite_score(
        score_profile,
        vfp_standard_score,
        naturalness_score,
        has_f0,
        f0_hz
    ).final_score;
}

pitchee_status_t pitchee_composite_score(
    pitchee_score_profile_t score_profile,
    double vfp_standard_score,
    double naturalness_score,
    double f0_hz,
    int32_t has_f0,
    pitchee_composite_score_t* out_score
) {
    if (!valid_score_profile(score_profile) || !out_score) {
        return PITCHEE_ERROR_INVALID_ARGUMENT;
    }
    const auto score = pitchee::calculate_composite_score(
        score_profile,
        vfp_standard_score,
        naturalness_score,
        has_f0 != 0,
        f0_hz
    );
    out_score->base_score = score.base_score;
    out_score->final_score = score.final_score;
    out_score->has_score_cap = score.has_score_cap ? 1 : 0;
    out_score->score_cap = score.score_cap;
    out_score->score_limited = score.score_limited ? 1 : 0;
    out_score->score_boosted = score.score_boosted ? 1 : 0;
    std::memset(out_score->score_rule, 0, sizeof(out_score->score_rule));
    std::memcpy(
        out_score->score_rule,
        score.score_rule.c_str(),
        std::min(
            sizeof(out_score->score_rule) - 1,
            score.score_rule.size()
        )
    );
    return PITCHEE_SUCCESS;
}

void pitchee_string_free(char* value) {
    delete[] value;
}

}  // extern "C"
