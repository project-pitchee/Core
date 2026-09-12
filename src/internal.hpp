#ifndef PITCHEE_INTERNAL_HPP
#define PITCHEE_INTERNAL_HPP

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace pitchee {

constexpr int kSampleRate = 16000;
constexpr int kPatchSamples = 24240;
constexpr int kStrideSamples = 1600;
constexpr int kEmbeddingBatchSize = 8;
constexpr int kEmbeddingDimensions = 192;
constexpr double kMaximumSeconds = 20.0;

struct VadSegment {
    double source_start_seconds = 0.0;
    double source_end_seconds = 0.0;
    double speech_start_seconds = 0.0;
    double speech_end_seconds = 0.0;
};

struct VadResult {
    std::vector<VadSegment> segments;
    double input_seconds = 0.0;
    double speech_seconds = 0.0;
    int silero_segment_count = 0;
    int discarded_breath_like_count = 0;
    int trimmed_segment_count = 0;
};

struct CompositeScore {
    double base_score = 0.0;
    double final_score = 0.0;
    bool has_score_cap = false;
    double score_cap = 0.0;
    bool score_limited = false;
    bool score_boosted = false;
    std::string score_rule;
};

struct AnalysisResult {
    double raw_female_score = 0.0;
    double standard_score = 0.0;
    int window_count = 0;
    std::vector<double> window_raw_scores;
    std::vector<double> window_starts_seconds;
    double window_duration_seconds = 0.0;
    int naturalness_patch_count = 0;
    int source_sample_rate = 0;
    int source_channels = 0;
    double source_seconds = 0.0;
    double analyzed_seconds = 0.0;
    bool has_f0 = false;
    double f0_mean_hz = 0.0;
    double f0_standard_deviation_hz = 0.0;
    int voiced_frame_count = 0;
    double naturalness_score = 0.0;
    CompositeScore score;
    VadResult vad;
};

struct PitchResult {
    std::vector<float> pitch_hz;
    std::vector<float> confidence;
    std::vector<float> timestamps;
    std::vector<uint8_t> voicing;
    bool has_mean_f0 = false;
    double mean_f0_hz = 0.0;
    double standard_deviation_f0_hz = 0.0;
    int voiced_frame_count = 0;
};

class OrtModel;

class NaturalnessModel {
public:
    NaturalnessModel(
        const std::filesystem::path& path,
        int intra_op_threads
    );
    ~NaturalnessModel();
    double score(const std::vector<float>& features) const;

private:
    std::unique_ptr<OrtModel> model_;
};

CompositeScore calculate_composite_score(
    double standard_score,
    double naturalness_score,
    bool has_f0,
    double f0_hz
);

std::string result_to_json(const AnalysisResult& result);
std::string escape_json(const std::string& value);

std::vector<float> resample_mono(
    const float* interleaved,
    size_t sample_count,
    int channels,
    int source_rate,
    int target_rate = kSampleRate
);

std::vector<float> concatenate_speech(
    const std::vector<float>& samples,
    const std::vector<VadSegment>& segments
);

std::vector<float> crop_patch(const std::vector<float>& signal, size_t start);
std::vector<size_t> sliding_patch_starts(size_t sample_count);
std::vector<size_t> naturalness_patch_starts(size_t sample_count);

void normalize_l2(float* values, size_t size);

}  // namespace pitchee

#endif
