#include "formant_tracker.hpp"

#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace pitchee {
namespace {

constexpr int kFftSize = 2048;
constexpr int kGridSize = 320;
constexpr double kMinHz = 80.0;
constexpr double kMaxHz = 5000.0;
constexpr double kTrackHistorySeconds = 0.50;
constexpr double kPi = 3.14159265358979323846;

constexpr std::array<std::pair<double, double>, 4> kF1Ranges{{
    {250.0, 550.0}, {250.0, 650.0}, {500.0, 1050.0}, {600.0, 1150.0},
}};
constexpr std::array<std::pair<double, double>, 4> kF2Ranges{{
    {1700.0, 3200.0}, {600.0, 1500.0}, {1300.0, 2600.0}, {900.0, 2000.0},
}};
constexpr std::array<std::pair<double, double>, 4> kF3Ranges{{
    {2400.0, 3800.0}, {2000.0, 3400.0}, {2200.0, 3600.0}, {2200.0, 3600.0},
}};
constexpr std::array<std::pair<double, double>, 4> kF4Ranges{{
    {3100.0, 4700.0}, {3000.0, 4600.0}, {3200.0, 4800.0}, {3200.0, 4800.0},
}};

std::array<double, 4> prior_center(int vowel_index) {
    const auto index = static_cast<size_t>(vowel_index);
    return {
        std::sqrt(kF1Ranges[index].first * kF1Ranges[index].second),
        std::sqrt(kF2Ranges[index].first * kF2Ranges[index].second),
        std::sqrt(kF3Ranges[index].first * kF3Ranges[index].second),
        std::sqrt(kF4Ranges[index].first * kF4Ranges[index].second),
    };
}

std::pair<double, double> range_for(
    int vowel_index,
    int formant_index
) {
    const auto index = static_cast<size_t>(vowel_index);
    switch (formant_index) {
        case 0: return kF1Ranges[index];
        case 1: return kF2Ranges[index];
        case 2: return kF3Ranges[index];
        default: return kF4Ranges[index];
    }
}

void fft_radix2(std::vector<std::complex<double>>& values) {
    const size_t size = values.size();
    for (size_t index = 1, reversed = 0; index < size; ++index) {
        size_t bit = size >> 1;
        for (; reversed & bit; bit >>= 1) reversed ^= bit;
        reversed ^= bit;
        if (index < reversed) std::swap(values[index], values[reversed]);
    }
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

void gaussian_smooth(std::vector<double>& values, double sigma) {
    const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
    std::vector<double> kernel(static_cast<size_t>(2 * radius + 1), 0.0);
    double sum = 0.0;
    for (int offset = -radius; offset <= radius; ++offset) {
        const double value = std::exp(
            -0.5 * offset * offset / (sigma * sigma)
        );
        kernel[static_cast<size_t>(offset + radius)] = value;
        sum += value;
    }
    for (double& value : kernel) value /= sum;

    std::vector<double> output(values.size(), 0.0);
    for (size_t index = 0; index < values.size(); ++index) {
        double value = 0.0;
        for (int offset = -radius; offset <= radius; ++offset) {
            const int source = std::clamp(
                static_cast<int>(index) + offset,
                0,
                static_cast<int>(values.size()) - 1
            );
            value += values[static_cast<size_t>(source)]
                * kernel[static_cast<size_t>(offset + radius)];
        }
        output[index] = value;
    }
    values.swap(output);
}

struct Peak {
    double hz = 0.0;
    double prominence = 0.0;
};

std::vector<Peak> find_peaks(
    const std::vector<double>& envelope,
    const std::vector<double>& grid
) {
    std::vector<Peak> candidates;
    for (size_t index = 1; index + 1 < envelope.size(); ++index) {
        if (!(envelope[index] > envelope[index - 1]
              && envelope[index] >= envelope[index + 1])) {
            continue;
        }
        double left_min = envelope[index];
        for (size_t cursor = index; cursor > 0; --cursor) {
            left_min = std::min(left_min, envelope[cursor - 1]);
            if (envelope[cursor - 1] > envelope[index]) break;
        }
        double right_min = envelope[index];
        for (size_t cursor = index + 1; cursor < envelope.size(); ++cursor) {
            right_min = std::min(right_min, envelope[cursor]);
            if (envelope[cursor] > envelope[index]) break;
        }
        const double prominence = envelope[index]
            - std::max(left_min, right_min);
        if (prominence >= 0.10) {
            candidates.push_back({grid[index], prominence});
        }
    }

    std::vector<Peak> peaks;
    size_t last_index = 0;
    for (size_t index = 0; index < candidates.size(); ++index) {
        const size_t candidate_index = static_cast<size_t>(
            std::lower_bound(grid.begin(), grid.end(), candidates[index].hz)
            - grid.begin()
        );
        if (peaks.empty() || candidate_index >= last_index + 5) {
            peaks.push_back(candidates[index]);
            last_index = candidate_index;
        } else if (candidates[index].prominence
                   > peaks.back().prominence) {
            peaks.back() = candidates[index];
            last_index = candidate_index;
        }
    }
    return peaks;
}

std::vector<std::pair<double, double>> frame_envelope(
    const std::vector<float>& samples,
    double f0_hz
) {
    std::vector<double> frame(samples.begin(), samples.end());
    if (frame.size() < 256) {
        frame.insert(frame.begin(), 256 - frame.size(), 0.0);
    }
    const double mean = frame.empty()
        ? 0.0
        : std::accumulate(frame.begin(), frame.end(), 0.0)
            / static_cast<double>(frame.size());
    for (double& value : frame) value -= mean;
    frame.resize(static_cast<size_t>(kFftSize), 0.0);
    std::vector<std::complex<double>> spectrum(frame.size(), 0.0);
    for (size_t index = 0; index < frame.size(); ++index) {
        const double window = 0.5 - 0.5 * std::cos(
            2.0 * kPi * static_cast<double>(index)
            / static_cast<double>(kFftSize - 1)
        );
        spectrum[index] = std::complex<double>(frame[index] * window, 0.0);
    }
    fft_radix2(spectrum);

    std::vector<double> harmonic_hz;
    std::vector<double> levels;
    const int harmonic_count = static_cast<int>(std::floor(kMaxHz / f0_hz));
    for (int harmonic = 1; harmonic <= harmonic_count; ++harmonic) {
        const double frequency = harmonic * f0_hz;
        if (frequency > kMaxHz) break;
        const double half_band = std::max(20.0, 0.18 * f0_hz);
        int low = static_cast<int>(std::ceil(
            (frequency - half_band) * kFftSize / kSampleRate
        ));
        int high = static_cast<int>(std::floor(
            (frequency + half_band) * kFftSize / kSampleRate
        ));
        low = std::max(low, 0);
        high = std::min(high, kFftSize / 2);
        double magnitude = 1e-7;
        for (int bin = low; bin <= high; ++bin) {
            magnitude = std::max(
                magnitude,
                std::abs(spectrum[static_cast<size_t>(bin)])
            );
        }
        if (frequency >= kMinHz) {
            harmonic_hz.push_back(frequency);
            levels.push_back(20.0 * std::log10(std::max(magnitude, 1e-7)));
        }
    }

    std::vector<double> grid(kGridSize);
    for (int index = 0; index < kGridSize; ++index) {
        const double fraction = static_cast<double>(index)
            / static_cast<double>(kGridSize - 1);
        grid[static_cast<size_t>(index)] = kMinHz
            * std::pow(kMaxHz / kMinHz, fraction);
    }
    std::vector<double> envelope(kGridSize, 0.0);
    if (harmonic_hz.size() < 4) return {{0.0, 0.0}};
    for (int index = 0; index < kGridSize; ++index) {
        const double frequency = grid[static_cast<size_t>(index)];
        const auto upper = std::lower_bound(
            harmonic_hz.begin(),
            harmonic_hz.end(),
            frequency
        );
        if (upper == harmonic_hz.begin()) {
            envelope[static_cast<size_t>(index)] = levels.front();
            continue;
        }
        if (upper == harmonic_hz.end()) {
            envelope[static_cast<size_t>(index)] = levels.back();
            continue;
        }
        const size_t right = static_cast<size_t>(upper - harmonic_hz.begin());
        const size_t left = right - 1;
        const double fraction = (
            std::log(frequency) - std::log(harmonic_hz[left])
        ) / (
            std::log(harmonic_hz[right]) - std::log(harmonic_hz[left])
        );
        envelope[static_cast<size_t>(index)] = levels[left]
            + fraction * (levels[right] - levels[left]);
    }
    gaussian_smooth(envelope, 3.0);
    std::vector<std::pair<double, double>> result;
    result.reserve(grid.size());
    for (size_t index = 0; index < grid.size(); ++index) {
        result.emplace_back(grid[index], envelope[index]);
    }
    return result;
}

std::array<double, 4> select_peaks(
    int vowel_index,
    const std::vector<std::pair<double, double>>& envelope,
    const std::array<double, 4>& previous
) {
    std::vector<double> grid;
    std::vector<double> values;
    grid.reserve(envelope.size());
    values.reserve(envelope.size());
    for (const auto& [hz, value] : envelope) {
        grid.push_back(hz);
        values.push_back(value);
    }
    const auto peaks = find_peaks(values, grid);
    std::array<double, 4> selected{};
    double lower_bound = kMinHz;
    const auto prior = prior_center(vowel_index);
    for (int formant = 0; formant < 4; ++formant) {
        const auto [low, high] = range_for(vowel_index, formant);
        std::vector<Peak> candidates;
        for (const Peak& peak : peaks) {
            if (peak.hz >= low && peak.hz <= high && peak.hz > lower_bound) {
                candidates.push_back(peak);
            }
        }
        if (candidates.empty()) {
            double center = std::clamp(prior[static_cast<size_t>(formant)], low, high);
            if (previous[static_cast<size_t>(formant)] > center) {
                center = std::min(previous[static_cast<size_t>(formant)], high);
            }
            selected[static_cast<size_t>(formant)] = center;
            lower_bound = center;
            continue;
        }
        double best_cost = std::numeric_limits<double>::infinity();
        double best_hz = candidates.front().hz;
        for (const Peak& candidate : candidates) {
            const double cost =
                std::abs(std::log(candidate.hz / prior[static_cast<size_t>(formant)]))
                + 0.45 * std::abs(std::log(
                    candidate.hz / std::max(previous[static_cast<size_t>(formant)], 1.0)
                ))
                - 0.12 * candidate.prominence;
            if (cost < best_cost) {
                best_cost = cost;
                best_hz = candidate.hz;
            }
        }
        selected[static_cast<size_t>(formant)] = best_hz;
        lower_bound = best_hz + 80.0;
    }
    return selected;
}

std::array<double, 4> median_history(
    const std::vector<std::array<double, 4>>& history
) {
    std::array<double, 4> result{};
    for (int formant = 0; formant < 4; ++formant) {
        std::vector<double> values;
        values.reserve(history.size());
        for (const auto& item : history) {
            values.push_back(item[static_cast<size_t>(formant)]);
        }
        const size_t middle = values.size() / 2;
        std::nth_element(
            values.begin(),
            values.begin() + static_cast<std::ptrdiff_t>(middle),
            values.end()
        );
        result[static_cast<size_t>(formant)] = values[middle];
    }
    return result;
}

struct StableResonanceReference {
    std::array<int, 2> indices;
    std::array<double, 2> male_median_log;
    std::array<double, 2> female_median_log;
    std::array<double, 2> weights;
};

constexpr std::array<StableResonanceReference, 4> kStableReferences{{
    {{{1, 2}}, {{7.775998134061836, 8.02532757264223}},
     {{7.905384598912772, 8.056622033666084}},
     {{0.7604975713694999, 0.23950242863050009}}},
    {{{0, 2}}, {{5.916086596829337, 7.852548913740458}},
     {{6.087256995947913, 7.879653056896393}},
     {{0.6787324685617498, 0.3212675314382501}}},
    {{{1, 2}}, {{7.531157692509552, 7.909891045818759}},
     {{7.5909162980005025, 7.949155797802636}},
     {{0.6571044602350229, 0.3428955397649772}}},
    {{{1, 3}}, {{7.204325589749411, 8.27200275955394}},
     {{7.2819576661276955, 8.289984604526841}},
     {{0.6482793224726272, 0.3517206775273729}}},
}};

}  // namespace

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
    if (index >= kStableReferences.size()) return -1.0f;
    const auto& reference = kStableReferences[index];
    double position = 0.0;
    for (size_t feature = 0; feature < 2; ++feature) {
        const size_t formant = static_cast<size_t>(reference.indices[feature]);
        const double value = std::log(std::max(
            static_cast<double>(formants_hz[formant]),
            1.0
        ));
        const double normalized = (
            value - reference.male_median_log[feature]
        ) / (
            reference.female_median_log[feature]
            - reference.male_median_log[feature]
        );
        position += reference.weights[feature] * normalized;
    }
    const double logit = -2.0 * std::log(1.0 / 3.0) * (position - 0.5);
    return static_cast<float>(100.0 / (1.0 + std::exp(-logit)));
}

HarmonicFormantTracker::HarmonicFormantTracker(
    int vowel_index,
    size_t hop_samples
) : vowel_index_(vowel_index) {
    history_limit_ = std::max<size_t>(
        1,
        static_cast<size_t>(std::llround(
            kTrackHistorySeconds * kSampleRate
            / static_cast<double>(std::max<size_t>(hop_samples, 1))
        ))
    );
    reset();
}

void HarmonicFormantTracker::reset() {
    initialized_ = false;
    previous_ = prior_center(vowel_index_);
    velocity_.fill(0.0);
    output_ = previous_;
    pending_.fill(std::numeric_limits<double>::quiet_NaN());
    pending_count_.fill(0);
    history_.clear();
}

std::array<float, 4> HarmonicFormantTracker::process(
    const std::vector<float>& samples,
    double f0_hz
) {
    if (f0_hz <= 0.0 || !std::isfinite(f0_hz)) {
        return {
            static_cast<float>(output_[0]), static_cast<float>(output_[1]),
            static_cast<float>(output_[2]), static_cast<float>(output_[3]),
        };
    }
    const auto envelope = frame_envelope(samples, f0_hz);
    if (envelope.size() < 4 || envelope.front().second == 0.0) {
        return {
            static_cast<float>(output_[0]), static_cast<float>(output_[1]),
            static_cast<float>(output_[2]), static_cast<float>(output_[3]),
        };
    }
    const auto measurement = select_peaks(vowel_index_, envelope, previous_);
    history_.push_back(measurement);
    if (history_.size() > history_limit_) history_.erase(history_.begin());
    if (!initialized_) {
        previous_ = measurement;
        output_ = measurement;
        initialized_ = true;
        return {
            static_cast<float>(output_[0]), static_cast<float>(output_[1]),
            static_cast<float>(output_[2]), static_cast<float>(output_[3]),
        };
    }

    const std::array<double, 4> gate{50.0, 100.0, 130.0, 160.0};
    const auto history_center = median_history(history_);
    std::array<double, 4> predicted{};
    std::array<bool, 4> accepted{};
    for (int formant = 0; formant < 4; ++formant) {
        predicted[static_cast<size_t>(formant)] =
            previous_[static_cast<size_t>(formant)]
            + 0.65 * velocity_[static_cast<size_t>(formant)];
        velocity_[static_cast<size_t>(formant)] *= 0.80;
        const double residual = measurement[static_cast<size_t>(formant)]
            - predicted[static_cast<size_t>(formant)];
        accepted[static_cast<size_t>(formant)] =
            std::abs(residual) <= gate[static_cast<size_t>(formant)]
            || std::abs(
                measurement[static_cast<size_t>(formant)]
                - history_center[static_cast<size_t>(formant)]
            ) <= gate[static_cast<size_t>(formant)];
        if (accepted[static_cast<size_t>(formant)]) {
            pending_[static_cast<size_t>(formant)] =
                std::numeric_limits<double>::quiet_NaN();
            pending_count_[static_cast<size_t>(formant)] = 0;
            continue;
        }
        const double tolerance = std::max(
            35.0,
            0.35 * gate[static_cast<size_t>(formant)]
        );
        if (std::isfinite(pending_[static_cast<size_t>(formant)])
            && std::abs(
                measurement[static_cast<size_t>(formant)]
                - pending_[static_cast<size_t>(formant)]
            ) <= tolerance) {
            ++pending_count_[static_cast<size_t>(formant)];
        } else {
            pending_[static_cast<size_t>(formant)] =
                measurement[static_cast<size_t>(formant)];
            pending_count_[static_cast<size_t>(formant)] = 1;
        }
        if (pending_count_[static_cast<size_t>(formant)] >= 2) {
            accepted[static_cast<size_t>(formant)] = true;
            pending_[static_cast<size_t>(formant)] =
                std::numeric_limits<double>::quiet_NaN();
            pending_count_[static_cast<size_t>(formant)] = 0;
        }
    }

    for (int formant = 0; formant < 4; ++formant) {
        const double gated = accepted[static_cast<size_t>(formant)]
            ? measurement[static_cast<size_t>(formant)]
            : predicted[static_cast<size_t>(formant)];
        const double gain = 0.12 / (
            1.0 + std::abs(
                gated - predicted[static_cast<size_t>(formant)]
            ) / 140.0
        );
        previous_[static_cast<size_t>(formant)] =
            predicted[static_cast<size_t>(formant)]
            + gain * (
                gated - predicted[static_cast<size_t>(formant)]
            );
        velocity_[static_cast<size_t>(formant)] +=
            0.08 * gain * (
                gated - predicted[static_cast<size_t>(formant)]
            );
        output_[static_cast<size_t>(formant)] +=
            0.30 * (
                previous_[static_cast<size_t>(formant)]
                - output_[static_cast<size_t>(formant)]
            );
    }
    std::sort(output_.begin(), output_.end());
    return {
        static_cast<float>(output_[0]), static_cast<float>(output_[1]),
        static_cast<float>(output_[2]), static_cast<float>(output_[3]),
    };
}

}  // namespace pitchee
