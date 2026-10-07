#include "pitchee/pitchee.h"

#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Metrics {
    double vfp = std::nan("");
    double naturalness = std::nan("");
    double f0 = std::nan("");
    double elapsed_ms = 0.0;
};

double extract_number(
    const std::string& json,
    const std::string& key,
    size_t start = 0
) {
    const size_t key_position = json.find(key, start);
    if (key_position == std::string::npos) return std::nan("");
    const size_t colon = json.find(':', key_position + key.size());
    if (colon == std::string::npos) return std::nan("");
    const char* begin = json.c_str() + colon + 1;
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    return end == begin ? std::nan("") : value;
}

double extract_naturalness(const std::string& json) {
    const size_t section = json.find("\"naturalness\"");
    if (section == std::string::npos) return std::nan("");
    return extract_number(json, "\"score\"", section);
}

Metrics analyze_file(
    pitchee_analyzer_t* analyzer,
    const std::filesystem::path& wav_path
) {
    char* json = nullptr;
    char error[2048] = {};
    const auto start = std::chrono::steady_clock::now();
    const auto status = pitchee_analyze_wav_file(
        analyzer,
        wav_path.c_str(),
        PITCHEE_SCORE_PROFILE_FEMINIZATION,
        nullptr,
        nullptr,
        &json,
        error,
        sizeof(error)
    );
    const auto end = std::chrono::steady_clock::now();
    if (status != PITCHEE_SUCCESS) {
        throw std::runtime_error(
            std::string("analyze failed: ") + wav_path.filename().string()
            + ": " + error
        );
    }

    const std::string result(json);
    pitchee_string_free(json);
    return {
        extract_number(result, "\"vfp_standard_score\""),
        extract_naturalness(result),
        extract_number(result, "\"mean_hz\""),
        std::chrono::duration<double, std::milli>(end - start).count(),
    };
}

void log_multiline(const std::string& report) {
    std::istringstream lines(report);
    std::string line;
    while (std::getline(lines, line)) {
        __android_log_print(
            ANDROID_LOG_INFO,
            "PitcheeNcnnBenchmark",
            "%s",
            line.c_str()
        );
    }
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_space_pitchee_benchmark_MainActivity_nativeRunNcnnBenchmark(
    JNIEnv* env,
    jobject,
    jstring model_directory,
    jstring audio_directory,
    jint threads
) {
    const char* model_chars = env->GetStringUTFChars(model_directory, nullptr);
    const char* audio_chars = env->GetStringUTFChars(audio_directory, nullptr);
    std::string report;
    pitchee_analyzer_t* analyzer = nullptr;

    try {
        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(audio_chars)) {
            if (entry.is_regular_file() && entry.path().extension() == ".wav") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) {
            throw std::runtime_error("no WAV files in audio directory");
        }

        char error[2048] = {};
        const auto status = pitchee_analyzer_create(
            model_chars,
            threads,
            0,
            75.0f,
            600.0f,
            0.9f,
            &analyzer,
            error,
            sizeof(error)
        );
        if (status != PITCHEE_SUCCESS) {
            throw std::runtime_error(std::string("analyzer create failed: ") + error);
        }

        analyze_file(analyzer, files.front());
        double total_ms = 0.0;
        std::ostringstream output;
        output << "PitcheeCore Android ncnn-only benchmark\n"
               << "device threads=" << threads << "\n"
               << "file | VFP | Naturalness | F0 | ms\n\n";
        for (size_t index = 0; index < files.size(); ++index) {
            const Metrics metrics = analyze_file(analyzer, files[index]);
            total_ms += metrics.elapsed_ms;
            output << std::setw(2) << std::setfill('0') << index + 1
                   << std::setfill(' ') << " " << files[index].filename().string()
                   << " | " << std::fixed << std::setprecision(3)
                   << metrics.vfp << " | "
                   << metrics.naturalness << " | "
                   << metrics.f0 << " | "
                   << metrics.elapsed_ms << "\n";
        }
        output << "\ntotal_ms=" << total_ms
               << " average_ms=" << total_ms / files.size();
        report = output.str();
        log_multiline(report);
    } catch (const std::exception& error) {
        report = std::string("Native ncnn benchmark error: ") + error.what();
        __android_log_print(
            ANDROID_LOG_ERROR,
            "PitcheeNcnnBenchmark",
            "%s",
            report.c_str()
        );
    }

    if (analyzer) pitchee_analyzer_destroy(analyzer);
    env->ReleaseStringUTFChars(model_directory, model_chars);
    env->ReleaseStringUTFChars(audio_directory, audio_chars);
    return env->NewStringUTF(report.c_str());
}
