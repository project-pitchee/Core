#include "pitchee/pitchee.h"

#include <cstdio>
#include <exception>
#include <iostream>
#include <string>

namespace {

void phase_callback(pitchee_analysis_phase_t phase, void*) {
    const char* label = "unknown";
    switch (phase) {
        case PITCHEE_PHASE_PREPARING_MODELS: label = "preparing models"; break;
        case PITCHEE_PHASE_LOADING_AUDIO: label = "loading audio"; break;
        case PITCHEE_PHASE_ANALYZING: label = "analyzing"; break;
        case PITCHEE_PHASE_COMPLETED: label = "completed"; break;
    }
    std::fprintf(stderr, "[pitchee] %s\n", label);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: pitchee_cli <model-directory> <audio.wav>\n";
        return 2;
    }

    char error[1024] = {};
    pitchee_analyzer_options_t options{};
    options.intra_op_threads = 2;
    options.use_coreml = 1;
    pitchee_analyzer_t* analyzer = nullptr;
    pitchee_status_t status = pitchee_analyzer_create(
        argv[1],
        &options,
        &analyzer,
        error,
        sizeof(error)
    );
    if (status != PITCHEE_SUCCESS) {
        std::cerr << "analyzer_create failed: " << error << "\n";
        return static_cast<int>(status);
    }

    char* json = nullptr;
    status = pitchee_analyzer_analyze_wav_file(
        analyzer,
        argv[2],
        phase_callback,
        nullptr,
        &json,
        error,
        sizeof(error)
    );
    if (status != PITCHEE_SUCCESS) {
        std::cerr << "analysis failed: " << error << "\n";
        pitchee_analyzer_destroy(analyzer);
        return static_cast<int>(status);
    }
    std::cout << json << "\n";
    pitchee_string_free(json);
    pitchee_analyzer_destroy(analyzer);
    return 0;
}
