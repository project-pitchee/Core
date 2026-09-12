#include "pitchee/pitchee.h"

#include <cstdio>
#include <cinttypes>
#include <exception>
#include <iostream>
#include <string>

namespace {

void progress_callback(const pitchee_progress_t* progress, void*) {
    std::fprintf(
        stderr,
        "[pitchee-progress] stage=%d completed=%" PRIu64
        " total=%" PRIu64 " fraction=%.6f\n",
        static_cast<int>(progress->stage),
        progress->completed,
        progress->total,
        progress->fraction
    );
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
    status = pitchee_analyzer_analyze_wav_file_with_progress(
        analyzer,
        argv[2],
        progress_callback,
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
