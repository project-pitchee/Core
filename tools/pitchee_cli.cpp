#include "pitchee/pitchee.h"

#include <cstring>
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
    if (argc != 4) {
        std::cerr << "usage: pitchee_cli <model-directory> <audio.wav> "
                     "<feminization|masculinization>\n";
        return 2;
    }

    pitchee_score_profile_t score_profile;
    if (std::strcmp(argv[3], "feminization") == 0) {
        score_profile = PITCHEE_SCORE_PROFILE_FEMINIZATION;
    } else if (std::strcmp(argv[3], "masculinization") == 0) {
        score_profile = PITCHEE_SCORE_PROFILE_MASCULINIZATION;
    } else {
        std::cerr << "invalid score profile: " << argv[3] << "\n";
        return 2;
    }

    char error[1024] = {};
    pitchee_analyzer_t* analyzer = nullptr;
    pitchee_status_t status = pitchee_analyzer_create(
        argv[1],
        2,
        1,
        75.0f,
        600.0f,
        0.9f,
        &analyzer,
        error,
        sizeof(error)
    );
    if (status != PITCHEE_SUCCESS) {
        std::cerr << "analyzer_create failed: " << error << "\n";
        return static_cast<int>(status);
    }

    char* json = nullptr;
    status = pitchee_analyze_wav_file(
        analyzer,
        argv[2],
        score_profile,
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
