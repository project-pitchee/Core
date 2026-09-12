#include "internal.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

bool contains(const std::string& value, const std::string& part) {
    return value.find(part) != std::string::npos;
}

}  // namespace

int main() {
    pitchee::AnalysisResult result;
    result.vfp_standard_score = 12.5;
    result.window_count = 1;
    result.window_duration_seconds = 1.5;
    result.vfp_windows.push_back({0.0, 1.5, 12.5});
    result.naturalness_score = 67.5;
    result.naturalness_window_duration_seconds = 1.5;
    result.naturalness_windows.push_back({0.0, 1.5, 70.0});
    result.voiced_frame_count = 20;
    result.voiced_window_count = 2;
    result.f0_windows.push_back({0.0, 0.1, false, 0.0});
    result.f0_windows.push_back({0.1, 0.2, true, 180.0});

    const std::string json = pitchee::result_to_json(result);
    require(contains(json, "\"schema_version\":2"), "schema version");
    require(!contains(json, "\"models\""), "models removed");
    require(!contains(json, "raw_female_score"), "raw score removed");
    require(contains(json, "\"f0\":{\"window_seconds\":0.1"), "f0 section");
    require(contains(json, "\"vfp\":{\"vfp_standard_score\":12.5"), "vfp section");
    require(
        contains(json, "\"naturalness\":{\"score\":67.5"),
        "naturalness section"
    );
    require(
        contains(json, "\"windows\":[{\"start_seconds\":0,\"end_seconds\":1.5,\"score\":70}"),
        "naturalness windows"
    );
    require(
        contains(json, "\"vfp_standard_score\":12.5"),
        "window vfp score"
    );
    std::cout << "PitcheeCore JSON test passed\n";
    return 0;
}
