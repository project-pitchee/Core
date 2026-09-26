#include "wav_reader.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << "\n";
        std::exit(1);
    }
}

}  // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path()
        / ("pitchee-wav-test-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()
        ) + ".wav");
    std::vector<unsigned char> bytes{
        'R', 'I', 'F', 'F', 40, 0, 0, 0, 'W', 'A', 'V', 'E',
        'f', 'm', 't', ' ', 16, 0, 0, 0,
        1, 0, 1, 0, 0x80, 0x3e, 0, 0, 0, 0x7d, 0, 0,
        2, 0, 16, 0,
        'd', 'a', 't', 'a', 4, 0, 0, 0, 0, 0x80, 0, 0x40
    };
    const auto write_fixture = [&] {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    write_fixture();
    const auto wav = pitchee::read_wav(path);
    std::filesystem::remove(path);
    require(wav.sample_rate == 16000 && wav.channels == 1, "PCM format");
    require(wav.samples == std::vector<float>({-1.0f, 0.5f}), "signed PCM16 samples");

    // A declared data size with its high bit set must remain unsigned and be
    // rejected as truncated, without a signed shift overflow or an over-read.
    bytes[43] = 0x80;
    write_fixture();
    bool rejected = false;
    try {
        pitchee::read_wav(path);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    std::filesystem::remove(path);
    require(rejected, "oversized WAV chunk must be rejected");
    std::cout << "PitcheeCore WAV reader test passed\n";
    return 0;
}
