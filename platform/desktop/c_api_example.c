#include "pitchee/pitchee.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: c_api_example <models> <float32-pcm-file>\n");
        return 2;
    }

    FILE* audio = fopen(argv[2], "rb");
    if (!audio) {
        perror("audio");
        return 1;
    }
    fseek(audio, 0, SEEK_END);
    const long byte_count = ftell(audio);
    fseek(audio, 0, SEEK_SET);
    if (byte_count <= 0 || byte_count % 4 != 0) {
        fprintf(stderr, "expected raw Float32 PCM\n");
        fclose(audio);
        return 1;
    }

    float* samples = (float*)malloc((size_t)byte_count);
    if (!samples || fread(samples, 1, (size_t)byte_count, audio)
        != (size_t)byte_count) {
        fprintf(stderr, "failed to read PCM\n");
        free(samples);
        fclose(audio);
        return 1;
    }
    fclose(audio);

    pitchee_analyzer_options_t options = {2, 0, 0};
    pitchee_analyzer_t* analyzer = NULL;
    char error[1024] = {0};
    pitchee_status_t status = pitchee_analyzer_create(
        argv[1], &options, &analyzer, error, sizeof(error)
    );
    if (status != PITCHEE_SUCCESS) {
        fprintf(stderr, "%s\n", error);
        free(samples);
        return (int)status;
    }

    char* json = NULL;
    status = pitchee_analyzer_analyze_pcm(
        analyzer,
        samples,
        (size_t)byte_count / sizeof(float),
        16000,
        1,
        NULL,
        NULL,
        &json,
        error,
        sizeof(error)
    );
    if (status == PITCHEE_SUCCESS) {
        puts(json);
        pitchee_string_free(json);
    } else {
        fprintf(stderr, "%s\n", error);
    }

    pitchee_analyzer_destroy(analyzer);
    free(samples);
    return status == PITCHEE_SUCCESS ? 0 : (int)status;
}
