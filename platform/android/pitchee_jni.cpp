#include "pitchee/pitchee.h"

#include <jni.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void throw_java(JNIEnv* env, const char* message) {
    jclass exception = env->FindClass("java/lang/IllegalStateException");
    env->ThrowNew(exception, message && *message ? message : "PitcheeCore error");
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_space_pitchee_core_PitcheeAnalyzer_nativeCreate(
    JNIEnv* env,
    jclass,
    jstring model_directory,
    jint threads
) {
    const char* model_chars = env->GetStringUTFChars(model_directory, nullptr);
    pitchee_analyzer_options_t options{};
    options.intra_op_threads = threads;
    options.use_coreml = 0;
    pitchee_analyzer_t* analyzer = nullptr;
    char error[1024] = {};
    const auto status = pitchee_analyzer_create(
        model_chars,
        &options,
        &analyzer,
        error,
        sizeof(error)
    );
    env->ReleaseStringUTFChars(model_directory, model_chars);
    if (status != PITCHEE_SUCCESS) {
        throw_java(env, error);
        return 0;
    }
    return reinterpret_cast<jlong>(analyzer);
}

extern "C" JNIEXPORT void JNICALL
Java_space_pitchee_core_PitcheeAnalyzer_nativeDestroy(
    JNIEnv*,
    jobject,
    jlong handle
) {
    pitchee_analyzer_destroy(reinterpret_cast<pitchee_analyzer_t*>(handle));
}

extern "C" JNIEXPORT jstring JNICALL
Java_space_pitchee_core_PitcheeAnalyzer_nativeAnalyze(
    JNIEnv* env,
    jobject,
    jlong handle_value,
    jfloatArray samples,
    jint sample_rate,
    jint channels
) {
    auto* analyzer = reinterpret_cast<pitchee_analyzer_t*>(handle_value);
    const jsize count = env->GetArrayLength(samples);
    std::vector<float> pcm(static_cast<size_t>(count));
    env->GetFloatArrayRegion(samples, 0, count, pcm.data());
    char* json = nullptr;
    char error[1024] = {};
    const auto status = pitchee_analyzer_analyze_pcm(
        analyzer,
        pcm.data(),
        pcm.size(),
        sample_rate,
        channels,
        nullptr,
        nullptr,
        &json,
        error,
        sizeof(error)
    );
    if (status != PITCHEE_SUCCESS) {
        throw_java(env, error);
        return nullptr;
    }
    jstring result = env->NewStringUTF(json);
    pitchee_string_free(json);
    return result;
}
