cmake_minimum_required(VERSION 3.22.1)
project(pitchee_benchmark LANGUAGES C CXX)

set(PITCHEE_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(PITCHEE_BUILD_CLI OFF CACHE BOOL "" FORCE)
set(PITCHEE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PITCHEE_ENABLE_ORT OFF CACHE BOOL "" FORCE)
set(PITCHEE_ENABLE_NCNN ON CACHE BOOL "" FORCE)

add_subdirectory("${PITCHEE_CORE_DIR}" pitchee-core)

add_library(pitchee_benchmark SHARED benchmark_jni.cpp)
target_compile_features(pitchee_benchmark PRIVATE cxx_std_17)
target_include_directories(pitchee_benchmark PRIVATE
    "${PITCHEE_CORE_DIR}/include"
    "${PITCHEE_CORE_DIR}/src"
)
target_link_libraries(pitchee_benchmark PRIVATE pitchee_core log)
target_link_options(pitchee_benchmark PRIVATE "-Wl,-z,max-page-size=16384")
target_link_options(pitchee_benchmark PRIVATE "-static-openmp")
