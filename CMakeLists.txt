cmake_minimum_required(VERSION 3.20)

project(
    PitcheeCore
    VERSION 0.1.0
    DESCRIPTION "Cross-platform on-device voice analysis core"
    LANGUAGES C CXX
)

if(APPLE AND CMAKE_SYSTEM_NAME STREQUAL "iOS"
   AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 15.1)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "15.1" CACHE STRING
        "Minimum iOS deployment target" FORCE)
endif()

option(PITCHEE_BUILD_SHARED "Build pitchee_core as a shared library" ON)
option(PITCHEE_BUILD_CLI "Build the desktop validation CLI" ON)
option(PITCHEE_BUILD_TESTS "Build core tests" ON)
option(PITCHEE_ENABLE_ORT "Enable ONNX Runtime model inference" ON)
set(PITCHEE_ORT_ROOT "" CACHE PATH "Directory containing ONNX Runtime headers and libraries")
set(PITCHEE_ORT_INCLUDE_DIR "" CACHE PATH "Explicit ONNX Runtime include directory")
set(PITCHEE_ORT_LIBRARY "" CACHE FILEPATH "Explicit ONNX Runtime library path")

if(PITCHEE_BUILD_SHARED)
    set(PITCHEE_LIBRARY_TYPE SHARED)
else()
    set(PITCHEE_LIBRARY_TYPE STATIC)
endif()

add_library(pitchee_core ${PITCHEE_LIBRARY_TYPE}
    src/analyzer.cpp
    src/audio.cpp
    src/json.cpp
    src/naturalness.cpp
    src/ort_runtime.cpp
    src/scoring.cpp
    src/spectrum.cpp
    src/vad.cpp
    src/wav_reader.cpp
)

add_library(Pitchee::Core ALIAS pitchee_core)

target_compile_features(pitchee_core PUBLIC cxx_std_17)
target_compile_definitions(pitchee_core PRIVATE PITCHEE_CORE_BUILD=1)
if(NOT PITCHEE_BUILD_SHARED)
    target_compile_definitions(pitchee_core PUBLIC PITCHEE_STATIC=1)
endif()
target_include_directories(pitchee_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
set_target_properties(pitchee_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME pitchee_core
)

if(MSVC)
    target_compile_options(pitchee_core PRIVATE /W4 /permissive- /EHsc)
else()
    target_compile_options(pitchee_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(PITCHEE_ENABLE_ORT)
    if(PITCHEE_ORT_INCLUDE_DIR)
        set(PITCHEE_ORT_INCLUDE "${PITCHEE_ORT_INCLUDE_DIR}")
    else()
        find_path(PITCHEE_ORT_INCLUDE onnxruntime_cxx_api.h
            HINTS
                "${PITCHEE_ORT_ROOT}/include"
                "$ENV{ONNXRUNTIME_ROOT}/include"
        )
    endif()

    if(PITCHEE_ORT_LIBRARY)
        set(PITCHEE_ORT_LIB "${PITCHEE_ORT_LIBRARY}")
    else()
        find_library(PITCHEE_ORT_LIB
            NAMES onnxruntime
            HINTS
                "${PITCHEE_ORT_ROOT}/lib"
                "${PITCHEE_ORT_ROOT}/lib64"
                "${PITCHEE_ORT_ROOT}/bin"
                "$ENV{ONNXRUNTIME_ROOT}/lib"
                "$ENV{ONNXRUNTIME_ROOT}/lib64"
        )
    endif()

    if(PITCHEE_ORT_INCLUDE AND PITCHEE_ORT_LIB)
        add_library(pitchee_onnxruntime UNKNOWN IMPORTED)
        set_target_properties(pitchee_onnxruntime PROPERTIES
            IMPORTED_LOCATION "${PITCHEE_ORT_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${PITCHEE_ORT_INCLUDE}"
        )
        target_link_libraries(pitchee_core PRIVATE pitchee_onnxruntime)
        target_compile_definitions(pitchee_core PRIVATE PITCHEE_ENABLE_ORT=1)
        message(STATUS "PitcheeCore: ONNX Runtime enabled")
        message(STATUS "  include: ${PITCHEE_ORT_INCLUDE}")
        message(STATUS "  library: ${PITCHEE_ORT_LIB}")
    else()
        message(WARNING "PitcheeCore: ONNX Runtime not found; model inference returns PITCHEE_ERROR_ORT_UNAVAILABLE")
    endif()
endif()

if(APPLE)
    target_link_libraries(pitchee_core PRIVATE
        "-framework Foundation"
        "-framework CoreML"
        "-framework Accelerate"
        "-framework CoreGraphics"
    )
endif()

if(PITCHEE_BUILD_CLI)
    add_executable(pitchee_cli tools/pitchee_cli.cpp)
    target_link_libraries(pitchee_cli PRIVATE pitchee_core)
endif()

if(PITCHEE_BUILD_TESTS)
    enable_testing()
    add_executable(pitchee_core_tests tests/core_tests.cpp)
    target_link_libraries(pitchee_core_tests PRIVATE pitchee_core)
    add_test(NAME pitchee_core_tests COMMAND pitchee_core_tests)

    add_executable(pitchee_resample_test tests/resample_test.cpp src/audio.cpp)
    target_compile_features(pitchee_resample_test PRIVATE cxx_std_17)
    target_include_directories(pitchee_resample_test PRIVATE src)
    add_test(NAME pitchee_resample_test COMMAND pitchee_resample_test)

    add_executable(pitchee_json_test tests/json_test.cpp src/json.cpp)
    target_compile_features(pitchee_json_test PRIVATE cxx_std_17)
    target_include_directories(pitchee_json_test PRIVATE src)
    add_test(NAME pitchee_json_test COMMAND pitchee_json_test)

    add_executable(pitchee_spectrum_test tests/spectrum_test.cpp)
    target_link_libraries(pitchee_spectrum_test PRIVATE pitchee_core)
    add_test(NAME pitchee_spectrum_test COMMAND pitchee_spectrum_test)

    add_executable(pitchee_c_abi_test tests/c_abi_test.c)
    target_link_libraries(pitchee_c_abi_test PRIVATE pitchee_core)
    add_test(NAME pitchee_c_abi_test COMMAND pitchee_c_abi_test)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/PitcheeCoreConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/PitcheeCoreConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PitcheeCore
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/PitcheeCoreConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(TARGETS pitchee_core
    EXPORT PitcheeCoreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY models/ DESTINATION share/pitchee/models)
install(FILES LICENSE DESTINATION share/pitchee)
install(EXPORT PitcheeCoreTargets
    FILE PitcheeCoreTargets.cmake
    NAMESPACE Pitchee::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PitcheeCore
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/PitcheeCoreConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/PitcheeCoreConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/PitcheeCore
)
