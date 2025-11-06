cmake_minimum_required(VERSION 3.18)
project(soundstream_cpp LANGUAGES CXX)

option(SOUNDSTREAM_BUILD_CLI "Build the soundstream_cli binary" ON)
option(SOUNDSTREAM_BUILD_PYTHON "Build the soundstream_light._native Python module" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TFLITE_ROOT "" CACHE PATH "Path to a pre-installed TensorFlow Lite tree (include+lib). Leave empty when using a source build.")
set(TFLITE_SOURCE_DIR "" CACHE PATH "Path to the cloned TensorFlow source tree (tflite-src) when building from source")
set(TFLITE_BUILD_DIR "" CACHE PATH "Path to the TensorFlow Lite CMake build directory (cmake_build) when building from source")
set(TFLITE_FORCE_LOAD_OPTIONS "")

set(SOUNDSTREAM_TFLITE_MODE "")
if(TFLITE_SOURCE_DIR AND TFLITE_BUILD_DIR)
  set(SOUNDSTREAM_TFLITE_MODE "build")
elseif(TFLITE_ROOT)
  set(SOUNDSTREAM_TFLITE_MODE "install")
else()
  set(_tflite_bootstrap_script "${CMAKE_CURRENT_SOURCE_DIR}/scripts/setup_tflite_from_source.sh")
  if(EXISTS "${_tflite_bootstrap_script}")
    find_program(BASH_EXECUTABLE bash)
    if(NOT BASH_EXECUTABLE)
      message(FATAL_ERROR
        "TensorFlow Lite paths were not provided and 'bash' was not found to run ${_tflite_bootstrap_script}. "
        "Install bash or set TFLITE_ROOT / TFLITE_SOURCE_DIR + TFLITE_BUILD_DIR explicitly.")
    endif()

    message(STATUS
      "TensorFlow Lite paths not provided; bootstrapping via ${_tflite_bootstrap_script}")
    execute_process(
      COMMAND "${BASH_EXECUTABLE}" "${_tflite_bootstrap_script}" --branch v2.15.0 --generator "Unix Makefiles" --build-type Release
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE _tflite_bootstrap_result
    )
    if(NOT _tflite_bootstrap_result EQUAL 0)
      message(FATAL_ERROR
        "Automatic TensorFlow Lite bootstrap failed (exit code ${_tflite_bootstrap_result}). "
        "Provide TFLITE_ROOT or TFLITE_SOURCE_DIR/TFLITE_BUILD_DIR manually.")
    endif()

    if(NOT TFLITE_SOURCE_DIR)
      set(TFLITE_SOURCE_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite-src"
        CACHE PATH "Path to the cloned TensorFlow source tree (auto-configured)" FORCE)
    endif()
    if(NOT TFLITE_BUILD_DIR)
      set(TFLITE_BUILD_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tflite-src/cmake_build"
        CACHE PATH "Path to the TensorFlow Lite CMake build directory (auto-configured)" FORCE)
    endif()
    set(SOUNDSTREAM_TFLITE_MODE "build")
  endif()
endif()

if(NOT SOUNDSTREAM_TFLITE_MODE)
  message(FATAL_ERROR
    "Set either TFLITE_ROOT (for an installed layout) or both TFLITE_SOURCE_DIR and TFLITE_BUILD_DIR (for the source build).")
endif()

find_package(Threads REQUIRED)

if(SOUNDSTREAM_TFLITE_MODE STREQUAL "install")
  find_path(TFLITE_INCLUDE_DIR tensorflow/lite/interpreter.h
    HINTS "${TFLITE_ROOT}/include" "${TFLITE_ROOT}"
    REQUIRED)

  find_library(TFLITE_LIBRARY
    NAMES tensorflowlite tensorflow-lite
    HINTS "${TFLITE_ROOT}/lib" "${TFLITE_ROOT}/lib64" "${TFLITE_ROOT}"
    REQUIRED)

  set(TFLITE_INCLUDE_DIRS "${TFLITE_INCLUDE_DIR}")
  set(TFLITE_LIBS "${TFLITE_LIBRARY}")
else()
  if(NOT IS_DIRECTORY "${TFLITE_SOURCE_DIR}")
    message(FATAL_ERROR "TFLITE_SOURCE_DIR ('${TFLITE_SOURCE_DIR}') does not exist or is not a directory.")
  endif()
  if(NOT IS_DIRECTORY "${TFLITE_BUILD_DIR}")
    message(FATAL_ERROR "TFLITE_BUILD_DIR ('${TFLITE_BUILD_DIR}') does not exist or is not a directory.")
  endif()

  set(TFLITE_CORE_LIB "${TFLITE_BUILD_DIR}/libtensorflow-lite.a")
  if(NOT EXISTS "${TFLITE_CORE_LIB}")
    message(FATAL_ERROR "TensorFlow Lite core library not found at '${TFLITE_CORE_LIB}'. Build it with the setup script first.")
  endif()

  set(_candidate_include_dirs
    "${TFLITE_SOURCE_DIR}"
    "${TFLITE_SOURCE_DIR}/tensorflow"
    "${TFLITE_SOURCE_DIR}/tensorflow/lite"
    "${TFLITE_SOURCE_DIR}/third_party"
    "${TFLITE_SOURCE_DIR}/third_party/xla/third_party/tsl"
    "${TFLITE_BUILD_DIR}/abseil-cpp"
    "${TFLITE_BUILD_DIR}/flatbuffers/include"
    "${TFLITE_BUILD_DIR}/_deps/flatbuffers-build/include"
    "${TFLITE_BUILD_DIR}/eigen"
    "${TFLITE_BUILD_DIR}/_deps/ml_dtypes-src/ml_dtypes/include"
    "${TFLITE_BUILD_DIR}/pthreadpool-source/include"
    "${TFLITE_BUILD_DIR}/cpuinfo/include"
    "${TFLITE_BUILD_DIR}/xnnpack/include"
  )
  set(TFLITE_INCLUDE_DIRS "")
  foreach(dir IN LISTS _candidate_include_dirs)
    if(IS_DIRECTORY "${dir}")
      list(APPEND TFLITE_INCLUDE_DIRS "${dir}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES TFLITE_INCLUDE_DIRS)

  file(GLOB TFLITE_CORE_DEPS CONFIGURE_DEPENDS
    "${TFLITE_BUILD_DIR}/libtensorflowlite_c.*"
    "${TFLITE_BUILD_DIR}/flatbuffers/lib*.a"
    "${TFLITE_BUILD_DIR}/flatbuffers/lib*.dylib"
    "${TFLITE_BUILD_DIR}/_deps/flatbuffers-build/lib*.a"
    "${TFLITE_BUILD_DIR}/_deps/flatbuffers-build/lib*.dylib"
    "${TFLITE_BUILD_DIR}/_deps/cpuinfo-build/lib*.a"
    "${TFLITE_BUILD_DIR}/_deps/gemmlowp-build/lib*.a"
    "${TFLITE_BUILD_DIR}/_deps/farmhash-build/lib*.a"
    "${TFLITE_BUILD_DIR}/_deps/fft2d-build/lib*.a"
    "${TFLITE_BUILD_DIR}/_deps/xnnpack-build/lib*.a"
    "${TFLITE_BUILD_DIR}/pthreadpool/lib*.a"
    "${TFLITE_BUILD_DIR}/pthreadpool/lib*.dylib"
  )
  file(GLOB TFLITE_CORE_DEPS_TOPLEVEL CONFIGURE_DEPENDS
    "${TFLITE_BUILD_DIR}/_deps/cpuinfo-build/lib*.a"
    "${TFLITE_BUILD_DIR}/pthreadpool/lib*.a"
    "${TFLITE_BUILD_DIR}/pthreadpool/lib*.dylib"
    "${TFLITE_BUILD_DIR}/_deps/farmhash-build/lib*.a"
    "${TFLITE_BUILD_DIR}/_deps/fft2d-build/lib*.a"
    "${TFLITE_BUILD_DIR}/flatbuffers/lib*.a"
    "${TFLITE_BUILD_DIR}/_deps/xnnpack-build/lib*.a"
  )
  file(GLOB TFLITE_CORE_DEPS_RUY CONFIGURE_DEPENDS
    "${TFLITE_BUILD_DIR}/ruy/libruy*.a"
    "${TFLITE_BUILD_DIR}/ruy/libruy*.dylib"
    "${TFLITE_BUILD_DIR}/_deps/ruy-build/ruy/libruy*.a"
  )
  file(GLOB_RECURSE TFLITE_ABSL_LIBS CONFIGURE_DEPENDS "${TFLITE_BUILD_DIR}/_deps/abseil-cpp-build/absl*/libabsl_*.a")

  set(TFLITE_LIBS "${TFLITE_CORE_LIB}")
  set(TFLITE_FORCE_LOAD_OPTIONS "")
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    foreach(lib IN LISTS TFLITE_CORE_DEPS TFLITE_CORE_DEPS_TOPLEVEL TFLITE_CORE_DEPS_RUY TFLITE_ABSL_LIBS)
      if(EXISTS "${lib}")
        list(APPEND TFLITE_LIBS "${lib}")
        list(APPEND TFLITE_FORCE_LOAD_OPTIONS "-Wl,-force_load,${lib}")
      endif()
    endforeach()
  else()
    foreach(lib IN LISTS TFLITE_CORE_DEPS TFLITE_CORE_DEPS_TOPLEVEL TFLITE_CORE_DEPS_RUY TFLITE_ABSL_LIBS)
      if(EXISTS "${lib}")
        list(APPEND TFLITE_LIBS "${lib}")
      endif()
    endforeach()
  endif()
  list(REMOVE_DUPLICATES TFLITE_LIBS)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  find_library(COREFOUNDATION_FRAMEWORK CoreFoundation REQUIRED)
  list(APPEND TFLITE_LIBS "${COREFOUNDATION_FRAMEWORK}")
endif()

set(SOUNDSTREAM_SOURCES
  src/model_wrapper.cc
  src/encoder.cc
  src/decoder.cc
  src/embedding_io.cc
  src/wav_io.cc
  src/pipeline.cc
)

add_library(soundstream STATIC ${SOUNDSTREAM_SOURCES})
target_include_directories(soundstream
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TFLITE_INCLUDE_DIRS}
)
target_link_libraries(soundstream
  PUBLIC
    ${TFLITE_LIBS}
    Threads::Threads
)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  if(TFLITE_FORCE_LOAD_OPTIONS)
    target_link_options(soundstream PUBLIC ${TFLITE_FORCE_LOAD_OPTIONS})
  endif()
endif()

if(SOUNDSTREAM_BUILD_CLI)
  add_executable(soundstream_cli src/main.cpp)
  target_link_libraries(soundstream_cli PRIVATE soundstream)
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(soundstream_cli PRIVATE ${TFLITE_LIBS})
  else()
    target_link_libraries(soundstream_cli PRIVATE "-Wl,--start-group" ${TFLITE_LIBS} "-Wl,--end-group")
  endif()

  add_executable(soundstream_roundtrip_test
    ${PROJECT_SOURCE_DIR}/tests/roundtrip_test.cc)
  target_link_libraries(soundstream_roundtrip_test PRIVATE soundstream)
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(soundstream_roundtrip_test PRIVATE ${TFLITE_LIBS})
  else()
    target_link_libraries(soundstream_roundtrip_test PRIVATE "-Wl,--start-group" ${TFLITE_LIBS} "-Wl,--end-group")
  endif()
  target_compile_definitions(soundstream_roundtrip_test PRIVATE
    SOUNDSTREAM_MODELS_PATH="${PROJECT_SOURCE_DIR}/models"
    SOUNDSTREAM_SAMPLE_WAV="${PROJECT_SOURCE_DIR}/test.wav")
  add_test(NAME soundstream_roundtrip COMMAND soundstream_roundtrip_test)
endif()

if(SOUNDSTREAM_BUILD_PYTHON)
  find_package(pybind11 CONFIG QUIET)
  if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)
  endif()

  pybind11_add_module(soundstream_light_native MODULE src/python_module.cc)
  target_link_libraries(soundstream_light_native PRIVATE soundstream)
  set_target_properties(soundstream_light_native PROPERTIES OUTPUT_NAME "_native")
  install(TARGETS soundstream_light_native DESTINATION soundstream_light)
endif()

enable_testing()

