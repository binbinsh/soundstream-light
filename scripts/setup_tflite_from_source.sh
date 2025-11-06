#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${ROOT_DIR}/third_party/tflite-src"
INSTALL_DIR="${ROOT_DIR}/third_party/tflite"

# Prevent Git from walking into parent cache directories that may contain
# placeholder .git files without valid contents (e.g. uv's sdist cache).
ROOT_PARENT="$(cd -- "${ROOT_DIR}/.." && pwd)"
if [[ -n "${GIT_CEILING_DIRECTORIES:-}" ]]; then
  export GIT_CEILING_DIRECTORIES="${GIT_CEILING_DIRECTORIES}:${ROOT_PARENT}"
else
  export GIT_CEILING_DIRECTORIES="${ROOT_PARENT}"
fi

BRANCH="${TFLITE_BRANCH:-master}"
GENERATOR="${CMAKE_GENERATOR:-Ninja}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

usage() {
  cat <<'EOU'
Usage: setup_tflite_from_source.sh [--branch <git-ref>] [--generator <cmake-generator>] [--build-type <type>] [--clean]

Environment Overrides:
  TFLITE_BRANCH       Git branch or tag to checkout (default: master)
  CMAKE_GENERATOR     CMake generator to use (default: Ninja)
  CMAKE_BUILD_TYPE    Build profile (default: Release)

Options:
  --branch <ref>      Git branch or tag.
  --generator <name>  CMake generator.
  --build-type <type> CMake build type.
  --clean             Remove existing source/install directories before building.
  -h, --help          Show this message.
EOU
}

CLEAN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --branch)
      BRANCH="$2"; shift 2 ;;
    --generator)
      GENERATOR="$2"; shift 2 ;;
    --build-type)
      CMAKE_BUILD_TYPE="$2"; shift 2 ;;
    --clean)
      CLEAN=true; shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1 ;;
  esac
done

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }
command -v cmake >/dev/null 2>&1 || { echo "cmake is required" >&2; exit 1; }

if [[ "${GENERATOR}" == "Ninja" ]]; then
  command -v ninja >/dev/null 2>&1 || { echo "ninja is required for Ninja generator" >&2; exit 1; }
fi

if [[ "${CLEAN}" == true ]]; then
  rm -rf "${SRC_DIR}" "${INSTALL_DIR}" "${ROOT_DIR}/third_party/tflite-deps"
fi

# Some sdists may already contain extracted sources without their .git metadata.
# Remove those snapshots so we always work with a real clone.
if [[ -d "${SRC_DIR}" && ! -d "${SRC_DIR}/.git" ]]; then
  echo "Found TensorFlow Lite sources without git metadata at ${SRC_DIR}; removing before fresh clone." >&2
  rm -rf "${SRC_DIR}"
fi

mkdir -p "${ROOT_DIR}/third_party"

if [[ ! -d "${SRC_DIR}" ]]; then
  git clone --branch "${BRANCH}" https://github.com/tensorflow/tensorflow.git "${SRC_DIR}"
else
  pushd "${SRC_DIR}" >/dev/null
  git fetch origin --tags
  git checkout "${BRANCH}"
  git pull --ff-only
  popd >/dev/null
fi

BUILD_DIR="${SRC_DIR}/cmake_build"

DEPS_DIR="${ROOT_DIR}/third_party/tflite-deps"
PSIMD_LOCAL_DIR="${DEPS_DIR}/psimd"

mkdir -p "${DEPS_DIR}"

if [[ -d "${PSIMD_LOCAL_DIR}" && ! -d "${PSIMD_LOCAL_DIR}/.git" ]]; then
  echo "Found psimd sources without git metadata at ${PSIMD_LOCAL_DIR}; refreshing clone." >&2
  rm -rf "${PSIMD_LOCAL_DIR}"
fi

if [[ ! -d "${PSIMD_LOCAL_DIR}" ]]; then
  git clone --depth 1 https://github.com/Maratyszcza/psimd.git "${PSIMD_LOCAL_DIR}"
  perl -0pi -e 's/CMAKE_MINIMUM_REQUIRED\(VERSION [0-9.]+ FATAL_ERROR\)/CMAKE_MINIMUM_REQUIRED(VERSION 3.5 FATAL_ERROR)/' "${PSIMD_LOCAL_DIR}/CMakeLists.txt"
else
  pushd "${PSIMD_LOCAL_DIR}" >/dev/null
  git fetch origin --depth 1
  git reset --hard origin/main
  popd >/dev/null
fi

export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_POLICY_VERSION_COMPATIBLE=3.27"

# Ensure downstream dependencies inherit modern policy settings.
XNNPACK_CMAKE="${SRC_DIR}/tensorflow/lite/tools/cmake/modules/xnnpack/CMakeLists.txt"
if [[ -f "${XNNPACK_CMAKE}" ]] && ! grep -q "cmake_policy(PUSH" "${XNNPACK_CMAKE}"; then
  perl -0pi -e 's|add_subdirectory\(\$\{xnnpack_SOURCE_DIR\}\s*\$\{xnnpack_BINARY_DIR\}\)|cmake_policy(PUSH)\ncmake_policy(VERSION 3.27)\nadd_subdirectory(${xnnpack_SOURCE_DIR} ${xnnpack_BINARY_DIR})\ncmake_policy(POP)|' "${XNNPACK_CMAKE}"
fi

cmake_call() {
  cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_POLICY_VERSION_COMPATIBLE=3.27 \
        "$@"
}

cmake_call -S "${SRC_DIR}/tensorflow/lite" \
      -B "${BUILD_DIR}" \
      -G "${GENERATOR}" \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -DTFLITE_ENABLE_XNNPACK=ON \
      -DTFLITE_ENABLE_INSTALL=OFF \
      -DPSIMD_SOURCE_DIR="${PSIMD_LOCAL_DIR}"

cmake --build "${BUILD_DIR}" --target tensorflow-lite

cat <<EOF

TensorFlow Lite build completed.

Libraries are available under:
  ${BUILD_DIR}/libtensorflow-lite.a
  ${BUILD_DIR}/libtensorflowlite_c.dylib (when generated)
  ${BUILD_DIR}/_deps/*-build/*.a

Use headers directly from the cloned source tree, for example:
  ${SRC_DIR}/tensorflow
  ${BUILD_DIR}/abseil-cpp
  ${BUILD_DIR}/_deps/flatbuffers-*/include

EOF

