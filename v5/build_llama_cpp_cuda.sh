#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR=""
PYTHON_PATH=""
PIP_PATH=""
ALLOW_CPU_FALLBACK="${AGENT_ALLOW_CPU_FALLBACK:-1}"
FORCE_CPU="0"
YES_MODE="0"
CUDA_ROOT_DEFAULT="${CUDA_ROOT_DEFAULT:-/usr/local/cuda}"
LLAMA_CUDA_ARCH="${LLAMA_CUDA_ARCH:-}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$ROOT_DIR/llama.cpp}"
LLAMA_CPP_REPO_URL="${LLAMA_CPP_REPO_URL:-https://github.com/ggml-org/llama.cpp}"
LLAMA_CPP_GIT_REF="${LLAMA_CPP_GIT_REF:-master}"
LOG_DIR="${SETUP_LOG_DIR:-$ROOT_DIR/.setup_logs}"
CUDA_BUILD_LOG="$LOG_DIR/llama_cpp_cuda_build.log"
LAST_CMAKE_LOG="$LOG_DIR/llama_cpp_cuda_cmake_probe.log"
LLAMA_CPP_BUILD_LOG="$LOG_DIR/llama_cpp_repo_build.log"

usage() {
  cat <<USAGE
Usage: $0 [--env-dir PATH | --python PATH] [--allow-cpu-fallback 0|1] [--yes] [--force-cpu]

Options:
  --env-dir PATH            Virtualenv directory to target.
  --python PATH             Exact python interpreter to use.
  --allow-cpu-fallback N    Install CPU build if CUDA build fails. Default: $ALLOW_CPU_FALLBACK.
  --yes                     Skip interactive confirmation prompts.
  --force-cpu               Skip CUDA build attempts and install CPU build only.
  --llama-cpp-dir PATH      Clone or reuse llama.cpp here. Default: $LLAMA_CPP_DIR.
  --cuda-arch VALUE         Override CMAKE_CUDA_ARCHITECTURES for both builds.
  --log-dir PATH            Directory for build logs.
  --help                    Show this help.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --env-dir)
      ENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_PATH="$2"
      shift 2
      ;;
    --allow-cpu-fallback)
      ALLOW_CPU_FALLBACK="$2"
      shift 2
      ;;
    --yes)
      YES_MODE="1"
      shift
      ;;
    --force-cpu)
      FORCE_CPU="1"
      shift
      ;;
    --log-dir)
      LOG_DIR="$2"
      CUDA_BUILD_LOG="$LOG_DIR/llama_cpp_cuda_build.log"
      LAST_CMAKE_LOG="$LOG_DIR/llama_cpp_cuda_cmake_probe.log"
      LLAMA_CPP_BUILD_LOG="$LOG_DIR/llama_cpp_repo_build.log"
      shift 2
      ;;
    --llama-cpp-dir)
      LLAMA_CPP_DIR="$2"
      shift 2
      ;;
    --cuda-arch)
      LLAMA_CUDA_ARCH="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_DIR"

if [ -n "$ENV_DIR" ]; then
  PYTHON_PATH="${PYTHON_PATH:-$ENV_DIR/bin/python}"
  PIP_PATH="$ENV_DIR/bin/pip"
fi

if [ -z "$PYTHON_PATH" ]; then
  PYTHON_PATH="$(command -v python3 || true)"
fi

if [ -z "$PYTHON_PATH" ] || [ ! -x "$PYTHON_PATH" ]; then
  echo "Python interpreter not found." >&2
  exit 1
fi

if [ -z "$PIP_PATH" ]; then
  PIP_PATH="$(dirname "$PYTHON_PATH")/pip"
fi

if [ ! -x "$PIP_PATH" ]; then
  echo "pip not found for interpreter: $PYTHON_PATH" >&2
  exit 1
fi

list_cuda_roots() {
  local candidate
  local seen=""
  for candidate in \
    "${CUDA_HOME:-}" \
    "${CUDAToolkit_ROOT:-}" \
    "$CUDA_ROOT_DEFAULT" \
    /usr/local/cuda \
    /usr/local/cuda-13.1 \
    /usr/local/cuda-12.6 \
    /usr/local/cuda-12.4 \
    /usr/local/cuda-[0-9]*; do
    if [ -z "$candidate" ] || [ ! -d "$candidate" ] || [ ! -x "$candidate/bin/nvcc" ]; then
      continue
    fi
    case " $seen " in
      *" $candidate "*) continue ;;
    esac
    printf '%s\n' "$candidate"
    seen="$seen $candidate"
  done
}

cuda_toolkit_ready() {
  local cuda_root="$1"
  [ -n "$cuda_root" ] || return 1
  [ -x "$cuda_root/bin/nvcc" ] || return 1
  [ -f "$cuda_root/include/cublas_v2.h" ] || return 1
  [ -f "$cuda_root/targets/x86_64-linux/lib/libcublas.so" ] || [ -f "$cuda_root/lib64/libcublas.so" ] || return 1
}

write_cuda_env() {
  local cuda_root="$1"
  export CUDA_HOME="$cuda_root"
  export CUDAToolkit_ROOT="$cuda_root"
  export CUDACXX="$cuda_root/bin/nvcc"
  export LD_LIBRARY_PATH="$cuda_root/lib64:$cuda_root/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
}

cuda_arch_cmake_args() {
  if [ -n "$LLAMA_CUDA_ARCH" ]; then
    printf '%s' "-DCMAKE_CUDA_ARCHITECTURES=$LLAMA_CUDA_ARCH -DGGML_NATIVE=OFF"
  fi
}

ensure_build_prereqs() {
  "$PIP_PATH" install --upgrade pip setuptools wheel cmake ninja
}

confirm_build() {
  if [ "$YES_MODE" = "1" ]; then
    return 0
  fi
  if [ ! -t 0 ]; then
    return 0
  fi
  local response
  read -r -p "Attempt CUDA llama-cpp-python build now? [Y/n] " response
  response="${response:-y}"
  case "$response" in
    y|Y|yes|YES|"") return 0 ;;
    *) return 1 ;;
  esac
}

run_cmake_probe() {
  local cuda_root="$1"
  local probe_dir
  probe_dir="$(mktemp -d)"
  cat > "$probe_dir/CMakeLists.txt" <<PROBE
cmake_minimum_required(VERSION 3.24)
project(cuda_probe LANGUAGES C CXX CUDA)
find_package(CUDAToolkit REQUIRED)
if(TARGET CUDA::cublas)
  message(STATUS "CUDA::cublas target exists")
else()
  message(FATAL_ERROR "CUDA::cublas target missing")
endif()
PROBE
  if cmake -S "$probe_dir" -B "$probe_dir/build" -DCUDAToolkit_ROOT="$cuda_root" -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" >"$LAST_CMAKE_LOG" 2>&1; then
    rm -rf "$probe_dir"
    return 0
  fi
  rm -rf "$probe_dir"
  return 1
}

install_cpu() {
  echo "Installing llama-cpp-python CPU build..."
  "$PIP_PATH" install --upgrade llama-cpp-python
}

ensure_llama_cpp_checkout() {
  local repo_dir="$1"
  if [ -d "$repo_dir/.git" ]; then
    echo "Reusing existing llama.cpp checkout: $repo_dir"
    return 0
  fi
  if [ -e "$repo_dir" ] && [ ! -d "$repo_dir/.git" ]; then
    echo "Path exists but is not a git checkout: $repo_dir" >&2
    return 1
  fi
  echo "Cloning llama.cpp into $repo_dir"
  git clone --depth 1 --branch "$LLAMA_CPP_GIT_REF" "$LLAMA_CPP_REPO_URL" "$repo_dir"
}

build_llama_cpp_repo() {
  local cuda_root="$1"
  local -a cmake_args
  cmake_args=(
    -DGGML_CUDA=ON
    -DCMAKE_BUILD_TYPE=Release
    -DCUDAToolkit_ROOT="$cuda_root"
    -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc"
  )
  if [ -n "$LLAMA_CUDA_ARCH" ]; then
    cmake_args+=("-DCMAKE_CUDA_ARCHITECTURES=$LLAMA_CUDA_ARCH" "-DGGML_NATIVE=OFF")
  fi

  echo "Building llama.cpp with CUDA from $cuda_root"
  if [ -n "$LLAMA_CUDA_ARCH" ]; then
    echo "Using CUDA architecture override: $LLAMA_CUDA_ARCH"
  fi

  if cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" "${cmake_args[@]}" \
    >"$LLAMA_CPP_BUILD_LOG" 2>&1 && \
    cmake --build "$LLAMA_CPP_DIR/build" --config Release >>"$LLAMA_CPP_BUILD_LOG" 2>&1; then
    return 0
  fi

  echo "llama.cpp CUDA build failed. Build log: $LLAMA_CPP_BUILD_LOG" >&2
  tail -n 40 "$LLAMA_CPP_BUILD_LOG" >&2 || true
  return 1
}

attempt_cuda_install() {
  local attempt_name="$1"
  local cmake_args="$2"

  echo
  echo "CUDA build attempt: $attempt_name"
  echo "CMAKE_ARGS=$cmake_args"
  if env CMAKE_ARGS="$cmake_args" FORCE_CMAKE=1 \
    "$PYTHON_PATH" -m pip install --no-cache-dir --force-reinstall --no-binary=llama-cpp-python llama-cpp-python \
    >"$CUDA_BUILD_LOG" 2>&1; then
    echo "CUDA build succeeded."
    return 0
  fi

  echo "CUDA build attempt failed: $attempt_name" >&2
  echo "Build log: $CUDA_BUILD_LOG" >&2
  tail -n 40 "$CUDA_BUILD_LOG" >&2 || true
  return 1
}

main() {
  local ready_cuda_roots=()
  local incomplete_cuda_roots=()
  local cuda_root=""
  "$PIP_PATH" uninstall -y llama-cpp-python >/dev/null 2>&1 || true

  if [ "$FORCE_CPU" = "1" ]; then
    install_cpu
    return 0
  fi

  if ! confirm_build; then
    echo "Skipped CUDA build by user choice."
    if [ "$ALLOW_CPU_FALLBACK" = "1" ] || [ "$ALLOW_CPU_FALLBACK" = "true" ]; then
      install_cpu
      return 0
    fi
    return 1
  fi

  while IFS= read -r cuda_root; do
    if cuda_toolkit_ready "$cuda_root"; then
      ready_cuda_roots+=("$cuda_root")
    else
      incomplete_cuda_roots+=("$cuda_root")
    fi
  done < <(list_cuda_roots)

  if [ "${#ready_cuda_roots[@]}" -eq 0 ]; then
    echo "No complete CUDA toolkit root detected." >&2
    if [ "${#incomplete_cuda_roots[@]}" -gt 0 ]; then
      printf 'Incomplete CUDA toolkits were found but skipped:\n' >&2
      printf '  %s\n' "${incomplete_cuda_roots[@]}" >&2
    fi
    if [ "$ALLOW_CPU_FALLBACK" = "1" ] || [ "$ALLOW_CPU_FALLBACK" = "true" ]; then
      install_cpu
      return 0
    fi
    return 1
  fi

  printf 'CUDA build candidates:\n'
  if [ -n "$LLAMA_CUDA_ARCH" ]; then
    printf 'CUDA architecture override: %s\n' "$LLAMA_CUDA_ARCH"
  fi
  printf '  %s\n' "${ready_cuda_roots[@]}"
  if [ "${#incomplete_cuda_roots[@]}" -gt 0 ]; then
    printf 'Skipped incomplete CUDA toolkits:\n'
    printf '  %s\n' "${incomplete_cuda_roots[@]}"
  fi

  ensure_build_prereqs
  ensure_llama_cpp_checkout "$LLAMA_CPP_DIR"

  for cuda_root in "${ready_cuda_roots[@]}"; do
    write_cuda_env "$cuda_root"
    echo "Detected CUDA toolkit root: $cuda_root"

    if ! run_cmake_probe "$cuda_root"; then
      echo "CMake CUDA probe failed for $cuda_root. See: $LAST_CMAKE_LOG" >&2
      continue
    fi

    if ! build_llama_cpp_repo "$cuda_root"; then
      continue
    fi

    pinned_args="-DGGML_CUDA=on -DCUDAToolkit_ROOT=$cuda_root -DCMAKE_CUDA_COMPILER=$cuda_root/bin/nvcc -DCMAKE_LIBRARY_PATH=$cuda_root/targets/x86_64-linux/lib -DCMAKE_INCLUDE_PATH=$cuda_root/include"
    if [ -n "$LLAMA_CUDA_ARCH" ]; then
      pinned_args="$pinned_args -DCMAKE_CUDA_ARCHITECTURES=$LLAMA_CUDA_ARCH -DGGML_NATIVE=OFF"
    fi
    if attempt_cuda_install "pinned-toolkit-paths" "$pinned_args"; then
      return 0
    fi

    minimal_args="-DGGML_CUDA=on -DCUDAToolkit_ROOT=$cuda_root -DCMAKE_CUDA_COMPILER=$cuda_root/bin/nvcc"
    if [ -n "$LLAMA_CUDA_ARCH" ]; then
      minimal_args="$minimal_args -DCMAKE_CUDA_ARCHITECTURES=$LLAMA_CUDA_ARCH -DGGML_NATIVE=OFF"
    fi
    if attempt_cuda_install "minimal-cuda" "$minimal_args"; then
      return 0
    fi
  done

  echo "Repeated CUDA build failures detected. The agent will not use GPU until this is fixed." >&2
  echo "CMake probe log: $LAST_CMAKE_LOG" >&2
  echo "Build log: $CUDA_BUILD_LOG" >&2
  echo "llama.cpp repo build log: $LLAMA_CPP_BUILD_LOG" >&2

  if [ "$ALLOW_CPU_FALLBACK" = "1" ] || [ "$ALLOW_CPU_FALLBACK" = "true" ]; then
    echo "Falling back to CPU build after CUDA build failures." >&2
    install_cpu
    return 0
  fi

  echo "CPU fallback is disabled. Leaving llama-cpp-python uninstalled." >&2
  return 1
}

main
