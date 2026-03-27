#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_ENV_DIR="${AGENT_ENV_DIR:-$ROOT_DIR/.venvs/agent}"
USER_ENV_DIR="${USER_ENV_DIR:-$ROOT_DIR/.venvs/user}"
USER_PYTHON_VERSION="${USER_PYTHON_VERSION:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH_DEFAULT="$ROOT_DIR/llama.cpp/IQuest_Coder_V1_40B_Q4_K_M.gguf"
AGENT_ENABLE_CUDA="${AGENT_ENABLE_CUDA:-auto}"
AGENT_ALLOW_CPU_FALLBACK="${AGENT_ALLOW_CPU_FALLBACK:-1}"
AGENT_CONFIRM_CUDA_BUILD="${AGENT_CONFIRM_CUDA_BUILD:-ask}"
CUDA_ROOT_DEFAULT="${CUDA_ROOT_DEFAULT:-/usr/local/cuda}"
LLAMA_CUDA_ARCH="${LLAMA_CUDA_ARCH:-}"
SETUP_LOG_DIR="${SETUP_LOG_DIR:-$ROOT_DIR/.setup_logs}"

mkdir -p "$ROOT_DIR/.venvs"
mkdir -p "$SETUP_LOG_DIR"

resolve_conda_exe() {
  local candidate
  for candidate in "${CONDA_EXE:-}" "$(command -v conda 2>/dev/null || true)" /opt/miniforge3/bin/conda /opt/miniforge3/condabin/conda; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      printf "%s\n" "$candidate"
      return 0
    fi
  done
  return 1
}

CONDA_EXE_PATH="$(resolve_conda_exe || true)"

echo "Agent env: $AGENT_ENV_DIR"
echo "User env:  $USER_ENV_DIR"
if [ -n "$CONDA_EXE_PATH" ]; then
  echo "Conda exe: $CONDA_EXE_PATH"
fi
echo "CUDA mode: $AGENT_ENABLE_CUDA"
echo "CPU fallback allowed: $AGENT_ALLOW_CPU_FALLBACK"
if [ -n "$LLAMA_CUDA_ARCH" ]; then
  echo "CUDA arch override: $LLAMA_CUDA_ARCH"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -d "$AGENT_ENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$AGENT_ENV_DIR"
fi

if [ -z "$CONDA_EXE_PATH" ]; then
  echo "Conda executable not found; cannot provision user environment." >&2
  exit 1
fi

if [ -d "$USER_ENV_DIR" ] && [ ! -d "$USER_ENV_DIR/conda-meta" ] && [ -f "$USER_ENV_DIR/pyvenv.cfg" ]; then
  USER_ENV_BACKUP="$USER_ENV_DIR.venv-backup.$(date +%Y%m%d%H%M%S)"
  echo "Existing user venv detected at $USER_ENV_DIR; moving it to $USER_ENV_BACKUP before creating conda env." 
  mv "$USER_ENV_DIR" "$USER_ENV_BACKUP"
fi

if [ ! -d "$USER_ENV_DIR/conda-meta" ]; then
  if [ -z "$USER_PYTHON_VERSION" ]; then
    USER_PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(str(sys.version_info.major) + "." + str(sys.version_info.minor))')"
  fi
  mkdir -p "$(dirname "$USER_ENV_DIR")"
  "$CONDA_EXE_PATH" create -y -p "$USER_ENV_DIR" "python=$USER_PYTHON_VERSION" pip
fi

AGENT_PY="$AGENT_ENV_DIR/bin/python"
AGENT_PIP="$AGENT_ENV_DIR/bin/pip"
USER_PY="$USER_ENV_DIR/bin/python"
USER_PIP="$USER_ENV_DIR/bin/pip"

install_agent_requirements() {
  local filtered_requirements
  filtered_requirements="$(mktemp)"
  grep -Ev '^[[:space:]]*llama-cpp-python([[:space:]]|[<>=!~].*)?$' "$ROOT_DIR/requirements-agent.txt" > "$filtered_requirements"
  "$AGENT_PIP" install -r "$filtered_requirements"
  rm -f "$filtered_requirements"
}

cuda_requested() {
  [ "$AGENT_ENABLE_CUDA" = "1" ] || [ "$AGENT_ENABLE_CUDA" = "true" ] || {
    [ "$AGENT_ENABLE_CUDA" = "auto" ] && command -v nvidia-smi >/dev/null 2>&1
  }
}

detect_cuda_root() {
  local candidate
  for candidate in "${CUDA_HOME:-}" "${CUDAToolkit_ROOT:-}" "$CUDA_ROOT_DEFAULT" /usr/local/cuda /usr/local/cuda-13.1 /usr/local/cuda-12.6 /usr/local/cuda-12.4; do
    if [ -n "$candidate" ] && [ -d "$candidate" ] && [ -x "$candidate/bin/nvcc" ] && [ -f "$candidate/include/cublas_v2.h" ]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

cuda_yes_flag() {
  if [ "$AGENT_CONFIRM_CUDA_BUILD" = "1" ] || [ "$AGENT_CONFIRM_CUDA_BUILD" = "true" ] || [ "$AGENT_CONFIRM_CUDA_BUILD" = "yes" ]; then
    printf '%s\n' "--yes"
    return 0
  fi
  if [ "$AGENT_CONFIRM_CUDA_BUILD" = "0" ] || [ "$AGENT_CONFIRM_CUDA_BUILD" = "false" ] || [ "$AGENT_CONFIRM_CUDA_BUILD" = "no" ]; then
    return 0
  fi
  if [ -t 0 ]; then
    return 0
  fi
  printf '%s\n' "--yes"
}

"$AGENT_PY" -m pip install --upgrade pip setuptools wheel cmake ninja
install_agent_requirements

if cuda_requested; then
  echo "CUDA-capable GPU detected or requested. Launching dedicated CUDA llama-cpp-python builder."
  CUDA_CONFIRM_FLAG="$(cuda_yes_flag)"
  if [ -n "$CUDA_CONFIRM_FLAG" ]; then
    bash "$ROOT_DIR/build_llama_cpp_cuda.sh" --env-dir "$AGENT_ENV_DIR" --allow-cpu-fallback "$AGENT_ALLOW_CPU_FALLBACK" $CUDA_CONFIRM_FLAG --log-dir "$SETUP_LOG_DIR"
  else
    bash "$ROOT_DIR/build_llama_cpp_cuda.sh" --env-dir "$AGENT_ENV_DIR" --allow-cpu-fallback "$AGENT_ALLOW_CPU_FALLBACK" --log-dir "$SETUP_LOG_DIR"
  fi
else
  echo "CUDA not requested or not detected. Installing CPU llama-cpp-python build."
  "$AGENT_PIP" install --upgrade llama-cpp-python
fi

if [ -f "$ROOT_DIR/requirements-user.txt" ]; then
  "$USER_PY" -m pip install --upgrade pip setuptools wheel
  if grep -Eq '^[[:space:]]*[^#[:space:]]' "$ROOT_DIR/requirements-user.txt"; then
    "$USER_PIP" install -r "$ROOT_DIR/requirements-user.txt"
  else
    echo "requirements-user.txt is empty; skipping user project dependency install."
  fi
fi

CUDA_ROOT_FOR_ENV="$(detect_cuda_root || true)"

cat > "$ROOT_DIR/agent_env.sh" <<ENVEOF
#!/usr/bin/env bash
export AGENT_ENV_DIR="$AGENT_ENV_DIR"
export USER_ENV_DIR="$USER_ENV_DIR"
export CONDA_EXE="\${CONDA_EXE:-$CONDA_EXE_PATH}"
export DEBUG_AGENT_MODEL_PATH="\${DEBUG_AGENT_MODEL_PATH:-$MODEL_PATH_DEFAULT}"
export CUDA_HOME="\${CUDA_HOME:-$CUDA_ROOT_FOR_ENV}"
export CUDAToolkit_ROOT="\${CUDAToolkit_ROOT:-\${CUDA_HOME:-}}"
export LLAMA_CUDA_ARCH="\${LLAMA_CUDA_ARCH:-${LLAMA_CUDA_ARCH}}"
if [ -n "\${CUDA_HOME:-}" ]; then
  export CUDACXX="\${CUDACXX:-\$CUDA_HOME/bin/nvcc}"
  export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$CUDA_HOME/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}"
fi
alias activate_agent_env='source "$AGENT_ENV_DIR/bin/activate"'
activate_user_env() {
  local conda_exe="\${CONDA_EXE:-$CONDA_EXE_PATH}"
  local conda_base="\$(dirname "\$(dirname "\$conda_exe")")"
  if [ -f "\$conda_base/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "\$conda_base/etc/profile.d/conda.sh"
    conda activate "\$USER_ENV_DIR"
  else
    export PATH="\$USER_ENV_DIR/bin:\$PATH"
    export CONDA_PREFIX="\$USER_ENV_DIR"
  fi
}
echo "Agent env ready: $AGENT_ENV_DIR"
echo "User env available: $USER_ENV_DIR"
echo "Model path: \$DEBUG_AGENT_MODEL_PATH"
if [ -n "\${CUDA_HOME:-}" ]; then
  echo "CUDA root: \$CUDA_HOME"
fi
echo "Use activate_agent_env or activate_user_env as needed."
ENVEOF
chmod +x "$ROOT_DIR/agent_env.sh"

cat > "$ROOT_DIR/run_agent.sh" <<ENVEOF
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$ROOT_DIR"
AGENT_ENV_DIR="$AGENT_ENV_DIR"
export USER_ENV_DIR="$USER_ENV_DIR"
export CONDA_EXE="\${CONDA_EXE:-$CONDA_EXE_PATH}"
export DEBUG_AGENT_MODEL_PATH="\${DEBUG_AGENT_MODEL_PATH:-$MODEL_PATH_DEFAULT}"
export CUDA_HOME="\${CUDA_HOME:-$CUDA_ROOT_FOR_ENV}"
export CUDAToolkit_ROOT="\${CUDAToolkit_ROOT:-\${CUDA_HOME:-}}"
export LLAMA_CUDA_ARCH="\${LLAMA_CUDA_ARCH:-${LLAMA_CUDA_ARCH}}"
if [ -n "\${CUDA_HOME:-}" ]; then
  export CUDACXX="\${CUDACXX:-\$CUDA_HOME/bin/nvcc}"
  export LD_LIBRARY_PATH="\$CUDA_HOME/lib64:\$CUDA_HOME/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}"
fi
"$AGENT_ENV_DIR/bin/python" "$ROOT_DIR/preflight_agent.py" --root-dir "$ROOT_DIR"
exec "$AGENT_ENV_DIR/bin/python" "$ROOT_DIR/system_debug_cli.py" "$@"
ENVEOF
chmod +x "$ROOT_DIR/run_agent.sh"

cat > "$ROOT_DIR/run_user_code.sh" <<ENVEOF
#!/usr/bin/env bash
set -euo pipefail
USER_ENV_DIR="$USER_ENV_DIR"
if [ "\$#" -lt 1 ]; then
  echo "Usage: $ROOT_DIR/run_user_code.sh <script.py> [args ...]" >&2
  exit 1
fi
exec "$USER_ENV_DIR/bin/python" "$@"
ENVEOF
chmod +x "$ROOT_DIR/run_user_code.sh"

echo
echo "Setup complete."
echo "Activate helpers with:"
echo "  source \"$ROOT_DIR/agent_env.sh\""
echo "Run the agent with:"
echo "  \"$ROOT_DIR/run_agent.sh\""
echo "Run user code with:"
echo "  \"$ROOT_DIR/run_user_code.sh\" your_script.py"
echo "Standalone CUDA builder:"
echo "  \"$ROOT_DIR/build_llama_cpp_cuda.sh\" --env-dir \"$AGENT_ENV_DIR\""
echo "CUDA build log:"
echo "  \"$SETUP_LOG_DIR/llama_cpp_cuda_build.log\""
echo "CUDA probe log:"
echo "  \"$SETUP_LOG_DIR/llama_cpp_cuda_cmake_probe.log\""
echo "llama.cpp repo build log:"
echo "  \"$SETUP_LOG_DIR/llama_cpp_repo_build.log\""
