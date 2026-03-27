#!/usr/bin/env bash
export AGENT_ENV_DIR="/workspace/.venvs/agent"
export USER_ENV_DIR="/workspace/.venvs/user"
export CONDA_EXE="${CONDA_EXE:-/opt/miniforge3/bin/conda}"
export DEBUG_AGENT_MODEL_PATH="${DEBUG_AGENT_MODEL_PATH:-/workspace/llama.cpp/IQuest_Coder_V1_40B_Q4_K_M.gguf}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-${CUDA_HOME:-}}"
export LLAMA_CUDA_ARCH="${LLAMA_CUDA_ARCH:-90}"
if [ -n "${CUDA_HOME:-}" ]; then
  export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi
alias activate_agent_env='source "/workspace/.venvs/agent/bin/activate"'
activate_user_env() {
  local conda_exe="${CONDA_EXE:-/opt/miniforge3/bin/conda}"
  local conda_base="$(dirname "$(dirname "$conda_exe")")"
  if [ -f "$conda_base/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$USER_ENV_DIR"
  else
    export PATH="$USER_ENV_DIR/bin:$PATH"
    export CONDA_PREFIX="$USER_ENV_DIR"
  fi
}
echo "Agent env ready: /workspace/.venvs/agent"
echo "User env available: /workspace/.venvs/user"
echo "Model path: $DEBUG_AGENT_MODEL_PATH"
if [ -n "${CUDA_HOME:-}" ]; then
  echo "CUDA root: $CUDA_HOME"
fi
echo "Use activate_agent_env or activate_user_env as needed."
