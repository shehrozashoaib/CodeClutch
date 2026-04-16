#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="/workspace"
AGENT_ENV_DIR="/workspace/.venvs/agent"
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
"/workspace/.venvs/agent/bin/python" "/workspace/preflight_agent.py" --root-dir "/workspace"
exec "/workspace/.venvs/agent/bin/python" "/workspace/system_debug_cli.py" "$@"
