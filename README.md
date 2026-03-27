# CodeClutch

CodeClutch is a terminal-first autonomous Python debugging agent.

This repository keeps major iterations in separate folders. The latest release is in [`v5`](./v5).

## Latest Release

- Current version: [`v5`](./v5)
- Main launcher: [`v5/run_agent.sh`](./v5/run_agent.sh)
- Setup script: [`v5/setup_agent_env.sh`](./v5/setup_agent_env.sh)
- Main CLI: [`v5/system_debug_cli.py`](./v5/system_debug_cli.py)

## What v5 Adds

- cleaner startup flow and CodeClutch operator console
- agent/user runtime separation
- approval-based action flow for writes and terminal commands
- better environment-versus-user-code classification
- stronger handling for dependency mismatches and repeated env loops
- minimal-input plus mocking workflow for expensive pipelines
- better GPU handoff so the agent model does not intentionally sit on VRAM during user runs

## Quick Start

1. Go to the latest version folder.

```bash
cd v5
```

2. Set up the environments.

```bash
bash setup_agent_env.sh
```

3. Point the agent at a GGUF model.

```bash
export DEBUG_AGENT_MODEL_PATH="/path/to/your/model.gguf"
```

4. Start CodeClutch.

```bash
./run_agent.sh
```

## CUDA Build Notes

On the original workstation for this release, CUDA needed an explicit architecture override.

Working setup command:

```bash
AGENT_ENABLE_CUDA=1 AGENT_CONFIRM_CUDA_BUILD=yes AGENT_ALLOW_CPU_FALLBACK=0 CUDA_HOME=/usr/local/cuda-12.6 CUDAToolkit_ROOT=/usr/local/cuda-12.6 LLAMA_CUDA_ARCH=90 bash setup_agent_env.sh
```

If you need to rebuild only the CUDA llama.cpp integration:

```bash
CUDA_HOME=/usr/local/cuda-12.6 CUDAToolkit_ROOT=/usr/local/cuda-12.6 LLAMA_CUDA_ARCH=90 AGENT_ALLOW_CPU_FALLBACK=0 ./build_llama_cpp_cuda.sh --env-dir .venvs/agent --yes
```

## Model Downloads

You can use your own GGUF model, or download from the referenced model repos mentioned in the release documentation inside [`v5/README.md`](./v5/README.md).

## Version Folders

- [`Version_1_Refactoring_and_Minimal_Inputs_Testing`](./Version_1_Refactoring_and_Minimal_Inputs_Testing)
- [`Version_2_Stable_General_Solution`](./Version_2_Stable_General_Solution)
- [`debugging_v3`](./debugging_v3)
- [`debugging_v4`](./debugging_v4)
- [`v5`](./v5)
