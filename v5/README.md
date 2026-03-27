# Autonomous Debug Agent

## Overview

This repository contains CodeClutch, a local autonomous Python debugging agent built around `llama-cpp-python`.

It is designed to behave more like a careful software engineer than a blind fixer:

- run the target script
- inspect the traceback and nearby code
- ask for approval before important actions
- distinguish user-code bugs from environment and dependency problems
- keep agent runtime dependencies separate from user project dependencies
- use RAG/search as support, not as a substitute for local reasoning

This file documents the `v5` release.

## Clone And Run v5

Clone the repo:

```bash
git clone git@github.com:shehrozashoaib/CodeClutch.git
cd CodeClutch/v5
```

Set up the environments:

```bash
bash setup_agent_env.sh
```

Point the agent at a GGUF model:

```bash
export DEBUG_AGENT_MODEL_PATH="/path/to/your/model.gguf"
```

Start the agent:

```bash
./run_agent.sh
```

## What We Are Building

The goal is not just a script runner. The goal is a reusable autonomous debugger that can:

- debug arbitrary Python projects
- reason about code changes versus environment fixes
- avoid blind edits to third-party libraries
- install or repair dependencies in the correct runtime
- support CUDA-backed local LLM inference through `llama.cpp`

The system files in `v5` are the debugger infrastructure:

- [`v5/system_debug_cli.py`](./system_debug_cli.py): terminal launcher and startup flow
- [`v5/system_llm_agent.py`](./system_llm_agent.py): main debugging loop and reasoning logic
- [`v5/system_tools.py`](./system_tools.py): script execution, terminal commands, file reads/writes, output capture

## Runtime Model

There are two separate runtimes.

- Agent runtime: `v5/.venvs/agent`
  - runs the debugger itself
  - contains `llama-cpp-python` and the agent-side dependencies
- User runtime: `v5/.venvs/user`
  - runs the user's project code
  - is the environment the agent installs project dependencies into
  - stays separate from the agent runtime

This separation is important. User project installs should not pollute the agent runtime, and agent-side model/runtime dependencies should not control the user project.

## Setup

Run:

```bash
bash setup_agent_env.sh
```

This script:

- prepares the agent environment at `v5/.venvs/agent`
- prepares the user runtime at `v5/.venvs/user`
- installs agent dependencies
- builds CUDA-enabled `llama.cpp` / `llama-cpp-python` when available
- generates helper scripts:
  - [`v5/agent_env.sh`](./agent_env.sh)
  - [`v5/run_agent.sh`](./run_agent.sh)
  - [`v5/run_user_code.sh`](./run_user_code.sh)

## CUDA / llama.cpp

This workstation needed a specific workaround.

What we found:

- complete usable toolkit: `/usr/local/cuda-12.6`
- `/usr/local/cuda-13.1` existed but was incomplete for builds
- native GPU architecture detection selected `compute_120a`
- CUDA 12.6 `nvcc` could not compile that architecture

So the working build path uses an explicit architecture override:

```bash
AGENT_ENABLE_CUDA=1 AGENT_CONFIRM_CUDA_BUILD=yes AGENT_ALLOW_CPU_FALLBACK=0 CUDA_HOME=/usr/local/cuda-12.6 CUDAToolkit_ROOT=/usr/local/cuda-12.6 LLAMA_CUDA_ARCH=90 bash setup_agent_env.sh
```

Why `LLAMA_CUDA_ARCH=90` matters here:

- it avoids the unsupported native autodetect path on this machine
- it is the current stable workaround for building the CUDA version successfully

To rerun just the CUDA builder:

```bash
CUDA_HOME=/usr/local/cuda-12.6 CUDAToolkit_ROOT=/usr/local/cuda-12.6 LLAMA_CUDA_ARCH=90 AGENT_ALLOW_CPU_FALLBACK=0 ./build_llama_cpp_cuda.sh --env-dir .venvs/agent --yes
```

Useful logs:

- `v5/.setup_logs/llama_cpp_cuda_build.log`
- `v5/.setup_logs/llama_cpp_cuda_cmake_probe.log`
- `v5/.setup_logs/llama_cpp_repo_build.log`

## Model Path

The agent reads the model path from `DEBUG_AGENT_MODEL_PATH`.

To set it permanently in your shell:

```bash
echo 'export DEBUG_AGENT_MODEL_PATH="/path/to/your/model.gguf"' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
echo "$DEBUG_AGENT_MODEL_PATH"
```

Using the environment variable is preferred over editing [`v5/run_agent.sh`](./run_agent.sh), because setup can regenerate the launcher script.

## GGUF Model Sources

If you need GGUF model files for local `llama.cpp` / `llama-cpp-python` runs, these Hugging Face repos are the current workspace references:

- Qwen 2.5 Coder GGUF: <https://huggingface.co/shehrozashoaib/Qwen_2.5_Coder_GGUF>
- IQuest V1 40B GGUF 4KM: <https://huggingface.co/shehrozashoaib/IQuest_V1_40B_GGUF_4KM>

These are model download sources only. After downloading a `.gguf` file, point `DEBUG_AGENT_MODEL_PATH` at the file you want the agent to use.

## Starting The Agent

Run:

```bash
./run_agent.sh
```

The launcher performs preflight checks, then starts the interactive debugger.

The CLI remembers the last startup values for:

- target file
- initial script arguments

Those values are reused as defaults on the next run.

## Running User Code Directly

To run project code in the user runtime directly:

```bash
./run_user_code.sh your_script.py [args ...]
```

## Helper Activations

You can load helper functions with:

```bash
source ./agent_env.sh
```

Then use:

```bash
activate_agent_env
activate_user_env
```

## How The Debugger Behaves

The agent is designed to imitate an actual engineer rather than blindly editing code.

Current behavior:

- it runs the target script and records stdout/stderr/exit code
- it offers output review menus so you can inspect failures before continuing
- it offers approval prompts before major actions such as terminal commands or writes
- it keeps a history of repeated failures and tries to detect cyclic behavior
- it uses RAG/search as supporting evidence, not as a direct replacement for local inspection
- it supports a minimal-input plus mocking workflow for expensive pipelines
- it releases the GPU-loaded model before user runs and restores it afterward

The important design rule is:

- user-code failures should lead to reading and editing user code
- environment/dependency failures should lead to inspecting and repairing the environment
- third-party library source should generally not be edited

## Environment And Dependency Handling

The debugger treats repeated third-party import/API failures as environment problems.

Examples of this class:

- `ModuleNotFoundError` inside `site-packages`
- `ImportError: cannot import name ...` coming from third-party packages
- package API/version drift across installed libraries

For those failures, the agent should:

1. stop reading `site-packages` files as if they were user code
2. inspect installed package versions in the user environment
3. propose a targeted upgrade/downgrade/pin command
4. rerun the script

This is intended to be generic behavior, not a hardcoded fix for one library.

## RAG Behavior

The RAG pipeline is meant to support debugging, not dominate it.

For import-related and compatibility-related failures, it can use:

- the import line and import path when available
- package-index style search queries
- package candidate extraction from search results
- runtime context for CUDA-sensitive dependencies like `torch`
- local environment facts collected during the run

The desired behavior is:

- avoid defaulting to `pip install <import-name>` when that is likely wrong
- prefer evidence-backed package candidates
- for runtime-sensitive libraries, choose a build that matches the local machine context

## Human-In-The-Loop Controls

The debugger is intentionally interactive.

You can:

- inspect stdout/stderr before the agent continues
- approve or reject proposed actions
- provide your own guidance when rejecting a step
- force more investigation instead of accepting a weak proposal
- allow a one-shot library patch only when you explicitly want that local hotfix

## Typical v5 Workflow

1. Clone the repo and `cd CodeClutch/v5`.
2. Run `bash setup_agent_env.sh`.
3. Export `DEBUG_AGENT_MODEL_PATH`.
4. Start the agent with `./run_agent.sh`.
5. Choose `full run`, `minimal inputs + mocking`, or `ask inside agent`.
6. Review outputs and approve or redirect actions as needed.

## Notes

- Files beginning with `system_` are debugger infrastructure.
- `python-magic` is optional. The debugger falls back to simpler file-type heuristics if it is unavailable.
- Hidden setup logs live under `v5/.setup_logs`.
- The latest operator-facing release is `v5`.
