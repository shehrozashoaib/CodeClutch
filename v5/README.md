# CodeClutch v5

CodeClutch v5 is a terminal-first autonomous Python debugging agent that behaves more like an engineer than a blind fixer.

It runs your code, inspects the traceback, separates user-code bugs from environment issues, asks for approval before risky actions, and keeps the agent runtime separate from the user runtime.

## Quick Start

1. Set up the environments.

```bash
bash setup_agent_env.sh
```

2. Point the agent at a GGUF model.

```bash
export DEBUG_AGENT_MODEL_PATH="/path/to/your/model.gguf"
```

3. Start CodeClutch.

```bash
./run_agent.sh
```

## Runtime Layout

CodeClutch uses two separate environments.

- Agent runtime: `.venvs/agent`
- User runtime: `.venvs/user`

This matters in practice:

- the agent itself runs in the agent env
- your target script runs in the user env
- `python`, `pip`, and `conda install` actions proposed by the agent are routed to the user env

## Launch Flow

When you start `./run_agent.sh`, CodeClutch will:

- show the `CodeClutch` startup screen
- run quiet preflight checks
- ask for the target file, script args, model path, context length, GPU layers, threads, and batch size
- ask for the debugging strategy
- start the agent loop

The CLI remembers your last target file and raw script arguments and reuses them as defaults.

## Strategies

You can choose one of three startup modes.

- `full run`: run the script as-is and debug the real failure directly
- `minimal inputs + mocking`: shrink expensive runs first and mock selected expensive functions when useful
- `ask inside agent`: let the agent decide after it starts

Use `minimal inputs + mocking` for expensive training or data pipelines. Use `full run` when you want the most direct reproduction of the real failure.

## Typical Session

```text
./run_agent.sh

Target file [train_alignn.py]:
Initial script arguments [--root_dir MP_json ...]:
Max agent steps / trials [15]:
Strategy [1=full run, 2=minimal inputs + mocking, 3=ask inside agent] [3]:
...

Step 1/15  phase=START | mocking=setup
  Next action    RunScriptInput
```

Then CodeClutch will iterate through:

- `RunScriptInput`
- output review
- file reads
- proposed writes
- targeted terminal commands
- optional RAG investigation

## Output Review

After a script run or terminal command, CodeClutch can show a review menu.

```text
Output options [Enter continue, 1 stdout, 2 stderr, 3 combined tail, 4 saved log, 5 disable prompts]
```

## Approval Flow

Before mutating actions, it will ask for approval.

```text
Options:
  [y] approve and execute
  [n] reject and provide guidance
  [p] reject, add suggested path to user-protected list, and re-plan
  [r] reject and force RAG / external investigation
```

For one-shot third-party hotfixes, a library write may also offer:

```text
  [l] allow this exact library patch once
```

## CUDA / llama.cpp Setup

On the original workstation for this release, CUDA needed an explicit architecture override.

Working setup command:

```bash
AGENT_ENABLE_CUDA=1 AGENT_CONFIRM_CUDA_BUILD=yes AGENT_ALLOW_CPU_FALLBACK=0 CUDA_HOME=/usr/local/cuda-12.6 CUDAToolkit_ROOT=/usr/local/cuda-12.6 LLAMA_CUDA_ARCH=90 bash setup_agent_env.sh
```

To rerun just the CUDA builder:

```bash
CUDA_HOME=/usr/local/cuda-12.6 CUDAToolkit_ROOT=/usr/local/cuda-12.6 LLAMA_CUDA_ARCH=90 AGENT_ALLOW_CPU_FALLBACK=0 ./build_llama_cpp_cuda.sh --env-dir .venvs/agent --yes
```

## Model Setup

CodeClutch reads the model path from `DEBUG_AGENT_MODEL_PATH`.

Example:

```bash
export DEBUG_AGENT_MODEL_PATH="/path/to/your/model.gguf"
```

If you want that permanently:

```bash
echo 'export DEBUG_AGENT_MODEL_PATH="/path/to/your/model.gguf"' >> ~/.bashrc
source ~/.bashrc
```

## Direct Commands

Run user code directly in the user runtime:

```bash
./run_user_code.sh your_script.py [args ...]
```

Load helper shell functions:

```bash
source ./agent_env.sh
activate_agent_env
activate_user_env
```

## Recommended Workflow

1. Start with `./run_agent.sh`.
2. Use `minimal inputs + mocking` for expensive ML/debug sessions.
3. Read the failure output before approving the next step.
4. Approve only the action you actually want.
5. Reject and guide the agent when it chooses the wrong path.
6. Use RAG when the problem is version-sensitive, obscure, or cyclic.

## Success State

A clean run ends with a success summary rather than a neutral stop.

```text
Session Summary
CodeClutch completed successfully.
Console clear. Target run finished without runtime errors.
```
