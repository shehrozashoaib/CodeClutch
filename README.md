# CodeClutch v5

CodeClutch is a terminal-first autonomous Python debugging agent that behaves more like an engineer than a blind fixer.

It runs your code, inspects the traceback, separates user-code bugs from environment issues, asks for approval before risky actions, and keeps the agent runtime separate from the user runtime.

## Quick Start

1. Set up the environments.

```bash
bash /workspace/setup_agent_env.sh
```

2. Point the agent at a GGUF model.

```bash
export DEBUG_AGENT_MODEL_PATH="/workspace/Qwen_2.5_Coder/model-q4_k_m.gguf"
```

3. Start CodeClutch.

```bash
/workspace/run_agent.sh
```

## Runtime Layout

CodeClutch uses two separate environments.

- Agent runtime: `/workspace/.venvs/agent`
- User runtime: `/workspace/.venvs/user`

This matters in practice:

- the agent itself runs in the agent env
- your target script runs in the user env
- `python`, `pip`, and `conda install` actions proposed by the agent are routed to the user env

## Launch Flow

When you start `/workspace/run_agent.sh`, CodeClutch will:

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

A normal run looks like this:

```text
/workspace/run_agent.sh

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

Use this to inspect:

- stdout only
- stderr only
- a combined tail
- the saved execution log
- or disable review prompts for the rest of the session

## Approval Flow

CodeClutch is intentionally human-in-the-loop.

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

That library-patch option is narrow on purpose. By default, CodeClutch avoids editing `site-packages`.

## What CodeClutch Tries To Do

The agent is designed to choose different workflows for different failure classes.

- User-code bug: read your file, understand the local context, propose a focused fix
- Dependency or compatibility issue: inspect versions in the user env, then repair the environment instead of editing library source
- Parser or config issue: inspect the actual malformed artifact, not parser internals
- CUDA or resource issue: treat it as runtime/config pressure first, not as a random code-edit problem

It also tries to avoid several bad patterns:

- repeating the same successful `pip show` / `conda list` loop forever
- defaulting to `pip install <import-name>` when the import/provider mapping is unclear
- editing library files to patch around dependency drift
- proposing impossible writes with bogus line numbers

## GPU Behavior During Runs

If the model is GPU-loaded, CodeClutch releases the agent model before the user script runs, then restores it afterward. This is important for VRAM-heavy jobs.

That means:

- the agent does not intentionally sit on GPU memory while your script is running
- training or inference jobs get the VRAM back during the user run
- if GPU reload fails, CodeClutch can recover instead of crashing the whole session

## Mocking Workflow

When you use `minimal inputs + mocking`, CodeClutch can identify expensive functions and let you keep only the ones you want to mock.

Typical flow:

```text
MOCK SELECTION
1. train.py::train_dgl
2. train_alignn.py::setup
3. dataset.py::get_torch_dataset
...
```

After selection, it prepares a smaller run and can reuse the expensive function output to keep later iterations fast.

## Direct Commands

Run user code directly in the user runtime:

```bash
/workspace/run_user_code.sh your_script.py [args ...]
```

Load helper shell functions:

```bash
source /workspace/agent_env.sh
activate_agent_env
activate_user_env
```

## Model Setup

CodeClutch reads the model path from `DEBUG_AGENT_MODEL_PATH`.

Example:

```bash
export DEBUG_AGENT_MODEL_PATH="/workspace/Qwen_2.5_Coder/model-q4_k_m.gguf"
```

If you want that permanently:

```bash
echo 'export DEBUG_AGENT_MODEL_PATH="/workspace/Qwen_2.5_Coder/model-q4_k_m.gguf"' >> ~/.bashrc
source ~/.bashrc
```

## Recommended Workflow

1. Start with `/workspace/run_agent.sh`.
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

## Notes

- The main launcher is [`run_agent.sh`](/workspace/run_agent.sh).
- The main CLI is [`system_debug_cli.py`](/workspace/system_debug_cli.py).
- The remembered startup defaults are stored in [`/workspace/.debug_cli_defaults.json`](/workspace/.debug_cli_defaults.json).
- Setup logs live under [`/workspace/.setup_logs`](/workspace/.setup_logs).

## Release Snapshot

This README describes the current v5 operator workflow from the terminal side.
