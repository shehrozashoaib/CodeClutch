# Anchor

**Debug AI-generated Python inside your real environment, with a local LLM. No token burn, no blind edits, no code leaving your machine.**

<p align="center"><img src="docs/demo.gif" alt="Anchor demo" width="800"/></p>

---

## Why Anchor

AI assistants write Python fast, but they don't patiently debug it. When something breaks, Copilot-style tools miss the details of your actual environment — your CUDA version, your numpy pin, your half-installed package — and burn tokens guessing at fixes from the outside.

Anchor runs your code where it actually lives, reads the real traceback, and proposes targeted repairs. A self-hosted LLM does the heavy reasoning on your workstation, so there's no per-token bill and nothing leaves the machine. A paid API and web RAG are available as escalation paths, not the default.

## Design Principles

- **Local-first.** The reasoning model runs on your GPU via `llama.cpp`. Your code and stack traces stay on your machine.
- **Ask before risky edits.** Writes, installs, and shell commands require your approval. The agent is interactive by design.
- **Separate runtimes.** The agent lives in `.venvs/agent`; your project lives in `.venvs/user`. Debugger dependencies can never collide with your project's `numpy` or `torch`.
- **Fix the right layer.** User-code failures lead to user code. Environment/dependency failures lead to the environment. Third-party library source is not edited.
- **RAG as support, not substitute.** Web search and package-index lookups back up local reasoning — they don't replace it.

## Requirements

- Linux workstation (Ubuntu tested)
- `python3`, `git`, Conda or Miniforge
- **GPU: 12 GB VRAM recommended** for the default model; CPU fallback works but is slow
- For CUDA builds: NVIDIA drivers + a working `nvidia-smi` + a CUDA toolkit installed
- One GGUF model file (see below)

### Default model

Start with **Qwen 2.5 Coder (GGUF)** — small enough for a 12 GB card, strong on Python:

- <https://huggingface.co/shehrozashoaib/Qwen_2.5_Coder_GGUF>

For larger workstations, IQuest V1 40B (4 K M quant) gives better reasoning at the cost of VRAM:

- <https://huggingface.co/shehrozashoaib/IQuest_V1_40B_GGUF_4KM>

## Quickstart

```bash
git clone git@github.com:shehrozashoaib/Anchor.git
cd Anchor/v5
bash setup_agent_env.sh
export DEBUG_AGENT_MODEL_PATH="/path/to/your/model.gguf"
./run_agent.sh
```

`setup_agent_env.sh` must run before `run_agent.sh` — it builds the two runtimes and regenerates the launcher scripts with paths that match your local clone. The launchers checked into git are templates; do not run them directly.

Persist the model path across shells:

```bash
echo 'export DEBUG_AGENT_MODEL_PATH="/path/to/your/model.gguf"' >> ~/.bashrc
source ~/.bashrc
```

## How It Works

```
    ┌──────────┐      ┌────────────┐      ┌──────────────┐      ┌──────────┐
    │  Your    │  →   │ Anchor │  →   │ Review &     │  →   │ Fix      │
    │  script  │      │ runs it    │      │ approve      │      │ applied  │
    └──────────┘      └─────┬──────┘      └──────┬───────┘      └──────────┘
                            │                    │
                            ▼                    ▼
                      ┌────────────┐      ┌──────────────┐
                      │ Traceback  │      │ Escalate to  │
                      │ + env info │      │ paid LLM/RAG │
                      └────────────┘      │ if stuck     │
                                          └──────────────┘
```

1. **Run** — the agent executes your script in the user runtime and captures stdout, stderr, and exit code.
2. **Inspect** — on failure, it reads the traceback, looks at nearby code, and checks installed package versions.
3. **Diagnose** — it classifies the failure as a user-code bug or an environment/dependency problem.
4. **Ask** — it proposes an action (edit, install, rerun, shell command) and waits for your approval.
5. **Escalate** — if local reasoning is stuck, it can call out to RAG or a paid LLM, with your consent.

## Worked Example

A real case from converting an LFM2 checkpoint to GGUF. The user code references a hyperparameter key that doesn't exist in the model config:

```python
class LFM2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.LFM2

    def _add_feed_forward_length(self):
        ff_dim = self.find_hparam(["block_ff_dim", "intermediate_size"])
        auto_adjust_ff_dim = self.hparams["block_auto_adjust_ff_dim"]
        ffn_dim_multiplier = self.hparams["block_ffn_dim_multiplier"]
        multiple_of = self.hparams["ffn_multiple_of"]   # wrong key
```

A session looks like:

```
$ ./run_agent.sh
[target] convert_hf_to_gguf.py  (args: /models/lfm2-ckpt --outtype f16)

--- run 1 ---
KeyError: 'ffn_multiple_of'
  at convert_hf_to_gguf.py:<LINE>, in LFM2Model._add_feed_forward_length

[agent] This is a user-code bug, not an environment problem.
[agent] Inspecting self.hparams keys from the loaded checkpoint...
[agent] Keys present: block_ff_dim, block_auto_adjust_ff_dim,
        block_ffn_dim_multiplier, block_multiple_of, ...
[agent] Siblings follow the `block_*` naming convention. `ffn_multiple_of`
        looks like a typo; the actual key in this config is `block_multiple_of`.
[agent] Propose: edit LFM2Model._add_feed_forward_length to read
        `self.hparams["block_multiple_of"]`.
Approve edit? [y/n/guide] y

--- run 2 ---
[OK] wrote lfm2-ckpt.f16.gguf  (exit 0)
[agent] Done.
```

Notice what the agent did **not** do:

- it didn't `pip install` a mystery package to paper over a `KeyError`
- it didn't edit `gguf`, `transformers`, or any third-party library
- it read the actual `hparams` dict from the running process to confirm the real key before proposing a fix
- no paid tokens were consumed — the entire loop ran on the local GGUF model

## Status

Private beta with ~8 ML teams. **DM [@shehrozashoaib](https://github.com/shehrozashoaib) to try it.**

## Repository Layout

The latest operator-facing release is **`v5`**. Earlier iterations are kept for reference:

- [`v5`](./v5) — current
- [`debugging_v4`](./debugging_v4), [`debugging_v3`](./debugging_v3) — previous iterations
- [`Version_2_Stable_General_Solution`](./Version_2_Stable_General_Solution), [`Version_1_Refactoring_and_Minimal_Inputs_Testing`](./Version_1_Refactoring_and_Minimal_Inputs_Testing) — early prototypes

Core v5 files:

- [`v5/system_debug_cli.py`](./v5/system_debug_cli.py) — terminal launcher and startup flow
- [`v5/system_llm_agent.py`](./v5/system_llm_agent.py) — main debugging loop and reasoning
- [`v5/system_tools.py`](./v5/system_tools.py) — script execution, shell, file I/O, output capture

## Advanced Setup

### Runtimes

`setup_agent_env.sh` creates two isolated environments inside `v5/`:

| Path | Purpose |
| --- | --- |
| `.venvs/agent` | Debugger runtime — `llama-cpp-python`, agent-side deps |
| `.venvs/user`  | Your project's runtime — agent installs project deps here |

Load helpers into your shell if you want to poke around manually:

```bash
source ./agent_env.sh
activate_agent_env   # or activate_user_env
```

Run user code directly (without the agent):

```bash
./run_user_code.sh your_script.py [args ...]
```

### CUDA builds

Anchor builds `llama-cpp-python` against `llama.cpp` with CUDA support when it's available. Some workstations need an explicit GPU architecture override.

**Symptom.** `setup_agent_env.sh` fails in the CUDA build step with an error like `nvcc does not support compute_120a` (or similar), typically because auto-detection picked a target your installed `nvcc` can't compile.

**Fix.** Pin the CUDA toolkit and architecture explicitly:

```bash
AGENT_ENABLE_CUDA=1 \
AGENT_CONFIRM_CUDA_BUILD=yes \
AGENT_ALLOW_CPU_FALLBACK=0 \
CUDA_HOME=/usr/local/cuda-12.6 \
CUDAToolkit_ROOT=/usr/local/cuda-12.6 \
LLAMA_CUDA_ARCH=90 \
bash setup_agent_env.sh
```

Adjust `CUDA_HOME` to point at your complete toolkit (check `ls /usr/local/cuda-*`). `LLAMA_CUDA_ARCH=90` is the stable workaround for avoiding bad native autodetect.

To rerun only the CUDA builder:

```bash
CUDA_HOME=/usr/local/cuda-12.6 \
CUDAToolkit_ROOT=/usr/local/cuda-12.6 \
LLAMA_CUDA_ARCH=90 \
AGENT_ALLOW_CPU_FALLBACK=0 \
./build_llama_cpp_cuda.sh --env-dir .venvs/agent --yes
```

Build logs land under `v5/.setup_logs/`:

- `llama_cpp_cuda_build.log`
- `llama_cpp_cuda_cmake_probe.log`
- `llama_cpp_repo_build.log`

## Troubleshooting

**`./run_agent.sh` fails on a fresh clone.** The launcher scripts committed to git are templates. Run `bash setup_agent_env.sh` first — it rewrites them with the correct paths for your machine.

**Launcher behaves like it points at a different path.** Same fix: rerun `bash setup_agent_env.sh`.

**CUDA build errors.** See *Advanced Setup → CUDA builds* and check the three `.setup_logs/` files listed there.

**Model not loading.** Confirm `echo "$DEBUG_AGENT_MODEL_PATH"` points at an existing `.gguf` file and the file isn't an LFS pointer.

**`python-magic` warnings.** Optional. The agent falls back to simpler file-type heuristics.

## Human-in-the-Loop Controls

While the agent is running you can:

- inspect stdout/stderr before it continues
- approve, reject, or redirect each proposed action
- force further investigation instead of accepting a weak proposal
- allow a one-shot library patch if you explicitly want a local hotfix

The CLI remembers the last target file and script arguments as defaults for the next run.

## Roadmap

- Broader language support beyond Python
- IDE integration (VS Code extension)
- Richer repo-wide context (code graph)
- Optional cloud model fallback with strict data controls

## Contact

- GitHub: [@shehrozashoaib](https://github.com/shehrozashoaib)
- Issues / beta requests: open a GitHub issue on this repo

## License

MIT — see [`LICENSE`](./LICENSE).
