# RELAY: Learned Relay Representations for Forward-Thinking Discrete Diffusion Models

Repository for the paper "Learned Relay Representations for Forward-Thinking Discrete Diffusion Models."

When Masked Diffusion Models (MDMs) generate sequences through iterative refinement, the rich internal computation accumulated over masked positions is discarded at the end of each forward pass—forcing every subsequent denoising step to start from scratch. We call this the *hard reset* problem. To address it, we propose **RELAY**, a method that makes MDMs *forward-thinking*: at each denoising step the model carries its last-layer hidden states forward as a learned relay, giving the next forward pass direct access to prior continuous computation. The relay is trained end-to-end via truncated backpropagation through time (BPTT), shaping it to be maximally informative for the next several denoising steps. RELAY is architecture-agnostic, leaves the inference-time decoding procedure unchanged, and is compatible with block diffusion and KV caching.

We validate design choices on a Sudoku-based planning task, then scale RELAY to Fast-dLLM v2 1.5B—a state-of-the-art diffusion language model—outperforming standard supervised fine-tuning on coding tasks by up to 3.7% in accuracy and 32% in inference latency.

---

## Installation

Each subdirectory has its own environment. Both require Python 3.11.10.

### `sudoku/`

```bash
git clone --recurse-submodules <repo-url>
cd relay/sudoku
conda create -p .venv_bptt python=3.11.10 pip ipykernel -y
conda activate ./.venv_bptt
pip install -e xlm-core
pip install -e xlm-core/xlm-models
pip install -r requirements.txt
```

Copy `.env.example` (or create a `.env` file at the project root) with at minimum:

```bash
WANDB_ENTITY=<your-entity>
WANDB_PROJECT=BPTT
DATA_DIR=data
HF_HOME=hf_home
LOG_DIR=logs
PROJECT_ROOT=.
```

### `fast-dllm-v2/`

```bash
cd relay/fast-dllm-v2
conda create -p .venv_dllm python=3.11.10 pip ipykernel -y
conda activate ./.venv_dllm
conda install mpi4py -y
cd v2
pip install -e .           # core package
pip install -e '.[eval]'   # adds lm-eval for benchmark evaluation
```

---

## Repository structure

```
relay/
├── sudoku/          # ablation study on a Sudoku planning task
└── fast-dllm-v2/    # scaling RELAY to Fast-dLLM v2 1.5B on coding benchmarks
```

---

## `sudoku/` — design study

This subdirectory contains the code for the controlled experiments on Sudoku (and related combinatorial planning tasks such as n-queens and graph coloring) that motivate and justify the RELAY design choices.

**Framework.** Training and evaluation are built on top of `xlm-core`, a PyTorch Lightning–based framework for masked language models, accessed here as a git submodule. Experiments are configured with Hydra and submitted to SLURM via scripts in `slurm_scripts/`. The main package is `doublebackprop/`.

**Key source files.**

| File | Role |
|---|---|
| `doublebackprop/model.py` | Model definitions. `RotaryTransformerLoopholingModel` is the relay-augmented transformer: it injects `LayerNorm(h_t)` (the relay from the previous step) into the token embedding layer and reads the new relay `h_s` from the final encoder hidden state. Also defines `BottleneckMLPBridge` (a small bottleneck MLP that transforms the carry) and `CrossAttentionBridge` (CAB, a cross-attention module that attends from token embeddings to the carried state). |
| `doublebackprop/loss.py` | Loss functions. `LoopholingBPTTLoss` unrolls the model for T denoising steps and applies truncated BPTT through the relay. `LoopholingBPTTPumaLoss` adds a PUMA-style streaming buffer that maintains persistent relay states across training batches. |
| `doublebackprop/predictor.py` | Inference-time decoding: confidence-threshold unmasking with relay forwarding across steps. |
| `doublebackprop/datamodule.py` | Data loading and batching for Sudoku (extreme/hard/deduction difficulties), n-queens, and graph coloring. |
| `doublebackprop/streaming_batch.py` | `StreamingBatch`: per-slot persistent storage of relay states for PUMA-style training. |
| `doublebackprop/history.py` | Trajectory history tracking for multi-step BPTT unrolls. |
| `doublebackprop/metrics.py` | Sudoku solution validity and completion metrics. |

**Training variants** (configured via Hydra experiment files):

- `sudoku_extreme_mlm_uniform` — standard MLM baseline, no relay.
- `sudoku_extreme_loopholing_bptt` — RELAY with truncated BPTT; `stop_grad_h_s=false` enables full gradient flow through the relay, `stop_grad_h_s=true` is the no-BPTT ablation.
- `sudoku_extreme_loopholing_bptt_puma` — RELAY with BPTT + PUMA streaming buffer.

See `SUDOKU_COMMANDS.md` and `PUZZLE_TRAINING_COMMANDS.md` for complete training commands. Plotting notebooks live under `plotting_scripts/`, and decode trajectory visualization tools under `visualization/`.

---

## `fast-dllm-v2/` — scaling to a 1.5B diffusion language model

This subdirectory scales RELAY to Fast-dLLM v2 1.5B (a block-diffusion LLM based on Qwen2.5) and evaluates it on coding benchmarks (HumanEval+ and MBPP+).

**Framework.** Training is built on LMFlow (in `v2/src/lmflow/`) with a custom Fast-dLLM model class under `v2/src/lmflow/models/fast_dllm/`. Generation at inference time is handled by `v2/generation_functions.py`, which patches `QwenForCausalLM.batch_sample` to forward the relay alongside committed tokens across block-diffusion steps.

**Carry modes.** Three relay architectures are supported, selected via `CARRY_MODE`:

| `CARRY_MODE` | Mechanism |
|---|---|
| `loopguard` | Layer-norm applied to `h_t`; injected as a residual at the model input (lightest carry). |
| `mlp` | `BottleneckMLPBridge`: two-layer bottleneck MLP with LayerNorm, transforms `h_t` before injection. |
| `cab` | `CrossAttentionBridge`: cross-attention from token embeddings to the carried state, with a bottleneck projection. |

**Key source files.**

| File | Role |
|---|---|
| `v2/generation_functions.py` | Block-diffusion generation loop with optional relay forwarding (`use_carry=True`). Handles KV caching and sub-block parallelization. |
| `v2/train_scripts/finetune_magicoder_oss_bptt.sbatch` | SLURM script for BPTT + PUMA fine-tuning on Magicoder-OSS code data, parameterized by `CARRY_MODE`. |
| `v2/train_scripts/finetune_magicoder_oss.sbatch` | Vanilla SFT baseline (no relay). |
| `v2/scripts/generate_evalplus_jsonl.py` | Generates EvalPlus JSONL samples from a checkpoint for HumanEval+ / MBPP+ scoring. |
| `v2/eval.py` | lm-evaluation-harness integration for non-code tasks (GSM8K, MATH, MMLU, etc.). |
| `v2/scripts/submit_eval.py` | SLURM wrapper for eval jobs. |

**Typical training progression** (each stage fine-tunes from the same base Fast-dLLM v2 1.5B checkpoint):

1. Vanilla SFT on Magicoder-OSS (`CARRY_MODE` not set).
2. BPTT + PUMA, no relay (`CARRY_MODE=none`): introduces the BPTT training objective without relay injection; serves as a strong baseline.
3. BPTT + PUMA + Loopguard (`CARRY_MODE=loopguard`).
4. BPTT + PUMA + MLP carry (`CARRY_MODE=mlp`).
5. BPTT + PUMA + CAB (`CARRY_MODE=cab`).

See `v2/CODE_COMMANDS.md` for full training and evaluation commands.
