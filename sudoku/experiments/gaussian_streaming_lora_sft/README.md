# Gaussian One-step Streaming LoRA-SFT

This experiment differs from the validated one-step Streaming LoRA-SFT baseline
only in its hidden-state transition during training:

\[
h_{k+1}=\operatorname{sg}(\mu_k+\sigma_{\mathrm{train}}\epsilon_k),
\qquad \epsilon_k\sim\mathcal N(0,I).
\]

The default experiment uses \(\sigma_{\mathrm{train}}=1\). The sampled hidden
state is stored in the same `StreamingBatch` slot and becomes the hidden input
to the next optimizer update. The initial state remains \(h_0=0\).

Every nonterminal state receives one supervised CE update. Position selection
uses the confidence rule, selected tokens are teacher forced from the ground
truth, and the hidden state is detached at every optimizer boundary. A puzzle
is replaced only after its complete trajectory finishes.

All pretrained RELAY parameters remain frozen. Only attention and FFN LoRA
parameters are trained, with rank 32 and alpha 64.

## Recommended staged run

First train and evaluate one training seed:

```bash
SEEDS=1 GPU=1 bash sudoku/experiments/gaussian_streaming_lora_sft/scripts/run.sh train
TRAIN_SEEDS=1 GPU=1 bash sudoku/experiments/gaussian_streaming_lora_sft/scripts/run.sh evaluate
```

The one-seed evaluation command also writes a one-training-seed summary. Use it
for screening only; the final uncertainty estimate requires all three training seeds.

After the smoke experiment is satisfactory, complete all three training seeds
and run the final matched evaluation:

```bash
GPU=1 bash sudoku/experiments/gaussian_streaming_lora_sft/scripts/run.sh train
GPU=1 bash sudoku/experiments/gaussian_streaming_lora_sft/scripts/run.sh evaluate
```

Evaluation uses the first 2,000 test puzzles, evaluation seeds 1, 2, and 3,
\(G=8\), \(T_{\mathrm{conf}}=2\), \(\tau_{\mathrm{tok}}=0.3\), and
\(\sigma_{\mathrm{eval}}\in\{0,1\}\).


## Core Implementation Files

The Training-Time Gaussian-Noise Streaming LoRA-SFT implementation is organized across the following files:

- [`streaming_lora_sft/train.py`](../streaming_lora_sft/train.py): Implements full-trajectory streaming supervision, one-step cross-entropy, teacher forcing, and the detached Gaussian hidden transition
  \[
  h_{k+1}
  =
  \operatorname{sg}
  \left(\mu_k+\sigma_{\mathrm{train}}\epsilon_k\right),
  \qquad
  \epsilon_k\sim\mathcal N(0,I).
  \]
- [`streaming_lora_sft/evaluate.py`](../streaming_lora_sft/evaluate.py): Performs matched evaluation of the newly evaluated pretrained baseline and the Gaussian-trained adapter under \(\sigma_{\mathrm{eval}}\in\{0,1\}\).
- [`scripts/`](./scripts): Contains the training and evaluation launchers for one or three independent training seeds.
- [`summarize_3seeds.py`](./summarize_3seeds.py): Aggregates three evaluation seeds within each adapter and then reports mean and standard deviation across training seeds.
- [`lora.py`](../../relay/lora.py): Implements LoRA injection, pretrained-parameter freezing, validation, and adapter serialization.
- [`streaming_batch.py`](../../relay/streaming_batch.py): Preserves token and hidden states across optimizer updates until each trajectory is completed.
- [`gaussian_grpo.py`](../../relay/gaussian_grpo.py): Provides the shared confidence-based `select_positions()` implementation.
- [`gaussian_hidden_grpo/train.py`](../gaussian_hidden_grpo/train.py): Provides shared checkpoint-loading, dataset-collation, hashing, rollout, and evaluation utilities.