# One-step Streaming LoRA-SFT

This experiment isolates full-trajectory supervised post-training from GRPO.

For every occupied streaming slot at state \((x_k,h_k)\), one optimizer update performs exactly one RELAY forward:

\[
(\Lambda_k,\mu_k)=f_\theta(x_k,h_k),\qquad
\mathcal L_k=-\frac{1}{|M_k|}\sum_{i\in M_k}\log p_\theta(y_i\mid x_k,h_k).
\]

Here, \(M_k\) contains all remaining mutable masked positions. Position selection uses the RELAY cumulative-confidence rule. The selected positions are teacher forced and the hidden state is detached:

\[
x_{k+1,i}=y_i\ (i\in U_k),\qquad h_{k+1}=\operatorname{sg}(\mu_k).
\]

The resulting state is returned to the same `StreamingBatch` slot. A slot receives a new puzzle only after its current puzzle has no remaining mutable masks. Thus every nonterminal state on every completed trajectory receives CE supervision, while gradients span one transition only.

Only LoRA matrices on attention and FFN projections are trainable. This baseline is locked to rank 32 and alpha 64; all pretrained parameters remain frozen. The training manifest records and checks a hash of the frozen parameters.

## Train

```bash
bash sudoku/experiments/streaming_lora_sft/scripts/train.sh
```

The default target is 5,000 completed trajectories. Progress is measured by `completed_trajectories` in `metrics.jsonl`, rather than optimizer updates.

## Matched evaluation

```bash
bash sudoku/experiments/streaming_lora_sft/scripts/evaluate.sh
```

The evaluator compares the pretrained checkpoint and the trained adapter on the same first 2,000 test puzzles with seeds 1, 2, and 3. It runs hidden-state \(\sigma=0\) and \(\sigma=1\), using \(T_{conf}=2\), full-vocabulary token temperature \(\tau_{tok}=0.3\), and group size 8.

At \(\sigma=0\), the relayed hidden state is deterministic. Token decoding remains categorical because \(\tau_{tok}=0.3\); the whole rollout is therefore not fully deterministic.
