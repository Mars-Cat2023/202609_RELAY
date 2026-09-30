# RELAY Table 1: first 2000 test puzzles

Run from the project root in the `relay-sudoku` environment:

```bash
python sudoku/experiments/table1_first2000/run.py \
  --output logs/table1_first2000_300k_ema_NEW_RUN --device cuda:0
```

## Protocol

- Load the original cached Hugging Face test split and select `range(2000)` directly, in dataset order. No shuffle, deduction-only filtering, or dataloader worker sharding.
- All four objectives, tied/untied embeddings, seeds 1/2/3: 24 checkpoints.
- Use each run's `40-300000.ckpt`, loading EMA parameters into the correct saved model architecture. No checkpoint modification, training, or best-checkpoint selection.
- Use the original `ConfidenceBasedPredictor`: threshold 0.15, top_prob confidence, top_k=1, top_p=None, max_steps=64 (plus the original unconditional final forward). Temperature 1, no added hidden noise. Hidden relay enabled only for RELAY/RELAY-sg.
- Model eval mode, BF16 autocast, batch size 512, full vocabulary. One deterministic prediction per puzzle. Score exact match of all 81 token IDs.
- Summary reports mean and sample standard deviation (ddof=1) over training seeds; population SD is also saved in `table1.json`. Paper values are copied from the user's table; no assumption about the paper's SD convention.
- RELAY tied seed1 reuses the prior baseline only after checking checkpoint/source hashes, data identities, all eight deterministic repeats and an additional original-predictor first-batch comparison. Other 23 runs are evaluated anew.

## Outputs

`manifest.json`: protocol/data/source hashes; `puzzles.jsonl`: ordered puzzle IDs/questions/answers; one predictions JSONL per checkpoint; `per_seed.json/csv`: scores and checkpoint hashes; `table1.json/md`: all-seed summary.

The script refuses to overwrite an existing run manifest. All existing experiment files are preserved. NFE counts actual forwards on whole batches, including finished rows and the final forward.

## Interpretation of the 2026-09-30 run

The final checkpoints for Rollout tied seed1 and RELAY tied seed2 score 0/2000. Both raw and EMA tensors in those checkpoints are finite. The main table includes these runs; they must not silently be removed or replaced by earlier `best.ckpt` files.

The original user-provided reproduction table reported n=2 for these two cells. An explicitly labeled supplementary summary of the two nonzero seeds is provided for context, but the original table alone does not establish which seeds it used. Dataset-size changes and changes in the included seeds must be distinguished.

These first 2000 examples have already been used in prior temperature/noise experiments. This evaluation follows the requested Table 1 subset; it is not a newly untouched holdout for future post-training model selection.
