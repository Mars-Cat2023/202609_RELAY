"""Offline dump of per-step decoding trajectories (checkpoint inference).

Run from repo root with Hydra overrides, same style as ``xlm``::

    python -m relay.dump_decode_trajectories \\
      experiment=sudoku_extreme_mlm_uniform \\
      +generation.ckpt_path=/path/to.ckpt \\
      +dump.output_path=trajectories.jsonl \\
      +dump.max_examples=200 \\
      +dump.limit_val_batches=5 \\
      per_device_batch_size=1 \\
      global_batch_size=1

Requires ``xlm_models.json`` (or equivalent) so Hydra resolves ``relay`` configs.

See ``visualization/README.md`` at the repo root for a full workflow, shell script, and Jupyter viewer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import dotenv
import torch
import hydra
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from xlm.utils.model_loading import load_model_for_inference
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

# Register Hydra search paths / resolvers (same side effects as ``xlm`` main).
import xlm.commands.lightning_main  # noqa: F401, E402

import xlm  # noqa: E402

_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(
        (Path(xlm.__file__).resolve().parent / "configs" / "lightning_train").resolve()
    ),
    "config_name": "config.yaml",
}


def _find_val_prediction_dataloader_idx(datamodule: Any) -> int:
    names = datamodule.dataloader_names["val"]
    for i in sorted(names.keys()):
        n = names[i]
        if "prediction" in n.lower() or "infill" in n.lower():
            return int(i)
    raise ValueError(
        f"No val dataloader with 'prediction' or 'infill' in name: {names}"
    )


def _exact_match_batch(
    pred: torch.Tensor, target: torch.Tensor
) -> List[bool]:
    return (pred == target).all(dim=-1).cpu().tolist()


def run_dump_decode_trajectories(cfg: DictConfig) -> None:
    if "PROJECT_ROOT" not in os.environ:
        os.environ["PROJECT_ROOT"] = "."
    if cfg.get("seed") is not None:
        seed_everything(int(cfg.seed), workers=True)

    dump = OmegaConf.select(cfg, "dump", default=None) or OmegaConf.create({})
    output_path = OmegaConf.select(dump, "output_path", default="decode_trajectories.jsonl")
    max_examples = int(OmegaConf.select(dump, "max_examples", default=10_000))
    limit_val_batches = OmegaConf.select(dump, "limit_val_batches", default=None)
    if limit_val_batches is not None:
        limit_val_batches = int(limit_val_batches)

    ckpt = OmegaConf.select(cfg, "generation.ckpt_path", default=None)
    if not ckpt:
        raise ValueError(
            "Set +generation.ckpt_path=/path/to.ckpt (full Lightning checkpoint)."
        )

    global_components: Dict[str, Any] = hydra.utils.instantiate(cfg.global_components)
    OmegaConf.clear_resolver("global_components")
    OmegaConf.register_new_resolver(
        "global_components", lambda x: global_components[x]
    )

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    tokenizer = datamodule.tokenizer
    OmegaConf.clear_resolver("tokenizer")
    OmegaConf.register_new_resolver(
        "tokenizer", lambda x: getattr(tokenizer, x)
    )
    OmegaConf.clear_resolver("datamodule")
    OmegaConf.register_new_resolver(
        "datamodule", lambda x: getattr(datamodule, x)
    )

    datamodule.no_trainer_mode = True
    datamodule.prepare_data()
    datamodule.setup("fit")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module, _ = load_model_for_inference(
        cfg,
        datamodule,
        tokenizer,
        config_prefix="generation",
        manual_ema_restore=True,
        move_to_device=device,
        set_eval_mode=True,
        enable_hub_support=True,
        allow_random_init=False,
    )

    predictor = module.predictor
    predictor.capture_trajectory = True
    predictor.log_rollout_diagnostics = True

    dl_idx = _find_val_prediction_dataloader_idx(datamodule)
    val_loaders = datamodule.val_dataloader()
    if not isinstance(val_loaders, list):
        val_loaders = [val_loaders]
    dl = val_loaders[dl_idx]
    dl_name = datamodule.dataloader_names["val"][dl_idx]

    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    example_index = 0
    n_written = 0
    batch_count = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for batch_idx, batch in enumerate(dl):
            if limit_val_batches is not None and batch_idx >= limit_val_batches:
                break
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            preds = module.predictor.predict(
                batch, batch_idx=batch_idx, dataloader_idx=dl_idx, dataloader_name=dl_name
            )
            if "trajectory" not in preds:
                raise RuntimeError(
                    "Predictor did not return trajectory; ensure capture_trajectory is True."
                )

            ids = preds["ids"]
            traj = preds["trajectory"]
            target = batch.get("target_ids")
            if target is None:
                raise ValueError("Batch missing target_ids (needed for exact match).")

            bsz = ids.shape[0]
            ems = _exact_match_batch(ids, target)
            rps = preds.get("rollout_steps_per_sample")
            if rps is None:
                rps = [max(0, len(traj[b]) - 1) for b in range(bsz)]

            tags = OmegaConf.to_container(cfg.get("tags", {}), resolve=True)
            skip_spl = getattr(predictor, "skip_special_tokens", True)
            row_base: Dict[str, Any] = {
                "dataloader_name": dl_name,
                "checkpoint": str(ckpt),
                "tags": tags,
            }
            for b in range(bsz):
                if n_written >= max_examples:
                    break
                ti = target[b].detach().cpu().tolist()
                pi = ids[b].detach().cpu().tolist()
                row = {
                    **row_base,
                    "example_index": example_index,
                    "batch_idx": batch_idx,
                    "sample_in_batch": b,
                    "exact_match": ems[b],
                    "rollout_steps_per_sample": int(rps[b]),
                    "pred_ids": pi,
                    "target_ids": ti,
                    "pred_text": tokenizer.decode(pi, skip_special_tokens=skip_spl),
                    "truth_text": tokenizer.decode(ti, skip_special_tokens=skip_spl),
                    "trajectory": traj[b],
                }
                f.write(json.dumps(row) + "\n")
                n_written += 1
                example_index += 1

            batch_count += 1
            if n_written >= max_examples:
                break

    logger.info(
        f"Wrote {n_written} examples to {out_path} "
        f"({batch_count} val batches, dataloader={dl_name})."
    )


@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig) -> None:
    dotenv.load_dotenv(
        dotenv_path=".env",
        override=True,
    )
    dotenv.load_dotenv(".secrets.env", override=True)
    run_dump_decode_trajectories(cfg)


if __name__ == "__main__":
    main()
