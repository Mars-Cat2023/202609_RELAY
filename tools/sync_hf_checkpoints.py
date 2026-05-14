"""Repackage local Fast-dLLM v2 RELAY checkpoints under the new
``use_relay`` / ``relay_layer_norm`` naming and (optionally) push them to
the Hugging Face Hub.

This script is **read-only** with respect to the source checkpoint -- it
copies everything under a new staging directory, rewrites
``config.json`` keys (``use_loopholing`` -> ``use_relay``,
``loophole_layer`` -> ``relay_layer``, drops deprecated CAB / MLP-carry
fields), renames the ``model.h_t_layer_norm.*`` tensors to
``model.relay_layer_norm.*`` inside the safetensors shard, and copies in
the public-release ``configuration.py`` / ``modeling.py`` from
``relay/fast-dllm-v2/v2/src/lmflow/models/fast_dllm/``.

It is the inverse of the backwards-compat translation in
``Fast_dLLM_QwenConfig.__init__`` -- new training runs save with the
canonical names, but the two reported checkpoints predate the rename, so
we run them through this once before uploading.

Usage::

    # 1. Stage both reported checkpoints into ./hf_staging/.
    #    Replace the --src paths with your local Trainer ``checkpoint-200``
    #    dirs from the OpenCode/OpenMath c40m60 RELAY and RELAY (sg) runs.
    python tools/sync_hf_checkpoints.py prepare \
        --src /path/to/<run-dir>/checkpoint-200 \
        --dst hf_staging/relay-fastdllm-v2-c40m60-relay-step200 \
        --variant relay --step 200

    python tools/sync_hf_checkpoints.py prepare \
        --src /path/to/<run-sg-dir>/checkpoint-200 \
        --dst hf_staging/relay-fastdllm-v2-c40m60-relay-sg-step200 \
        --variant relay-sg --step 200

    # 2. Smoke-test each staged dir (loads through the public modeling.py).
    python tools/sync_hf_checkpoints.py smoke \
        --staged hf_staging/relay-fastdllm-v2-c40m60-relay-step200

    # 3. Push to the Hub (requires ``huggingface-cli login`` first).
    python tools/sync_hf_checkpoints.py push \
        --staged hf_staging/relay-fastdllm-v2-c40m60-relay-step200 \
        --repo <your-hf-user>/relay-fastdllm-v2-c40m60-relay-step200

The on-disk training checkpoints under ``--src`` are NEVER modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_FAST_DLLM = (
    REPO_ROOT / "fast-dllm-v2" / "v2" / "src" / "lmflow" / "models" / "fast_dllm"
)


_DEPRECATED_CONFIG_KEYS = (
    "use_cab", "cab_bottleneck_dim", "cab_n_heads", "cab_n_kv_heads",
    "cab_mlp_expansion_dim", "read_layers", "only_mask_tokens",
    "use_mlp_carry", "loophole_position_guard",
)


def _rewrite_config(src_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate legacy keys to the public-release schema."""
    cfg = dict(src_cfg)
    use_relay = cfg.pop("use_loopholing", None)
    relay_layer = cfg.pop("loophole_layer", None)
    if use_relay is None:
        # Old vanilla SFT checkpoints were written without either key.
        use_relay = False
    cfg["use_relay"] = bool(use_relay)
    cfg["relay_layer"] = int(relay_layer) if relay_layer is not None else -1
    for k in _DEPRECATED_CONFIG_KEYS:
        cfg.pop(k, None)
    cfg["auto_map"] = {
        "AutoConfig": "configuration.Fast_dLLM_QwenConfig",
        "AutoModel": "modeling.Fast_dLLM_QwenModel",
        "AutoModelForCausalLM": "modeling.Fast_dLLM_QwenForCausalLM",
    }
    return cfg


def _rewrite_safetensors(src: Path, dst: Path) -> None:
    """Copy ``src/model.safetensors`` to ``dst`` with renamed weight keys."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    rename_map = {
        "model.h_t_layer_norm.weight": "model.relay_layer_norm.weight",
        "model.h_t_layer_norm.bias": "model.relay_layer_norm.bias",
    }
    new_state: Dict[str, "torch.Tensor"] = {}  # noqa: F821
    metadata: Dict[str, str] = {}
    with safe_open(str(src), framework="pt") as f:
        if f.metadata():
            metadata = dict(f.metadata())
        for k in f.keys():
            new_key = rename_map.get(k, k)
            new_state[new_key] = f.get_tensor(k)
    metadata.setdefault("format", "pt")
    save_file(new_state, str(dst), metadata=metadata)


def _copy_release_modeling(staging_dir: Path) -> None:
    """Drop ``configuration.py`` + ``modeling.py`` from the public release."""
    if not SRC_FAST_DLLM.exists():
        raise SystemExit(
            f"Could not find public-release modeling at {SRC_FAST_DLLM}. "
            "Run this script from the relay repo root."
        )
    for fname in ("configuration.py", "modeling.py"):
        src = SRC_FAST_DLLM / fname
        if not src.exists():
            raise SystemExit(f"Missing {src}")
        shutil.copy2(src, staging_dir / fname)


def _write_model_card(staging_dir: Path, *, variant: str, step: int) -> None:
    pretty_variant = {"relay": "RELAY", "relay-sg": "RELAY (sg)"}.get(
        variant, variant
    )
    # We deliberately do not embed the eventual Hub repo id in the card so
    # the same staging dir can be uploaded under any user account; the card
    # uses ``<this-repo>`` placeholders that the user can search-replace
    # post-upload.
    card = f"""---
license: apache-2.0
language: en
library_name: transformers
tags:
  - text-generation
  - diffusion
  - relay
  - fast-dllm-v2
base_model: Efficient-Large-Model/Fast_dLLM_v2_1.5B
---

# {pretty_variant} adaptation of Fast-dLLM v2 1.5B (c40m60, step {step})

Released alongside the paper *Learned Relay Representations for
Forward-Thinking Discrete Diffusion Models*. Reproduces the
**{pretty_variant}** row of Table 2.

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "<this-repo>"  # e.g. <your-hf-user>/relay-fastdllm-v2-c40m60-{variant}-step{step}
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
```

`config.json` ships with `use_relay=True` and `relay_layer=-1`; the
safetensors shard contains a `model.relay_layer_norm.{{weight,bias}}` tensor
that the bundled `modeling.py` instantiates and consumes inside the
2-step relay forward (paper Algorithm 1).

## Training

200 optimizer steps starting from
[`Efficient-Large-Model/Fast_dLLM_v2_1.5B`](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B)
on the 60k OpenCodeInstruct + OpenMathInstruct-2 c40m60 mixture
(24 000 code + 36 000 math rows). Effective batch size 32, learning rate
5e-6, BD block 32 / sub-block 8, threshold 0.85. The {pretty_variant}
variant uses
`bptt_use_relay=1, bptt_stop_grad_h_s={int(variant == 'relay-sg')}`.

Full training command (from `relay/fast-dllm-v2/v2/`):

```bash
USE_RELAY=1 BPTT_STOP_GRAD_H_S={int(variant == 'relay-sg')} \\
  sbatch train_scripts/finetune_opencode_openmath_bptt.sbatch
```

See the public release of the training code for the full pipeline.
"""
    (staging_dir / "README.md").write_text(card, encoding="utf-8")


def cmd_prepare(args: argparse.Namespace) -> None:
    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"Source checkpoint dir does not exist: {src}")
    if dst.exists():
        if not args.force:
            raise SystemExit(
                f"Destination {dst} already exists. Re-run with --force to overwrite."
            )
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    cfg_path = src / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        src_cfg = json.load(f)
    new_cfg = _rewrite_config(src_cfg)
    with (dst / "config.json").open("w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        f"[prepare] rewrote config.json: use_relay={new_cfg['use_relay']} "
        f"relay_layer={new_cfg['relay_layer']}"
    )

    src_st = src / "model.safetensors"
    if not src_st.exists():
        raise SystemExit(
            f"Missing {src_st}. Sharded checkpoints are not yet supported "
            "by this tool; run zero_to_fp32.py to merge first."
        )
    _rewrite_safetensors(src_st, dst / "model.safetensors")
    print(f"[prepare] rewrote model.safetensors with renamed relay tensors")

    # Copy tokenizer / generation config / chat template etc. as-is.
    _COPY_VERBATIM = (
        "added_tokens.json",
        "chat_template.jinja",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    )
    for name in _COPY_VERBATIM:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
    print(f"[prepare] copied tokenizer / chat-template files")

    _copy_release_modeling(dst)
    print(
        f"[prepare] dropped public-release configuration.py / modeling.py "
        f"(from {SRC_FAST_DLLM})"
    )

    _write_model_card(dst, variant=args.variant, step=args.step)
    print(f"[prepare] wrote README.md model card (variant={args.variant})")

    print(f"[prepare] staging dir ready: {dst}")
    print(
        "          next: `python tools/sync_hf_checkpoints.py smoke --staged "
        f"{dst}`"
    )


def cmd_smoke(args: argparse.Namespace) -> None:
    staged = Path(args.staged)
    if not staged.is_dir():
        raise SystemExit(f"Staged dir does not exist: {staged}")
    # Add the in-tree source so we don't depend on auto_map round-trips.
    src_root = REPO_ROOT / "fast-dllm-v2" / "v2" / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    import torch  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[smoke] loading {staged} via trust_remote_code=True ...")
    tok = AutoTokenizer.from_pretrained(str(staged), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(staged), trust_remote_code=True
    )
    cfg = model.config
    if not getattr(cfg, "use_relay", False):
        raise SystemExit(
            "[smoke] FAIL: loaded config has use_relay=False; expected True"
        )
    base = model.model
    if not hasattr(base, "relay_layer_norm"):
        raise SystemExit(
            "[smoke] FAIL: model is missing relay_layer_norm; weight rename "
            "must have failed"
        )
    w = base.relay_layer_norm.weight.detach().float().norm().item()
    b = base.relay_layer_norm.bias.detach().float().norm().item()
    print(
        f"[smoke] OK: use_relay=True, relay_layer={cfg.relay_layer}, "
        f"relay_layer_norm.weight L2={w:.4f}, bias L2={b:.4f}"
    )
    print(f"[smoke] tokenizer vocab size = {tok.vocab_size}")


def cmd_push(args: argparse.Namespace) -> None:
    staged = Path(args.staged)
    if not staged.is_dir():
        raise SystemExit(f"Staged dir does not exist: {staged}")
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub not installed; `pip install huggingface_hub`"
        ) from e

    print(f"[push] uploading {staged} -> {args.repo}")
    create_repo(args.repo, exist_ok=True, private=args.private)
    api = HfApi()
    api.upload_folder(
        folder_path=str(staged),
        repo_id=args.repo,
        repo_type="model",
        commit_message=args.commit_message
        or f"Resync to public-release naming (use_relay / relay_layer_norm)",
    )
    print(f"[push] OK: https://huggingface.co/{args.repo}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_prepare = sub.add_parser("prepare", help="Stage a checkpoint with renamed config + tensors.")
    sp_prepare.add_argument("--src", required=True, help="Source checkpoint dir (read-only).")
    sp_prepare.add_argument("--dst", required=True, help="Destination staging dir.")
    sp_prepare.add_argument("--variant", required=True, choices=("relay", "relay-sg"))
    sp_prepare.add_argument("--step", type=int, required=True, help="Checkpoint step (for the model card).")
    sp_prepare.add_argument("--force", action="store_true", help="Overwrite an existing --dst.")
    sp_prepare.set_defaults(func=cmd_prepare)

    sp_smoke = sub.add_parser("smoke", help="Round-trip load a staged dir to confirm key/tensor names.")
    sp_smoke.add_argument("--staged", required=True)
    sp_smoke.set_defaults(func=cmd_smoke)

    sp_push = sub.add_parser("push", help="Upload a staged dir to the HF Hub.")
    sp_push.add_argument("--staged", required=True)
    sp_push.add_argument("--repo", required=True, help="HF repo id, e.g. user/name.")
    sp_push.add_argument("--private", action="store_true")
    sp_push.add_argument("--commit_message", default=None)
    sp_push.set_defaults(func=cmd_push)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
