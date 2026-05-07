"""Experiment grid rows for manage_evals (lm_eval / submit_eval).

Each entry in RUN_SPECS is a template dict. Values wrapped in P(...) form one axis of a
Cartesian product. Reuse the same P instance for multiple keys so they stay in lockstep
(e.g. model_path and evalplus_tokenizer). Plain dicts (no P) expand to a single row.

Zipping / paired variation without a full product is not supported: use separate RUN_SPECS
entries instead.

Imported by manage_evals.py as RUNS (fully expanded).
"""

from __future__ import annotations

import itertools
from typing import Any

# --- shared paths (see cluster output_models layout) ---------------------------------
_WORK = "/work/pi_mccallum_umass_edu/brozonoyer_umass_edu/Fast-dLLM/v2/output_models"
_HUB = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
_NUMINA_CKPTS = (250, 500, 750, 1000, 1248)
_TULU_CKPTS = (500, 1000, 1500, 2000)
_MAGICODER_CKPTS = (250, 500, 750, 1000, 1175)
_VANILLA_CKPTS = (750, 1000, 1175)
_OC_OM_CKPTS = (200, 400, 600, 800)
_OC_OM_C40M60_CKPTS = (200, 400)
_OC_OM_C40M60_DECODEALIGNED8_CKPTS = (200, 400)
_OC_OM_C40M60_LAYERS_CKPTS = (200, 400)
class P:
    """One Cartesian factor: each expanded row picks one of ``values``."""

    __slots__ = ("values",)

    def __init__(self, *values: Any) -> None:
        if len(values) == 1:
            v0 = values[0]
            if isinstance(v0, (str, bytes)):
                self.values = (v0,)
            elif isinstance(v0, (list, tuple)):
                self.values = tuple(v0)
            else:
                self.values = tuple(v0)
        else:
            self.values = tuple(values)
        if not self.values:
            raise ValueError("P(...) requires at least one value")
    
    def __add__(self, other: P) -> P:
        return P(*(self.values + other.values))


CODE_EVAL_TASKS = P("humaneval_plus_evalplus", "mbpp_plus_evalplus")
MATH_EVAL_TASKS = P("math500", "gsm8k")
THRESHOLDS = P(0.85, 0.95, 1.0)
SMALL_BLOCK_SIZES = P(8, 4) #P(8)

def _expand_one(spec: dict[str, Any]) -> list[dict[str, Any]]:
    unique_ps: list[P] = []
    seen_ids: set[int] = set()
    for k in sorted(spec.keys()):
        v = spec[k]
        if isinstance(v, P):
            pid = id(v)
            if pid not in seen_ids:
                seen_ids.add(pid)
                unique_ps.append(v)
    if not unique_ps:
        for v in spec.values():
            if isinstance(v, P):
                raise RuntimeError("unreachable")
        return [dict(spec)]

    factors = [p.values for p in unique_ps]
    rows: list[dict[str, Any]] = []
    for combo in itertools.product(*factors):
        assign = {id(p): c for p, c in zip(unique_ps, combo)}
        row: dict[str, Any] = {}
        for k, v in spec.items():
            if isinstance(v, P):
                row[k] = assign[id(v)]
            else:
                row[k] = v
        rows.append(row)
    return rows


def expand_run_specs(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for spec in specs:
        out.extend(_expand_one(spec))
    return out


def _numina_paths(subdir: str) -> P:
    base = f"{_WORK}/numina_1p5B/{subdir}"
    return P(*(f"{base}/checkpoint-{s}" for s in _NUMINA_CKPTS))

def _oc_om_paths(subdir: str) -> P:
    base = f"{_WORK}/opencode_openmath_60k_1p5B/{subdir}"
    return P(*(f"{base}/checkpoint-{s}" for s in _OC_OM_CKPTS))

def _oc_om_40_60_paths(subdir: str) -> P:
    base = f"{_WORK}/opencode_openmath_60k_c40m60_1p5B/{subdir}"
    return P(*(f"{base}/checkpoint-{s}" for s in _OC_OM_C40M60_CKPTS))

def _oc_om_40_60_decodealigned8_paths(subdir: str) -> P:
    base = f"{_WORK}/opencode_openmath_60k_c40m60_decodealigned8_1p5B/{subdir}"
    return P(*(f"{base}/checkpoint-{s}" for s in _OC_OM_C40M60_CKPTS))

def _oc_om_40_60_layers_paths(subdir: str) -> P:
    base = f"{_WORK}/opencode_openmath_60k_c40m60_loopguard_layers_1p5B/{subdir}"
    return P(*(f"{base}/checkpoint-{s}" for s in _OC_OM_C40M60_LAYERS_CKPTS))

def _magicoder_paths(subdir: str) -> P:
    base = f"{_WORK}/magicoder_oss_full_1p5B/{subdir}"
    return P(*(f"{base}/checkpoint-{s}" for s in _MAGICODER_CKPTS))

def _tulu_paths(subdir: str) -> P:
    base = f"{_WORK}/tulu3_sft_1p5B/{subdir}"
    return P(*(f"{base}/checkpoint-{s}" for s in _TULU_CKPTS))

def _vanilla_paths(subdir: str) -> P:
    base = f"{_WORK}/magicoder_oss_full_1p5B/{subdir}"
    return P(*(f"{base}/checkpoint-{s}" for s in _VANILLA_CKPTS))


# Disambiguates default run_tag when checkpoint parent dirname matches another training run (submit_eval --training_data).


# fmt: off
RUN_SPECS: list[dict[str, Any]] = [
    {
        "experiment": "numina_1p5B_bptt_loopguard_puma_bs2x16_ep2",
        "task": MATH_EVAL_TASKS,
        "model_path": _numina_paths("bptt_loopguard_puma_bs2x16_ep2"),
        "use_carry": True,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "small_block_size": 32,
    },
    {
        "experiment": "numina_1p5B_bptt_mlp_puma_bs2x16_ep2",
        "task": MATH_EVAL_TASKS,
        "model_path": _numina_paths("bptt_mlp_puma_bs2x16_ep2"),
        "use_carry": True,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
    },
    {
        "experiment": "numina_1p5B_bptt_nocarry_puma_bs2x16_ep2",
        "task": MATH_EVAL_TASKS,
        "model_path": _numina_paths("bptt_nocarry_puma_bs2x16_ep2"),
        "use_carry": False,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
    },
    {
        "experiment": "tulu3_sft_1p5B_bptt_loopguard_puma_nopack1024_bs2x16x8_ep1_lr1e-5",
        "task": MATH_EVAL_TASKS,
        "model_path": _tulu_paths("bptt_loopguard_puma_nopack1024_bs2x16x8_ep1_lr1e-5"),
        "use_carry": True,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "small_block_size": SMALL_BLOCK_SIZES,
    },
    {"experiment": "math500_baselines_apr27", "task": MATH_EVAL_TASKS, "model_path": _HUB, "use_carry": False, "threshold": 0.85},
    {
        "experiment": "opencode_openmath_60k_1p5B_bptt_loopguard_puma_nopack2048_bs2x16x2_ep3_lr5e-6",
        "task": MATH_EVAL_TASKS + CODE_EVAL_TASKS,
        "model_path": _oc_om_paths("bptt_loopguard_puma_nopack2048_bs2x16x2_ep3_lr5e-6") + _oc_om_paths("vanilla_nopack2048_bs2x16x2_ep3_lr5e-6") + _oc_om_paths("bptt_loopguard_stopgrad_puma_nopack2048_bs2x16x2_ep3_lr5e-6") + _oc_om_paths("bptt_nocarry_puma_nopack2048_bs2x16x2_ep3_lr5e-6"),
        "use_carry": True,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "small_block_size": SMALL_BLOCK_SIZES,
        "evalplus_tokenizer": _HUB,
    },
    {
        "experiment": "opencode_openmath_60k_c40m60_1p5B_bptt_loopguard_puma_nopack2048_bs2x16x2_ep3_lr5e-6",
        "task": MATH_EVAL_TASKS + CODE_EVAL_TASKS,
        "model_path": (_oc_om_40_60_paths("bptt_loopguard_puma_nopack2048_bs2x16x2_ep3_lr5e-6") + _oc_om_40_60_paths("bptt_loopguard_stopgrad_puma_nopack2048_bs2x16x2_ep3_lr5e-6") + _oc_om_40_60_decodealigned8_paths("bptt_loopguard_decodealigned8_puma_nopack2048_bs2x16x2_ep3_lr5e-6") + _oc_om_40_60_decodealigned8_paths("bptt_nocarry_decodealigned8_puma_nopack2048_bs2x16x2_ep3_lr5e-6")),
        "use_carry": True,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "training_data": "oc_om_60k_c40m60",
        "evalplus_tokenizer": _HUB,
    },
    {
        "experiment": "opencode_openmath_60k_c40m60_loopguard_layers_1p5B",
        "task": MATH_EVAL_TASKS + CODE_EVAL_TASKS,
        "model_path": (_oc_om_40_60_layers_paths("bptt_loopguard_layer13_puma_nopack2048_bs2x16x2_ep3_lr5e-6") + _oc_om_40_60_layers_paths("bptt_loopguard_layer17_puma_nopack2048_bs2x16x2_ep3_lr5e-6") + _oc_om_40_60_layers_paths("bptt_loopguard_layer21_puma_nopack2048_bs2x16x2_ep3_lr5e-6")+ _oc_om_40_60_layers_paths("bptt_loopguard_layer25_puma_nopack2048_bs2x16x2_ep3_lr5e-6")),
        "use_carry": True,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "training_data": "oc_om_60k_c40m60",
        "evalplus_tokenizer": _HUB,
    },
    {
        "experiment": "opencode_openmath_60k_c40m60_ck200_nfe_t085",
        "task": MATH_EVAL_TASKS + CODE_EVAL_TASKS,
        "model_path": P(
            f"{_WORK}/opencode_openmath_60k_c40m60_1p5B/bptt_loopguard_puma_nopack2048_bs2x16x2_ep3_lr5e-6/checkpoint-200",
            f"{_WORK}/opencode_openmath_60k_c40m60_1p5B/bptt_loopguard_stopgrad_puma_nopack2048_bs2x16x2_ep3_lr5e-6/checkpoint-200",
        ),
        "use_carry": True,
        "threshold": 0.85,
        "batch_size": 32,
        "num_fewshot": 0,
        "training_data": "oc_om_60k_c40m60_ck200_nfe",
        "evalplus_tokenizer": _HUB,
        "output_root": "eval_results_nfe",
    },
    {
        "experiment": "opencode_openmath_60k_c40m60_ck200_nfe_t085",
        "task": MATH_EVAL_TASKS + CODE_EVAL_TASKS,
        "model_path": P(
            f"{_WORK}/opencode_openmath_60k_c40m60_1p5B/vanilla_nopack2048_bs2x16x2_ep3_lr5e-6/checkpoint-200",
            f"{_WORK}/hf_baseline_Fast_dLLM_v2_1.5B",
        ),
        "use_carry": False,
        "threshold": 0.85,
        "batch_size": 32,
        "num_fewshot": 0,
        "training_data": "oc_om_60k_c40m60_ck200_nfe",
        "evalplus_tokenizer": _HUB,
        "output_root": "eval_results_nfe",
    },
    {
        "experiment": "opencode_openmath_60k_c40m60_1p5B_bptt_cab_mlp",
        "task": MATH_EVAL_TASKS + CODE_EVAL_TASKS,
        "model_path": (_oc_om_40_60_paths("bptt_mlp_puma_nopack2048_bs2x16x2_ep3_lr5e-6") + _oc_om_40_60_paths("bptt_cab_puma_nopack2048_bs2x16x2_ep3_lr5e-6") ),
        "use_carry": True,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "training_data": "oc_om_60k_c40m60",
        "evalplus_tokenizer": _HUB,
    },
    # one bespoke run where we switch off carry only during inference
    {
        "experiment": "opencode_openmath_60k_c40m60_1p5B_bptt_loopguard_puma_nopack2048_bs2x16x2_ep3_lr5e-6_nocarry",
        "task": P("math500"), #MATH_EVAL_TASKS,
        "model_path": (_oc_om_40_60_paths("bptt_loopguard_puma_nopack2048_bs2x16x2_ep3_lr5e-6") + _oc_om_40_60_paths("bptt_loopguard_stopgrad_puma_nopack2048_bs2x16x2_ep3_lr5e-6") ),
        "use_carry": False,
        "threshold": 0.85, #THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "training_data": "oc_om_60k_c40m60",
        "evalplus_tokenizer": _HUB,
    },
    {
        "experiment": "opencode_openmath_60k_c40m60_1p5B_bptt_loopguard_puma_nopack2048_bs2x16x2_ep3_lr5e-6",
        "task": MATH_EVAL_TASKS + CODE_EVAL_TASKS,
        "model_path": (_oc_om_40_60_paths("bptt_nocarry_puma_nopack2048_bs2x16x2_ep3_lr5e-6") + _oc_om_40_60_paths("vanilla_nopack2048_bs2x16x2_ep3_lr5e-6") ),
        "use_carry": False,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "training_data": "oc_om_60k_c40m60",
        "evalplus_tokenizer": _HUB,
    },
    # Same as numina grid above, small_block_size=32 (run tag gains sb32 via submit_eval)
    {
        "experiment": "numina_1p5B_bptt_loopguard_puma_bs2x16_ep2",
        "task": MATH_EVAL_TASKS,
        "model_path": _numina_paths("bptt_loopguard_puma_bs2x16_ep2"),
        "use_carry": True,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "small_block_size": SMALL_BLOCK_SIZES,
    },
    {
        "experiment": "numina_1p5B_bptt_mlp_puma_bs2x16_ep2",
        "task": MATH_EVAL_TASKS,
        "model_path": _numina_paths("bptt_mlp_puma_bs2x16_ep2"),
        "use_carry": True,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "small_block_size": SMALL_BLOCK_SIZES,
    },
    {
        "experiment": "numina_1p5B_bptt_nocarry_puma_bs2x16_ep2",
        "task": MATH_EVAL_TASKS,
        "model_path": _numina_paths("bptt_nocarry_puma_bs2x16_ep2"),
        "use_carry": False,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "small_block_size": SMALL_BLOCK_SIZES,
    },
    {"experiment": "math500_baselines_apr27", "task": MATH_EVAL_TASKS, "model_path": _HUB, "use_carry": False, "threshold": 0.85, "small_block_size": 32},
    # humaneval_plus_evalplus / mbpp_plus_evalplus (EvalPlus-aligned lm-eval tasks).
    {
        "experiment": "evalplus_baselines_apr27",
        "task": CODE_EVAL_TASKS,
        "model_path": _HUB,
        "use_carry": False,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "evalplus_tokenizer": _HUB,
    },
]

# Per-run magicoder / vanilla: product(task × checkpoint); shared P for path + tokenizer.
for _exp, _subdir, _carry in (
    ("magicoder_oss_full_1p5B_bptt_loopguard_puma_nopack1024_bs2x16_ep1_lr1e-5", "bptt_loopguard_puma_nopack1024_bs2x16_ep1_lr1e-5", True),
    ("magicoder_oss_full_1p5B_bptt_mlp_puma_nopack1024_bs2x16_ep1_lr1e-5", "bptt_mlp_puma_nopack1024_bs2x16_ep1_lr1e-5", True),
    ("magicoder_oss_full_1p5B_bptt_nocarry_puma_nopack1024_bs2x16_ep1_lr1e-5", "bptt_nocarry_puma_nopack1024_bs2x16_ep1_lr1e-5", False),
    ("tulu3_sft_1p5B_bptt_loopguard_puma_nopack1024_bs2x16x8_ep1_lr1e-5", "bptt_loopguard_puma_nopack1024_bs2x16x8_ep1_lr1e-5", True),
):
    _ck = _magicoder_paths(_subdir) if "magicoder" in _exp else _tulu_paths(_subdir)
    RUN_SPECS.append({
        "experiment": _exp,
        "task": CODE_EVAL_TASKS,
        "model_path": _ck,
        "use_carry": _carry,
        "threshold": THRESHOLDS,
        "batch_size": 32,
        "num_fewshot": 0,
        "evalplus_tokenizer": _ck,
        "small_block_size": SMALL_BLOCK_SIZES,
    })

_vpaths = _vanilla_paths("vanilla_nopack1024_bs2x16_ep1_lr1e-5")
RUN_SPECS.append({
    "experiment": "magicoder_oss_full_1p5B_vanilla_nopack1024_bs2x16_ep1_lr1e-5",
    "task": CODE_EVAL_TASKS,
    "model_path": _vpaths,
    "use_carry": False,
    "threshold": THRESHOLDS,
    "batch_size": 32,
    "num_fewshot": 0,
    "evalplus_tokenizer": _vpaths,
    "small_block_size": SMALL_BLOCK_SIZES,
})
# fmt: on

RUNS: list[dict[str, Any]] = expand_run_specs(RUN_SPECS)

__all__ = ["P", "RUNS", "RUN_SPECS", "expand_run_specs"]
