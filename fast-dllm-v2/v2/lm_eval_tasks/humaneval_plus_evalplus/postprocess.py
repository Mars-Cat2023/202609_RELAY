# SPDX-License-Identifier: Apache-2.0
"""Loads shared postprocessor for YAML ``!function postprocess.build_predictions``."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SPEC = spec_from_file_location(
    "_ep_post_humaneval",
    Path(__file__).resolve().parents[1] / "_evalplus_common" / "postprocess.py",
)
assert _SPEC and _SPEC.loader
_mod = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)

build_predictions = _mod.build_predictions

__all__ = ["build_predictions"]
