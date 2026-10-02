"""Minimal LoRA support for RELAY without modifying pretrained weights."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.enabled = True
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.lora_A.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_B.to(device=base.weight.device, dtype=base.weight.dtype)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if self.enabled:
            output = output + self.lora_B(self.lora_A(inputs)) * self.scaling
        return output


@dataclass(frozen=True)
class LoRAReport:
    target_names: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


def _resolve_parent(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    components = qualified_name.split(".")
    parent = model
    for component in components[:-1]:
        parent = parent[int(component)] if component.isdigit() else getattr(parent, component)
    return parent, components[-1]


def inject_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    target_suffixes: Sequence[str] = ("attn_qkv", "o_proj", "mlp.0", "mlp.2"),
) -> LoRAReport:
    """Freeze ``model`` and replace matching linear layers with LoRA wrappers."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and not isinstance(module, LoRALinear)
        and any(name.endswith(suffix) for suffix in target_suffixes)
    ]
    if not matches:
        raise ValueError(f"No linear modules matched LoRA targets {tuple(target_suffixes)}")
    for name, module in matches:
        parent, child_name = _resolve_parent(model, name)
        replacement = LoRALinear(module, rank=rank, alpha=alpha)
        if child_name.isdigit():
            parent[int(child_name)] = replacement  # type: ignore[index]
        else:
            setattr(parent, child_name, replacement)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return LoRAReport(tuple(name for name, _ in matches), trainable, total)


def iter_lora_modules(model: nn.Module) -> Iterator[LoRALinear]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module


@contextmanager
def adapters_disabled(model: nn.Module):
    modules = list(iter_lora_modules(model))
    previous = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous):
            module.enabled = enabled


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if ".lora_A." in name or ".lora_B." in name
    }


def load_lora_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    expected = set(lora_state_dict(model))
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(f"LoRA state mismatch; missing={missing}, unexpected={unexpected}")
    current = model.state_dict()
    with torch.no_grad():
        for name, tensor in state.items():
            current[name].copy_(tensor)


def assert_only_lora_trainable(model: nn.Module) -> None:
    invalid = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and ".lora_A." not in name
        and ".lora_B." not in name
    ]
    if invalid:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {invalid}")
