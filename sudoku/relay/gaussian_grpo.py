"""Gaussian-hidden GRPO primitives for RELAY.

The implementation follows the agreed EGSPO-style surrogate: position sets
sampled by the old rollout are held fixed while token and hidden likelihoods
are recomputed under the current policy.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import ContextManager, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .lora import adapters_disabled


@dataclass
class RolloutStep:
    x: torch.Tensor
    h: torch.Tensor
    selected: torch.Tensor
    sampled_tokens: torch.Tensor
    old_token_log_probs: torch.Tensor
    old_selected_log_prob: torch.Tensor
    old_mean: torch.Tensor
    sampled_hidden: torch.Tensor
    valid: torch.Tensor
    hidden_valid: torch.Tensor
    force_all_positions: bool

    def pin_memory(self) -> "RolloutStep":
        if not torch.cuda.is_available():
            return self
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.pin_memory())
        return self


@dataclass
class RolloutBatch:
    steps: List[RolloutStep]
    rewards: torch.Tensor
    advantages: torch.Tensor
    final_ids: torch.Tensor
    group_size: int
    prompts: int


def confidence(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("confidence temperature must be nonnegative")
    if temperature == 0:
        ties = (logits == logits.amax(dim=-1, keepdim=True)).sum(dim=-1)
        return ties.to(logits.dtype).reciprocal()
    return (logits / temperature).softmax(dim=-1).amax(dim=-1)


@torch.no_grad()
def select_positions(
    logits: torch.Tensor,
    masked_mutable: torch.Tensor,
    *,
    temperature: float,
    threshold: float,
    active: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RELAY cumulative-uncertainty selection with the top-position fallback."""
    scores = confidence(logits, temperature).clone()
    scores[~masked_mutable] = float("-inf")
    uncertainty, indices = torch.sort(1 - scores, dim=-1)
    selected = (torch.cumsum(uncertainty, dim=-1) < threshold).scatter(
        -1, indices, torch.cumsum(uncertainty, dim=-1) < threshold
    )
    needs_fallback = ~selected.any(-1) & masked_mutable.any(-1)
    fallback = torch.zeros_like(selected).scatter_(1, indices[:, :1], True)
    selected |= fallback & needs_fallback[:, None]
    selected &= masked_mutable
    if active is not None:
        selected &= active[:, None]
    return selected


def _autocast(device: torch.device, dtype: Optional[torch.dtype]) -> ContextManager:
    if device.type == "cuda" and dtype is not None:
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    # These tensors outlive the rollout CUDA tensors. A synchronous device-to-host
    # copy prevents the evaluator/updater from reading an unfinished transfer.
    return tensor.detach().to(device="cpu", non_blocking=False).contiguous()


@torch.inference_mode()
def collect_rollouts(
    model: nn.Module,
    *,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    fixed: torch.Tensor,
    mask_token_id: int,
    group_size: int,
    sigma: float,
    confidence_temperature: float,
    token_temperature: float,
    threshold: float,
    max_steps: int,
    generator: Optional[torch.Generator] = None,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
    pin_memory: bool = True,
) -> RolloutBatch:
    if sigma < 0:
        raise ValueError("hidden-noise sigma must be nonnegative")
    if token_temperature <= 0:
        raise ValueError("token temperature must be positive")
    if group_size < 2:
        raise ValueError("GRPO group_size must be at least 2")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    model.eval()
    device = next(model.parameters()).device
    prompts = prompt_ids.shape[0]
    expand = lambda value: value.repeat_interleave(group_size, dim=0).to(device)
    x = expand(prompt_ids)
    targets = expand(target_ids)
    fixed_expanded = expand(fixed)
    h = torch.zeros(x.shape[0], x.shape[1], model.d_model, device=device)
    active = torch.ones(x.shape[0], dtype=torch.bool, device=device)
    steps: List[RolloutStep] = []

    for step_index in range(max_steps):
        state_x = x.clone()
        state_h = h.clone()
        masked = (state_x == mask_token_id) & ~fixed_expanded
        with _autocast(device, autocast_dtype):
            logits, old_mean_raw = model(state_x, state_h)
        old_mean = old_mean_raw.float()
        log_probs = F.log_softmax(logits.float() / token_temperature, dim=-1)
        if step_index == max_steps - 1:
            selected = masked & active[:, None]
        else:
            selected = select_positions(
                logits.float(), masked, temperature=confidence_temperature,
                threshold=threshold, active=active,
            )

        probabilities = log_probs.exp().reshape(-1, log_probs.shape[-1])
        sampled_tokens = torch.multinomial(
            probabilities, 1, replacement=True, generator=generator
        ).reshape(x.shape)
        selected_token_log_prob = (
            log_probs.gather(-1, sampled_tokens[..., None]).squeeze(-1) * selected
        ).sum(-1)
        x = torch.where(selected, sampled_tokens, state_x)
        terminal = ~(((x == mask_token_id) & ~fixed_expanded).any(-1))
        # The last forced-fill step has no successor, even if MASK was sampled.
        hidden_valid = active & ~terminal & (step_index < max_steps - 1)
        noise = torch.randn(
            old_mean.shape, device=device, dtype=torch.float32, generator=generator
        )
        sampled_hidden = old_mean + sigma * noise
        h = torch.where(hidden_valid[:, None, None], sampled_hidden, state_h)

        record = RolloutStep(
            x=_to_cpu(state_x),
            h=_to_cpu(state_h),
            selected=_to_cpu(selected),
            sampled_tokens=_to_cpu(sampled_tokens),
            old_token_log_probs=_to_cpu(log_probs),
            old_selected_log_prob=_to_cpu(selected_token_log_prob),
            old_mean=_to_cpu(old_mean),
            sampled_hidden=_to_cpu(sampled_hidden),
            valid=_to_cpu(active),
            hidden_valid=_to_cpu(hidden_valid),
            force_all_positions=step_index == max_steps - 1,
        )
        steps.append(record.pin_memory() if pin_memory else record)
        active &= ~terminal
        if not active.any():
            break

    rewards = ((x == targets).all(-1) & ~(((x == mask_token_id) & ~fixed_expanded).any(-1))).float()
    grouped_rewards = rewards.view(prompts, group_size)
    advantages = grouped_rewards - grouped_rewards.mean(dim=1, keepdim=True)
    return RolloutBatch(
        steps=steps,
        rewards=_to_cpu(grouped_rewards),
        advantages=_to_cpu(advantages.reshape(-1)),
        final_ids=_to_cpu(x),
        group_size=group_size,
        prompts=prompts,
    )


def _move(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device=device, non_blocking=True)


def diagonal_gaussian_log_ratio(
    action: torch.Tensor,
    current_mean: torch.Tensor,
    old_mean: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """Log q_current(action|s) - log q_old(action|s), summed per sample."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    delta = current_mean.float() - old_mean.float()
    centered_action = action.float() - old_mean.float()
    return (
        centered_action * delta / (sigma * sigma)
        - delta.square() / (2 * sigma * sigma)
    ).flatten(1).sum(-1)


def equal_variance_gaussian_kl(
    first_mean: torch.Tensor,
    second_mean: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """KL(N(first_mean,sigma^2 I) || N(second_mean,sigma^2 I))."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return (
        (first_mean.float() - second_mean.float()).square().flatten(1).sum(-1)
        / (2 * sigma * sigma)
    )


def grpo_update_epoch(
    model: nn.Module,
    rollout: RolloutBatch,
    optimizer: torch.optim.Optimizer,
    *,
    sigma: float,
    token_temperature: float,
    confidence_temperature: float,
    threshold: float,
    clip_epsilon: float,
    kl_beta: float,
    max_grad_norm: float,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> Dict[str, float]:
    """Run one PPO epoch over a fixed old-policy rollout and update LoRA."""
    if sigma <= 0 or token_temperature <= 0:
        raise ValueError("sigma and token_temperature must be positive")
    model.eval()
    device = next(model.parameters()).device
    advantages = _move(rollout.advantages, device).float()
    denominator = float(rollout.prompts * rollout.group_size)
    optimizer.zero_grad(set_to_none=True)
    sums: Dict[str, float] = {
        "policy_loss": 0.0, "reference_kl": 0.0,
        "token_kl_old": 0.0, "hidden_kl_old": 0.0, "joint_kl_old": 0.0,
        "token_reference_kl": 0.0, "hidden_reference_kl": 0.0,
        "token_log_ratio": 0.0, "hidden_log_ratio": 0.0,
        "joint_log_ratio": 0.0, "joint_clip_count": 0.0,
        "token_clip_count": 0.0, "hidden_clip_count": 0.0,
        "position_exact_count": 0.0, "position_jaccard": 0.0,
        "valid_count": 0.0,
    }

    for old in rollout.steps:
        x = _move(old.x, device)
        h = _move(old.h, device).float()
        selected = _move(old.selected, device)
        sampled_tokens = _move(old.sampled_tokens, device)
        valid = _move(old.valid, device)
        hidden_valid = _move(old.hidden_valid, device)
        old_log_probs = _move(old.old_token_log_probs, device).float()
        old_selected_log_prob = _move(old.old_selected_log_prob, device).float()
        old_mean = _move(old.old_mean, device).float()
        sampled_hidden = _move(old.sampled_hidden, device).float()

        with _autocast(device, autocast_dtype):
            current_logits, current_mean_raw = model(x, h)
        current_mean = current_mean_raw.float()
        current_log_probs = F.log_softmax(
            current_logits.float() / token_temperature, dim=-1
        )
        current_selected_log_prob = (
            current_log_probs.gather(-1, sampled_tokens[..., None]).squeeze(-1)
            * selected
        ).sum(-1)
        token_log_ratio = current_selected_log_prob - old_selected_log_prob
        hidden_log_ratio = diagonal_gaussian_log_ratio(
            sampled_hidden, current_mean, old_mean, sigma
        )
        hidden_log_ratio = torch.where(hidden_valid, hidden_log_ratio, 0.0)
        joint_log_ratio = token_log_ratio + hidden_log_ratio
        ratio = torch.exp(joint_log_ratio.clamp(-20, 20))
        clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
        surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)

        old_probs = old_log_probs.exp()
        token_kl_old = (
            (old_probs * (old_log_probs - current_log_probs)).sum(-1) * selected
        ).sum(-1)
        hidden_kl_old = equal_variance_gaussian_kl(old_mean, current_mean, sigma)
        hidden_kl_old = torch.where(hidden_valid, hidden_kl_old, 0.0)

        with torch.no_grad(), adapters_disabled(model), _autocast(device, autocast_dtype):
            reference_logits, reference_mean_raw = model(x, h)
        reference_log_probs = F.log_softmax(
            reference_logits.float() / token_temperature, dim=-1
        )
        current_probs = current_log_probs.exp()
        token_reference_kl = (
            (current_probs * (current_log_probs - reference_log_probs)).sum(-1)
            * selected
        ).sum(-1)
        hidden_reference_kl = equal_variance_gaussian_kl(
            current_mean, reference_mean_raw.float(), sigma
        )
        hidden_reference_kl = torch.where(hidden_valid, hidden_reference_kl, 0.0)
        reference_kl = token_reference_kl + hidden_reference_kl

        masked = (x == getattr(model, "mask_idx", 2))
        current_selection = (
            masked & valid[:, None]
            if old.force_all_positions
            else select_positions(
                current_logits.detach().float(), masked,
                temperature=confidence_temperature, threshold=threshold, active=valid,
            )
        )
        intersection = (current_selection & selected).sum(-1).float()
        union = (current_selection | selected).sum(-1).float().clamp_min(1)
        exact = (current_selection == selected).all(-1)

        valid_float = valid.float()
        step_policy = -(surrogate * valid_float).sum() / denominator
        step_kl = (reference_kl * valid_float).sum() / denominator
        step_loss = step_policy + kl_beta * step_kl
        step_loss.backward()

        with torch.no_grad():
            count = valid_float.sum().item()
            sums["policy_loss"] += step_policy.item()
            sums["reference_kl"] += step_kl.item()
            sums["valid_count"] += count
            for name, value in (
                ("token_kl_old", token_kl_old),
                ("hidden_kl_old", hidden_kl_old),
                ("joint_kl_old", token_kl_old + hidden_kl_old),
                ("token_reference_kl", token_reference_kl),
                ("hidden_reference_kl", hidden_reference_kl),
                ("token_log_ratio", token_log_ratio),
                ("hidden_log_ratio", hidden_log_ratio),
                ("joint_log_ratio", joint_log_ratio),
            ):
                sums[name] += (value * valid_float).sum().item()
            lower_log_clip = math.log(1 - clip_epsilon)
            upper_log_clip = math.log(1 + clip_epsilon)
            sums["joint_clip_count"] += (
                ((joint_log_ratio < lower_log_clip) | (joint_log_ratio > upper_log_clip))
                * valid
            ).sum().item()
            sums["token_clip_count"] += (
                ((token_log_ratio < lower_log_clip) | (token_log_ratio > upper_log_clip))
                * valid
            ).sum().item()
            sums["hidden_clip_count"] += (
                ((hidden_log_ratio < lower_log_clip) | (hidden_log_ratio > upper_log_clip))
                * valid
            ).sum().item()
            sums["position_exact_count"] += (exact & valid).sum().item()
            sums["position_jaccard"] += ((intersection / union) * valid_float).sum().item()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    optimizer.step()
    count = max(sums.pop("valid_count"), 1.0)
    return {
        "loss": sums["policy_loss"] + kl_beta * sums["reference_kl"],
        "policy_loss": sums["policy_loss"],
        "reference_kl_per_step": sums["reference_kl"] * denominator / count,
        "token_kl_old_per_step": sums["token_kl_old"] / count,
        "hidden_kl_old_per_step": sums["hidden_kl_old"] / count,
        "joint_kl_old_per_step": sums["joint_kl_old"] / count,
        "token_reference_kl_per_step": sums["token_reference_kl"] / count,
        "hidden_reference_kl_per_step": sums["hidden_reference_kl"] / count,
        "token_log_ratio_mean": sums["token_log_ratio"] / count,
        "hidden_log_ratio_mean": sums["hidden_log_ratio"] / count,
        "joint_log_ratio_mean": sums["joint_log_ratio"] / count,
        "joint_clip_fraction": sums["joint_clip_count"] / count,
        "token_threshold_fraction": sums["token_clip_count"] / count,
        "hidden_threshold_fraction": sums["hidden_clip_count"] / count,
        "position_exact_fraction": sums["position_exact_count"] / count,
        "position_jaccard": sums["position_jaccard"] / count,
        "grad_norm": float(grad_norm),
    }


def supervised_lora_loss(
    model: nn.Module,
    *,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    fixed: torch.Tensor,
    mask_token_id: int,
    num_steps: int,
    confidence_temperature: float,
    threshold: float,
    sigma: float,
    gaussian_hidden: bool,
    generator: Optional[torch.Generator] = None,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    """Two-step teacher-forced LoRA-SFT objective, optionally with hidden noise."""
    if gaussian_hidden and sigma <= 0:
        raise ValueError("Gaussian SFT requires sigma > 0")
    device = next(model.parameters()).device
    x = prompt_ids.to(device)
    targets = target_ids.to(device)
    fixed = fixed.to(device)
    h = torch.zeros(x.shape[0], x.shape[1], model.d_model, device=device)
    losses: List[torch.Tensor] = []
    for _ in range(num_steps):
        masked = (x == mask_token_id) & ~fixed
        with _autocast(device, autocast_dtype):
            logits, mean = model(x, h)
        per_token = F.cross_entropy(
            logits.float().transpose(1, 2), targets, reduction="none"
        )
        losses.append((per_token * masked).sum() / masked.sum().clamp_min(1))
        selected = select_positions(
            logits.detach().float(), masked, temperature=confidence_temperature,
            threshold=threshold,
        )
        x = torch.where(selected, targets, x)
        h = mean.float()
        if gaussian_hidden:
            noise = torch.randn(
                h.shape, device=h.device, dtype=h.dtype, generator=generator
            )
            h = h + sigma * noise
    return torch.stack(losses).sum()
