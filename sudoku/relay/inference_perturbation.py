"""Inference-only interventions; the training predictor and model are unchanged."""
import math

import torch

from .predictor import ConfidenceBasedPredictor


class PerturbedRelayPredictor(ConfidenceBasedPredictor):
    def __init__(self, *, confidence_temperature=1.0, hidden_sigma=0.0,
                 noise_generator=None, **kwargs):
        super().__init__(**kwargs)
        if not math.isfinite(confidence_temperature) or confidence_temperature < 0:
            raise ValueError("confidence_temperature must be nonnegative and finite")
        if not math.isfinite(hidden_sigma) or hidden_sigma < 0:
            raise ValueError("hidden_sigma must be nonnegative and finite")
        if not self.with_relay or self.confidence != "top_prob":
            raise ValueError("This experiment requires relay and top_prob confidence")
        self.confidence_temperature = confidence_temperature
        self.hidden_sigma = hidden_sigma
        self.noise_generator = noise_generator

    def compute_confidence(self, logits):
        # Preserve the original numerical path exactly for T=1.
        if self.confidence_temperature == 1.0:
            return super().compute_confidence(logits)
        if self.confidence_temperature == 0.0:
            # Exact T -> 0+ limit: uniform mass on tied maxima, zero elsewhere.
            ties = (logits == logits.amax(dim=-1, keepdim=True)).sum(dim=-1)
            return ties.to(logits.dtype).reciprocal()
        return super().compute_confidence(logits / self.confidence_temperature)

    def _record_relay_rollout_step_diagnostics(self, *args, **kwargs):
        # Avoid the original diagnostic's per-step GPU synchronization.
        pass

    def predict_single_step(self, step_results, final_step=False):
        result = super().predict_single_step(step_results, final_step=final_step)
        self.forward_calls += 1
        if self.forward_calls == 1:
            self.first_step_remaining_masks = (
                (result["x"] == self.tokenizer.mask_token_id) & ~result["fixed"]
            ).sum(-1)
        filled = ~((result["x"] == self.tokenizer.mask_token_id) & ~result["fixed"]).any(-1)
        self.first_filled = torch.where(
            (self.first_filled < 0) & filled,
            result["steps_taken"].long(), self.first_filled,
        )
        # This is AFTER logits and token updates. The sampled hidden is fed to
        # the next forward. Perturb every hidden coordinate, including clues.
        # No noise is needed after the unconditional terminal forward.
        if self.hidden_sigma > 0 and not final_step:
            h = result["h"].float()
            noise = torch.randn(h.shape, device=h.device, dtype=torch.float32,
                                generator=self.noise_generator)
            result["h"] = h + self.hidden_sigma * noise
        return result

    @torch.inference_mode()
    def predict(self, batch, **kwargs):
        self.forward_calls = 0
        self.first_filled = torch.full(
            (batch["input_ids"].shape[0],), -1,
            dtype=torch.long, device=batch["input_ids"].device,
        )
        initially_filled = ~((batch["input_ids"] == self.tokenizer.mask_token_id)
                             & ~batch["fixed"]).any(-1)
        self.first_filled[initially_filled] = 0
        result = super().predict(batch, **kwargs)
        result["first_step_remaining_masks"] = self.first_step_remaining_masks.cpu().tolist()
        result["actual_forward_calls"] = self.forward_calls
        result["first_filled_step"] = self.first_filled.cpu().tolist()
        return result
