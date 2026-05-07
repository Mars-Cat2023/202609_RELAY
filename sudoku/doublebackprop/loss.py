from typing import Any, Dict, List, Optional, Tuple, cast, TypedDict

import torch
from .flex_attn_utils import block_mask_from_segment_ids
from .streaming_batch import StreamingBatch
from .types import MLMBatch, MLMLossDict, MLMModel
from xlm.harness import LossFunction, Harness
from xlm.datamodule import Tokenizer
from xlm.utils.nn import masked_mean, select_random_indices
from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


def loopholing_forward_kwargs_from_storage(
    storage: Optional[Dict[str, Any]],
    target_device: torch.device,
    sb: Optional[StreamingBatch] = None,
) -> Dict[str, Any]:
    """Build kwargs for ``RotaryTransformerLoopholingModel.forward`` (xlm-core flex design).

    ``block_mask`` is set on ``sb`` from ``segment_ids`` in **storage** (after eviction), not
    from the collator's pre-built mask, so batch row ``b`` always matches storage row ``b``.
    """
    if storage is None:
        return {}
    out: Dict[str, Any] = {}
    pos = storage.get("positions")
    if pos is not None:
        out["positions"] = pos.to(device=target_device)
    if sb is not None and sb.block_mask is not None:
        out["block_mask"] = sb.block_mask
    return out


def ensure_batch_fixed_for_streaming(
    batch: MLMBatch,
    batch_shape_key: str = "input_ids",
) -> None:
    """Mutate ``batch`` in place: add ``fixed`` if absent.

    ``StreamingBatch`` and PUMA losses expect a bool ``fixed`` mask (True = never
    re-mask / clue positions). Collators like ``mlm.datamodule_mlm.DefaultMLMCollator``
    do not set it; for plain MLM (e.g. OWT) no positions are pinned, so use all False.
    """
    if isinstance(batch.get("fixed"), torch.Tensor):
        return
    if batch_shape_key not in batch:
        raise KeyError(batch_shape_key)
    ref = batch[batch_shape_key]
    if not isinstance(ref, torch.Tensor):
        raise TypeError(f"{batch_shape_key} must be a torch.Tensor")
    batch["fixed"] = torch.zeros(
        ref.shape,
        dtype=torch.bool,
        device=ref.device,
    )


class MLMLoss(LossFunction[MLMBatch, MLMLossDict]):
    def __init__(
        self,
        loss_on_visible_tokens: bool = False,
        model: Optional[MLMModel] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.loss_on_visible_tokens = loss_on_visible_tokens
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token_id_tensor = None

    def configure(self, pl_module: Harness):
        self.mask_token_id_tensor = torch.tensor(  # type: ignore
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )

    def __call__(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        loss_dict = self.loss_fn(
            batch, batch_idx, dataloader_idx, dataloader_name
        )
        return loss_dict

    def loss_fn(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:

        input_ids = batch["input_ids"]
        targets = batch["target_ids"]
        assert targets is not None

        model = cast(MLMModel, self.model)
        logits = model(input_ids)

        ignore = torch.zeros_like(input_ids, dtype=torch.bool)
        if not self.loss_on_visible_tokens:
            ignore = ignore.logical_or(input_ids != self.mask_token_id_tensor)  # type: ignore
        targets[ignore] = -100
        if ignore.all():
            # Need to do this manually because pytorch doesn't do the logical thing. See https://github.com/pytorch/pytorch/issues/70348
            return {
                "loss": torch.tensor(
                    0.0,
                    device=logits.device,
                    dtype=logits.dtype,
                    requires_grad=True,
                )
            }

        logits_T = logits.transpose(1, 2)

        ce_per_token = torch.nn.functional.cross_entropy(
            logits_T, targets, reduction="none", ignore_index=-100
        )
        # Match ``mlm.loss_mlm.MLMLoss``: average over masked positions only, not full seq.
        ce = masked_mean(ce_per_token.flatten(), ~ignore.flatten(), dim=-1)

        return {"loss": ce}


class LoopholingBPTTPumaLoss(LossFunction[MLMBatch, MLMLossDict]):
    """Streaming PUMA batch with optional multi-step Loopholing BPTT.

    Maintains a StreamingBatch of partially unmasked examples. With
    ``with_loopholing=True`` the storage also includes persistent h_s (last hidden state):
    each call evicts/fills slots, runs num_steps loopholing unroll on the buffer, persists
    detached h_s and updated input_ids, then updates eviction state. With
    ``with_loopholing=False`` the same PUMA buffer and teacher-forced on-policy unmask
    schedule are used, but each step is a plain ``model(x)`` forward with no hidden carry.

    **Diagnostic logging:** optional train-only metrics are marked in-code with
    ``[LOSS_DIAGNOSTIC]``. Grep for that tag to find flags, ``loss_fn`` hooks, helpers
    (``_replay_*``, ``_compute_*grad*``, ``_do_log``), and related Hydra keys for removal.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Tokenizer] = None,
        predictor: Optional[Any] = None,
        loss_on_visible_tokens: bool = False,
        use_model_confidence: bool = True,
        batch_shape_key: str = "input_ids",
        num_steps: int = 2,
        stop_grad_h_s: bool = False,
        with_loopholing: bool = True,
        weighted_ce: bool = False,
        threshold_sampling_sigma: float = 0.0,
        # --- [LOSS_DIAGNOSTIC] Optional train metrics (default off; safe to delete as a group) ---
        log_puzzle_diagnostics: bool = False,
        log_gradient_norms: bool = False,
        log_l1_bptt_via_h_diagnosis: bool = False,
        # --- end [LOSS_DIAGNOSTIC] __init__ flags ---
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.predictor = predictor
        self.loss_on_visible_tokens = loss_on_visible_tokens
        self.use_model_confidence = use_model_confidence
        self.batch_shape_key = batch_shape_key
        self.num_steps = num_steps
        self.stop_grad_h_s = stop_grad_h_s
        self.with_loopholing = with_loopholing
        self.weighted_ce = weighted_ce
        self.threshold_sampling_sigma = threshold_sampling_sigma
        self.log_puzzle_diagnostics = log_puzzle_diagnostics
        self.log_gradient_norms = log_gradient_norms
        self.log_l1_bptt_via_h_diagnosis = log_l1_bptt_via_h_diagnosis
        self.pl_module = None
        self.mask_token_id_tensor = None
        # [LOSS_DIAGNOSTIC] Cached in configure(); only for log_gradient_norms /
        # log_l1_bptt_via_h_diagnosis (remove with those code paths).
        self._grad_params: Optional[List[torch.nn.Parameter]] = None
        if self.threshold_sampling_sigma < 0:
            raise ValueError(
                f"threshold_sampling_sigma must be >= 0, got {self.threshold_sampling_sigma}."
            )

    def configure(self, pl_module: Harness) -> None:
        self.pl_module = pl_module
        if self.predictor is None:
            self.predictor = pl_module.predictor
        self.mask_token_id_tensor = torch.tensor(
            self.tokenizer.mask_token_id,
            dtype=torch.long,
            device=pl_module.device,
        )
        self.streaming_batch = StreamingBatch()
        self.val_streaming_batch = StreamingBatch()
        # [LOSS_DIAGNOSTIC] See __init__ gradient / L1-BPTT flags.
        self._grad_params = [
            p for p in pl_module.model.parameters() if p.requires_grad
        ]

    def __call__(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        return self.loss_fn(batch, batch_idx, dataloader_idx, dataloader_name)

    def _compute_ce(
        self,
        logits: torch.Tensor,
        x_clean: torch.Tensor,
        x_input: torch.Tensor,
        fixed: torch.Tensor,
    ) -> torch.Tensor:
        targets = x_clean.clone()
        ignore = torch.zeros_like(x_input, dtype=torch.bool)
        if not self.loss_on_visible_tokens:
            ignore = ignore.logical_or(x_input != self.mask_token_id_tensor)
        else:
            ignore = ignore.logical_or(fixed)
        targets[ignore] = -100
        ce = torch.nn.functional.cross_entropy(
            logits.transpose(1, 2), targets, reduction="none", ignore_index=-100,
        )
        
        if self.weighted_ce:
            # Per-position weight: w_i = (1 + conf_i). Detach so confidence acts as a fixed
            # scalar multiplier: without detach the product rule introduces a second term
            # ∂(CE_i * conf_i)/∂θ = CE_i * ∂conf_i/∂θ that incentivises confidence on
            # high-loss positions — opposite of the intended effect.
            confidence = self.predictor.compute_confidence(logits).detach()  # (B, L)
            weights = 1.0 + confidence  # (B, L)
            ce = ce * weights
        
        return masked_mean(ce.flatten(), ~ignore.flatten(), dim=-1)

    def _select_unmask(
        self,
        logits: torch.Tensor,
        masked_mutable: torch.Tensor,
        confidence_threshold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.use_model_confidence:
                return self.predictor.compute_confidence_unmask(
                    logits,
                    masked_mutable,
                    threshold=confidence_threshold,
                )
            num_unmask = (masked_mutable.sum(dim=-1) // 2).clamp(min=1)
            return select_random_indices(
                inp_shape=logits.shape[:2],
                num_unmask=num_unmask,
                select_from_mask=masked_mutable,
                selection_score=None,
                selection_mode="sample",
            )

    def _sample_confidence_threshold(
        self,
        *,
        device: torch.device,
        training: bool,
    ) -> Optional[float]:
        if not self.use_model_confidence:
            return None
        mu = getattr(self.predictor, "threshold", None)
        if mu is None:
            return None
        mu = float(mu)
        if (not training) or self.threshold_sampling_sigma == 0:
            return mu

        sampled = mu + self.threshold_sampling_sigma * torch.randn(
            (), device=device
        ).item() # DP: Never call .item() on tensors in training code. It is not compile friendly and introduces a GPU->CPU transfer.
        # Keep thresholds non-negative; fallback in _select_unmask still guarantees >=1 unmask.
        return max(0.0, sampled)

    def loss_fn(
        self,
        batch: MLMBatch,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
        dataloader_name: Optional[str] = None,
    ) -> MLMLossDict:
        is_training = bool(self.pl_module.training) if self.pl_module is not None else True

        ensure_batch_fixed_for_streaming(batch, self.batch_shape_key)
        if not is_training:
            # Validation: reset so evict_and_fill re-initializes from the current batch,
            # sidestepping the training buffer's fixed-capacity constraint. Each val call
            # is independent (no cross-batch carry), which is the correct semantics.
            self.val_streaming_batch.reset()
        sb = self.val_streaming_batch if not is_training else self.streaming_batch
        # Slots that will be filled this step (before evict_and_fill mutates state)
        evicted_candidates: Optional[torch.Tensor] = None
        if sb.storage is not None:
            evicted_candidates = sb.ready_to_evict.nonzero(as_tuple=True)[0]

        evicted_count = sb.evict_and_fill(
            batch,
            mask_token_id=self.mask_token_id_tensor,
            batch_shape_key=self.batch_shape_key,
        )

        device = sb.storage["input_ids"].device
        sb.block_mask = None
        if getattr(self.model, "use_flex_attn", False):
            # Build from ``storage['segment_ids']`` after evict_and_fill — row-aligned with
            # ``input_ids`` even when only *some* slots were evicted.  The collator's
            # ``batch['block_mask']`` is only valid for a contiguous microbatch layout, not
            # for a mixed-age buffer after partial eviction.
            seg_stored = sb.storage.get("segment_ids")
            if seg_stored is None:
                raise ValueError(
                    "use_flex_attn=True requires segment_ids in the batch / storage "
                    "(PackedMLMCollator)."
                )
            sb.block_mask = block_mask_from_segment_ids(seg_stored, device)
        capacity, seq_len = sb.capacity, sb.seq_len

        if self.with_loopholing:
            d_model = self.model.d_model
            # Ensure h_s exists in storage; zero for freshly filled slots (detach across calls)
            if "h_s" not in sb.storage:
                sb.storage["h_s"] = torch.zeros(
                    capacity, seq_len, d_model,
                    device=device,
                    dtype=torch.float32,
                )
            if evicted_count > 0 and evicted_candidates is not None:
                evicted_slots = evicted_candidates[:evicted_count]
                sb.storage["h_s"][evicted_slots] = 0.0
        else:
            sb.storage.pop("h_s", None)

        x = sb.storage["input_ids"].clone()
        x_clean = sb.storage["target_ids"]
        fixed = sb.storage["fixed"]
        # Clone so we never use the buffer in the autograd graph; we copy back after the loop.
        h: Optional[torch.Tensor] = (
            sb.storage["h_s"].clone() if self.with_loopholing else None
        )
        # Flex: positions from storage + patched block_mask on ``sb``; OWT/puzzles: often {}.
        forward_kw = loopholing_forward_kwargs_from_storage(sb.storage, device, sb)
        # [LOSS_DIAGNOSTIC] L1 BPTT replay tape only when that diagnosis runs (e.g. puzzle YAMLs).
        # Skipped for OWT / default runs so we do not clone x,h or unmasks every step.
        need_l1_replay_tape = (
            self.with_loopholing
            and h is not None
            and self.log_l1_bptt_via_h_diagnosis
            and not self.stop_grad_h_s
            and self.num_steps >= 2
            and bool(self._grad_params)
            and self.pl_module is not None
            and self.pl_module.training
        )
        x_unroll_start: Optional[torch.Tensor] = None
        h_unroll_start: Optional[torch.Tensor] = None
        recorded_unmasks: List[torch.Tensor] = []
        if need_l1_replay_tape:
            x_unroll_start = x.clone()
            assert h is not None
            h_unroll_start = h.clone()

        loss_terms: List[torch.Tensor] = []
        masks_per_step: List[float] = []
        last_unmask: Optional[torch.Tensor] = None
        sampled_thresholds: List[float] = []

        # [LOSS_DIAGNOSTIC] log_puzzle_diagnostics (buffer / hidden RMS, etc.)
        h_carry_rms: List[float] = []
        h_s_rms: List[float] = []
        h_delta_rms: List[float] = []
        unmask_frac: List[float] = []

        for t in range(self.num_steps):
            masked_mutable = (x == self.tokenizer.mask_token_id) & ~fixed
            has_any_mask = masked_mutable.any()

            masks_per_step.append(masked_mutable.sum().float().item() / capacity)

            # [LOSS_DIAGNOSTIC] (log_puzzle_diagnostics)
            if self.log_puzzle_diagnostics and h is not None:
                with torch.no_grad():
                    hf = h.detach().float()
                    h_carry_rms.append(torch.sqrt(hf.pow(2).mean()).item())

            if self.with_loopholing:
                assert h is not None
                logits, h_s = self.model(x, h, **forward_kw)
            else:
                logits = self.model(x)
                h_s = None

            L_t = self._compute_ce(logits, x_clean, x, fixed)
            # Keep a gradient path to model parameters even when there are no masked
            # positions on this rank. This prevents DDP bucket-shape divergence.
            anchor = 0.0 * logits.sum()
            L_t = L_t + anchor
            loss_terms.append(L_t)

            sampled_threshold = self._sample_confidence_threshold(
                device=logits.device,
                training=is_training,
            )
            threshold_t = (
                torch.tensor(sampled_threshold, device=logits.device, dtype=logits.dtype)
                if sampled_threshold is not None
                else None
            )
            if sampled_threshold is not None:
                sampled_thresholds.append(sampled_threshold)
            unmask = self._select_unmask(
                logits.detach(),
                masked_mutable,
                confidence_threshold=threshold_t,
            )
            # Preserve no-op semantics when no masked mutable positions remain.
            unmask = unmask & masked_mutable
            last_unmask = unmask
            if need_l1_replay_tape:
                recorded_unmasks.append(unmask.clone())

            x = x.clone()
            with torch.no_grad():
                x[unmask] = x_clean[unmask]

            if self.with_loopholing:
                assert h is not None and h_s is not None
                # Match prior behavior: when nothing is masked, keep h unchanged.
                h_old = h
                h_next = h_s.detach() if self.stop_grad_h_s else h_s
                h_new = torch.where(has_any_mask, h_next, h_old)

                # [LOSS_DIAGNOSTIC] (log_puzzle_diagnostics)
                if self.log_puzzle_diagnostics:
                    with torch.no_grad():
                        hs = h_s.detach().float()
                        h_s_rms.append(torch.sqrt(hs.pow(2).mean()).item())
                        delta = h_new.float() - h_old.float()
                        h_delta_rms.append(torch.sqrt(delta.pow(2).mean()).item())
                        unmask_frac.append(unmask.float().mean().item())

                h = h_new
            elif self.log_puzzle_diagnostics:
                unmask_frac.append(unmask.float().mean().item())

        # loss_terms always has num_steps entries because the loop no longer breaks early.
        # torch.stack+sum avoids a Python-level list fold and is fully compile-friendly.
        total_loss = torch.stack(loss_terms).sum()
        # Persist state for next call (detach across calls; replace tensor references,
        # never copy_ into live storage so there is no aliasing into the autograd graph).
        with torch.no_grad():
            sb.storage["input_ids"] = x.detach().clone()
            if self.with_loopholing:
                assert h is not None
                sb.storage["h_s"] = h.detach().clone()
        # last_unmask is always set because the loop always runs at least one step.
        sb.update_after_unmask(last_unmask, x_clean, self.mask_token_id_tensor)

        # [LOSS_DIAGNOSTIC] (log_puzzle_diagnostics)
        loss_share: List[float] = []
        if self.log_puzzle_diagnostics and loss_terms:
            td = total_loss.detach().abs().clamp_min(1e-8)
            loss_share = [(li.detach() / td).item() for li in loss_terms]

        # ------------------------------------------------------------------
        # [LOSS_DIAGNOSTIC] Extra autograd + replay (log_l1_bptt_via_h_diagnosis,
        # log_gradient_norms). Runs before _do_log; remove this block + helpers below.
        # L1 path runs first so ∇L1 can be reused for log_gradient_norms when both on.
        # ------------------------------------------------------------------
        l1_bptt_diag: Optional[Dict[str, float]] = None
        reuse_flat_grad_L1: Optional[torch.Tensor] = None
        if (
            need_l1_replay_tape
            and len(recorded_unmasks) == self.num_steps
            and x_unroll_start is not None
            and h_unroll_start is not None
        ):
            l1_bptt_diag, reuse_flat_grad_L1 = (
                self._compute_l1_grad_via_h_decomposition(
                    loss_terms,
                    x_unroll_start,
                    h_unroll_start,
                    x_clean,
                    fixed,
                    recorded_unmasks,
                    forward_kw=forward_kw,
                )
            )

        grad_norm_per_L: Optional[List[float]] = None
        grad_cosine_Li: Optional[List[float]] = None
        total_grad_norm: Optional[float] = None
        if (
            self.log_gradient_norms
            and self._grad_params
            and self.pl_module is not None
            and self.pl_module.training
        ):
            grad_norm_per_L, grad_cosine_Li, total_grad_norm = (
                self._compute_per_loss_grad_stats(
                    loss_terms,
                    total_loss,
                    reuse_flat_grad_L1=reuse_flat_grad_L1,
                )
            )
        # --- end [LOSS_DIAGNOSTIC] extra autograd / replay ---

        # [LOSS_DIAGNOSTIC] Lightning train/* logs (see _do_log); no extra compute.
        if self.pl_module is not None and self.pl_module.training and (
            self.log_puzzle_diagnostics
            or grad_norm_per_L is not None
            or l1_bptt_diag
        ):
            self._do_log(
                x,
                loss_terms,
                masks_per_step,
                evicted_count,
                capacity,
                sampled_thresholds,
                h_carry_rms=h_carry_rms,
                h_s_rms=h_s_rms,
                h_delta_rms=h_delta_rms,
                unmask_frac=unmask_frac,
                loss_share=loss_share,
                grad_norm_per_L=grad_norm_per_L,
                grad_cosine_Li=grad_cosine_Li,
                total_grad_norm=total_grad_norm,
                l1_bptt_diag=l1_bptt_diag,
            )
        # --- end [LOSS_DIAGNOSTIC] _do_log call ---

        result: MLMLossDict = {"loss": total_loss}
        #for i, L_i in enumerate(loss_terms):
        #    result[f"L{i}"] = L_i.detach()
        return result

    # =========================================================================
    # [LOSS_DIAGNOSTIC] Helpers: L1 BPTT-through-h decomposition + per-loss grad norms.
    # Safe to delete entire section with __init__ flags, configure._grad_params,
    # loss_fn snapshot/replay hooks, and _do_log branches.
    # =========================================================================

    def _replay_unroll_detached_h_carry(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        x_clean: torch.Tensor,
        fixed: torch.Tensor,
        unmask_per_step: List[torch.Tensor],
        forward_kw: Optional[Dict[str, Any]] = None,
    ) -> List[torch.Tensor]:
        """[LOSS_DIAGNOSTIC] Replay unroll with ``h_s.detach()`` between steps (same as
        ``stop_grad_h_s=True``), using a fixed unmask schedule so the forward matches the
        main run. Used only to build the blocked graph for ∇L1 decomposition.
        """
        if forward_kw is None:
            forward_kw = {}
        loss_terms: List[torch.Tensor] = []
        for t in range(self.num_steps):
            masked_mutable = (x == self.tokenizer.mask_token_id) & ~fixed
            has_any_mask = masked_mutable.any()

            if self.with_loopholing:
                logits, h_s = self.model(x, h, **forward_kw)
            else:
                logits = self.model(x)
                h_s = None

            L_t = self._compute_ce(logits, x_clean, x, fixed)
            anchor = 0.0 * logits.sum()
            L_t = L_t + anchor
            loss_terms.append(L_t)

            unmask = unmask_per_step[t] & masked_mutable
            x = x.clone()
            with torch.no_grad():
                x[unmask] = x_clean[unmask]

            if self.with_loopholing:
                assert h_s is not None
                h_old = h
                h_next = h_s.detach()
                h_new = torch.where(has_any_mask, h_next, h_old)
                h = h_new

        return loss_terms

    @torch.compiler.disable
    def _compute_l1_grad_via_h_decomposition(
        self,
        loss_terms: List[torch.Tensor],
        x0: torch.Tensor,
        h0: torch.Tensor,
        x_clean: torch.Tensor,
        fixed: torch.Tensor,
        unmask_per_step: List[torch.Tensor],
        forward_kw: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, float], Optional[torch.Tensor]]:
        """[LOSS_DIAGNOSTIC] Decompose ∇L1 into paths through vs around the carry:
        g_full − g_block ≈ via-h.

        ``g_full`` is ∇L1 on the training graph (``stop_grad_h_s=False``). ``g_block`` is
        ∇L1′ where L1′ is the second-step loss on a replay with detached carry (same
        unmasks). Their difference is the gradient attributable to backprop through the
        first forward into parameters. Extra forward + ``autograd.grad``; keep off unless
        ``log_l1_bptt_via_h_diagnosis=True``.

        Returns ``(metrics, g_full_flat)`` so ``log_gradient_norms`` can reuse ``g_full_flat``
        and avoid a second ``autograd.grad(L1)``.
        """
        params = self._grad_params
        if not params or len(loss_terms) < 2:
            return {}, None

        ref = loss_terms[0]

        def flatten(
            grads: Tuple[Optional[torch.Tensor], ...],
        ) -> torch.Tensor:
            parts: List[torch.Tensor] = []
            for g in grads:
                if g is not None:
                    parts.append(g.detach().reshape(-1).float())
            if not parts:
                return torch.zeros(1, device=ref.device, dtype=torch.float32)
            return torch.cat(parts)

        L1 = loss_terms[1]
        g_full = flatten(
            torch.autograd.grad(
                L1,
                params,
                retain_graph=True,
                allow_unused=True,
            )
        )

        replay_terms = self._replay_unroll_detached_h_carry(
            x0,
            h0,
            x_clean,
            fixed,
            unmask_per_step,
            forward_kw=forward_kw,
        )
        L1_blocked = replay_terms[1]

        g_block = flatten(
            torch.autograd.grad(
                L1_blocked,
                params,
                retain_graph=True,
                allow_unused=True,
            )
        )

        g_via_h = g_full - g_block

        nf = torch.linalg.vector_norm(g_full)
        nb = torch.linalg.vector_norm(g_block)
        nh = torch.linalg.vector_norm(g_via_h)

        def _cos(u: torch.Tensor, v: torch.Tensor) -> float:
            nu = torch.linalg.vector_norm(u)
            nv = torch.linalg.vector_norm(v)
            if float(nu.item()) < 1e-12 or float(nv.item()) < 1e-12:
                return 0.0
            return float(((u * v).sum() / (nu * nv).clamp_min(1e-12)).item())

        scalar_delta = float((L1 - L1_blocked).detach().abs().item())

        metrics = {
            "l1_grad_norm_full": float(nf.item()),
            "l1_grad_norm_second_forward_only": float(nb.item()),
            "l1_grad_norm_via_h_path": float(nh.item()),
            "l1_grad_norm_ratio_via_h_over_full": float(
                (nh / nf.clamp_min(1e-12)).item()
            ),
            "l1_grad_norm_ratio_via_h_over_second_forward_only": float(
                (nh / nb.clamp_min(1e-12)).item()
            ),
            "l1_grad_cosine_via_h_vs_full": _cos(g_via_h, g_full),
            "l1_grad_cosine_full_vs_second_forward_only": _cos(g_full, g_block),
            "l1_replay_scalar_abs_delta": scalar_delta,
        }
        return metrics, g_full

    @torch.compiler.disable
    def _compute_per_loss_grad_stats(
        self,
        loss_terms: List[torch.Tensor],
        total_loss: torch.Tensor,
        *,
        reuse_flat_grad_L1: Optional[torch.Tensor] = None,
    ) -> Tuple[List[float], List[float], float]:
        """[LOSS_DIAGNOSTIC] L2 norms of flattened ∇_θ L_i, cosines between consecutive ∇L_i,
        and ‖∇_θ total_loss‖₂.

        Extra ``torch.autograd.grad`` passes (``retain_graph=True``) so Lightning can still
        backward ``total_loss``. Expensive — use only with ``log_gradient_norms=True``.
        If ``reuse_flat_grad_L1`` is set (from ``_compute_l1_grad_via_h_decomposition``),
        skips ``autograd.grad(L1)`` for index 1.
        """
        params = self._grad_params
        if not params or not loss_terms:
            return [], [], 0.0

        ref = loss_terms[0]

        def flatten(
            grads: Tuple[Optional[torch.Tensor], ...],
        ) -> torch.Tensor:
            parts: List[torch.Tensor] = []
            for g in grads:
                if g is not None:
                    parts.append(g.detach().reshape(-1).float())
            if not parts:
                return torch.zeros(1, device=ref.device, dtype=torch.float32)
            return torch.cat(parts)

        norms: List[float] = []
        flat_vecs: List[torch.Tensor] = []
        for i, li in enumerate(loss_terms):
            if reuse_flat_grad_L1 is not None and i == 1:
                v = reuse_flat_grad_L1
            else:
                grads = torch.autograd.grad(
                    li,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                v = flatten(grads)
            flat_vecs.append(v)
            norms.append(float(torch.linalg.vector_norm(v).item()))

        cosines: List[float] = []
        for i in range(len(flat_vecs) - 1):
            a, b = flat_vecs[i], flat_vecs[i + 1]
            na = torch.linalg.vector_norm(a)
            nb = torch.linalg.vector_norm(b)
            if float(na.item()) < 1e-12 and float(nb.item()) < 1e-12:
                cosines.append(0.0)
            else:
                cos = (a * b).sum() / (na * nb).clamp_min(1e-12)
                cosines.append(float(cos.item()))

        g_total = torch.autograd.grad(
            total_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        v_total = flatten(g_total)
        total_norm = float(torch.linalg.vector_norm(v_total).item())

        return norms, cosines, total_norm

    @torch.compiler.disable
    def _do_log(
        self,
        x: torch.Tensor,
        loss_terms: List[torch.Tensor],
        masks_per_step: List[float],
        evicted_count: int,
        capacity: int,
        sampled_thresholds: List[float],
        *,
        h_carry_rms: Optional[List[float]] = None,
        h_s_rms: Optional[List[float]] = None,
        h_delta_rms: Optional[List[float]] = None,
        unmask_frac: Optional[List[float]] = None,
        loss_share: Optional[List[float]] = None,
        grad_norm_per_L: Optional[List[float]] = None,
        grad_cosine_Li: Optional[List[float]] = None,
        total_grad_norm: Optional[float] = None,
        l1_bptt_diag: Optional[Dict[str, float]] = None,
    ) -> None:
        """[LOSS_DIAGNOSTIC] Lightning ``train/*`` logs only (train step); no backward.

        Puzzle metrics: ``log_puzzle_diagnostics``. Gradient norms: ``log_gradient_norms``.
        L1 BPTT carry path: ``log_l1_bptt_via_h_diagnosis`` (with ``stop_grad_h_s=False``).
        Uses ``.item()`` and Lightning ``log``; ``@torch.compiler.disable``.
        """
        if self.pl_module is None:
            return
        # --- [LOSS_DIAGNOSTIC] branch: log_puzzle_diagnostics ---
        if self.log_puzzle_diagnostics:
            if loss_terms:
                masks_in_buffer = (x == self.mask_token_id_tensor).sum().float() / capacity
                self.pl_module.log(
                    "train/masks_in_buffer",
                    masks_in_buffer,
                    on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                )
            self.pl_module.log(
                "train/evicted_count",
                float(evicted_count),
                on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
            )
            for i, m in enumerate(masks_per_step):
                self.pl_module.log(
                    f"train/masks_at_x_{i}",
                    m,
                    on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                )
            if len(masks_per_step) >= 1:
                self.pl_module.log("train/masks_at_x_i", masks_per_step[0], on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)
            if len(masks_per_step) >= 2:
                self.pl_module.log("train/masks_at_x_i_plus_1", masks_per_step[1], on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)
            for i, L_i in enumerate(loss_terms):
                self.pl_module.log(
                    f"train/L{i}",
                    L_i.detach(),
                    on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                )
            for i, threshold in enumerate(sampled_thresholds):
                self.pl_module.log(
                    f"train/conf_threshold_t_{i}",
                    threshold,
                    on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                )

            if h_carry_rms:
                for i, v in enumerate(h_carry_rms):
                    self.pl_module.log(
                        f"train/h_carry_rms_t{i}",
                        v,
                        on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                    )
            if h_s_rms:
                for i, v in enumerate(h_s_rms):
                    self.pl_module.log(
                        f"train/h_s_rms_t{i}",
                        v,
                        on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                    )
            if h_delta_rms:
                for i, v in enumerate(h_delta_rms):
                    self.pl_module.log(
                        f"train/h_delta_rms_t{i}",
                        v,
                        on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                    )
            if unmask_frac:
                for i, v in enumerate(unmask_frac):
                    self.pl_module.log(
                        f"train/unmask_frac_t{i}",
                        v,
                        on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                    )
            if loss_share:
                for i, v in enumerate(loss_share):
                    self.pl_module.log(
                        f"train/loss_share_L{i}",
                        v,
                        on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                    )
                if len(loss_share) >= 2:
                    self.pl_module.log(
                        "train/loss_share_ratio_L1_over_L0",
                        loss_share[1] / max(loss_share[0], 1e-8),
                        on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                    )
        # --- end log_puzzle_diagnostics ---

        # --- [LOSS_DIAGNOSTIC] branch: log_gradient_norms ---
        if grad_norm_per_L:
            for i, gn in enumerate(grad_norm_per_L):
                self.pl_module.log(
                    f"train/grad_norm_L{i}",
                    gn,
                    on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                )
            if total_grad_norm is not None:
                self.pl_module.log(
                    "train/grad_norm_total_loss",
                    total_grad_norm,
                    on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                )
            if grad_cosine_Li:
                for i, c in enumerate(grad_cosine_Li):
                    self.pl_module.log(
                        f"train/grad_cosine_L{i}_L{i + 1}",
                        c,
                        on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                    )
            if len(grad_norm_per_L) >= 2:
                self.pl_module.log(
                    "train/grad_norm_ratio_L1_over_L0",
                    grad_norm_per_L[1] / max(grad_norm_per_L[0], 1e-12),
                    on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False,
                )
        # --- end log_gradient_norms ---

        # --- [LOSS_DIAGNOSTIC] branch: log_l1_bptt_via_h_diagnosis ---
        if l1_bptt_diag:
            key_to_metric = {
                "l1_grad_norm_full": "train/l1_grad_norm_full",
                "l1_grad_norm_second_forward_only": "train/l1_grad_norm_second_forward_only",
                "l1_grad_norm_via_h_path": "train/l1_grad_norm_via_h_path",
                "l1_grad_norm_ratio_via_h_over_full": "train/l1_grad_norm_ratio_via_h_over_full",
                "l1_grad_norm_ratio_via_h_over_second_forward_only": (
                    "train/l1_grad_norm_ratio_via_h_over_second_forward_only"
                ),
                "l1_grad_cosine_via_h_vs_full": "train/l1_grad_cosine_via_h_vs_full",
                "l1_grad_cosine_full_vs_second_forward_only": (
                    "train/l1_grad_cosine_full_vs_second_forward_only"
                ),
                "l1_replay_scalar_abs_delta": "train/l1_replay_scalar_abs_delta",
            }
            for k, metric in key_to_metric.items():
                if k in l1_bptt_diag:
                    self.pl_module.log(
                        metric,
                        l1_bptt_diag[k],
                        on_step=True,
                        on_epoch=False,
                        prog_bar=False,
                        logger=True,
                        sync_dist=False,
                    )
        # --- end log_l1_bptt_via_h_diagnosis ---


