"""GPU test: verify BPTT loss shift alignment matches model's internal loss.

The model's own ``ForCausalLMLoss`` left-shifts labels (``labels[1:]``) to
achieve next-token prediction.  Our BPTT loss instead right-shifts logits via
``_shift_logits`` so that position indices stay aligned with mask/unmask
tensors.

This test verifies the two approaches are numerically equivalent on a
fully-masked block:

  1. Forward the model normally (``bypass_noising=False``), let it compute
     ``out.loss`` via its internal ``ForCausalLMLoss`` on its own random
     masking.
  2. Take the same ``out.logits`` (already halved to ``(B, L)``), apply
     ``_shift_logits``, and compute per-position CE against the same
     labels using ``F.cross_entropy(..., ignore_index=-100)``.
  3. Average over the same ``labels != -100`` positions the model used.
  4. Assert the two scalars match to within bf16 rounding tolerance.

Also tests that ``_shift_logits`` is equivalent to the shift in
``eval.py:get_logits`` on random tensors.

Requires GPU (flex_attention). Run from Fast-dLLM/v2/:
    python scripts/test_bptt_shift_alignment.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
import torch.nn.functional as F


def test_shift_logits_matches_eval_py():
    """``_shift_logits`` == ``eval.py``'s ``get_logits`` shift."""
    from lmflow.pipeline.utils.block_bptt_loss import FastDLLMBlockBPTTLoss

    B, L, V = 2, 16, 100
    logits = torch.randn(B, L, V)
    shifted = FastDLLMBlockBPTTLoss._shift_logits(logits)

    eval_py_shifted = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
    assert torch.equal(shifted, eval_py_shifted), "shift_logits != eval.py shift"
    assert shifted.shape == (B, L, V), f"wrong shape {shifted.shape}"
    print("[PASS] _shift_logits matches eval.py convention")


def test_shifted_ce_matches_model_internal_loss():
    """Our shifted per-position CE matches ForCausalLMLoss on identical logits."""
    from lmflow.pipeline.utils.block_bptt_loss import FastDLLMBlockBPTTLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[SKIP] no CUDA device — this test requires GPU")
        return

    import lmflow.models.fast_dllm  # noqa: F401 — register Auto* classes
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.for_model("Fast_dLLM_Qwen")
    config.num_hidden_layers = 2
    config.hidden_size = 128
    config.intermediate_size = 256
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.bd_size = 32
    config.layer_types = ["full_attention"] * 2
    config.conplemenrary_mask = True

    model = AutoModelForCausalLM.from_config(config).to(device).to(torch.bfloat16)
    model.train()

    MASK_ID = 151665
    BD = config.bd_size
    B = 2
    L = BD * 2

    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (B, L), device=device)
    labels = input_ids.clone()
    labels[:, :BD] = -100

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)

    model_loss = out.loss
    raw_logits = out.logits  # (B, L, V) — already halved inside the model

    shifted = FastDLLMBlockBPTTLoss._shift_logits(raw_logits.float())
    per_pos_ce = F.cross_entropy(
        shifted.transpose(1, 2), labels, reduction="none", ignore_index=-100,
    )
    valid = labels != -100
    our_loss = per_pos_ce[valid].mean()

    print(f"  model internal loss (ForCausalLMLoss): {model_loss.item():.6f}")
    print(f"  our shifted per-position CE mean:      {our_loss.item():.6f}")
    diff = abs(model_loss.item() - our_loss.item())
    print(f"  absolute diff:                         {diff:.6e}")

    # NOTE: there will be a small but nonzero diff because:
    #   - ForCausalLMLoss's shift drops position 0 and appends -100 at the end,
    #     meaning it computes CE over positions [1..L-1] of the labels.
    #   - Our shift keeps all L positions, but position 0 gets a garbage logit
    #     (duplicated logits[:, 0]) which labels[:, 0] == -100 ignores anyway.
    #   - Position L-1: ForCausalLMLoss drops it (shifted label = -100 pad).
    #     Our shift uses logits[:, L-2] to predict labels[:, L-1].
    # So we differ by exactly 1 position (the last) out of N valid positions.
    # For this test the difference should be well below 1.0 (bf16 rounding
    # plus one extra position).
    #
    # A stricter check: exclude the last response position from both sides.
    labels_trimmed = labels.clone()
    labels_trimmed[:, -1] = -100
    per_pos_ce_trimmed = F.cross_entropy(
        shifted.transpose(1, 2), labels_trimmed, reduction="none", ignore_index=-100,
    )
    valid_trimmed = labels_trimmed != -100
    our_loss_trimmed = per_pos_ce_trimmed[valid_trimmed].mean()

    # ForCausalLMLoss effectively ignores the last position too (shifted pad),
    # so model_loss should be very close to our_loss_trimmed.
    diff_strict = abs(model_loss.item() - our_loss_trimmed.item())
    print(f"  strict diff (excl last pos):           {diff_strict:.6e}")
    assert diff_strict < 0.05, (
        f"Shifted CE diverges from model loss by {diff_strict:.4f} "
        f"(model={model_loss.item():.4f}, ours={our_loss_trimmed.item():.4f}). "
        "Logit shift is likely wrong."
    )
    print("[PASS] shifted CE matches model internal loss (strict)")


def test_shift_does_not_break_shapes():
    """Sanity: _shift_logits preserves shape and doesn't NaN."""
    from lmflow.pipeline.utils.block_bptt_loss import FastDLLMBlockBPTTLoss

    for shape in [(1, 1, 10), (2, 64, 100), (4, 128, 152064)]:
        B, L, V = shape
        logits = torch.randn(B, L, V)
        shifted = FastDLLMBlockBPTTLoss._shift_logits(logits)
        assert shifted.shape == logits.shape, f"shape mismatch for {shape}"
        assert not shifted.isnan().any(), f"NaN for {shape}"
    print("[PASS] _shift_logits shape & NaN checks")


if __name__ == "__main__":
    test_shift_logits_matches_eval_py()
    test_shift_does_not_break_shapes()
    test_shifted_ce_matches_model_internal_loss()
    print("\nAll tests passed.")
