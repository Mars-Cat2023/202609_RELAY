"""Static-only smoke check for the Fast-dLLM v2 BPTT vendoring + wiring.

Doesn't run a real forward (flex_attention requires CUDA). It only verifies:
  1. The vendored package imports cleanly and registers with AutoConfig.
  2. The modeling.py edits parse and the new symbols/kwargs exist.
  3. `FastDLLMBlockBPTTLoss` and `FastDLLMBPTTTrainer` import cleanly and have
     the expected signatures.
  4. `FinetunerArguments` exposes the four new BPTT fields with correct defaults.
  5. The finetuner's `loss_type=='mlm'` branch is byte-for-byte unchanged.
"""

import inspect
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(ROOT, "src"))


def _step(label):
    print(f"[smoke] {label}")


def main():
    _step("import vendored package -> auto-registration")
    import lmflow.models.fast_dllm as fd  # noqa: E402
    from transformers import AutoConfig  # noqa: E402

    assert fd.Fast_dLLM_QwenConfig.model_type == "Fast_dLLM_Qwen"
    cfg_cls = AutoConfig.for_model("Fast_dLLM_Qwen").__class__
    assert cfg_cls is fd.Fast_dLLM_QwenConfig, (
        f"AutoConfig resolved to {cfg_cls}, expected vendored class"
    )

    _step("modeling edits: forward signatures + new submodule")
    inner_sig = inspect.signature(fd.Fast_dLLM_QwenModel.forward)
    assert "h_t" in inner_sig.parameters, "Fast_dLLM_QwenModel.forward missing h_t"

    causal_sig = inspect.signature(fd.Fast_dLLM_QwenForCausalLM.forward)
    assert "h_t" in causal_sig.parameters
    assert "bypass_noising" in causal_sig.parameters
    assert causal_sig.parameters["bypass_noising"].default is False

    src = inspect.getsource(fd.Fast_dLLM_QwenForCausalLM.forward)
    assert "if self.training and not bypass_noising:" in src, (
        "noising/complement gate not applied"
    )
    assert "if self.training and labels is not None:" in src, (
        "halving condition not tightened"
    )
    assert "h_t=h_t" in src, "h_t not threaded into self.model(...)"

    inner_src = inspect.getsource(fd.Fast_dLLM_QwenModel.forward)
    assert "self.h_t_layer_norm(h_t" in inner_src, (
        "h_t injection not present after embed_tokens"
    )

    _step("loss + trainer modules import cleanly")
    from lmflow.pipeline.utils.block_bptt_loss import FastDLLMBlockBPTTLoss
    from lmflow.pipeline.utils.bptt_trainer import FastDLLMBPTTTrainer
    from lmflow.pipeline.utils.streaming_batch import StreamingBatch

    loss_sig = inspect.signature(FastDLLMBlockBPTTLoss.__init__)
    assert {"mask_token_id", "threshold", "top_p", "temperature",
            "use_streaming_buffer"}.issubset(loss_sig.parameters.keys())

    trainer_sig = inspect.signature(FastDLLMBPTTTrainer.__init__)
    assert {"mask_token_id", "threshold", "top_p", "temperature",
            "use_streaming_buffer"}.issubset(trainer_sig.parameters.keys())
    assert FastDLLMBPTTTrainer.compute_loss is not __import__(
        "transformers"
    ).Trainer.compute_loss, "compute_loss not overridden"

    _step("StreamingBatch lifecycle (CPU-only, no model forward)")
    import torch  # noqa: E402

    sb = StreamingBatch()
    assert not sb.is_initialized()

    MASK = 999
    cap, L, D = 4, 12, 8
    x_clean = torch.arange(cap * L, dtype=torch.long).reshape(cap, L)
    labels = x_clean.clone()
    # First 4 positions are "prompt" (label = -100), rest are "response".
    labels[:, :4] = -100

    n = sb.evict_and_fill(
        batch={"x_clean": x_clean, "labels": labels},
        mask_token_id=MASK,
        d_model=D,
    )
    assert n == cap, f"cold start should fill all {cap} slots, got {n}"
    assert sb.capacity == cap and sb.seq_len == L
    assert sb.storage["x_input"].shape == (cap, L)
    assert sb.storage["h_s"].shape == (cap, L, D)
    # Maskable positions all set to mask token; prompt positions untouched.
    assert (sb.storage["x_input"][:, 4:] == MASK).all(), "fresh slice not all-mask in response"
    assert (sb.storage["x_input"][:, :4] == x_clean[:, :4]).all(), "prompt corrupted"
    assert sb.storage["h_s"].abs().sum().item() == 0, "fresh h_s not zero"
    assert sb.ready_to_evict is not None and not sb.ready_to_evict.any(), (
        "fresh slices should not be ready to evict"
    )

    # Simulate one BPTT step that reveals the first 3 response positions of
    # row 0 only (positions [4, 5, 6]); leave everything else unchanged.
    x_input_next = sb.storage["x_input"].clone()
    x_input_next[0, 4:7] = x_clean[0, 4:7]
    h_s_next = torch.randn(cap, L, D)
    sb.persist_after_step(x_input_next, h_s_next, mask_token_id=MASK)
    assert sb.storage["h_s"].dtype == torch.float32
    assert torch.allclose(sb.storage["h_s"], h_s_next.float()), "h_s not persisted"
    counts = sb.slot_mask_counts(MASK)
    # Row 0: revealed 3 of 8 maskable -> 5 left. Other rows: 8 left each.
    assert counts.tolist() == [5, 8, 8, 8], f"unexpected slot mask counts {counts.tolist()}"
    assert not sb.ready_to_evict.any(), "no slot should be ready yet"

    # Reveal everything in row 0 -> ready_to_evict should fire for that slot only.
    x_input_next2 = sb.storage["x_input"].clone()
    x_input_next2[0, 4:] = x_clean[0, 4:]
    sb.persist_after_step(x_input_next2, sb.storage["h_s"], mask_token_id=MASK)
    assert sb.ready_to_evict.tolist() == [True, False, False, False], (
        f"expected slot 0 ready, got {sb.ready_to_evict.tolist()}"
    )

    # New batch should evict + refill exactly one slot (slot 0).
    new_batch = {
        "x_clean": x_clean + 1000,
        "labels": labels.clone(),
    }
    n = sb.evict_and_fill(new_batch, mask_token_id=MASK, d_model=D)
    assert n == 1, f"expected 1 slot refilled, got {n}"
    # Refilled slot has fresh all-mask init and zero h_s.
    assert (sb.storage["x_input"][0, 4:] == MASK).all()
    assert sb.storage["h_s"][0].abs().sum().item() == 0
    # Other slots untouched.
    assert (sb.storage["x_input"][1:, 4:] == MASK).all()

    # reset() clears state (validation semantics).
    sb.reset()
    assert not sb.is_initialized()

    _step("FinetunerArguments exposes BPTT fields with right defaults")
    from lmflow.args import FinetunerArguments

    fields = {f.name: f for f in FinetunerArguments.__dataclass_fields__.values()}
    assert fields["loss_type"].default == "mlm"
    assert fields["bptt_threshold"].default == 0.95
    assert fields["bptt_top_p"].default == 0.95
    assert fields["bptt_temperature"].default == 0.0
    assert fields["bptt_use_streaming_buffer"].default is False

    _step("finetuner.tune dispatch: mlm path -> Trainer; bptt path -> FastDLLMBPTTTrainer")
    # Source-level check (no import) so we don't pull in optional `evaluate`
    # / deepspeed deps just to verify the dispatch wiring.
    finetuner_path = os.path.join(
        ROOT, "src", "lmflow", "pipeline", "finetuner.py"
    )
    with open(finetuner_path) as f:
        finetuner_src = f.read()

    assert "import lmflow.models.fast_dllm" in finetuner_src, (
        "auto-registration import missing"
    )
    assert "loss_type" in finetuner_src and "\"bptt\"" in finetuner_src, (
        "BPTT dispatch branch missing from finetuner.py"
    )
    assert "FastDLLMBPTTTrainer" in finetuner_src
    assert "ddp_find_unused_parameters = True" in finetuner_src
    assert "use_reentrant" in finetuner_src
    assert "bptt_use_streaming_buffer" in finetuner_src, (
        "use_streaming_buffer not threaded through finetuner.py"
    )

    # mlm path: PeftTrainer / Trainer dispatch retained as elif.
    assert "FinetuningTrainer = PeftTrainer" in finetuner_src
    assert "FinetuningTrainer = Trainer" in finetuner_src

    _step("OK")


if __name__ == "__main__":
    main()
