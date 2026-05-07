# v2
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
from jaxtyping import Float, Integer, Bool
from torch import Tensor as TT

from xlm.modules.rotary_transformer import (
    RotaryTransformerFinalLayer,
    RotaryTransformerLayer,
    RotaryTransformerLayerList,
    RotaryEmbedding,
)

from xlm.utils.rank_zero import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)

########################################################
# region: Rotary Transformer


def _get_activation(activation: str) -> nn.Module:
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "relu":
        return nn.ReLU()
    raise ValueError(f"Activation {activation} not supported")


def _zero_init_linear(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class BottleneckMLPBridge(nn.Module):
    """Bottleneck carry transform with the same MLP shape as RotaryTransformerLayer."""

    def __init__(
        self,
        d_model: int,
        bridge_dim: int,
        activation: str = "relu",
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        zero_init_output: bool = True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.fc1 = nn.Linear(d_model, bridge_dim, bias=True)
        self.activation = _get_activation(activation)
        self.fc2 = nn.Linear(bridge_dim, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        if zero_init_output:
            _zero_init_linear(self.fc2)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        x = self.fc1(self.norm(h_t))
        x = self.activation(x)
        x = self.fc2(x)
        return self.dropout(x)


class CrossAttentionBridge(nn.Module):
    """Bottleneck cross-attention bridge from token embeddings to carried state."""

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int = 128,
        n_heads: int = 4,
        n_kv_heads: int = 4,
        dim_feedforward: Optional[int] = None,
        activation: str = "relu",
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        zero_init_output: bool = True,
    ):
        super().__init__()
        if bottleneck_dim % n_heads != 0:
            raise ValueError("bottleneck_dim must be divisible by n_heads")
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = bottleneck_dim // n_heads

        self.w_down_x = nn.Linear(d_model, bottleneck_dim, bias=True)
        self.w_down_h = nn.Linear(d_model, bottleneck_dim, bias=True)

        self.attn_norm_x = nn.LayerNorm(bottleneck_dim, eps=layer_norm_eps)
        self.attn_norm_h = nn.LayerNorm(bottleneck_dim, eps=layer_norm_eps)
        self.w_q = nn.Linear(bottleneck_dim, bottleneck_dim, bias=False)
        self.w_k = nn.Linear(
            bottleneck_dim, n_kv_heads * self.head_dim, bias=False
        )
        self.w_v = nn.Linear(
            bottleneck_dim, n_kv_heads * self.head_dim, bias=False
        )
        self.w_out = nn.Linear(bottleneck_dim, bottleneck_dim, bias=True)

        mlp_dim = dim_feedforward or bottleneck_dim
        self.mlp_norm = nn.LayerNorm(bottleneck_dim, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim, mlp_dim, bias=True),
            _get_activation(activation),
            nn.Linear(mlp_dim, bottleneck_dim, bias=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.w_up = nn.Linear(bottleneck_dim, d_model, bias=True)
        if zero_init_output:
            _zero_init_linear(self.w_up)

    def forward(self, token_emb: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = token_emb.shape
        _, carry_len, _ = h_t.shape

        x_b = self.w_down_x(token_emb)
        h_b = self.w_down_h(h_t)
        x_norm = self.attn_norm_x(x_b)
        h_norm = self.attn_norm_h(h_b)

        q = self.w_q(x_norm).view(
            batch_size, seq_len, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = self.w_k(h_norm).view(
            batch_size, carry_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.w_v(h_norm).view(
            batch_size, carry_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        if self.n_heads != self.n_kv_heads:
            num_groups = self.n_heads // self.n_kv_heads
            q = q.reshape(
                batch_size, self.n_kv_heads, num_groups, seq_len, self.head_dim
            )
            k = k.unsqueeze(2)
            v = v.unsqueeze(2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            attn = attn.reshape(
                batch_size, self.n_heads, seq_len, self.head_dim
            )
        else:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, -1)
        x_b = x_b + self.dropout(self.w_out(attn))
        x_b = x_b + self.dropout(self.mlp(self.mlp_norm(x_b)))
        return self.dropout(self.w_up(x_b))


class RotaryTransformerModel(torch.nn.Module):
    "Rotary embedding based transformer decoder."

    def __init__(
        self,
        num_embeddings: int,  # vocab plus mask and padding other special tokens
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
        final_layer_without_normalization: bool = False,
        weight_decay_all_params: bool = False,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.weight_decay_all_params = weight_decay_all_params
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.d_model = d_model
        self.embed_tokens = nn.Embedding(
            num_embeddings, d_model, padding_idx=padding_idx
        )
        self.dim_feedforward = dim_feedforward or 4 * d_model
        encoder_layer = RotaryTransformerLayer(
            d_model,
            nhead,
            self.dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            force_flash_attn=force_flash_attn,
        )
        self.max_length = max_length
        self.encoder = RotaryTransformerLayerList.from_layer(
            encoder_layer,
            num_layers,
            RotaryEmbedding(
                rotary_emb_dim, head_first=True, cache_size=max_length
            ),
        )
        self.output_layer = RotaryTransformerFinalLayer(
            d_model,
            num_embeddings,
            layer_norm_eps,
            use_final_layer_norm=not final_layer_without_normalization,
            zero_init=False,  # zero init important for mdlm, mlm?
        )
        if tie_embeddings:
            # nn.Embedding inits with N(0, 1); a fresh LM-head Linear(d_model, V)
            # would init with std ≈ 1/sqrt(d_model). If we tie without re-initing,
            # the LM head ends up ~sqrt(d_model)× too large, producing huge initial
            # logits and forcing thousands of steps of "scale recovery" before the
            # model can actually learn. Re-init at LM-head scale before tying.
            nn.init.normal_(
                self.embed_tokens.weight, mean=0.0, std=d_model ** -0.5
            )
            with torch.no_grad():
                self.embed_tokens.weight[padding_idx].zero_()
            # output_layer.linear.weight is (V, d) — same shape as embed_tokens.weight
            self.output_layer.linear.weight = self.embed_tokens.weight

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
    ) -> Float[TT, " *batch seq_len vocab_size"]:
        """
        Args:
            x_t: The input tokens of shape (*batch, seq_len)
            attention_mask: The attention mask of shape (*batch, seq_len), which is True for non-padding tokens.
            positions: The positions of the tokens of shape (*batch, seq_len)
        """
        attention_mask = torch.ones_like(x_t, dtype=torch.bool, device=x_t.device)
        positions = (attention_mask.cumsum(dim=1) - 1)

        x = self.embed_tokens(x_t)  # shape (batch_size, seq_len, d_model)

        for block in self.encoder:
            x = block(x, attention_mask, positions=positions)

        vocab_logits = self.output_layer(
            x,
        )  # shape (batch_size, seq_len, vocab_size)
        return vocab_logits

    def get_named_params_for_weight_decay(self):
        # all parameters except biases and layer-norm parameters (unless weight_decay_all_params)
        for name, param in self.named_parameters():
            if not self.weight_decay_all_params and ("bias" in name or "norm" in name):
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        # biases and layer-norm parameters (empty when weight_decay_all_params is True)
        if self.weight_decay_all_params:
            return
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                yield (name, param)


class RotaryTransformerLoopholingModel(torch.nn.Module):
    """Rotary transformer with loopholing: accepts h_t and returns (logits, h_s).

    Injection: x = token_emb + LayerNorm(carry_transform(h_t)) before layer 0.
    The base class uses an identity carry transform so the injection reduces to
    ``token_emb + LayerNorm(h_t)``. Subclasses (MLP, CAB) override
    ``_carry_transform`` to learn a richer carry path; the shared
    ``h_t_layer_norm`` always normalizes the result before adding to ``token_emb``.

    Readout: h_s is the hidden state at `loophole_layer` (default -1 = last layer,
    supports negative indexing). vocab_logits always come from the final layer.
    """

    def __init__(
        self,
        num_embeddings: int,
        d_model: int,
        num_layers: int,
        nhead: int,
        padding_idx: int = 0,
        mask_idx: int = 1,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        rotary_emb_dim: int = 64,
        max_length: int = 1024,
        force_flash_attn: bool = False,
        use_flex_attn: bool = False,
        final_layer_without_normalization: bool = False,
        weight_decay_all_params: bool = False,
        loophole_layer: int = -1,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.weight_decay_all_params = weight_decay_all_params
        self.padding_idx = padding_idx
        self.mask_idx = mask_idx
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_flex_attn = use_flex_attn
        # Resolve negative indices once so forward doesn't recompute. -1 → last layer.
        self._loophole_layer_idx = loophole_layer % num_layers
        self.embed_tokens = nn.Embedding(
            num_embeddings, d_model, padding_idx=padding_idx
        )
        self.h_t_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dim_feedforward = dim_feedforward or 4 * d_model
        encoder_layer = RotaryTransformerLayer(
            d_model,
            nhead,
            self.dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            force_flash_attn=force_flash_attn,
        )
        self.max_length = max_length
        self.encoder = RotaryTransformerLayerList.from_layer(
            encoder_layer,
            num_layers,
            RotaryEmbedding(
                rotary_emb_dim, head_first=True, cache_size=max_length
            ),
        )
        self.output_layer = RotaryTransformerFinalLayer(
            d_model,
            num_embeddings,
            layer_norm_eps,
            use_final_layer_norm=not final_layer_without_normalization,
            zero_init=False,
        )
        if tie_embeddings:
            # nn.Embedding inits with N(0, 1); a fresh LM-head Linear(d_model, V)
            # would init with std ≈ 1/sqrt(d_model). If we tie without re-initing,
            # the LM head ends up ~sqrt(d_model)× too large, producing huge initial
            # logits and forcing thousands of steps of "scale recovery" before the
            # model can actually learn. Re-init at LM-head scale before tying.
            nn.init.normal_(
                self.embed_tokens.weight, mean=0.0, std=d_model ** -0.5
            )
            with torch.no_grad():
                self.embed_tokens.weight[padding_idx].zero_()
            # output_layer.linear.weight is (V, d) — same shape as embed_tokens.weight
            self.output_layer.linear.weight = self.embed_tokens.weight

    def forward(
        self,
        x_t: Integer[TT, " *batch seq_len"],
        h_t: Float[TT, " *batch seq_len d_model"],
        positions: Optional[Integer[TT, " *batch seq_len"]] = None,
        block_mask: Any = None,
    ) -> Tuple[Float[TT, " *batch seq_len vocab_size"], Float[TT, " *batch seq_len d_model"]]:
        """
        Same flex contract as ``RotaryTransformerMLMModel`` in xlm-core: when
        ``use_flex_attn=True`` and both ``positions`` and ``block_mask`` are provided (training /
        packed batches), uses FlexAttention. If either is omitted (e.g. unconditional predictor),
        falls back to dense full attention with monotonic positions.

        When ``use_flex_attn=False``, always uses dense full attention and monotonic positions if
        ``positions`` is None.
        """
        attention_mask: Optional[torch.Tensor] = None

        use_flex = (
            self.use_flex_attn and positions is not None and block_mask is not None
        )
        if use_flex:
            attention_mask = None
        else:
            attention_mask = torch.ones_like(x_t, dtype=torch.bool)
            if positions is None:
                positions = (attention_mask.cumsum(dim=1) - 1)

        token_emb = self.embed_tokens(x_t)  # (*batch, seq_len, d_model)
        x = token_emb + self.h_t_layer_norm(self._carry_transform(token_emb, h_t))

        flex_block = block_mask if use_flex else None
        h_s = None
        for i, block in enumerate(self.encoder):
            x = block(x, attention_mask, positions=positions, block_mask=flex_block)
            if i == self._loophole_layer_idx:
                h_s = x

        vocab_logits = self.output_layer(x)
        return vocab_logits, h_s

    def _carry_transform(
        self,
        token_emb: Float[TT, " *batch seq_len d_model"],
        h_t: Float[TT, " *batch seq_len d_model"],
    ) -> Float[TT, " *batch seq_len d_model"]:
        del token_emb
        return h_t

    def get_named_params_for_weight_decay(self):
        for name, param in self.named_parameters():
            if not self.weight_decay_all_params and ("bias" in name or "norm" in name):
                continue
            yield (name, param)

    def get_named_params_for_no_weight_decay(self):
        if self.weight_decay_all_params:
            return
        for name, param in self.named_parameters():
            if "bias" in name or "norm" in name:
                yield (name, param)


class RotaryTransformerLoopholingMLPModel(RotaryTransformerLoopholingModel):
    """Loopholing transformer with a bottleneck MLP carry bridge."""

    def __init__(
        self,
        *args: Any,
        bridge_dim: int = 128,
        bridge_zero_init_output: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.h_t_bridge = BottleneckMLPBridge(
            d_model=self.d_model,
            bridge_dim=bridge_dim,
            activation=kwargs.get("activation", "relu"),
            dropout=kwargs.get("dropout", 0.1),
            layer_norm_eps=kwargs.get("layer_norm_eps", 1e-5),
            zero_init_output=bridge_zero_init_output,
        )

    def _carry_transform(
        self,
        token_emb: Float[TT, " *batch seq_len d_model"],
        h_t: Float[TT, " *batch seq_len d_model"],
    ) -> Float[TT, " *batch seq_len d_model"]:
        del token_emb
        return self.h_t_bridge(h_t)


class RotaryTransformerLoopholingCABModel(RotaryTransformerLoopholingModel):
    """Loopholing transformer with a bottleneck cross-attention carry bridge."""

    def __init__(
        self,
        *args: Any,
        cab_bottleneck_dim: int = 128,
        cab_n_heads: int = 4,
        cab_n_kv_heads: int = 4,
        cab_dim_feedforward: Optional[int] = None,
        cab_zero_init_output: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.cross_attention_bridge = CrossAttentionBridge(
            d_model=self.d_model,
            bottleneck_dim=cab_bottleneck_dim,
            n_heads=cab_n_heads,
            n_kv_heads=cab_n_kv_heads,
            dim_feedforward=cab_dim_feedforward,
            activation=kwargs.get("activation", "relu"),
            dropout=kwargs.get("dropout", 0.1),
            layer_norm_eps=kwargs.get("layer_norm_eps", 1e-5),
            zero_init_output=cab_zero_init_output,
        )

    def _carry_transform(
        self,
        token_emb: Float[TT, " *batch seq_len d_model"],
        h_t: Float[TT, " *batch seq_len d_model"],
    ) -> Float[TT, " *batch seq_len d_model"]:
        return self.cross_attention_bridge(token_emb, h_t)
