# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# EAGLE 4: Speculative Decoding with Feature-Level Autoregressive Drafter
# Based on EAGLE (arXiv:2401.07851), EAGLE-2 (arXiv:2406.16858), EAGLE-3 (arXiv:2501.01805)

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3 import MLP

from dflash_mlx.model import (
    DFlashAttention,
    DFlashDraftModel,
    DFlashDraftModelArgs,
    DraftRuntimeCapabilities,
)

_EAGLE4_ARCHITECTURE = "Eagle4DraftModel"
_EAGLE4_REQUIRED_FIELDS = frozenset(
    (
        "target_layer_ids",
        "feature_fusion_layers",
        "tree_width",
        "tree_depth",
        "confidence_threshold",
        "use_feature_prediction",
    )
)


@dataclass(frozen=True)
class EagleProposal:
    token_ids: mx.array
    q_token_ids: Optional[mx.array] = None
    q_probs: Optional[mx.array] = None
    tree_nodes: Optional[mx.array] = None
    acceptance_probs: Optional[mx.array] = None
    candidate_ids: Optional[mx.array] = None
    probabilities: Optional[mx.array] = None


def has_eagle4_architecture(config: dict[str, Any]) -> bool:
    return _EAGLE4_ARCHITECTURE in tuple(str(v) for v in config.get("architectures") or ())


def normalize_eagle4_config(config: dict[str, Any]) -> dict[str, Any]:
    data = dict(config)
    if not has_eagle4_architecture(data):
        raise ValueError("EAGLE 4 draft config must declare architectures=['Eagle4DraftModel']")
    eagle_config = data.get("eagle_config")
    if not isinstance(eagle_config, dict):
        raise ValueError("EAGLE 4 draft config requires object eagle_config")
    missing = sorted(_EAGLE4_REQUIRED_FIELDS - set(eagle_config))
    if missing:
        raise ValueError(
            "EAGLE 4 draft config missing eagle_config field(s): " + ", ".join(missing)
        )
    if data.get("is_causal") is not False:
        raise ValueError("EAGLE 4 draft config requires is_causal=false")
    layer_types = tuple(data.get("layer_types") or ())
    if not layer_types:
        raise ValueError("EAGLE 4 draft config requires layer_types")
    if len(layer_types) != int(data["num_hidden_layers"]):
        raise ValueError(
            "EAGLE 4 draft layer_types length must match num_hidden_layers: "
            f"{len(layer_types)} != {int(data['num_hidden_layers'])}"
        )
    if set(layer_types) != {"sliding_attention"}:
        raise ValueError("EAGLE 4 MLX support requires all draft layers to be sliding_attention")
    rope_parameters = data.get("rope_parameters")
    rope_scaling = data.get("rope_scaling")
    rope = rope_parameters or rope_scaling or {}
    if "rope_theta" not in data:
        data["rope_theta"] = rope.get("rope_theta", 10000.0)
    if rope_scaling is None and isinstance(rope_parameters, dict):
        rope_type = (
            rope_parameters.get("type")
            or rope_parameters.get("rope_type")
            or "default"
        )
        data["rope_scaling"] = (
            None
            if rope_type == "default"
            else {
                key: value
                for key, value in rope_parameters.items()
                if key != "rope_theta"
            }
        )
    for key, value in eagle_config.items():
        data[key] = value
    return data


@dataclass
class Eagle4DraftModelArgs(DFlashDraftModelArgs):
    target_layer_ids: tuple[int, ...] = ()
    feature_fusion_layers: tuple[int, ...] = (0, 4, 8)
    tree_width: int = 4
    tree_depth: int = 3
    confidence_threshold: float = 0.8
    use_feature_prediction: bool = True
    conv_kernel_size: int = 2
    conv_group_size: int = 16
    selector_rank: int = 256
    selector_top_k: int = 16
    is_causal: bool = False
    final_logit_softcapping: Optional[float] = None
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "Eagle4DraftModelArgs":
        data = normalize_eagle4_config(params)
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.is_causal is not False:
            raise ValueError("EAGLE 4 draft attention must be non-causal")
        if int(self.conv_kernel_size) != 2:
            raise ValueError("EAGLE 4 MLX support requires two-tap dynamic conv")
        if int(self.conv_group_size) != 16:
            raise ValueError("EAGLE 4 MLX support requires dynamic conv group size 16")
        if int(self.hidden_size) % int(self.conv_group_size) != 0:
            raise ValueError("EAGLE 4 hidden_size must be divisible by conv_group_size")
        if set(self.layer_types) != {"sliding_attention"}:
            raise ValueError("EAGLE 4 MLX support requires all draft layers to be sliding_attention")
        if len(self.target_layer_ids) == 0:
            raise ValueError("EAGLE 4 requires target_layer_ids for feature extraction")
        if len(self.feature_fusion_layers) == 0:
            raise ValueError("EAGLE 4 requires feature_fusion_layers")


@mx.compile
def _grouped_dynamic_convolve(
    hidden: mx.array,
    dynamic: mx.array,
    base: mx.array,
    group_size: int,
) -> mx.array:
    batch, length, hidden_size = hidden.shape
    groups = hidden_size // int(group_size)
    blocks = hidden.reshape(batch, length, groups, int(group_size))
    dynamic = dynamic.reshape(batch, length, int(base.shape[0]), groups, 1)
    if int(base.shape[0]) == 2:
        b0 = base[0].reshape(1, 1, groups, int(group_size)).astype(hidden.dtype)
        out0 = (b0 + dynamic[:, :, 0]) * blocks
        if length <= 1:
            return out0.reshape(hidden.shape)
        b1 = base[1].reshape(1, 1, groups, int(group_size)).astype(hidden.dtype)
        out1 = (b1 + dynamic[:, 1:, 1]) * blocks[:, :-1]
        return mx.concatenate([out0[:, :1], out0[:, 1:] + out1], axis=1).reshape(hidden.shape)
    output = mx.zeros_like(blocks)
    for offset in range(int(base.shape[0])):
        if offset == 0:
            values = blocks
        else:
            values = mx.concatenate(
                (mx.zeros_like(blocks[:, :offset]), blocks[:, :-offset]),
                axis=1,
            )
        kernel = base[offset].reshape(1, 1, groups, int(group_size)).astype(hidden.dtype)
        output = output + kernel * values
        output = output + dynamic[:, :, offset] * values
    return output.reshape(hidden.shape)


_S_EAGLE_CONV_PREP = r"""
    const uint g_idx = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint total_g = 320;
    if (g_idx >= total_g) return;
    const uint d0 = g_idx * 16;
    const bfloat dyn0 = dynamic[((t * 2 + 0) * 2 + 0) * total_g + g_idx];
    const bfloat dyn1 = (t > 0) ? dynamic[((t * 2 + 0) * 2 + 1) * total_g + g_idx] : bfloat(0.0f);
    const device bfloat* x_cur = hidden + t * 5120 + d0;
    const device bfloat* x_prev = (t > 0) ? (hidden + (t - 1) * 5120 + d0) : x_cur;
    const device bfloat* b0 = base + 0 * 5120 + d0;
    const device bfloat* b1 = base + 1 * 5120 + d0;
    device bfloat* out_ptr = out + t * 5120 + d0;
    #pragma unroll
    for (uint c = 0; c < 16; ++c) {
        bfloat v0 = (b0[c] + dyn0) * x_cur[c];
        if (t > 0) {
            bfloat v1 = (b1[c] + dyn1) * x_prev[c];
            v0 = bfloat(float(v0) + float(v1));
        }
        out_ptr[c] = v0;
    }
"""

_S_EAGLE_CONV_FIN_RES = r"""
    const uint g_idx = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    const uint total_g = 320;
    if (g_idx >= total_g) return;
    const uint d0 = g_idx * 16;
    const bfloat dyn0 = dynamic[((t * 2 + 1) * 2 + 0) * total_g + g_idx];
    const bfloat dyn1 = (t > 0) ? dynamic[((t * 2 + 1) * 2 + 1) * total_g + g_idx] : bfloat(0.0f);
    const device bfloat* x_cur = hidden + t * 5120 + d0;
    const device bfloat* x_prev = (t > 0) ? (hidden + (t - 1) * 5120 + d0) : x_cur;
    const device bfloat* b0 = base + 2 * 5120 + d0;
    const device bfloat* b1 = base + 3 * 5120 + d0;
    const device bfloat* res = residual + t * 5120 + d0;
    device bfloat* out_ptr = out + t * 5120 + d0;
    #pragma unroll
    for (uint c = 0; c < 16; ++c) {
        bfloat v0 = (b0[c] + dyn0) * x_cur[c];
        if (t > 0) {
            bfloat v1 = (b1[c] + dyn1) * x_prev[c];
            v0 = bfloat(float(v0) + float(v1));
        }
        out_ptr[c] = bfloat(float(res[c]) + float(v0));
    }
"""

_kernel_eagle_conv_prep = None
_kernel_eagle_conv_fin_res = None


def _get_eagle_conv_kernels():
    global _kernel_eagle_conv_prep, _kernel_eagle_conv_fin_res
    if _kernel_eagle_conv_prep is None:
        _kernel_eagle_conv_prep = mx.fast.metal_kernel(
            name="eagle4_conv_prep",
            input_names=["hidden", "dynamic", "base"],
            output_names=["out"],
            source=_S_EAGLE_CONV_PREP,
        )
        _kernel_eagle_conv_fin_res = mx.fast.metal_kernel(
            name="eagle4_conv_fin_res",
            input_names=["hidden", "dynamic", "base", "residual"],
            output_names=["out"],
            source=_S_EAGLE_CONV_FIN_RES,
        )
    return _kernel_eagle_conv_prep, _kernel_eagle_conv_fin_res


class GroupedDynamicCausalConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, group_size: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.group_size = int(group_size)
        groups = int(hidden_size) // self.group_size
        self.base_kernel = mx.zeros(
            (2, self.kernel_size, int(hidden_size)),
            dtype=mx.bfloat16,
        )
        self.kernel_projection = nn.Linear(
            int(hidden_size),
            2 * self.kernel_size * groups,
            bias=False,
        )

    def _can_use_metal_fastpath(self, hidden: mx.array) -> bool:
        return (
            hidden.ndim == 3
            and int(hidden.shape[0]) == 1
            and int(hidden.shape[1]) > 0
            and int(hidden.shape[2]) == 5120
            and self.kernel_size == 2
            and self.group_size == 16
            and hidden.dtype == mx.bfloat16
            and self.base_kernel.dtype == mx.bfloat16
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
        )

    def prepare_fast(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        k_prep, _ = _get_eagle_conv_kernels()
        L = int(hidden.shape[1])
        dynamic_raw = self.kernel_projection(hidden)
        out = k_prep(
            inputs=[hidden, dynamic_raw, self.base_kernel],
            grid=(320, L, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(1, L, 5120)],
            output_dtypes=[mx.bfloat16],
        )[0]
        return out, dynamic_raw

    def finish_residual_fast(
        self,
        hidden: mx.array,
        dynamic_raw: mx.array,
        residual: mx.array,
    ) -> mx.array:
        _, k_fin = _get_eagle_conv_kernels()
        L = int(hidden.shape[1])
        return k_fin(
            inputs=[hidden, dynamic_raw, self.base_kernel, residual],
            grid=(320, L, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(1, L, 5120)],
            output_dtypes=[mx.bfloat16],
        )[0]

    def prepare(self, hidden: mx.array) -> tuple[mx.array, mx.array]:
        groups = int(hidden.shape[-1]) // self.group_size
        dynamic = self.kernel_projection(hidden).reshape(
            *hidden.shape[:-1],
            2,
            self.kernel_size,
            groups,
        )
        return (
            _grouped_dynamic_convolve(
                hidden,
                dynamic[..., 0, :, :],
                self.base_kernel[0],
                self.group_size,
            ),
            dynamic[..., 1, :, :],
        )

    def finish(self, hidden: mx.array, dynamic: mx.array) -> mx.array:
        return _grouped_dynamic_convolve(
            hidden,
            dynamic,
            self.base_kernel[1],
            self.group_size,
        )


class FeatureFusionMLP(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, fusion_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.fusion_dim = fusion_dim
        self.layer_projections = nn.ModuleList([
            nn.Linear(hidden_size, fusion_dim, bias=False)
            for _ in range(num_layers)
        ])
        self.fusion_norm = nn.RMSNorm(fusion_dim, eps=1e-6)
        self.fusion_mlp = MLP(fusion_dim, fusion_dim * 4)
        self.output_proj = nn.Linear(fusion_dim, hidden_size, bias=False)

    def __call__(self, layer_outputs: list[mx.array]) -> mx.array:
        projected = [proj(out) for proj, out in zip(self.layer_projections, layer_outputs)]
        fused = sum(projected) / len(projected)
        fused = self.fusion_norm(fused)
        fused = self.fusion_mlp(fused)
        return self.output_proj(fused)


class Eagle4Attention(DFlashAttention):
    def __init__(self, args: Eagle4DraftModelArgs, layer_idx: int):
        super().__init__(args, layer_idx)
        self.is_causal = bool(args.is_causal)

    def _attention_mask(
        self,
        *,
        block_len: int,
        query_offset: int,
        key_len: int,
        key_positions: Optional[mx.array] = None,
    ) -> Optional[mx.array]:
        if self.sliding_window is None and not self.is_causal:
            return None
        if (
            self.sliding_window is not None
            and not self.is_causal
            and int(query_offset) + int(block_len) <= int(self.sliding_window)
        ):
            return None
        query_positions = mx.arange(
            query_offset,
            query_offset + int(block_len),
            dtype=mx.int32,
        )
        if key_positions is None:
            key_start = query_offset + int(block_len) - int(key_len)
            key_positions = mx.arange(key_start, key_start + int(key_len), dtype=mx.int32)
        if self.sliding_window is None:
            return key_positions[None, :] <= query_positions[:, None]
        context = (key_positions[None, :] < int(query_offset)) & (
            query_positions[:, None] - key_positions[None, :] < int(self.sliding_window)
        )
        block = key_positions[None, :] >= int(query_offset)
        if self.is_causal:
            block = block & (key_positions[None, :] <= query_positions[:, None])
        return context | block


class Eagle4DecoderLayer(nn.Module):
    def __init__(self, args: Eagle4DraftModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Eagle4Attention(args, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.attention_conv = GroupedDynamicCausalConv(
            args.hidden_size,
            args.conv_kernel_size,
            args.conv_group_size,
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            args.hidden_size,
            args.conv_kernel_size,
            args.conv_group_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        target_hidden: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.attention_conv._can_use_metal_fastpath(hidden_states):
            residual = hidden_states
            hidden_states, dynamic = self.attention_conv.prepare_fast(
                self.input_layernorm(hidden_states)
            )
            attn_out = self.self_attn(
                hidden_states,
                target_hidden=target_hidden,
                cache=cache,
            )
            hidden_states = self.attention_conv.finish_residual_fast(
                attn_out,
                dynamic,
                residual,
            )
            residual = hidden_states
            hidden_states, dynamic = self.mlp_conv.prepare_fast(
                self.post_attention_layernorm(hidden_states)
            )
            return self.mlp_conv.finish_residual_fast(
                self.mlp(hidden_states),
                dynamic,
                residual,
            )
        residual = hidden_states
        hidden_states, dynamic = self.attention_conv.prepare(
            self.input_layernorm(hidden_states)
        )
        hidden_states = residual + self.attention_conv.finish(
            self.self_attn(
                hidden_states,
                target_hidden=target_hidden,
                cache=cache,
            ),
            dynamic,
        )
        residual = hidden_states
        hidden_states, dynamic = self.mlp_conv.prepare(
            self.post_attention_layernorm(hidden_states)
        )
        return residual + self.mlp_conv.finish(self.mlp(hidden_states), dynamic)

    def advance_projected_context_cache(
        self,
        *,
        target_hidden: mx.array,
        cache: Any,
    ) -> None:
        self.self_attn.append_projected_context_cache(
            target_hidden=target_hidden,
            cache=cache,
        )


class FeaturePredictionHead(nn.Module):
    def __init__(self, hidden_size: int, target_hidden_size: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * 2, target_hidden_size, bias=False),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.predictor(x)


class DynamicDraftTree(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        tree_width: int = 4,
        tree_depth: int = 3,
        confidence_threshold: float = 0.8,
    ):
        super().__init__()
        self.tree_width = tree_width
        self.tree_depth = tree_depth
        self.confidence_threshold = confidence_threshold
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1, bias=False),
            nn.Sigmoid(),
        )

    def build_tree(
        self,
        draft_hidden: mx.array,
        target_lm_head: nn.Module,
        temperature: float = 0.0,
    ) -> tuple[mx.array, mx.array, mx.array]:
        B, M, H = draft_hidden.shape
        tree_tokens = []
        tree_probs = []
        tree_confidences = []

        current_hidden = draft_hidden[:, 0:1, :]
        for depth in range(self.tree_depth):
            level_tokens = []
            level_probs = []
            level_confidences = []

            for width_idx in range(self.tree_width):
                logits = self.lm_head(current_hidden)
                if temperature > 0:
                    probs = mx.softmax(logits.astype(mx.float32) / temperature, axis=-1)
                    if temperature > 0:
                        next_token = mx.random.categorical(mx.log(probs))
                    else:
                        next_token = mx.argmax(probs, axis=-1)
                else:
                    next_token = mx.argmax(logits, axis=-1)
                    probs = mx.softmax(logits.astype(mx.float32), axis=-1)

                confidence = self.confidence_head(current_hidden)
                level_tokens.append(next_token)
                level_probs.append(probs)
                level_confidences.append(confidence)

            level_tokens = mx.stack(level_tokens, axis=1)
            level_probs = mx.stack(level_probs, axis=1)
            level_confidences = mx.stack(level_confidences, axis=1)

            tree_tokens.append(level_tokens)
            tree_probs.append(level_probs)
            tree_confidences.append(level_confidences)

            best_idx = mx.argmax(level_confidences, axis=1)
            current_hidden = mx.take_along_axis(
                current_hidden.expand_dims(1).repeat(self.tree_width, axis=1),
                best_idx[:, None, None, None].repeat(H, axis=-1),
                axis=1,
            )[:, 0:1, :]

            if mx.all(level_confidences.squeeze(-1) < self.confidence_threshold):
                break

        tree_tokens = mx.concatenate(tree_tokens, axis=1)
        tree_probs = mx.concatenate(tree_probs, axis=1)
        tree_confidences = mx.concatenate(tree_confidences, axis=1)

        return tree_tokens, tree_probs, tree_confidences

    def select_best_path(
        self,
        tree_tokens: mx.array,
        tree_probs: mx.array,
        tree_confidences: mx.array,
    ) -> mx.array:
        B, W, V = tree_probs.shape
        max_conf_idx = mx.argmax(tree_confidences, axis=1)
        best_tokens = mx.take_along_axis(
            tree_tokens,
            max_conf_idx[:, None, None].repeat(V, axis=-1),
            axis=1,
        )[:, 0, :]
        return best_tokens


class Eagle4CandidateSelector(nn.Module):
    def __init__(self, args: Eagle4DraftModelArgs):
        super().__init__()
        self.top_k = int(args.selector_top_k)
        self.predecessor_codebook = nn.Embedding(args.vocab_size, int(args.selector_rank))
        self.successor_codebook = nn.Embedding(args.vocab_size, int(args.selector_rank))
        self.hidden_projection = nn.Linear(args.hidden_size, int(args.selector_rank), bias=False)

    def _edge_scores(
        self,
        predecessor_ids: mx.array,
        successor_vectors: mx.array,
        hidden: mx.array,
    ) -> mx.array:
        return mx.sum(
            self.predecessor_codebook(predecessor_ids)[:, :, None]
            * hidden[:, None, None]
            * successor_vectors[:, None],
            axis=-1,
        )

    def _select_ancestral(
        self,
        *,
        candidates: mx.array,
        unary: mx.array,
        hidden: mx.array,
        anchor_ids: mx.array,
        successors: mx.array,
        temperature: float,
        capture_q: bool,
    ):
        predecessor = anchor_ids
        path = []
        q_rows = []
        for position in range(int(hidden.shape[1])):
            edges = self._edge_scores(
                predecessor[:, None],
                successors[:, position],
                hidden[:, position],
            )[:, 0]
            scores = unary[:, position] + edges
            if float(temperature) > 0 or capture_q:
                q = mx.softmax(scores.astype(mx.float32) / float(temperature), axis=-1)
                q_rows.append(q)
            if float(temperature) > 0:
                selected = mx.random.categorical(mx.log(q))
            else:
                selected = mx.argmax(scores, axis=-1)
            predecessor = mx.take_along_axis(
                candidates[:, position],
                selected[:, None],
                axis=-1,
            )[:, 0]
            path.append(predecessor)
        sparse_q = mx.stack(q_rows, axis=1) if q_rows else None
        return EagleProposal(
            token_ids=mx.stack(path, axis=1),
            candidate_ids=candidates,
            probabilities=sparse_q,
        )

    def select(
        self,
        hidden: mx.array,
        logits: mx.array,
        anchor_ids: mx.array,
        temperature: float,
        capture_q: bool = False,
    ) -> EagleProposal:
        if len(hidden.shape) != 3 or len(logits.shape) != 3:
            raise ValueError("EAGLE 4 selector requires 3D hidden and logits")
        if tuple(hidden.shape[:2]) != tuple(logits.shape[:2]):
            raise ValueError("EAGLE 4 selector hidden/logits batch and length must match")
        if int(logits.shape[-1]) < self.top_k:
            raise ValueError("EAGLE 4 selector vocabulary must be at least top_k")
        anchor_ids = anchor_ids.reshape(-1)
        if int(anchor_ids.shape[0]) != int(hidden.shape[0]):
            raise ValueError("EAGLE 4 selector anchor batch must match hidden batch")

        candidates = mx.argpartition(logits, -self.top_k, axis=-1)[..., -self.top_k:]
        unary = mx.take_along_axis(logits, candidates, axis=-1)
        hidden = self.hidden_projection(hidden)
        successors = self.successor_codebook(candidates)
        return self._select_ancestral(
            candidates=candidates,
            unary=unary,
            hidden=hidden,
            anchor_ids=anchor_ids,
            successors=successors,
            temperature=float(temperature),
            capture_q=bool(capture_q),
        )


class Eagle4DraftModel(DFlashDraftModel):
    def __init__(self, args: Eagle4DraftModelArgs):
        super().__init__(args)
        self.model_type = "eagle4_qwen3"
        self.args = args
        self.candidate_selector = Eagle4CandidateSelector(args)
        self.feature_fusion = FeatureFusionMLP(
            args.hidden_size,
            len(args.target_layer_ids),
            args.hidden_size,
        )
        if args.use_feature_prediction:
            self.feature_prediction = FeaturePredictionHead(
                args.hidden_size,
                args.hidden_size,
            )
        self.draft_tree = DynamicDraftTree(
            args.hidden_size,
            args.vocab_size,
            args.tree_width,
            args.tree_depth,
            args.confidence_threshold,
        )
        self.capabilities = DraftRuntimeCapabilities(
            default_block_tokens=min(8, int(args.block_size)),
            max_block_tokens=int(args.block_size),
            supports_copyspec=True,
            supports_ddtree=True,
            supports_early_rollback_launch=True,
        )

    def _build_layers(self, args: DFlashDraftModelArgs) -> list[nn.Module]:
        if not isinstance(args, Eagle4DraftModelArgs):
            raise TypeError("EAGLE 4 layer construction requires Eagle4DraftModelArgs")
        return [
            Eagle4DecoderLayer(args, layer_idx)
            for layer_idx in range(args.num_hidden_layers)
        ]

    def bind_target_model(self, target_model: Any, *, target_ops: Any) -> None:
        super().bind_target_model(target_model, target_ops=target_ops)
        self.embed_scale *= float(self.args.input_embedding_scale)
        self.target_ops = target_ops
        self.target_model = target_model

    def extract_target_features(self, target_model: Any, input_ids: mx.array) -> list[mx.array]:
        layer_outputs = []
        target_ops = self.target_ops
        hidden = target_ops.embed_tokens(target_model)(input_ids[None])
        for layer_idx, layer in enumerate(target_model.model.layers):
            hidden = layer(hidden)
            if layer_idx in self.args.target_layer_ids:
                layer_outputs.append(hidden[:, -1:, :])
        return layer_outputs

    def compute_fused_features(self, layer_outputs: list[mx.array]) -> mx.array:
        return self.feature_fusion(layer_outputs)

    def compute_logits(self, logits: mx.array) -> mx.array:
        logits = logits * float(self.args.output_multiplier)
        cap = self.args.final_logit_softcapping
        if cap is not None and float(cap) > 0:
            cap_value = float(cap)
            logits = mx.tanh(logits / cap_value) * cap_value
        return logits

    def select_proposal(
        self,
        *,
        draft_hidden: mx.array,
        logits: mx.array,
        anchor_ids: mx.array,
        temperature: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        capture_q: bool = False,
    ) -> EagleProposal:
        del top_p, min_p, top_k
        proposal = self.candidate_selector.select(
            draft_hidden,
            self.compute_logits(logits),
            anchor_ids,
            temperature,
            capture_q=capture_q,
        )
        return EagleProposal(
            token_ids=proposal.token_ids.squeeze(0),
            q_token_ids=(
                proposal.candidate_ids.squeeze(0)
                if proposal.probabilities is not None
                else None
            ),
            q_probs=(
                proposal.probabilities.squeeze(0)
                if proposal.probabilities is not None
                else None
            ),
        )

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        remapped = dict(weights)
        for name in ("predecessor_codebook", "successor_codebook"):
            old_key = f"candidate_selector.{name}"
            new_key = f"{old_key}.weight"
            if old_key not in remapped:
                continue
            if new_key in remapped:
                raise ValueError(
                    f"EAGLE 4 safetensors contain both {old_key!r} and {new_key!r}"
                )
            remapped[new_key] = remapped.pop(old_key)
        return remapped