"""
modeling_qwen2_5_vl.py

Slim model-structure file for Qwen2.5-VL with DynVid compression.
All compression algorithms live in compression_unified.py.
"""

from typing import Callable, Optional, Union, List, Tuple
import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, is_flash_attn_available
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    Qwen2_5_VLAttention,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLTextModel,
    Qwen2_5_VLModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModelOutputWithPast,
    repeat_kv,
)
try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
except ImportError:
    Qwen2_5_VLDecoderLayer = None

from .compression_unified import (
    _td_logger,
    dynvid_compression_qwen,
    _dyseg_group_frames,
    _compute_group_budget,
)

# ============================================================================
# TextModel forward (with sliding window + soft/hard pruning, no DeepStack)
# ============================================================================


def Qwen2_5_VLDecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    """Qwen2.5-VL DecoderLayer forward, pass through output_attentions and return (h, attn_weights).

    The official latest DecoderLayer.forward does not accept output_attentions nor return attn_weights,
    we must monkey-patch it to support attention collection needed for LLM internal pruning.
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention - pass output_attentions + cache_position through to self_attn
    attn_outputs = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    if isinstance(attn_outputs, tuple):
        hidden_states = attn_outputs[0]
        attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
    else:
        hidden_states = attn_outputs
        attn_weights = None

    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return (hidden_states, attn_weights)



def _query_prune_qwen(
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    attentions_list: List[torch.Tensor],
    cache_position: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    text_position_ids: Optional[torch.Tensor],
    position_embeddings,
    visual_token_range: Tuple[int, int],
    target_budget: int,
    prune_method: str = "text_token",
):
    """Query guided pruning.

    Inputs
        hidden_states:       (B, S, D)
        causal_mask:         (B, 1, q, k) or None - full_attention mask
        attentions_list:     list of (B, H, q_eff, k_eff) attention weights
                             collected from observe layers (including qp_layer-1)
        cache_position:      (S,) or None
        position_ids:        (3, B, S) - M-RoPE 3D positions
        text_position_ids:   (B, S) or None
        position_embeddings: (cos, sin), each (B, S, head_dim) or (S, head_dim)
        visual_token_range:  (v_start, v_end)
        target_budget:       number of visual tokens to keep
        prune_method:        "text_token" | "all_token"

    Returns
        (hidden_states, causal_mask, position_ids, text_position_ids,
         cache_position, position_embeddings, keep_full_idx, keep_visual_idx)
    """
    B, S, D = hidden_states.shape
    v_start, v_end = visual_token_range
    num_visual = v_end - v_start

    # ---- 1. Aggregate multi-layer attention to (B, V) ----
    # Each layer attn shape: (B, H, q_eff, k_eff)
    # k_eff = S (full K), we only take the V segment
    per_layer_scores = []
    for attn in attentions_list:
        if attn is None:
            continue
        # Take visual segment of K
        attn_v = attn[..., v_start:v_end]  # (B, H, q_eff, V)

        # Aggregate query dimension by method
        q_eff = attn_v.shape[2]
        if prune_method == "text_token":
            # q_eff is the number of text tokens (prefix + suffix)
            # max over query -> maximum attention each visual token receives from all text tokens
            scores_layer = attn_v.max(dim=2).values.mean(dim=1)  # (B, V)
        else:  # all_token
            scores_layer = attn_v.max(dim=2).values.mean(dim=1)  # (B, V)
        per_layer_scores.append(scores_layer)

    if len(per_layer_scores) == 0:
        # No available attention, skip pruning
        return (hidden_states, causal_mask, position_ids, text_position_ids,
                cache_position, position_embeddings, None, None)

    # Multi-layer mean
    scores = torch.stack(per_layer_scores, dim=0).mean(dim=0)  # (B, V)

    # ---- 2. Top-k selection ----
    k = min(target_budget, num_visual)
    _, top_idx = scores.topk(k, dim=1)
    top_idx = top_idx.sort(dim=1).values  # (B, k)
    keep_visual_idx = top_idx[0]  # (k,) - relative index within V segment

    # ---- 3. Construct global keep_full_idx ----
    device = hidden_states.device
    keep_full_idx = torch.cat([
        torch.arange(0, v_start, device=device, dtype=torch.long),
        v_start + keep_visual_idx,
        torch.arange(v_end, S, device=device, dtype=torch.long),
    ])  # (S - num_pruned,)

    # ---- 4. Synchronized slicing ----
    new_hidden = hidden_states[:, keep_full_idx, :].contiguous()

    # position_ids: (3, B, S) -> (3, B, S_new)
    new_position_ids = position_ids[..., keep_full_idx].contiguous() if position_ids is not None else None

    # text_position_ids: (B, S)
    new_text_position_ids = text_position_ids[..., keep_full_idx].contiguous() if text_position_ids is not None else None

    # cache_position: (S,)
    new_cache_position = cache_position[keep_full_idx].contiguous() if cache_position is not None else None

    # position_embeddings: (cos, sin)
    if position_embeddings is not None:
        cos, sin = position_embeddings
        if cos.ndim == 3:  # (B, S, d) or (3, S, d) for M-RoPE expanded
            new_cos = cos[..., keep_full_idx, :].contiguous()
            new_sin = sin[..., keep_full_idx, :].contiguous()
        elif cos.ndim == 4:  # (3, B, S, d)
            new_cos = cos[..., keep_full_idx, :].contiguous()
            new_sin = sin[..., keep_full_idx, :].contiguous()
        elif cos.ndim == 2:  # (S, d)
            new_cos = cos[keep_full_idx, :].contiguous()
            new_sin = sin[keep_full_idx, :].contiguous()
        else:
            new_cos = cos
            new_sin = sin
        new_position_embeddings = (new_cos, new_sin)
    else:
        new_position_embeddings = None

    # causal_mask: (B, 1, q, k) - slice both q and k dimensions
    if causal_mask is not None:
        if causal_mask.ndim == 4:
            new_causal_mask = causal_mask[:, :, keep_full_idx, :][:, :, :, keep_full_idx].contiguous()
        else:
            new_causal_mask = causal_mask
    else:
        new_causal_mask = None

    return (new_hidden, new_causal_mask, new_position_ids, new_text_position_ids,
            new_cache_position, new_position_embeddings, keep_full_idx, keep_visual_idx)


def Qwen2_5_VLTextModel_forward(
    self: Qwen2_5_VLTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    """Qwen2.5-VL TextModel forward, LLM internal pruning.

    Enable output_attentions at layer [qp - 1] to collect attention,
    trigger query_guided_prune synchronized slicing before entering layer at layer_idx == qp_layer.
    """
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You must specify exactly one of input_ids or inputs_embeds"
        )

    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(
            3, inputs_embeds.shape[0], -1
        )
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(
            3, position_ids.shape[0], -1
        )

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = None

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = (
                create_sliding_window_causal_mask(**mask_kwargs)
            )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    # ---- Assert all full_attention, single mask ----
    # Compatible with old/new transformers: new field is layer_type, old is attention_type
    def _get_layer_attn_type(layer):
        # Prefer self_attn.layer_type (new), fall back to attention_type (old/custom)
        sa = getattr(layer, 'self_attn', None)
        if sa is not None:
            lt = getattr(sa, 'layer_type', None)
            if lt is not None:
                return lt
        return getattr(layer, 'attention_type', 'full_attention')

    assert all(
        _get_layer_attn_type(decoder_layer) == "full_attention"
        for decoder_layer in self.layers[: self.config.num_hidden_layers]
    ), (
        "LLM pruning requires all decoder layers to be "
        "full_attention. Set use_sliding_window=False in model config."
    )
    causal_mask = causal_mask_mapping["full_attention"]

    # ---- Parse pruning configuration ----
    td_config = getattr(self, 'td_config', None)
    if td_config is None:
        for parent_attr in ('model', 'language_model'):
            parent = getattr(self, parent_attr, None)
            if parent is not None:
                td_config = getattr(parent, 'td_config', None)
                if td_config is not None:
                    break

    visual_token_range = getattr(td_config, '_visual_token_range', None) if td_config else None
    target_budget = getattr(td_config, '_target_budget', 0) if td_config else 0

    _qp_layer = -1
    if td_config is not None and visual_token_range is not None and target_budget > 0:
        _raw_qp = getattr(td_config, 'query_prune_layer', -1)
        if isinstance(_raw_qp, str):
            _raw_qp = _raw_qp.strip()
            _qp_layer = int(_raw_qp) if _raw_qp else -1
        elif isinstance(_raw_qp, (int, float)):
            _qp_layer = int(_raw_qp)

        if _qp_layer >= 0:
            _v_start, _v_end = visual_token_range
            _current_visual = _v_end - _v_start
            if _current_visual <= target_budget:
                _qp_layer = -1

    # observe_layers: only qp_layer - 1 (single-layer observation for hard pruning)
    _observe_layers = set()
    if _qp_layer > 0:
        _observe_layers.add(_qp_layer - 1)

    _prune_method = getattr(td_config, 'llm_prune_method', 'text_token') if td_config else 'text_token'

    is_prefill = inputs_embeds.shape[1] > 1

    _attn_history = []  # list of (B, H, q_eff, S)
    _output_attentions = output_attentions

    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # ---- Trigger prune before the pruning layer ----
        if (
            is_prefill
            and _qp_layer > 0
            and layer_idx == _qp_layer
            and visual_token_range is not None
            and len(_attn_history) > 0
        ):
            _v_start_now, _v_end_now = visual_token_range
            _num_visual_now = _v_end_now - _v_start_now
            if _num_visual_now > target_budget and hidden_states.shape[1] > _v_end_now:
                (
                    hidden_states,
                    causal_mask,
                    position_ids,
                    text_position_ids,
                    cache_position,
                    position_embeddings,
                    keep_full_idx,
                    keep_visual_idx,
                ) = _query_prune_qwen(
                    hidden_states=hidden_states,
                    causal_mask=causal_mask,
                    attentions_list=_attn_history,
                    cache_position=cache_position,
                    position_ids=position_ids,
                    text_position_ids=text_position_ids,
                    position_embeddings=position_embeddings,
                    visual_token_range=visual_token_range,
                    target_budget=target_budget,
                    prune_method=_prune_method,
                )

                if keep_visual_idx is not None:
                    new_v_count = keep_visual_idx.shape[0]
                    visual_token_range = (_v_start_now, _v_start_now + new_v_count)
                    if td_config is not None:
                        td_config._visual_token_range = visual_token_range
                    _td_logger.info(
                        f"[Query-Guided Prune] {_num_visual_now} -> {new_v_count} visual tokens "
                        f"(layer {_qp_layer})"
                    )

        # ---- Determine whether to enable output_attentions for this layer (for attn collection) ----
        _need_attn = (
            is_prefill
            and layer_idx in _observe_layers
            and visual_token_range is not None
        )
        if _need_attn:
            kwargs["prune_method"] = _prune_method
            kwargs["visual_token_start"] = visual_token_range[0]
            kwargs["visual_token_end"] = visual_token_range[1]
            kwargs["use_rope_for_pruning"] = getattr(td_config, 'use_rope_for_pruning', False) if td_config else False
            _layer_output_attentions = True
        else:
            kwargs.pop("prune_method", None)
            kwargs.pop("visual_token_start", None)
            kwargs.pop("visual_token_end", None)
            kwargs.pop("use_rope_for_pruning", None)
            _layer_output_attentions = _output_attentions

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            output_attentions=_layer_output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]
        attn_weights = layer_outputs[1] if len(layer_outputs) > 1 else None

        # ---- Collect attn for pruning ----
        if _need_attn and attn_weights is not None:
            _attn_history.append(attn_weights)

        if _output_attentions:
            all_self_attns += (attn_weights,)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                past_key_values,
                all_hidden_states,
                all_self_attns,
            ]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )



# ============================================================================
# VisionTransformerPretrainedModel forward 
# ============================================================================


def Qwen2_5_VisionTransformerPretrainedModel_forward(
    self: Qwen2_5_VisionTransformerPretrainedModel,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    """Qwen2.5-VL Vision Transformer forward.

    Identical to native forward, only additionally computes column-wise mean
    attention (CLS attention surrogate) and stores in td_config._vit_cls_attn at the last full-attention layer.
    """
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(
        seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
    )
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(
        seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
    )
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

    # Convert window_index to GPU tensor for subsequent reverse use
    if not isinstance(window_index, torch.Tensor):
        window_index = torch.tensor(window_index, device=hidden_states.device, dtype=torch.long)
    else:
        window_index = window_index.to(device=hidden_states.device)

    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # Find the index of the last full-attention layer for computing col_mean CLS attention
    td_config = getattr(self, 'td_config', None)
    last_fullatt_idx = max(self.fullatt_block_indexes) if self.fullatt_block_indexes else -1

    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens

        # Only pass td_config at the last full-attention layer to trigger col_mean computation
        extra_kwargs = {}
        if layer_num == last_fullatt_idx and td_config is not None:
            extra_kwargs['td_config'] = td_config

        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings,
            **extra_kwargs,
        )

    hidden_states = self.merger(hidden_states)

    # Reverse window_index to restore original order
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    # Also reorder _vit_cls_attn by reverse_indices to restore original order
    # (col_mean is computed in pre-merger dimension; here we do spatial_merge_unit group mean + reverse)
    if td_config is not None and getattr(td_config, '_vit_cls_attn', None) is not None:
        cls_attn_raw = td_config._vit_cls_attn
        smu = self.spatial_merge_unit
        n_groups = cls_attn_raw.shape[0] // smu
        cls_attn_merged = cls_attn_raw.view(n_groups, smu).mean(dim=1)
        td_config._vit_cls_attn = cls_attn_merged[reverse_indices]

    return hidden_states



# ============================================================================
# VisionBlock forward
# ============================================================================


def Qwen2_5_VLVisionBlock_forward(
    self: Qwen2_5_VLVisionBlock,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """Qwen2.5-VL VisionBlock forward."""
    residual = hidden_states
    hidden_states = self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.mlp(self.norm2(hidden_states))
    hidden_states = residual + hidden_states
    return hidden_states


# ============================================================================
# VisionAttention forward (CLS attention computation)
# ============================================================================


def Qwen2_5_VLVisionAttention_forward(
    self: Qwen2_5_VLVisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """Qwen2.5-VL Vision Attention forward.

    Identical to the native attention forward, except when td_config is passed via kwargs,
    it additionally computes CLS attention and stores it in td_config._vit_cls_attn.
    - col_mean: column-mean CLS attention
    """
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin
    )

    # ---- Extra CLS attention computation (only in layer where td_config is passed) ----
    td_config_ref = kwargs.pop('td_config', None)
    if td_config_ref is not None:
        with torch.no_grad():
            head_dim = query_states.shape[-1]
            num_heads = query_states.shape[1]
            total_tokens = query_states.shape[0]
            T = cu_seqlens.shape[0] - 1
            assert total_tokens % T == 0, \
                f"cls_attn requires equal-length segments: total={total_tokens}, T={T}"
            L = total_tokens // T

            # (T*L, H, d) -> (T, H, L, d)
            Q = query_states.view(T, L, num_heads, head_dim).permute(0, 2, 1, 3)
            K = key_states.view(T, L, num_heads, head_dim).permute(0, 2, 1, 3)

            chunk_t = 8
            scores = torch.empty(T, L, device=query_states.device, dtype=torch.float32)
            
            # col_mean CLS attention computation
            for s in range(0, T, chunk_t):
                e = min(s + chunk_t, T)
                logits = torch.matmul(Q[s:e], K[s:e].transpose(-2, -1)) / math.sqrt(head_dim)
                w = logits.softmax(dim=-1)
                # Column mean: first mean over query dim (dim=-2), then mean over heads (dim=1)
                scores[s:e] = w.mean(dim=-2).mean(dim=1)

            td_config_ref._vit_cls_attn = scores.reshape(T * L).to(torch.float32)

    # ---- Normal Flash Attention forward (identical to native) ----
    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[
            self.config._attn_implementation
        ]

    assert self.config._attn_implementation == "flash_attention_2"
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    attn_output, _ = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask=None,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=max_seqlen,
        max_length_k=max_seqlen,
        is_causal=False,
    )

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output


# ============================================================================
# LLM Attention forward (unmodified)
# ============================================================================


def Qwen2_5_VLAttention_forward(
    self: Qwen2_5_VLAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    """Qwen2.5-VL LLM Attention forward, with output_attentions fallback for pruning."""
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    # Save Q/K before RoPE (for use_rope_for_pruning=False ablation)
    _q_before_rope = query_states
    _k_before_rope = key_states

    cos, sin = position_embeddings
    # Compatible with both versions: new rope_parameters["mrope_section"], old rope_scaling["mrope_section"]
    if hasattr(self, 'rope_scaling') and self.rope_scaling is not None and "mrope_section" in self.rope_scaling:
        _mrope_section = self.rope_scaling["mrope_section"]
    else:
        _mrope_section = self.config.rope_parameters["mrope_section"]
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        _mrope_section,
    )

    if past_key_values is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
        }
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # Extract pruning-related kwargs (not passed to attention_interface)
    # NOTE: output_attentions is a formal function parameter, not in kwargs; must read from function args
    _output_attentions = output_attentions
    kwargs.pop("output_attentions", None)  # Defensive pop to avoid passing duplicate to attention_interface
    _prune_method = kwargs.pop("prune_method", "text_token")
    _v_start = kwargs.pop("visual_token_start", 0)
    _v_end = kwargs.pop("visual_token_end", 0)
    _use_rope_for_pruning = kwargs.pop("use_rope_for_pruning", False)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[
            self.config._attn_implementation
        ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,
        **kwargs,
    )

    # ---- Fallback: manually compute pruning attn_weights (Flash Attention 2 does not return them) ----
    # Returns full (B, H, q_eff, S) matrix; V-segment slicing is handled by _query_prune_qwen
    if _output_attentions and attn_weights is None:
        # Select Q/K source based on use_rope_for_pruning
        if _use_rope_for_pruning:
            _q_src = query_states
            _k_src = key_states
        else:
            _q_src = _q_before_rope
            _k_src = _k_before_rope

        with torch.no_grad():
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import repeat_kv
            key_states_for_score = repeat_kv(_k_src, self.num_key_value_groups)
            # K keeps full sequence: (B, H, S, d)
            k_full = key_states_for_score

            if _prune_method == "text_token":
                q_prefix = _q_src[:, :, :_v_start, :]
                q_suffix = _q_src[:, :, _v_end:, :]
                q_for_score = torch.cat([q_prefix, q_suffix], dim=2)  # (B, H, T_text, d)
            else:  # all_token
                q_for_score = _q_src  # (B, H, S, d)

            _scale = self.head_dim ** -0.5
            _logits = torch.matmul(q_for_score, k_full.transpose(-2, -1)) * _scale
            # Return full (B, H, q_eff, S) matrix
            attn_weights = _logits.softmax(dim=-1, dtype=torch.float32)

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights


# ============================================================================
# generate (transparent passthrough)
# ============================================================================


@torch.no_grad()
def Qwen2_5_VLForConditionalGeneration_generate(
    self: Qwen2_5_VLForConditionalGeneration,
    **kwargs,
):
    """Transparent passthrough to the original generate method."""
    return self.generate_ori(**kwargs)



def Qwen2_5_VLModel_get_video_features(
    self: Qwen2_5_VLModel,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    """get_video_features"""
    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)

    split_sizes = (
        video_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2
    ).tolist()

    video_embeds = torch.split(video_embeds, split_sizes)
    return video_embeds



# ============================================================================
# Model forward 
# ============================================================================


def Qwen2_5_VLModel_forward(
    self: Qwen2_5_VLModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:
    """Qwen2.5-VL Model forward, integrated with DynVid video token compression.
    """
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # ---- Image embedding (no compression) ----
    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # ---- Video embedding + compression ----
    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(
            pixel_values_videos, video_grid_thw
        )

        td_config = getattr(self, 'td_config', None)

        # === Determine compression path ===
        # Important debug/fix path:
        # If before_LLM_retention_ratio >= 1.0, do NOT enter the pre-LLM reconstruction
        # path. We keep the native Qwen2.5-VL video scatter and only record the
        # visual token range for optional inner-LLM pruning. This avoids changing
        # position_ids / cache_position / rope_deltas when vision-side compression
        # is disabled.
        _vis_ratio = 1.0
        _llm_ratio = 1.0
        if td_config is not None:
            _vis_ratio = getattr(td_config, 'before_LLM_retention_ratio', 1.0)
            _llm_ratio = getattr(td_config, 'llm_prune_ratio', 1.0)

        if td_config is not None and _vis_ratio >= 1.0:
            # ===== Native video scatter + optional LLM prune bookkeeping =====
            video_embeds_cat = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds_cat,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, video_embeds_cat
            )

            # Only record the contiguous video-token range for inner-LLM pruning.
            # Do not manually set position_ids / cache_position / rope_deltas here;
            # let the native get_rope_index branch below compute them.
            video_indices = (input_ids[0] == self.config.video_token_id).nonzero(as_tuple=True)[0]
            if video_indices.numel() > 0:
                video_start = video_indices[0].item()
                video_end = video_indices[-1].item() + 1
                td_config._visual_token_range = (video_start, video_end)
                td_config._target_budget = max(1, int(video_indices.numel() * _llm_ratio))
                _td_logger.info(
                    f"[NativeScatter+LLMPrune] visual=[{video_start},{video_end}), "
                    f"current={video_indices.numel()}, target={td_config._target_budget}, "
                    f"llm_ratio={_llm_ratio}"
                )

            # Prevent the fallback block below from scattering video a second time.
            pixel_values_videos = None

        elif td_config is not None:
            # ===== Compression path =====

            # Step 1: Scatter video embeddings into inputs_embeds (same as native path)
            video_embeds_cat = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds_cat,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)

            # Step 2: Run DynVid compression on each video to get kept_indices
            batch_idx = 0
            _video_token_id = self.config.video_token_id
            video_mask_1d_raw = (input_ids[batch_idx] == _video_token_id)
            video_token_indices = video_mask_1d_raw.nonzero(as_tuple=True)[0]

            # Compute position_ids (on the full sequence)
            orig_position_ids, orig_rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = orig_rope_deltas

            # Step 3: For each video segment, run compression and collect kept global indices
            keep_visual_global_indices_list = []
            video_offset = 0

            for i, v_embed in enumerate(video_embeds):
                T_i = video_grid_thw[i, 0].item()
                H_i = video_grid_thw[i, 1].item()
                W_i = video_grid_thw[i, 2].item()
                merge_size = self.visual.spatial_merge_size
                N_i_orig = (H_i * W_i) // (merge_size * merge_size)
                orig_count = T_i * N_i_orig
                N_i = N_i_orig

                # Position IDs for this video segment
                v_pos_start = video_offset
                v_pos_end = video_offset + orig_count
                video_offset += orig_count

                this_video_indices = video_token_indices[v_pos_start:v_pos_end]

                all_pos_ids = orig_position_ids[:, batch_idx, this_video_indices]

                if all_pos_ids.shape[0] == 4:
                    video_pos_ids = all_pos_ids[1:]         # (3, T*N)
                else:
                    video_pos_ids = all_pos_ids             # (3, T*N)

                comp_embeds, comp_positions, kept_indices = dynvid_compression_qwen(
                    video_embeds=v_embed,
                    position_ids_video=video_pos_ids,
                    num_frames=T_i,
                    tokens_per_frame=N_i,
                    config=td_config,
                )

                # kept_indices are local indices within the video segment; convert to global sequence indices
                kept_global = this_video_indices[kept_indices]
                keep_visual_global_indices_list.append(kept_global)

                # Write compressed embeddings back to the corresponding positions in inputs_embeds
                inputs_embeds[0, kept_global] = comp_embeds.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )

            # Step 4: Build keep_global_indices (prefix + kept_video + suffix)
            seq_len = inputs_embeds.shape[1]
            video_start = video_token_indices[0].item()
            video_end = video_token_indices[-1].item() + 1

            prefix_indices = torch.arange(video_start, device=inputs_embeds.device)
            suffix_indices = torch.arange(video_end, seq_len, device=inputs_embeds.device)
            kept_video_indices = torch.cat(keep_visual_global_indices_list, dim=0)

            keep_global_indices = torch.cat(
                [prefix_indices, kept_video_indices, suffix_indices], dim=0
            ).sort().values

            # Step 5: index selection
            bsz, _, hidden_size = inputs_embeds.shape
            inputs_embeds = torch.gather(
                inputs_embeds,
                dim=1,
                index=keep_global_indices.view(1, -1, 1).expand(bsz, -1, hidden_size),
            )
            position_ids = orig_position_ids[:, :, keep_global_indices]
            if attention_mask is not None:
                attention_mask = attention_mask[:, keep_global_indices]
            cache_position = torch.arange(seq_len, device=inputs_embeds.device)[keep_global_indices]

            # Record visual token positions for LLM pruning
            n_kept_video = kept_video_indices.shape[0]
            new_video_start = video_start
            new_video_end = video_start + n_kept_video
            td_config._visual_token_range = (new_video_start, new_video_end)
            _llm_ratio = getattr(td_config, 'llm_prune_ratio', 1.0)
            td_config._target_budget = max(1, int(n_kept_video * _llm_ratio))

            pixel_values_videos = None

        else:
            # ===== Native path (no compression or bypass) =====
            video_embeds_cat = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds_cat,
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask, video_embeds_cat
            )

    # ---- Fallback ----
    if pixel_values_videos is not None:
        video_embeds_native = self.get_video_features(
            pixel_values_videos, video_grid_thw
        )
        video_embeds_native = torch.cat(video_embeds_native, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        _, video_mask = self.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds_native,
        )
        inputs_embeds = inputs_embeds.masked_scatter(
            video_mask, video_embeds_native
        )

    # ---- Position ID computation ----
    if position_ids is None:
        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (
                past_key_values is None
                or past_key_values.get_seq_length() == 0
            )
        )
        if (
            prefill_compiled_stage or prefill_noncompiled_stage
        ) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            position_ids = torch.arange(
                seq_length, device=inputs_embeds.device
            )
            position_ids = position_ids.view(1, 1, -1).expand(
                3, batch_size, -1
            )
            if cache_position is not None:
                delta = (cache_position[0] + self.rope_deltas).to(
                    inputs_embeds.device
                )
            else:
                delta = torch.zeros(
                    (batch_size, seq_length), device=inputs_embeds.device
                )
            delta = delta.repeat_interleave(
                batch_size // delta.shape[0], dim=1
            )
            position_ids = position_ids + delta.to(position_ids.device)

    # ---- LLM forward ----
    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    output = Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )
    return output if return_dict else output.to_tuple()