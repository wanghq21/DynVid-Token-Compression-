"""
modeling_qwen3_vl.py

Qwen3-VL model architecture (DynVid compression integration).
The compression algorithm is implemented in compression_unified.py; this file only contains
the model architecture and thin wrappers.

Architecture features:
  - Global attention ViT (no window attention)
  - DeepStack: multi-layer ViT feature injection into LLM
  - q_norm / k_norm (Qwen3-specific)
  - No sliding window

"""

from typing import Callable, Optional, Union, List, Tuple
import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs, is_flash_attn_available
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    Qwen3VLVisionAttention,
    Qwen3VLVisionBlock,
    Qwen3VLVisionModel,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLForConditionalGeneration,
    repeat_kv,
)

from .compression_unified import (
    _td_logger,
    dynvid_compression_qwen,
    _deepstack_process,
)



# ============================================================================
# Qwen3-VL specific: Attention-score-based query-guided pruning (Plan A)
# ============================================================================

def query_guided_pruning_qwen3(
    hidden_states: torch.Tensor,       # (B, S, D)
    visual_token_range: tuple,         # (start, end)
    target_budget: int,                # final number of visual tokens to retain
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    text_position_ids: Optional[torch.Tensor] = None,
    cache_position: Optional[torch.Tensor] = None,
    past_key_values=None,
    td_config=None,
    position_embeddings=None,          # RoPE embeddings (cos, sin)
    attn_scores: Optional[torch.Tensor] = None,   # (B, V) produced by the previous decoder layer's normal forward
) -> tuple:
    """LLM Decoder internal attention-score-based pruning (Qwen3-specific, Plan A).

    Differences from query_guided_pruning_qwen:
      - Does not compute Q/K attention internally (avoids missing q_norm/k_norm/RoPE)
      - Directly receives attn_scores produced by the decoder layer's normal forward
      - attn_scores are already correct (processed through q_norm + k_norm + RoPE + softmax)

    Args:
        hidden_states: (B, S, D)
        visual_token_range: (start, end)
        target_budget: number of visual tokens to retain after pruning
        attention_mask, position_ids, text_position_ids, cache_position: sequence tensors
        past_key_values: KV cache
        td_config: DynVidConfig
        position_embeddings: (cos, sin) RoPE embeddings, used for reconstruction after pruning
        attn_scores: (B, V) importance score for each visual token
            Correctly computed in Qwen3VLTextAttention_forward (with q_norm/k_norm/RoPE)

    Returns:
        (hidden_states, attention_mask, position_ids, text_position_ids,
         cache_position, num_pruned, keep_idx)
    """
    B, S, D = hidden_states.shape
    assert B == 1, "query_guided_pruning_qwen3 currently supports batch_size=1 only"
    v_start, v_end = visual_token_range
    num_visual = v_end - v_start

    # Safety check
    if v_end > S or v_start >= S or num_visual <= 0:
        return hidden_states, attention_mask, position_ids, text_position_ids, cache_position, 0, None

    if num_visual <= target_budget:
        return hidden_states, attention_mask, position_ids, text_position_ids, cache_position, 0, None

    # ========================================================================
    # Use externally provided attn_scores
    # ========================================================================
    if attn_scores is not None:
        scores = attn_scores  # (B, V) — correctly computed by the decoder layer
    else:
        _td_logger.warning(
            "[QueryPrune-Q3] attn_scores not available, falling back to cosine similarity"
        )
        visual_hidden = hidden_states[:, v_start:v_end, :]
        text_prefix = hidden_states[:, :v_start, :]
        text_suffix = hidden_states[:, v_end:, :]

        if text_prefix.shape[1] + text_suffix.shape[1] == 0:
            return hidden_states, attention_mask, position_ids, text_position_ids, cache_position, 0, None

        text_all = torch.cat([text_prefix, text_suffix], dim=1)
        query = text_all.mean(dim=1, keepdim=True)

        query_normed = F.normalize(query, dim=-1)
        visual_normed = F.normalize(visual_hidden, dim=-1)
        scores = (visual_normed * query_normed).sum(dim=-1)  # (B, num_visual)


    # ========================================================================
    # Top-k selection & physical removal
    # ========================================================================
    visual_hidden = hidden_states[:, v_start:v_end, :]
    k = min(target_budget, num_visual)
    _, top_indices = scores.topk(k, dim=1)
    top_indices = top_indices.sort(dim=1).values

    keep_idx = top_indices[0]  # (k,)
    kept_visual = visual_hidden[0, keep_idx, :].unsqueeze(0)

    # Reassemble the sequence
    text_prefix = hidden_states[:, :v_start, :]
    text_suffix = hidden_states[:, v_end:, :]
    new_hidden = torch.cat([text_prefix, kept_visual, text_suffix], dim=1)

    # Update attention_mask
    new_attention_mask = attention_mask
    if attention_mask is not None and attention_mask.ndim == 2:
        prefix_mask = attention_mask[:, :v_start]
        suffix_mask = attention_mask[:, v_end:]
        visual_mask = torch.ones(B, k, device=attention_mask.device, dtype=attention_mask.dtype)
        new_attention_mask = torch.cat([prefix_mask, visual_mask, suffix_mask], dim=1)

    # Update 3D RoPE position_ids
    new_position_ids = position_ids
    if position_ids is not None:
        prefix_pos = position_ids[:, :, :v_start]
        suffix_pos = position_ids[:, :, v_end:]
        visual_pos = position_ids[:, :, v_start:v_end]
        kept_pos = visual_pos[:, :, keep_idx]
        new_position_ids = torch.cat([prefix_pos, kept_pos, suffix_pos], dim=2)

    # Update 2D text_position_ids
    new_text_position_ids = text_position_ids
    if text_position_ids is not None and text_position_ids.ndim == 2:
        tp_prefix = text_position_ids[:, :v_start]
        tp_suffix = text_position_ids[:, v_end:]
        tp_visual = text_position_ids[:, v_start:v_end]
        tp_kept = tp_visual[:, keep_idx]
        new_text_position_ids = torch.cat([tp_prefix, tp_kept, tp_suffix], dim=1)

    # Update cache_position
    new_cache_position = cache_position
    if cache_position is not None:
        new_cache_position = torch.arange(0, new_hidden.shape[1], device=cache_position.device)

    # Clean up past_key_values
    if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
        _kv_keep = torch.cat([
            torch.arange(v_start, device=keep_idx.device),
            v_start + keep_idx,
            torch.arange(v_end, S, device=keep_idx.device),
        ])
        for _kv_layer_idx in range(len(past_key_values.key_cache)):
            if past_key_values.key_cache[_kv_layer_idx].numel() > 0:
                kv_len = past_key_values.key_cache[_kv_layer_idx].shape[2]
                if kv_len == S:
                    past_key_values.key_cache[_kv_layer_idx] = past_key_values.key_cache[_kv_layer_idx].index_select(2, _kv_keep)
                    past_key_values.value_cache[_kv_layer_idx] = past_key_values.value_cache[_kv_layer_idx].index_select(2, _kv_keep)

    num_pruned = num_visual - k
    _td_logger.info(f"[QueryPrune] {num_visual} -> {k} visual tokens (pruned {num_pruned})")

    return new_hidden, new_attention_mask, new_position_ids, new_text_position_ids, new_cache_position, num_pruned, keep_idx





def Qwen3VLTextModel_forward(
    self: Qwen3VLTextModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    # args for deepstack
    visual_pos_masks: Optional[torch.Tensor] = None,
    deepstack_visual_embeds: Optional[list] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, BaseModelOutputWithPast]:
    """Qwen3-VL TextModel forward, integrating:
    - DeepStack per-layer visual feature injection
    - Single-layer Hard Pruning
    - Post-Hard-Pruning synchronized trimming of DeepStack embeds and visual_pos_masks
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    attention_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # ---- DynVid pruning config ----
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

    # Parse query_prune_layer (single layer, integer)
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

    is_prefill = inputs_embeds.shape[1] > 1
    _pruned_done = False  # Fix-Bug4: prevent re-entry into hard pruning

    # Layers that need output_attentions: only qp_layer - 1 (used for hard pruning)
    _need_attn_layers = set()
    if _qp_layer > 0:
        _need_attn_layers.add(_qp_layer - 1)
    _last_prune_attn = None  # Store attn_scores from qp_layer-1 for pruning

    for layer_idx, decoder_layer in enumerate(self.layers):

        # Decide whether attention weights are needed
        _need_attn = (
            layer_idx in _need_attn_layers
            and is_prefill
            and visual_token_range is not None
        )
        _prune_method = getattr(td_config, 'llm_prune_method', 'text_token') if td_config else 'text_token'
        if _need_attn:
            kwargs["output_attentions"] = True
            kwargs["prune_method"] = _prune_method
            kwargs["visual_token_start"] = visual_token_range[0]
            kwargs["visual_token_end"] = visual_token_range[1]
            kwargs["use_rope_for_pruning"] = getattr(td_config, 'use_rope_for_pruning', True) if td_config else True
        else:
            kwargs.pop("output_attentions", None)
            kwargs.pop("prune_method", None)
            kwargs.pop("visual_token_start", None)
            kwargs.pop("visual_token_end", None)
            kwargs.pop("use_rope_for_pruning", None)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        # -- DeepStack: inject ViT intermediate layer features in the first N layers --
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            if visual_pos_masks is not None:
                hidden_states = _deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )

        # Save attn_weights from qp_layer-1 for pruning
        if (
            _qp_layer > 0
            and layer_idx == _qp_layer - 1
            and _need_attn
            and layer_outputs[1] is not None
        ):
            _last_prune_attn = layer_outputs[1]  # (B, V)

        # -- Hard Pruning: use real attention scores from the previous layer --
        if (
            layer_idx == _qp_layer
            and is_prefill
            and not _pruned_done
            and visual_token_range is not None
            and hidden_states.shape[1] > visual_token_range[1]
        ):
            _v_start_now, _v_end_now = visual_token_range
            _num_visual_now = _v_end_now - _v_start_now

            if _num_visual_now > target_budget:
                hidden_states, attention_mask, position_ids, text_position_ids, cache_position, num_pruned, _keep_idx = (
                    query_guided_pruning_qwen3(
                        hidden_states=hidden_states,
                        visual_token_range=visual_token_range,
                        target_budget=target_budget,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        text_position_ids=text_position_ids,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        td_config=td_config,
                        position_embeddings=position_embeddings,
                        attn_scores=_last_prune_attn,
                    )
                )

                if num_pruned > 0:
                    # Update visual_token_range
                    visual_token_range = (_v_start_now, _v_start_now + target_budget)
                    if td_config is not None:
                        td_config._visual_token_range = visual_token_range

                    # Rebuild RoPE
                    if position_ids is not None and position_ids.ndim == 3:
                        position_embeddings = self.rotary_emb(hidden_states, position_ids)

                    # Fix DynamicCache._seen_tokens
                    if past_key_values is not None and hasattr(past_key_values, '_seen_tokens'):
                        past_key_values._seen_tokens = hidden_states.shape[1]

                    # Fix-Bug1: Clear KV cache for all layers after _qp_layer.
                    # Otherwise subsequent layers' update() would append new K/V to existing old cache,
                    # causing cache length to double (S_new + S_new = 2 * S_new).
                    if past_key_values is not None and hasattr(past_key_values, 'key_cache'):
                        for _l_after in range(layer_idx + 1, len(past_key_values.key_cache)):
                            if past_key_values.key_cache[_l_after].numel() > 0:
                                past_key_values.key_cache[_l_after] = past_key_values.key_cache[_l_after][:, :, :0, :]
                                past_key_values.value_cache[_l_after] = past_key_values.value_cache[_l_after][:, :, :0, :]

                    # Rebuild causal_mask (Qwen3-VL: no sliding window)
                    attention_mask = create_causal_mask(
                        config=self.config,
                        input_embeds=hidden_states,
                        attention_mask=attention_mask,
                        cache_position=cache_position,
                        past_key_values=None,
                        position_ids=text_position_ids,
                    )

                    # -- Synchronized trimming of DeepStack embeds + visual_pos_masks --
                    if _keep_idx is not None:
                        # Trim visual_pos_masks
                        if visual_pos_masks is not None:
                            # Rebuild: prefix + kept_visual + suffix
                            _vpm_prefix = visual_pos_masks[:, :_v_start_now]
                            _vpm_suffix = visual_pos_masks[:, _v_end_now:]
                            _vpm_visual = visual_pos_masks[:, _v_start_now:_v_end_now]
                            _vpm_kept = _vpm_visual[:, _keep_idx]
                            visual_pos_masks = torch.cat([_vpm_prefix, _vpm_kept, _vpm_suffix], dim=1)

                        # Trim DeepStack embeds: _keep_idx is visual-local index.
                        # To avoid DeepStack list and LLM layer_idx being in different index spaces,
                        # synchronize all DeepStack entries to the current visual token set.
                        if deepstack_visual_embeds is not None:
                            for _ds_i in range(len(deepstack_visual_embeds)):
                                deepstack_visual_embeds[_ds_i] = deepstack_visual_embeds[_ds_i][_keep_idx]

                    # After pruning, need to update visual_token_range in _need_attn_layers
                    # (subsequent layers still in _need_attn_layers would need the new range).
                    # However, since hard pruning executes only once (single layer), no further attn is needed.
                    _pruned_done = True  # Fix-Bug4: mark pruning as done

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )



def Qwen3VLVisionAttention_forward(
    self: Qwen3VLVisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple] = None,
    return_logits: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Qwen3-VL Vision Attention forward.

    When kwargs contains compute_cls_attn=True, computes CLS attention (col_mean: attention matrix column mean).

    When return_logits=True, additionally returns per-frame attention weights.
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

    # ---- CLS attention computation (DynVid, col_mean) ----
    compute_cls_attn = kwargs.pop('compute_cls_attn', False)
    td_config_ref = kwargs.pop('td_config', None)

    if compute_cls_attn and td_config_ref is not None:
        with torch.no_grad():
            head_dim = query_states.shape[-1]
            cls_scores = torch.zeros(seq_length, device=hidden_states.device, dtype=torch.float32)

            num_segments = cu_seqlens.shape[0] - 1
            for seg_idx in range(num_segments):
                seg_start = cu_seqlens[seg_idx].item()
                seg_end = cu_seqlens[seg_idx + 1].item()
                seg_len = seg_end - seg_start
                if seg_len <= 0:
                    continue

                Q_seg = query_states[seg_start:seg_end]
                K_seg = key_states[seg_start:seg_end]

                # col_mean: attention matrix column mean
                attn_logits = torch.bmm(
                    Q_seg.transpose(0, 1),
                    K_seg.transpose(0, 1).transpose(-2, -1)
                ) / math.sqrt(head_dim)
                attn_weights = attn_logits.softmax(dim=-1)
                cls_scores[seg_start:seg_end] = attn_weights.mean(dim=-2).mean(dim=0)

            td_config_ref._vit_cls_attn = cls_scores

    # ---- Flash Attention 2 ----
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
        **kwargs,
    )

    # ---- Optional: return attention weights  ----
    attn_weights = None
    if return_logits:
        num_frames = cu_seqlens.shape[0] - 1
        q, k = query_states.squeeze(0), key_states.squeeze(0)
        q, k = q.transpose(0, 1), k.transpose(0, 1)
        q = q.reshape(num_frames, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(num_frames, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / self.head_dim**0.5
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = attn_weights.mean(1).mean(1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output, attn_weights


def Qwen3VLVisionBlock_forward(
    self: Qwen3VLVisionBlock,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Qwen3-VL VisionBlock forward, passes kwargs through to attention."""
    residual = hidden_states
    hidden_states, attn_weights = self.attn(
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
    return hidden_states, attn_weights


def Qwen3VLVisionModel_forward(
    self: Qwen3VLVisionModel,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, list, Optional[torch.Tensor]]:
    """Qwen3-VL ViT forward.

    Key differences from Qwen2.5-VL:
    1. Pure global attention, no window attention -> no window_index / cu_window_seqlens
    2. Added DeepStack: collects features at specified intermediate layers, projects via deepstack_merger_list
    3. CLS attention is computed at the last layer (all layers are global, so the last layer suffices)

    Returns:
        hidden_states: (post_merger_seq, D) merged visual features
        deepstack_feature_lists: list of (post_merger_seq, D) DeepStack features
        attn_weights: Optional last-layer attention weights 
    """
    hidden_states = self.patch_embed(hidden_states)

    # Qwen3-VL: fast positional embedding interpolation
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    # Qwen3-VL: pure global attention, cu_seqlens is simply cumulative per frame
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    td_config = getattr(self, 'td_config', None)
    num_blocks = len(self.blocks)
    # CLS attention: computed at the last block (Qwen3-VL has global attention in all layers)
    last_block_idx = num_blocks - 1


    # ---- DeepStack collection ----
    deepstack_feature_lists = []

    for layer_num, blk in enumerate(self.blocks):
        return_logits = (layer_num == last_block_idx)

        # CLS attention kwargs (col_mean)
        extra_kwargs = {}
        if layer_num == last_block_idx and td_config is not None:
            extra_kwargs['compute_cls_attn'] = True
            extra_kwargs['td_config'] = td_config

        hidden_states, attn_weights = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            return_logits=return_logits,
            **extra_kwargs,
        )

        # DeepStack: collect intermediate layer features
        if hasattr(self, 'deepstack_visual_indexes') and layer_num in self.deepstack_visual_indexes:
            ds_idx = self.deepstack_visual_indexes.index(layer_num)
            deepstack_feature = self.deepstack_merger_list[ds_idx](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

        # ================================================================

    # ---- Compute ViT internal per-frame spread (pre-merger) ----
    if td_config is not None:
        with torch.no_grad():
            _smu_spread = self.spatial_merge_unit
            _n_sg = hidden_states.shape[0] // _smu_spread
            _sg_feats = hidden_states.float().view(_n_sg, _smu_spread, -1).mean(dim=1)
            # Qwen3-VL has no window reordering: already in original order, no argsort needed
            _sg_feats_norm = F.normalize(_sg_feats, dim=-1)

            _cu_sg = (cu_seqlens.float() / _smu_spread).long()
            _n_frames_sp = _cu_sg.shape[0] - 1
            _vit_spread = torch.zeros(_n_frames_sp, device=hidden_states.device, dtype=torch.float32)

            for _fidx in range(_n_frames_sp):
                _f_start = _cu_sg[_fidx].item()
                _f_end = _cu_sg[_fidx + 1].item()
                _f_feat = _sg_feats_norm[_f_start:_f_end]
                if _f_feat.shape[0] > 1:
                    _sim = _f_feat @ _f_feat.T
                    _ut_mask = torch.triu(torch.ones_like(_sim, dtype=torch.bool), diagonal=1)
                    _vit_spread[_fidx] = 1.0 - _sim[_ut_mask].mean()

            td_config._vit_spread = _vit_spread

    # ---- merger (spatial merge) ----
    hidden_states = self.merger(hidden_states)
    # Qwen3-VL has no window reordering: no reverse_indices needed

    # CLS attn: spatial merge unit group mean (no reverse needed)
    if td_config is not None and hasattr(td_config, '_vit_cls_attn') and td_config._vit_cls_attn is not None:
        smu = self.spatial_merge_unit
        cls_attn_raw = td_config._vit_cls_attn
        _cur_seq = cls_attn_raw.shape[0]
        n_groups = _cur_seq // smu
        td_config._vit_cls_attn = cls_attn_raw.view(n_groups, smu).mean(dim=1)


    # Record fusion split sizes

    return hidden_states, deepstack_feature_lists, attn_weights



# ============================================================================
# Qwen3-VL get_video_features / get_image_features
# ============================================================================


def Qwen3VLModel_get_video_features(
    self: Qwen3VLModel,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    """get_video_features: returns (video_embeds_list, deepstack_embeds, attn_weights)."""
    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
    video_embeds, deepstack_video_embeds, attn_weights = self.visual(
        pixel_values_videos, grid_thw=video_grid_thw
    )

    split_sizes = (
        video_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2
    ).tolist()

    video_embeds = torch.split(video_embeds, split_sizes)
    return video_embeds, deepstack_video_embeds, attn_weights


def Qwen3VLModel_get_image_features(
    self: Qwen3VLModel,
    pixel_values: torch.FloatTensor,
    image_grid_thw: Optional[torch.LongTensor] = None,
):
    """get_image_features: returns (image_embeds_list, deepstack_embeds)."""
    pixel_values = pixel_values.type(self.visual.dtype)
    image_embeds, deepstack_image_embeds, _ = self.visual(
        pixel_values, grid_thw=image_grid_thw
    )
    split_sizes = (
        image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2
    ).tolist()
    image_embeds = torch.split(image_embeds, split_sizes)
    return image_embeds, deepstack_image_embeds

# Qwen3-VL LLM Attention Forward (Qwen3-specific: q_norm / k_norm)
# ============================================================================


def Qwen3VLTextAttention_forward(
    self: Qwen3VLTextAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple:
    """Qwen3-VL LLM Attention forward (with q_norm/k_norm).

    Addition: when kwargs contains output_attentions=True and Flash Attention 2 does not return attn_weights,
    manually computes the attention scores needed for pruning.
    At this point query_states/key_states have already gone through q_norm + k_norm + RoPE, so results are fully correct.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Save pre-RoPE Q/K (for use_rope_for_pruning=False ablation)
    _q_before_rope = query_states
    _k_before_rope = key_states

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # Extract pruning-related kwargs (do not pass to attention_interface)
    _output_attentions = kwargs.pop("output_attentions", False)
    _prune_method = kwargs.pop("prune_method", "text_token")
    _v_start = kwargs.pop("visual_token_start", 0)
    _v_end = kwargs.pop("visual_token_end", 0)
    _use_rope_for_pruning = kwargs.pop("use_rope_for_pruning", False)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    # ---- Fallback: manually compute pruning attn_weights (Flash Attention 2 does not return them) ----
    if _output_attentions and attn_weights is None:
        # Select Q/K source based on use_rope_for_pruning:
        #   True  -> use post-RoPE query_states/key_states (with q_norm + k_norm + RoPE)
        #   False -> use pre-RoPE _q_before_rope/_k_before_rope (only q_norm + k_norm)
        if _use_rope_for_pruning:
            _q_src = query_states
            _k_src = key_states
        else:
            _q_src = _q_before_rope
            _k_src = _k_before_rope

        with torch.no_grad():
            key_states_for_score = repeat_kv(_k_src, self.num_key_value_groups)
            # Only take K columns corresponding to visual tokens
            k_visual = key_states_for_score[:, :, _v_start:_v_end, :]  # (B, H, V, d)

            if _prune_method == "text_token":
                q_prefix = _q_src[:, :, :_v_start, :]
                q_suffix = _q_src[:, :, _v_end:, :]
                q_for_score = torch.cat([q_prefix, q_suffix], dim=2)  # (B, H, T_text, d)
            else:  # all_token
                q_for_score = _q_src  # (B, H, S, d)

            _scale = self.head_dim ** -0.5
            _logits = torch.matmul(q_for_score, k_visual.transpose(-2, -1)) * _scale  # (B, H, Q, V)
            _weights = _logits.softmax(dim=-1, dtype=torch.float32)

            attn_weights = _weights.max(dim=2).values.mean(dim=1)  # (B, V)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def Qwen3VLTextDecoderLayer_forward(
    self: Qwen3VLTextDecoderLayer,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple:
    """Qwen3-VL Decoder Layer forward."""
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states, attn_weights


@torch.no_grad()
def Qwen3VLForConditionalGeneration_generate(
    self: Qwen3VLForConditionalGeneration,
    **kwargs,
):
    """Transparent passthrough to the original generate method."""
    return self.generate_ori(**kwargs)


def Qwen3VLModel_forward(
    self: Qwen3VLModel,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen3VLModelOutputWithPast]:
    """Qwen3-VL Model forward, integrating DynVid video token compression + DeepStack.

    Pipeline:
    1. ViT -> extract visual features + DeepStack intermediate layer features
    2. DynVid compression -> returns compressed_embeds + kept_global_indices
    3. Use kept_global_indices to synchronously trim DeepStack embeds
    4. Construct visual_pos_masks
    5. Rebuild sequence -> LLM forward (with DeepStack injection + pruning)
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    td_config = getattr(self, 'td_config', None)

    # Initialize visual_pos_masks / deepstack (the compression path sets these internally)
    visual_pos_masks = None
    deepstack_visual_embeds = None

    # ---- Image embedding (no compression, but collect DeepStack) ----
    image_mask = None
    deepstack_image_embeds = None
    if pixel_values is not None:
        image_embeds, deepstack_image_embeds = Qwen3VLModel_get_image_features(
            self, pixel_values, image_grid_thw
        )
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    # ---- Video embedding + compression ----
    video_mask = None
    deepstack_video_embeds = None
    n_video_tokens = 0
    if pixel_values_videos is not None:
        video_embeds_list, deepstack_video_embeds, cls_attention = Qwen3VLModel_get_video_features(
            self, pixel_values_videos, video_grid_thw
        )

        if td_config is not None:
            _vis_ratio = getattr(td_config, 'before_LLM_retention_ratio', 1.0)
            _llm_ratio = getattr(td_config, 'llm_prune_ratio', 1.0)

            if _vis_ratio >= 1.0:
                # ===== Native video scatter + optional LLM prune bookkeeping =====
                video_embeds_cat = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                n_video_tokens = video_embeds_cat.shape[0]
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds_cat
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)

                video_indices = (input_ids[0] == self.config.video_token_id).nonzero(as_tuple=True)[0]
                if video_indices.numel() > 0:
                    video_start = video_indices[0].item()
                    video_end = video_indices[-1].item() + 1
                    td_config._visual_token_range = (video_start, video_end)
                    td_config._target_budget = max(1, int(video_indices.numel() * _llm_ratio))

                pixel_values_videos = None

            else:
                # ===== DynVid compression path =====
                # Core idea: scatter first -> get_rope_index on full sequence ->
                # DynVid compression -> select by global indices -> no sequence rebuild

                # Step 1: Compute position_ids on FULL original sequence
                if position_ids is None:
                    attention_mask_tensor = (
                        attention_mask if not isinstance(attention_mask, dict) else attention_mask.get("full_attention", attention_mask)
                    )
                    if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                        attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                        if attention_mask_tensor.dtype.is_floating_point:
                            attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                            attention_mask_tensor = (1.0 - attention_mask_tensor).int()

                    orig_position_ids, orig_rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        attention_mask=attention_mask_tensor if isinstance(attention_mask_tensor, torch.Tensor) else attention_mask,
                    )
                    self.rope_deltas = orig_rope_deltas  # DO NOT MODIFY
                else:
                    orig_position_ids = position_ids
                    orig_rope_deltas = getattr(self, 'rope_deltas', None)

                batch_idx = 0
                _video_token_id = self.config.video_token_id
                video_mask_1d_raw = (input_ids[batch_idx] == _video_token_id)
                video_token_indices = video_mask_1d_raw.nonzero(as_tuple=True)[0]

                # Step 2: Scatter video embedding into inputs_embeds (same as native path)
                video_embeds_cat = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
                n_video_tokens = video_embeds_cat.shape[0]
                video_mask_3d = video_mask_1d_raw.unsqueeze(0).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask_3d, video_embeds_cat)

                # Step 2.5: Construct visual_pos_masks + deepstack (before compression, on the full sequence)
                _vid_mask_1d_full = video_mask_1d_raw.unsqueeze(0)  # (1, seq_len)
                if image_mask is not None:
                    _img_mask_1d_full = image_mask[..., 0]  # (1, seq_len)
                    _visual_pos_masks_full = _img_mask_1d_full | _vid_mask_1d_full
                    _deepstack_visual_embeds_full = []
                    _img_joint = _img_mask_1d_full[_visual_pos_masks_full]
                    _vid_joint = _vid_mask_1d_full[_visual_pos_masks_full]
                    _ds_img = deepstack_image_embeds or []
                    _ds_vid = deepstack_video_embeds or []
                    _n_ds = max(len(_ds_img), len(_ds_vid))
                    for _di in range(_n_ds):
                        _embed_joint = torch.zeros(_visual_pos_masks_full.sum(), _ds_img[0].shape[-1] if _ds_img else _ds_vid[0].shape[-1], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                        if _di < len(_ds_img):
                            _embed_joint[_img_joint] = _ds_img[_di].to(inputs_embeds.dtype)
                        if _di < len(_ds_vid):
                            _embed_joint[_vid_joint] = _ds_vid[_di].to(inputs_embeds.dtype)
                        _deepstack_visual_embeds_full.append(_embed_joint)
                else:
                    _visual_pos_masks_full = _vid_mask_1d_full
                    _deepstack_visual_embeds_full = deepstack_video_embeds

                # Step 3: Run DynVid compression, collect kept GLOBAL indices
                keep_visual_global_indices_list = []
                all_kept_local_indices = []  # For DeepStack synchronization (local to concat video)

                video_offset = 0

                for i, v_embed in enumerate(video_embeds_list):
                    T_i = video_grid_thw[i, 0].item()
                    H_i = video_grid_thw[i, 1].item()
                    W_i = video_grid_thw[i, 2].item()
                    merge_size = self.visual.spatial_merge_size
                    N_i_orig = (H_i * W_i) // (merge_size * merge_size)
                    orig_count = T_i * N_i_orig
                    N_i = N_i_orig

                    v_pos_start = video_offset
                    v_pos_end = video_offset + orig_count
                    video_offset += orig_count

                    this_video_indices = video_token_indices[v_pos_start:v_pos_end]
                    all_pos_ids = orig_position_ids[:, batch_idx, this_video_indices]

                    assert all_pos_ids.shape[-1] == T_i * N_i, (
                        f"Position mismatch: got {all_pos_ids.shape[-1]}, expected {T_i * N_i}"
                    )

                    if all_pos_ids.shape[0] == 4:
                        video_pos_ids = all_pos_ids[1:]
                    else:
                        video_pos_ids = all_pos_ids

                    # DynVid compression
                    comp_embeds, comp_positions, kept_indices = dynvid_compression_qwen(
                        video_embeds=v_embed,
                        position_ids_video=video_pos_ids,
                        num_frames=T_i,
                        tokens_per_frame=N_i,
                        config=td_config,
                    )

                    # Convert kept_indices (local to this video) -> global indices in full sequence
                    kept_global = this_video_indices[kept_indices]
                    keep_visual_global_indices_list.append(kept_global)

                    # Write compressed embeddings back to inputs_embeds at original global positions
                    inputs_embeds[0, kept_global] = comp_embeds.to(inputs_embeds.device, inputs_embeds.dtype)

                    # For DeepStack sync: local indices relative to concatenated video embeds
                    all_kept_local_indices.append(kept_indices + v_pos_start)

                assert inputs_embeds.shape[0] == 1, "Compression currently supports batch_size=1"

                # Step 4: Build keep_global_indices (prefix + kept video + suffix)
                seq_len = inputs_embeds.shape[1]
                video_start = video_token_indices[0].item()
                video_end = video_token_indices[-1].item() + 1

                prefix_indices = torch.arange(video_start, device=inputs_embeds.device)
                suffix_indices = torch.arange(video_end, seq_len, device=inputs_embeds.device)
                kept_video_indices = torch.cat(keep_visual_global_indices_list, dim=0)
                keep_global_indices = torch.cat([prefix_indices, kept_video_indices, suffix_indices], dim=0).sort().values

                # Step 5: index selection
                bsz, _, hidden_size = inputs_embeds.shape
                inputs_embeds = torch.gather(
                    inputs_embeds, dim=1,
                    index=keep_global_indices.view(1, -1, 1).expand(bsz, -1, hidden_size)
                )
                position_ids = orig_position_ids[:, :, keep_global_indices]

                if attention_mask is not None:
                    if isinstance(attention_mask, torch.Tensor):
                        attention_mask = attention_mask[:, keep_global_indices]

                cache_position = torch.arange(seq_len, device=inputs_embeds.device)[keep_global_indices]

                # rope_deltas remains unchanged (DO NOT MODIFY)
                self.rope_deltas = orig_rope_deltas

                # ---- synchronized trimming of visual_pos_masks + DeepStack ----
                # visual_pos_masks: directly index from the full sequence mask using keep_global_indices
                visual_pos_masks = _visual_pos_masks_full[:, keep_global_indices]

                # deepstack_visual_embeds: trim video portion using kept_local_all (anchor indices within video segment)
                kept_local_all = torch.cat(all_kept_local_indices)
                if _deepstack_visual_embeds_full is not None:
                    if image_mask is not None:
                        # image+video scenario: need to reorganize from joint embeds based on new visual_pos_masks
                        # Trim video portion of DeepStack, keep image portion unchanged
                        _ds_vid_pruned = [ds_embed[kept_local_all] for ds_embed in (deepstack_video_embeds or [])]
                        _ds_img = deepstack_image_embeds or []
                        _n_ds = max(len(_ds_img), len(_ds_vid_pruned))
                        # Rebuild joint embeds in the compressed sequence
                        _img_mask_compressed = image_mask[:, keep_global_indices, :][..., 0] if image_mask is not None else None
                        _vid_mask_compressed = torch.zeros(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)
                        _vid_mask_compressed[:, video_start:video_start + kept_video_indices.shape[0]] = True
                        deepstack_visual_embeds = []
                        for _di in range(_n_ds):
                            _embed_joint = torch.zeros(visual_pos_masks.sum(), _ds_img[0].shape[-1] if _ds_img else _ds_vid_pruned[0].shape[-1], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                            _img_joint_c = _img_mask_compressed[visual_pos_masks] if _img_mask_compressed is not None else None
                            _vid_joint_c = _vid_mask_compressed[visual_pos_masks]
                            if _di < len(_ds_img) and _img_joint_c is not None:
                                _embed_joint[_img_joint_c] = _ds_img[_di].to(inputs_embeds.dtype)
                            if _di < len(_ds_vid_pruned):
                                _embed_joint[_vid_joint_c] = _ds_vid_pruned[_di].to(inputs_embeds.dtype)
                            deepstack_visual_embeds.append(_embed_joint)
                    else:
                        # Pure video scenario: directly trim using kept_local_all
                        deepstack_visual_embeds = [
                            ds_embed[kept_local_all] for ds_embed in _deepstack_visual_embeds_full
                        ]
                else:
                    deepstack_visual_embeds = None

                # [Approach 1] DeepStack injection strength global scaling lambda (unified config, dataset-independent)
                #   ds_injected = lambda * ds, lambda in [0, 1]
                #   lambda = 1.0 -> full injection (benefits MVBench/MLVU fine-grained/temporal tasks)
                #   lambda = 0.0 -> equivalent to no injection (benefits VideoMME long-video global semantic tasks), bit-identical to pre-modification behavior
                #   0 < lambda < 1 -> single parameter compromise, all datasets share the same lambda
                # Default 0.0: preserves current (no injection) behavior; increasing lambda gradually restores DeepStack injection.
                _ds_scale = float(getattr(td_config, 'deepstack_inject_scale', 0.5)) if td_config is not None else 0.5
                if deepstack_visual_embeds is not None:
                    if _ds_scale == 0.0:
                        deepstack_visual_embeds = None
                    elif _ds_scale != 1.0:
                        deepstack_visual_embeds = [(_ds * _ds_scale) for _ds in deepstack_visual_embeds]

                # Record visual token positions (in the compressed sequence), for LLM pruning use
                n_kept_video = kept_video_indices.shape[0]
                new_v_start = video_start  # prefix length unchanged
                new_v_end = video_start + n_kept_video
                td_config._visual_token_range = (new_v_start, new_v_end)

                _llm_ratio = getattr(td_config, 'llm_prune_ratio', 1.0)
                td_config._target_budget = max(1, int(n_kept_video * _llm_ratio))

                # If there are images, image_mask also needs synchronized trimming to the new sequence length
                if image_mask is not None:
                    image_mask = image_mask[:, keep_global_indices, :]

                pixel_values_videos = None

        else:
            # ===== No compression path =====
            video_embeds_cat = torch.cat(video_embeds_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            n_video_tokens = video_embeds_cat.shape[0]
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds_cat
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_cat)

    # ---- Fallback: if video has not been processed yet ----
    if pixel_values_videos is not None:
        video_embeds_native_list, deepstack_video_embeds, _ = Qwen3VLModel_get_video_features(
            self, pixel_values_videos, video_grid_thw
        )
        video_embeds_native = torch.cat(video_embeds_native_list, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        n_video_tokens = video_embeds_native.shape[0]
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds_native
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds_native)

    # ---- Construct visual_pos_masks (non-compression path only) ----
    # The compression path already set visual_pos_masks and deepstack_visual_embeds in Step 5
    if visual_pos_masks is None:
        deepstack_visual_embeds = None
        if image_mask is not None or video_mask is not None:
            _img_mask_1d = image_mask[..., 0] if image_mask is not None else None
            _vid_mask_1d = video_mask[..., 0] if video_mask is not None else None

            if _img_mask_1d is not None and _vid_mask_1d is not None:
                visual_pos_masks = _img_mask_1d | _vid_mask_1d
                deepstack_visual_embeds = []
                _img_joint = _img_mask_1d[visual_pos_masks]
                _vid_joint = _vid_mask_1d[visual_pos_masks]
                _ds_img = deepstack_image_embeds or []
                _ds_vid = deepstack_video_embeds or []
                _n_ds = max(len(_ds_img), len(_ds_vid))
                for _di in range(_n_ds):
                    _embed_joint = torch.zeros(visual_pos_masks.sum(), _ds_img[0].shape[-1] if _ds_img else _ds_vid[0].shape[-1], device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    if _di < len(_ds_img):
                        _embed_joint[_img_joint] = _ds_img[_di].to(inputs_embeds.dtype)
                    if _di < len(_ds_vid):
                        _embed_joint[_vid_joint] = _ds_vid[_di].to(inputs_embeds.dtype)
                    deepstack_visual_embeds.append(_embed_joint)
            elif _img_mask_1d is not None:
                visual_pos_masks = _img_mask_1d
                deepstack_visual_embeds = deepstack_image_embeds
            elif _vid_mask_1d is not None:
                visual_pos_masks = _vid_mask_1d
                deepstack_visual_embeds = deepstack_video_embeds

    # ---- Position ID computation ----
    if position_ids is None:
        attention_mask_tensor = (
            attention_mask if not isinstance(attention_mask, dict) else attention_mask.get("full_attention", attention_mask)
        )
        if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
            attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
            if attention_mask_tensor.dtype.is_floating_point:
                attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

        prefill_compiled_stage = is_torchdynamo_compiling() and (
            (input_ids is not None and input_ids.shape[1] != 1)
            or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
        )
        prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
            (cache_position is not None and cache_position[0] == 0)
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        )
        if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask_tensor if isinstance(attention_mask_tensor, torch.Tensor) else attention_mask,
            )
            self.rope_deltas = rope_deltas
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    # ---- LLM forward (with DeepStack + pruning) ----
    outputs = Qwen3VLTextModel_forward(
        self.language_model,
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )

    return Qwen3VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        rope_deltas=self.rope_deltas,
    )