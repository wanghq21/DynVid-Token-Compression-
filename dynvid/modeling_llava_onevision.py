"""
modeling_llava_onevision.py

DynVid: Dynamism-Aware Adaptive Video Token Compression - LLaVA-OneVision / LLaVA-Video version.

Architecture notes:
  LLaVA-OneVision and LLaVA-Video share the same `LlavaQwenForCausalLM` model class,
  this file adapts both.

Pipeline:
  SigLIP ViT -> mm_projector -> 2D Pooling (27 * 27 -> 13 * 13 = 169 tokens/frame)
    -> DySeg segmentation -> ERA group budget allocation (Participation Ratio)
    -> DATS anchor selection within groups (Facility Location) -> Top-K soft fusion
    -> Frame-mode newline insertion -> Rebuild embedding list
    -> Qwen2 LLM (standard 1D RoPE)
    -> (optional) LLM internal attention-based hard pruning

Key differences from Qwen2.5-VL version:
  - No M-RoPE (standard 1D RoPE, position_ids is arange)
  - No DeepStack
  - No ViT internal token fusion
  - Tokens inserted into LLM via "rebuild embedding list" instead of masked_scatter
  - Compression function returns keep_visual_indices (not M-RoPE positions)
  - image_newline appended at end of each frame in frame mode

Monkey-patch targets (7):
  1. SigLipAttention.forward           -> returns per-token importance
  2. SigLipVisionTower.forward         -> returns (features, cls_attentions)
  3. LlavaMetaForCausalLM.encode_images -> calls modified ViT
  4. LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal -> inserts DynVid compression
  5. Qwen2Attention.forward            -> pruning layer returns attn_weights
  6. Qwen2DecoderLayer.forward         -> passes attn_weights
  7. Qwen2Model.forward                -> triggers LLM-side pruning
"""

from __future__ import annotations

import logging
import math
import random
import re
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .compression_unified import (
    dynvid_compression_llava,
)

# ============================================================================
# Logger
# ============================================================================
_td_logger = logging.getLogger("dynvid.llava")


# ============================================================================
# LLaVA OneVision: LLM internal attention-based pruning (1D RoPE)
# ============================================================================


def query_guided_pruning_llava(
    hidden_states: Tensor,       # (B, S, D)
    visual_token_range: tuple,   # (start, end)
    target_budget: int,          # final number of visual tokens to keep
    attention_mask: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    cache_position: Optional[Tensor] = None,
    past_key_values=None,
    td_config=None,
    decoder_layer=None,
    position_embeddings=None,
    attn_scores: Optional[Tensor] = None,  # (B, V) produced by decoder layer output_attentions
) -> tuple:
    """LLM internal attention-score-based pruning (LLaVA version).

    Differences from Qwen2.5-VL version:
      - position_ids: 2D (B, S) instead of 3D (B, 3, S)
      - No text_position_ids
      - No M-RoPE section handling
      - cos/sin: (B, S, head_dim) standard 1D RoPE

    When attn_scores is not None, use directly (output_attentions approach);
    otherwise fall back to manual q_proj/k_proj computation (backward compatible).

    Returns:
        (hidden_states, attention_mask, position_ids,
         cache_position, position_embeddings, num_pruned, keep_idx)
    """
    B, S, D_h = hidden_states.shape
    v_start, v_end = visual_token_range
    num_visual = v_end - v_start

    if v_end > S or v_start >= S or num_visual <= 0:
        return hidden_states, attention_mask, position_ids, cache_position, position_embeddings, 0, None

    if num_visual <= target_budget:
        return hidden_states, attention_mask, position_ids, cache_position, position_embeddings, 0, None

    prune_method = "text_token"

    # ========================================================================
    # Prefer externally provided attn_scores (output_attentions approach)
    # ========================================================================
    if attn_scores is not None:
        scores = attn_scores  # (B, V) - already correctly computed by decoder layer
        # _td_logger.info(
        #     f"[QueryPrune-LLaVA] Using decoder-layer attn_scores: "
        #     f"[{scores.min().item():.6f}, {scores.max().item():.6f}]"
        # )
    elif decoder_layer is not None and position_embeddings is not None:
        self_attn = decoder_layer.self_attn
        head_dim = self_attn.head_dim
        num_heads = self_attn.config.num_attention_heads
        num_kv_heads = self_attn.config.num_key_value_heads
        num_kv_groups = num_heads // num_kv_heads

        with torch.no_grad():
            if prune_method == "text_token":
                text_prefix_h = hidden_states[:, :v_start, :]
                text_suffix_h = hidden_states[:, v_end:, :]
                if text_prefix_h.shape[1] + text_suffix_h.shape[1] == 0:
                    return hidden_states, attention_mask, position_ids, cache_position, position_embeddings, 0, None
                query_hidden = torch.cat([text_prefix_h, text_suffix_h], dim=1)
            else:
                query_hidden = hidden_states

            visual_hidden = hidden_states[:, v_start:v_end, :]

            q_states = self_attn.q_proj(query_hidden)
            k_states = self_attn.k_proj(visual_hidden)

            q_len = q_states.shape[1]
            v_len = k_states.shape[1]
            q_states = q_states.view(B, q_len, num_heads, head_dim).transpose(1, 2)
            k_states = k_states.view(B, v_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            cos, sin = position_embeddings
            if cos.ndim == 3:
                if prune_method == "text_token":
                    q_cos = torch.cat([cos[:, :v_start, :], cos[:, v_end:, :]], dim=1)
                    q_sin = torch.cat([sin[:, :v_start, :], sin[:, v_end:, :]], dim=1)
                else:
                    q_cos = cos
                    q_sin = sin
                k_cos = cos[:, v_start:v_end, :]
                k_sin = sin[:, v_start:v_end, :]
            elif cos.ndim == 2:
                if prune_method == "text_token":
                    q_cos = torch.cat([cos[:v_start, :], cos[v_end:, :]], dim=0).unsqueeze(0)
                    q_sin = torch.cat([sin[:v_start, :], sin[v_end:, :]], dim=0).unsqueeze(0)
                else:
                    q_cos = cos.unsqueeze(0)
                    q_sin = sin.unsqueeze(0)
                k_cos = cos[v_start:v_end, :].unsqueeze(0)
                k_sin = sin[v_start:v_end, :].unsqueeze(0)
            else:
                q_cos = k_cos = cos
                q_sin = k_sin = sin

            # GQA: repeat K
            from transformers.models.qwen2.modeling_qwen2 import repeat_kv as _repeat_kv
            if num_kv_groups > 1:
                k_states = _repeat_kv(k_states, num_kv_groups)

            scale = 1.0 / math.sqrt(head_dim)
            attn_logits = torch.matmul(q_states, k_states.transpose(-2, -1)) * scale
            attn_weights = attn_logits.softmax(dim=-1)

            # max over query, mean over heads
            scores = attn_weights.max(dim=2).values.mean(dim=1)  # (B, V)

    else:
        # Fallback: cosine similarity
        _td_logger.warning("[QueryPrune] fallback to cosine similarity")
        visual_hidden = hidden_states[:, v_start:v_end, :]
        text_all = torch.cat([hidden_states[:, :v_start, :], hidden_states[:, v_end:, :]], dim=1)
        if text_all.shape[1] == 0:
            return hidden_states, attention_mask, position_ids, cache_position, position_embeddings, 0, None
        query = text_all.mean(dim=1, keepdim=True)
        query_normed = F.normalize(query, dim=-1)
        visual_normed = F.normalize(visual_hidden, dim=-1)
        scores = (visual_normed * query_normed).sum(dim=-1)


    # Top-k selection
    k = min(target_budget, num_visual)
    _, top_indices = scores.topk(k, dim=1)
    top_indices = top_indices.sort(dim=1).values
    keep_idx = top_indices[0]

    kept_visual = hidden_states[0, v_start + keep_idx, :].unsqueeze(0)
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

    # Update position_ids (1D RoPE)
    new_position_ids = position_ids
    if position_ids is not None:
        if position_ids.ndim == 2:
            # Standard 1D: (B, S)
            prefix_pos = position_ids[:, :v_start]
            suffix_pos = position_ids[:, v_end:]
            visual_pos = position_ids[:, v_start:v_end]
            kept_pos = visual_pos[:, keep_idx]
            new_position_ids = torch.cat([prefix_pos, kept_pos, suffix_pos], dim=1)
        elif position_ids.ndim == 3:
            # Safety: 3D M-RoPE (shouldn't happen for LLaVA, but handle gracefully)
            prefix_pos = position_ids[:, :, :v_start]
            suffix_pos = position_ids[:, :, v_end:]
            visual_pos = position_ids[:, :, v_start:v_end]
            kept_pos = visual_pos[:, :, keep_idx]
            new_position_ids = torch.cat([prefix_pos, kept_pos, suffix_pos], dim=2)

    # Update cache_position
    new_cache_position = cache_position
    if cache_position is not None:
        new_cache_position = torch.arange(0, new_hidden.shape[1], device=cache_position.device)

    # Update position_embeddings
    new_pos_emb = position_embeddings
    if position_embeddings is not None:
        cos, sin = position_embeddings
        # Rebuild by keeping non-visual + kept visual + suffix
        keep_global = torch.cat([
            torch.arange(v_start, device=keep_idx.device),
            v_start + keep_idx,
            torch.arange(v_end, S, device=keep_idx.device),
        ])
        if cos.ndim == 3:
            new_cos = cos[:, keep_global, :]
            new_sin = sin[:, keep_global, :]
        elif cos.ndim == 2:
            new_cos = cos[keep_global, :]
            new_sin = sin[keep_global, :]
        else:
            new_cos = cos
            new_sin = sin
        new_pos_emb = (new_cos, new_sin)

    # Clean KV cache
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

    return new_hidden, new_attention_mask, new_position_ids, new_cache_position, new_pos_emb, num_pruned, keep_idx




# ============================================================================
# ---  Monkey-patch functions: SigLIP ViT
# ============================================================================

@torch.no_grad()
def SigLipVisionTower_forward(self, images: torch.Tensor):
    """Modified SigLipVisionTower.forward: returns (features, cls_attentions)."""
    if not isinstance(images, torch.Tensor):
        raise ValueError(f"Unexpected data type: {type(images)}")
    image_forward_outs = self.vision_tower(
        images.to(device=self.device, dtype=self.dtype),
        output_attentions=True,
        output_hidden_states=True,
    )
    image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
    cls_attentions = image_forward_outs.attentions[-1].to(images.dtype)
    assert image_features.shape[-2] == 729, (
        f"Expected 729 patches, got {image_features.shape[-2]}"
    )
    return image_features, cls_attentions


def SigLipAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Modified SigLipAttention.forward: returns per-token importance.
    Supports two methods:
      - col_mean: mean over heads, mean over queries (original method)
      - row_topk: each query votes for top-k keys, counts votes
    """
    batch_size, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    k_v_seq_len = key_states.shape[-2]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)

    SIGLIP_CLS_ATTN_METHOD = "col_mean_debiased"   # change method here: col_mean / col_mean_debiased
    # --- Compute per-token CLS attention ---
    if SIGLIP_CLS_ATTN_METHOD == "col_mean_debiased":
        raw = attn_weights.mean(1).mean(1)  # (B, 729)
        # positional bias = mean over all frames (content-independent fixed pattern)
        bias = raw.mean(dim=0, keepdim=True)  # (1, 729)
        importance = (raw - bias).clamp(min=0)
        importance = importance / importance.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    else:
        # col_mean: original method
        importance = attn_weights.mean(1).mean(1)

    return attn_output, importance


# ============================================================================
# ---  Monkey-patch functions: LlavaMetaForCausalLM
# ============================================================================

def LlavaMetaForCausalLM_encode_images(self, images: torch.Tensor):
    """Call modified ViT, return (features, cls_attentions)."""
    image_features, cls_attentions = self.get_model().get_vision_tower()(images)
    image_features = self.get_model().mm_projector(image_features)
    return image_features, cls_attentions


def LlavaMetaForCausalLM_prepare_inputs_labels_for_multimodal(
    self,
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    labels,
    images,
    modalities=["image"],
    image_sizes=None,
):
    """Core: insert DynVid compression.

    Modifications:
      1. Video path: use dynvid_compression_llava
      2. Frame-mode newline: use keep_visual_indices for per-frame allocation
      3. Record visual_token_start_index and visual_token_length for LLM pruning
    """
    from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
    from llava.mm_utils import get_anyres_image_grid_shape
    from llava.model.llava_arch import unpad_image
    from llava.utils import rank0_print

    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if isinstance(modalities, str):
        modalities = [modalities]

    td_config = getattr(self, 'td_config', None)

    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

        video_idx_in_batch = []
        for _ in range(len(modalities)):
            if modalities[_] == "video":
                video_idx_in_batch.append(_)

        images_list = []
        for image in images:
            if image.ndim == 4:
                images_list.append(image)
            else:
                images_list.append(image.unsqueeze(0))

        concat_images = torch.cat([image for image in images_list], dim=0)
        split_sizes = [image.shape[0] for image in images_list]
        encoded_image_features, cls_attentions = self.encode_images(concat_images)

        encoded_image_features = torch.split(encoded_image_features, split_sizes)
        image_features = []
        assert len(encoded_image_features) == 1, "Only support single video in a batch for now."

        # Store compression results for frame-mode newline
        _compression_results = {}

        for idx, image_feat in enumerate(encoded_image_features):
            if idx in video_idx_in_batch and td_config is not None:
                # ---- DynVid compression ----
                visual_token_start_index = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0].item()

                pooled_image_feature = self.get_2dPool(image_feat)
                pooled_cls_attentions = self.get_2dPool(
                    cls_attentions[:split_sizes[idx]].unsqueeze(-1)
                ).squeeze(-1)

                num_frames, num_visual_tokens = pooled_image_feature.shape[:2]

                # Extract text embeddings for query bonus
                text_token_mask = (input_ids[0] != IMAGE_TOKEN_INDEX)
                if text_token_mask.any():
                    text_ids = input_ids[0][text_token_mask]
                    text_embeds = self.get_model().embed_tokens(text_ids)
                    td_config._text_embeds = text_embeds
                else:
                    td_config._text_embeds = None

                compressed_visual_tokens, keep_visual_indices = dynvid_compression_llava(
                    video_embeds=pooled_image_feature,
                    cls_attention=pooled_cls_attentions,
                    num_frames=num_frames,
                    tokens_per_frame=num_visual_tokens,
                    config=td_config,
                )

                _compression_results[idx] = {
                    'compressed_visual_tokens': compressed_visual_tokens,
                    'keep_visual_indices': keep_visual_indices,
                    'pooled_image_feature': pooled_image_feature,
                    'num_frames': num_frames,
                    'num_visual_tokens': num_visual_tokens,
                    'visual_token_start_index': visual_token_start_index,
                }

                image_features.append(compressed_visual_tokens)
            elif idx in video_idx_in_batch:
                # No td_config: no compression, direct 2D pooling
                pooled_image_feature = self.get_2dPool(image_feat)
                image_features.append(pooled_image_feature)
            else:
                image_features.append(image_feat)

        mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
        mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

        if mm_patch_merge_type == "flat":
            image_features = [x.flatten(0, 1) for x in image_features]

        elif mm_patch_merge_type.startswith("spatial"):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_idx in video_idx_in_batch:
                    if mm_newline_position == "grid":
                        if image_idx in _compression_results:
                            # Grid mode incompatible with compressed 1D tokens, fallback to frame-mode newline
                            cr = _compression_results[image_idx]
                            compressed_visual_tokens = cr['compressed_visual_tokens']
                            keep_visual_indices = cr['keep_visual_indices']
                            num_frames = cr['num_frames']
                            num_visual_tokens = cr['num_visual_tokens']

                            compressed_visual_token_list = []
                            for frame_idx in range(num_frames):
                                start_idx = frame_idx * num_visual_tokens
                                end_idx = start_idx + num_visual_tokens
                                ind = torch.where(
                                    (keep_visual_indices >= start_idx) & (keep_visual_indices < end_idx)
                                )[0]
                                frame_visual_tokens = compressed_visual_tokens[ind]
                                frame_visual_tokens = torch.cat(
                                    (frame_visual_tokens,
                                     self.model.image_newline[None].to(image_feature.device)),
                                    dim=0,
                                )
                                compressed_visual_token_list.append(frame_visual_tokens)

                            image_feature = torch.cat(compressed_visual_token_list, dim=0)
                            if td_config is not None:
                                td_config._visual_token_range = (
                                    cr['visual_token_start_index'],
                                    cr['visual_token_start_index'] + image_feature.shape[0],
                                )
                                td_config._visual_token_length = image_feature.shape[0]
                        else:
                            image_feature = self.add_token_per_grid(image_feature)
                        new_image_features.append(image_feature)

                    elif mm_newline_position == "frame":
                        # ---- Frame-mode newline: append image_newline at end of each frame ----
                        if image_idx in _compression_results:
                            cr = _compression_results[image_idx]
                            compressed_visual_tokens = cr['compressed_visual_tokens']
                            keep_visual_indices = cr['keep_visual_indices']
                            num_frames = cr['num_frames']
                            num_visual_tokens = cr['num_visual_tokens']

                            compressed_visual_token_list = []
                            for frame_idx in range(num_frames):
                                start_idx = frame_idx * num_visual_tokens
                                end_idx = start_idx + num_visual_tokens
                                ind = torch.where(
                                    (keep_visual_indices >= start_idx) & (keep_visual_indices < end_idx)
                                )[0]
                                frame_visual_tokens = compressed_visual_tokens[ind]
                                frame_visual_tokens = torch.cat(
                                    (frame_visual_tokens,
                                     self.model.image_newline[None].to(image_feature.device)),
                                    dim=0,
                                )
                                compressed_visual_token_list.append(frame_visual_tokens)

                            image_feature = torch.cat(compressed_visual_token_list, dim=0)

                            # Update visual token info in td_config for LLM pruning
                            if td_config is not None:
                                td_config._visual_token_range = (
                                    cr['visual_token_start_index'],
                                    cr['visual_token_start_index'] + image_feature.shape[0],
                                )
                                td_config._visual_token_length = image_feature.shape[0]
                        else:
                            # Uncompressed: original frame-mode (flatten)
                            image_feature = image_feature.flatten(0, 1)

                        new_image_features.append(image_feature)

                    elif mm_newline_position == "one_token":
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (image_feature, self.model.image_newline[None].to(image_feature.device)),
                                dim=0,
                            )
                        # Update visual token info after compression for LLM pruning
                        if image_idx in _compression_results and td_config is not None:
                            cr = _compression_results[image_idx]
                            td_config._visual_token_range = (
                                cr['visual_token_start_index'],
                                cr['visual_token_start_index'] + image_feature.shape[0],
                            )
                            td_config._visual_token_length = image_feature.shape[0]
                        new_image_features.append(image_feature)

                    elif mm_newline_position == "no_token":
                        if image_idx in _compression_results:
                            # Already 1D after compression, no need to flatten
                            if td_config is not None:
                                cr = _compression_results[image_idx]
                                td_config._visual_token_range = (
                                    cr['visual_token_start_index'],
                                    cr['visual_token_start_index'] + image_feature.shape[0],
                                )
                                td_config._visual_token_length = image_feature.shape[0]
                        else:
                            image_feature = image_feature.flatten(0, 1)
                        new_image_features.append(image_feature)
                    else:
                        raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")

                elif image_feature.shape[0] > 1:
                    # Multi-image operations (keep original logic)
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]

                    if "anyres_max" in image_aspect_ratio:
                        matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                        if matched_anyres_max_num_patches:
                            max_num_patches = int(matched_anyres_max_num_patches.group(1))

                    if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                        if hasattr(self.get_vision_tower(), "image_size"):
                            vision_tower_image_size = self.get_vision_tower().image_size
                        else:
                            raise ValueError("vision_tower_image_size not found")
                        try:
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size
                            )
                        except Exception as e:
                            rank0_print(f"Error: {e}")
                            num_patch_width, num_patch_height = 2, 2
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        image_feature = image_feature.view(2, 2, height, width, -1)

                    if "maxpool2x2" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = nn.functional.max_pool2d(image_feature, 2)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                        unit = image_feature.shape[2]
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        c, h, w = image_feature.shape
                        times = math.sqrt(h * w / (max_num_patches * unit**2))
                        if times > 1.1:
                            image_feature = image_feature[None]
                            image_feature = nn.functional.interpolate(
                                image_feature, [int(h // times), int(w // times)], mode="bilinear"
                            )[0]
                        image_feature = torch.cat(
                            (image_feature,
                             self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    elif "unpad" in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat(
                            (image_feature,
                             self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)),
                            dim=-1,
                        )
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)

                    if "nobase" not in mm_patch_merge_type:
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    new_image_features.append(image_feature)
                else:
                    image_feature = image_feature[0]
                    if "unpad" in mm_patch_merge_type:
                        image_feature = torch.cat(
                            (image_feature, self.model.image_newline[None]),
                            dim=0,
                        )
                    new_image_features.append(image_feature)

            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        image_features = self.encode_images(images)

    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
        raise NotImplementedError

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0

    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                try:
                    cur_image_features = image_features[cur_image_idx]
                except IndexError:
                    cur_image_features = image_features[cur_image_idx - 1]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(
                    torch.full((cur_image_features.shape[0],), IGNORE_INDEX,
                               device=cur_labels.device, dtype=cur_labels.dtype)
                )

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)

    tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
    new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
    new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, "tokenizer_padding_side", "right") == "left":
            new_input_embeds_padded.append(
                torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                           cur_new_embed), dim=0)
            )
            if cur_len > 0:
                new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(
                torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                                       dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
            )
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = new_labels_padded

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    if getattr(self.config, "use_pos_skipping", False) and self.training:
        position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0)
        split_position = random.randint(0, new_input_embeds.size(1))
        left_add = random.randint(0, self.config.pos_skipping_range)
        right_add = random.randint(left_add, self.config.pos_skipping_range)
        position_ids[:, :split_position] += left_add
        position_ids[:, split_position:] += right_add

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


# ============================================================================
# ---  Monkey-patch functions: Qwen2 LLM
# ============================================================================

def Qwen2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple,
    attention_mask: Optional[torch.Tensor],
    past_key_values=None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple:
    """Qwen2Attention.forward: return attn_weights at pruning layer."""
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb,
        repeat_kv,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # Use eager or flash attention
    attention_interface = None
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation in ALL_ATTENTION_FUNCTIONS:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    if attention_interface is None:
        from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
        attention_interface = eager_attention_forward

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=getattr(self, 'sliding_window', None),
        **kwargs,
    )

    # If output_attentions needed but FA2 did not return them
    if kwargs.get("output_attentions", False) and attn_weights is None:
        last_query = query_states[:, :, -1:, :]
        key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
        attn_weights = torch.matmul(last_query, key_states_expanded.transpose(2, 3)) * self.scaling
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def Qwen2DecoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
    """Qwen2DecoderLayer.forward: return attn_weights to the next layer."""
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


def Qwen2Model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
):
    """Qwen2Model.forward: hard pruning trigger (query_guided_pruning_llava)."""
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # Causal mask
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if getattr(self, 'has_sliding_layers', False):
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # Get td_config
    td_config = getattr(self, 'td_config', None)
    is_prefill = hidden_states.shape[1] > 1
    _qp_layer = getattr(td_config, 'query_prune_layer', -1) if td_config else -1
    llm_prune_ratio = getattr(td_config, 'llm_prune_ratio', 1.0) if td_config else 1.0

    # Get visual_token_range
    visual_token_range = getattr(td_config, '_visual_token_range', None) if td_config else None

    causal_mask = causal_mask_mapping.get("full_attention", None)


    for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):

        hidden_states, attn_weights = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # --- Hard Pruning: query-guided physical token removal ---
        if (
            layer_idx == _qp_layer
            and is_prefill
            and visual_token_range is not None
            and hidden_states.shape[1] > visual_token_range[1]
        ):
            v_start, v_end = visual_token_range
            num_visual = v_end - v_start

            if llm_prune_ratio < 1.0 and num_visual > 0:
                target_budget = max(1, int(num_visual * llm_prune_ratio))

                (
                    hidden_states,
                    attention_mask_updated,
                    position_ids,
                    cache_position,
                    position_embeddings,
                    num_pruned,
                    keep_idx,
                ) = query_guided_pruning_llava(
                    hidden_states=hidden_states,
                    visual_token_range=visual_token_range,
                    target_budget=target_budget,
                    attention_mask=None,  # use causal_mask
                    position_ids=position_ids,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    td_config=td_config,
                    decoder_layer=decoder_layer,
                    position_embeddings=position_embeddings,
                )

                if num_pruned > 0:
                    # Update visual_token_range
                    new_v_end = v_start + (num_visual - num_pruned)
                    td_config._visual_token_range = (v_start, new_v_end)
                    visual_token_range = (v_start, new_v_end)

                    # Rebuild causal_mask
                    new_seq_len = hidden_states.shape[1]
                    if causal_mask is not None:
                        causal_mask = causal_mask[:, :, :new_seq_len, :new_seq_len]

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )