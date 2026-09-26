"""
compression_unified.py

DynVid: Dynamism-Aware Adaptive Video Token Compression — Unified Compression Module.
Shared by modeling_qwen3_vl.py, modeling_qwen2_5_vl.py, and modeling_llava_onevision.py.

Pipeline:
  Post-projector signal computation -> DySeg grouping -> ERA group budget allocation (Participation Ratio)
  -> Intra-group DATS anchor selection (Facility Location) -> Top-K soft fusion
  -> (Optional) LLM internal query-guided hard pruning
"""

from typing import Callable, Optional, Union, List, Tuple
import numpy as np
import logging
import math
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import tqdm
from torch import Tensor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv


class TqdmHandler(logging.Handler):
    """Output via tqdm.write to avoid being overwritten by the progress bar."""
    def emit(self, record):
        msg = self.format(record)
        tqdm.tqdm.write(msg)

_td_logger = logging.getLogger("dynvid")


def _td_normalize(x: torch.Tensor) -> torch.Tensor:
    """Z-score + sigmoid normalize to ~[0, 1].
    """
    mu = x.mean()
    sigma = x.std()
    if sigma < 1e-8:
        return torch.full_like(x, 0.5)  # No discriminability -> neutral value
    return torch.sigmoid((x - mu) / sigma)

def _td_normalize_batched(x: torch.Tensor) -> torch.Tensor:
    """Per-group version of _td_normalize: x (B, P) -> (B, P), independent z-score + sigmoid per group.

    Exactly equivalent to _td_normalize (normalized along group dimension dim=1):
      Groups with sigma < 1e-8 degenerate to neutral value 0.5.
    """
    mu = x.mean(dim=1, keepdim=True)
    sigma = x.std(dim=1, keepdim=True)
    out = torch.sigmoid((x - mu) / sigma)
    flat = (sigma < 1e-8).expand_as(out)
    out = torch.where(flat, torch.full_like(out, 0.5), out)
    return out


_td_logger.setLevel(logging.INFO)
_td_logger.propagate = False
_local_rank = int(os.environ.get("LOCAL_RANK", 0))
if not _td_logger.handlers and _local_rank == 0:
    _td_tqdm_handler = TqdmHandler()
    _td_tqdm_handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    )
    _td_logger.addHandler(_td_tqdm_handler)



# ============================================================================
# Core Algorithm Functions
# ============================================================================

def _facility_location_select(
    features: torch.Tensor,
    importance: torch.Tensor,
    k: int,
    init_selected: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Facility Location greedy, (1-1/e) guarantee, reduced GPU-CPU synchronization."""
    N, D = features.shape
    if k >= N:
        return torch.arange(N, device=features.device)

    feat_norm = F.normalize(features.float(), dim=-1)
    imp = _td_normalize(importance.float()).clamp(min=1e-4)

    # Similarity matrix
    sim = feat_norm @ feat_norm.T
    weighted_sim = sim * imp.unsqueeze(1)  # (N, N)

    # Current coverage value
    current_max = torch.zeros(N, device=features.device)

    # Store selection results in tensor, no CPU transfer
    selected = torch.empty(k, dtype=torch.long, device=features.device)
    mask = torch.zeros(N, device=features.device, dtype=torch.bool)

    # Handle pre-selected seeds
    n_init = 0
    if init_selected is not None and init_selected.numel() > 0:
        n_init = init_selected.shape[0]
        for i in range(n_init):
            idx = init_selected[i]
            selected[i] = idx
            mask[idx] = True
            current_max = torch.max(current_max, weighted_sim[:, idx])

    # Greedy loop - all on GPU, no .item() calls
    for step in range(n_init, k):
        # Marginal gain
        marginal_gain = (weighted_sim - current_max.unsqueeze(1)).clamp(min=0).sum(dim=0)
        marginal_gain[mask] = -1.0

        # argmax stays on GPU
        best = marginal_gain.argmax()
        selected[step] = best
        mask[best] = True

        # Update coverage
        current_max = torch.max(current_max, weighted_sim[:, best])

    return selected[:k].sort().values



def _facility_location_select_batched(
    feats: torch.Tensor,
    imp: torch.Tensor,
    K_list: torch.Tensor,
    init_selected: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched exact greedy Facility Location, parallel computation for equal-sized groups.
    Strictly bit-equivalent to per-group _facility_location_select (exact greedy).
    """
    B, P, D = feats.shape
    device = feats.device

    # ---- Reorder by K descending (active groups -> prefix) ----
    order = torch.argsort(K_list, descending=True)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(B, device=device)
    feats = feats[order]
    imp = imp[order]
    K_sorted = K_list[order]
    if init_selected is not None and init_selected.numel() > 0:
        init_selected = init_selected[order]

    fn = F.normalize(feats.float(), dim=-1)
    impn = _td_normalize_batched(imp.float()).clamp(min=1e-4)
    W = torch.bmm(fn, fn.transpose(1, 2)) * impn.unsqueeze(2)  # (B, P, P)

    cur_max = torch.zeros(B, P, device=device)
    mask = torch.zeros(B, P, dtype=torch.bool, device=device)
    K_max = int(K_sorted[0].item())                   # After descending sort, index 0 is the maximum
    sel = torch.empty(B, K_max, dtype=torch.long, device=device)
    arangeB = torch.arange(B, device=device)

    # ---- Pre-selected seeds: all groups have the same n_init, fill in entirely ----
    n_init = 0
    if init_selected is not None and init_selected.numel() > 0:
        n_init = init_selected.shape[1]
        for i in range(n_init):
            idx = init_selected[:, i]
            sel[:, i] = idx
            mask[arangeB, idx] = True
            cur_max = torch.max(cur_max, W[arangeB, :, idx])

    # ---- Greedy: step-th iteration only processes prefix groups with K_b > step ----
    K_cpu = K_sorted.tolist()        # CPU-side B_active advancement, avoids per-step GPU sync
    B_active = B
    for step in range(n_init, K_max):
        while B_active > 0 and K_cpu[B_active - 1] <= step:
            B_active -= 1
        if B_active == 0:
            break
        ar = arangeB[:B_active]
        Wa = W[:B_active]                                       # (Ba, P, P)
        cur_a = cur_max[:B_active]                             # (Ba, P)
        marginal = (Wa - cur_a.unsqueeze(2)).clamp(min=0).sum(dim=1)  # (Ba, P)
        marginal[mask[:B_active]] = -1.0
        best = marginal.argmax(dim=1)                          # (Ba,)
        sel[:B_active, step] = best
        mask[ar, best] = True
        cur_max[:B_active] = torch.max(cur_a, W[ar, :, best])

    return sel[inv]   # Restore input order




def _topk_importance_select(
    features: torch.Tensor,          # Only for signature alignment, Top-K baseline does not use this
    importance: torch.Tensor,
    k: int,
    init_selected: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pure importance Top-K baseline, strictly aligned with _facility_location_select interface/return.
    """
    N, D = features.shape
    if k >= N:
        return torch.arange(N, device=features.device)

    # importance is monotonic -> topk only cares about ordering, _td_normalize does not change ranking, so normalization is skipped
    imp = importance.float().clone()                 # clone: in-place masking of seeds below
    device = features.device

    selected = torch.empty(k, dtype=torch.long, device=device)

    n_init = 0
    if init_selected is not None and init_selected.numel() > 0:
        n_init = init_selected.shape[0]
        selected[:n_init] = init_selected
        # Set importance of already-selected seeds to -inf to prevent re-selection during fill phase
        imp[init_selected] = float("-inf")

    if k > n_init:
        _, idx = imp.topk(k - n_init)                # Fill remaining slots, descending
        selected[n_init:] = idx

    return selected[:k].sort().values



@torch.no_grad()
def _topk_trash_fuse(
    all_feats: torch.Tensor,        # (P, D)
    all_pos: torch.Tensor,          # (P,) 1D global index (LLaVA) or (3, P) M-RoPE (Qwen)
    anchor_indices: torch.Tensor,   # (K,)
    top_k: int = 10,
    anchor_weight: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Top-K soft assignment + mean fusion.
    fused = alpha * anchor + (1-alpha) * drop_centroid
    """
    P, D = all_feats.shape
    K = anchor_indices.shape[0]
    device = all_feats.device

    is_anchor = torch.zeros(P, dtype=torch.bool, device=device)
    is_anchor[anchor_indices] = True

    anchor_feats = all_feats[anchor_indices].float()         # (K, D)
    _pos_2d = (all_pos.dim() == 2)
    anchor_pos = all_pos[:, anchor_indices] if _pos_2d else all_pos[anchor_indices]

    drop_mask = ~is_anchor
    drop_mask_indices = drop_mask.nonzero(as_tuple=True)[0]  # (N_drop,)
    N_drop = drop_mask_indices.shape[0]

    if N_drop == 0:
        return anchor_feats.to(all_feats.dtype), anchor_pos

    drop_feats = all_feats[drop_mask_indices].float()        # (N_drop, D)

    # ---- Step 1: drop -> anchor similarity ----
    drop_norm = F.normalize(drop_feats, dim=-1)
    anchor_norm = F.normalize(anchor_feats, dim=-1)
    sims = drop_norm @ anchor_norm.T                         # (N_drop, K)

    # ---- Step 2: Top-K ----
    actual_k = min(top_k, K)
    topk_sims, topk_ids = sims.topk(actual_k, dim=1)        # (N_drop, actual_k)

    # ---- Step 3: Softmax (only over top-k anchors) ----
    drop_anchor_weights = F.softmax(topk_sims / 0.01, dim=1)  # (N_drop, actual_k)

    P_mat = torch.zeros(N_drop, K, device=device, dtype=torch.float32)
    P_mat.scatter_(1, topk_ids, drop_anchor_weights)

    # ---- Step 4: mean fusion ----
    weighted_drop = P_mat.T @ drop_feats                    # (K, D)
    drop_weight_sum = P_mat.sum(dim=0)                      # (K,)
    safe_sum = drop_weight_sum.clamp(min=1e-8)
    drop_centroid = weighted_drop / safe_sum.unsqueeze(-1)
    has_drop = (drop_weight_sum > 1e-8).float().unsqueeze(-1)
    fused = has_drop * (anchor_weight * anchor_feats + (1.0 - anchor_weight) * drop_centroid) \
            + (1.0 - has_drop) * anchor_feats

    return fused.to(all_feats.dtype), anchor_pos




def _dyseg_group_frames(frame_features_norm: List[torch.Tensor],
                        threshold: float,
                        min_segment_num: int = 0,
                        complementary_segment: bool = True) -> List[List[int]]:
    """DySeg: Group adjacent frames by cosine similarity, with minimum segment count constraint.

    Adjacent frames whose mean feature cosine similarity > threshold -> same group.
    If the number of groups < min_segment_num and complementary_segment=True,
    select the positions with lowest similarity among remaining positions to add cuts,
    until the minimum segment count is satisfied.

    Args:
        frame_features_norm: List of (K_t, D) normalized frame features.
        threshold: Similarity threshold; higher values produce finer grouping.
        min_segment_num: Minimum number of segments, 0 means no constraint.
        complementary_segment: Whether to automatically add complementary cuts when below min_segment_num.

    Returns:
        groups: List of List[int], frame indices contained in each group.
    """
    T = len(frame_features_norm)
    if T <= 1:
        return [list(range(T))]

    # Compute mean feature for each frame
    frame_means = torch.stack([f.mean(dim=0) for f in frame_features_norm])  # (T, D)
    frame_means = F.normalize(frame_means, dim=-1)

    # Adjacent frame similarities (T-1,)
    transition_sims = F.cosine_similarity(frame_means[:-1], frame_means[1:], dim=-1)
    transition_sims = _td_normalize(transition_sims)

    # Step 1: Threshold-based splitting - cut at positions where similarity < threshold
    cut_indices = (transition_sims < threshold).nonzero(as_tuple=True)[0].tolist()

    # Step 2: Complementary cuts - when segment count is below min_segment_num, select remaining positions with lowest similarity
    num_segments = len(cut_indices) + 1
    if min_segment_num > 0 and num_segments < min_segment_num and complementary_segment:
        num_needed = min_segment_num - num_segments
        # Set similarity of already-cut positions to 1.0 (exclude them)
        remaining_sims = transition_sims.clone()
        for idx in cut_indices:
            remaining_sims[idx] = 1.0
        # Select top-K positions with lowest similarity from remaining
        k = min(num_needed, remaining_sims.shape[0])
        if k > 0:
            extra_indices = torch.topk(remaining_sims, k=k, largest=False).indices.tolist()
            cut_indices = sorted(set(cut_indices + extra_indices))

    # Step 3: Build groups from cut points
    cut_indices_sorted = sorted(cut_indices)
    groups = []
    prev = 0
    for c in cut_indices_sorted:
        # cut at position c means: frames [prev..c] are one group, [c+1..] starts next
        groups.append(list(range(prev, c + 1)))
        prev = c + 1
    groups.append(list(range(prev, T)))

    return groups



def _compute_group_budget(
    groups: List[List[int]],
    frame_features_norm: List[torch.Tensor],  # list of (N, D) normalized
    total_budget: int,
    method: str,  #  "uniform", "pr"
    N_real: List[int],
    device=None,
) -> List[int]:
    """ERA (Effective Rank-based Adaptive Allocation): Compute token budget for each group.

    Supported methods:
        Uniform:
            group_weights = P, i.e., allocate by total token count per group, equivalent to per-token uniform budget.

        PR (Participation Ratio, no SVD):
            Dimensionality term PR = tr(G)^2/||G||_F^2 (= second-moment approximation of singular value entropy, only one matrix multiply),

    Args:
        groups: DySeg grouping result
        frame_features_norm: Normalized features per frame
        total_budget: Total token budget
        method: "uniform" / "pr"
        N_real: Token count per frame
        device: Computation device

    Returns:
        group_budgets: Token budget per group (integer list, sum = total_budget)
    """
    num_groups = len(groups)
    group_weights = torch.zeros(num_groups, device=device)

    # ---- Single pass to collect token pool per group (cat is lightweight, no decomposition) ----
    pool_feats_list = []
    P_list = []
    for group in groups:
        pool_feats = torch.cat([frame_features_norm[t] for t in group], dim=0)  # (P, D)
        pool_feats_list.append(pool_feats)
        P_list.append(pool_feats.shape[0])

    if method == "uniform":
        for i in range(num_groups):
            group_weights[i] = 1.0 if P_list[i] <= 1 else P_list[i]
    elif method == "pr":
        D = pool_feats_list[0].shape[1]
        alpha = 0.05
        for i in range(num_groups):
            if P_list[i] <= 1:
                group_weights[i] = 1.0
        buckets: dict = {}
        for i in range(num_groups):
            if P_list[i] > 1:
                buckets.setdefault(P_list[i], []).append(i)
        for P_val, ids in buckets.items():
            B = len(ids)
            feats_bucket = torch.stack(
                [pool_feats_list[i].float() for i in ids], dim=0
            )  # (B, P_val, D)
            # Take the smaller-side Gram (non-zero spectrum is identical, just to save compute)
            if P_val < D:
                gram = torch.bmm(feats_bucket, feats_bucket.transpose(1, 2))  # (B, P, P)
            else:
                gram = torch.bmm(feats_bucket.transpose(1, 2), feats_bucket)  # (B, D, D)
            fro2 = (gram * gram).flatten(1).sum(dim=1)                 # (B,) = sum_lambda^2
            tr = torch.diagonal(gram, dim1=1, dim2=2).sum(dim=1)       # (B,) = sum_lambda (approx P)
            pr = (tr * tr) / (fro2 + 1e-8)                             # (B,) in [1, min(P,D)]
            w = (1.0 - alpha) * pr + alpha * float(P_val)       # (B,)
            for b, i in enumerate(ids):
                group_weights[i] = w[b]

    else:
        raise NotImplementedError(
            f"group_budget method '{method}' is not supported. "
            f"Only 'uniform' / 'pr' are supported."
        )

    # Normalize and allocate budget
    if group_weights.sum() <= 0:
        group_weights = torch.ones(num_groups, device=device)
    ratio = group_weights / group_weights.sum()
    group_budgets_float = ratio * total_budget
    group_budgets = group_budgets_float.floor().int().tolist()

    # Ensure each group gets at least 1 token
    group_budgets = [max(1, b) for b in group_budgets]
    # Limit each group's budget to not exceed its actual token count
    for g_idx, group in enumerate(groups):
        max_tokens = sum(N_real[t] for t in group)
        group_budgets[g_idx] = min(group_budgets[g_idx], max_tokens)

    # Allocate remainder (by descending weight)
    remainder = total_budget - sum(group_budgets)
    sorted_groups = sorted(range(num_groups), key=lambda i: -group_weights[i].item())
    idx = 0
    while remainder > 0 and idx < num_groups * 10:
        g = sorted_groups[idx % num_groups]
        max_tokens = sum(N_real[t] for t in groups[g])
        if group_budgets[g] < max_tokens:
            group_budgets[g] += 1
            remainder -= 1
        idx += 1

    # Fix over-allocation
    if sum(group_budgets) > total_budget:
        sorted_asc = sorted(range(num_groups), key=lambda i: group_weights[i].item())
        excess = sum(group_budgets) - total_budget
        idx = 0
        while excess > 0 and idx < num_groups * 10:
            g = sorted_asc[idx % num_groups]
            if group_budgets[g] > 1:
                group_budgets[g] -= 1
                excess -= 1
            idx += 1

    return group_budgets



# ============================================================================
# Main Compression Function: DynVid (DySeg + ERA + DATS + Soft Fusion)
# ============================================================================

@torch.no_grad()
def dynvid_compression_qwen(
    video_embeds: torch.Tensor,
    position_ids_video: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    config,
    precomputed_groups=None,
    precomputed_group_budgets=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Post-projector Compression Pipeline (DynVid):
      DySeg grouping -> ERA group budget allocation -> Intra-group DATS anchor selection (Facility Location) -> Top-K soft fusion

    All signals (CLS attn, dynamism) are computed here.
    ViT is completely unmodified.

    Args:
        video_embeds: (T * N, D) all video tokens (projector output).
        position_ids_video: (3, T * N) M-RoPE position IDs.
        num_frames: T.
        tokens_per_frame: N.
        config: DynVidConfig instance.
        precomputed_groups: Optional, DySeg grouping pre-computed by the caller (modeling).
            When not None, skips internal DySeg grouping and reuses directly (supports pre-computation/overlap execution).
        precomputed_group_budgets: Optional, group budgets paired with precomputed_groups.
            Both must be provided together to take effect; otherwise falls back to internal computation.

    Returns:
        compressed_embeds: (budget, D) compressed video tokens.
        compressed_positions: (3, budget) compressed position IDs.
        kept_global_indices: (budget,) global index of each retained token in the original flat visual sequence.
    """
    T = num_frames
    N = tokens_per_frame
    D = video_embeds.shape[-1]
    device = video_embeds.device
    dtype = video_embeds.dtype

    N_original = getattr(config, '_original_tokens_per_frame', None) or N
    _s2_ratio = getattr(config, 'before_LLM_retention_ratio', 0.20) or 0.20
    budget = max(1, int(T * N_original * _s2_ratio))

    # If compression is not needed
    if budget >= T * N:
        # _td_logger.info(f"[Compress] SKIP: budget={budget} >= T*N={T*N}")
        return video_embeds, position_ids_video, torch.arange(T * N, device=device)

    # ---- Split into per-frame ----
    frame_features = video_embeds.view(T, N, D)
    frame_positions = position_ids_video.view(3, T, N)

    frame_features_list = [frame_features[t] for t in range(T)]       # list of (N, D)
    frame_positions_list = [frame_positions[:, t, :] for t in range(T)]  # list of (3, N)

    # ---- Read config parameters ----
    dynamism_window = 1  # DynVid fixed: only look at adjacent frames (window=1)
    dyseg_threshold = config.dyseg_threshold
    min_segment_num = getattr(config, 'min_segment_num', 4)           # Minimum segment count (0=no constraint)
    complementary_segment = getattr(config, 'complementary_segment', True)  # Automatically add complementary cuts when insufficient
    topk_fusion = config.topk_fusion           # K for Top-K soft assignment
    anchor_weight = getattr(config, 'anchor_weight', 0.25)
    anchor_method = getattr(config, 'anchor_method', 'facility_location')
    group_budget_method = config.group_budget_method  # "pr" or "uniform"
    min_tokens_per_frame = 1  # Minimum 1 anchor retained per frame

    # Normalize features for similarity computation
    frame_features_norm = [F.normalize(f.float(), dim=-1) for f in frame_features_list]

    # # Signal 1: ViT col_mean CLS attention (production only supports col_mean)
    # _cls_method = getattr(config, 'cls_attn_method', 'col_mean')
    # if _cls_method != 'col_mean':
    #     raise NotImplementedError(
    #         f"Only cls_attn_method='col_mean' is supported, got '{_cls_method}'"
    #     )
    _vit_cls_attn = getattr(config, '_vit_cls_attn', None)
    assert _vit_cls_attn is not None and _vit_cls_attn.shape[0] == T * N, (
        f"col_mean signal not available: shape="
        f"{_vit_cls_attn.shape if _vit_cls_attn is not None else None}, expected=({T*N},)"
    )
    token_norms_per_frame = list(_vit_cls_attn.float().view(T, N))  # list of (N,)

    # Signal 2: Dynamism - dyn(v_j) = novelty(v_j) + asymmetry(v_j)
    # novelty = 1 - max(fwd_max, bwd_max)  -> neither forward nor backward frames are similar = new content
    # asymmetry = |fwd_max - bwd_max|       -> forward-backward asymmetry = motion direction change
    w = dynamism_window

    fwd_max_all = torch.zeros(T, N, device=device, dtype=torch.float32)
    bwd_max_all = torch.zeros(T, N, device=device, dtype=torch.float32)

    if T > 1:
        # Stack list into (T, N, D), note frame_features_norm[t] is already float32
        frames_stack = torch.stack(frame_features_norm, dim=0)  # (T, N, D)

        for d in range(1, min(w, T - 1) + 1):
            # For all t in [0, T-d): compute frames[t] @ frames[t+d].T
            sim_batch = torch.bmm(
                frames_stack[: T - d],                       # (T-d, N, D)
                frames_stack[d:].transpose(-2, -1),          # (T-d, D, N)
            )  # (T-d, N, N)
            fwd_d = sim_batch.max(dim=2).values              # (T-d, N)
            bwd_d = sim_batch.max(dim=1).values              # (T-d, N)
            fwd_max_all[: T - d] = torch.max(fwd_max_all[: T - d], fwd_d)
            bwd_max_all[d:]      = torch.max(bwd_max_all[d:],      bwd_d)

    # Compute dynamism in one pass: novelty = 1 - max(fwd, bwd), asymmetry = |fwd - bwd|
    novelty_all = (1.0 - torch.max(fwd_max_all, bwd_max_all)).clamp(min=0)  # (T, N)
    asymmetry_all = torch.abs(fwd_max_all - bwd_max_all)                       # (T, N)
    token_dyn_all = novelty_all + asymmetry_all      # (T, N)  - Paper Eq: dyn = novelty + asymmetry
    token_dynamism_per_frame = list(token_dyn_all)  # list of (N,) views

    # Normalize signals
    # dynamism: keep raw values, normalization moved to intra-group (cross-frame absolute values are comparable)
    token_norms_per_frame = [_td_normalize(tn) for tn in token_norms_per_frame]

    # ---- DySeg Grouping ----
    _use_precomputed = (
        precomputed_groups is not None and precomputed_group_budgets is not None
    )
    if _use_precomputed:
        groups = precomputed_groups
    else:
        groups = _dyseg_group_frames(frame_features_norm, dyseg_threshold, min_segment_num, complementary_segment)

    # Token count per frame (no padding, all frames equal length)
    _N_real = [N] * T

    # ---- Pre-compute per-frame token importance ----
    # cls_attn: already intra-frame normalized (relative ranking within frame, not comparable across frames)
    # dynamism: keep raw values, intra-group normalization (cross-frame absolute values are comparable)
    # Final importance = cls_attn + dynamism (equal-weight addition), computed within the group loop

    token_imp_per_frame = []  # Frame-level temporary storage: only cls component (already normalized)
    for t in range(T):
        _t_norms = token_norms_per_frame[t]
        # Frame-level only stores cls component temporarily
        token_imp_per_frame.append(_t_norms)

    # ---- Group budget allocation ----
    if _use_precomputed:
        group_budgets = precomputed_group_budgets
    else:
        group_budgets = _compute_group_budget(
            groups=groups,
            frame_features_norm=frame_features_norm,
            total_budget=budget,
            method=group_budget_method,
            N_real=_N_real,
            device=device,
        )

    # ---- Intra-group anchor selection + Top-K Soft Fusion ----
    all_fused_feats = []
    all_fused_pos = []
    all_kept_global_indices = []

    # Pre-stack: avoid per-group list.append + torch.cat host overhead
    frame_feats_stacked = torch.stack([f.float() for f in frame_features_list])  # (T, N, D)
    frame_pos_stacked = torch.stack(frame_positions_list, dim=1)                 # (3, T, N)
    cls_stacked = torch.stack(token_norms_per_frame)                             # (T, N)
    dyn_stacked = torch.stack(token_dynamism_per_frame)                          # (T, N)
    ar_N = torch.arange(N, device=device, dtype=torch.long)                      # (N,)

    # ---- First pass: vectorized collection of each group's pool, compute importance / init_selected ----
    # Groups with K_group >= P are kept entirely (passthrough); the rest enter pending for anchor selection.
    # Use entries in group order to record each group's output method, ensuring output sequence order matches original implementation.
    entries = []  # list of dict: {"type": "passthrough"/"compress", ...}
    pending = []  # Only compress groups, for batched facility_location

    for g_idx, group in enumerate(groups):
        K_group = group_budgets[g_idx]
        G = len(group)
        group_t = torch.tensor(group, device=device, dtype=torch.long)              # (G,)

        # index_select + reshape: no Python loop / no cat
        pool_feats_cat = frame_feats_stacked[group_t].reshape(G * N, D)              # (P, D)
        pool_pos_cat = frame_pos_stacked[:, group_t, :].reshape(3, G * N)           # (3, P)
        pool_norms_cat = cls_stacked[group_t].reshape(-1)                           # (P,)
        pool_dyn_cat = dyn_stacked[group_t].reshape(-1)                             # (P,)

        # ---- Intra-group normalization: dynamism ----
        pool_dyn_cat = _td_normalize(pool_dyn_cat)

        # ---- Intra-group importance = cls_attn + dynamism (Paper Eq: q_j = a_j + dyn(v_j)) ----
        pool_imp_cat = pool_norms_cat + pool_dyn_cat
        if pool_imp_cat.sum() == 0:
            pool_imp_cat = torch.ones_like(pool_imp_cat)

        pool_frame_ids_t = torch.arange(G, device=device, dtype=torch.long).repeat_interleave(N)  # (P,)
        pool_frame_global_t = group_t.repeat_interleave(N)                                         # (P,)

        P = G * N
        K_group = min(K_group, P)

        pool_to_global = (group_t.unsqueeze(1) * N + ar_N).flatten()                # (P,)

        if K_group >= P:
            # passthrough: keep entire group (output deferred to third pass, maintaining group order)
            entries.append({
                "type": "passthrough",
                "feats": pool_feats_cat,
                "pos": pool_pos_cat,
                "to_global": pool_to_global,
            })
            continue

        K_anchor = K_group

        # # Guarantee min_per_frame anchors per frame (temporal coverage guarantee)
        # _n_force = min(min_tokens_per_frame, N)
        # if _n_force > 0 and P == G * N:
        #     imp_per_frame = pool_imp_cat.view(G, N)                          # (G, N)
        #     _, idx_in_frame = imp_per_frame.topk(_n_force, dim=1)            # (G, k)
        #     offsets = torch.arange(G, device=device, dtype=torch.long).unsqueeze(1) * N
        #     init_selected = (idx_in_frame + offsets).flatten()               # (G*k,)
        # else:
        init_selected = None

        gd = {
            "type": "compress",
            "group": group,
            "feats": pool_feats_cat,
            "pos": pool_pos_cat,
            "imp": pool_imp_cat,
            "frame_ids": pool_frame_ids_t,
            "frame_global": pool_frame_global_t,
            "to_global": pool_to_global,
            "P": P,
            "K": K_anchor,
            "init_selected": init_selected,
            "anchor_indices": None,
        }
        entries.append(gd)
        pending.append(gd)

    # ---- Second pass: anchor selection ----
    if anchor_method == "facility_location":
        # Bucket by (P, n_init), same-bucket groups use batched exact greedy (with init_selected seeds, bit-equivalent)
        buckets = {}
        for gd in pending:
            n_init = 0 if gd["init_selected"] is None else int(gd["init_selected"].shape[0])
            buckets.setdefault((gd["P"], n_init), []).append(gd)
        for (P_val, n_init), gds in buckets.items():
            feats_b = torch.stack([gd["feats"] for gd in gds])   # (B, P, D)
            imp_b = torch.stack([gd["imp"] for gd in gds])       # (B, P)
            K_list = torch.tensor([gd["K"] for gd in gds], device=device, dtype=torch.long)
            init_b = None
            if n_init > 0:
                init_b = torch.stack([gd["init_selected"] for gd in gds])  # (B, n_init)
            # When init_selected count >= K_anchor, the per-group version directly truncates to init_selected[:K_anchor],
            # here n_init = min_tokens_per_frame*G = G (=1*G), K >= G always holds (each frame gets at least quota>=1),
            # so no special handling needed; the regular greedy with K>n_init is retained.
            sel = _facility_location_select_batched(feats_b, imp_b, K_list, init_selected=init_b)
            for b, gd in enumerate(gds):
                Kb = gd["K"]
                gd["anchor_indices"] = sel[b, :Kb].sort().values
    else:
        for gd in pending:
            pool_feats_cat = gd["feats"]
            pool_imp_cat = gd["imp"]
            P = gd["P"]
            K_anchor = gd["K"]
            init_selected = gd["init_selected"]

            if init_selected is not None and init_selected.shape[0] >= K_anchor:
                anchor_indices = init_selected[:K_anchor]
            elif anchor_method == "facility_location_original":
                anchor_indices = _facility_location_select(
                    features=pool_feats_cat,
                    importance=pool_imp_cat,
                    k=K_anchor,
                    init_selected=init_selected,
                )
            elif anchor_method == "topk":
                anchor_indices = _topk_importance_select(
                    features=pool_feats_cat,
                    importance=pool_imp_cat,
                    k=K_anchor,
                    init_selected=init_selected,
                )

            gd["anchor_indices"] = anchor_indices

    # ---- Third pass: Top-K soft fusion + output (in original group order) ----
    for gd in entries:
        if gd["type"] == "passthrough":
            # Keep entire group, output in group order
            all_fused_feats.append(gd["feats"].to(dtype))
            all_fused_pos.append(gd["pos"])
            all_kept_global_indices.append(gd["to_global"])
            continue

        pool_feats_cat = gd["feats"]
        pool_pos_cat = gd["pos"]
        pool_imp_cat = gd["imp"]
        pool_frame_ids_t = gd["frame_ids"]
        pool_to_global = gd["to_global"]
        P = gd["P"]
        anchor_indices = gd["anchor_indices"]

        # Build anchor/drop partition, call Top-K soft fusion
        anchor_mask = torch.zeros(P, dtype=torch.bool, device=device)
        anchor_mask[anchor_indices] = True

        anchor_feats = pool_feats_cat[anchor_indices]      # (K_anchor, D)
        anchor_pos = pool_pos_cat[:, anchor_indices]        # (3, K_anchor)

        drop_mask = ~anchor_mask
        if drop_mask.any():
            fused_feat, fused_pos = _topk_trash_fuse(
                all_feats=pool_feats_cat,
                all_pos=pool_pos_cat,
                anchor_indices=anchor_indices,
                top_k=topk_fusion,
                anchor_weight=anchor_weight,
            )
        else:
            fused_feat = anchor_feats
            fused_pos = anchor_pos

        fused_feat = fused_feat.to(dtype)

        # Output entire block (anchor_indices ascending + pool frame-major order -> already correct order)
        all_fused_feats.append(fused_feat)
        all_fused_pos.append(fused_pos)
        all_kept_global_indices.append(pool_to_global[anchor_indices])

    compressed_embeds = torch.cat(all_fused_feats, dim=0)
    compressed_positions = torch.cat(all_fused_pos, dim=1)
    kept_global_indices = torch.cat(all_kept_global_indices)

    actual_total = compressed_embeds.shape[0]
    _total_real = sum(_N_real)
    _td_logger.info(
        f"[Compress] Done: {_total_real} -> {actual_total} tokens "
        f"(budget={budget}, groups={len(groups)}, method=DATS+SoftFusion)"
    )

    return compressed_embeds, compressed_positions.to(frame_positions_list[0].dtype), kept_global_indices



# ============================================================================
# LLaVA OneVision: Post-projector DynVid Compression (1D global indices)
# ============================================================================


def dynvid_compression_llava(
    video_embeds: Tensor,
    cls_attention: Tensor,
    num_frames: int,
    tokens_per_frame: int,
    config,
) -> Tuple[Tensor, Tensor]:
    """Post-projector DynVid compression (LLaVA version).

    Differences from the Qwen2.5-VL version:
      - Input: (T, N, D) per-frame + (T, N) cls_attn  
      - Output: keep_visual_indices global indices 
      - No valid_mask / padding logic 
      - No position_ids_video 

    Args:
        video_embeds: (T, N, D) video tokens after 2D pooling.
        cls_attention: (T, N) SigLIP last layer per-token importance.
        num_frames: T.
        tokens_per_frame: N (169 after 2D pooling).
        config: DynVidConfig instance.

    Returns:
        compressed_embeds: (budget, D) compressed video tokens.
        keep_visual_indices: (budget,) global indices (frame_idx * N + token_idx).
    """
    T = num_frames
    N = tokens_per_frame
    D = video_embeds.shape[-1]
    device = video_embeds.device
    dtype = video_embeds.dtype

    _s2_ratio = getattr(config, 'before_LLM_retention_ratio', 0.20) or 0.20
    budget = max(1, int(T * N * _s2_ratio))

    # If compression is not needed
    if budget >= T * N:
        all_indices = torch.arange(T * N, device=device, dtype=torch.long)
        return video_embeds.reshape(T * N, D), all_indices

    # ---- Split into per-frame ----
    frame_features_list = [video_embeds[t] for t in range(T)]       # list of (N, D)

    # ---- Read config parameters ----
    dynamism_window = 1  # DynVid fixed: only look at adjacent frames (window=1)
    dyseg_threshold = config.dyseg_threshold
    min_segment_num = getattr(config, 'min_segment_num', 0)
    complementary_segment = getattr(config, 'complementary_segment', True)
    topk_fusion = config.topk_fusion
    anchor_weight = getattr(config, 'anchor_weight', 0.5)
    anchor_method = getattr(config, 'anchor_method', 'facility_location')
    group_budget_method = config.group_budget_method
    min_tokens_per_frame = 1  # Minimum 1 anchor retained per frame (temporal coverage guarantee)

    # Normalize features for similarity
    frame_features_norm = [F.normalize(f.float(), dim=-1) for f in frame_features_list]

    # Signal 1: CLS attention from SigLIP
    # LLaVA uses SigLIP's attn_weights.mean(heads).mean(queries) as per-token importance
    token_norms_per_frame = [cls_attention[t].float() for t in range(T)]

    # Signal 2: Dynamism
    w = dynamism_window
    all_frames_stacked = torch.stack(frame_features_norm)  # (T, N, D)

    fwd_max_all = torch.zeros(T, N, device=device, dtype=torch.float32)
    bwd_max_all = torch.zeros(T, N, device=device, dtype=torch.float32)

    for d in range(1, min(w, T - 1) + 1):
        curr = all_frames_stacked[:T - d]   # (T-d, N, D)
        next_ = all_frames_stacked[d:]      # (T-d, N, D)
        sim_batch = torch.bmm(curr, next_.transpose(1, 2))  # (T-d, N, N)
        fwd_max_all[:T - d] = torch.max(fwd_max_all[:T - d], sim_batch.max(dim=2).values)
        bwd_max_all[d:] = torch.max(bwd_max_all[d:], sim_batch.max(dim=1).values)

    # Vectorized dynamism computation
    novelty_all = (1.0 - torch.max(fwd_max_all, bwd_max_all)).clamp(min=0)  # (T, N)
    asymmetry_all = torch.abs(fwd_max_all - bwd_max_all)                     # (T, N)
    token_dyn_all = novelty_all + asymmetry_all    # (T, N)
    token_dynamism_per_frame = [token_dyn_all[t] for t in range(T)]

    # Normalize CLS attention (intra-frame normalization)
    token_norms_per_frame = [_td_normalize(tn) for tn in token_norms_per_frame]

    # ---- DySeg Grouping ----
    groups = _dyseg_group_frames(frame_features_norm, dyseg_threshold, min_segment_num, complementary_segment)

    # ---- Group budget allocation ----
    N_real = [N] * T
    group_budgets = _compute_group_budget(
        groups=groups,
        frame_features_norm=frame_features_norm,
        total_budget=budget,
        method=group_budget_method,
        N_real=N_real,
        device=device,
    )

    # ---- Intra-group anchor selection + Top-K Soft Fusion ----
    all_fused_feats = []
    all_fused_indices = []  # Global indices

    # Pre-stack: avoid per-group list.append + torch.cat + torch.tensor(list) host/H2D overhead
    frame_feats_stacked = torch.stack([f.float() for f in frame_features_list])  # (T, N, D)
    cls_stacked = torch.stack(token_norms_per_frame)        # (T, N)
    dyn_stacked = torch.stack(token_dynamism_per_frame)     # (T, N)
    ar_N = torch.arange(N, device=device, dtype=torch.long)  # (N,)

    # ---- First pass: vectorized collection of each group's pool, compute importance ----
    # Groups with K_group >= P are kept entirely; the rest enter pending for anchor selection.
    pending = []
    for g_idx, group in enumerate(groups):
        K_group = group_budgets[g_idx]
        G = len(group)
        group_t = torch.tensor(group, device=device, dtype=torch.long)  # (G,)

        # index_select + reshape: no Python loop / no cat / no H2D sync
        pool_feats_cat = frame_feats_stacked[group_t].reshape(G * N, D)              # (P, D)
        pool_gidx_cat = ((group_t * N).view(G, 1) + ar_N).reshape(-1)                # (P,)
        pool_cls_cat = cls_stacked[group_t].reshape(-1)                              # (P,)
        pool_dyn_cat = dyn_stacked[group_t].reshape(-1)                              # (P,)
        pool_frame_ids_t = torch.arange(G, device=device, dtype=torch.long).repeat_interleave(N)  # (P,)

        P = G * N
        K_group = min(K_group, P)

        if K_group >= P:
            all_fused_feats.append(pool_feats_cat.to(dtype))
            all_fused_indices.append(pool_gidx_cat)
            continue

        # Intra-group normalization of dynamism
        pool_dyn_cat = _td_normalize(pool_dyn_cat)

        # Intra-group importance = convex combination
        pool_imp_cat = pool_cls_cat + pool_dyn_cat
        if pool_imp_cat.sum() == 0:
            pool_imp_cat = torch.ones_like(pool_imp_cat)

        # # Guarantee min_per_frame anchors per frame (temporal coverage guarantee)
        # _n_force = min(min_tokens_per_frame, N)
        # if _n_force > 0 and P == G * N:
        #     imp_per_frame = pool_imp_cat.view(G, N)                          # (G, N)
        #     _, idx_in_frame = imp_per_frame.topk(_n_force, dim=1)            # (G, k)
        #     offsets = torch.arange(G, device=device, dtype=torch.long).unsqueeze(1) * N
        #     init_selected = (idx_in_frame + offsets).flatten()               # (G*k,)
        # else:
        init_selected = None

        pending.append({
            "feats": pool_feats_cat,
            "gidx": pool_gidx_cat,
            "imp": pool_imp_cat,
            "frame_ids": pool_frame_ids_t,
            "P": P,
            "K": K_group,
            "init_selected": init_selected,
            "anchor_indices": None,
        })

    # ---- Second pass: anchor selection ----
    if anchor_method == "facility_location":
        # Bucket by (P, n_init), same-bucket groups use batched exact greedy (with init_selected seeds, bit-equivalent)
        buckets = {}
        for gd in pending:
            n_init = 0 if gd["init_selected"] is None else int(gd["init_selected"].shape[0])
            buckets.setdefault((gd["P"], n_init), []).append(gd)
        for (P_val, n_init), gds in buckets.items():
            feats_b = torch.stack([gd["feats"] for gd in gds])   # (B, P, D)
            imp_b = torch.stack([gd["imp"] for gd in gds])       # (B, P)
            K_list = torch.tensor([gd["K"] for gd in gds], device=device, dtype=torch.long)
            init_b = None
            if n_init > 0:
                init_b = torch.stack([gd["init_selected"] for gd in gds])  # (B, n_init)
            sel = _facility_location_select_batched(feats_b, imp_b, K_list, init_selected=init_b)
            for b, gd in enumerate(gds):
                Kb = gd["K"]
                gd["anchor_indices"] = sel[b, :Kb].sort().values
    else:
        for gd in pending:
            pool_feats_cat = gd["feats"]
            pool_imp_cat = gd["imp"]
            P = gd["P"]
            K_anchor = gd["K"]
            init_selected = gd["init_selected"]

            if init_selected is not None and init_selected.shape[0] >= K_anchor:
                anchor_indices = init_selected[:K_anchor]
            elif anchor_method == "facility_location_original":
                anchor_indices = _facility_location_select(
                    features=pool_feats_cat,
                    importance=pool_imp_cat,
                    k=K_anchor,
                    init_selected=init_selected,
                )
            elif anchor_method == "topk":
                anchor_indices = _topk_importance_select(
                    features=pool_feats_cat,
                    importance=pool_imp_cat,
                    k=K_anchor,
                    init_selected=init_selected,
                )

            gd["anchor_indices"] = anchor_indices

    # ---- Third pass: Top-K soft fusion ----
    for gd in pending:
        pool_feats_cat = gd["feats"]
        pool_gidx_cat = gd["gidx"]
        pool_imp_cat = gd["imp"]
        pool_frame_ids_t = gd["frame_ids"]
        P = gd["P"]
        anchor_indices = gd["anchor_indices"]

        # Top-K soft fusion (using unified _topk_trash_fuse, passing 1D positions)
        anchor_mask = torch.zeros(P, dtype=torch.bool, device=device)
        anchor_mask[anchor_indices] = True
        drop_mask = ~anchor_mask

        if drop_mask.any():
            fused_feat, fused_gidx = _topk_trash_fuse(
                all_feats=pool_feats_cat,
                all_pos=pool_gidx_cat,       # (P,) 1D global indices
                anchor_indices=anchor_indices,
                top_k=topk_fusion,
                anchor_weight=anchor_weight,
            )
        else:
            fused_feat = pool_feats_cat[anchor_indices].float()
            fused_gidx = pool_gidx_cat[anchor_indices]

        fused_feat = fused_feat.to(dtype)

        all_fused_feats.append(fused_feat)
        all_fused_indices.append(fused_gidx)


    compressed_embeds = torch.cat(all_fused_feats, dim=0)
    keep_visual_indices = torch.cat(all_fused_indices, dim=0)

    # Sort by global index (maintain temporal order)
    sort_order = keep_visual_indices.argsort()
    compressed_embeds = compressed_embeds[sort_order]
    keep_visual_indices = keep_visual_indices[sort_order]

    actual_total = compressed_embeds.shape[0]
    _td_logger.info(
        f"[Compress] Done: {T*N} -> {actual_total} tokens "
        f"(budget={budget}, groups={len(groups)}, method=DATS+SoftFusion)"
    )

    return compressed_embeds, keep_visual_indices




def _deepstack_process(
    hidden_states: torch.Tensor,
    visual_pos_masks: torch.Tensor,
    deepstack_embed: torch.Tensor,
) -> torch.Tensor:
    """DeepStack residual injection: performs hidden_states += deepstack_embed at visual positions.

    Args:
        hidden_states: (B, S, D)
        visual_pos_masks: (B, S) bool mask, True = visual token
        deepstack_embed: (num_visual, D) DeepStack features for this layer
    """
    hidden_states = hidden_states.clone()
    hidden_states[visual_pos_masks] = hidden_states[visual_pos_masks] + deepstack_embed
    return hidden_states