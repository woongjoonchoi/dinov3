"""
Window representative token selection policies.

All methods: O(nd) per window, no pairwise distances, no SVD/PCA.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _pick_space(X_pre: Tensor, X_post: Tensor, space: str) -> Tensor:
    if space == "pre_ln":
        return X_pre
    return X_post


def _gather_by_index(G: Tensor, idx: Tensor) -> Tensor:
    """G: [B, W, n, d], idx: [B, W] → [B, W, d]"""
    d = G.shape[-1]
    return G.gather(2, idx[:, :, None, None].expand(*idx.shape, 1, d)).squeeze(2)


def _masked_mean(Z: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Z: [B, W, n, d], mask: [B, W, n] bool → [B, W, d]"""
    if mask is None:
        return Z.mean(dim=2)
    m = mask.float().unsqueeze(-1)
    return (Z * m).sum(2) / (m.sum(2) + 1e-8)


def _index_to_position(positions: Optional[Tensor], idx: Tensor) -> Optional[Tensor]:
    """positions: [n, 2] or None, idx: [B, W] → [B, W, 2] or None"""
    if positions is None:
        return None
    B, W = idx.shape
    return positions[idx.reshape(-1)].reshape(B, W, 2)


# ── V1: Mean Centroid ─────────────────────────────────────────────────────────

def select_v1_mean_centroid(
    X_pre: Tensor,
    X_post: Tensor,
    selector_space: str,
    gather_space: str,
    valid_mask: Optional[Tensor] = None,
    positions: Optional[Tensor] = None,
    **kwargs,
) -> Dict[str, Tensor]:
    """Synthetic centroid; no actual token selected. gather_space unused."""
    Z = _pick_space(X_pre, X_post, selector_space)
    return {"rep_vector": _masked_mean(Z, valid_mask)}


# ── V2: Centroid Proximal (Approximate Medoid) ───────────────────────────────

def select_v2_centroid_proximal(
    X_pre: Tensor,
    X_post: Tensor,
    selector_space: str,
    gather_space: str,
    valid_mask: Optional[Tensor] = None,
    positions: Optional[Tensor] = None,
    **kwargs,
) -> Dict[str, Tensor]:
    """Token closest to window centroid (squared L2, min-based → invalid=+inf)."""
    Z = _pick_space(X_pre, X_post, selector_space)
    G = _pick_space(X_pre, X_post, gather_space)
    mu = _masked_mean(Z, valid_mask)
    dist_sq = ((Z - mu.unsqueeze(2)) ** 2).sum(-1)  # [B, W, n]
    if valid_mask is not None:
        dist_sq = dist_sq.masked_fill(~valid_mask, float("inf"))
    idx = dist_sq.argmin(dim=2)
    result = {"rep_vector": _gather_by_index(G, idx), "selected_index": idx}
    pos = _index_to_position(positions, idx)
    if pos is not None:
        result["rep_position"] = pos
    return result


# ── V3: Deviation from Mean ───────────────────────────────────────────────────

def select_v3_deviation_from_mean(
    X_pre: Tensor,
    X_post: Tensor,
    selector_space: str,
    gather_space: str,
    valid_mask: Optional[Tensor] = None,
    positions: Optional[Tensor] = None,
    **kwargs,
) -> Dict[str, Tensor]:
    """Token maximally deviating from window mean (max-based → invalid=-inf)."""
    Z = _pick_space(X_pre, X_post, selector_space)
    G = _pick_space(X_pre, X_post, gather_space)
    mu = _masked_mean(Z, valid_mask)
    scores = ((Z - mu.unsqueeze(2)) ** 2).sum(-1)  # [B, W, n]
    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, float("-inf"))
    idx = scores.argmax(dim=2)
    result = {
        "rep_vector": _gather_by_index(G, idx),
        "selected_index": idx,
        "score": scores,
    }
    pos = _index_to_position(positions, idx)
    if pos is not None:
        result["rep_position"] = pos
    return result


# ── V4: Norm Based ────────────────────────────────────────────────────────────

def select_v4_norm_based(
    X_pre: Tensor,
    X_post: Tensor,
    selector_space: str,
    gather_space: str,
    valid_mask: Optional[Tensor] = None,
    positions: Optional[Tensor] = None,
    **kwargs,
) -> Dict[str, Tensor]:
    """Token with largest squared L2 norm (max-based → invalid=-inf).
    Note: post-LN norms are near-uniform, making this near-degenerate post-LN."""
    Z = _pick_space(X_pre, X_post, selector_space)
    G = _pick_space(X_pre, X_post, gather_space)
    scores = (Z * Z).sum(-1)  # [B, W, n]
    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, float("-inf"))
    idx = scores.argmax(dim=2)
    result = {
        "rep_vector": _gather_by_index(G, idx),
        "selected_index": idx,
        "score": scores,
    }
    pos = _index_to_position(positions, idx)
    if pos is not None:
        result["rep_position"] = pos
    return result


# ── V5: Finite Difference Saliency ───────────────────────────────────────────

def select_v5_finite_difference_saliency(
    X_pre: Tensor,
    X_post: Tensor,
    selector_space: str,
    gather_space: str,
    valid_mask: Optional[Tensor] = None,
    positions: Optional[Tensor] = None,
    **kwargs,
) -> Dict[str, Tensor]:
    """Token with maximum 4-neighbor local contrast (max-based → invalid=-inf).
    Requires n to be a perfect square (M*M)."""
    Z = _pick_space(X_pre, X_post, selector_space)
    G = _pick_space(X_pre, X_post, gather_space)
    B, num_windows, n, d = Z.shape
    M = int(math.isqrt(n))
    assert M * M == n, f"n={n} must be a perfect square"

    Z_grid = Z.reshape(B, num_windows, M, M, d)

    # Squared differences for horizontal and vertical neighbor pairs
    diff_h = Z_grid[:, :, :, :-1, :] - Z_grid[:, :, :, 1:, :]  # [B,W,M,M-1,d]
    diff_v = Z_grid[:, :, :-1, :, :] - Z_grid[:, :, 1:, :, :]  # [B,W,M-1,M,d]
    score_h = (diff_h * diff_h).sum(-1)  # [B,W,M,M-1]
    score_v = (diff_v * diff_v).sum(-1)  # [B,W,M-1,M]

    # Accumulate 4-neighbor contrast symmetrically
    score_grid = Z_grid.new_zeros(B, num_windows, M, M)
    score_grid[:, :, :, :-1].add_(score_h)
    score_grid[:, :, :, 1:].add_(score_h)
    score_grid[:, :, :-1, :].add_(score_v)
    score_grid[:, :, 1:, :].add_(score_v)

    scores = score_grid.reshape(B, num_windows, n)
    if valid_mask is not None:
        scores = scores.masked_fill(~valid_mask, float("-inf"))
    idx = scores.argmax(dim=2)
    result = {
        "rep_vector": _gather_by_index(G, idx),
        "selected_index": idx,
        "score": scores,
    }
    pos = _index_to_position(positions, idx)
    if pos is not None:
        result["rep_position"] = pos
    return result


# ── V6: Weighted Quadrature Pooling ──────────────────────────────────────────

def select_v6_weighted_quadrature_pooling(
    X_pre: Tensor,
    X_post: Tensor,
    selector_space: str,
    gather_space: str,
    valid_mask: Optional[Tensor] = None,
    positions: Optional[Tensor] = None,
    tau: float = 1.0,
    score_type: str = "norm",
    **kwargs,
) -> Dict[str, Tensor]:
    """Softmax-weighted feature sum. score_type: 'norm' | 'deviation'."""
    Z = _pick_space(X_pre, X_post, selector_space)
    G = _pick_space(X_pre, X_post, gather_space)

    if score_type == "deviation":
        mu = _masked_mean(Z, valid_mask)
        raw_scores = ((Z - mu.unsqueeze(2)) ** 2).sum(-1)
    else:  # "norm"
        raw_scores = (Z * Z).sum(-1)

    if valid_mask is not None:
        raw_scores = raw_scores.masked_fill(~valid_mask, float("-inf"))

    weights = torch.softmax(raw_scores / tau, dim=2)  # [B, W, n]
    if valid_mask is not None:
        weights = weights.masked_fill(~valid_mask, 0.0)

    rep_vector = (weights.unsqueeze(-1) * G).sum(2)  # [B, W, d]
    idx = weights.argmax(dim=2)

    return {"rep_vector": rep_vector, "weights": weights, "selected_index": idx}


# ── V7: Weighted Position Estimation ─────────────────────────────────────────

def select_v7_weighted_position_estimation(
    X_pre: Tensor,
    X_post: Tensor,
    selector_space: str,
    gather_space: str,
    valid_mask: Optional[Tensor] = None,
    positions: Optional[Tensor] = None,
    tau: float = 1.0,
    score_type: str = "norm",
    **kwargs,
) -> Dict[str, Tensor]:
    """Softmax-weighted position estimation + feature sum.
    Returns both rep_position [B,W,2] and rep_vector [B,W,d]."""
    Z = _pick_space(X_pre, X_post, selector_space)
    G = _pick_space(X_pre, X_post, gather_space)
    B, num_windows, n, d = Z.shape

    if score_type == "deviation":
        mu = _masked_mean(Z, valid_mask)
        raw_scores = ((Z - mu.unsqueeze(2)) ** 2).sum(-1)
    else:  # "norm"
        raw_scores = (Z * Z).sum(-1)

    if valid_mask is not None:
        raw_scores = raw_scores.masked_fill(~valid_mask, float("-inf"))

    weights = torch.softmax(raw_scores / tau, dim=2)  # [B, W, n]
    if valid_mask is not None:
        weights = weights.masked_fill(~valid_mask, 0.0)

    rep_vector = (weights.unsqueeze(-1) * G).sum(2)  # [B, W, d]

    # Build integer grid positions if none provided
    if positions is None:
        M = int(math.isqrt(n))
        u = torch.arange(M, device=Z.device, dtype=Z.dtype)
        v = torch.arange(M, device=Z.device, dtype=Z.dtype)
        gu, gv = torch.meshgrid(u, v, indexing="ij")
        pos_grid = torch.stack([gu.reshape(-1), gv.reshape(-1)], dim=1)  # [n, 2]
    else:
        pos_grid = positions.to(dtype=Z.dtype, device=Z.device)

    rep_position = (weights.unsqueeze(-1) * pos_grid[None, None]).sum(2)  # [B, W, 2]

    # Nearest discrete token to estimated position
    pos_dist_sq = ((pos_grid[None, None] - rep_position[:, :, None, :]) ** 2).sum(-1)  # [B, W, n]
    idx = pos_dist_sq.argmin(dim=2)

    return {
        "rep_vector": rep_vector,
        "rep_position": rep_position,
        "weights": weights,
        "selected_index": idx,
    }


# ── V8: Haar LL ───────────────────────────────────────────────────────────────

def select_v8_haar_ll(
    X_pre: Tensor,
    X_post: Tensor,
    selector_space: str,
    gather_space: str,
    valid_mask: Optional[Tensor] = None,
    positions: Optional[Tensor] = None,
    **kwargs,
) -> Dict[str, Tensor]:
    """Block-level 2×2 Haar LL: select highest-energy block, then closest token.
    Requires n to be a perfect square. Odd M is padded with replicate padding."""
    Z = _pick_space(X_pre, X_post, selector_space)
    G = _pick_space(X_pre, X_post, gather_space)
    B, num_windows, n, d = Z.shape
    M = int(math.isqrt(n))
    assert M * M == n, f"n={n} must be a perfect square"

    Z_grid = Z.reshape(B, num_windows, M, M, d)

    if M % 2 != 0:
        # Replicate-pad last row and col to make M even
        Z_grid = torch.nn.functional.pad(
            Z_grid.permute(0, 1, 4, 2, 3), (0, 1, 0, 1), mode="replicate"
        ).permute(0, 1, 3, 4, 2)
        M_pad = M + 1
    else:
        M_pad = M

    # 2×2 block LL averages: [B, W, Hb, Hb, d] where Hb = M_pad // 2
    ll = (
        Z_grid[:, :, 0::2, 0::2, :] + Z_grid[:, :, 1::2, 0::2, :]
        + Z_grid[:, :, 0::2, 1::2, :] + Z_grid[:, :, 1::2, 1::2, :]
    ) / 4.0

    Hb = M_pad // 2
    block_energy = (ll * ll).sum(-1).reshape(B, num_windows, Hb * Hb)
    best_block = block_energy.argmax(dim=2)  # [B, W]
    best_a = best_block // Hb
    best_b = best_block % Hb

    # LL vector of the best block: [B, W, d]
    BW = B * num_windows
    arange_BW = torch.arange(BW, device=Z.device)
    best_ll_vec = (
        ll.reshape(BW, Hb, Hb, d)[arange_BW, best_a.reshape(BW), best_b.reshape(BW)]
        .reshape(B, num_windows, d)
    )

    # 4 candidate flat indices within the best block (clamped to valid M range)
    r0 = (best_a * 2).clamp(max=M - 1)
    r1 = (best_a * 2 + 1).clamp(max=M - 1)
    c0 = (best_b * 2).clamp(max=M - 1)
    c1 = (best_b * 2 + 1).clamp(max=M - 1)
    cand_flat = torch.stack([r0 * M + c0, r0 * M + c1, r1 * M + c0, r1 * M + c1], dim=2)  # [B,W,4]

    # Squared distance from each candidate (in Z space) to the LL vector
    z_cands = Z.gather(2, cand_flat[:, :, :, None].expand(B, num_windows, 4, d))  # [B,W,4,d]
    dist_sq = ((z_cands - best_ll_vec.unsqueeze(2)) ** 2).sum(-1)  # [B,W,4]
    local_idx = dist_sq.argmin(dim=2)  # [B,W] ∈ {0,1,2,3}

    flat_idx = cand_flat.gather(2, local_idx.unsqueeze(2)).squeeze(2)  # [B,W]
    rep_vector = _gather_by_index(G, flat_idx)

    result = {"rep_vector": rep_vector, "selected_index": flat_idx}
    pos = _index_to_position(positions, flat_idx)
    if pos is not None:
        result["rep_position"] = pos
    return result


# ── V9: Entropy Based ─────────────────────────────────────────────────────────

def select_v9_entropy_based(
    X_pre: Tensor,
    X_post: Tensor,
    selector_space: str,
    gather_space: str,
    valid_mask: Optional[Tensor] = None,
    positions: Optional[Tensor] = None,
    tau_c: float = 1.0,
    entropy_mode: str = "high",
    **kwargs,
) -> Dict[str, Tensor]:
    """Token selected by channel-distribution entropy.
    entropy_mode='high': feature-rich token; 'low': sharp/confident token."""
    Z = _pick_space(X_pre, X_post, selector_space)
    G = _pick_space(X_pre, X_post, gather_space)

    # Channel softmax with max-stability
    z_max = Z.amax(dim=-1, keepdim=True)
    q = torch.softmax((Z - z_max) / tau_c, dim=-1)  # [B, W, n, d]
    entropy = -(q * (q + 1e-8).log()).sum(-1)  # [B, W, n]

    scores = entropy.clone()
    if valid_mask is not None:
        fill = float("-inf") if entropy_mode == "high" else float("inf")
        scores = scores.masked_fill(~valid_mask, fill)

    idx = scores.argmax(dim=2) if entropy_mode == "high" else scores.argmin(dim=2)

    result = {
        "rep_vector": _gather_by_index(G, idx),
        "selected_index": idx,
        "entropy": entropy,
    }
    pos = _index_to_position(positions, idx)
    if pos is not None:
        result["rep_position"] = pos
    return result
