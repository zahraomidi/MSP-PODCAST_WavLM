#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soft_label_utils.py
----------------------------------
Optional soft-label and augmentation utilities for categorical emotion tasks.

The public example configuration uses distributional labels from the dataset
and disables MixUp/CutMix by default.

Includes:
 - Optional synthetic soft-target builders
 - Optional skew per class
 - One-hot / label smoothing fallbacks
 - Optional MixUp / CutMix helpers for SpeechBrain training
"""

import torch
import torch.nn.functional as F
from typing import Callable, Tuple

# -------------------------------------------------
# Soft-label utilities 
# -------------------------------------------------

__all__ = ["one_hot", "smooth_labels", "gaussian_neighbor_smoothing", \
           "ordinal_gaussian_labels", "apply_mixup", "apply_cutmix", \
           "embedding_mixup", "embedding_cutmix", "apply_cutmix_plot"]

def one_hot(y_idx: torch.Tensor, C: int) -> torch.Tensor:
    return F.one_hot(y_idx, num_classes=C).float()

def smooth_labels(y_idx, num_classes, eps=0.1):
    """
    Standard label smoothing.

    Args:
      y_idx: LongTensor [B] with class indices
      num_classes: total number of classes
      eps: smoothing factor (0.0 = one-hot, 0.1 = 10% uniform smoothing)
    Returns:
      smoothed_targets: FloatTensor [B, C]
    """
    oh = F.one_hot(y_idx, num_classes=num_classes).float()
    return oh * (1 - eps) + (eps / num_classes) * torch.ones_like(oh)

def gaussian_neighbor_smoothing(
    y_idx: torch.Tensor,
    num_classes: int,
    eps: float = 0.1,
    sigma_per_class=None,
    skew_per_class=None,
) -> torch.Tensor:
    """
    Gaussian neighbor smoothing with controllable asymmetry (skewness).

    Args:
        y_idx: [B] LongTensor of class indices
        num_classes: int, number of classes
        eps: smoothing factor
        sigma_per_class: list/tensor of base sigma per class (len=C)
        skew_per_class: list/tensor of skew per class in [-1,1] (len=C)
    Returns:
        smoothed_targets: [B, C] FloatTensor
    """
    device = y_idx.device
    B = y_idx.size(0)

    # defaults
    if sigma_per_class is None:
        sigma_per_class = torch.full((num_classes,), 0.7, device=device)
    else:
        sigma_per_class = torch.as_tensor(sigma_per_class, device=device, dtype=torch.float)

    if skew_per_class is None:
        skew_per_class = torch.zeros(num_classes, device=device)
    else:
        skew_per_class = torch.as_tensor(skew_per_class, device=device, dtype=torch.float)

    # per-sample parameters
    sigma = sigma_per_class[y_idx]  # [B]
    skew = skew_per_class[y_idx]    # [B] in [-1,1]

    # class axis and diffs
    classes = torch.arange(num_classes, device=device).float()  # [C]
    diff = classes.unsqueeze(0) - y_idx.float().unsqueeze(1)    # [B,C]

    # asymmetric scaling
    # right side: (1 + skew), left side: (1 - skew)
    right_mask = (diff > 0).float()
    left_mask = 1.0 - right_mask
    sigma_eff = sigma.unsqueeze(1) * (right_mask * (1 + skew).unsqueeze(1)
                                     + left_mask * (1 - skew).unsqueeze(1))
    
    # Gaussian weights (no external imports, fully differentiable)
    weights = torch.exp(-0.5 * (diff / sigma_eff).pow(2))
    weights[torch.arange(B, device=device), y_idx] = 0.0

    # normalize and blend
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    oh = F.one_hot(y_idx, num_classes=num_classes).float()
    tgt = (1 - eps) * oh + eps * weights
    return tgt


def ordinal_gaussian_labels(y_idx: torch.Tensor, num_classes: int, sigma: float) -> torch.Tensor:
    """
    Generate ordinal Gaussian labels.
    Args:
      y_idx: LongTensor [B] with class indices
      num_classes: total number of classes
      sigma: standard deviation of the Gaussian
    Returns:
      soft_targets: FloatTensor [B, C]
    """
    c = torch.arange(num_classes, device=y_idx.device).float()[None, :]  # (1, C)
    y = y_idx.float()[:, None]                                           # (B, 1)
    logits = -0.5 * ((c - y) ** 2) / (sigma ** 2 + 1e-12)
    return F.softmax(logits, dim=-1)

# -------------------------------------------------
# Mixup and CutMix for audio inputs and soft labels  
# -------------------------------------------------
def apply_mixup(
    wavs: torch.Tensor,                 # [B,T] or [B,1,T]
    y_idx: torch.Tensor,                # [B]
    C: int,                             # number of classes
    alpha: float = 0.4,                 # Beta distribution parameter
    per_sample: bool = True,            # if False, use shared lambda for all
    batch_size: int | None = None,      # optionally override (for partial mini-batches)
    build_targets: Callable[[torch.Tensor], torch.Tensor] | None = None,
    soft_targets: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply Mixup augmentation to a batch of waveforms and labels.

    Args:
        wavs: Tensor [B, T] or [B, 1, T].
        y_idx: Class indices [B].
        C: Number of classes (for one-hot).
        alpha: Beta distribution parameter controlling mix strength.
        per_sample: Whether to sample lambda per item or use one for the batch.
        batch_size: Optional override (useful when wavs may contain more items).
        build_targets: Optional function returning smoothed/soft targets.
        soft_targets: Optional precomputed soft targets [B, C].

    Returns:
        mixed_wavs: [B, 1, T] mixed audio tensors
        mixed_targets: [B, C] mixed label distributions
        lam: [B] or [1] tensor of mixing ratios
    """
    device = wavs.device
    B = batch_size or wavs.size(0)

    # --- sanity checks ---
    if B < 2:
        raise ValueError("Mixup requires at least 2 samples in the batch.")
    # if wavs.dim() == 2:
    #     wavs = wavs.unsqueeze(1)  # [B, 1, T]

    # --- create a non-trivial random permutation ---
    perm = torch.randperm(B, device=device)
    # Avoid self-mixing for small batches
    if torch.all(perm == torch.arange(B, device=device)):
        perm = torch.roll(perm, 1)

    wavs_p, y_p = wavs[perm], y_idx[perm]

    # --- sample lambda(s) ---
    lam = torch.distributions.Beta(alpha, alpha).rsample((B,)).to(device)
    if not per_sample:
        lam = lam.mean().expand(B)
    lam_w = lam.view(B, *([1] * (wavs.dim() - 1)))

    # --- mix waveforms ---
    mixed_wavs = lam_w * wavs + (1 - lam_w) * wavs_p

    if mixed_wavs.dim() > 2:
        mixed_wavs = mixed_wavs.squeeze(1)

    # --- mix labels (supports soft_targets, one-hot or custom builder) ---
    if soft_targets is not None:
        t1 = soft_targets
        t2 = soft_targets[perm]
    elif build_targets is None:
        t1 = F.one_hot(y_idx, num_classes=C).float()
        t2 = F.one_hot(y_p,   num_classes=C).float()
    else:
        t1, t2 = build_targets(y_idx), build_targets(y_p)
    mixed_targets = lam.view(B, 1) * t1 + (1 - lam.view(B, 1)) * t2
    assert mixed_wavs.dim() == 2, f"MixUp output must be [B,T], got {mixed_wavs.shape}"
    return mixed_wavs, mixed_targets, lam


def apply_cutmix(
    wavs: torch.Tensor,                 
    lens: torch.Tensor,                 # [B], normalized [0,1]
    y_idx: torch.Tensor,                
    C: int,
    alpha: float = 0.4,
    per_sample: bool = True,
    build_targets: Callable[[torch.Tensor], torch.Tensor] | None = None,
    soft_targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply CutMix augmentation to a batch of variable-length audio waveforms.

    Replaces a random contiguous segment of each waveform with the same-length
    segment from another waveform in the batch, according to a Beta(α, α) ratio.

    Returns:
        mixed_wavs: Tensor [B, 1, T]  — augmented waveforms
        mixed_targets: Tensor [B, C]  — mixed (soft) label distributions
        lam_eff: Tensor [B]           — effective retained ratios per sample
    """
    B, T, device = wavs.size(0), wavs.size(-1), wavs.device

    # Random permutation
    perm = torch.randperm(B, device=device)
    # Force at least one swap (avoid identity)
    while torch.all(perm == torch.arange(B, device=device)):
        perm = torch.randperm(B, device=device)
    wavs_p, y_p = wavs[perm], y_idx[perm]

    # Sample λ ∈ [0,1]
    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.rsample((B,)).to(device) if per_sample else beta.rsample(()).to(device).expand(B)
    cut_frac = 1.0 - lam  # fraction to replace

    # Compute actual cut lengths (respecting utterance lengths)
    eff_len = (lens * T).round().clamp_min(1).long()
    cut_len = (cut_frac * eff_len.float()).round().clamp_min(1).long().clamp_max(eff_len)

    # Random start positions
    max_start = (eff_len - cut_len + 1).clamp_min(1)
    starts = (torch.rand_like(eff_len.float()) * max_start.float()).floor().long()

    # Build boolean mask for replaced region
    idx = torch.arange(T, device=device).view(1, T)
    mask = ((idx >= starts.view(-1,1)) & (idx < (starts + cut_len).view(-1,1))).bool()
    # if wavs.dim() == 3:
    #     mask = mask.unsqueeze(1)
    mask = mask.to(dtype=wavs.dtype, device=wavs.device)

    # Replace region
    mixed_wavs = wavs * (1.0 - mask) + wavs_p * mask

    if mixed_wavs.dim() > 2:
        mixed_wavs = mixed_wavs.squeeze(1)

    # Effective λ = proportion kept from original
    lam_eff = torch.ones(B, device=device)
    valid = eff_len > 0
    lam_eff[valid] = 1.0 - (cut_len.float()[valid] / eff_len.float()[valid])

    # Mix labels
    if soft_targets is not None:
        t1 = soft_targets
        t2 = soft_targets[perm]
    elif build_targets is None:
        t1 = F.one_hot(y_idx, num_classes=C).float()
        t2 = F.one_hot(y_p,   num_classes=C).float()
    else:
        t1, t2 = build_targets(y_idx), build_targets(y_p)
    mixed_targets = lam_eff.view(B,1) * t1 + (1 - lam_eff.view(B,1)) * t2
    mixed_wavs = mixed_wavs / (mixed_wavs.abs().max(dim=1, keepdim=True)[0] + 1e-6)
    assert mixed_wavs.dim() == 2, f"CutMix output must be [B,T], got {mixed_wavs.shape}"
    return mixed_wavs, mixed_targets, lam_eff

def apply_cutmix_plot(
    wavs: torch.Tensor,
    lens: torch.Tensor,
    y_idx: torch.Tensor,
    C: int,
    alpha: float = 0.4,
    per_sample: bool = True,
    build_targets: Callable[[torch.Tensor], torch.Tensor] | None = None,
    soft_targets: torch.Tensor | None = None,
):
    """
    Plot wrapper for apply_cutmix: returns extra details (start, cut_len) for visualization.
    """
    # --- copy core CutMix logic ---
    B, T, device = wavs.size(0), wavs.size(-1), wavs.device
    perm = torch.randperm(B, device=device)
    # Force at least one swap (avoid identity)
    while torch.all(perm == torch.arange(B, device=device)):
        perm = torch.randperm(B, device=device)
    wavs_p, y_p = wavs[perm], y_idx[perm]

    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.rsample((B,)).to(device) if per_sample else beta.rsample(()).to(device).expand(B)
    cut_frac = 1.0 - lam

    eff_len = (lens * T).round().clamp_min(1).long()
    cut_len = (cut_frac * eff_len.float()).round().clamp_min(1).long().clamp_max(eff_len)
    max_start = (eff_len - cut_len + 1).clamp_min(1)
    starts = (torch.rand_like(eff_len.float()) * max_start.float()).floor().long()

    idx = torch.arange(T, device=device).view(1, T)
    mask = ((idx >= starts.view(-1, 1)) & (idx < (starts + cut_len).view(-1, 1))).bool()
    if wavs.dim() == 3:
        mask = mask.unsqueeze(1)
    # --- ensure mask shape and dtype match exactly ---
    mask = mask.to(dtype=wavs.dtype, device=wavs.device) 

    mixed_wavs = wavs * (1.0 - mask) + wavs_p * mask
    mixed_wavs = mixed_wavs / (mixed_wavs.abs().max(dim=1, keepdim=True)[0] + 1e-6)

    lam_eff = torch.ones(B, device=device)
    lam_eff[eff_len > 0] = 1.0 - (cut_len.float()[eff_len > 0] / eff_len.float()[eff_len > 0])

    if soft_targets is not None:
        t1 = soft_targets
        t2 = soft_targets[perm]
    elif build_targets is None:
        t1 = F.one_hot(y_idx, num_classes=C).float()
        t2 = F.one_hot(y_p, num_classes=C).float()
    else:
        t1, t2 = build_targets(y_idx), build_targets(y_p)
    mixed_targets = lam_eff.view(B, 1) * t1 + (1 - lam_eff.view(B, 1)) * t2

    # --- return both training outputs and visualization info ---
    return {
        "mixed_wavs": mixed_wavs,
        "mixed_targets": mixed_targets,
        "lam_eff": lam_eff,
        "starts": starts,
        "cut_lens": cut_len,
        "perm": perm
    }


# -------------------------------------------------
# Embedding-level MixUp / CutMix (for pooled or framewise embeddings)
# -------------------------------------------------

def embedding_mixup(
    embeds: torch.Tensor,               # pooled: [B, D] OR framewise: [B, T, D]
    y_idx: torch.Tensor,                # [B]
    C: int,
    alpha: float = 0.4,
    per_sample: bool = True,
    build_targets: Callable[[torch.Tensor], torch.Tensor] | None = None,
    soft_targets: torch.Tensor | None = None,
):
    """
    MixUp applied directly on embeddings (pooled or framewise).

    Args:
        embeds: [B,D] or [B,T,D]
        y_idx: [B]
        C: number of classes
        alpha: Beta distribution parameter
        per_sample: lambda per sample vs shared
        build_targets: optional label smoothing function

    Returns:
        mixed_embeds: same shape as embeds
        mixed_targets: [B,C]
        lam: [B] mixing ratio
    """
    device = embeds.device
    B = embeds.size(0)

    # permutation
    perm = torch.randperm(B, device=device)
    if torch.all(perm == torch.arange(B, device=device)):
        perm = torch.roll(perm, 1)

    emb_p, y_p = embeds[perm], y_idx[perm]

    # lambda sampling
    lam = torch.distributions.Beta(alpha, alpha).rsample((B,)).to(device)
    if not per_sample:
        lam = lam.mean().expand(B)

    # reshape for broadcasting
    if embeds.dim() == 2:
        # pooled [B,D]
        lam_w = lam.view(B, 1)
    else:
        # framewise [B,T,D]
        lam_w = lam.view(B, 1, 1)

    mixed_embeds = lam_w * embeds + (1 - lam_w) * emb_p

    # target mixing
    if soft_targets is not None:
        t1 = soft_targets
        t2 = soft_targets[perm]
    elif build_targets is None:
        t1 = F.one_hot(y_idx, num_classes=C).float()
        t2 = F.one_hot(y_p, num_classes=C).float()
    else:
        t1, t2 = build_targets(y_idx), build_targets(y_p)
    mixed_targets = lam.view(B,1) * t1 + (1 - lam.view(B,1)) * t2
    return mixed_embeds, mixed_targets, lam


def embedding_cutmix(
    embeds: torch.Tensor,               # pooled: [B,D] OR framewise: [B,T,D]
    y_idx: torch.Tensor,                
    C: int,
    alpha: float = 0.4,
    per_sample: bool = True,
    build_targets: Callable[[torch.Tensor], torch.Tensor] | None = None,
    soft_targets: torch.Tensor | None = None,
):
    """
    CutMix applied on embeddings.

    - For pooled embeddings: falls back to MixUp-like behavior.
    - For framewise embeddings: replaces a contiguous time segment.

    Args:
        embeds: [B,D] or [B,T,D]
        y_idx: [B]
        C: number of classes
    """
    device = embeds.device
    B = embeds.size(0)

    perm = torch.randperm(B, device=device)
    if torch.all(perm == torch.arange(B, device=device)):
        perm = torch.roll(perm, 1)

    emb_p, y_p = embeds[perm], y_idx[perm]

    # lambda sampling
    lam = torch.distributions.Beta(alpha, alpha).rsample((B,)).to(device)
    if not per_sample:
        lam = lam.mean().expand(B)

    # If pooled: [B,D] → behave like MixUp
    if embeds.dim() == 2:
        lam_w = lam.view(B,1)
        mixed_embeds = lam_w * embeds + (1 - lam_w) * emb_p
        lam_eff = lam.clone()
    else:
        # framewise: [B,T,D]
        _, T, _ = embeds.shape
        cut_frac = 1.0 - lam
        cut_len = (cut_frac * T).round().clamp_min(1).long()
        max_start = (T - cut_len + 1).clamp_min(1)
        starts = (torch.rand_like(cut_len.float()) * max_start.float()).floor().long()

        idx = torch.arange(T, device=device).view(1, T)
        mask = ((idx >= starts.view(-1,1)) & (idx < (starts + cut_len).view(-1,1))).float().unsqueeze(-1)
        # [B,T,1]

        mixed_embeds = embeds * (1 - mask) + emb_p * mask
        lam_eff = 1.0 - (cut_len.float() / T)

    # mix labels
    if soft_targets is not None:
        t1 = soft_targets
        t2 = soft_targets[perm]
    elif build_targets is None:
        t1 = F.one_hot(y_idx, num_classes=C).float()
        t2 = F.one_hot(y_p,   num_classes=C).float()
    else:
        t1, t2 = build_targets(y_idx), build_targets(y_p)
    mixed_targets = lam_eff.view(B,1) * t1 + (1 - lam_eff.view(B,1)) * t2
    return mixed_embeds, mixed_targets, lam_eff