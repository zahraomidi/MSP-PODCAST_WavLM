#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soft_label_utils.py
----------------------------------
Utilities for one-hot categorical targets and optional waveform MixUp/CutMix.

The public example configuration uses dataset-provided label distributions
and keeps MixUp/CutMix disabled by default.

Includes:
 - One-hot categorical targets
 - Optional waveform MixUp helper
 - Optional waveform CutMix helper
"""

import torch
import torch.nn.functional as F
from typing import Callable, Tuple

# -------------------------------------------------
# Target and waveform augmentation utilities
# -------------------------------------------------

__all__ = ["one_hot", "apply_mixup", "apply_cutmix"]

def one_hot(y_idx: torch.Tensor, C: int) -> torch.Tensor:
    return F.one_hot(y_idx, num_classes=C).float()

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
