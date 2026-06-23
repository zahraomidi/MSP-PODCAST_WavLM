#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metric_utils.py
----------------------------------
Metric utilities for VAD regression and categorical speech emotion recognition.

Provides:
 - Concordance correlation coefficient (CCC)
 - Root mean square error (RMSE)
 - Categorical accuracy, unweighted accuracy, precision, recall, and F1
 - Top-k accuracy
 - Calibration and expected calibration error (ECE)
 - Distribution statistics
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score



# ======================================================
# ---------- REGRESSION METRICS ----------
# ======================================================

@torch.no_grad()
def ccc(x: torch.Tensor, y: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Concordance correlation coefficient per dimension.
    Args:
        x, y: [N, D] tensors (e.g., [N, 3] for V/A/D)
        dim: dimension to reduce over
    Returns:
        Tensor [D] of CCC values
    """
    x_mean = x.mean(dim=dim)
    y_mean = y.mean(dim=dim)
    x_var = x.var(dim=dim, unbiased=False)
    y_var = y.var(dim=dim, unbiased=False)
    cov = ((x - x_mean) * (y - y_mean)).mean(dim=dim)
    return (2 * cov) / (x_var + y_var + (x_mean - y_mean).pow(2) + 1e-8)


@torch.no_grad()
def rmse(x: torch.Tensor, y: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Root mean square error per dimension."""
    return torch.sqrt(F.mse_loss(x, y, reduction="none").mean(dim=dim))


# ======================================================
# ---------- CATEGORICAL METRICS ----------
# ======================================================

@torch.no_grad()
def cat_metrics(
    cls_logp: torch.Tensor,
    y_idx: torch.Tensor,
    class_names=None,
) -> dict:
    """
    Compute categorical accuracy, unweighted accuracy (UA), and confusion matrix.

    Args:
        cls_logp: [N,C] log-probabilities
        y_idx:    [N]
        class_names: optional list of class strings

    Returns:
        dict with acc, ua, per-class acc, and confusion matrix
    """
    C = cls_logp.size(-1)
    pred = cls_logp.argmax(dim=-1)
    acc = (pred == y_idx).float().mean().item()

    per_class_acc = {}
    ua_vals = []
    for c in range(C):
        mask = (y_idx == c)
        if mask.any():
            pc_acc = (pred[mask] == c).float().mean().item()
            ua_vals.append(pc_acc)
            cname = class_names[c] if class_names else str(c)
            per_class_acc[cname] = pc_acc

    ua = float(np.mean(ua_vals)) if ua_vals else 0.0

    try:
        cm = confusion_matrix(
            y_idx.cpu().numpy(),
            pred.cpu().numpy(),
            labels=list(range(C)),
        ).tolist()
    except Exception:
        cm = None

    return dict(acc=acc, ua=ua, per_class_acc=per_class_acc, confmat=cm)


def topk_accuracy(logp: torch.Tensor, y: torch.Tensor, ks=(1,2,3)):
    """
    logp: [B, C] log-probs
    y:    [B] true labels
    """
    with torch.no_grad():
        probs = torch.exp(logp)
        maxk = max(ks)
        topk = probs.topk(maxk, dim=-1).indices  # [B, maxk]

        acc = {}
        for k in ks:
            correct = (topk[:, :k] == y.unsqueeze(1)).any(dim=1)
            acc[f"top{k}_acc"] = correct.float().mean().item()
        return acc
    

@torch.no_grad()
def compute_cls_extra_metrics(logp, y_true, class_names, zero_division=0) -> dict:
    """Compute robust classification metrics from log-probs.

    Args:
        logp: [N,C] log-probabilities
        y_true: [N] int labels
        class_names: list[str] of length C
        zero_division: forwarded to sklearn precision/recall

    Returns:
        dict with macro/weighted F1, per-class recall/precision.
    """
    y_pred = logp.argmax(dim=-1).detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()

    metrics = {}
    metrics["macro_f1"] = float(f1_score(y_true_np, y_pred, average="macro"))
    metrics["weighted_f1"] = float(f1_score(y_true_np, y_pred, average="weighted"))

    labels = list(range(len(class_names)))
    recalls = recall_score(
        y_true_np, y_pred, labels=labels, average=None, zero_division=zero_division
    )
    precisions = precision_score(
        y_true_np, y_pred, labels=labels, average=None, zero_division=zero_division
    )

    for i, cname in enumerate(class_names):
        metrics[f"recall_{cname}"] = float(recalls[i])
        metrics[f"precision_{cname}"] = float(precisions[i])

    return metrics

@torch.no_grad()
def confusion_matrix_from_logp(logp: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute integer confusion matrix from log-probs and true labels.

    Args:
        logp: [N,C] log-probabilities
        y_true: [N] int labels
        num_classes: C

    Returns:
        conf: [C,C] int64 tensor on CPU (rows=true, cols=pred)
    """
    preds = logp.argmax(dim=-1).view(-1).to(torch.int64)
    y = y_true.view(-1).to(torch.int64)

    # Clamp labels defensively (avoid crash if bad label slips in debug)
    y = torch.clamp(y, 0, num_classes - 1)
    preds = torch.clamp(preds, 0, num_classes - 1)

    idx = y * num_classes + preds
    bins = torch.bincount(idx.cpu(), minlength=num_classes * num_classes)
    conf = bins.view(num_classes, num_classes).to(torch.int64)
    return conf

@torch.no_grad()
def ece_from_bin_sums(conf_sum: torch.Tensor, acc_sum: torch.Tensor, count: torch.Tensor) -> float:
    """Compute Expected Calibration Error (ECE) from pre-accumulated per-bin sums.

    Args:
        conf_sum: [B] sum of confidences per bin
        acc_sum:  [B] sum of accuracies (0/1) per bin
        count:    [B] sample count per bin

    Returns:
        float ECE in [0,1]
    """
    conf_sum = conf_sum.to(torch.float32)
    acc_sum = acc_sum.to(torch.float32)
    count = count.to(torch.float32)

    total = float(count.sum().item())
    if total <= 0:
        return 0.0

    nonzero = count > 0
    avg_conf = torch.zeros_like(conf_sum)
    avg_acc = torch.zeros_like(acc_sum)
    avg_conf[nonzero] = conf_sum[nonzero] / count[nonzero]
    avg_acc[nonzero] = acc_sum[nonzero] / count[nonzero]

    ece = (torch.abs(avg_acc - avg_conf) * (count / total)).sum()
    return float(ece.item())


@torch.no_grad()
def ece_bin_sums_from_logp(logp: torch.Tensor, y_true: torch.Tensor, n_bins: int = 10):
    """Accumulate ECE bin sums from log-probs.

    Args:
        logp: [N,C] log-probabilities
        y_true: [N]
        n_bins: number of confidence bins

    Returns:
        conf_sum, acc_sum, count: each [n_bins] on CPU
    """
    probs = torch.exp(logp)
    conf, pred = probs.max(dim=-1)  # [N]
    correct = (pred == y_true).to(torch.float32)  # [N]

    # Bin edges in [0,1]
    conf = torch.clamp(conf, 0.0, 1.0)
    bin_idx = torch.clamp((conf * n_bins).to(torch.int64), 0, n_bins - 1)  # [N]

    conf_sum = torch.zeros(n_bins, dtype=torch.float32)
    acc_sum = torch.zeros(n_bins, dtype=torch.float32)
    count = torch.zeros(n_bins, dtype=torch.float32)

    # CPU accumulation (stable across devices / SB)
    bi = bin_idx.detach().cpu()
    conf_cpu = conf.detach().cpu().to(torch.float32)
    cor_cpu = correct.detach().cpu().to(torch.float32)

    for b in range(n_bins):
        m = (bi == b)
        if m.any():
            count[b] = float(m.sum().item())
            conf_sum[b] = conf_cpu[m].sum()
            acc_sum[b] = cor_cpu[m].sum()

    return conf_sum, acc_sum, count


# distribution_stats(dist) -> dict(mean_ent, mean_maxp, mean_margin, …)
@torch.no_grad()
def distribution_stats(dist: torch.Tensor, normalize: bool = True, eps: float = 1e-8) -> dict:
    """Compute summary statistics from categorical distributions.

    Works for both:
      - predicted probabilities (softmax outputs)
      - soft label vectors (annotator distributions)

    Args:
        dist: [N,C] tensor of nonnegative values (not necessarily normalized)
        normalize: whether to renormalize dist to sum=1 per row
        eps: numerical stability

    Returns:
        dict with:
          - mean_ent: mean entropy H(p)
          - mean_ent_norm: mean entropy normalized by log(C)
          - mean_maxp: mean max probability
          - mean_margin: mean (top1 - top2)
    """
    if dist is None:
        return {}
    if dist.ndim != 2:
        raise ValueError(f"distribution_stats expects [N,C], got shape={tuple(dist.shape)}")

    p = dist.to(torch.float32)
    p = torch.clamp(p, min=0.0)
    if normalize:
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)

    C = int(p.size(-1))
    # entropy
    p_safe = torch.clamp(p, min=eps)
    ent = -(p_safe * torch.log(p_safe)).sum(dim=-1)  # [N]
    ent_norm = ent / max(math.log(max(C, 2)), eps)
    ent_norm = torch.clamp(ent_norm, 0.0, 1.0)

    # max prob + margin
    top2 = torch.topk(p, k=min(2, C), dim=-1).values
    maxp = top2[:, 0]
    if C >= 2:
        margin = top2[:, 0] - top2[:, 1]
    else:
        margin = torch.zeros_like(maxp)

    return {
        "mean_ent": float(ent.mean().item()),
        "mean_ent_norm": float(ent_norm.mean().item()),
        "mean_maxp": float(maxp.mean().item()),
        "mean_margin": float(margin.mean().item()),
    }
