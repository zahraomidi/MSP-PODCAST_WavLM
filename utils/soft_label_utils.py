"""One-hot categorical target utility."""

import torch
import torch.nn.functional as F

__all__ = ["one_hot"]


def one_hot(y_idx: torch.Tensor, C: int) -> torch.Tensor:
    return F.one_hot(y_idx, num_classes=C).float()
