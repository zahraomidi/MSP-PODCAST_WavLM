"""State, scheduling, and summary helpers for entropy curriculum training."""

import numpy as np
import torch


class EntropyCurriculumState:
    """Minimal persisted state for entropy curriculum recovery."""

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "none",
        phase: str = "disabled",
        last_epoch: int = -1,
        last_threshold=None,
        last_alpha=None,
        last_min_weight=None,
    ):
        self.enabled = bool(enabled)
        self.mode = str(mode)
        self.phase = str(phase)
        self.last_epoch = int(last_epoch)
        self.last_threshold = last_threshold
        self.last_alpha = last_alpha
        self.last_min_weight = last_min_weight
        self._restored_from_ckpt = False

    def state_dict(self):
        return {
            "enabled": bool(self.enabled),
            "mode": str(self.mode),
            "phase": str(self.phase),
            "last_epoch": int(self.last_epoch),
            "last_threshold": self.last_threshold,
            "last_alpha": self.last_alpha,
            "last_min_weight": self.last_min_weight,
        }

    def load_state_dict(self, state):
        self.enabled = bool(state.get("enabled", False))
        self.mode = str(state.get("mode", "none"))
        self.phase = str(state.get("phase", "disabled"))
        self.last_epoch = int(state.get("last_epoch", -1))
        self.last_threshold = state.get("last_threshold", None)
        self.last_alpha = state.get("last_alpha", None)
        self.last_min_weight = state.get("last_min_weight", None)
        self._restored_from_ckpt = True


def _save_curriculum_state(obj, path):
    """Save EntropyCurriculumState as a plain checkpoint dictionary."""
    torch.save(obj.state_dict(), path)


def _load_curriculum_state(obj, path, end_of_epoch):
    """Restore EntropyCurriculumState from a checkpoint dictionary."""
    state = torch.load(path, map_location="cpu")
    obj.load_state_dict(state)


def _get_entropy_bin_edges(raw=None):
    """Return four monotonic edges for low/mid/high entropy bins."""
    default = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
    if raw is None:
        raw = default
    try:
        edges = [float(x) for x in raw]
        if len(edges) != 4:
            return default
        if not np.all(np.isfinite(edges)):
            return default
        if any(edges[i] > edges[i + 1] for i in range(3)):
            return default
        edges[0] = 0.0
        edges[-1] = 1.0
        return edges
    except Exception:
        return default


def _entropy_curriculum_policy(cfg, epoch: int):
    """Return the active curriculum mode, threshold, alpha, and minimum weight."""
    cfg = cfg or {}
    if not bool(cfg.get("enabled", False)):
        return "none", None, None, None

    mode = str(cfg.get("mode", "none")).lower()
    warmup = max(0, int(cfg.get("warmup_epochs", 0)))
    ramp = max(0, int(cfg.get("ramp_epochs", 0)))

    if epoch is not None and int(epoch) <= warmup:
        return "none", None, None, None

    if epoch is None:
        t = 1.0
    elif ramp <= 1:
        t = 1.0
    else:
        e = max(0, int(epoch) - warmup - 1)
        t = min(1.0, max(0.0, e / float(ramp - 1)))

    if mode in {"filter", "filter_rev"}:
        q_sched = cfg.get("quantile_schedule", None)
        e_sched = cfg.get("schedule_epochs", None)
        if (
            isinstance(q_sched, (list, tuple))
            and isinstance(e_sched, (list, tuple))
            and len(q_sched) > 0
            and len(q_sched) == len(e_sched)
        ):
            cur_epoch = 0 if epoch is None else int(epoch)
            stage_idx = 0
            for i, e0 in enumerate(e_sched):
                if cur_epoch >= int(e0):
                    stage_idx = i
            q = float(q_sched[min(stage_idx, len(q_sched) - 1)])
            q = min(1.0, max(0.0, q))
            return mode, float(q), None, None

        start_thr = float(cfg.get("start_thr", 0.25))
        end_thr = float(cfg.get("end_thr", 1.0))
        thr = start_thr + t * (end_thr - start_thr)
        return mode, float(thr), None, None

    if mode in {"weight", "weight_rev"}:
        a0 = float(cfg.get("alpha_start", 2.0))
        a1 = float(cfg.get("alpha_end", 0.0))
        alpha = a0 + t * (a1 - a0)
        min_w = float(cfg.get("min_weight", 0.10))
        return mode, None, float(alpha), float(min_w)

    return "none", None, None, None


def _summarize_entropy_weight_epoch(values, min_count, total_count):
    """Summarize TRAIN weights and the fraction clamped to minimum weight."""
    n_min = int(min_count or 0)
    n_tot = int(total_count or 0)
    frac_at_min = (float(n_min) / float(n_tot)) if n_tot > 0 else None
    if not values:
        return None, None, None, None, None, None, frac_at_min

    try:
        weights = torch.cat(values, dim=0).to(dtype=torch.float32).view(-1)
    except Exception:
        return None, None, None, None, None, None, frac_at_min

    if int(weights.numel()) == 0:
        return None, None, None, None, None, None, frac_at_min

    try:
        quantiles = torch.quantile(
            weights,
            torch.tensor(
                [0.10, 0.50, 0.90],
                dtype=weights.dtype,
                device=weights.device,
            ),
        )
        return (
            float(weights.mean().item()),
            float(weights.min().item()),
            float(weights.max().item()),
            float(quantiles[0].item()),
            float(quantiles[1].item()),
            float(quantiles[2].item()),
            frac_at_min,
        )
    except Exception:
        array = weights.detach().cpu().numpy()
        return (
            float(array.mean()),
            float(array.min()),
            float(array.max()),
            float(np.percentile(array, 10)),
            float(np.percentile(array, 50)),
            float(np.percentile(array, 90)),
            frac_at_min,
        )
