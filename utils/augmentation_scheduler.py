class AugmentationScheduler:
    """
    Epoch-aware controller for augmentation probabilities.

    - base_aug_cfg: dict like:
        {
          "noise": {"p": 0.15, "snr_min": 10, ...},
          "time_mask": {"p": 0.1, "max_mask_pct": 0.05, ...},
          ...
        }

    - scheduler_cfg: dict from YAML under `augmentation_scheduler`.
    """

    def __init__(self, scheduler_cfg, base_aug_cfg):
        self.enabled = scheduler_cfg.get("enabled", False)
        self.phases = scheduler_cfg.get("phases", [])
        self.base_aug_cfg = base_aug_cfg or {}

    def get_phase(self, epoch: int):
        """Return the active phase for this epoch, or None if none matches."""
        for ph in self.phases:
            if ph["epoch_start"] <= epoch <= ph["epoch_end"]:
                return ph
        return None

    def compute_scale(self, phase, epoch: int) -> float:
        """Compute global scale factor for a phase at the given epoch."""
        if "scale" in phase:
            return float(phase["scale"])

        if "scale_start" in phase and "scale_end" in phase:
            start = phase["epoch_start"]
            end = phase["epoch_end"]
            if end <= start:
                return float(phase["scale_end"])
            progress = (epoch - start) / (end - start)
            return float(phase["scale_start"] + progress * (phase["scale_end"] - phase["scale_start"]))

        return 1.0

    def get_effective_aug_config(self, epoch: int, return_phase_info=False):
        """
        Returns:
          effective_cfg: dict with same structure as base_aug_cfg but scaled `p`
          allowed_augs: set of aug names allowed in this phase (or empty set)
          allow_combo:  bool flag for multi-aug per sample
        """
        if not self.enabled:
            # No scheduling at all → just return base config, but we still
            # expose allowed_augs=None and allow_combo=False for logging.
            if return_phase_info:
                return self.base_aug_cfg, None, False, "disabled", 1.0
            return self.base_aug_cfg, None, False

        phase = self.get_phase(epoch)
        if phase is None:
            # Outside any defined phase → fallback to base config
            if return_phase_info:
                return self.base_aug_cfg, None, False, "none", 1.0
            return self.base_aug_cfg, None, False

        scale = self.compute_scale(phase, epoch)
        allowed = set(phase.get("allowed_augs", []))
        allow_combo = bool(phase.get("allow_combinations", False))

        effective_cfg = {}
        for aug_name, cfg in self.base_aug_cfg.items():
            cfg = dict(cfg)  # shallow copy
            base_p = float(cfg.get("p", 0.0) or 0.0)

            if allowed and aug_name not in allowed:
                cfg["p"] = 0.0
            else:
                cfg["p"] = float(base_p * scale)

            effective_cfg[aug_name] = cfg

        # Optional debug logging
        if phase is not None:
            print(f"[AUG-SCHED][Epoch {epoch}] phase={phase.get('name')} scale={scale} allowed={allowed} allow_combo={allow_combo}")
            print(f"[AUG-SCHED][Epoch {epoch}] effective_cfg={effective_cfg}")

        if return_phase_info:
            return effective_cfg, allowed, allow_combo, phase.get("name", None), scale

        # Optional debug logging
        if phase is not None:
            print(f"[AUG-SCHED][Epoch {epoch}] phase={phase.get('name')} scale={scale} allowed={allowed} allow_combo={allow_combo}")
            print(f"[AUG-SCHED][Epoch {epoch}] effective_cfg={effective_cfg}")

        return effective_cfg, allowed, allow_combo