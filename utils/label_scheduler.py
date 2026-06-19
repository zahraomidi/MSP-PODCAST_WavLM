# --- LabelScheduler helper ---
class LabelScheduler:
    """
    Simple epoch-based scheduler for classifier target sources.

    Expects a config dict in hparams under 'label_scheduler', with:
      label_scheduler:
        phases:
          - {start: 1, end: 5, mode: "hard"}
          - {start: 6, end: 15, mode: "primary"}
          - {start: 16, end: 100, mode: "secondary"}

    Supported modes (lowercased):
      - "hard"      -> use hard one-hot labels (y_idx)
      - "primary"   -> use emo_dist_primary from JSON (via dist_pipeline)
      - "secondary" -> use emo_dist_secondary
      - "merged"    -> use merged emo_dist
    """
    def __init__(self, cfg):
        self.phases = []
        self.default_mode = "merged"

        if not cfg:
            return

        # Allow top-level default_mode override
        if isinstance(cfg, dict) and "default_mode" in cfg:
            self.default_mode = str(cfg["default_mode"]).lower()

        phases = cfg.get("phases", []) if isinstance(cfg, dict) else []
        for ph in phases:
            if not isinstance(ph, dict):
                continue
            start = int(ph.get("start", 1))
            end = int(ph.get("end", start))
            mode = str(ph.get("mode", self.default_mode)).lower()
            self.phases.append((start, end, mode))

        if self.phases:
            # If no explicit default_mode was given, use the last phase's mode
            if "default_mode" not in (cfg if isinstance(cfg, dict) else {}):
                self.default_mode = self.phases[-1][2]

    def get_mode(self, epoch: int) -> str:
        """Return the active mode for the given epoch (1-based)."""
        if not self.phases:
            return self.default_mode

        for start, end, mode in self.phases:
            if start <= epoch <= end:
                return mode

        # Before first phase: use its mode; after last: use last mode
        if epoch < self.phases[0][0]:
            return self.phases[0][2]
        return self.phases[-1][2]
