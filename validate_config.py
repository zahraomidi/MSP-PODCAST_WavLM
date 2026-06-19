# validate_config.py
import os

def _req(hp, key, typ=None, allow_none=False):
    if key not in hp:
        raise RuntimeError(f"[YAML] Missing required key: '{key}'")
    val = hp[key]
    if val is None and not allow_none:
        raise RuntimeError(f"[YAML] '{key}' is None but must be set")
    if typ is not None and val is not None and not isinstance(val, typ):
        raise RuntimeError(f"[YAML] '{key}' has wrong type: expected {typ}, got {type(val)}")
    return val

def _exists(path, keyname):
    if path and not os.path.exists(path):
        raise RuntimeError(f"[YAML] {keyname} path does not exist: {path}")

def validate_hparams(hp: dict) -> None:
    # ---- identity ----
    _req(hp, "seed", int)
    _req(hp, "pretrained_model", str)

    # ---- paths ----
    _exists(_req(hp, "data_folder", str), "data_folder")
    _exists(_req(hp, "train_annotation", str), "train_annotation")
    _exists(_req(hp, "valid_annotation", str), "valid_annotation")
    _exists(_req(hp, "test_annotation",  str), "test_annotation")
    _req(hp, "output_folder", str)
    _req(hp, "save_folder", str)

    # ---- class space ----
    C = int(_req(hp, "out_n_neurons_cls", int))
    emo_classes = _req(hp, "emo_classes", (list, tuple))
    if len(emo_classes) != C:
        raise RuntimeError(f"[YAML] len(emo_classes)={len(emo_classes)} != out_n_neurons_cls={C}")

    # ---- label mode ----
    dist_mode = str(_req(hp, "dist_mode", str)).lower()
    if dist_mode not in {"hard", "primary", "secondary", "merged"}:
        raise RuntimeError(f"[YAML] dist_mode invalid: {dist_mode}")

    # ---- curriculum ----
    key = str(hp.get("ambiguity_signal_key", "emo_entropy_norm")).lower()
    if key not in {
        "emo_entropy_norm",
        "emo_entropy_norm_merged",
        "emo_margin",
        "emo_maxprob",
        "ambiguity_signal",
    }:
        raise RuntimeError(f"[YAML] ambiguity_signal_key unsupported: {key}")

    ecfg = hp.get("entropy_curriculum", None)
    if ecfg is not None:
        if not isinstance(ecfg, dict):
            raise RuntimeError(f"[YAML] entropy_curriculum must be dict, got {type(ecfg)}")

        if bool(ecfg.get("enabled", False)):
            mode = str(ecfg.get("mode", "")).lower()
            if mode not in {"filter", "filter_rev", "weight", "weight_rev"}:
                raise RuntimeError(
                    f"[YAML] entropy_curriculum.mode must be filter|filter_rev|weight|weight_rev, got {mode}"
                )

            for k in ("warmup_epochs", "ramp_epochs"):
                if k not in ecfg or not isinstance(ecfg[k], int):
                    raise RuntimeError(f"[YAML] entropy_curriculum.{k} must exist and be int")
                if int(ecfg[k]) < 0:
                    raise RuntimeError(f"[YAML] entropy_curriculum.{k} must be >= 0")

            if mode in {"filter", "filter_rev"}:
                for k in ("start_thr", "end_thr"):
                    if k not in ecfg or not isinstance(ecfg[k], (int, float)):
                        raise RuntimeError(f"[YAML] entropy_curriculum.{k} must exist and be float")
                start_thr = float(ecfg["start_thr"])
                end_thr = float(ecfg["end_thr"])
                if not (0.0 <= start_thr <= 1.0 and 0.0 <= end_thr <= 1.0):
                    raise RuntimeError("[YAML] entropy_curriculum filter thresholds must be in [0,1]")
                if start_thr > end_thr:
                    raise RuntimeError("[YAML] entropy_curriculum.start_thr must be <= end_thr")

            if mode in {"weight", "weight_rev"}:
                for k in ("alpha_start", "alpha_end", "min_weight"):
                    if k not in ecfg or not isinstance(ecfg[k], (int, float)):
                        raise RuntimeError(f"[YAML] entropy_curriculum.{k} must exist and be float")
                min_weight = float(ecfg["min_weight"])
                if not (0.0 <= min_weight <= 1.0):
                    raise RuntimeError("[YAML] entropy_curriculum.min_weight must be in [0,1]")

    # ---- optimizer lrs ----
    _req(hp, "ssl_lr", (int, float))
    _req(hp, "head_lr", (int, float))

    # ---- augmentation sanity ----
    sr = int(_req(hp, "sample_rate", int))
    if sr != 16000:
        raise RuntimeError(f"[YAML] sample_rate must be 16000 (pipeline assumption). Got {sr}")

    aug = hp.get("augmentation", {})
    if aug is not None and not isinstance(aug, dict):
        raise RuntimeError("[YAML] augmentation must be a dict")

    # ---- schedulers ----
    # If your code expects these for resume reproducibility, make sure they exist.
    if "lr_annealing" not in hp:
        raise RuntimeError("[YAML] Missing lr_annealing")
    if "lr_annealing_ssl" not in hp:
        raise RuntimeError("[YAML] Missing lr_annealing_ssl")
