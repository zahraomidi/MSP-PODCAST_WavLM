"""Checkpoint metadata, path resolution, and reload helpers."""

import math
import os
import re

import torch

from utils.entropy_curriculum import (
    _load_curriculum_state,
    _save_curriculum_state,
)
from utils.unfreeze_state import _load_unfreeze_state, _save_unfreeze_state

__all__ = [
    "_checkpoint_meta_key",
    "_find_ckpt",
    "_is_ckpt_dir",
    "_load_state_dict_safe",
    "_metric_key_candidates",
    "_missing_recoverables_in_ckpt",
    "_normalize_metric_name",
    "_normalize_path",
    "_read_epoch_from_ckpt_yaml",
    "_resolve_ckpt_paths",
    "_resolve_metric_from_stats",
    "load_from_ckpt",
]


def _normalize_metric_name(metric_name):
    """Normalize configurable metric names to a stable internal form."""
    return str(metric_name or "").strip().lower().replace("-", "_")


def _metric_key_candidates(metric_name):
    """Return acceptable stats/meta keys for a configured metric."""
    metric_name = _normalize_metric_name(metric_name)
    alias_map = {
        "macrof1": ("macro_f1",),
        "macro_f1": ("macro_f1",),
        "ua": ("uar", "CAT_UA"),
        "uar": ("uar", "CAT_UA"),
        "cat_ua": ("CAT_UA", "uar"),
        "acc": ("acc", "CAT_ACC"),
        "cat_acc": ("CAT_ACC", "acc"),
        "ccc": ("ccc_avg", "ccc"),
        "ccc_avg": ("ccc_avg", "ccc"),
        "vad_ccc": ("ccc_avg", "ccc"),
        "kld": ("kld",),
        "jsd": ("jsd",),
    }
    return alias_map.get(metric_name, (metric_name,))


def _checkpoint_meta_key(metric_name):
    """Return the canonical checkpoint metadata key for a configured metric."""
    metric_name = _normalize_metric_name(metric_name)
    if metric_name in {"ua", "uar", "cat_ua"}:
        return "uar"
    if metric_name in {"acc", "cat_acc"}:
        return "acc"
    if metric_name in {"ccc", "ccc_avg", "vad_ccc"}:
        return "ccc_avg"
    if metric_name in {"macrof1", "macro_f1"}:
        return "macro_f1"
    return metric_name


def _resolve_metric_from_stats(stats, metric_name):
    """Resolve a configured metric name against a stage stats dict."""
    for key in _metric_key_candidates(metric_name):
        if key not in stats or stats.get(key) is None:
            continue
        try:
            value = float(stats[key])
        except Exception:
            continue
        if math.isfinite(value):
            return value, key
    return None, None


def _normalize_path(path) -> str:
    """Normalize filesystem paths for robust comparisons."""
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def _is_ckpt_dir(path: str) -> bool:
    """True when `path` points to a concrete SpeechBrain checkpoint directory."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "CKPT.yaml"))


def _resolve_ckpt_paths(path: str):
    """Resolve checkpoint input into search, exact checkpoint, and payload dirs."""
    p = os.path.normpath(os.path.expanduser(str(path)))

    if os.path.isfile(p):
        p = os.path.dirname(p)

    if _is_ckpt_dir(p):
        return os.path.dirname(p), p, p

    return p, None, p


def _read_epoch_from_ckpt_yaml(ckpt_dir: str):
    """Read `epoch` field from CKPT.yaml if present."""
    try:
        meta_path = os.path.join(str(ckpt_dir), "CKPT.yaml")
        if not os.path.isfile(meta_path):
            return None
        epoch_val = None
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                match = re.match(r"^\s*epoch\s*:\s*([0-9]+)\s*$", line)
                if match:
                    epoch_val = int(match.group(1))
        return epoch_val
    except Exception:
        return None


def _missing_recoverables_in_ckpt(ckpt_dir: str, recoverables: dict):
    """Return recoverable keys whose `<key>.ckpt` payload is missing."""
    missing = []
    try:
        if not ckpt_dir or not os.path.isdir(ckpt_dir):
            return list(recoverables.keys())
        for name, obj in (recoverables or {}).items():
            if obj is None:
                continue
            path = os.path.join(ckpt_dir, f"{name}.ckpt")
            if not os.path.isfile(path):
                missing.append(str(name))
    except Exception:
        pass
    return missing


def _find_ckpt(path: str, names):
    """Return first existing ckpt file matching any name in `names`."""
    for name in names:
        ckpt = os.path.join(path, f"{name}.ckpt")
        if os.path.isfile(ckpt):
            return ckpt
    return None


@torch.no_grad()
def _load_state_dict_safe(module: torch.nn.Module, ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(
            f"[warn] Loaded {os.path.basename(ckpt_path)} "
            f"with missing={len(missing)}, unexpected={len(unexpected)}"
        )


def load_from_ckpt(
    ckpt_path,
    device,
    ssl_model=None,
    cat_mlp=None,
    vad_mlp=None,
    tc_gru_head=None,
    model=None,
    brain=None,
    label_encoder=None,
    optimizer=None,
    lr_annealing=None,
    lr_annealing_ssl=None,
    epoch_counter=None,
    unfreeze_state=None,
    curriculum_state=None,
    dataloader_train=None,
    mode="backbone+heads",
):
    """Load training state from a SpeechBrain checkpoint directory or .ckpt.

    mode:
      - 'all'             : resume training state (models + optimizer + schedulers + counters [+ optional dataloader])
      - 'backbone'        : load only ssl_model weights
      - 'backbone+heads'  : load ssl_model + heads (cat/vad) weights

    Notes:
      - This pipeline uses a *single* optimizer with param groups (SSL vs heads).
        Therefore there is no separate ssl_opt.
      - lr_annealing_ssl is the SSL-group scheduler (compatibility alias handled by caller).
    """
    from speechbrain.utils.checkpoints import Checkpointer

    search_dir, target_ckpt_dir, payload_dir = _resolve_ckpt_paths(ckpt_path)
    restore_info = {
        "loaded": False,
        "search_dir": str(search_dir),
        "target_ckpt_dir": str(target_ckpt_dir) if target_ckpt_dir is not None else None,
        "loaded_ckpt_dir": None,
        "meta_epoch": None,
        "missing_recoverables": [],
        "optimizer_state_restored": None,
        "full_state": None,
    }

    if mode == "all":
        counter_before = None
        if epoch_counter is not None and hasattr(epoch_counter, "current"):
            try:
                counter_before = int(getattr(epoch_counter, "current"))
            except Exception:
                counter_before = None

        recoverables = {}
        if ssl_model is not None:
            recoverables["ssl_model"] = ssl_model
        if cat_mlp is not None:
            recoverables["cat_mlp"] = cat_mlp
        if vad_mlp is not None:
            recoverables["vad_mlp"] = vad_mlp
        if tc_gru_head is not None:
            recoverables["tc_gru_head"] = tc_gru_head
        if model is not None:
            recoverables["model"] = model
        if brain is not None:
            recoverables["brain"] = brain
        if label_encoder is not None:
            recoverables["label_encoder"] = label_encoder
        if optimizer is not None:
            recoverables["optimizer"] = optimizer
        if lr_annealing is not None:
            recoverables["lr_annealing"] = lr_annealing
        if lr_annealing_ssl is not None:
            recoverables["lr_annealing_ssl"] = lr_annealing_ssl
        if epoch_counter is not None:
            recoverables["counter"] = epoch_counter
        if unfreeze_state is not None:
            recoverables["unfreeze_state"] = unfreeze_state
        if curriculum_state is not None:
            recoverables["curriculum_state"] = curriculum_state

        if dataloader_train is not None:
            recoverables["dataloader-TRAIN"] = dataloader_train

        cp = Checkpointer(checkpoints_dir=search_dir, recoverables=recoverables)
        try:
            if hasattr(cp, "custom_save_hooks") and hasattr(cp, "custom_load_hooks"):
                if unfreeze_state is not None:
                    cp.custom_save_hooks["unfreeze_state"] = _save_unfreeze_state
                    cp.custom_load_hooks["unfreeze_state"] = _load_unfreeze_state
                if curriculum_state is not None:
                    cp.custom_save_hooks["curriculum_state"] = _save_curriculum_state
                    cp.custom_load_hooks["curriculum_state"] = _load_curriculum_state
            if hasattr(cp, "optional_recoverables"):
                if unfreeze_state is not None:
                    cp.optional_recoverables["unfreeze_state"] = True
                if curriculum_state is not None:
                    cp.optional_recoverables["curriculum_state"] = True
        except Exception:
            pass

        recovered_ckpt = None
        if target_ckpt_dir is not None:
            target_norm = _normalize_path(target_ckpt_dir)
            ckpt_pred = (
                lambda ckpt, _target_norm=target_norm:
                _normalize_path(getattr(ckpt, "path", "")) == _target_norm
            )
            try:
                recovered_ckpt = cp.recover_if_possible(
                    allow_partial=True,
                    ckpt_predicate=ckpt_pred,
                )
            except TypeError:
                try:
                    recovered_ckpt = cp.recover_if_possible(ckpt_predicate=ckpt_pred)
                except TypeError:
                    recovered_ckpt = cp.recover_if_possible()
        else:
            try:
                recovered_ckpt = cp.recover_if_possible(allow_partial=True)
            except TypeError:
                recovered_ckpt = cp.recover_if_possible()

        counter_after = None
        if epoch_counter is not None and hasattr(epoch_counter, "current"):
            try:
                counter_after = int(getattr(epoch_counter, "current"))
            except Exception:
                counter_after = None

        loaded = recovered_ckpt is not None
        if (not loaded) and (counter_before is not None) and (counter_after is not None):
            loaded = counter_after != counter_before

        used_ckpt_dir = None
        if recovered_ckpt is not None:
            try:
                used_ckpt_dir = str(getattr(recovered_ckpt, "path", None) or "")
            except Exception:
                used_ckpt_dir = None
        if not used_ckpt_dir:
            used_ckpt_dir = target_ckpt_dir

        meta_epoch = None
        if recovered_ckpt is not None:
            try:
                meta_epoch = (getattr(recovered_ckpt, "meta", {}) or {}).get("epoch", None)
                if meta_epoch is not None:
                    meta_epoch = int(meta_epoch)
            except Exception:
                meta_epoch = None
        if meta_epoch is None and used_ckpt_dir is not None:
            meta_epoch = _read_epoch_from_ckpt_yaml(used_ckpt_dir)

        expected_state = {
            "optimizer": optimizer,
            "lr_annealing": lr_annealing,
            "lr_annealing_ssl": lr_annealing_ssl,
            "counter": epoch_counter,
            "unfreeze_state": unfreeze_state,
            "curriculum_state": curriculum_state,
        }
        missing = _missing_recoverables_in_ckpt(used_ckpt_dir, expected_state)
        opt_state_restored = None
        if optimizer is not None:
            try:
                opt_state_restored = len(getattr(optimizer, "state", {})) > 0
            except Exception:
                opt_state_restored = None

        restore_info.update(
            {
                "loaded": bool(loaded),
                "loaded_ckpt_dir": str(used_ckpt_dir) if used_ckpt_dir is not None else None,
                "meta_epoch": int(meta_epoch) if meta_epoch is not None else None,
                "missing_recoverables": list(missing),
                "optimizer_state_restored": opt_state_restored,
                "full_state": len(missing) == 0,
            }
        )
        if restore_info["loaded"]:
            print("[info] Full resume from", restore_info["loaded_ckpt_dir"] or search_dir)
            if missing:
                print("[warn] Partial restore: missing recoverables:", ",".join(missing))
            if opt_state_restored is False:
                print("[warn] Optimizer state appears empty after restore (LR/momenta likely reset).")
        else:
            print(
                "[warn] Full resume failed (no checkpoint matched).",
                f"ckpt_path={ckpt_path} resolved_search_dir={search_dir}",
            )
        return restore_info

    def _load_head(name, module, aliases):
        if module is None:
            return False
        ckpt = _find_ckpt(payload_dir, aliases)
        if ckpt:
            _load_state_dict_safe(module, ckpt, device)
            print(f"[info] {name} loaded from {os.path.basename(ckpt)}")
            return True
        print(f"[warn] no checkpoint found for {name} (looked for {aliases})")
        return False

    loaded_any = False
    if mode in ("backbone", "backbone+heads", "all"):
        loaded_any = _load_head("ssl_model", ssl_model, ["ssl_model"]) or loaded_any

    if mode in ("backbone+heads", "all"):
        loaded_any = _load_head("cat_mlp", cat_mlp, ["cat_mlp", "clf_mlp"]) or loaded_any
        loaded_any = _load_head("vad_mlp", vad_mlp, ["vad_mlp", "output_mlp"]) or loaded_any
        loaded_any = _load_head("tc_gru_head", tc_gru_head, ["tc_gru_head", "tc_gru"]) or loaded_any
    else:
        print("[info] mode='backbone': heads not loaded")

    print("[info] Retune mode: optimizers/schedulers start fresh.")
    restore_info["loaded"] = bool(loaded_any)
    restore_info["loaded_ckpt_dir"] = str(payload_dir)
    restore_info["meta_epoch"] = _read_epoch_from_ckpt_yaml(payload_dir)
    return restore_info
