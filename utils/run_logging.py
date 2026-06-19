from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _emit_debug(brain: Any, msg: str) -> None:
    fn = getattr(brain, "_debug", None)
    if callable(fn):
        fn(msg)
    else:
        print(f"[DEBUG] {msg}", flush=True)


def _get_module(brain: Any, name: str) -> Any:
    mods = getattr(brain, "modules", None)
    if mods is None:
        return None
    if hasattr(mods, name):
        return getattr(mods, name)
    try:
        return mods[name]
    except Exception:
        return None


def _resolve_ckpt_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = str(path)
    if p.endswith(".yaml") or p.endswith(".ckpt"):
        return os.path.dirname(p)
    return p


def _resolve_precision(brain: Any) -> str:
    run_opts = getattr(brain, "run_opts", {}) or {}
    if isinstance(run_opts, dict):
        if run_opts.get("precision") is not None:
            return str(run_opts.get("precision"))
        if bool(run_opts.get("bfloat16_mix_prec", False)):
            return "bf16"
        if bool(run_opts.get("auto_mix_prec", False)):
            return "fp16"
    hp_prec = _cfg_get(getattr(brain, "hparams", None), "precision", None)
    if hp_prec is not None:
        return str(hp_prec)
    return "fp32"


def _normalize_label_mode(mode: Any) -> str:
    m = str(mode if mode is not None else "unknown").strip().lower()
    if m == "hard":
        return "hard"
    if m.startswith("primary"):
        return "primary-dist"
    if m.startswith("merged"):
        return "merged-dist"
    if m.startswith("secondary"):
        return "secondary-dist"
    return m


def _resolve_label_mode(brain: Any) -> str:
    mode = getattr(brain, "current_label_mode", None)
    if mode is None:
        sched = getattr(brain, "label_scheduler", None)
        if sched is not None and hasattr(sched, "get_mode"):
            try:
                mode = sched.get_mode(1)
            except Exception:
                mode = None
    if mode is None:
        mode = _cfg_get(getattr(brain, "hparams", None), "dist_mode", "merged")
    return _normalize_label_mode(mode)


def _count_params(module: Any) -> tuple[int, int]:
    if module is None:
        return 0, 0
    total = 0
    trainable = 0
    for p in module.parameters():
        n = int(p.numel())
        total += n
        if bool(getattr(p, "requires_grad", False)):
            trainable += n
    return total, trainable


def _count_trainable_all(brain: Any) -> int:
    mods = getattr(brain, "modules", None)
    if mods is None:
        return 0
    try:
        return sum(int(p.numel()) for _, p in mods.named_parameters() if p.requires_grad)
    except Exception:
        return 0


def _format_unfreeze_steps(steps: Any) -> str:
    if not steps:
        return "none"
    parts = []
    for step in steps:
        e = _cfg_get(step, "epoch", None)
        k = _cfg_get(step, "layers", None)
        lr = _cfg_get(step, "lr_factor", 1.0)
        if e is None or k is None:
            continue
        try:
            parts.append(f"e{int(e)}->L{int(k)}@{float(lr):.2f}")
        except Exception:
            continue
    return ",".join(parts) if parts else "none"


def build_run_config(brain: Any, resume_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    hparams = getattr(brain, "hparams", None)
    exp_dir = str(_cfg_get(hparams, "output_folder", _cfg_get(hparams, "save_folder", ".")))

    seed = int(_cfg_get(hparams, "seed", -1))
    device = str(getattr(brain, "device", _cfg_get(getattr(brain, "run_opts", {}), "device", "unknown")))
    precision = _resolve_precision(brain)
    label_mode = _resolve_label_mode(brain)

    cat_loss_cfg = _cfg_get(hparams, "cat_loss", {}) or {}
    cat_loss_type = str(getattr(brain, "cat_loss_type", _cfg_get(cat_loss_cfg, "type", "unknown"))).lower()
    lambda_cat = float(_cfg_get(hparams, "lambda_cat", 0.0))
    lambda_vad = float(_cfg_get(hparams, "lambda_vad", 0.0))
    lambda_ccc = float(getattr(brain, "lambda_ccc", _cfg_get(hparams, "lambda_ccc", 0.0)))
    use_ccc_loss = bool(_cfg_get(hparams, "use_ccc_loss", False))
    vad_out_dim = int(getattr(brain, "vad_out_dim", _cfg_get(hparams, "out_n_neurons", 3)))
    if lambda_vad <= 0.0:
        vad_loss_name = "disabled"
    else:
        vad_loss_name = "gaussian_nll" if vad_out_dim == 6 else "smoothl1"
        if use_ccc_loss and lambda_ccc > 0.0:
            vad_loss_name = f"{vad_loss_name}+ccc"

    ssl_model = _get_module(brain, "ssl_model")
    enc_total, enc_trainable = _count_params(ssl_model)
    if enc_total == 0:
        enc_status = "unknown"
    elif enc_trainable == 0:
        enc_status = "frozen"
    elif enc_trainable == enc_total:
        enc_status = "fully-unfrozen"
    else:
        enc_status = "partially-unfrozen"

    tc_gru_enabled = bool(getattr(brain, "use_tc_gru_head", False))
    tc_gru_cfg = {
        "conv_channels": int(_cfg_get(hparams, "tc_gru_conv_channels", 256)),
        "conv_kernel": int(_cfg_get(hparams, "tc_gru_conv_kernel", 3)),
        "gru_hidden": int(_cfg_get(hparams, "tc_gru_gru_hidden", 256)),
        "gru_layers": int(_cfg_get(hparams, "tc_gru_gru_layers", 2)),
        "emb_dim": int(_cfg_get(hparams, "tc_gru_emb_dim", 256)),
        "dropout": float(_cfg_get(hparams, "tc_gru_dropout", 0.1)),
        "bidirectional": bool(_cfg_get(hparams, "tc_gru_bidirectional", False)),
    } if tc_gru_enabled else {}

    entropy_cfg = _cfg_get(hparams, "entropy_curriculum", {}) or {}
    entropy_enabled = bool(getattr(brain, "entropy_curriculum_enabled", _cfg_get(entropy_cfg, "enabled", False)))
    entropy_mode = str(getattr(brain, "entropy_curriculum_mode", _cfg_get(entropy_cfg, "mode", "none"))).lower()

    unfreeze_steps = _cfg_get(hparams, "unfreeze_steps", []) or []
    resume = {
        "requested": False,
        "active": False,
        "mode": str(_cfg_get(hparams, "mode", "scratch")).lower(),
        "load_mode": None,
        "ckpt_path": None,
        "ckpt_dir": None,
        "restored_epoch": None,
        "restored_step": None,
        "unfreeze_state_restored": bool(
            getattr(getattr(brain, "unfreeze_state", None), "_restored_from_ckpt", False)
        ),
        "curriculum_state_restored": bool(
            getattr(getattr(brain, "curriculum_state", None), "_restored_from_ckpt", False)
        ),
    }
    if resume_meta:
        resume.update(resume_meta)
    if not resume.get("ckpt_dir", None):
        resume["ckpt_dir"] = _resolve_ckpt_dir(resume.get("ckpt_path", None))

    run_cfg = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exp_dir": exp_dir,
        "seed": seed,
        "device": device,
        "precision": precision,
        "label_mode": label_mode,
        "loss": {
            "cat_loss_type": cat_loss_type,
            "lambda_cat": lambda_cat,
            "lambda_vad": lambda_vad,
            "vad_loss": vad_loss_name,
            "use_ccc_loss": use_ccc_loss,
            "lambda_ccc": lambda_ccc,
            "cbce_weights_enabled": bool(getattr(brain, "_cbce_class_weights", None) is not None),
            "weighted_kld_enabled": bool(getattr(brain, "_kld_class_weights", None) is not None),
        },
        "ssl_backbone": {
            "name": type(ssl_model).__name__ if ssl_model is not None else "unknown",
            "encoder_trainable_params": int(enc_trainable),
            "encoder_total_params": int(enc_total),
            "status": enc_status,
        },
        "tc_gru": {
            "enabled": tc_gru_enabled,
            "params": tc_gru_cfg,
        },
        "heads": {
            "cat_mlp": bool(_get_module(brain, "cat_mlp") is not None),
            "vad_mlp": bool(_get_module(brain, "vad_mlp") is not None),
        },
        "curriculum": {
            "entropy_enabled": entropy_enabled,
            "entropy_mode": entropy_mode,
            "label_scheduler_enabled": bool(getattr(brain, "label_scheduler", None) is not None),
        },
        "unfreeze_schedule": {
            "enabled": bool(_cfg_get(hparams, "gradual_unfreeze", False)),
            "freeze_head_epochs": int(_cfg_get(hparams, "freeze_head_epochs", 0)),
            "summary": _format_unfreeze_steps(unfreeze_steps),
        },
        "resume": resume,
        "trainable_params_total": int(_count_trainable_all(brain)),
    }
    return run_cfg


def write_run_config(exp_dir: str, run_cfg: Dict[str, Any]) -> str:
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2, sort_keys=True)
    return path


def emit_run_header(brain: Any, run_cfg: Dict[str, Any], run_config_path: Optional[str] = None) -> None:
    loss = run_cfg.get("loss", {})
    ssl = run_cfg.get("ssl_backbone", {})
    tcg = run_cfg.get("tc_gru", {})
    heads = run_cfg.get("heads", {})
    curr = run_cfg.get("curriculum", {})
    unfr = run_cfg.get("unfreeze_schedule", {})
    resume = run_cfg.get("resume", {})

    tc_desc = "disabled"
    if bool(tcg.get("enabled", False)):
        p = tcg.get("params", {}) or {}
        tc_desc = (
            f"enabled(conv={p.get('conv_channels')} k={p.get('conv_kernel')} "
            f"gru_h={p.get('gru_hidden')} L={p.get('gru_layers')} "
            f"emb={p.get('emb_dim')} bidir={p.get('bidirectional')})"
        )

    lines = [
        "[RUN-HEADER] ----------------------------------------------------------------",
        (
            f"[RUN-HEADER] exp_dir={run_cfg.get('exp_dir')} | seed={run_cfg.get('seed')} "
            f"| device={run_cfg.get('device')} | precision={run_cfg.get('precision')}"
        ),
        f"[RUN-HEADER] label_mode={run_cfg.get('label_mode')}",
        (
            f"[RUN-HEADER] loss: cat={loss.get('cat_loss_type')} (lambda_cat={loss.get('lambda_cat')}) "
            f"| vad={loss.get('vad_loss')} (lambda_vad={loss.get('lambda_vad')}, "
            f"use_ccc={loss.get('use_ccc_loss')}, lambda_ccc={loss.get('lambda_ccc')}) "
            f"| cat_weights(cbce={loss.get('cbce_weights_enabled')}, kld={loss.get('weighted_kld_enabled')})"
        ),
        (
            f"[RUN-HEADER] ssl: backbone={ssl.get('name')} status={ssl.get('status')} "
            f"encoder_trainable={ssl.get('encoder_trainable_params')}/{ssl.get('encoder_total_params')}"
        ),
        f"[RUN-HEADER] tc_gru={tc_desc}",
        f"[RUN-HEADER] heads: cat_mlp={heads.get('cat_mlp')} vad_mlp={heads.get('vad_mlp')}",
        (
            f"[RUN-HEADER] curriculum: entropy_enabled={curr.get('entropy_enabled')} "
            f"mode={curr.get('entropy_mode')} label_scheduler={curr.get('label_scheduler_enabled')}"
        ),
        (
            f"[RUN-HEADER] unfreeze_schedule: enabled={unfr.get('enabled')} "
            f"freeze_head_epochs={unfr.get('freeze_head_epochs')} steps={unfr.get('summary')}"
        ),
        (
            f"[RUN-HEADER] resume={resume.get('active')} requested={resume.get('requested')} "
            f"mode={resume.get('mode')} load_mode={resume.get('load_mode')} "
            f"ckpt_dir={resume.get('ckpt_dir')} restored_epoch={resume.get('restored_epoch')} "
            f"restored_step={resume.get('restored_step')}"
        ),
        (
            f"[RUN-HEADER] restore_state: unfreeze_state={resume.get('unfreeze_state_restored')} "
            f"curriculum_state={resume.get('curriculum_state_restored')}"
        ),
    ]
    if run_config_path:
        lines.append(f"[RUN-HEADER] run_config_json={run_config_path}")
    lines.append("[RUN-HEADER] ----------------------------------------------------------------")

    for line in lines:
        _emit_debug(brain, line)


def _resolve_encoder_layers(brain: Any):
    ssl = _get_module(brain, "ssl_model")
    if ssl is None:
        return []
    if hasattr(ssl, "model") and hasattr(ssl.model, "encoder") and hasattr(ssl.model.encoder, "layers"):
        return list(ssl.model.encoder.layers)
    if hasattr(ssl, "encoder") and hasattr(ssl.encoder, "layers"):
        return list(ssl.encoder.layers)
    return []


def _format_layer_ids(ids: List[int]) -> str:
    if not ids:
        return "none"
    spans = []
    start = ids[0]
    prev = ids[0]
    for idx in ids[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        spans.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = idx
    spans.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(spans)


def maybe_log_unfreeze_event(brain: Any, epoch: Optional[int] = None, reason: str = "") -> None:
    layers = _resolve_encoder_layers(brain)
    trainable_layers = []
    for i, layer in enumerate(layers):
        try:
            if any(bool(p.requires_grad) for p in layer.parameters()):
                trainable_layers.append(i)
        except Exception:
            continue

    trainable_params = int(_count_trainable_all(brain))
    phase = str(getattr(brain, "current_phase", "unknown"))
    state_key = (tuple(trainable_layers), trainable_params, phase)
    if state_key == getattr(brain, "_debug_last_unfreeze_event", None):
        return
    brain._debug_last_unfreeze_event = state_key

    ep = "NA" if epoch is None else str(int(epoch))
    layer_str = _format_layer_ids(trainable_layers)
    why = reason if reason else "state_change"
    _emit_debug(
        brain,
        (
            f"[STATE][UNFREEZE] epoch={ep} phase={phase} reason={why} "
            f"trainable_layers={layer_str} trainable_params={trainable_params}"
        ),
    )


def maybe_log_curriculum_event(brain: Any, epoch: Optional[int] = None) -> None:
    enabled = bool(getattr(brain, "entropy_curriculum_enabled", False))
    if enabled and hasattr(brain, "_entropy_curriculum_policy"):
        mode, thr, alpha, min_w = brain._entropy_curriculum_policy(epoch=epoch)
    else:
        mode, thr, alpha, min_w = ("none", None, None, None)

    if not enabled or str(mode).lower() == "none":
        phase = "disabled"
        sampler_mode = "none"
    elif str(mode).lower() == "filter":
        phase = "filter"
        sampler_mode = "entropy_filter"
    elif str(mode).lower() == "filter_rev":
        phase = "filter_rev"
        sampler_mode = "entropy_filter_rev"
    elif str(mode).lower() == "weight":
        phase = "weight"
        sampler_mode = "entropy_weight"
    elif str(mode).lower() == "weight_rev":
        phase = "weight_rev"
        sampler_mode = "entropy_weight_rev"
    else:
        phase = str(mode).lower()
        sampler_mode = str(mode).lower()

    key = (
        phase,
        sampler_mode,
        None if thr is None else round(float(thr), 6),
        None if alpha is None else round(float(alpha), 6),
        None if min_w is None else round(float(min_w), 6),
    )
    if key == getattr(brain, "_debug_last_curriculum_event", None):
        return
    brain._debug_last_curriculum_event = key

    ep = "NA" if epoch is None else str(int(epoch))
    thr_txt = "n/a" if thr is None else f"{float(thr):.4f}"
    _emit_debug(
        brain,
        (
            f"[STATE][CURRICULUM] epoch={ep} phase={phase} "
            f"entropy_thr={thr_txt} sampler_mode={sampler_mode}"
        ),
    )


def maybe_log_loss_schedule_event(brain: Any, epoch: Optional[int] = None) -> None:
    hparams = getattr(brain, "hparams", None)
    cat_loss = str(getattr(brain, "cat_loss_type", _cfg_get(_cfg_get(hparams, "cat_loss", {}), "type", "unknown"))).lower()
    lambda_cat = float(_cfg_get(hparams, "lambda_cat", 0.0))
    lambda_vad = float(_cfg_get(hparams, "lambda_vad", 0.0))
    lambda_ccc = float(getattr(brain, "lambda_ccc", _cfg_get(hparams, "lambda_ccc", 0.0)))

    key = (cat_loss, round(lambda_cat, 6), round(lambda_vad, 6), round(lambda_ccc, 6))
    if key == getattr(brain, "_debug_last_loss_schedule_event", None):
        return
    brain._debug_last_loss_schedule_event = key

    ep = "NA" if epoch is None else str(int(epoch))
    _emit_debug(
        brain,
        (
            f"[STATE][LOSS-SCHED] epoch={ep} cat_loss={cat_loss} "
            f"lambda_cat={lambda_cat:.6f} lambda_vad={lambda_vad:.6f} lambda_ccc={lambda_ccc:.6f}"
        ),
    )
