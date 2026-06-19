import os
from typing import Optional, Any


def init_debug_logger(brain: Any) -> None:
    """Initialize a line-buffered debug log file handle on the brain."""
    output_folder = getattr(brain.hparams, "output_folder", None)
    if output_folder is None:
        raise AttributeError("brain.hparams.output_folder is required for debug logging")

    os.makedirs(output_folder, exist_ok=True)
    debug_path = os.path.join(output_folder, "debug_log.txt")
    brain._debug_fh = open(debug_path, "a", buffering=1)


def debug(brain: Any, msg: str) -> None:
    """Write a debug message to the debug file if available, else print."""
    fh = getattr(brain, "_debug_fh", None)
    if fh is not None:
        fh.write(msg + "\n")
    else:
        print(f"[DEBUG] {msg}", flush=True)


def log(brain: Any, msg: str, tag: str = "INFO") -> None:
    """Tag + forward a message to debug()."""
    debug(brain, f"[{tag}] {msg}")


def print_param_stats(brain: Any, tag: str = "") -> None:
    """Print total/trainable parameter counts for brain.modules."""
    total = 0
    trainable = 0
    for _, p in brain.modules.named_parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n

    log(
        brain,
        f"[PARAMS-{tag}] total={total/1e6:.2f}M, trainable={trainable/1e6:.2f}M",
        tag="PARAMS",
    )


def print_stage_summary(
    brain: Any,
    stage: Any,
    epoch: Optional[int],
    stage_loss: float,
    cat_metrics_out: Optional[dict],
    extra_cls_metrics: Optional[dict],
) -> None:
    """Compact console summary aligned with ambiguity-aware training."""

    # ---- Encoder trainability (for gradual unfreezing sanity) ----
    enc_params = list(brain.modules.ssl_model.parameters())
    enc_total = len(enc_params)
    enc_train = sum(p.requires_grad for p in enc_params)
    enc_pct = 100.0 * enc_train / max(enc_total, 1)

    # ---- Augmentation phase (scheduler) ----
    aug_phase = getattr(getattr(brain, "aug_scheduler", None), "current_phase", "NA")

    # ---- Label mode (hard/primary/merged/etc.) ----
    label_mode = getattr(brain, "current_label_mode", None)
    if label_mode is None:
        label_mode = str(getattr(brain.hparams, "dist_mode", "merged"))

    def _mget(d: Optional[dict], k: str, default=None):
        return default if d is None else d.get(k, default)

    # ---- Core cls metrics ----
    # UA may come as `ua` (preferred) or `CAT_UA` depending on caller.
    ua = _mget(cat_metrics_out, "ua", _mget(cat_metrics_out, "CAT_UA", 0.0))

    # Macro-F1 is expected in `extra_cls_metrics["macro_f1"]`, but some callers
    # may pass it via `cat_metrics_out` as `macro_f1` or `MACRO_F1`.
    macro_f1 = _mget(
        extra_cls_metrics,
        "macro_f1",
        _mget(cat_metrics_out, "macro_f1", _mget(cat_metrics_out, "MACRO_F1", 0.0)),
    )

    # ---- Ambiguity / confidence summaries ----
    pred_ent = _mget(extra_cls_metrics, "PRED_ENT", _mget(cat_metrics_out, "PRED_ENT", None))
    pred_maxp = _mget(extra_cls_metrics, "PRED_MAXP", _mget(cat_metrics_out, "PRED_MAXP", None))
    pred_margin = _mget(extra_cls_metrics, "PRED_MARGIN", _mget(cat_metrics_out, "PRED_MARGIN", None))

    label_ent = _mget(extra_cls_metrics, "LABEL_ENT", _mget(cat_metrics_out, "LABEL_ENT", None))
    label_maxp = _mget(extra_cls_metrics, "LABEL_MAXP", _mget(cat_metrics_out, "LABEL_MAXP", None))
    label_margin = _mget(extra_cls_metrics, "LABEL_MARGIN", _mget(cat_metrics_out, "LABEL_MARGIN", None))

    # Accept either camelCase or snake_case keys
    top1_acc = _mget(extra_cls_metrics, "top1_acc", _mget(cat_metrics_out, "top1_acc", None))
    top2_acc = _mget(extra_cls_metrics, "top2_acc", _mget(cat_metrics_out, "top2_acc", None))
    top3_acc = _mget(extra_cls_metrics, "top3_acc", _mget(cat_metrics_out, "top3_acc", None))

    ece = _mget(extra_cls_metrics, "CAT_ECE", _mget(cat_metrics_out, "CAT_ECE", None))

    # Entropy-stratified categorical metrics (VALID/TEST)
    ent_f1_low = _mget(extra_cls_metrics, "macro_f1_bin_low", _mget(extra_cls_metrics, "macro_f1_ent_low", None))
    ent_f1_mid = _mget(extra_cls_metrics, "macro_f1_bin_mid", _mget(extra_cls_metrics, "macro_f1_ent_mid", None))
    ent_f1_high = _mget(extra_cls_metrics, "macro_f1_bin_high", _mget(extra_cls_metrics, "macro_f1_ent_high", None))
    n_ent_low = _mget(extra_cls_metrics, "n_bin_low", _mget(extra_cls_metrics, "n_ent_low", None))
    n_ent_mid = _mget(extra_cls_metrics, "n_bin_mid", _mget(extra_cls_metrics, "n_ent_mid", None))
    n_ent_high = _mget(extra_cls_metrics, "n_bin_high", _mget(extra_cls_metrics, "n_ent_high", None))

    keep_frac = getattr(brain, "_entropy_keep_frac", None)
    ent_avg = getattr(brain, "_entropy_avg", None)

    def _fmt3(x):
        return "NA" if x is None else f"{float(x):.3f}"

    print("\n" + "=" * 30)
    print(f"STAGE: {stage.name} | Epoch: {epoch}")
    print("=" * 30)

    print("\nLoss:")
    print(f"  loss          : {stage_loss:.4f}")

    print("\nClassification:")
    print(f"  UA            : {float(ua):.3f}")
    print(f"  Macro-F1      : {float(macro_f1):.3f}")

    print("\nTop-K Accuracy:")
    print(f"  Top-1 Acc     : {_fmt3(top1_acc)}")
    print(f"  Top-2 Acc     : {_fmt3(top2_acc)}")
    print(f"  Top-3 Acc     : {_fmt3(top3_acc)}")

    if (
        ent_f1_low is not None
        or ent_f1_mid is not None
        or ent_f1_high is not None
        or n_ent_low is not None
        or n_ent_mid is not None
        or n_ent_high is not None
    ):
        print("\nEntropy-Stratified F1:")
        print(f"  F1 Low/Mid/High : {_fmt3(ent_f1_low)} / {_fmt3(ent_f1_mid)} / {_fmt3(ent_f1_high)}")
        def _fmt_n(x):
            if x is None:
                return "NA"
            try:
                return str(int(x))
            except Exception:
                return "NA"
        print(f"  N  Low/Mid/High : {_fmt_n(n_ent_low)} / {_fmt_n(n_ent_mid)} / {_fmt_n(n_ent_high)}")

    print("\nAmbiguity / Confidence:")
    if pred_ent is not None:
        print(f"  Pred Entropy  : {float(pred_ent):.3f}")
    if pred_maxp is not None:
        print(f"  Pred MaxP     : {float(pred_maxp):.3f}")
    if pred_margin is not None:
        print(f"  Pred Margin   : {float(pred_margin):.3f}")

    if label_ent is not None:
        print(f"  Label Entropy : {float(label_ent):.3f}")
    if label_maxp is not None:
        print(f"  Label MaxP    : {float(label_maxp):.3f}")
    if label_margin is not None:
        print(f"  Label Margin  : {float(label_margin):.3f}")

    if keep_frac is not None or ent_avg is not None:
        kf = "NA" if keep_frac is None else f"{float(keep_frac):.3f}"
        ea = "NA" if ent_avg is None else f"{float(ent_avg):.3f}"
        print(f"  Curriculum    : keep_frac={kf} | avg_amb={ea}")

    print("\nCalibration:")
    if ece is not None:
        print(f"  ECE           : {float(ece):.3f}")

    print("\nModel State:")
    print(f"  Encoder train : {enc_pct:.1f} %")
    print(f"  Augmentation  : {aug_phase}")
    print(f"  Label mode    : {label_mode}")

    print("-" * 30, flush=True)
