import os, random, math, sys, re, json 
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
import torch
import soundfile as sf
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import torch.optim as _optim
from speechbrain.utils.checkpoints import Checkpointer
from sklearn.metrics import f1_score, recall_score, precision_score

from utils.augment import WaveformAugmenter
from utils.soft_label_utils import apply_mixup, apply_cutmix
from utils.soft_label_utils import one_hot
from utils.accuracy import AccuracyStats
from utils.freeze_utils import apply_unfreeze_schedule, _save_unfreeze_state, _load_unfreeze_state 
from utils.model_utils import concordance_cc, _to_float_tensor, _to_long_tensor, save_model_info, export_test_predictions
from utils.unfreeze_state import UnfreezeState
from utils.attention_pooling import MultiHeadSelfAttentionPooling
from utils.metric_utils import ccc, rmse, cat_metrics 
from utils.metric_utils import (
    compute_cls_extra_metrics,
    confusion_matrix_from_logp,
    ece_from_bin_sums,
    topk_accuracy,
    distribution_stats,
    ece_bin_sums_from_logp,
)
# from utils.subsampler import BalancedSubsetPerEpochSampler

from utils.augmentation_scheduler import AugmentationScheduler
from utils.model_utils import TemporalCNN
from validate_config import validate_hparams
from dataio_msp_podcast import dataio_prep

from utils.utils import set_global_seed, _hp_get
from utils.model_utils import get_ssl_model, WhisperFeatureExtractor, TCGRUHead
from utils.unfreeze_controller import (
    _reapply_unfreeze_from_state,
    _resolve_encoder_layers,
    _maybe_update_unfreeze,
) 

from utils.logging_utils import init_debug_logger, log, debug, print_stage_summary, print_param_stats
from utils.run_logging import (
    build_run_config,
    emit_run_header,
    maybe_log_curriculum_event,
    maybe_log_loss_schedule_event,
    write_run_config,
)

# Minimal persisted scaffold for entropy curriculum recovery.
# This is intentionally no-op when curriculum is disabled.
class CurriculumState:
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
    """Save CurriculumState as a plain dict for SpeechBrain checkpoints."""
    torch.save(obj.state_dict(), path)


def _load_curriculum_state(obj, path, end_of_epoch):
    """Restore CurriculumState from SpeechBrain checkpoint payload."""
    state = torch.load(path, map_location="cpu")
    obj.load_state_dict(state)


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

# =============================================================
# Brain class for Speech Emotion Recognition
# =============================================================
class SerBrain(sb.Brain):
    def _init_tc_gru_head(self):
        """Initialize paper-style TC-GRU head. If enabled, we bypass attention/mean pooling."""
        self.use_tc_gru_head = bool(getattr(self.hparams, "use_tc_gru_head", False))
        if not self.use_tc_gru_head:
            self.tc_gru_head = None
            return

        in_dim = int(getattr(self.hparams, "ssl_hidden_dim", 0) or 0)
        if in_dim <= 0:
            # Fallback: infer from cat_mlp first Linear if possible
            cat_mlp = self.modules["cat_mlp"] if "cat_mlp" in self.modules else None
            if isinstance(cat_mlp, nn.Sequential):
                for layer in cat_mlp:
                    if isinstance(layer, nn.Linear):
                        in_dim = int(layer.in_features)
                        break

        if in_dim <= 0:
            raise RuntimeError("[TC-GRU] Could not infer ssl feature dim. Set hparams.ssl_hidden_dim.")

        conv_channels = int(getattr(self.hparams, "tc_gru_conv_channels", 256))
        conv_kernel   = int(getattr(self.hparams, "tc_gru_conv_kernel", 3))
        gru_hidden    = int(getattr(self.hparams, "tc_gru_gru_hidden", 256))
        gru_layers    = int(getattr(self.hparams, "tc_gru_gru_layers", 2))
        emb_dim       = int(getattr(self.hparams, "tc_gru_emb_dim", 256))
        dropout       = float(getattr(self.hparams, "tc_gru_dropout", 0.1))
        bidir         = bool(getattr(self.hparams, "tc_gru_bidirectional", False))

        self.tc_gru_head = TCGRUHead(
            in_dim=in_dim,
            conv_channels=conv_channels,
            conv_kernel=conv_kernel,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            emb_dim=emb_dim,
            dropout=dropout,
            bidirectional=bidir,
        ).to(self.device)

        self.modules["tc_gru_head"] = self.tc_gru_head
        self._log(
            f"[TC-GRU] Enabled: in_dim={in_dim} conv={conv_channels} k={conv_kernel} "
            f"gru_h={gru_hidden} L={gru_layers} emb={emb_dim} bidir={bidir} drop={dropout}",
            tag="HEAD",
        )
        
    def _init_debug_logger(self):
        """Backwards-compatible wrapper around utils.logging_utils.init_debug_logger."""
        return init_debug_logger(self)
    
    def _debug(self, msg: str):
        """Backwards-compatible wrapper around utils.logging_utils.debug."""
        return debug(self, msg)
    
    def _log(self, msg: str, tag: str = "INFO"):
        """Backwards-compatible wrapper around utils.logging_utils.log."""
        return log(self, msg, tag=tag)
    
    def _print_stage_summary(self, stage, epoch, stage_loss, cat_metrics_out, extra_cls_metrics):
        """Backwards-compatible wrapper around utils.logging_utils.print_stage_summary."""
        return print_stage_summary(
            self,
            stage=stage,
            epoch=epoch,
            stage_loss=stage_loss,
            cat_metrics_out=cat_metrics_out,
            extra_cls_metrics=extra_cls_metrics,
        )
    
    def print_param_stats(self, tag=""):
        """Backwards-compatible wrapper around utils.logging_utils.print_param_stats."""
        return print_param_stats(self, tag=tag)

    def _append_train_log_fallback(self, text: str):
        """Best-effort append to output_folder/train_log.txt when logger APIs fail."""
        try:
            out_dir = str(getattr(self.hparams, "output_folder", "") or "")
            if not out_dir:
                return
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "train_log.txt")
            with open(path, "a", encoding="utf-8") as f:
                f.write(str(text))
        except Exception:
            pass


    def _drop_optimizer_recoverables(self):
        """Remove ANY optimizer objects from the checkpointer, no matter the key."""
        if getattr(self, "checkpointer", None) and hasattr(self.checkpointer, "recoverables"):
            for k, obj in list(self.checkpointer.recoverables.items()):
                if isinstance(obj, _optim.Optimizer):
                    self.checkpointer.recoverables.pop(k, None)

    def _get_param_group_indices(self):
        """Return (ssl_group_idx, head_group_idx) for the single optimizer."""
        ssl_idx = getattr(self, "_ssl_group_idx", None)
        head_idx = getattr(self, "_head_group_idx", None)
        if ssl_idx is None or head_idx is None:
            # fallback: group 0=ssl, group 1=heads (as built in init_optimizers)
            ssl_idx, head_idx = 0, 1
        return int(ssl_idx), int(head_idx)

    def _set_group_lr(self, group_idx: int, new_lr: float):
        """Set LR for a specific param group in the single optimizer."""
        if getattr(self, "optimizer", None) is None:
            return
        if group_idx < 0 or group_idx >= len(self.optimizer.param_groups):
            return
        self.optimizer.param_groups[group_idx]["lr"] = float(new_lr)

    def _scale_group_lr(self, group_idx: int, factor: float):
        """Multiply LR for a specific param group in the single optimizer."""
        if getattr(self, "optimizer", None) is None:
            return
        if group_idx < 0 or group_idx >= len(self.optimizer.param_groups):
            return
        self.optimizer.param_groups[group_idx]["lr"] *= float(factor)

    def _train_update_cm_topk(self, clf_logp: torch.Tensor, y_idx: torch.Tensor, ks=(1, 2, 3)):
        """Streaming TRAIN metrics to avoid storing full logits on CPU.

        Updates:
        - self._train_cm: [C,C] confusion matrix on CPU (int64)
        - self._train_topk_correct{k}: correct counts
        - self._train_topk_total: total samples
        """
        with torch.no_grad():
            y_true = y_idx
            if hasattr(y_true, "data"):
                y_true = y_true.data
            if y_true.ndim > 1:
                y_true = y_true.squeeze(-1)
            y_true = y_true.long().view(-1)

            y_pred = torch.argmax(clf_logp, dim=-1).long().view(-1)

            # Move only small vectors to CPU (cheap). DO NOT move [B,C].
            y_true_cpu = y_true.detach().to("cpu", non_blocking=True)
            y_pred_cpu = y_pred.detach().to("cpu", non_blocking=True)

            C = int(self.num_classes)
            idx = (y_true_cpu * C + y_pred_cpu).clamp_min(0)
            binc = torch.bincount(idx, minlength=C * C)
            self._train_cm += binc.view(C, C).to(self._train_cm.dtype)

            # Top-k counts computed on-device, then reduced
            max_k = int(max(ks))
            topk_idx = torch.topk(clf_logp, k=max_k, dim=-1).indices  # [B,max_k]
            y_true_dev = y_true.to(topk_idx.device)
            correct_mat = topk_idx.eq(y_true_dev.unsqueeze(-1))       # [B,max_k]

            for k in ks:
                k = int(k)
                corr = correct_mat[:, :k].any(dim=-1).sum().item()
                self._train_topk_correct[k] += int(corr)

            self._train_topk_total += int(y_true.numel())

    def on_fit_start(self, model=None):
        """Initializes the brain, including the SSL model and optimizers."""
        seed = getattr(self.hparams, "seed", 1415)
        cudnn_det = getattr(self.hparams, "cudnn_deterministic", False)
        set_global_seed(seed, cudnn_det)
        self._init_debug_logger()
        # Force-create train_log.txt early so logging never silently fails
        if hasattr(self.hparams, "train_logger"):
            try:
                self.hparams.train_logger.write(
                    f"--- Experiment started (seed={seed}) ---\n"
                )
            except Exception:
                pass
        self._debug(f"Run started | seed={seed}")
        self._debug(f"[PATHS]  output_folder={self.hparams.output_folder} save_folder={self.hparams.save_folder}")

        # --- helpers: hparams blocks can be dict-like or AttrDict-like ---
        def _cfg_get(cfg, key, default=None):
            if cfg is None:
                return default
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        # ---------------------------------------
        # Build Augmentation + Scheduler
        # ---------------------------------------
        # Base augmentation config from YAML
        self.base_aug_cfg = self.hparams.augmentation

        # Scheduler config from YAML
        sched_cfg = getattr(self.hparams, "augmentation_scheduler", {})

        # Initialize scheduler
        self.aug_scheduler = AugmentationScheduler(
            scheduler_cfg=sched_cfg,
            base_aug_cfg=self.base_aug_cfg
        )

        # Initialize augmenter from full hparams so it can resolve top-level
        # paths like `rir_folder` (not present inside augmentation sub-dict).
        self.augment = WaveformAugmenter(self.hparams, device=self.device)

        # Log initial augmentation summary (pre-scheduler)
        any_aug = any(
            (self.base_aug_cfg.get(k, {}).get("p", 0) or 0) > 0
            for k in self.base_aug_cfg
        )
        # Keep augmentation summary out of SpeechBrain train_logger (too verbose on console).
        self._debug(
            f"[AUG-INIT]  any={any_aug} "
            f"noise_p={self.base_aug_cfg.get('noise', {}).get('p', 0)} "
            f"time_mask_p={self.base_aug_cfg.get('time_mask', {}).get('p', 0)} "
            f"rir_p={self.base_aug_cfg.get('rir', {}).get('p', 0)} "
            f"params={self.base_aug_cfg}"
        )

        # --- SSL model init ---
        existing_ssl = self.modules["ssl_model"] if "ssl_model" in self.modules else None
        if existing_ssl is None:
            self.ssl_model = get_ssl_model(self.hparams)
        else:
            self.ssl_model = existing_ssl

        if self.ssl_model is None:
            raise ValueError("Failed to initialize self.ssl_model. get_ssl_model(self.hparams) returned None.")

        self.ssl_model = self.ssl_model.to(self.device)
        self.modules["ssl_model"] = self.ssl_model

        self._log(f"Initialized SSL model: {type(self.ssl_model).__name__}")
        self.modules.update({"ssl_model": self.ssl_model})

        # --- optional paper-style TC-GRU head (replaces pooling) ---
        self._init_tc_gru_head()

        # --- attention pooling over time (used only when TC-GRU head is disabled) ---
        self._init_attention_pool()

        # --- optional hybrid CNN head ---
        self.use_hybrid_head = bool(getattr(self.hparams, "use_hybrid_head", False))
        if self.use_hybrid_head:
            cnn_channels = getattr(self.hparams, "cnn_channels", [256, 256])
            cnn_kernel   = int(getattr(self.hparams, "cnn_kernel", 5))
            cnn_dropout  = float(getattr(self.hparams, "cnn_dropout", 0.1))

            self.temporal_cnn = TemporalCNN(
                feat_dim=int(getattr(self.hparams, "ssl_hidden_dim", 768)),
                channels=cnn_channels,
                kernel_size=cnn_kernel,
                dropout=cnn_dropout,
            ).to(self.device)

            self.modules["temporal_cnn"] = self.temporal_cnn
            self._log(
                f"[HYBRID] TemporalCNN enabled: channels={cnn_channels}, "
                f"kernel={cnn_kernel}, dropout={cnn_dropout}"
            )
        else:
            self.temporal_cnn = None

        # ---- Save model architecture for inspection (AFTER all heads exist) ----
        # NOTE: save_model_info now prints optional heads (tc_gru_head / attn_pool / temporal_cnn),
        # so do NOT append here (avoids stale/overwritten dumps).
        try:
            save_model_info(self, self.hparams.save_folder)
        except Exception as e:
            self._log(f"[WARN] save_model_info failed: {e}")

        # --- initial freeze: encoder off, heads on ---
        for p in self.modules["ssl_model"].parameters():
            p.requires_grad = False
        for k in ("vad_mlp", "cat_mlp", "attn_pool", "temporal_cnn", "tc_gru_head"):
            if k in self.modules:
                for p in self.modules[k].parameters():
                    p.requires_grad = True

        # --- classification meta (single source of truth; must exist before recoverables & checkpoint recovery) ---
        self.label_encoder = _hp_get(self.hparams, "label_encoder", None)
        if self.label_encoder is None:
            raise RuntimeError("label_encoder is None. dataio_prep must set hparams['label_encoder'].")

        self.num_classes = int(self.label_encoder.expected_len)
        self.class_names = [self.label_encoder.ind2lab[i] for i in range(self.num_classes)]

        # --- unfreeze runtime state (recoverable) ---
        if not hasattr(self, "unfreeze_state"):
            self.unfreeze_state = UnfreezeState()

        # --- entropy curriculum runtime state (recoverable scaffold) ---
        if not hasattr(self, "curriculum_state"):
            ecfg = _hp_get(self.hparams, "entropy_curriculum", {}) or {}
            ecfg_enabled = bool(ecfg.get("enabled", False)) if isinstance(ecfg, dict) else bool(getattr(ecfg, "enabled", False))
            ecfg_mode = str(ecfg.get("mode", "none")).lower() if isinstance(ecfg, dict) else str(getattr(ecfg, "mode", "none")).lower()
            self.curriculum_state = CurriculumState(
                enabled=ecfg_enabled,
                mode=ecfg_mode,
                phase=("active" if ecfg_enabled else "disabled"),
            )

        # mirror onto convenience attrs
        self.unfrozen_count = self.unfreeze_state.unfrozen_count
        self.current_phase  = self.unfreeze_state.phase

        # If resuming with gradual unfreeze, re-apply requires_grad flags
        if getattr(self.hparams, "gradual_unfreeze", False) and self.unfreeze_state.unfrozen_count > 0:
            try:
                _reapply_unfreeze_from_state(self)
                self._log(
                    f"[GradualUnfreeze] Re-applied state: "
                    f"{self.unfreeze_state.unfrozen_count} encoder layer(s) unfrozen "
                    f"(phase='{self.unfreeze_state.phase}')."
                )
            except Exception as e:
                self._log(f"[GradualUnfreeze][WARN] Failed to re-apply unfreeze state: {e}")

        # --- register recoverables ---
        if getattr(self, "checkpointer", None) is not None:
            # heads
            for k in ("vad_mlp", "cat_mlp", "tc_gru_head", "attn_pool", "temporal_cnn"):
                if k in self.modules:
                    self.checkpointer.add_recoverable(k, self.modules[k])

            # encoder
            self.checkpointer.add_recoverable("ssl_model", self.modules["ssl_model"])
            self.checkpointer.add_recoverable("label_encoder", self.label_encoder)
            model_obj = _hp_get(self.hparams, "model", None)
            if model_obj is not None:
                self.checkpointer.add_recoverable("model", model_obj)
                self.checkpointer.optional_recoverables["model"] = True
            counter_obj = _hp_get(self.hparams, "epoch_counter", None)
            if counter_obj is not None:
                self.checkpointer.add_recoverable("counter", counter_obj)
                self.checkpointer.optional_recoverables["counter"] = True
            self.checkpointer.add_recoverable("brain", self)
            self.checkpointer.optional_recoverables["brain"] = True

            # Single-optimizer schedulers (heads vs SSL group)
            for name in ("lr_annealing", "lr_annealing_ssl", "lr_annealing_wavlm"):
                sched = getattr(self.hparams, name, None)
                if sched is not None:
                    self.checkpointer.add_recoverable(name, sched)
                    self.checkpointer.optional_recoverables[name] = True

            self.checkpointer.custom_save_hooks["unfreeze_state"] = _save_unfreeze_state
            self.checkpointer.custom_load_hooks["unfreeze_state"] = _load_unfreeze_state
            self.checkpointer.custom_save_hooks["curriculum_state"] = _save_curriculum_state
            self.checkpointer.custom_load_hooks["curriculum_state"] = _load_curriculum_state

            self.checkpointer.add_recoverable("unfreeze_state", self.unfreeze_state)
            self.checkpointer.add_recoverable("curriculum_state", self.curriculum_state)
            self.checkpointer.optional_recoverables["curriculum_state"] = True

        # Build optimizers AFTER requires_grad is set (encoder frozen at start)
        self.init_optimizers()

        # Keep optimizer state by default so future full resumes are possible.
        # Drop it only when explicitly requested via `resume: model_only` (or alias).
        pending_cfg = getattr(self, "_pending_resume", None) or {}
        pending_mode = str(pending_cfg.get("mode", getattr(self.hparams, "mode", "scratch"))).lower()
        pending_load_mode = str(pending_cfg.get("load_mode", getattr(self.hparams, "load_mode", "backbone+heads"))).lower()
        resume_policy = str(getattr(self.hparams, "resume", "strict")).lower()
        weights_only = resume_policy in {"model_only", "weights_only", "no_optimizer", "none"}
        keep_for_full_resume = (pending_mode == "resume" and pending_load_mode == "all")
        if weights_only and not keep_for_full_resume:
            self._drop_optimizer_recoverables()
            self._log("[RESUME] Optimizer recoverables disabled by policy (weights-only checkpoints).")

        # ---------------------------------------------------------
        # Deferred resume/fine-tune loading (after SSL exists)
        # ---------------------------------------------------------
        pending = getattr(self, "_pending_resume", None)
        mode_for_resume = str(getattr(self.hparams, "mode", "scratch")).lower()
        self._run_resume_meta = {
            "requested": bool(pending) or mode_for_resume in {"resume", "ft"},
            "active": False,
            "mode": mode_for_resume,
            "load_mode": None,
            "ckpt_path": None,
            "ckpt_dir": None,
        }
        manual_resume_loaded = False
        if pending:
            ckpt_path = pending["ckpt_path"]
            load_mode = pending.get("load_mode", "backbone+heads")
            mode = pending.get("mode", "scratch")
            reset_head = pending.get("reset_head", False)
            reset_clf = pending.get("reset_clf", False)
            reset_reg = pending.get("reset_reg", False)
            freeze_pat = pending.get("freeze_pat", "")
            self._run_resume_meta.update(
                {
                    "requested": True,
                    "mode": str(mode),
                    "load_mode": str(load_mode),
                    "ckpt_path": str(ckpt_path),
                    "ckpt_dir": (
                        os.path.dirname(str(ckpt_path))
                        if str(ckpt_path).endswith((".yaml", ".ckpt"))
                        else str(ckpt_path)
                    ),
                }
            )

            ssl_sched = getattr(self.hparams, "lr_annealing_ssl", None)
            if ssl_sched is None:
                ssl_sched = getattr(self.hparams, "lr_annealing_wavlm", None)  # legacy fallback

            load_info = load_from_ckpt(
                ckpt_path=ckpt_path,
                device=self.device,
                ssl_model=self.modules["ssl_model"] if "ssl_model" in self.modules else None,
                cat_mlp=self.modules["cat_mlp"] if "cat_mlp" in self.modules else None,
                vad_mlp=self.modules["vad_mlp"] if "vad_mlp" in self.modules else None,
                tc_gru_head=self.modules["tc_gru_head"] if "tc_gru_head" in self.modules else None,
                model=_hp_get(self.hparams, "model", None),
                brain=self,
                label_encoder=getattr(self, "label_encoder", None),
                optimizer=self.optimizer if load_mode == "all" else None,
                lr_annealing=getattr(self.hparams, "lr_annealing", None) if load_mode == "all" else None,
                lr_annealing_ssl=ssl_sched if load_mode == "all" else None,
                epoch_counter=getattr(self.hparams, "epoch_counter", None) if load_mode == "all" else None,
                unfreeze_state=getattr(self, "unfreeze_state", None) if load_mode == "all" else None,
                curriculum_state=getattr(self, "curriculum_state", None) if load_mode == "all" else None,
                dataloader_train=None,
                mode=load_mode,
            )
            load_info = dict(load_info or {})
            manual_resume_loaded = bool(load_info.get("loaded", False))
            self._run_resume_meta["active"] = bool(manual_resume_loaded)
            if load_info.get("loaded_ckpt_dir"):
                self._run_resume_meta["loaded_ckpt_dir"] = str(load_info.get("loaded_ckpt_dir"))
            missing_recoverables = list(load_info.get("missing_recoverables", []) or [])
            if missing_recoverables:
                self._run_resume_meta["missing_recoverables"] = list(missing_recoverables)
                self._log(
                    "[RESUME][WARN] Partial state restore; missing recoverables: "
                    + ", ".join(missing_recoverables)
                )
            if load_info.get("optimizer_state_restored", None) is False:
                self._log(
                    "[RESUME][WARN] Optimizer state was not restored. "
                    "Expect transient loss spike and LR/scheduler reset effects."
                )

            # EpochCounter must reflect last completed epoch so fit() continues
            # from the next one instead of restarting at epoch 1.
            if load_mode == "all":
                counter_obj = getattr(self.hparams, "epoch_counter", None)
                meta_epoch = load_info.get("meta_epoch", None)
                if counter_obj is not None and meta_epoch is not None and hasattr(counter_obj, "current"):
                    try:
                        cur = int(getattr(counter_obj, "current"))
                    except Exception:
                        cur = None
                    try:
                        meta_epoch = int(meta_epoch)
                    except Exception:
                        meta_epoch = None
                    if meta_epoch is not None and (cur is None or cur < meta_epoch):
                        try:
                            counter_obj.current = int(meta_epoch)
                            manual_resume_loaded = True
                            self._run_resume_meta["active"] = True
                            self._log(
                                f"[RESUME] Counter corrected from {cur} to {meta_epoch} using CKPT.yaml metadata."
                            )
                        except Exception as e:
                            self._log(f"[RESUME][WARN] Failed to set epoch_counter.current from metadata: {e}")

                # Don't silently continue a "resume" run from scratch.
                if str(mode).lower() == "resume" and not manual_resume_loaded:
                    raise RuntimeError(
                        f"[RESUME] Could not recover training state from '{ckpt_path}'. "
                        "Pass a valid CKPT+... directory, CKPT.yaml, or checkpoints root."
                    )
                if str(mode).lower() == "resume" and ("counter" in set(missing_recoverables)):
                    raise RuntimeError(
                        f"[RESUME] Checkpoint '{ckpt_path}' is missing counter.ckpt, "
                        "so epoch continuity cannot be guaranteed."
                    )

            # Keep runtime flags synchronized with restored unfreeze state
            # before any epoch-level unfreeze decisions.
            if load_mode == "all" and getattr(self, "unfreeze_state", None) is not None:
                self.unfrozen_count = int(getattr(self.unfreeze_state, "unfrozen_count", 0))
                self.current_phase = str(getattr(self.unfreeze_state, "phase", "heads_only"))
                if bool(getattr(self.hparams, "gradual_unfreeze", False)):
                    try:
                        _reapply_unfreeze_from_state(self)
                    except Exception as e:
                        self._log(f"[GradualUnfreeze][WARN] Post-resume re-apply failed: {e}")

            def _maybe_reset_module(mod):
                if mod is None:
                    return
                if hasattr(mod, "reset_parameters"):
                    try:
                        mod.reset_parameters()
                        return
                    except Exception:
                        pass
                if isinstance(mod, nn.Sequential):
                    for layer in mod:
                        if hasattr(layer, "reset_parameters"):
                            try:
                                layer.reset_parameters()
                            except Exception:
                                pass

            # Fine-tune optional head resets AFTER restore
            if mode == "ft":
                if reset_head:
                    _maybe_reset_module(self.modules["cat_mlp"] if "cat_mlp" in self.modules else None)
                    _maybe_reset_module(self.modules["vad_mlp"] if "vad_mlp" in self.modules else None)
                else:
                    if reset_clf:
                        _maybe_reset_module(self.modules["cat_mlp"] if "cat_mlp" in self.modules else None)
                    if reset_reg:
                        _maybe_reset_module(self.modules["vad_mlp"] if "vad_mlp" in self.modules else None)

            # Optional freeze regex AFTER restore
            if freeze_pat:
                try:
                    rx = re.compile(freeze_pat)
                except Exception as e:
                    raise ValueError(f"Invalid freeze regex: {freeze_pat} ({e})")
                for name, p in self.modules.named_parameters():
                    if rx.search(name):
                        p.requires_grad = False

            self._log(f"[RESUME] Loaded from {ckpt_path} with mode={mode} load_mode={load_mode}")
            self._pending_resume = None

        self._grad_accum = int(getattr(self.hparams, "grad_accum_steps", 1))
        # Preserve recovered value if brain.ckpt restored it.
        self._step_counter = int(getattr(self, "_step_counter", 0))

        # ---- TRAIN metrics throttling (perf) ----
        # Updating confusion/top-k every batch can serialize and kill throughput.
        # Default: update every 50 TRAIN batches unless overridden in hparams.
        self._train_metrics_every = int(getattr(self.hparams, "train_metrics_every", 100))
        if self._train_metrics_every < 1:
            self._train_metrics_every = 1
        self._train_metrics_batch_i = 0

        try:
            v = getattr(self.hparams, "ckpt_interval_minutes", None)
            if v is not None and float(v) <= 0:
                self.hparams.ckpt_interval_minutes = 1e9
                self._log("[CKPT] Disabled core checkpointing; using VALID save_and_keep_only only.", tag="CKPT")
        except Exception:
            pass

        # If manual resume already loaded a specific checkpoint, avoid a second
        # implicit recover_if_possible() inside SpeechBrain on_fit_start.
        _orig_recover_if_possible = None
        if manual_resume_loaded and getattr(self, "checkpointer", None) is not None:
            _orig_recover_if_possible = getattr(self.checkpointer, "recover_if_possible", None)
            if callable(_orig_recover_if_possible):
                def _skip_recover(*args, **kwargs):
                    return None
                self.checkpointer.recover_if_possible = _skip_recover
        try:
            super().on_fit_start()
        finally:
            if callable(_orig_recover_if_possible):
                self.checkpointer.recover_if_possible = _orig_recover_if_possible

        self.print_param_stats(tag="start")

        # =====================================================
        # Phase-3 OPTIONAL Head-Only Fine-Tune State
        # Controls patience for head-only fine-tuning before switching phases.
        # =====================================================
        # Do not clobber CBCE weights built above
        if not hasattr(self, "_cbce_class_weights"):
            self._cbce_class_weights = None
        self._phase3_active = False
        self._phase3_epochs_left = 0
        self._phase3_triggered = False
        self._phase3_patience = 3  # Number of epochs to wait for improvement in phase-3 head-only fine-tuning
        self._best_cat_ua = -1.0
        self._cat_ua_plateau = 0
        self._log("[Phase3] Initialized head-only fine-tuning controller.")

        # =====================================================
        # Early stopping (VALID-stage metric) + checkpoint selection
        # NOTE: SpeechBrain does NOT early-stop by default. We implement it here.
        # =====================================================
        es_cfg = _hp_get(self.hparams, "early_stopping", None)
        es_enabled = False
        es_patience = 0
        es_min_delta = 0.0
        es_metric = "macro_f1"   # default
        es_mode = "max"          # macro-F1/UA/ACC are maximized

        try:
            if es_cfg is not None:
                # es_cfg may be dict-like or AttrDict-like
                es_enabled = bool(getattr(es_cfg, "enabled", es_cfg.get("enabled", False)))
                es_patience = int(getattr(es_cfg, "patience", es_cfg.get("patience", 0)))
                es_min_delta = float(getattr(es_cfg, "min_delta", es_cfg.get("min_delta", 0.0)))
                es_metric = str(getattr(es_cfg, "metric", es_cfg.get("metric", "macro_f1"))).lower()
                es_mode = str(getattr(es_cfg, "mode", es_cfg.get("mode", "max"))).lower()
        except Exception:
            # safest fallback
            es_enabled = False

        self._es_enabled = bool(es_enabled and es_patience > 0)
        self._es_patience = int(es_patience)
        self._es_min_delta = float(es_min_delta)
        self._es_metric = str(es_metric)
        self._es_mode = str(es_mode)
        self._es_best = -float("inf") if self._es_mode == "max" else float("inf")
        self._es_bad_epochs = 0

        if self._es_enabled:
            self._log(
                f"[EarlyStopping] enabled=True metric={self._es_metric} mode={self._es_mode} "
                f"patience={self._es_patience} min_delta={self._es_min_delta}",
                tag="ES",
            )
        else:
            self._log("[EarlyStopping] enabled=False", tag="ES")

        # =====================================================
        # Checkpoint selection policy (VALID-stage metric)
        # =====================================================
        ckpt_cfg = _hp_get(self.hparams, "checkpoint_selection", None)
        try:
            ckpt_metric = _normalize_metric_name(
                _cfg_get(ckpt_cfg, "metric", getattr(self.hparams, "checkpoint_metric", "macro_f1"))
            )
            ckpt_mode = str(_cfg_get(ckpt_cfg, "mode", "max")).lower()
            ckpt_num_keep = int(_cfg_get(ckpt_cfg, "save_top_k", getattr(self.hparams, "ckpt_num_to_keep", 3)))
            ckpt_save_last = bool(_cfg_get(ckpt_cfg, "save_last", getattr(self.hparams, "ckpt_save_latest", True)))
            ckpt_min_delta = float(_cfg_get(ckpt_cfg, "min_delta", getattr(self.hparams, "ckpt_min_delta", 0.0)))
        except Exception:
            ckpt_metric = _normalize_metric_name(getattr(self.hparams, "checkpoint_metric", "macro_f1"))
            ckpt_mode = "max"
            ckpt_num_keep = int(getattr(self.hparams, "ckpt_num_to_keep", 3))
            ckpt_save_last = bool(getattr(self.hparams, "ckpt_save_latest", True))
            ckpt_min_delta = float(getattr(self.hparams, "ckpt_min_delta", 0.0))

        if ckpt_mode not in {"min", "max"}:
            ckpt_mode = "max"

        self._ckpt_metric = str(ckpt_metric)
        self._ckpt_mode = str(ckpt_mode)
        self._ckpt_num_keep = max(1, int(ckpt_num_keep))
        self._ckpt_save_latest = bool(ckpt_save_last)
        self._ckpt_min_delta = float(ckpt_min_delta)
        self._ckpt_meta_key = _checkpoint_meta_key(self._ckpt_metric)
        self._log(
            f"[CheckpointSelection] metric={self._ckpt_metric} meta_key={self._ckpt_meta_key} "
            f"mode={self._ckpt_mode} save_top_k={self._ckpt_num_keep} "
            f"save_last={self._ckpt_save_latest} min_delta={self._ckpt_min_delta}",
            tag="CKPT",
        )

        # -----------------------------
        # Best-checkpoint tracking (for VALID-only selection)
        # -----------------------------
        self._best_ckpt_score = None
        self._best_ckpt_epoch = None
        self._best_ckpt_path  = None

        # --- regression (VAD) meta ---
        # Two supported heads:
        #   - mean-only:  vad_mlp outputs [B,3]
        #   - mean+uncertainty: vad_mlp outputs [B,6] = [mu(3), logvar(3)]
        self.vad_out_dim = int(getattr(self.hparams, "out_n_neurons", 3))
        if self.vad_out_dim not in (3, 6):
            raise RuntimeError(f"out_n_neurons must be 3 (mu) or 6 (mu+logvar), got {self.vad_out_dim}")

        self.vad_weights = torch.tensor(
            getattr(self.hparams, "vad_dim_weights", [1.2, 1.0, 1.0]),
            device=self.device, dtype=torch.float32,
        )

        # Pre-build loss modules used every batch (avoid re-allocations)
        self._vad_huber = nn.SmoothL1Loss(
            beta=float(getattr(self.hparams, "huber_beta", 0.2)),
            reduction="none",
        )

        # log-variance clamp for stability in heteroscedastic NLL
        self._logvar_min = float(getattr(self.hparams, "logvar_clip_min", -10.0))
        self._logvar_max = float(getattr(self.hparams, "logvar_clip_max",  10.0))


        # ---- categorical loss selection (paper comparability knobs) ----
        # Preferred knob (single source of truth): `cat_loss.type` in YAML
        #   one of: ce | cbce | kld | jsd
        # Back-compat: if `cat_loss` is missing but `loss.type` is set to kld/jsd,
        # we treat that as the intended categorical loss.
        cat_loss_cfg = _hp_get(self.hparams, "cat_loss", None)
        cat_type = str(_cfg_get(cat_loss_cfg, "type", "kld")).lower()

        if cat_type not in ("ce", "cbce", "kld", "jsd"):
            self._log(f"[LOSS][WARN] Unknown cat_loss.type='{cat_type}', falling back to 'ce'.", tag="LOSS")
            cat_type = "ce"

        self.cat_loss_type = cat_type

        # soft targets are required for divergence losses
        self.use_soft_targets = bool(self.cat_loss_type in ("kld", "jsd"))

        # temperature for soft labels (only applied when using emo_vec / soft targets)
        self.soft_temp = float(getattr(self.hparams, "soft_temp", 1.0))

        # --- loss modules (pre-built) ---
        self.ce = nn.CrossEntropyLoss()
        self._ce_none = nn.CrossEntropyLoss(reduction="none")

        # KLDiv expects log-prob input
        self.kld = nn.KLDivLoss(reduction="batchmean")
        self._kld_none = nn.KLDivLoss(reduction="none")


        # --- CBCE (Class-Balanced CE) config ---
        cbce_cfg = _hp_get(self.hparams, "cbce", None)
        self.cbce_beta = float(_cfg_get(cbce_cfg, "beta", 0.9999))
        self.cbce_reduction = str(_cfg_get(cbce_cfg, "reduction", "mean")).lower()

        # Gate CBCE weights: only when CBCE is selected OR explicitly enabled.
        cbce_enabled = (str(getattr(self, "cat_loss_type", "")).lower() == "cbce")
        cbce_enabled = cbce_enabled or bool(_cfg_get(cbce_cfg, "enabled", False))

        # Build weights aligned to label_encoder order.
        self._cbce_class_weights = None
        if cbce_enabled:
            raw = _cfg_get(cbce_cfg, "class_counts", None)
            try:
                if raw is None:
                    raise ValueError("cbce.class_counts is missing")
                if not isinstance(raw, dict):
                    raise TypeError(f"cbce.class_counts must be a dict {{name: count}}, got {type(raw)}")

                counts_list = [int(raw.get(name, 0)) for name in self.class_names]
                counts_t = torch.tensor([max(c, 0) for c in counts_list], dtype=torch.float32, device=self.device)
                beta = float(self.cbce_beta)

                eff_num = 1.0 - torch.pow(torch.tensor(beta, device=self.device), counts_t)
                w = (1.0 - beta) / torch.clamp(eff_num, min=1e-8)
                w = w / torch.clamp(w.mean(), min=1e-8)

                self._cbce_class_weights = w
                used = ", ".join([f"{n}:{c}" for n, c in zip(self.class_names, counts_list)])
                self._log(f"[LOSS] CBCE weights built (label_encoder order): {used}", tag="LOSS")
            except Exception as e:
                self._log(f"[LOSS][WARN] Failed to build CBCE class weights; CBCE will behave like CE: {e}", tag="LOSS")
                self._cbce_class_weights = None

        # Optional weighting for CCC term in VAD loss
        self.lambda_ccc = float(getattr(self.hparams, "lambda_ccc", 1.0))

        self._log(
            f"[LOSS] cat_loss.type={self.cat_loss_type} (use_soft_targets={self.use_soft_targets}) | "
            f"soft_temp={self.soft_temp} | cbce_beta={self.cbce_beta}",
            tag="LOSS",
        )

        # --- optional Weighted-KLD (class-weighted KL for soft targets) ---
        # Applies to ANY soft-label mode (primary / merged / secondary) because it operates on the
        # final [B,C] target distribution returned by _get_cls_targets().
        kldw_cfg = _hp_get(self.hparams, "kld_weighting", None)

        def _as_dict(cfg):
            if cfg is None:
                return None
            if isinstance(cfg, dict):
                return cfg
            # AttrDict-like
            try:
                return dict(cfg)
            except Exception:
                return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}

        self._kld_class_weights = None  # tensor[C] on device
        try:
            kldw = _as_dict(kldw_cfg) or {}
            kldw_enabled = bool(kldw.get("enabled", False))
            kldw_mode = str(kldw.get("mode", "class_counts")).lower()
            kldw_power = float(kldw.get("power", 1.0))
            kldw_norm = str(kldw.get("normalize", "mean")).lower()
            kldw_min = float(kldw.get("min_w", 0.25))
            kldw_max = float(kldw.get("max_w", 5.0))

            if kldw_enabled:
                C = int(self.num_classes)
                names = list(self.class_names)  # label_encoder order

                if kldw_mode == "weights":
                    raw_w = kldw.get("weights", None)
                    if raw_w is None:
                        raise ValueError("kld_weighting.mode='weights' but kld_weighting.weights is missing")
                    if len(raw_w) != C:
                        raise ValueError(f"kld_weighting.weights length {len(raw_w)} != num_classes {C}")
                    w = torch.tensor([float(x) for x in raw_w], device=self.device, dtype=torch.float32)

                else:
                    # default: derive weights from class_counts (inverse freq^power)
                    raw_counts = kldw.get("class_counts", None)
                    if raw_counts is None or not isinstance(raw_counts, dict):
                        raise ValueError("kld_weighting.mode='class_counts' requires dict kld_weighting.class_counts")
                    counts_list = [float(raw_counts.get(n, 0)) for n in names]
                    counts_t = torch.tensor([max(1.0, c) for c in counts_list], device=self.device, dtype=torch.float32)
                    w = torch.pow(1.0 / counts_t, kldw_power)

                # normalize weights so they don't explode the loss scale
                if kldw_norm == "mean":
                    w = w / torch.clamp(w.mean(), min=1e-8)
                elif kldw_norm == "sum":
                    w = w / torch.clamp(w.sum(), min=1e-8) * float(C)

                # clamp
                w = torch.clamp(w, min=kldw_min, max=kldw_max)
                self._kld_class_weights = w

                # log a compact summary
                w_cpu = w.detach().cpu().numpy()
                self._log(
                    f"[LOSS] Weighted-KLD enabled | mode={kldw_mode} power={kldw_power} norm={kldw_norm} "
                    f"clamp=[{kldw_min},{kldw_max}] | w(min/mean/max)="
                    f"{w_cpu.min():.3f}/{w_cpu.mean():.3f}/{w_cpu.max():.3f}",
                    tag="LOSS",
                )
        except Exception as e:
            # If config is missing/bad, proceed with vanilla KLD
            self._kld_class_weights = None
            if kldw_cfg is not None:
                self._log(f"[LOSS][WARN] Failed to build Weighted-KLD class weights; using vanilla KLD: {e}", tag="LOSS")
        self.mixup_enabled = bool(getattr(self.hparams, "mixup_enabled", False))
        self.mixup_alpha   = float(getattr(self.hparams, "mixup_alpha", 0.4))
        self.mixup_p       = float(getattr(self.hparams, "mixup_p", 0.0))
        self.mixup_per_smp = bool(getattr(self.hparams, "mixup_per_smp", True))

        self.cutmix_enabled = bool(getattr(self.hparams, "cutmix_enabled", False))
        self.cutmix_alpha   = float(getattr(self.hparams, "cutmix_alpha", 0.4))
        self.cutmix_p       = float(getattr(self.hparams, "cutmix_p", 0.0))
        self.cutmix_per_smp = bool(getattr(self.hparams, "cutmix_per_smp", True))

        # Experiments currently exclude MixUp/CutMix; keep toggles off by default unless explicitly re-enabled.
        if not bool(getattr(self.hparams, "enable_mix_aug", False)):
            self.mixup_enabled = False
            self.cutmix_enabled = False

        self.frame_w = float(getattr(self.hparams, "frame_loss_weight", 0.0))
        self._cat_soft_targets = None  # holds [B,C] for current TRAIN batch if mixup/cutmix produced soft labels
        # self._init_augmenter()  

        self.current_label_mode = str(_hp_get(self.hparams, "dist_mode", "merged")).lower()

        # ---- classification loss mode ----
        # NOTE: use_soft_targets is derived from cat_loss.type (single source of truth).
        if self.use_soft_targets:
            loss_name = f"{self.cat_loss_type.upper()} (soft targets)"
        else:
            loss_name = f"{self.cat_loss_type.upper()} (hard targets)"
        self._log(f"[LOSS] Classification loss = {loss_name}")

        # ---- entropy curriculum (sample filter/weighting) ----
        # Uses batch-provided: emo_entropy_norm / emo_maxprob / emo_margin (from dataio_prep).
        # Modes:
        #   1) "filter": keep low-entropy samples (H <= threshold)
        #   2) "filter_rev": keep high-entropy samples (H >= threshold)
        #   3) "weight": weight low-entropy samples higher via (1 - H)^alpha
        #   4) "weight_rev": weight high-entropy samples higher via H^alpha
        self.entropy_curriculum = getattr(self.hparams, "entropy_curriculum", None)
        if self.entropy_curriculum is None:
            self.entropy_curriculum = {}
        self.entropy_curriculum_enabled = bool(self.entropy_curriculum.get("enabled", False))
        self.entropy_curriculum_mode = str(self.entropy_curriculum.get("mode", "none")).lower()

        # bookkeeping (logged per TRAIN epoch)
        self._entropy_keep_frac = None
        self._entropy_avg = None
        self._entropy_kept = 0
        self._entropy_seen = 0
        self._entropy_weight_values = []
        self._entropy_weight_min_count = 0
        self._entropy_weight_total_count = 0
        self._entropy_active_quantile = None
        self._entropy_cutoff_sum = 0.0
        self._entropy_cutoff_n = 0
        self._entropy_kept_amb_sum = 0.0
        self._entropy_kept_amb_n = 0
        self._entropy_kept_amb_min = None
        self._entropy_kept_amb_max = None
        self._entropy_kept_class_counts = None
        # Optional per-batch weight curriculum debug (very verbose).
        # 0 disables batch-level logging; epoch-level CURR summaries stay enabled.
        self._entropy_weight_batch_debug_every = int(
            getattr(self.hparams, "entropy_weight_batch_debug_every", 0)
        )
        self._entropy_weight_batch_debug_counter = 0

        if self.entropy_curriculum_enabled:
            self._log(
                f"[EntropyCurriculum] enabled=True mode={self.entropy_curriculum_mode} key={getattr(self.hparams, 'ambiguity_signal_key', 'emo_entropy_norm')} cfg={self.entropy_curriculum}",
            )
        else:
            self._log("[EntropyCurriculum] enabled=False", tag="CURR")

        # ---------------------------------------------------------
        # IMPORTANT: disable SpeechBrain core periodic/end-of-epoch
        # checkpointing (it will otherwise write CKPT every epoch
        # when ckpt_interval_minutes<=0).
        # We do our own VALID-only model selection via save_and_keep_only.
        # ---------------------------------------------------------
        try:
            v = getattr(self.hparams, "ckpt_interval_minutes", None)
            if v is not None:
                v = float(v)
                if v <= 0:
                    setattr(self.hparams, "ckpt_interval_minutes", 1e9)
                    self._log(
                        "[CKPT] Disabled core checkpointing (ckpt_interval_minutes<=0). Using VALID save_and_keep_only only.",
                        tag="CKPT",
                    )
        except Exception:
            pass

        # ---- One-time run header + machine-readable run config ----
        if not bool(getattr(self, "_run_header_logged", False)):
            restored_epoch = None
            counter = getattr(self.hparams, "epoch_counter", None)
            if counter is not None and hasattr(counter, "current"):
                try:
                    restored_epoch = int(getattr(counter, "current"))
                except Exception:
                    restored_epoch = None

            resume_meta = dict(getattr(self, "_run_resume_meta", {}) or {})
            resume_meta.setdefault("requested", False)
            resume_meta.setdefault("active", bool(manual_resume_loaded))
            resume_meta["restored_epoch"] = restored_epoch
            resume_meta["restored_step"] = int(getattr(self, "_step_counter", 0))
            unfreeze_restored = bool(
                getattr(getattr(self, "unfreeze_state", None), "_restored_from_ckpt", False)
            )
            curriculum_restored = bool(
                getattr(getattr(self, "curriculum_state", None), "_restored_from_ckpt", False)
            )
            resume_meta["unfreeze_state_restored"] = unfreeze_restored
            resume_meta["curriculum_state_restored"] = curriculum_restored
            if not bool(resume_meta.get("active", False)) and (unfreeze_restored or curriculum_restored):
                resume_meta["active"] = True

            run_cfg = build_run_config(self, resume_meta=resume_meta)
            run_cfg_path = None
            try:
                run_cfg_path = write_run_config(run_cfg["exp_dir"], run_cfg)
            except Exception as e:
                self._debug(f"[RUN-HEADER][WARN] Failed to write run_config.json: {e}")

            emit_run_header(self, run_cfg, run_config_path=run_cfg_path)
            self._run_header_logged = True


    def _init_attention_pool(self):
        """
        Initialize multi-head attention pooling.

        Uses:
        - hparams.ssl_hidden_dim if available
        - otherwise infers input_dim from first Linear in vad_mlp

        Controlled by flat hparams:
        - use_attn_pooling   (bool, default: True)
        - attn_num_heads     (int,  default: 4)
        - attn_hidden_dim    (int,  default: None -> input_dim)
        - attn_dropout       (float, default: 0.1)
        - attn_temperature   (float, default: 1.0)
        """
        use_attn = bool(getattr(self.hparams, "use_attn_pooling", True))
        if not use_attn:
            self._log("[ATTN] Attention pooling disabled; using simple mean pooling.")
            self.attn_pool = None
            return

        # Prefer explicit ssl_hidden_dim
        in_dim = int(getattr(self.hparams, "ssl_hidden_dim", 0) or 0)

        # Fallback: infer from vad_mlp
        if in_dim <= 0:
            vad_mlp = self.modules["vad_mlp"] if "vad_mlp" in self.modules else None
            if isinstance(vad_mlp, nn.Sequential):
                for layer in vad_mlp:
                    if isinstance(layer, nn.Linear):
                        in_dim = layer.in_features
                        break

        if in_dim <= 0:
            self._log("[ATTN][WARN] Could not infer ssl feature dim; falling back to mean pooling.")
            self.attn_pool = None
            return

        num_heads   = int(getattr(self.hparams, "attn_num_heads", 4))
        hidden_dim  = getattr(self.hparams, "attn_hidden_dim", None)
        if hidden_dim is not None:
            hidden_dim = int(hidden_dim)
        dropout     = float(getattr(self.hparams, "attn_dropout", 0.1))
        temperature = float(getattr(self.hparams, "attn_temperature", 1.0))

        self.attn_pool = MultiHeadSelfAttentionPooling(
            input_dim=in_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=temperature,
        ).to(self.device)

        # register for checkpointing
        self.modules["attn_pool"] = self.attn_pool
        self._log(
            f"[ATTN] Init multi-head attention pooling: "
            f"in_dim={in_dim}, heads={num_heads}, hidden_dim={hidden_dim}, "
            f"dropout={dropout}, tau={temperature:.3f}"
        )

    def _init_augmenter(self):
        """Initialize waveform augmentation pipeline."""
        aug_cfg = getattr(self.hparams, "augmentation", None)
        if not aug_cfg:
            self._log("[AUG] No augmentation config found; skipping.")
            self.augment = None
            return

        try:
            from utils.augment import WaveformAugmenter
            self.augment = WaveformAugmenter(
                self.hparams,
                device=self.device,
                dtype=torch.float32,
            )
            self._log("[AUG] WaveformAugmenter initialized successfully.")
        except Exception as e:
            self._log(f"[AUG][WARN] Failed to initialize augmentation: {e}")
            self.augment = None

    def _build_targets(self, y_idx: torch.Tensor) -> torch.Tensor:
        """Return [B,C] one-hot targets (synthetic target smoothing removed)."""
        return one_hot(y_idx, self.num_classes)
        
    def _build_cutmix_targets(self, y, y_perm, lam_eff):
        ta = self._build_targets(y); tb = self._build_targets(y_perm)
        return lam_eff.view(-1,1) * ta + (1.0 - lam_eff.view(-1,1)) * tb

    @staticmethod
    def _normalize_distribution(dist: torch.Tensor) -> torch.Tensor:
        """Clamp + renormalize categorical distributions."""
        dist = torch.clamp(dist, min=1e-8)
        return dist / dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _get_entropy_bin_edges(self):
        """Return 4 monotonic edges for low/mid/high entropy bins."""
        default = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]
        raw = getattr(self.hparams, "entropy_metric_bin_edges", default)
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

    def _extract_eval_label_entropy_norm(self, batch):
        """Per-utterance label entropy in [0,1] for VALID/TEST stratified metrics.

        Priority:
          1) merged/fixed entropy provided by dataio (`emo_entropy_norm_merged`)
          2) existing batch field (`emo_entropy_norm`)
          3) fallback from label distribution (`emo_vec`)
        """
        for key in ("emo_entropy_norm_merged", "emo_entropy_norm"):
            val = getattr(batch, key, None)
            if val is not None:
                t = getattr(val, "data", val)
                t = t.to(self.device, dtype=torch.float32).view(-1)
                return torch.clamp(t, 0.0, 1.0)

        emo_vec = getattr(batch, "emo_vec", None)
        if emo_vec is not None:
            vec = getattr(emo_vec, "data", emo_vec).to(self.device, dtype=torch.float32)
            if vec.ndim == 1:
                vec = vec.unsqueeze(0)
            vec = self._normalize_distribution(vec)
            ent = -(vec * torch.log(vec.clamp_min(1e-8))).sum(dim=-1)
            C = int(vec.shape[-1]) if vec.ndim >= 2 else int(getattr(self, "num_classes", 2))
            ent = ent / math.log(max(C, 2))
            return torch.clamp(ent, 0.0, 1.0).view(-1)
        return None

    def _get_cls_targets(self, batch, stage, y_idx):
        """
        Returns the soft categorical targets to supervise the emotion head.

        Priority order:
        1. Augmentation-created mixtures (MixUp / CutMix) stored in _cat_soft_targets
        2. Configured `dist_mode`:
            - "hard"      -> one-hot from y_idx (no emo_vec)
            - otherwise   -> dataset-provided emo_vec distributions
        3. Fallback synthetic one-hot targets from _build_targets().
        """
        # 1) If MixUp / CutMix produced soft labels for this TRAIN batch, use them.
        if stage == sb.Stage.TRAIN and getattr(self, "_cat_soft_targets", None) is not None:
            return self._normalize_distribution(self._cat_soft_targets.to(self.device))

        # Determine current label mode from the configured dist_mode.
        label_mode = getattr(self, "current_label_mode", None)
        if label_mode is None:
            label_mode = getattr(self.hparams, "dist_mode", "merged")
        label_mode = str(label_mode).lower()

        # 2) Hard mode: ignore emo_vec and use strict one-hot from y_idx.
        if label_mode == "hard":
            hard_vec = one_hot(y_idx, self.num_classes).to(self.device)
            return self._normalize_distribution(hard_vec)

        # 3) Soft modes: rely on emo_vec (primary / secondary / merged selected by dist_pipeline)
        emo_vec = getattr(batch, "emo_vec", None)
        if emo_vec is not None:
            vec = getattr(emo_vec, "data", emo_vec)
            if vec is not None:
                vec = vec.to(self.device, dtype=torch.float32)
                bad_rows = (~torch.isfinite(vec)).any() or (vec.sum(dim=-1) <= 0).any()
                if bad_rows:
                    self._log("[WARN] emo_vec contains NaN/Inf/zero rows — falling back to hard labels.")
                    hard_vec = one_hot(y_idx, self.num_classes).to(self.device)
                    return self._normalize_distribution(hard_vec)

                vec = self._normalize_distribution(vec)
                temp = float(getattr(self, "soft_temp", 1.0))
                if temp != 1.0:
                    logits = torch.log(vec.clamp_min(1e-8))
                    vec = torch.softmax(logits / temp, dim=-1)
                return vec

        # 4) Fallback: synthetic targets from loss config (_build_targets).
        return self._build_targets(y_idx).to(self.device)

    
    def _entropy_curriculum_policy(self, epoch: int):
        """Return (mode, threshold, alpha, min_w).

        Schedules are controlled by hparams.entropy_curriculum:

        Curriculum 1 (filter):
          enabled: true
          mode: filter
          quantile_schedule: [0.5, 0.75, 1.0]   # optional staged quantile inclusion
          schedule_epochs: [0, 4, 8]            # optional stage start epochs
          warmup_epochs: 1
          ramp_epochs: 6
          start_thr: 0.20
          end_thr: 0.95

        Curriculum 2 (filter_rev):
          enabled: true
          mode: filter_rev
          quantile_schedule: [0.5, 0.75, 1.0]
          schedule_epochs: [0, 4, 8]
          warmup_epochs: 1
          ramp_epochs: 6
          start_thr: 0.20
          end_thr: 0.95

        Curriculum 3 (weight):
          enabled: true
          mode: weight
          warmup_epochs: 1
          ramp_epochs: 6
          alpha_start: 2.0
          alpha_end: 0.0
          min_weight: 0.15

        Curriculum 4 (weight_rev):
          enabled: true
          mode: weight_rev
          warmup_epochs: 1
          ramp_epochs: 6
          alpha_start: 2.0
          alpha_end: 0.0
          min_weight: 0.15

        Notes:
          - entropy inputs should be normalized to [0,1] (emo_entropy_norm).
          - warmup_epochs: curriculum is fully disabled for the first N epochs.
        """
        cfg = getattr(self, "entropy_curriculum", {}) or {}
        if not bool(cfg.get("enabled", False)):
            return "none", None, None, None

        mode = str(cfg.get("mode", "none")).lower()
        warmup = max(0, int(cfg.get("warmup_epochs", 0)))
        ramp = max(0, int(cfg.get("ramp_epochs", 0)))

        # Fully disable curriculum during warmup epochs.
        if epoch is not None and int(epoch) <= warmup:
            return "none", None, None, None

        if epoch is None:
            t = 1.0
        elif ramp <= 1:
            t = 1.0
        else:
            # First post-warmup epoch starts exactly at start_thr / alpha_start.
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

    def _summarize_entropy_weight_epoch(self):
        """Return (mean, min, max, p10, p50, p90, frac_at_min_weight) for TRAIN weights."""
        vals = getattr(self, "_entropy_weight_values", None)
        n_min = int(getattr(self, "_entropy_weight_min_count", 0) or 0)
        n_tot = int(getattr(self, "_entropy_weight_total_count", 0) or 0)
        frac_at_min = (float(n_min) / float(n_tot)) if n_tot > 0 else None
        if not vals:
            return None, None, None, None, None, None, frac_at_min

        try:
            w = torch.cat(vals, dim=0).to(dtype=torch.float32).view(-1)
        except Exception:
            return None, None, None, None, None, None, frac_at_min

        if int(w.numel()) == 0:
            return None, None, None, None, None, None, frac_at_min

        try:
            q = torch.quantile(
                w,
                torch.tensor([0.10, 0.50, 0.90], dtype=w.dtype, device=w.device),
            )
            return (
                float(w.mean().item()),
                float(w.min().item()),
                float(w.max().item()),
                float(q[0].item()),
                float(q[1].item()),
                float(q[2].item()),
                frac_at_min,
            )
        except Exception:
            arr = w.detach().cpu().numpy()
            return (
                float(arr.mean()),
                float(arr.min()),
                float(arr.max()),
                float(np.percentile(arr, 10)),
                float(np.percentile(arr, 50)),
                float(np.percentile(arr, 90)),
                frac_at_min,
            )

    def _log_entropy_curriculum_epoch_debug(self, epoch):
        """Emit an epoch-level curriculum diagnostic line."""
        if not bool(getattr(self, "entropy_curriculum_enabled", False)):
            return

        active_mode, active_thr, active_alpha, active_min_w = self._entropy_curriculum_policy(
            epoch=None if epoch is None else int(epoch)
        )
        active_mode = str(active_mode).lower()
        mode = str(getattr(self, "entropy_curriculum_mode", "none")).lower()
        phase = "active" if active_mode in {"filter", "filter_rev", "weight", "weight_rev"} else "warmup"
        entm = getattr(self, "_entropy_avg", None)

        if mode in {"filter", "filter_rev"}:
            frac = getattr(self, "_entropy_keep_frac", None)
            cfg = getattr(self, "entropy_curriculum", {}) or {}
            q_sched = cfg.get("quantile_schedule", None)
            e_sched = cfg.get("schedule_epochs", None)
            use_quantile = (
                isinstance(q_sched, (list, tuple))
                and isinstance(e_sched, (list, tuple))
                and len(q_sched) > 0
                and len(q_sched) == len(e_sched)
            )
            current_q = float(active_thr) if (use_quantile and active_thr is not None) else None
            cutoff_n = int(getattr(self, "_entropy_cutoff_n", 0) or 0)
            cutoff_sum = float(getattr(self, "_entropy_cutoff_sum", 0.0) or 0.0)
            threshold_val = (cutoff_sum / float(cutoff_n)) if cutoff_n > 0 else None
            num_kept = int(getattr(self, "_entropy_kept", 0) or 0)
            kept_amb_n = int(getattr(self, "_entropy_kept_amb_n", 0) or 0)
            kept_amb_sum = float(getattr(self, "_entropy_kept_amb_sum", 0.0) or 0.0)
            kept_amb_mean = (kept_amb_sum / float(kept_amb_n)) if kept_amb_n > 0 else None
            kept_amb_min = getattr(self, "_entropy_kept_amb_min", None)
            kept_amb_max = getattr(self, "_entropy_kept_amb_max", None)

            cc = getattr(self, "_entropy_kept_class_counts", None)
            cc_list = None
            cc_min = None
            cc_med = None
            cc_max = None
            if isinstance(cc, torch.Tensor) and int(cc.numel()) > 0:
                cc_cpu = cc.detach().to("cpu", dtype=torch.long).view(-1)
                cc_list = [int(x) for x in cc_cpu.tolist()]
                if len(cc_list) > 0:
                    cc_min = int(min(cc_list))
                    cc_med = float(np.median(np.asarray(cc_list, dtype=np.float32)))
                    cc_max = int(max(cc_list))

            msg = (
                f"Epoch {epoch} | EntropyCurriculum mode={mode} "
                f"phase={phase} current_q={current_q} "
                f"threshold_value={threshold_val} num_kept={num_kept} "
                f"active_quantile={current_q} entropy_cutoff_value={threshold_val} "
                f"keep_frac={frac} class_counts_kept={cc_list} "
                f"class_counts_kept_min={cc_min} class_counts_kept_median={cc_med} "
                f"class_counts_kept_max={cc_max} "
                f"kept_H_min={kept_amb_min} kept_H_mean={kept_amb_mean} kept_H_max={kept_amb_max} "
                f"ent_mean={entm}"
            )
        elif mode in {"weight", "weight_rev"}:
            mean_w, min_w, max_w, p10_w, p50_w, p90_w, frac_min_w = self._summarize_entropy_weight_epoch()
            msg = (
                f"Epoch {epoch} | EntropyCurriculum mode={mode} "
                f"phase={phase} "
                f"alpha={active_alpha} cfg_min_weight={active_min_w} "
                f"min_weight={min_w} max_weight={max_w} mean_weight={mean_w} "
                f"p10_weight={p10_w} p50_weight={p50_w} p90_weight={p90_w} "
                f"frac_at_min_weight={frac_min_w} ent_mean={entm}"
            )
        else:
            frac = getattr(self, "_entropy_keep_frac", None)
            msg = (
                f"Epoch {epoch} | EntropyCurriculum mode={mode} "
                f"phase={phase} keep_frac={frac} ent_mean={entm}"
            )

        try:
            self._log(msg, tag="CURR")
        except Exception:
            try:
                print(f"[CURR] {msg}", flush=True)
            except Exception:
                pass
    
    
    def _assert_label_payload(self, batch, stage):
        """Crash-fast sanity checks for label tensors.

        Runs once per stage/epoch (first batch) to prevent silent fallback.

        Rules:
          - If label_mode == "hard": require `y_idx` (integer class) and validate/synthesize `emo_encoded` (one-hot).
          - Else (soft/dist): require `emo_dist` or `emo_vec`, sums≈1, correct shape.
        """
        # Resolve label mode the same way as `_get_cls_targets()`.
        label_mode = getattr(self, "current_label_mode", None)
        if label_mode is None:
            label_mode = getattr(self.hparams, "dist_mode", "merged")
        label_mode = str(label_mode).lower()

        C = int(getattr(self, "num_classes", 0) or 0)
        if C <= 0:
            raise RuntimeError("[LABEL-ASSERT] num_classes is not set; cannot validate labels.")

        def _as_tensor(x):
            if x is None:
                return None
            return getattr(x, "data", x)

        if label_mode == "hard":
            # Hard mode MUST have y_idx; emo_encoded is optional (we can synthesize it).
            if not hasattr(batch, "y_idx"):
                raise RuntimeError("[LABEL-ASSERT] label_mode='hard' requires batch.y_idx to exist.")

            y = _as_tensor(getattr(batch, "y_idx", None))
            if y is None:
                raise RuntimeError("[LABEL-ASSERT] batch.y_idx is None.")
            y = y.to(self.device).view(-1).long()

            if not torch.isfinite(y.float()).all():
                raise RuntimeError("[LABEL-ASSERT] y_idx contains NaN/Inf.")
            if (y.min().item() < 0) or (y.max().item() >= C):
                raise RuntimeError(
                    f"[LABEL-ASSERT] y_idx out of range: min={int(y.min().item())}, max={int(y.max().item())}, C={C}"
                )

            # If emo_encoded is provided, validate it strictly.
            # If it is missing (common in TEST/VALID), synthesize one-hot from y_idx.
            enc = None
            if hasattr(batch, "emo_encoded"):
                enc = _as_tensor(getattr(batch, "emo_encoded", None))

            if enc is None:
                enc = one_hot(y, C).to(self.device, dtype=torch.float32)
                # Best-effort: attach for downstream debug (won't break PaddedBatch)
                try:
                    setattr(batch, "emo_encoded", enc)
                except Exception:
                    pass
            else:
                enc = enc.to(self.device)
                if enc.ndim != 2:
                    raise RuntimeError(
                        f"[LABEL-ASSERT] emo_encoded must be rank-2 [B,C], got shape={tuple(enc.shape)}"
                    )
                if int(enc.shape[-1]) != C:
                    raise RuntimeError(
                        f"[LABEL-ASSERT] emo_encoded last dim mismatch: got {int(enc.shape[-1])}, expected {C}"
                    )
                if not torch.isfinite(enc).all():
                    raise RuntimeError("[LABEL-ASSERT] emo_encoded contains NaN/Inf.")

                # One-hot sanity: each row sums to ~1 and max is ~1.
                rs = enc.sum(dim=-1)
                if not torch.allclose(rs, torch.ones_like(rs), atol=1e-3, rtol=1e-3):
                    raise RuntimeError(
                        f"[LABEL-ASSERT] emo_encoded row sums must be 1 (±1e-3). "
                        f"Got min={rs.min().item():.6f}, max={rs.max().item():.6f}"
                    )
                mx = enc.max(dim=-1).values
                if not torch.all(mx > 0.99):
                    raise RuntimeError(
                        f"[LABEL-ASSERT] emo_encoded rows must be one-hot-ish (max>0.99). "
                        f"Got min(max)={mx.min().item():.6f}"
                    )

            return  # hard mode OK

        # Soft / distribution mode: accept either emo_dist or emo_vec as the distribution payload.
        dist = None
        if hasattr(batch, "emo_dist"):
            dist = _as_tensor(getattr(batch, "emo_dist", None))
        if dist is None and hasattr(batch, "emo_vec"):
            dist = _as_tensor(getattr(batch, "emo_vec", None))

        if dist is None:
            raise RuntimeError(
                f"[LABEL-ASSERT] label_mode='{label_mode}' requires batch.emo_dist or batch.emo_vec to exist."
            )

        dist = dist.to(self.device, dtype=torch.float32)
        if dist.ndim != 2:
            raise RuntimeError(f"[LABEL-ASSERT] dist labels must be rank-2 [B,C], got shape={tuple(dist.shape)}")
        if int(dist.shape[-1]) != C:
            raise RuntimeError(f"[LABEL-ASSERT] dist last dim mismatch: got {int(dist.shape[-1])}, expected {C}")
        if not torch.isfinite(dist).all():
            raise RuntimeError("[LABEL-ASSERT] dist labels contain NaN/Inf.")

        # Distribution sanity: non-negative, sums to ~1.
        if (dist < -1e-6).any():
            mn = float(dist.min().detach().cpu().item())
            raise RuntimeError(f"[LABEL-ASSERT] dist labels contain negative values (min={mn}).")

        s = dist.sum(dim=-1)
        if not torch.allclose(s, torch.ones_like(s), atol=1e-2, rtol=1e-2):
            raise RuntimeError(
                f"[LABEL-ASSERT] dist row sums must be 1 (±1e-2). "
                f"Got min={s.min().item():.6f}, max={s.max().item():.6f}"
            )

        return

    def _maybe_assert_labels_first_batch(self, batch, stage):
        """Run label assertions exactly once per stage/epoch."""
        if not bool(getattr(self.hparams, "assert_label_sanity", True)):
            return
        if getattr(self, "_label_assert_done", False):
            return
        self._assert_label_payload(batch, stage)
        self._label_assert_done = True

    def compute_forward(self, batch, stage):
        """Minimal forward: SSL -> optional CNN -> pooling -> heads."""
        verbose = False

        batch = batch.to(self.device)
        wavs, lens = batch.sig

        # First-batch label sanity checks (crash-fast; prevents silent fallback).
        self._maybe_assert_labels_first_batch(batch, stage)

        # ----- Augmentation (train only) -----
        if stage == sb.Stage.TRAIN:
            # Reset per-batch soft targets
            self._cat_soft_targets = None

            # Waveform-level augmentation (noise, RIR, etc.)
            if hasattr(self, "augment") and self.augment is not None:
                wavs = self.augment(wavs, lens=lens, training=True)

            # Build base soft targets from the current label mode (hard / primary / merged).
            # These are what MixUp / CutMix will mix, so they always respect the configured label mode.
            y_idx_raw = _to_long_tensor(batch.y_idx, self.device)

            # NOTE: _get_cls_targets will ignore self._cat_soft_targets when it is None
            # and will return either one-hot (hard) or emo_vec-based soft distributions.
            base_soft_targets = self._get_cls_targets(batch, sb.Stage.TRAIN, y_idx_raw)

            # Determine whether current label mode is hard or soft.
            # Use the same resolution logic as `_get_cls_targets()`:
            #   - `self.current_label_mode` mirrors hparams.dist_mode
            #   - otherwise fall back to hparams.dist_mode (default: merged)
            # If soft labels (primary/merged), disable MixUp and CutMix to avoid over-smoothing.
            label_mode = getattr(self, "current_label_mode", None)
            if label_mode is None:
                label_mode = getattr(self.hparams, "dist_mode", "merged")
            label_mode = str(label_mode).lower()
            # MixUp/CutMix produce soft targets. Keep them only when the loss consumes soft targets.
            loss_uses_soft = str(getattr(self, "cat_loss_type", "ce")).lower() in ("kld", "jsd")
            allow_mix = (label_mode == "hard") and loss_uses_soft
            allow_cutmix = (label_mode == "hard") and loss_uses_soft

            # ---------- MixUp ----------
            if allow_mix and getattr(self, "mixup_enabled", False) and torch.rand(1, device=self.device).item() < self.mixup_p:
                mixed_wavs, mixed_targets, lam = apply_mixup(
                    wavs,
                    y_idx=y_idx_raw,
                    C=self.num_classes,
                    alpha=self.mixup_alpha,
                    per_sample=self.mixup_per_smp,
                    build_targets=None,  # already soft; let MixUp detect that
                    soft_targets=base_soft_targets,   # [B, C] soft distributions
                )
                wavs = mixed_wavs
                # Store soft label distributions for this batch
                self._cat_soft_targets = mixed_targets.to(self.device)
                if verbose:
                    self._log(f"[MIXUP] Applied with alpha={self.mixup_alpha}, λ≈{lam.mean().item():.3f}")

            # ---------- CutMix ----------
            elif allow_cutmix and getattr(self, "cutmix_enabled", False) and torch.rand(1, device=self.device).item() < self.cutmix_p:
                mixed_wavs, mixed_targets, lam_eff = apply_cutmix(
                    wavs,
                    lens=lens,
                    y_idx=y_idx_raw,
                    C=self.num_classes,
                    alpha=self.cutmix_alpha,
                    per_sample=self.cutmix_per_smp,
                    build_targets=None,  # already soft; let CutMix detect that
                    soft_targets=base_soft_targets,   # [B, C] soft distributions
                )
                wavs = mixed_wavs
                self._cat_soft_targets = mixed_targets.to(self.device)
                if verbose:
                    self._log(f"[CUTMIX] Applied with alpha={self.cutmix_alpha}, mean λ_eff={lam_eff.mean().item():.3f}")

        # ----- SSL encoder -----
        # print("SSL MODEL ATTRS:", dir(self.modules["ssl_model"]))
        ssl_out = self.modules["ssl_model"](wavs, lens)

        if verbose: 
            if isinstance(ssl_out, dict):
                self._log(f"SSL OUT KEYS: {list(ssl_out.keys())}")
                if "last_hidden_state" in ssl_out:
                    self._log(f"last_hidden_state shape = {ssl_out['last_hidden_state'].shape}")
                if "hidden_states" in ssl_out:
                    self._log(f"num hidden_states = {len(ssl_out['hidden_states'])}")
                    self._log(f"hidden_states[-1] shape = {ssl_out['hidden_states'][-1].shape}")
            else:
                self._log(f"SSL OUT TYPE: {type(ssl_out)}, shape: {ssl_out.shape}")


        # Handle different SSL wrappers: tensor or dict
        if isinstance(ssl_out, dict):
            # adapt to how your get_ssl_model returns outputs
            # most HF models use 'last_hidden_state'
            framewise = ssl_out.get("last_hidden_state", None)
            if framewise is None:
                raise RuntimeError(f"ssl_model returned dict without 'last_hidden_state' key: {ssl_out.keys()}")
        else:
            framewise = ssl_out  # [B, T, D]

        # ----- Optional Temporal CNN -----
        if self.use_hybrid_head and self.temporal_cnn is not None:
            framewise = self.temporal_cnn(framewise)  # [B, T, D]
            if verbose:
                self._log(f"[HYBRID] After TemporalCNN, framewise shape = {framewise.shape}")

        # ----- Pool / Head over time -----
        if getattr(self, "use_tc_gru_head", False) and getattr(self, "tc_gru_head", None) is not None:
            pooled = self.tc_gru_head(framewise, lens)  # [B, emb]
        elif hasattr(self, "attn_pool") and self.attn_pool is not None:
            pooled = self.attn_pool(framewise, lens)    # [B, D]
        else:
            pooled = framewise.mean(dim=1)              # [B, D]

        # --- ATTENTION POOLING SANITY CHECK ---
        if self.attn_pool is not None:
            if pooled is None:
                raise RuntimeError("ATTN POOLING FAILURE: pooled returned None.")
            if not torch.isfinite(pooled).all():
                raise RuntimeError("ATTN POOLING FAILURE: pooled contains NaN/Inf values.")
            if pooled.dim() != 2:
                raise RuntimeError(f"ATTN POOLING FAILURE: pooled has invalid shape {pooled.shape}, expected [B, D].")
            # Optional debug logging
            if getattr(self.hparams, "debug_attn_pool", False):
                self._log(f"[ATTN-POOL] pooled shape={tuple(pooled.shape)}, min={pooled.min().item():.4f}, max={pooled.max().item():.4f}")

        # --- TC-GRU SANITY CHECK ---
        if getattr(self, "use_tc_gru_head", False) and getattr(self, "tc_gru_head", None) is not None:
            if pooled is None:
                raise RuntimeError("TC-GRU FAILURE: pooled returned None.")
            if not torch.isfinite(pooled).all():
                raise RuntimeError("TC-GRU FAILURE: pooled contains NaN/Inf values.")
            if pooled.dim() != 2:
                raise RuntimeError(f"TC-GRU FAILURE: pooled has invalid shape {pooled.shape}, expected [B, D].")

        # If you insist on using avg_pool from hparams, use this instead:
        # pooled = self.hparams.avg_pool(framewise, lens).view(framewise.size(0), -1)

        # ----- Heads -----
        # VAD head predicts BOTH mean and uncertainty (std) per dimension.
        # Parameterize uncertainty via log-variance for stability.
        # Output convention: [B, 6] = [mu_V, mu_A, mu_D, logvar_V, logvar_A, logvar_D]
        vad_mu = None
        vad_logvar = None
        vad_std = None

        if getattr(self.hparams, "lambda_vad", 0.0) > 0.0 and "vad_mlp" in self.modules:
            out_vad = self.modules["vad_mlp"](pooled)  # [B, 6] (mu+logvar) or legacy [B, 3]
            if out_vad.size(-1) == 6:
                vad_mu, vad_logvar = out_vad[..., :3], out_vad[..., 3:]
                # std is for logging/debug; keep a numerically safe version
                lv_safe = torch.clamp(vad_logvar, min=self._logvar_min, max=self._logvar_max)
                vad_std = torch.exp(0.5 * lv_safe)
            elif out_vad.size(-1) == 3:
                vad_mu = out_vad
            else:
                raise RuntimeError(
                    f"vad_mlp output last dim {out_vad.size(-1)}; expected 3 (mu) or 6 (mu+logvar)"
                )

        logits_cat = self.modules["cat_mlp"](pooled)      # [B, C]
        if logits_cat.dim() != 2 or int(logits_cat.size(-1)) != int(self.num_classes):
            raise RuntimeError(
                f"cat_mlp output shape {tuple(logits_cat.shape)} incompatible with num_classes={self.num_classes}"
            )

        logp_cat = torch.log_softmax(logits_cat, dim=-1)

        # ----- Targets -----
        vad   = batch.vad
        y_idx = batch.y_idx

        vad   = _to_float_tensor(vad,  self.device)
        y_idx = _to_long_tensor(y_idx, self.device)

        preds = {
            **({"vad_mu": vad_mu} if vad_mu is not None else {}),
            **({"vad_logvar": vad_logvar} if vad_logvar is not None else {}),
            **({"vad_std": vad_std} if vad_std is not None else {}),
            "cat_logits": logits_cat,
            "cat_logp": logp_cat,
        }
        tgts = {
            **({"vad": vad} if vad_mu is not None else {}),
            "y_idx": y_idx,
        }
        return preds, tgts

    def compute_objectives(self, predictions, batch, stage):
        """Minimal multi-head loss: VAD + categorical."""
        preds, tgts = predictions
        reg_mu = preds.get("vad_mu", None)
        reg_logvar = preds.get("vad_logvar", None)
        clf_logp   = preds["cat_logp"]     # [B, C]

        vad = tgts.get("vad", None)
        y_idx = tgts["y_idx"]              # [B]

        # --- Minimal target normalization (avoid repeated work in hot loop) ---
        if hasattr(y_idx, "data"):  # PaddedData
            y_idx = y_idx.data
        y_idx = y_idx.view(-1).long()

        # ========= VAD loss =========
        # Supports:
        #  - mean-only regression: SmoothL1 on mu
        #  - heteroscedastic regression: Gaussian NLL using (mu, logvar)
        # In the heteroscedastic case we clamp logvar for numerical stability.
        reg_loss = torch.tensor(0.0, device=self.device)

        if reg_mu is not None and vad is not None and float(getattr(self.hparams, "lambda_vad", 0.0)) > 0.0:
            if reg_mu.size(-1) != 3:
                raise RuntimeError(f"vad_mu last dim must be 3, got {tuple(reg_mu.shape)}")

            # Ensure float tensors on device (avoid repeated conversions later)
            reg_mu = reg_mu.to(self.device, dtype=torch.float32)
            vad = vad.to(self.device, dtype=torch.float32)

            if reg_logvar is not None:
                if reg_logvar.size(-1) != 3:
                    raise RuntimeError(f"vad_logvar last dim must be 3, got {tuple(reg_logvar.shape)}")

                # Stable heteroscedastic Gaussian NLL:
                # NLL = 0.5 * (exp(-lv) * err^2 + lv)
                lv = reg_logvar.to(self.device, dtype=torch.float32)
                lv = torch.clamp(lv, min=self._logvar_min, max=self._logvar_max)
                err = (reg_mu - vad)  # [B,3]
                inv_var = torch.exp(-lv)
                nll_per_dim = 0.5 * (inv_var * (err ** 2) + lv)  # [B,3]
                reg_loss = (nll_per_dim * self.vad_weights).mean()
            else:
                # Mean-only SmoothL1 (Huber) on mu
                perdim = self._vad_huber(reg_mu, vad)  # [B,3]
                reg_loss = (perdim * self.vad_weights).mean()

            # Optional CCC term on the mean prediction only (weighted by lambda_ccc)
            if bool(getattr(self.hparams, "use_ccc_loss", True)) and float(getattr(self, "lambda_ccc", 1.0)) > 0.0:
                ccc_v = concordance_cc(reg_mu, vad)  # [3]
                ccc_term = 1.0 - (ccc_v * self.vad_weights).sum() / self.vad_weights.sum()
                reg_loss = reg_loss + float(self.lambda_ccc) * ccc_term.clamp_min(0.0)

        # ========= Categorical Loss =========
        # cat_loss.type in YAML controls the objective:
        #   - ce   : standard CrossEntropy (hard targets)
        #   - cbce : class-balanced CE (hard targets; requires/benefits from class counts)
        #   - kld  : KLDiv on (logp, target_dist)
        #   - jsd  : Jensen-Shannon divergence between predicted prob and target_dist
        # NOTE: when entropy curriculum is enabled, we apply it ONLY during TRAIN.

        cat_type = str(getattr(self, "cat_loss_type", "cbce")).lower()
        # --- DEBUG: log categorical loss type once (first time only) ---
        if not hasattr(self, "_loss_debug_logged"):
            self._loss_debug_logged = False
        if not self._loss_debug_logged:
            self._log(
                f"[DEBUG-LOSS] compute_objectives using cat_loss_type='{cat_type}' | "
                f"use_soft_targets={getattr(self, 'use_soft_targets', None)}",
                tag="LOSS",
            )
            self._loss_debug_logged = True
        # --- DEBUG: log hard/soft targets branch once (first time only) ---
        if not hasattr(self, "_targets_debug_logged"):
            self._targets_debug_logged = False

        # Resolve label_mode the same way as _get_cls_targets()
        label_mode_dbg = getattr(self, "current_label_mode", None)
        if label_mode_dbg is None:
            label_mode_dbg = getattr(self.hparams, "dist_mode", None)
        label_mode_dbg = str(label_mode_dbg).lower() if label_mode_dbg is not None else "merged"
        metric_kld_per_samp = None
        metric_jsd_per_samp = None

        if cat_type == "ce":
            # CE is HARD-label only
            if not self._targets_debug_logged:
                self._log(
                    f"[DEBUG-TARGETS] HARD targets branch | cat_type='{cat_type}' | "
                    f"label_mode={label_mode_dbg} | y_idx_shape={tuple(y_idx.shape)}",
                    tag="LOSS",
                )
                self._targets_debug_logged = True

            logits = preds["cat_logits"]
            per_samp = self._ce_none(logits, y_idx)  # [B]
            base_clf_loss = per_samp.mean()

        elif cat_type == "cbce":
            # CBCE supports BOTH:
            #  - hard labels (y_idx) when label_mode=='hard'
            #  - soft distributions (primary/merged/secondary) when label_mode!='hard'
            use_soft = (label_mode_dbg != "hard")

            if not self._targets_debug_logged:
                self._log(
                    f"[DEBUG-TARGETS] {'SOFT' if use_soft else 'HARD'} targets branch | cat_type='{cat_type}' | "
                    f"label_mode={label_mode_dbg} | emo_vec_present={getattr(batch, 'emo_vec', None) is not None}",
                    tag="LOSS",
                )
                self._targets_debug_logged = True

            logp = preds["cat_logp"]  # [B,C]

            if use_soft:
                # Soft CBCE: -sum_c w_c * q_c * log p_c
                soft_targets = self._get_cls_targets(batch, stage, y_idx).detach()  # [B,C]
                # Ensure numerical safety
                soft_targets = torch.clamp(soft_targets, min=1e-8)
                soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                w = getattr(self, "_cbce_class_weights", None)
                if w is not None:
                    w = w.to(logp.device, dtype=logp.dtype).view(1, -1)
                    per_samp = -((soft_targets * logp) * w).sum(dim=-1)  # [B]
                else:
                    per_samp = -(soft_targets * logp).sum(dim=-1)  # [B]

                base_clf_loss = per_samp.mean()

            else:
                # Hard CBCE: -w[y]*log p[y]
                logits = preds["cat_logits"]
                if getattr(self, "_cbce_class_weights", None) is not None:
                    logp_h = F.log_softmax(logits, dim=-1)
                    nll = -logp_h.gather(dim=-1, index=y_idx.view(-1, 1)).squeeze(-1)
                    w_y = self._cbce_class_weights.gather(dim=0, index=y_idx.view(-1))
                    per_samp = w_y * nll
                else:
                    per_samp = self._ce_none(logits, y_idx)  # [B]
                base_clf_loss = per_samp.mean()

        else:
            # Soft targets: divergence losses (KLD / JSD)
            if not self._targets_debug_logged:
                emo_vec_present = getattr(batch, "emo_vec", None) is not None
                self._log(
                    f"[DEBUG-TARGETS] SOFT targets branch | cat_type='{cat_type}' | "
                    f"label_mode={label_mode_dbg} | emo_vec_present={emo_vec_present}",
                    tag="LOSS",
                )
                self._targets_debug_logged = True

            soft_targets = self._get_cls_targets(batch, stage, y_idx).detach()  # [B,C]

            if cat_type == "jsd":
                # JSD(p||q) = 0.5*KL(p||m) + 0.5*KL(q||m), where m = 0.5*(p+q)
                logp = preds["cat_logp"]
                p = torch.exp(logp)
                q = soft_targets
                # numerical safety
                p = torch.clamp(p, 1e-8, 1.0)
                q = torch.clamp(q, 1e-8, 1.0)
                p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                m = 0.5 * (p + q)
                m = torch.clamp(m, 1e-8, 1.0)
                m = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1)  # [B]
                kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1)  # [B]
                per_samp = 0.5 * (kl_pm + kl_qm)
                base_clf_loss = per_samp.mean()

            else:
                # default: KLDiv
                # KLDivLoss(reduction='none') returns [B,C] elementwise: target * (log(target) - input)
                # Apply optional per-class weights (aligned to label_encoder order) BEFORE summing.
                kld_elem = self._kld_none(preds["cat_logp"], soft_targets)  # [B,C]

                w = getattr(self, "_kld_class_weights", None)
                if w is not None:
                    try:
                        w = w.to(kld_elem.device, dtype=kld_elem.dtype).view(1, -1)
                        if int(w.shape[-1]) == int(kld_elem.shape[-1]):
                            kld_elem = kld_elem * w
                    except Exception:
                        pass

                per_samp = kld_elem.sum(dim=-1)  # [B]
                base_clf_loss = per_samp.mean()

            try:
                metric_logp = preds["cat_logp"]
                metric_q = torch.clamp(soft_targets.detach(), min=1e-8, max=1.0)
                metric_q = metric_q / metric_q.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                metric_kld_per_samp = self._kld_none(metric_logp, metric_q).sum(dim=-1)

                metric_p = torch.exp(metric_logp).clamp(1e-8, 1.0)
                metric_p = metric_p / metric_p.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                metric_m = 0.5 * (metric_p + metric_q)
                metric_m = torch.clamp(metric_m, 1e-8, 1.0)
                metric_m = metric_m / metric_m.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                kl_pm_metric = (metric_p * (torch.log(metric_p) - torch.log(metric_m))).sum(dim=-1)
                kl_qm_metric = (metric_q * (torch.log(metric_q) - torch.log(metric_m))).sum(dim=-1)
                metric_jsd_per_samp = 0.5 * (kl_pm_metric + kl_qm_metric)
            except Exception:
                metric_kld_per_samp = None
                metric_jsd_per_samp = None

        # ---- ambiguity curriculum (sample filter/weighting) ----
        # Uses batch-provided: emo_entropy_norm / emo_maxprob / emo_margin (from dataio_prep).
        # Modes:
        #   1) "filter": keep low-ambiguity samples
        #   2) "filter_rev": keep high-ambiguity samples
        #   3) "weight": multiply per-sample clf loss by (1-ambiguity)^alpha
        #   4) "weight_rev": multiply per-sample clf loss by ambiguity^alpha
        clf_loss = base_clf_loss
        if stage == sb.Stage.TRAIN and getattr(self, "entropy_curriculum_enabled", False):
            mode, thr, alpha, min_w = self._entropy_curriculum_policy(epoch=getattr(self, "epoch", None))
            if getattr(self, "curriculum_state", None) is not None:
                self.curriculum_state.enabled = True
                self.curriculum_state.mode = str(mode)
                self.curriculum_state.phase = (
                    "active" if str(mode) in {"filter", "filter_rev", "weight", "weight_rev"} else "warmup"
                )
                self.curriculum_state.last_epoch = int(getattr(self, "epoch", -1) if getattr(self, "epoch", None) is not None else -1)
                self.curriculum_state.last_threshold = float(thr) if thr is not None else None
                self.curriculum_state.last_alpha = float(alpha) if alpha is not None else None
                self.curriculum_state.last_min_weight = float(min_w) if min_w is not None else None

            if str(mode) in {"filter", "filter_rev", "weight", "weight_rev"} and per_samp is not None:
                requested_sig_key = str(getattr(self.hparams, "ambiguity_signal_key", "emo_entropy_norm"))
                sig_key = requested_sig_key
                sig = getattr(batch, sig_key, None)
                if sig is None and sig_key != "emo_entropy_norm":
                    sig_key = "emo_entropy_norm"
                    sig = getattr(batch, sig_key, None)

                if sig is not None:
                    sig = getattr(sig, "data", sig).to(self.device, dtype=torch.float32).view(-1)

                    # Convert different confidence measures into a unified ambiguity scale in [0,1]
                    # Higher => more ambiguous / harder
                    if sig_key in ("emo_margin", "emo_maxprob"):
                        sig = 1.0 - torch.clamp(sig, 0.0, 1.0)
                    else:
                        sig = torch.clamp(sig, 0.0, 1.0)

                    amb = sig
                    self._entropy_avg = float(amb.mean().detach().cpu().item())

                    if mode in {"filter", "filter_rev"} and thr is not None:
                        cfg = getattr(self, "entropy_curriculum", {}) or {}
                        q_sched = cfg.get("quantile_schedule", None)
                        e_sched = cfg.get("schedule_epochs", None)
                        use_quantile = (
                            isinstance(q_sched, (list, tuple))
                            and isinstance(e_sched, (list, tuple))
                            and len(q_sched) > 0
                            and len(q_sched) == len(e_sched)
                        )

                        cutoff_val = None
                        if use_quantile:
                            q = min(1.0, max(0.0, float(thr)))
                            n = int(amb.numel())
                            k = int(math.ceil(q * float(n)))
                            k = max(0, min(n, k))
                            self._entropy_active_quantile = float(q)
                            keep = torch.zeros_like(amb, dtype=torch.bool)
                            if k > 0:
                                order = torch.argsort(amb, dim=0, descending=(mode == "filter_rev"))
                                keep[order[:k]] = True
                                cutoff_val = float(amb[order[k - 1]].detach().cpu().item())
                        else:
                            self._entropy_active_quantile = None
                            if mode == "filter_rev":
                                keep = (amb >= float(thr))
                            else:
                                keep = (amb <= float(thr))
                            cutoff_val = float(thr)

                        if cutoff_val is not None and math.isfinite(cutoff_val):
                            self._entropy_cutoff_sum += float(cutoff_val)
                            self._entropy_cutoff_n += 1

                        kept = int(keep.sum().detach().cpu().item())
                        seen = int(keep.numel())
                        self._entropy_kept += kept
                        self._entropy_seen += seen
                        if keep.any():
                            kept_amb = amb[keep]
                            kept_amb_sum = float(kept_amb.sum().detach().cpu().item())
                            kept_amb_n = int(kept_amb.numel())
                            kept_amb_min = float(kept_amb.min().detach().cpu().item())
                            kept_amb_max = float(kept_amb.max().detach().cpu().item())
                            self._entropy_kept_amb_sum += kept_amb_sum
                            self._entropy_kept_amb_n += kept_amb_n
                            if self._entropy_kept_amb_min is None:
                                self._entropy_kept_amb_min = kept_amb_min
                            else:
                                self._entropy_kept_amb_min = min(
                                    float(self._entropy_kept_amb_min), kept_amb_min
                                )
                            if self._entropy_kept_amb_max is None:
                                self._entropy_kept_amb_max = kept_amb_max
                            else:
                                self._entropy_kept_amb_max = max(
                                    float(self._entropy_kept_amb_max), kept_amb_max
                                )
                        try:
                            y_for_counts = y_idx
                            if hasattr(y_for_counts, "data"):
                                y_for_counts = y_for_counts.data
                            if y_for_counts.ndim > 1:
                                y_for_counts = y_for_counts.squeeze(-1)
                            y_for_counts = y_for_counts.long().view(-1)
                            if int(y_for_counts.numel()) == int(keep.numel()):
                                if self._entropy_kept_class_counts is None:
                                    self._entropy_kept_class_counts = torch.zeros(
                                        int(self.num_classes), dtype=torch.long
                                    )
                                kept_y = y_for_counts[keep].detach().to("cpu")
                                binc = torch.bincount(
                                    kept_y, minlength=int(self.num_classes)
                                ).to(dtype=torch.long)
                                self._entropy_kept_class_counts += binc
                        except Exception:
                            pass
                        if self._entropy_seen > 0:
                            self._entropy_keep_frac = float(self._entropy_kept / float(self._entropy_seen))
                        else:
                            self._entropy_keep_frac = None
                        clf_loss = per_samp[keep].mean() if keep.any() else per_samp.mean()

                    elif mode in {"weight", "weight_rev"} and alpha is not None:
                        if mode == "weight_rev":
                            # Keep pow stable at H=0 while preserving expected [0,1] support.
                            h = torch.clamp(amb, min=1e-8, max=1.0)
                            w = torch.pow(h, float(alpha))
                        else:
                            w = torch.pow(1.0 - amb, float(alpha))
                        if min_w is not None:
                            w = torch.clamp(w, min=float(min_w), max=1.0)
                        at_min = None
                        if min_w is not None:
                            at_min = (w <= (float(min_w) + 1e-12))
                            self._entropy_weight_min_count += int(at_min.sum().detach().cpu().item())
                            self._entropy_weight_total_count += int(at_min.numel())
                        mean_before_t = w.mean()
                        mean_before = float(mean_before_t.detach().cpu().item())
                        frac_min_weight = None
                        if min_w is not None:
                            if at_min is None:
                                at_min = (w <= (float(min_w) + 1e-12))
                            frac_min_weight = float(
                                at_min.to(dtype=torch.float32).mean().detach().cpu().item()
                            )
                        eps = 1e-8
                        if not math.isfinite(mean_before) or mean_before <= eps:
                            w = torch.ones_like(w)
                            mean_after = 1.0
                            self._log(
                                f"[CURR][WARN] Entropy weight mean too small/non-finite "
                                f"(mean_before={mean_before}); fallback to uniform weights.",
                                tag="CURR",
                            )
                        else:
                            w = w / mean_before_t
                            mean_after = float(w.mean().detach().cpu().item())
                        dbg_every = int(getattr(self, "_entropy_weight_batch_debug_every", 0))
                        if dbg_every > 0:
                            self._entropy_weight_batch_debug_counter = int(
                                getattr(self, "_entropy_weight_batch_debug_counter", 0)
                            ) + 1
                            if (self._entropy_weight_batch_debug_counter % dbg_every) == 0:
                                self._debug(
                                    f"[CURR][{mode}] mean_before={mean_before:.6f} "
                                    f"mean_after={mean_after:.6f} "
                                    f"frac_min_weight={frac_min_weight}"
                                )
                        w_cpu = w.detach().to("cpu", dtype=torch.float32).view(-1)
                        if int(w_cpu.numel()) > 0:
                            self._entropy_weight_values.append(w_cpu)
                        self._entropy_keep_frac = 1.0
                        clf_loss = (w * per_samp).mean()
                    else:
                        clf_loss = per_samp.mean()
                else:
                    warned_keys = getattr(self, "_warned_missing_ambiguity_signal", set())
                    if not isinstance(warned_keys, set):
                        warned_keys = set()
                    if requested_sig_key not in warned_keys:
                        self._log(
                            f"[CURR][WARN] ambiguity_signal_key='{requested_sig_key}' not found in batch "
                            "(fallback 'emo_entropy_norm' also missing). Curriculum skipped."
                        )
                        warned_keys.add(requested_sig_key)
                        self._warned_missing_ambiguity_signal = warned_keys
                    clf_loss = per_samp.mean()
            else:
                clf_loss = per_samp.mean()

        # ========= Total loss =========
        loss = (
            self.hparams.lambda_vad * reg_loss
            + self.hparams.lambda_cat * clf_loss
        )

        # ========= Metrics bookkeeping =========
        if stage == sb.Stage.TRAIN:
            # TRAIN: do NOT store logits/labels; optionally stream confusion + top-k.
            if getattr(self, "_train_cm", None) is not None:
                # Throttle TRAIN confusion/top-k updates for speed.
                self._train_metrics_batch_i = int(getattr(self, "_train_metrics_batch_i", 0)) + 1
                every = int(getattr(self, "_train_metrics_every", 100))
                if every < 1:
                    every = 1
                if (self._train_metrics_batch_i % every) == 0:
                    try:
                        self._train_update_cm_topk(clf_logp.detach(), y_idx.detach(), ks=(1, 2, 3))
                    except Exception:
                        pass

        else:
            # VALID/TEST: keep full logp/y for richer metrics
            try:
                self._stage_cls_lp.append(clf_logp.detach().cpu())
                self._stage_cls_y.append(y_idx.detach().cpu())
            except Exception:
                pass

            try:
                if metric_kld_per_samp is not None:
                    self._stage_kld_sum += float(metric_kld_per_samp.detach().sum().cpu().item())
                    self._stage_kld_count += int(metric_kld_per_samp.numel())
                if metric_jsd_per_samp is not None:
                    self._stage_jsd_sum += float(metric_jsd_per_samp.detach().sum().cpu().item())
                    self._stage_jsd_count += int(metric_jsd_per_samp.numel())
            except Exception:
                pass

            # Per-utterance label entropy for entropy-stratified VALID/TEST metrics.
            try:
                ent_norm = self._extract_eval_label_entropy_norm(batch)
                if ent_norm is not None:
                    self._stage_label_ent_norm.append(ent_norm.detach().cpu())
            except Exception:
                pass

            # ---- prediction confidence / ambiguity (via utils) ----
            try:
                probs = torch.exp(clf_logp.detach())  # [B,C]
                ds = distribution_stats(probs)        # dict with mean_* keys

                self._pred_ent_sum += float(ds.get("mean_ent", ds.get("ent", 0.0)))
                self._pred_maxp_sum += float(ds.get("mean_maxp", ds.get("maxp", 0.0)))
                self._pred_margin_sum += float(ds.get("mean_margin", ds.get("margin", 0.0)))
                self._pred_conf_n += 1

                # ECE bin accumulation (streaming)
                B = int(getattr(self, "_ece_bins", getattr(self.hparams, "ece_bins", 10)))
                conf_sum, acc_sum, count = ece_bin_sums_from_logp(
                    clf_logp.detach(),
                    y_idx.to(clf_logp.device),
                    n_bins=B,
                )

                # Support both list-based and tensor-based accumulators
                if isinstance(self._ece_conf_sum, list):
                    for bi in range(B):
                        self._ece_conf_sum[bi] += float(conf_sum[bi].detach().cpu().item())
                        self._ece_acc_sum[bi] += float(acc_sum[bi].detach().cpu().item())
                        self._ece_count[bi] += int(count[bi].detach().cpu().item())
                else:
                    self._ece_conf_sum = self._ece_conf_sum.to(conf_sum.device) + conf_sum
                    self._ece_acc_sum = self._ece_acc_sum.to(acc_sum.device) + acc_sum
                    self._ece_count = self._ece_count.to(count.device) + count
            except Exception:
                pass

            # ---- label ambiguity / confidence (requires emo_vec) ----
            try:
                emo_vec = getattr(batch, "emo_vec", None)
                if emo_vec is not None:
                    vec = getattr(emo_vec, "data", emo_vec).to(self.device, dtype=torch.float32)
                    vec = self._normalize_distribution(vec)

                    ls = distribution_stats(vec)
                    self._label_ent_sum += float(ls.get("mean_ent", ls.get("ent", 0.0)))
                    self._label_maxp_sum += float(ls.get("mean_maxp", ls.get("maxp", 0.0)))
                    self._label_margin_sum += float(ls.get("mean_margin", ls.get("margin", 0.0)))
                    self._label_conf_n += 1
            except Exception:
                pass

            # Track ids for export/debug
            try:
                if hasattr(batch, "id"):
                    self._stage_ids += list(batch.id)
            except Exception:
                pass

            # VAD metrics only for non-TRAIN stages (memory safety)
            if reg_mu is not None and vad is not None:
                try:
                    self._stage_preds.append(reg_mu.detach().cpu())
                    self._stage_tgts.append(vad.detach().cpu())
                except Exception:
                    pass

                # ---- VAD uncertainty metrics (heteroscedastic only) ----
                try:
                    if reg_logvar is not None:
                        lv = reg_logvar.detach().to(self.device, dtype=torch.float32)
                        lv = torch.clamp(lv, min=self._logvar_min, max=self._logvar_max)
                        err = (
                            reg_mu.detach().to(self.device, dtype=torch.float32)
                            - vad.detach().to(self.device, dtype=torch.float32)
                        )
                        inv_var = torch.exp(-lv)
                        nll_per_dim = 0.5 * (inv_var * (err ** 2) + lv)  # [B,3]
                        nll = nll_per_dim.mean(dim=-1)  # [B]
                        self._vad_nll_sum += float(nll.mean().cpu().item())
                        self._vad_nll_n += 1

                        std = preds.get("vad_std", None)
                        if std is None:
                            std = torch.exp(0.5 * lv)
                        self._vad_std_sum += (
                            std.detach().mean(dim=0).cpu().numpy().astype(np.float64)
                        )
                        self._vad_std_n += 1
                except Exception:
                    pass

        return loss

    def _resolve_eval_ckpt_meta(self, min_key=None, max_key=None):
        """Best-effort: figure out which checkpoint will/was used for TEST and return a dict.

        Returns:
          {"path": str, "meta": dict, "min_key": ..., "max_key": ...} or None
        """
        # 1) Prefer what we already computed earlier
        meta = getattr(self, "_eval_ckpt_meta", None)
        if isinstance(meta, dict) and meta.get("path"):
            return meta

        # 2) Some SpeechBrain versions keep the last chosen checkpoint here
        cp = getattr(self, "checkpointer", None)
        last = getattr(cp, "_last_ckpt", None) if cp is not None else None
        if isinstance(last, dict) and ("meta" in last or "path" in last):
            path = str(last.get("path", ""))
            m = last.get("meta", {}) or {}
            return {"path": path, "meta": dict(m), "min_key": min_key, "max_key": max_key}

        # 3) Fallback: ask the checkpointer again
        if cp is not None:
            try:
                chosen = cp.find_checkpoint(min_key=min_key, max_key=max_key)
                if chosen is not None:
                    return {
                        "path": str(getattr(chosen, "path", "")),
                        "meta": dict(getattr(chosen, "meta", {}) or {}),
                        "min_key": min_key,
                        "max_key": max_key,
                    }
            except Exception:
                pass

        return None

    def _print_eval_ckpt_banner(self, stage, min_key=None, max_key=None):
        """Always print (and log) which checkpoint epoch is used for TEST."""
        if stage != sb.Stage.TEST:
            return
        meta = self._resolve_eval_ckpt_meta(min_key=min_key, max_key=max_key)
        if isinstance(meta, dict):
            ep = (meta.get("meta", {}) or {}).get("epoch", None)
            metric_key = _checkpoint_meta_key(getattr(self, "_ckpt_metric", "macro_f1"))
            metric_val = (meta.get("meta", {}) or {}).get(metric_key, None)
            path = meta.get("path", "")
            msg = f"[TEST][CKPT] Using checkpoint epoch={ep} {metric_key}={metric_val} path={path}"
        else:
            msg = "[TEST][CKPT] Using checkpoint: unknown (could not resolve meta)"

        # print so it shows up even if logger filters tags
        try:
            print(msg, flush=True)
        except Exception:
            pass

        # also send through your logger
        try:
            self._log(msg, tag="CKPT")
        except Exception:
            pass

    def on_stage_start(self, stage, epoch=None):
        """Called at the beginning of each stage or epoch.
        Handles progressive unfreezing, dynamic weighting, and reset bookkeeping.
        """
        # Crash-fast label sanity checks should run once per stage/epoch (first batch).
        self._label_assert_done = False

        # ---- Reset accumulators every stage ----
        self._stage_preds, self._stage_tgts = [], []
        self._stage_cls_lp, self._stage_cls_y, self._stage_ids = [], [], []
        self._stage_margin = []
        self._stage_label_ent_norm = []
        self._stage_kld_sum = 0.0
        self._stage_kld_count = 0
        self._stage_jsd_sum = 0.0
        self._stage_jsd_count = 0

        # ---- confidence / ambiguity accumulators (memory-safe scalar sums) ----
        self._pred_ent_sum = 0.0
        self._pred_maxp_sum = 0.0
        self._pred_margin_sum = 0.0
        self._pred_conf_n = 0

        self._label_ent_sum = 0.0
        self._label_maxp_sum = 0.0
        self._label_margin_sum = 0.0
        self._label_conf_n = 0

        # Entropy-curriculum debug stats are stage/epoch scoped.
        self._entropy_keep_frac = None
        self._entropy_avg = None
        self._entropy_kept = 0
        self._entropy_seen = 0
        self._entropy_weight_values = []
        self._entropy_weight_min_count = 0
        self._entropy_weight_total_count = 0
        self._entropy_active_quantile = None
        self._entropy_cutoff_sum = 0.0
        self._entropy_cutoff_n = 0
        self._entropy_kept_amb_sum = 0.0
        self._entropy_kept_amb_n = 0
        self._entropy_kept_amb_min = None
        self._entropy_kept_amb_max = None
        self._entropy_kept_class_counts = None

        # Expected Calibration Error (ECE)
        self._ece_bins = int(getattr(self.hparams, "ece_bins", 10))
        B = self._ece_bins
        self._ece_conf_sum = [0.0] * B
        self._ece_acc_sum  = [0.0] * B
        self._ece_count    = [0] * B

        # VAD uncertainty accumulators (VALID/TEST only)
        self._vad_nll_sum = 0.0
        self._vad_nll_n = 0
        self._vad_std_sum = np.zeros(3, dtype=np.float64)
        self._vad_std_n = 0

        # ---- TRAIN streaming classification stats (avoid storing logits) ----
        self._train_cm = None
        self._train_topk_correct = {1: 0, 2: 0, 3: 0}
        self._train_topk_total = 0
        if stage == sb.Stage.TRAIN:
            C = int(getattr(self, "num_classes", 0) or 0)
            if C > 0:
                self._train_cm = torch.zeros((C, C), dtype=torch.int64, device="cpu")
            # reset TRAIN throttle counter each epoch
            self._train_metrics_batch_i = 0
            self._entropy_weight_batch_debug_counter = 0

        # ---- TEST: always print which checkpoint epoch is used ----
        if stage == sb.Stage.TEST:
            # Use keys stashed from __main__ (best effort)
            min_key = getattr(self, "_eval_min_key", None)
            max_key = getattr(self, "_eval_max_key", None)
            self._print_eval_ckpt_banner(stage, min_key=min_key, max_key=max_key)

        # ---- Training-specific logic ----
        if stage == sb.Stage.TRAIN and epoch is not None:
            self.epoch = int(epoch)
            self.current_label_mode = str(_hp_get(self.hparams, "dist_mode", "merged")).lower()
            self._log(f"Epoch {epoch} | label mode={self.current_label_mode}", tag="LABEL")
            maybe_log_curriculum_event(self, epoch=int(epoch))

            # Apply gradual-unfreeze schedule (selective freezing)
            _maybe_update_unfreeze(self, epoch)

            # Loss weighting (reflect actual hparams)
            self.reg_w = float(getattr(self.hparams, "lambda_vad", 1.0))
            self.clf_w = float(getattr(self.hparams, "lambda_cat", 0.8))
            self._log(
                f"Epoch {epoch} | lambda_vad={self.reg_w:.2f} | lambda_cat={self.clf_w:.2f}",
                tag="LOSS"
            )
            maybe_log_loss_schedule_event(self, epoch=int(epoch))

            # Parameter summary for encoder
            total_params = sum(p.numel() for p in self.modules["ssl_model"].parameters())
            trainable_params = sum(
                p.numel() for p in self.modules["ssl_model"].parameters() if p.requires_grad
            )
            if stage == sb.Stage.TRAIN:
                self._log(
                    f"Epoch {epoch} | encoder trainable {100 * trainable_params / total_params:.1f}%",
                    tag="ENC"
                )

            # -----------------------------
            # Update augmentation config for this epoch
            # -----------------------------
            effective_cfg, allowed_augs, allow_combo, phase_name, scale = \
                self.aug_scheduler.get_effective_aug_config(epoch, return_phase_info=True)

            if hasattr(self, "augment") and self.augment is not None:
                self.augment.update_config(effective_cfg, allow_combo)

            # Flatten probabilities for readability
            flat_probs = {f"{k}_p": float(v.get("p", 0.0)) for k, v in effective_cfg.items()}

            # Debug-only logging for AugmentScheduler (not train_logger)
            allowed_list = list(allowed_augs) if allowed_augs is not None else []
            self._debug(
                f"[AUG-SCHED] epoch={epoch} phase={phase_name} scale={scale:.2f} "
                f"allowed={allowed_list} probs={flat_probs}"
            )
            # Human-readable scheduler summary (for console/paper/debug)
            self._log(
                f"Epoch {epoch} | AUG phase={phase_name} | scale={scale:.2f} | augs={allowed_list}",
                tag="AUG",
            )


    def fit_batch(self, batch):
        """
        Gradient accumulation version of fit_batch.
        Works with:
        - param-group optimizer (SSL vs heads)
        - mixup/cutmix
        - frozen/partially unfrozen encoder
        - SB train loop
        """

        # forward
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        if loss is None:
            raise RuntimeError("compute_objectives returned None.")

        # scale loss
        accum_steps = self._grad_accum
        loss_scaled = loss / accum_steps
        loss_scaled.backward()

        # gradient clipping (only when stepping)
        self._step_counter += 1
        if self._step_counter % accum_steps == 0:

            gc = getattr(self.hparams, "grad_clip", None)
            if gc is not None:
                head_params = [p for n, p in self.modules.named_parameters() if not n.startswith("ssl_model.")]
                ssl_trainable = [p for p in self.modules["ssl_model"].parameters() if p.requires_grad]
                params_to_clip = head_params + ssl_trainable
                if params_to_clip:
                    nn.utils.clip_grad_norm_(params_to_clip, gc)
            
            # optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()  

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage)
            loss = self.compute_objectives(predictions, batch, stage)
        return loss


    def on_stage_end(self, stage, stage_loss, epoch=None):
        """
        Called at the end of each stage (TRAIN / VALID / TEST).
        Handles metric aggregation, LR scheduling, checkpointing, and test export.
        """

        # ---------------------------------------------------------
        # TEST stage does not have an epoch counter in SpeechBrain,
        # so `epoch` arrives as None. For reporting, replace it with
        # the selected checkpoint's epoch (from checkpoint meta).
        # ---------------------------------------------------------
        epoch_display = epoch
        if stage == sb.Stage.TEST and epoch is None:
            try:
                min_key = getattr(self, "_eval_min_key", None)
                max_key = getattr(self, "_eval_max_key", None)
                m = self._resolve_eval_ckpt_meta(min_key=min_key, max_key=max_key)
                if isinstance(m, dict):
                    epoch_display = (m.get("meta", {}) or {}).get("epoch", None)
            except Exception:
                pass


        # -----------------------------
        # Stage availability flags
        # -----------------------------
        have_vad = hasattr(self, "_stage_preds") and len(self._stage_preds) > 0
        have_cls = (
            (hasattr(self, "_stage_cls_lp") and len(self._stage_cls_lp) > 0)
            or (stage == sb.Stage.TRAIN and getattr(self, "_train_cm", None) is not None and int(self._train_cm.sum().item()) > 0)
        )

        # -----------------------------
        # 2. Regression metrics (CCC + RMSE)
        # -----------------------------
        vad_metrics = {}
        if have_vad:
            preds = torch.cat(self._stage_preds)
            tgts  = torch.cat(self._stage_tgts)
            ccc_vals  = ccc(preds, tgts)
            rmse_vals = rmse(preds, tgts)
            vad_metrics = {
                "CCC_V": float(ccc_vals[0]),
                "CCC_A": float(ccc_vals[1]),
                "CCC_D": float(ccc_vals[2]),
                "RMSE_V": float(rmse_vals[0]),
                "RMSE_A": float(rmse_vals[1]),
                "RMSE_D": float(rmse_vals[2]),
                "ccc_avg": float(ccc_vals.mean().item()),
            }

        # -----------------------------
        # 3. Categorical metrics
        # -----------------------------
        cat_metrics_out = {}
        extra_cls_metrics = {}
        ent_bin_metrics = {}
        conf_mat = None
        topk = {}

        if have_cls:
            if stage == sb.Stage.TRAIN and getattr(self, "_train_cm", None) is not None and int(self._train_cm.sum().item()) > 0:
                # TRAIN: compute metrics from streaming confusion matrix + top-k counters
                cm = self._train_cm
                conf_mat = cm

                tot = float(cm.sum().item())
                diag = torch.diag(cm).to(torch.float32)
                row_sum = cm.sum(dim=1).to(torch.float32)

                acc = float((diag.sum() / max(tot, 1.0)).item())
                ua = float((diag / row_sum.clamp_min(1.0)).mean().item())
                cat_metrics_out = {"acc": acc, "ua": ua}

                ttot = float(getattr(self, "_train_topk_total", 0) or 0)
                if ttot > 0:
                    topk = {
                        "top1_acc": float(self._train_topk_correct.get(1, 0) / ttot),
                        "top2_acc": float(self._train_topk_correct.get(2, 0) / ttot),
                        "top3_acc": float(self._train_topk_correct.get(3, 0) / ttot),
                    }
                else:
                    topk = {}

                # TRAIN: compute macro/weighted F1 directly from confusion matrix (no logits stored)
                try:
                    cmf = cm.to(torch.float32)
                    tp = torch.diag(cmf)
                    fp = cmf.sum(dim=0) - tp
                    fn = cmf.sum(dim=1) - tp

                    prec = tp / (tp + fp).clamp_min(1.0)
                    rec  = tp / (tp + fn).clamp_min(1.0)
                    f1   = 2.0 * prec * rec / (prec + rec).clamp_min(1e-8)

                    # Macro-F1: unweighted mean over classes that have support
                    support = cmf.sum(dim=1)
                    valid = support > 0
                    macro_f1 = float(f1[valid].mean().item()) if bool(valid.any().item()) else 0.0

                    # Weighted-F1: weighted by class support
                    weighted_f1 = float((f1 * support).sum().item() / support.sum().clamp_min(1.0).item())

                    extra_cls_metrics = {
                        "macro_f1": macro_f1,
                        "weighted_f1": weighted_f1,
                    }
                except Exception:
                    extra_cls_metrics = {}

            else:
                # VALID/TEST: full logp-based metrics
                cls_lp = torch.cat(self._stage_cls_lp, dim=0)
                cls_y  = torch.cat(self._stage_cls_y, dim=0)

                cat_metrics_out = cat_metrics(cls_lp, cls_y, self.class_names)

                try:
                    conf_mat = confusion_matrix_from_logp(cls_lp, cls_y, self.num_classes)
                except Exception:
                    conf_mat = None

                try:
                    extra_cls_metrics = compute_cls_extra_metrics(cls_lp, cls_y, self.class_names)
                except Exception:
                    extra_cls_metrics = {}

                try:
                    topk = topk_accuracy(cls_lp, cls_y, ks=(1, 2, 3))
                except Exception:
                    topk = {}

                # ---- Ensure macro_f1 exists for checkpointing (robust fallback) ----
                if "macro_f1" not in extra_cls_metrics or extra_cls_metrics.get("macro_f1", None) is None:
                    try:
                        if conf_mat is None:
                            conf_mat = confusion_matrix_from_logp(cls_lp, cls_y, self.num_classes)
                        cmf = conf_mat.to(torch.float32)
                        tp = torch.diag(cmf)
                        fp = cmf.sum(dim=0) - tp
                        fn = cmf.sum(dim=1) - tp
                        prec = tp / (tp + fp).clamp_min(1.0)
                        rec  = tp / (tp + fn).clamp_min(1.0)
                        f1   = 2.0 * prec * rec / (prec + rec).clamp_min(1e-8)
                        support = cmf.sum(dim=1)
                        valid = support > 0
                        extra_cls_metrics["macro_f1"] = float(f1[valid].mean().item()) if bool(valid.any().item()) else 0.0
                    except Exception:
                        extra_cls_metrics["macro_f1"] = 0.0

                if "weighted_f1" not in extra_cls_metrics or extra_cls_metrics.get("weighted_f1", None) is None:
                    try:
                        if conf_mat is None:
                            conf_mat = confusion_matrix_from_logp(cls_lp, cls_y, self.num_classes)
                        cmf = conf_mat.to(torch.float32)
                        tp = torch.diag(cmf)
                        fp = cmf.sum(dim=0) - tp
                        fn = cmf.sum(dim=1) - tp
                        prec = tp / (tp + fp).clamp_min(1.0)
                        rec  = tp / (tp + fn).clamp_min(1.0)
                        f1   = 2.0 * prec * rec / (prec + rec).clamp_min(1e-8)
                        support = cmf.sum(dim=1)
                        extra_cls_metrics["weighted_f1"] = float((f1 * support).sum().item() / support.sum().clamp_min(1.0).item())
                    except Exception:
                        extra_cls_metrics["weighted_f1"] = 0.0

        # ---- Entropy-stratified categorical metrics (VALID/TEST only) ----
        if stage in (sb.Stage.VALID, sb.Stage.TEST) and have_cls:
            try:
                cls_lp = torch.cat(self._stage_cls_lp, dim=0)
                y_true = torch.cat(self._stage_cls_y, dim=0).view(-1).cpu().numpy()
                y_pred = torch.argmax(cls_lp, dim=-1).view(-1).cpu().numpy()
                if hasattr(self, "_stage_label_ent_norm") and len(self._stage_label_ent_norm) > 0:
                    ent = torch.cat(self._stage_label_ent_norm, dim=0).view(-1).cpu().numpy()
                else:
                    ent = np.full((len(y_true),), np.nan, dtype=np.float32)

                n = min(len(y_true), len(y_pred), len(ent))
                y_true = y_true[:n]
                y_pred = y_pred[:n]
                ent = ent[:n]

                edges = self._get_entropy_bin_edges()
                names = ("low", "mid", "high")
                for i, name in enumerate(names):
                    lo, hi = float(edges[i]), float(edges[i + 1])
                    if i < len(names) - 1:
                        m = np.isfinite(ent) & (ent >= lo) & (ent < hi)
                    else:
                        m = np.isfinite(ent) & (ent >= lo) & (ent <= hi)

                    n_bin = int(np.sum(m))
                    ent_bin_metrics[f"n_bin_{name}"] = n_bin
                    ent_bin_metrics[f"n_ent_{name}"] = n_bin
                    if n_bin > 0:
                        mf1_bin = float(
                            f1_score(y_true[m], y_pred[m], average="macro", zero_division=0)
                        )
                        ent_bin_metrics[f"macro_f1_bin_{name}"] = mf1_bin
                        ent_bin_metrics[f"macro_f1_ent_{name}"] = mf1_bin
                    else:
                        ent_bin_metrics[f"macro_f1_bin_{name}"] = float("nan")
                        ent_bin_metrics[f"macro_f1_ent_{name}"] = float("nan")

                extra_cls_metrics.update(ent_bin_metrics)

                def _fmt(v):
                    return "nan" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{float(v):.4f}"

                self._debug(
                    f"{stage.name.lower()} "
                    f"macro_f1_ent_low={_fmt(ent_bin_metrics.get('macro_f1_bin_low'))}, "
                    f"macro_f1_ent_mid={_fmt(ent_bin_metrics.get('macro_f1_bin_mid'))}, "
                    f"macro_f1_ent_high={_fmt(ent_bin_metrics.get('macro_f1_bin_high'))}, "
                    f"n_low={int(ent_bin_metrics.get('n_bin_low', 0))}, "
                    f"n_mid={int(ent_bin_metrics.get('n_bin_mid', 0))}, "
                    f"n_high={int(ent_bin_metrics.get('n_bin_high', 0))}"
                )
            except Exception as e:
                self._debug(f"[ENT-BIN][WARN] Failed entropy-stratified metrics: {e}")

        # -----------------------------
        # 4. Aggregate all metrics
        # -----------------------------
        metrics = {**vad_metrics}
        metrics.update({f"CAT_{k.upper()}": v for k, v in cat_metrics_out.items()})
        metrics.update(extra_cls_metrics)
        metrics.update(topk)
        metrics.update(ent_bin_metrics)
        if getattr(self, "_stage_kld_count", 0) > 0:
            metrics["kld"] = float(self._stage_kld_sum / float(self._stage_kld_count))
        if getattr(self, "_stage_jsd_count", 0) > 0:
            metrics["jsd"] = float(self._stage_jsd_sum / float(self._stage_jsd_count))
        if conf_mat is not None:
            metrics["CAT_confusion"] = conf_mat.cpu().tolist()

        # Guardrail: checkpoint/early-stopping key must always exist
        if "macro_f1" not in metrics or metrics.get("macro_f1", None) is None:
            metrics["macro_f1"] = 0.0
        if "CAT_UA" in metrics and metrics.get("CAT_UA", None) is not None:
            metrics["uar"] = float(metrics["CAT_UA"])
        if "CAT_ACC" in metrics and metrics.get("CAT_ACC", None) is not None:
            metrics["acc"] = float(metrics["CAT_ACC"])
        if "ccc_avg" in metrics and metrics.get("ccc_avg", None) is not None:
            metrics["ccc"] = float(metrics["ccc_avg"])

        # ---- confidence / ambiguity summaries ----
        if getattr(self, "_pred_conf_n", 0) > 0:
            n = float(self._pred_conf_n)
            metrics["PRED_ENT"] = float(self._pred_ent_sum / n)
            metrics["PRED_MAXP"] = float(self._pred_maxp_sum / n)
            metrics["PRED_MARGIN"] = float(self._pred_margin_sum / n)

        if getattr(self, "_label_conf_n", 0) > 0:
            n = float(self._label_conf_n)
            metrics["LABEL_ENT"] = float(self._label_ent_sum / n)
            metrics["LABEL_MAXP"] = float(self._label_maxp_sum / n)
            metrics["LABEL_MARGIN"] = float(self._label_margin_sum / n)

        # ---- Expected Calibration Error (ECE) ----
        try:
            conf_sum = self._ece_conf_sum
            acc_sum  = self._ece_acc_sum
            count    = self._ece_count

            # Supports either list-based or tensor-based accumulators
            if isinstance(count, list):
                if sum(count) > 0:
                    metrics["CAT_ECE"] = float(ece_from_bin_sums(conf_sum, acc_sum, count))
            else:
                if int(count.sum().item()) > 0:
                    metrics["CAT_ECE"] = float(ece_from_bin_sums(conf_sum, acc_sum, count))
        except Exception:
            pass

        # ---- VAD uncertainty summaries (VALID/TEST only) ----
        if stage in (sb.Stage.VALID, sb.Stage.TEST) and getattr(self, "_vad_nll_n", 0) > 0:
            metrics["VAD_NLL"] = float(self._vad_nll_sum / float(self._vad_nll_n))
        if stage in (sb.Stage.VALID, sb.Stage.TEST) and getattr(self, "_vad_std_n", 0) > 0:
            stdm = self._vad_std_sum / float(self._vad_std_n)
            metrics["VAD_STD_V"] = float(stdm[0])
            metrics["VAD_STD_A"] = float(stdm[1])
            metrics["VAD_STD_D"] = float(stdm[2])

        # ccc_avg for stats: fallback to 0.0 if not available
        stats = {"loss": float(stage_loss)}
        if have_vad:
            stats["ccc_avg"] = float(vad_metrics.get("ccc_avg", 0.0))
        stats.update(metrics)

        # -----------------------------
        # 4.5 Checkpointing + Early stopping (VALID only)
        # Metric is configurable from YAML.
        # -----------------------------
        if stage == sb.Stage.VALID and getattr(self, "checkpointer", None) is not None:
            try:
                num_keep = int(getattr(self, "_ckpt_num_keep", getattr(self.hparams, "ckpt_num_to_keep", 3)))
            except Exception:
                num_keep = 3

            ckpt_meta = {
                "end-of-epoch": True,
                "epoch": int(epoch) if epoch is not None else -1,

                # key metrics
                "macro_f1": float(stats.get("macro_f1", 0.0)),
                "uar": float(stats.get("uar", stats.get("CAT_UA", 0.0))),
                "acc": float(stats.get("acc", stats.get("CAT_ACC", 0.0))),
                "kld": float(stats.get("kld", float("nan"))),
                "jsd": float(stats.get("jsd", float("nan"))),

                # cls report
                "weighted_f1": float(stats.get("weighted_f1", 0.0)),
                "CAT_ACC": float(stats.get("CAT_ACC", 0.0)),
                "CAT_UA": float(stats.get("CAT_UA", 0.0)),

                # vad report
                "ccc_avg": float(stats.get("ccc_avg", 0.0)),
                "CCC_V": float(stats.get("CCC_V", 0.0)),
                "CCC_A": float(stats.get("CCC_A", 0.0)),
                "CCC_D": float(stats.get("CCC_D", 0.0)),
                "RMSE_V": float(stats.get("RMSE_V", 0.0)),
                "RMSE_A": float(stats.get("RMSE_A", 0.0)),
                "RMSE_D": float(stats.get("RMSE_D", 0.0)),

                # extras (if present)
                "top1_acc": float(stats.get("top1_acc", 0.0)),
                "top2_acc": float(stats.get("top2_acc", 0.0)),
                "top3_acc": float(stats.get("top3_acc", 0.0)),
                "CAT_ECE": float(stats.get("CAT_ECE", 0.0)),
                "VAD_NLL": float(stats.get("VAD_NLL", 0.0)),
                "VAD_STD_V": float(stats.get("VAD_STD_V", 0.0)),
                "VAD_STD_A": float(stats.get("VAD_STD_A", 0.0)),
                "VAD_STD_D": float(stats.get("VAD_STD_D", 0.0)),
            }

            # Always save a resumable "latest" checkpoint every VALID epoch.
            # This avoids losing late-epoch progress when best-metric plateaus early.
            try:
                if bool(getattr(self, "_ckpt_save_latest", getattr(self.hparams, "ckpt_save_latest", True))):
                    latest_meta = dict(ckpt_meta)
                    latest_meta["ckpt_role"] = "latest"
                    latest_meta["is_latest"] = True
                    if hasattr(self.checkpointer, "save_checkpoint"):
                        try:
                            self.checkpointer.save_checkpoint(meta=latest_meta)
                        except TypeError:
                            # Compatibility with older SpeechBrain signatures.
                            self.checkpointer.save_checkpoint()
                    else:
                        self._log(
                            "[CKPT][WARN] Checkpointer has no save_checkpoint(); "
                            "latest-per-epoch checkpoint not saved.",
                            tag="CKPT",
                        )
            except Exception as e:
                self._log(f"[CKPT][WARN] Failed to save latest resume checkpoint: {e}", tag="CKPT")

            metric_name = str(getattr(self, "_ckpt_metric", "macro_f1"))
            metric_mode = str(getattr(self, "_ckpt_mode", "max")).lower()
            metric_key = str(getattr(self, "_ckpt_meta_key", _checkpoint_meta_key(metric_name)))
            score, score_key = _resolve_metric_from_stats(stats, metric_name)

            if score is None:
                self._log(
                    f"[CKPT][WARN] Selection metric '{metric_name}' was missing/non-finite at epoch={epoch}; "
                    "best-checkpoint candidate not updated.",
                    tag="CKPT",
                )
            else:
                min_delta = float(getattr(self, "_ckpt_min_delta", getattr(self.hparams, "ckpt_min_delta", 0.0)))
                best = getattr(self, "_best_ckpt_score", None)
                improved = (
                    (best is None)
                    or (score < (best - min_delta) if metric_mode == "min" else score > (best + min_delta))
                )

                if improved:
                    self._best_ckpt_score = score
                    self._best_ckpt_epoch = int(epoch) if epoch is not None else None

                save_kwargs = {
                    "meta": ckpt_meta,
                    "num_to_keep": num_keep,
                }
                if metric_mode == "min":
                    save_kwargs["min_keys"] = [metric_key]
                else:
                    save_kwargs["max_keys"] = [metric_key]

                self.checkpointer.save_and_keep_only(**save_kwargs)

                if improved:
                    self._log(
                        f"[CKPT] New best {score_key or metric_key}={score:.6f} at epoch={epoch} "
                        f"(mode={metric_mode}).",
                        tag="CKPT",
                    )
                else:
                    self._log(
                        f"[CKPT] Saved VALID candidate using {score_key or metric_key}={score:.6f} "
                        f"(mode={metric_mode}, keep_top_k={num_keep}).",
                        tag="CKPT",
                    )

        # -----------------------------
        # Early stopping (VALID only)
        # Metric is configurable from YAML.
        # -----------------------------
        if stage == sb.Stage.VALID and bool(getattr(self, "_es_enabled", False)):
            metric_name = str(getattr(self, "_es_metric", "macro_f1"))
            cur, resolved_key = _resolve_metric_from_stats(stats, metric_name)
            if cur is None:
                self._log(
                    f"[EarlyStopping][WARN] Metric '{metric_name}' was missing/non-finite at epoch={epoch}; "
                    "skipping early-stopping update for this epoch.",
                    tag="ES",
                )
                cur = None
            mode = str(getattr(self, "_es_mode", "max")).lower()
            min_delta = float(getattr(self, "_es_min_delta", 0.0))

            if cur is not None:
                improved = (cur > (self._es_best + min_delta)) if mode == "max" else (cur < (self._es_best - min_delta))

                if improved:
                    self._es_best = cur
                    self._es_bad_epochs = 0
                else:
                    self._es_bad_epochs = int(getattr(self, "_es_bad_epochs", 0)) + 1

                if self._es_bad_epochs >= int(getattr(self, "_es_patience", 0)):
                    self._log(
                        f"[EarlyStopping] STOP at epoch={epoch} | metric={resolved_key or metric_name} "
                        f"| best={self._es_best:.4f} | cur={cur:.4f} | bad_epochs={self._es_bad_epochs}",
                        tag="ES",
                    )
                    # End training after this epoch
                    try:
                        if hasattr(self.hparams, "epoch_counter") and self.hparams.epoch_counter is not None:
                            self.hparams.epoch_counter.limit = int(epoch)
                    except Exception:
                        pass
                    self._es_enabled = False
                
        # -----------------------------
        # 5. Logging
        # -----------------------------
        if stage.name == 'TEST':
            log_kwargs = {
                "stats_meta": {"Stage": stage.name, "Epoch": epoch_display},
            }
        else:        
            log_kwargs = {
                "stats_meta": {"Stage": stage.name, "Epoch": epoch},
            }

        if stage == sb.Stage.TRAIN:
            log_kwargs["train_stats"] = stats
        elif stage == sb.Stage.VALID:
            log_kwargs["valid_stats"] = stats
        elif stage == sb.Stage.TEST:
            log_kwargs["test_stats"] = stats

        # ---- Human-readable multiline logging (never miss epochs) ----
        msg = None
        try:
            lines = []
            lines.append(f"===== STAGE: {stage.name} | EPOCH: {epoch_display} =====")
            lines.append(f"loss: {stage_loss:.4f}")

            # VAD metrics
            if "ccc_avg" in metrics:
                lines.append(f"CCC_V: {metrics.get('CCC_V', 0):.3f}")
                lines.append(f"CCC_A: {metrics.get('CCC_A', 0):.3f}")
                lines.append(f"CCC_D: {metrics.get('CCC_D', 0):.3f}")
                lines.append(f"CCC_AVG: {metrics.get('ccc_avg', 0):.3f}")
            if "VAD_NLL" in metrics:
                lines.append(f"VAD_NLL: {metrics['VAD_NLL']:.4f}")
            if "VAD_STD_V" in metrics:
                lines.append(f"VAD_STD_V: {metrics['VAD_STD_V']:.4f}")
                lines.append(f"VAD_STD_A: {metrics['VAD_STD_A']:.4f}")
                lines.append(f"VAD_STD_D: {metrics['VAD_STD_D']:.4f}")

            # Classification metrics
            if cat_metrics_out:
                lines.append(f"CAT_ACC: {cat_metrics_out.get('acc', 0):.3f}")
                lines.append(f"CAT_UA: {cat_metrics_out.get('ua', 0):.3f}")
            if extra_cls_metrics:
                lines.append(f"MACRO_F1: {extra_cls_metrics.get('macro_f1', 0):.3f}")
                lines.append(f"WEIGHTED_F1: {extra_cls_metrics.get('weighted_f1', 0):.3f}")
            if stage in (sb.Stage.VALID, sb.Stage.TEST) and ("macro_f1_bin_low" in metrics):
                lines.append(
                    "ENT_BIN_MACRO_F1: "
                    f"low={metrics.get('macro_f1_bin_low', float('nan'))}, "
                    f"mid={metrics.get('macro_f1_bin_mid', float('nan'))}, "
                    f"high={metrics.get('macro_f1_bin_high', float('nan'))}, "
                    f"n_low={int(metrics.get('n_bin_low', 0))}, "
                    f"n_mid={int(metrics.get('n_bin_mid', 0))}, "
                    f"n_high={int(metrics.get('n_bin_high', 0))}"
                )
            
            if "PRED_ENT" in metrics:
                lines.append(f"PRED_ENT: {metrics['PRED_ENT']:.3f}")
                lines.append(f"PRED_MAXP: {metrics['PRED_MAXP']:.3f}")
                lines.append(f"PRED_MARGIN: {metrics['PRED_MARGIN']:.3f}")
            if "LABEL_ENT" in metrics:
                lines.append(f"LABEL_ENT: {metrics['LABEL_ENT']:.3f}")
                lines.append(f"LABEL_MAXP: {metrics['LABEL_MAXP']:.3f}")
                lines.append(f"LABEL_MARGIN: {metrics['LABEL_MARGIN']:.3f}")
            if "CAT_ECE" in metrics:
                lines.append(f"CAT_ECE: {metrics['CAT_ECE']:.4f}")

            lines.append("----------------------------------------")
            msg = "\n".join(lines) + "\n"
            wrote = False
            if hasattr(self.hparams, "train_logger") and hasattr(self.hparams.train_logger, "write"):
                self.hparams.train_logger.write(msg)
                wrote = True
            if not wrote:
                self._append_train_log_fallback(msg)
                wrote = True
            if not wrote:
                print(msg, flush=True)
        except Exception as e:
            warn_msg = f"[WARN] Failed multiline logging: {e}"
            print(warn_msg, flush=True)
            try:
                self._log(warn_msg, tag="LOG")
            except Exception:
                pass
            if msg:
                self._append_train_log_fallback(msg)

        if hasattr(self.hparams, "train_logger"):
            try:
                self.hparams.train_logger.log_stats(**log_kwargs)
            except Exception as e:
                try:
                    self._log(f"[LOG][WARN] train_logger.log_stats failed: {e}", tag="LOG")
                except Exception:
                    pass
                fallback_line = (
                    f"Stage: {stage.name}, Epoch: {epoch_display} - "
                    f"loss: {float(stage_loss):.6f}, macro_f1: {float(metrics.get('macro_f1', 0.0)):.6f}\n"
                )
                self._append_train_log_fallback(fallback_line)

        # ---- Entropy curriculum summary (TRAIN only) ----
        if stage == sb.Stage.TRAIN:
            self._log_entropy_curriculum_epoch_debug(epoch=epoch)

        # ---- Compact console summary ----
        # NOTE: print_stage_summary expects top-k / calibration / ambiguity keys in the dicts it receives.
        # We compute top-k into `metrics` above, so merge it here for display.
        summary_extra = dict(extra_cls_metrics) if extra_cls_metrics is not None else {}
        for k in (
            "top1_acc", "top2_acc", "top3_acc",
            "CAT_ECE",
            "PRED_ENT", "PRED_MAXP", "PRED_MARGIN",
            "LABEL_ENT", "LABEL_MAXP", "LABEL_MARGIN",
        ):
            if k in metrics and metrics[k] is not None:
                summary_extra[k] = metrics[k]

        self._print_stage_summary(
            stage=stage,
            epoch=epoch_display,
            stage_loss=stage_loss,
            cat_metrics_out=cat_metrics_out,
            extra_cls_metrics=summary_extra,
        )

        # Confusion matrix debug logging only
        if conf_mat is not None and stage == sb.Stage.VALID and epoch is not None:
            self._debug(f"[CONF][Epoch {epoch}]\n{conf_mat.cpu().numpy()}")

        # -----------------------------
        # 6. Scheduler updates (VALID only)
        # -----------------------------
        if stage == sb.Stage.VALID:
            self.unfreeze_state.unfrozen_count = int(self.unfrozen_count)
            self.unfreeze_state.phase = str(self.current_phase)

            # Head LR scheduler -> applies to head param group only
            if hasattr(self.hparams, "lr_annealing") and self.optimizer is not None:
                _, new_head_lr = self.hparams.lr_annealing(stage_loss)
                _, head_idx = self._get_param_group_indices()
                self._set_group_lr(head_idx, new_head_lr)

            # SSL LR scheduler (support either lr_annealing_ssl or legacy lr_annealing_wavlm) -> SSL group only
            ssl_sched = getattr(self.hparams, "lr_annealing_ssl", None)
            if ssl_sched is None:
                ssl_sched = getattr(self.hparams, "lr_annealing_wavlm", None)

            if ssl_sched is not None and self.optimizer is not None:
                _, new_ssl_lr = ssl_sched(stage_loss)
                ssl_idx, _ = self._get_param_group_indices()
                self._set_group_lr(ssl_idx, new_ssl_lr)

        # -----------------------------
        # 7. Test predictions export
        # -----------------------------
        elif stage == sb.Stage.TEST:
            export_test_predictions(self)

        # -----------------------------
        # 8. Reset accumulators
        # -----------------------------
        for attr in ["_stage_preds", "_stage_tgts", "_stage_cls_lp", "_stage_cls_y", "_stage_ids", "_stage_margin", "_stage_label_ent_norm"]:
            if hasattr(self, attr):
                getattr(self, attr).clear()

        # TRAIN streaming accumulators
        self._train_cm = None
        self._train_topk_correct = {1: 0, 2: 0, 3: 0}
        self._train_topk_total = 0


    def on_evaluate_start(self, max_key=None, min_key=None):
        # Don’t carry optimizer state into eval
        self._drop_optimizer_recoverables()

        # Remember which checkpoint will be used for eval
        self._eval_ckpt_meta = None
        if getattr(self, "checkpointer", None) is not None:
            try:
                ckpt_dir = getattr(self.checkpointer, "checkpoints_dir", None)
                self._log(f"[CKPT] checkpoints_dir={ckpt_dir}", tag="CKPT")

                # Debug: list what SpeechBrain sees
                try:
                    ckpts = list(self.checkpointer.list_checkpoints())
                except Exception:
                    ckpts = []
                self._log(f"[CKPT] discovered={len(ckpts)}", tag="CKPT")
                for c in ckpts:
                    try:
                        self._debug(f"[CKPT] {getattr(c,'path',None)} meta={getattr(c,'meta',None)}")
                    except Exception:
                        pass

                chosen = self.checkpointer.find_checkpoint(min_key=min_key, max_key=max_key)

                self._eval_ckpt_meta = {
                    "path": str(getattr(chosen, "path", "")),
                    "meta": getattr(chosen, "meta", {}) or {},
                    "min_key": min_key,
                    "max_key": max_key,
                }

                ep = self._eval_ckpt_meta["meta"].get("epoch", None)
                metric_key = _checkpoint_meta_key(getattr(self, "_ckpt_metric", "macro_f1"))
                metric_val = self._eval_ckpt_meta["meta"].get(metric_key, None)
                self._log(
                    f"[CKPT] selected for eval: epoch={ep} {metric_key}={metric_val} "
                    f"path={self._eval_ckpt_meta['path']}",
                    tag="CKPT",
                )
            except Exception as e:
                self._log(f"[CKPT][WARN] failed selecting ckpt: {e}", tag="CKPT")

        return super().on_evaluate_start(max_key=max_key, min_key=min_key)
    
    def load_state_dict(self, state_dict, strict=True):
        """Restore frozen/unfrozen state when resuming from checkpoint."""
        if "unfrozen_count" in state_dict:
            self.unfrozen_count = state_dict["unfrozen_count"]
            self.current_phase = state_dict.get("phase", "heads_only")
            self._log(f"[Resume] Restored unfreeze state: {self.unfrozen_count} layers unfrozen ({self.current_phase}).")

            # Physically re-apply correct requires_grad flags
            encoder_layers = _resolve_encoder_layers(self)
            for i, layer in enumerate(encoder_layers):
                trainable = (i >= len(encoder_layers) - self.unfrozen_count)
                for p in layer.parameters():
                    p.requires_grad = trainable
        return super().load_state_dict(state_dict, strict)

def init_optimizers(self):
    """Initialize a single optimizer with two param groups (SSL encoder vs heads).

    Design goals:
      - Works when the encoder is fully frozen (requires_grad=False)
      - Works with gradual unfreezing (params are already in optimizer; toggling requires_grad is enough)
      - Supports independent LRs for encoder vs heads
    """

    # -----------------------------
    # 1) Resolve encoder module
    # -----------------------------
    ssl_mod = None
    try:
        # SpeechBrain ModulesDict-like
        ssl_mod = self.modules["ssl_model"]
    except Exception:
        ssl_mod = getattr(self, "ssl_model", None)

    # -----------------------------
    # 2) Build parameter lists (keep frozen params in optimizer for later unfreeze)
    # -----------------------------
    ssl_params: list = []
    head_params: list = []

    for name, p in self.modules.named_parameters():
        # Typical names start with "ssl_model." for the encoder.
        if name.startswith("ssl_model."):
            ssl_params.append(p)
        else:
            head_params.append(p)

    if len(head_params) == 0:
        raise RuntimeError("init_optimizers(): No head parameters found (cat/vad heads missing?).")

    # -----------------------------
    # 3) Parameter summary (total vs trainable)
    # -----------------------------
    if ssl_mod is not None:
        enc_total = sum(p.numel() for p in ssl_mod.parameters())
        enc_trainable = sum(p.numel() for p in ssl_mod.parameters() if p.requires_grad)
    else:
        enc_total = 0
        enc_trainable = 0

    head_total = sum(p.numel() for p in head_params)
    head_trainable = sum(p.numel() for p in head_params if p.requires_grad)

    # Guard against divide-by-zero
    enc_pct = (100.0 * enc_trainable / max(enc_total, 1))

    self._log(
        f"[PARAMS] encoder total={enc_total:,} trainable={enc_trainable:,} ({enc_pct:.1f}%) | "
        f"heads total={head_total:,} trainable={head_trainable:,}",
        tag="MODEL",
    )
    try:
        self._debug(
            f"[PARAMS] encoder total={enc_total:,} trainable={enc_trainable:,} ({enc_pct:.1f}%) | "
            f"heads total={head_total:,} trainable={head_trainable:,}"
        )
    except Exception:
        pass

    # -----------------------------
    # 4) Retrieve LRs with defaults
    # -----------------------------
    ssl_lr = float(getattr(self.hparams, "ssl_lr", 1e-5))
    head_lr = float(getattr(self.hparams, "head_lr", 3e-4))

    # -----------------------------
    # 5) Create optimizer with param groups
    # -----------------------------
    optim_groups = []
    has_ssl = len(ssl_params) > 0
    if has_ssl:
        optim_groups.append({"params": ssl_params, "lr": ssl_lr})

    optim_groups.append({"params": head_params, "lr": head_lr})

    self.optimizer = self.hparams.opt_class(optim_groups)

    # Record param-group indices for later LR scheduling
    if has_ssl:
        self._ssl_group_idx = 0
        self._head_group_idx = 1
    else:
        self._ssl_group_idx = None
        self._head_group_idx = 0

    # -----------------------------
    # 6) Register recoverables + SpeechBrain dict
    # -----------------------------
    if getattr(self, "checkpointer", None) is not None:
        try:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)
        except Exception:
            pass

    self.optimizers_dict = {"optimizer": self.optimizer}

# Bind module-level optimizer initializer as SerBrain method (surgical fix).
SerBrain.init_optimizers = init_optimizers

# ==========================================================
# Checkpoint loading helpers
# =========================================================

def _normalize_path(path) -> str:
    """Normalize filesystem paths for robust comparisons."""
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def _is_ckpt_dir(path: str) -> bool:
    """True when `path` points to a concrete SpeechBrain checkpoint directory."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "CKPT.yaml"))


def _resolve_ckpt_paths(path: str):
    """Resolve checkpoint input into search dir + optional exact checkpoint dir.

    Returns:
      search_dir: directory containing CKPT+* folders (for Checkpointer)
      target_ckpt_dir: exact CKPT+* directory if user passed one, else None
      payload_dir: directory containing module *.ckpt files (for manual loading)
    """
    p = os.path.normpath(os.path.expanduser(str(path)))

    # Allow direct file input (CKPT.yaml or a payload .ckpt file).
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
                m = re.match(r"^\s*epoch\s*:\s*([0-9]+)\s*$", line)
                if m:
                    epoch_val = int(m.group(1))
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
            f = os.path.join(ckpt_dir, f"{name}.ckpt")
            if not os.path.isfile(f):
                missing.append(str(name))
    except Exception:
        pass
    return missing


def _find_ckpt(path: str, names):
    """Return first existing ckpt file matching any name in `names`."""
    for n in names:
        ckpt = os.path.join(path, f"{n}.ckpt")
        if os.path.isfile(ckpt):
            return ckpt
    return None

@torch.no_grad()
def _load_state_dict_safe(module: torch.nn.Module, ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(f"[warn] Loaded {os.path.basename(ckpt_path)} "
              f"with missing={len(missing)}, unexpected={len(unexpected)}")

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
      - lr_annealing_ssl is the SSL-group scheduler (legacy fallback handled by caller).
    """
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

        # Dataloader recovery is optional and version-dependent; keep guarded.
        if dataloader_train is not None:
            # SpeechBrain's saved key is typically "dataloader-TRAIN".
            recoverables["dataloader-TRAIN"] = dataloader_train

        cp = Checkpointer(checkpoints_dir=search_dir, recoverables=recoverables)
        # Temporary checkpointer must know custom object hooks too, otherwise
        # older SpeechBrain raises "Don't know how to load <UnfreezeState>".
        try:
            if hasattr(cp, "custom_save_hooks") and hasattr(cp, "custom_load_hooks"):
                if unfreeze_state is not None:
                    cp.custom_save_hooks["unfreeze_state"] = _save_unfreeze_state
                    cp.custom_load_hooks["unfreeze_state"] = _load_unfreeze_state
                if curriculum_state is not None:
                    cp.custom_save_hooks["curriculum_state"] = _save_curriculum_state
                    cp.custom_load_hooks["curriculum_state"] = _load_curriculum_state
            # Keep these optional for backward compatibility with older ckpts.
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
                    # Older SpeechBrain versions may not support ckpt_predicate.
                    recovered_ckpt = cp.recover_if_possible()
        else:
            try:
                recovered_ckpt = cp.recover_if_possible(allow_partial=True)
            except TypeError:
                # Older SpeechBrain versions don't support allow_partial
                recovered_ckpt = cp.recover_if_possible()

        counter_after = None
        if epoch_counter is not None and hasattr(epoch_counter, "current"):
            try:
                counter_after = int(getattr(epoch_counter, "current"))
            except Exception:
                counter_after = None

        loaded = recovered_ckpt is not None
        if (not loaded) and (counter_before is not None) and (counter_after is not None):
            loaded = (counter_after != counter_before)

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
                "full_state": (len(missing) == 0),
            }
        )
        if restore_info["loaded"]:
            print("[info] Full resume from", restore_info["loaded_ckpt_dir"] or search_dir)
            if missing:
                print(
                    "[warn] Partial restore: missing recoverables:",
                    ",".join(missing),
                )
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

    # Optimizers and schedulers are *not* loaded in retune mode.
    print("[info] Retune mode: optimizers/schedulers start fresh.")
    restore_info["loaded"] = bool(loaded_any)
    restore_info["loaded_ckpt_dir"] = str(payload_dir)
    restore_info["meta_epoch"] = _read_epoch_from_ckpt_yaml(payload_dir)
    return restore_info


# ==========================================================
# Main entry point
# RECIPE BEGINS!
# ==========================================================
if __name__ == "__main__":
    # SpeechBrain standard argument parsing:
    # - first positional arg: YAML hparams file
    # - remaining args: optional HyperPyYAML overrides (e.g., --seed 17)
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters (YAML is source of truth; CLI overrides optional)
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Ensure a label_encoder object exists in hparams (needed for Checkpointer recoverables).
    # NOTE: hparams can be a dict, so use dict-style access.
    try:
        has_le = ("label_encoder" in hparams) and (hparams["label_encoder"] is not None)
    except Exception:
        has_le = False

    if not has_le:
        try:
            hparams["label_encoder"] = sb.dataio.encoder.CategoricalEncoder()
        except Exception:
            hparams["label_encoder"] = None

    # validate_hparams(hparams)

    # Make run_opts visible (your get_ssl_model reads it)
    hparams["run_opts"] = run_opts

    # -----------------------------
    # Reproducibility
    # -----------------------------
    seed = int(_hp_get(hparams, "seed", 1411))
    deterministic = bool(_hp_get(hparams, "deterministic", False))
    set_global_seed(seed, deterministic)

    # Resolve per-fold output path *and* keep dependent paths in sync
    os.makedirs(hparams["output_folder"], exist_ok=True)
    os.makedirs(hparams["save_folder"], exist_ok=True)


    # Re-create train logger if the path changed after HyperPyYAML load
    if "train_logger" in hparams:
        new_log_path = os.path.join(hparams["output_folder"], "train_log.txt")
        cur_log_path = getattr(hparams["train_logger"], "save_file", None)
        if cur_log_path != new_log_path:
            hparams["train_logger"] = sb.utils.train_logger.FileTrainLogger(
                save_file=new_log_path
            )

    # Create experiment directory
    mode = str(_hp_get(hparams, "mode", "scratch")).lower()
    if mode != "resume":
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )

    # -----------------------------
    # Data
    # -----------------------------
    datasets = dataio_prep(hparams)

    # -----------------------------
    # Checkpointer
    # -----------------------------
    # If your YAML already constructs a checkpointer object, prefer it.
    # Otherwise create a default one here.
    from pathlib import Path
    ckpt_dir = Path(str(_hp_get(hparams, "save_folder", "./checkpoints")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Prefer YAML-constructed checkpointer if present; otherwise create a default one.
    checkpointer = hparams.get("checkpointer", None)
    if checkpointer is None:
        # Start with empty recoverables; SerBrain will add recoverables in on_fit_start/init_optimizers.
        checkpointer = Checkpointer(checkpoints_dir=ckpt_dir, recoverables={})
    else:
        # Keep its directory in sync with save_folder (SpeechBrain expects a Path here)
        try:
            checkpointer.checkpoints_dir = ckpt_dir
        except Exception:
            pass

    # -----------------------------
    # Initialize Brain
    # -----------------------------
    brain = SerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )

    # ==========================================================
    # Resume / Fine-tune metadata (actual load happens in on_fit_start)
    # ==========================================================
    mode = str(hparams.get("mode", "scratch")).lower()

    # accept both keys: ckpt_path and ckpt-path
    ckpt_path = str(hparams.get("ckpt_path", "") or "")
    if not ckpt_path:
        ckpt_path = str(hparams.get("ckpt-path", "") or "")

    reset_head = bool(hparams.get("reset_head", False) or hparams.get("reset-head", False))
    reset_clf  = bool(hparams.get("reset_clf", False) or hparams.get("reset-clf", False))
    reset_reg  = bool(hparams.get("reset_reg", False) or hparams.get("reset-reg", False))
    freeze_pat = str(hparams.get("freeze", "") or "")
    load_mode  = str(hparams.get("load_mode", "backbone+heads")).lower()
    if mode == "resume" and load_mode != "all":
        print(f"[RESUME][WARN] forcing load_mode='all' (got '{load_mode}').")
        load_mode = "all"

    if mode in {"resume", "ft"}:
        if not ckpt_path:
            raise ValueError("mode is 'resume'/'ft' but ckpt_path is empty. Set ckpt_path in YAML.")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ckpt_path does not exist: {ckpt_path}")
        # Stash pending resume config to run after modules/optimizers exist.
        brain._pending_resume = {
            "mode": mode,
            "ckpt_path": ckpt_path,
            "reset_head": reset_head,
            "reset_clf": reset_clf,
            "reset_reg": reset_reg,
            "freeze_pat": freeze_pat,
            "load_mode": load_mode,
        }

    # Ensure eval loaders don't shuffle or drop items by default
    train_loader_opts = dict(hparams.get("dataloader_options", {}))
    valid_loader_opts = dict(hparams.get("valid_dataloader_options", {}))
    if not valid_loader_opts:
        valid_loader_opts = {**train_loader_opts}
    valid_loader_opts.update({"shuffle": False, "drop_last": False})

    brain.fit(
        epoch_counter=brain.hparams.epoch_counter,
        train_set=datasets["train"],                
        valid_set=datasets["valid"],
        train_loader_kwargs=train_loader_opts,
        valid_loader_kwargs=valid_loader_opts,
    )

    ckpt_metric = _normalize_metric_name(
        getattr(
            brain,
            "_ckpt_metric",
            _hp_get(_hp_get(hparams, "checkpoint_selection", None), "metric", hparams.get("checkpoint_metric", "macro_f1")),
        )
    )
    ckpt_mode = str(
        getattr(
            brain,
            "_ckpt_mode",
            _hp_get(_hp_get(hparams, "checkpoint_selection", None), "mode", "max"),
        )
    ).lower()
    ckpt_meta_key = _checkpoint_meta_key(ckpt_metric)
    if ckpt_mode == "min":
        min_key, max_key = ckpt_meta_key, None
    else:
        min_key, max_key = None, ckpt_meta_key


    test_loader_opts = dict(valid_loader_opts)
    test_loader_opts.update(hparams.get("test_dataloader_options", {}))
    test_loader_opts.update({"shuffle": False, "drop_last": False})

    # ---------------------------------------------------------
    # Resolve and log which checkpoint will be used for TEST
    # (SpeechBrain's TEST stage prints epoch=None, so log ckpt meta explicitly)
    # ---------------------------------------------------------
    # Stash min_key/max_key for TEST checkpoint resolution/logging
    brain._eval_min_key = min_key
    brain._eval_max_key = max_key
    try:
        chosen_ckpt = None
        if getattr(brain, "checkpointer", None) is not None:
            # returns a Checkpoint object with `.path` and `.meta`
            chosen_ckpt = brain.checkpointer.find_checkpoint(min_key=min_key, max_key=max_key)

        if chosen_ckpt is not None:
            ckpt_path = str(getattr(chosen_ckpt, "path", ""))
            ckpt_meta = getattr(chosen_ckpt, "meta", {}) or {}
            ckpt_epoch = ckpt_meta.get("epoch", None)
            ckpt_metric_value = ckpt_meta.get(ckpt_meta_key, None)

            # Make visible to SerBrain.on_stage_start(TEST)
            brain._eval_ckpt_meta = {
                "path": ckpt_path,
                "meta": dict(ckpt_meta),
                "min_key": min_key,
                "max_key": max_key,
            }

            # Also print immediately
            print(f"[TEST][CKPT] Using checkpoint epoch={ckpt_epoch} {ckpt_meta_key}={ckpt_metric_value} path={ckpt_path}")
        else:
            brain._eval_ckpt_meta = None
            print("[TEST][CKPT] No checkpoint resolved (using default / latest behavior).")
    except Exception as e:
        brain._eval_ckpt_meta = None
        print(f"[TEST][CKPT][WARN] Failed to resolve checkpoint meta: {e}")

    test_stats = brain.evaluate(
        test_set=datasets["test"],
        min_key=min_key,
        max_key=max_key,
        test_loader_kwargs=test_loader_opts,
    )
