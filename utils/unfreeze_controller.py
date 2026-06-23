# Interspeech 2026 Unfreezing controller functions
# NOTE: Standalone helpers. Pass the Brain instance explicitly as `brain`.
# Written defensively to support different `brain.modules` container types (ModuleDict / SimpleNamespace / dict).

from __future__ import annotations

from typing import Any, Optional
from utils.run_logging import maybe_log_unfreeze_event


def _mod_get(mods: Any, key: str) -> Any:
    """Get a module from `brain.modules` regardless of its container type."""
    if mods is None:
        return None
    # Attribute access (SpeechBrain common pattern)
    if hasattr(mods, key):
        return getattr(mods, key)
    # Mapping access (ModuleDict / dict)
    try:
        return mods[key]
    except Exception:
        return None


def _iter_params(m: Any):
    if m is None:
        return []
    try:
        return list(m.parameters())
    except Exception:
        return []


def _resolve_encoder_layers(brain):
    """
    Returns the list/ModuleList of transformer layers for gradual unfreeze.
    Handles SpeechBrain HF-wrapped WavLM encoders in 1.0.2.
    """
    m = _mod_get(brain.modules, "ssl_model")
    if m is None:
        raise RuntimeError("[GradualUnfreeze] brain.modules has no 'ssl_model'.")

    # HF-style: ssl_model.model.encoder.layers
    if hasattr(m, "model") and hasattr(m.model, "encoder") and hasattr(m.model.encoder, "layers"):
        layers = m.model.encoder.layers
        brain._log(f"[GradualUnfreeze] Detected {len(layers)} transformer layers via 'model.encoder.layers'.")
        return layers

    # Fallback: ssl_model.encoder.layers (older wrappers)
    if hasattr(m, "encoder") and hasattr(m.encoder, "layers"):
        layers = m.encoder.layers
        brain._log(f"[GradualUnfreeze] Detected {len(layers)} transformer layers via 'encoder.layers'.")
        return layers

    raise RuntimeError(
        "Could not determine SSL transformer layers. "
        "Tried ssl_model.model.encoder.layers and ssl_model.encoder.layers."
    )

def _reapply_unfreeze_from_state(brain):
    """Use brain.unfreeze_state to set requires_grad on encoder layers."""
    layers = _resolve_encoder_layers(brain)
    L = len(layers)
    k = int(getattr(brain.unfreeze_state, "unfrozen_count", 0))

    ssl_model = _mod_get(brain.modules, "ssl_model")

    # Freeze all encoder params
    for p in _iter_params(ssl_model):
        p.requires_grad = False

    # Unfreeze top-k layers
    if k > 0:
        start = max(0, L - k)
        for i in range(start, L):
            for p in _iter_params(layers[i]):
                p.requires_grad = True

    # Heads always trainable
    for head in ("vad_mlp", "cat_mlp"):
        hm = _mod_get(brain.modules, head)
        for p in _iter_params(hm):
            p.requires_grad = True
    maybe_log_unfreeze_event(
        brain,
        epoch=getattr(brain, "epoch", None),
        reason="resume_reapply",
    )
    
def _unfreeze_top_layers(brain, num_layers):
    """Unfreeze top N transformer layers of the encoder."""
    encoder_layers = _resolve_encoder_layers(brain)
    total_layers = len(encoder_layers)

    num_layers = int(num_layers)
    num_layers = max(0, min(num_layers, total_layers))
    start = total_layers - num_layers

    for i in range(start, total_layers):
        for p in _iter_params(encoder_layers[i]):
            p.requires_grad = True

    brain.unfrozen_count = num_layers
    brain._log(f"Unfroze top {num_layers}/{total_layers} layers.")

def _adjust_ssl_lr(brain, factor: float):
    """Adjust SSL param-group LR after unfreezing new layers.

    Compatible with older/newer Brain helpers:
      - prefers brain._scale_group_lr(group_idx, factor) if present
      - otherwise multiplies optimizer.param_groups[group_idx]['lr'] by factor
    """
    ssl_idx, _ = brain._get_param_group_indices()

    if hasattr(brain, "_scale_group_lr"):
        brain._scale_group_lr(ssl_idx, factor)
        brain._log(f"Adjusted SSL group LR ×{factor:.2f}.")
        return

    # Fallback: direct optimizer edit
    try:
        g = brain.optimizer.param_groups[int(ssl_idx)]
        g["lr"] = float(g.get("lr", 0.0)) * float(factor)
        brain._log(f"Adjusted SSL group LR ×{factor:.2f}.")
    except Exception as e:
        brain._log(f"[Unfreeze][WARN] Failed to adjust SSL LR: {e}")

def _apply_gradual_unfreeze(brain, epoch: int):
    """
    Apply gradual-unfreeze schedule based on hparams.

    Expects optional hparams keys:
        - gradual_unfreeze: bool
        - freeze_head_epochs: int
        - max_unfrozen_layers: int
        - unfreeze_steps: list of dict/objects with fields:
            epoch: int
            layers: int
            lr_factor: float (multiplier for SSL LR)
    """
    if not getattr(brain.hparams, "gradual_unfreeze", False):
        return

    # Resolve encoder layers and schedule parameters
    encoder_layers = _resolve_encoder_layers(brain)
    total_layers = len(encoder_layers)
    max_layers = int(getattr(brain.hparams, "max_unfrozen_layers", total_layers) or total_layers)

    freeze_head_epochs = int(getattr(brain.hparams, "freeze_head_epochs", 0))
    steps = getattr(brain.hparams, "unfreeze_steps", [])

    # -----------------------
    # Phase 0: heads-only
    # -----------------------
    if epoch <= freeze_head_epochs:
        if brain.unfrozen_count != 0 or brain.current_phase != "heads_only":
            ssl_model = _mod_get(brain.modules, "ssl_model")
            # freeze entire encoder
            for p in _iter_params(ssl_model):
                p.requires_grad = False
            # keep heads trainable
            for head in ("vad_mlp", "cat_mlp"):
                hm = _mod_get(brain.modules, head)
                for p in _iter_params(hm):
                    p.requires_grad = True

            brain.unfrozen_count = 0
            brain.current_phase = "heads_only"
            brain.unfreeze_state.unfrozen_count = 0
            brain.unfreeze_state.phase = "heads_only"
            brain._log(f"[GradualUnfreeze] Epoch {epoch}: heads-only (encoder frozen).")
            maybe_log_unfreeze_event(brain, epoch=epoch, reason="heads_only")
        return

    # -----------------------
    # Later phases: follow unfreeze_steps
    # -----------------------
    if not steps:
        # no schedule defined; nothing to do
        return

    def _get_step_attr(step, key, default=None):
        if isinstance(step, dict):
            return step.get(key, default)
        return getattr(step, key, default)

    applied_any = False

    for step in steps:
        step_epoch = int(_get_step_attr(step, "epoch", -1))
        if step_epoch != epoch:
            continue

        target_layers = int(_get_step_attr(step, "layers", 0))
        lr_factor = float(_get_step_attr(step, "lr_factor", 1.0))

        if target_layers <= brain.unfrozen_count:
            # already at or beyond this unfreeze level
            continue

        new_k = min(target_layers, max_layers, total_layers)
        _unfreeze_top_layers(brain, new_k)
        _adjust_ssl_lr(brain, lr_factor)

        brain.unfrozen_count = new_k
        brain.current_phase = f"enc_{new_k}"
        brain.unfreeze_state.unfrozen_count = new_k
        brain.unfreeze_state.phase = brain.current_phase

        applied_any = True
        brain._log(
            f"[GradualUnfreeze] Epoch {epoch}: unfroze top {new_k}/{total_layers} "
            f"layers (lr_factor={lr_factor:.2f})."
        )
        maybe_log_unfreeze_event(brain, epoch=epoch, reason=f"step_epoch_{step_epoch}")

    if not applied_any and epoch > freeze_head_epochs:
        # No matching step, keep current state but log once per epoch
        brain._log(
            f"[GradualUnfreeze] Epoch {epoch}: no matching entry in unfreeze_steps; "
            f"keeping {brain.unfrozen_count} encoder layer(s) unfrozen."
        )

def _maybe_update_unfreeze(brain, epoch: int):
    """Compatibility wrapper for unfreezing logic.

    This helper accepts either a brain object or an epoch argument for backward-compatible calls.
    This method delegates to the configured unfreezing strategy:
        - if `gradual_unfreeze` is enabled, use `_apply_gradual_unfreeze(epoch)`.
        - otherwise, intentionally no-op to avoid undefined legacy calls.

    It must never crash the training loop.
    """
    try:
        if bool(getattr(brain.hparams, "gradual_unfreeze", False)):
            _apply_gradual_unfreeze(brain, int(epoch))
            return

        # Legacy/static schedules were part of older experiments.
        # Intentionally no-op here to avoid calling undefined helpers.
        return
    except Exception as e:
        brain._log(f"[Unfreeze][WARN] _maybe_update_unfreeze failed: {e}")
