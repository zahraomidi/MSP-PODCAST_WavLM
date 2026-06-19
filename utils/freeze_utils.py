import torch

# ---------------------------
# Freezing / Unfreezing Logic
# ---------------------------
def apply_unfreeze_schedule(self, epoch):
    """Apply freeze-heads-then-gradual-unfreeze schedule with resume support."""
    if not getattr(self.hparams, "gradual_unfreeze", False):
        return

    freeze_head_epochs = getattr(self.hparams, "freeze_head_epochs", 5)
    schedule = getattr(self.hparams, "unfreeze_steps", [])
    resumed_state = getattr(self, "unfrozen_count", 0)

    # ---- PHASE 1: HEAD-ONLY ----
    if epoch <= freeze_head_epochs and resumed_state == 0:
        for name, module in self.modules.items():
            if "mlp" in name.lower():  # heads
                for p in module.parameters():
                    p.requires_grad = True
            else:
                for p in module.parameters():
                    p.requires_grad = False
        self.current_phase = "heads_only"
        self._log(f"[Epoch {epoch}] SSL encoder frozen — training heads only.")
        self.print_param_stats(tag=f"after_unfreeze_epoch_{epoch}")
        return

    # ---- PHASE 2: GRADUAL UNFREEZE ----
    for step in schedule:
        if epoch == step["epoch"]:
            num_layers = step.get("layers", 0)
            lr_factor = step.get("lr_factor", 1.0)
            if num_layers <= self.unfrozen_count:
                return  # already done
            self._unfreeze_top_layers(num_layers)
            self._adjust_ssl_lr(lr_factor)
            self.unfrozen_count = num_layers
            self.current_phase = "unfreezing"
            msg = f"[Epoch {epoch}] Unfroze top {num_layers} SSL layers | LR ×{lr_factor:.2f}"
            self._log(msg)
            if hasattr(self.hparams, "train_logger"):
                self.hparams.train_logger.log_stats({"unfreeze_event": msg})
            self.print_param_stats(tag=f"after_unfreeze_epoch_{epoch}")
            break


def _freeze_all_but_heads(self):
    """Freeze the entire SSL encoder, keep heads trainable."""
    ssl_model = self.modules.ssl_model
    self._log("[FREEZE] Freezing encoder parameters...")
    for p in ssl_model.parameters():
        p.requires_grad = False

    # Unfreeze all heads
    for head in ["cat_mlp", "vad_mlp", "bin_mlp"]:
        if head in self.modules:
            for p in self.modules[head].parameters():
                p.requires_grad = True
            self._log(f"[FREEZE] Keeping head '{head}' trainable")

    self._log_trainable_params()  # optional verification
    

def _unfreeze_top_layers(self, num_layers):
    """Unfreeze top N transformer layers (or all)."""
    ssl_model = self.modules.ssl_model
    if not hasattr(ssl_model, "encoder"):
        self._log("[WARN] No encoder.layers found in ssl_model")
        return

    enc_layers = list(ssl_model.encoder.layers)
    if num_layers == "all":
        for p in ssl_model.parameters():
            p.requires_grad = True
        self._log("[UNFREEZE] All encoder layers unfrozen.")
    else:
        n = int(num_layers)
        for l in enc_layers[-n:]:
            for p in l.parameters():
                p.requires_grad = True
        self._log(f"[UNFREEZE] Top {n} transformer layers unfrozen.")

    self._log_trainable_params()  # optional verification


def _log_trainable_params(self):
    total = sum(p.numel() for p in self.modules.ssl_model.parameters())
    trainable = sum(p.numel() for p in self.modules.ssl_model.parameters() if p.requires_grad)
    self._log(f"[INFO] Trainable encoder params: {trainable}/{total} ({100 * trainable / total:.2f}%)")

def _save_unfreeze_state(obj, path):
    """obj is UnfreezeState instance; path is file path SB expects."""
    torch.save(obj.state_dict(), path)  # write dictionary only

def _load_unfreeze_state(obj, path, end_of_epoch):
    """obj: UnfreezeState instance; path: file to load; end_of_epoch ignored."""
    state = torch.load(path, map_location="cpu")
    obj.load_state_dict(state)