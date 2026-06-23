import torch
import numpy as np
import torch.nn as nn
import os
import sys
import json

def _hp_get(hparams, key, default=None):
    """Safe getter for both attr-style and dict-style hparams."""
    if isinstance(hparams, dict):
        return hparams.get(key, default)
    return getattr(hparams, key, default)

# SpeechBrain SSL model wrapper (path changed across SB versions)
try:
    # Newer SpeechBrain
    from speechbrain.integrations.huggingface import Wav2Vec2
except Exception:
    # Older SpeechBrain (deprecated but still works)
    from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2

# -------------------------------------------------
# ----------- Self-supervised Model ---------------
# -------------------------------------------------
def get_ssl_model(hparams):
    model_name = str(_hp_get(hparams, "pretrained_model", "wavlm")).lower()
    save_path = _hp_get(hparams, "save_folder", "./save")

    if model_name == "wavlm":
        # Prefer YAML-provided hub + folder (wavlm-base vs wavlm-large etc.)
        hub = str(_hp_get(hparams, "wavlm_hub", "microsoft/wavlm-base"))
        folder = str(_hp_get(hparams, "wavlm_folder", "")) or ""
        if not folder:
            folder = os.path.join(str(save_path), "wavlm")
        return Wav2Vec2(
            source=hub,
            save_path=folder,
            output_all_hiddens=False,
        )
    else:
        raise ValueError(
            f"Unsupported pretrained_model '{model_name}'. "
            "Only WavLM is supported in this public release."
        )
    

def concordance_cc(x, y, eps=1e-8):
    # x,y: [B,3]
    vx = x.var(dim=0, unbiased=False)
    vy = y.var(dim=0, unbiased=False)
    mx = x.mean(dim=0)
    my = y.mean(dim=0)
    cov = ((x - mx) * (y - my)).mean(dim=0)
    return (2 * cov) / (vx + vy + (mx - my).pow(2) + eps)  # [3]


def _to_float_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True).float()

    # 2) SpeechBrain "padded" containers (have .data -> Tensor)
    if hasattr(x, "data") and isinstance(x.data, torch.Tensor):
        return x.data.to(device, non_blocking=True).float()

    # 3) NumPy or other array-likes
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.as_tensor(x)
    return x.to(device, non_blocking=True).float()


def _to_long_tensor(x, device):
    t = _to_float_tensor(x, device)
    return t.long()

# =============================================================
# TC-GRU head (paper-style) for SER
# Temporal Conv (k=3) -> 2-layer GRU (256) -> embedding (256)
# =============================================================
class TCGRUHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        conv_channels: int = 256,
        conv_kernel: int = 3,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        emb_dim: int = 256,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.conv = nn.Conv1d(
            in_channels=self.in_dim,
            out_channels=int(conv_channels),
            kernel_size=int(conv_kernel),
            padding=int(conv_kernel) // 2,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=float(dropout))

        self.gru = nn.GRU(
            input_size=int(conv_channels),
            hidden_size=int(gru_hidden),
            num_layers=int(gru_layers),
            batch_first=True,
            dropout=float(dropout) if int(gru_layers) > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )

        out_dim = int(gru_hidden) * (2 if bidirectional else 1)
        # Skip connection: project conv summary to GRU hidden dim and add to h_last
        self.skip_proj = nn.Identity() if int(conv_channels) == out_dim else nn.Linear(int(conv_channels), out_dim)
        self.emb = nn.Sequential(
            nn.Linear(out_dim, int(emb_dim)),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
        )

    def _compute_h_last(self, framewise: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """framewise: [B,T,D], lens: [B] in (0,1]. Returns shared temporal state [B,H]."""
        # Conv1d expects [B,D,T]
        x = framewise.transpose(1, 2)  # [B,D,T]
        x = self.drop(self.act(self.conv(x)))
        x = x.transpose(1, 2)  # [B,T,C]

        # Pack for GRU to avoid padding waste
        # lens is fraction; convert to lengths
        B, T, _ = x.shape
        lengths_dev = (lens * T).long().clamp(min=1, max=T)     # on GPU
        lengths_cpu = lengths_dev.cpu()                         # only for pack
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths=lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, h_n = self.gru(packed)

        # Use last-layer hidden state (more stable than last frame)
        # h_n: [L*(dir), B, H]
        if self.gru.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [B,2H]
        else:
            h_last = h_n[-1]  # [B,H]

        # ---- Skip connection from conv output ----
        # x is currently conv output after transpose back: [B, T, C]
        # lengths is already computed: [B] on CPU
        lengths_dev = lengths_dev.to(x.device)
        T = x.size(1)
        mask = (torch.arange(T, device=x.device)[None, :] < lengths_dev[:, None]).to(x.dtype)  # [B,T]
        denom = lengths_dev.clamp_min(1).unsqueeze(1).float()  # [B,1]
        conv_mean = (x * mask.unsqueeze(-1)).sum(dim=1) / lengths_dev.clamp_min(1).unsqueeze(1).to(x.dtype)  # [B,C]

        h_last = h_last + self.skip_proj(conv_mean)
        return h_last

    def forward_shared(self, framewise: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """Return shared TC-GRU embedding right before the final dropout."""
        h_last = self._compute_h_last(framewise=framewise, lens=lens)
        if (
            isinstance(self.emb, nn.Sequential)
            and len(self.emb) >= 3
            and isinstance(self.emb[-1], nn.Dropout)
        ):
            x = h_last
            for layer in list(self.emb)[:-1]:
                x = layer(x)
            return x
        # Fallback for custom emb blocks without an explicit trailing dropout.
        return self.emb(h_last)

    def forward(self, framewise: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """framewise: [B,T,D], lens: [B] in (0,1]. Returns [B,emb_dim]."""
        h_last = self._compute_h_last(framewise=framewise, lens=lens)
        return self.emb(h_last)
    
def save_model_info(brain, save_dir=None):
    """Write a lightweight reproducibility snapshot for this run.

    This is a standalone utility (not a Brain method). We pass `brain` explicitly
    to avoid relying on `self` in module-level helpers.

    Saves:
      - module reprs (SSL, pooling, heads)
      - basic environment versions
      - key hparams (class order, dist mode, loss config)

    Never crashes training.
    """
    try:
        hps = getattr(brain, "hparams", None)

        def _hp_get(hp, key, default=None):
            if hp is None:
                return default
            # HyperPyYAML returns a dict-like object in many recipes
            if isinstance(hp, dict):
                return hp.get(key, default)
            return getattr(hp, key, default)

        # Resolve save directory
        if save_dir is None:
            save_dir = _hp_get(hps, "save_folder", None)
        if save_dir is None:
            save_dir = _hp_get(hps, "output_folder", "./save")
        save_dir = str(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        arch_path = os.path.join(save_dir, "model_architecture.txt")

        def _safe_write(fh, title: str, obj):
            fh.write(f"=== {title} ===\n")
            try:
                fh.write(repr(obj) + "\n\n")
            except Exception:
                try:
                    fh.write(str(obj) + "\n\n")
                except Exception as e:
                    fh.write(f"[WARN] could not stringify {title}: {e}\n\n")

        modules = getattr(brain, "modules", {})
        ssl_obj = getattr(brain, "ssl_model", None)
        if ssl_obj is None and isinstance(modules, dict):
            ssl_obj = modules.get("ssl_model", None)

        # Resolve the optional TC-GRU module from `brain` or `brain.modules`.
        tcgru_obj = getattr(brain, "tc_gru_head", None)
        if tcgru_obj is None:
            tcgru_obj = getattr(brain, "tcgru_head", None)
        if tcgru_obj is None and isinstance(modules, dict):
            tcgru_obj = (
                modules.get("tc_gru_head")
                or modules.get("tcgru_head")
                or modules.get("tc_gru")
                or modules.get("tcgru")
            )

        with open(arch_path, "w") as f:
            # ---- versions / environment ----
            f.write("=== ENV ===\n")
            f.write(f"python={sys.version.split()[0]}\n")
            f.write(f"torch={torch.__version__}\n")
            try:
                import speechbrain

                f.write(f"speechbrain={getattr(speechbrain, '__version__', 'NA')}\n")
            except Exception as e:
                f.write(f"[INFO] speechbrain version unavailable: {e}\n")
            try:
                import transformers

                f.write(f"transformers={getattr(transformers, '__version__', 'NA')}\n")
            except Exception as e:
                f.write(f"[INFO] transformers version unavailable: {e}\n")
            f.write("\n")

            # ---- key run metadata ----
            f.write("=== RUN ===\n")
            f.write(f"device={getattr(brain, 'device', 'NA')}\n")
            f.write(f"output_folder={_hp_get(hps, 'output_folder', 'NA')}\n")
            f.write(f"save_folder={_hp_get(hps, 'save_folder', save_dir)}\n")
            f.write(f"pretrained_model={_hp_get(hps, 'pretrained_model', 'NA')}\n")
            f.write(f"dist_mode={_hp_get(hps, 'dist_mode', 'NA')}\n")
            f.write(f"use_soft_targets={_hp_get(hps, 'use_soft_targets', 'NA')}\n")

            loss_cfg = _hp_get(hps, "loss", None)
            if loss_cfg is not None:
                # loss_cfg might be dict-like or an object
                if isinstance(loss_cfg, dict):
                    f.write(f"loss.type={loss_cfg.get('type', 'NA')}\n")
                    f.write(f"loss.epsilon={loss_cfg.get('epsilon', 'NA')}\n")
                else:
                    f.write(f"loss.type={getattr(loss_cfg, 'type', 'NA')}\n")
                    f.write(f"loss.epsilon={getattr(loss_cfg, 'epsilon', 'NA')}\n")
            f.write("\n")

            # ---- class order (if already available) ----
            class_names = getattr(brain, "class_names", None)
            if class_names is not None:
                f.write("=== CLASSES ===\n")
                try:
                    f.write("class_names=" + json.dumps(list(class_names)) + "\n\n")
                except Exception:
                    f.write("class_names=" + str(class_names) + "\n\n")

            # ---- modules ----
            _safe_write(f, "SSL MODEL", ssl_obj)
            _safe_write(f, "TCGRU HEAD", tcgru_obj)

            if isinstance(modules, dict):
                _safe_write(f, "VAD MLP", modules.get("vad_mlp", None))
                _safe_write(f, "CAT MLP", modules.get("cat_mlp", None))
            else:
                # Some SpeechBrain versions use ModuleDict-like containers
                _safe_write(f, "VAD MLP", getattr(modules, "vad_mlp", None))
                _safe_write(f, "CAT MLP", getattr(modules, "cat_mlp", None))

            # ---- optional sampler class (if present) ----
            f.write("=== SUBSAMPLER (optional) ===\n")
            try:
                from utils.subsampler import BalancedSubsetPerEpochSampler

                f.write(repr(BalancedSubsetPerEpochSampler) + "\n\n")
            except Exception as e:
                f.write(f"[INFO] no BalancedSubsetPerEpochSampler available: {e}\n\n")

        # Log if the brain provides a logger
        try:
            log_fn = getattr(brain, "_log", None)
            if callable(log_fn):
                log_fn(f"Model architecture saved to {arch_path}", tag="INFO")
        except Exception:
            pass

        return arch_path

    except Exception as e:
        # Never crash training for this
        try:
            log_fn = getattr(brain, "_log", None)
            if callable(log_fn):
                log_fn(f"Failed to save model architecture: {e}", tag="WARN")
        except Exception:
            pass
        return None


def export_test_predictions(brain):
    """Save per-utterance predictions after TEST stage.

    Standalone utility (not a Brain method). Pass `brain` explicitly.

    Writes CSV if pandas is available, otherwise falls back to JSONL.
    """
    import json as _json

    # Resolve output directory
    hps = getattr(brain, "hparams", None)
    outdir = "./results"
    if hps is not None:
        outdir = getattr(hps, "output_folder", getattr(hps, "save_folder", outdir))
    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)

    stage_preds = getattr(brain, "_stage_preds", [])
    stage_tgts = getattr(brain, "_stage_tgts", [])
    stage_ids = getattr(brain, "_stage_ids", [])
    stage_cls_lp = getattr(brain, "_stage_cls_lp", [])
    stage_cls_y = getattr(brain, "_stage_cls_y", [])

    if not stage_preds:
        try:
            brain._log("[TEST] No predictions to export.")
        except Exception:
            pass
        return None

    P = torch.cat(stage_preds, dim=0)
    T = torch.cat(stage_tgts, dim=0) if stage_tgts else None

    have_cls = bool(stage_cls_y) and bool(stage_cls_lp)
    if have_cls:
        LP = torch.cat(stage_cls_lp, dim=0)
        Y = torch.cat(stage_cls_y, dim=0)
        Yhat = LP.argmax(dim=-1)
        probs = torch.exp(LP)
        class_names = list(getattr(brain, "class_names", []))
    else:
        Y = Yhat = probs = None
        class_names = []

    rows = []
    n = int(P.shape[0])
    for i in range(n):
        rid = stage_ids[i] if i < len(stage_ids) else f"utt_{i}"
        p = P[i]
        row = {
            "uttid": rid,
            "pred_V": float(p[0]),
            "pred_A": float(p[1]),
            "pred_D": float(p[2]),
        }
        if T is not None:
            t = T[i]
            row.update(
                {
                    "true_V": float(t[0]),
                    "true_A": float(t[1]),
                    "true_D": float(t[2]),
                }
            )

        if have_cls and class_names:
            yi = int(Y[i].item())
            yhi = int(Yhat[i].item())
            # Guard against mismatch between label ids and class_names
            row["pred_label"] = class_names[yhi] if 0 <= yhi < len(class_names) else str(yhi)
            row["true_label"] = class_names[yi] if 0 <= yi < len(class_names) else str(yi)
            for cidx, cname in enumerate(class_names):
                row[f"prob_{cname}"] = float(probs[i, cidx])
            row["prob_vector"] = [float(x) for x in probs[i].tolist()]
            row["class_order"] = list(class_names)

        rows.append(row)

    # Prefer CSV if pandas exists; otherwise JSONL
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        csv_path = os.path.join(outdir, "test_predictions.csv")
        df.to_csv(csv_path, index=False)
        try:
            brain._log(f"[TEST] Saved predictions to {csv_path}")
        except Exception:
            pass
        return csv_path

    except Exception as e:
        jpath = os.path.join(outdir, "test_predictions.jsonl")
        with open(jpath, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(_json.dumps(row) + "\n")
        try:
            brain._log(f"[TEST] Wrote JSONL predictions to {jpath}. (CSV unavailable: {e})")
        except Exception:
            pass
        return jpath
