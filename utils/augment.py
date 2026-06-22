"""
augment.py — SpeechBrain-friendly waveform augmentation utilities.

Includes:
• RIR convolution with early reflection trimming
• Noise addition (SNR-controlled)
• Time masking
• Speed perturbation
• Band-limiting (biquad)
• RMS/anti-clip normalization
• Orchestration class: WaveformAugmenter
"""

import os, math, glob, inspect
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

# -------------------------------------------------------------------------
# Shape helpers
# -------------------------------------------------------------------------
def _ensure_bt(wavs):
    squeeze_back = False
    if wavs.ndim == 3 and wavs.shape[1] == 1:
        wavs = wavs.squeeze(1)
        squeeze_back = True
    return wavs, squeeze_back

def _restore_shape(wavs, squeeze_back):
    return wavs.unsqueeze(1) if squeeze_back else wavs


# -------------------------------------------------------------------------
# RIR utilities
# -------------------------------------------------------------------------
def _apply_per_sample(wavs, lens, fn):
    B, T = wavs.shape
    out = torch.zeros_like(wavs)
    for i in range(B):
        Li = int(lens[i].item() * T) if lens is not None else T
        xo = fn(wavs[i, :Li], Li)
        out[i, :xo.numel()] = xo
    return out

def _align_trim_rir(rir: torch.Tensor, sr: int,
                    onset_db_drop: float = 20.0, trim_after_ms: float = 80.0):
    """Trim RIR to first arrival, keeping early reflections only."""
    rir = rir - rir.mean()
    peak = rir.abs().max()
    if peak <= 0:
        return rir
    thr = peak * (10 ** (-onset_db_drop / 20.0))
    nz = torch.nonzero(rir.abs() >= thr, as_tuple=False)
    start = int(nz[0]) if nz.numel() else 0
    rir = rir[start:]
    maxL = int(sr * (trim_after_ms / 1000.0))
    return rir[:maxL]


def _load_audio_mono(path: str, target_sr: int | None = None):
    """Load audio robustly (torchaudio first, soundfile fallback), return mono float32."""
    wav, sr = None, None

    # Primary path (fast, native in most envs)
    try:
        wav, sr = torchaudio.load(path)
    except Exception as e_ta:
        # Fallback path for environments where torchaudio requires torchcodec
        # and torchcodec isn't installed.
        try:
            import soundfile as sf

            data, sr = sf.read(path, always_2d=True)
            if not np.issubdtype(data.dtype, np.floating):
                info = np.iinfo(data.dtype)
                scale = float(max(abs(info.min), info.max))
                data = data.astype(np.float32) / max(scale, 1.0)
            else:
                data = data.astype(np.float32)

            # soundfile -> [T, C], torchaudio convention -> [C, T]
            wav = torch.from_numpy(data.T)
        except Exception as e_sf:
            raise RuntimeError(
                f"torchaudio.load failed ({e_ta}); soundfile fallback failed ({e_sf})"
            ) from e_sf

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    wav = wav.to(torch.float32)
    wav_mono = wav.mean(0)

    if target_sr is not None and int(sr) != int(target_sr):
        wav_mono = torchaudio.functional.resample(wav_mono, int(sr), int(target_sr))
        sr = int(target_sr)

    return wav_mono, int(sr)


class RIRBank:
    def __init__(self, paths, target_sr, device, dtype,
                 preload=True, max_files=256, onset_db_drop=20.0, trim_after_ms=80.0):
        self.paths = list(paths or [])
        self.sr = int(target_sr)
        self.device, self.dtype = device, dtype
        self.onset_db_drop = float(onset_db_drop)
        self.trim_after_ms = float(trim_after_ms)
        self.kernels = []
        # ----- Preload all RIRs, record stats -----
        failed, loaded = 0, 0
        if preload and self.paths:
            for p in self.paths[:max_files]:
                try:
                    rir, _ = _load_audio_mono(p, target_sr=self.sr)
                    rir = _align_trim_rir(rir, self.sr, self.onset_db_drop, self.trim_after_ms)
                    self.kernels.append(rir.to(self.device, self.dtype))
                    loaded += 1
                except Exception as e:
                    print(f"[RIRBank] Failed to load RIR {p}: {e}", flush=True)
                    failed += 1

        # ----- Log diagnostic message -----
        msg = (f"[RIRBank] Preloaded {loaded}/{len(self.paths)} RIRs "
            f"(failed: {failed}); max_files={max_files}, trim={self.trim_after_ms} ms, "
            f"onset_drop={self.onset_db_drop} dB")

        # Integrate with SpeechBrain logger if present
        try:
            import inspect
            caller_frame = inspect.stack()[1]
            caller_locals = caller_frame.frame.f_locals
            hparams = caller_locals.get("hparams") or caller_locals.get("self")
            logger = getattr(hparams, "train_logger", None)
            if logger is not None and hasattr(logger, "write"):
                logger.write(msg + "\n")
            else:
                print(msg, flush=True)
        except Exception:
            print(msg, flush=True)

    def sample(self):
        if self.kernels:
            idx = int(torch.randint(0, len(self.kernels), (1,), device=self.kernels[0].device).item())
            return self.kernels[idx]
        # fallback: on-demand load (rare once preloaded)
        if not self.paths:
            return None
        import torchaudio
        j = int(torch.randint(0, len(self.paths), (1,)).item())
        path = self.paths[j]
        try:
            rir, _ = _load_audio_mono(path, target_sr=self.sr)
            rir = _align_trim_rir(rir, self.sr, self.onset_db_drop, self.trim_after_ms)
            return rir.to(self.device, self.dtype)
        except Exception:
            return None

def apply_rir_convolution(wavs, lens=None, rir_bank=None, rir_scale=0.3):
    """
    Per-sample RIR, aligned & early-only, correct grouped conv, RMS-preserving, no onset shift.
    Input wavs can be [B,T] or [B,1,T]; output is always [B,T].
    """
    wavs, sb = _ensure_bt(wavs)                       # wavs: [B, T]
    if rir_bank is None or not getattr(rir_bank, "paths", None):
        return _restore_shape(wavs, sb)

    B, T = wavs.shape

    # collect kernels
    kernels, maxL = [], 1
    for _ in range(B):
        r = rir_bank.sample()
        if r is None:
            r = wavs.new_tensor([1.0])               # identity impulse
        else:
            r = _align_trim_rir(r, sr=rir_bank.sr)
            r = r * float(rir_scale)
        kernels.append(r)
        maxL = max(maxL, r.numel())

    # weights [B,1,L], input as [1,B,T] -> depthwise conv with groups=B
    weight = wavs.new_zeros(B, 1, maxL)
    for i, r in enumerate(kernels):
        weight[i, 0, :r.numel()] = r

    # ---- CORRECT SHAPES (no transpose anywhere) ----
    x = wavs.unsqueeze(0)                             # [1, B, T]
    y = F.conv1d(x, weight, padding=maxL - 1, groups=B)  # [1, B, T+L-1]
    y = y[:, :, :T].squeeze(0)                        # [B, T]  <-- keep [B,T]

    # RMS preserve
    rms_in  = wavs.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    rms_out = y.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
    y = y * (rms_in / rms_out)

    # Always return [B,T] or [B,1,T] matching input rank
    return _restore_shape(torch.clamp(y, -1.0, 1.0), sb)



# -------------------------------------------------------------------------
# Augmentation primitives
# -------------------------------------------------------------------------
def add_noise_snr(wavs, lens=None, snr_db_min=5, snr_db_max=25):
    wavs, sb = _ensure_bt(wavs)
    def _one(xi, Li):
        ps = (xi[:Li]**2).mean() + 1e-8
        snr_db = float(torch.empty((), device=xi.device).uniform_(snr_db_min, snr_db_max).item())
        pn = ps / (10 ** (snr_db / 10))
        noise = torch.randn_like(xi[:Li]) * math.sqrt(pn)
        xo = xi.clone()
        xo[:Li] = xi[:Li] + noise
        return xo[:Li]
    out = _apply_per_sample(wavs, lens, _one)
    return _restore_shape(out, sb)

def time_mask_waveform(wavs, lens=None, max_mask_pct=0.1, num_masks=2):
    wavs, sb = _ensure_bt(wavs)
    def _one(xi, Li):
        xo = xi.clone()
        if Li <= 1: return xo
        Lmax = max(1, int(Li * max_mask_pct))
        for _ in range(num_masks):
            mlen = int(torch.randint(1, Lmax + 1, (1,), device=xi.device).item())
            start = int(torch.randint(0, max(1, Li - mlen + 1), (1,), device=xi.device).item())
            xo[start:start+mlen] = 0.0
        return xo
    out = _apply_per_sample(wavs, lens, _one)
    return _restore_shape(out, sb)


# ---------- helpers ----------
def _match_length(y: torch.Tensor, T_ref: int, center: bool = True) -> torch.Tensor:
    """Crop/pad each 1D sample in y to T_ref (y: [B,T])."""
    B, T = y.shape
    if T == T_ref:
        return y
    if T > T_ref:
        if center:
            start = (T - T_ref) // 2
        else:
            start = 0
        return y[:, start:start+T_ref]
    out = y.new_zeros(B, T_ref)
    out[:, :T] = y
    return out

# ---------- Speed perturbation (one resample, keep_len=True) ----------
def speed_perturb_batch(x: torch.Tensor, sr: int, factor: float, keep_len: bool = True) -> torch.Tensor:
    """
    x: [B,T]; factor>1 -> faster/shorter; factor<1 -> slower/longer
    Performs a single resample, then crops/pads back to T if keep_len=True.
    No peak normalization, no RMS override (preserves original amplitude/prosodic cues).
    """
    B, T = x.shape
    # For speed change, resample to sr/factor (e.g., 1.05 => sr/1.05)
    new_sr = max(1000, int(round(sr / max(factor, 1e-6))))
    y_list = []
    for i in range(B):
        y_i = AF.resample(x[i], sr, new_sr)
        y_list.append(y_i)
    max_len = max(t.numel() for t in y_list)
    y = x.new_zeros(B, max_len)
    for i, yi in enumerate(y_list):
        y[i, :yi.numel()] = yi
    if keep_len:
        y = _match_length(y, T_ref=T, center=True)
    return torch.clamp(y, -1.0, 1.0)

# ---------- Band-limit (HP/LP) with RMS preserve + optional dry ---------
def bandlimit_batch(
    x: torch.Tensor,
    sr: int,
    low_hz: float | None,
    high_hz: float | None,
    dry: float,
    rms_preserve: bool = True,
    **_ignore,  # absorb extra kwargs (e.g., lens)
):
    """
    x: (B, T) waveform, any dtype/device
    sr: sample rate (Hz)
    low_hz, high_hz: band edges (Hz) — None allowed
    dry: 0..1, out = dry*x + (1-dry)*bandlimited(x)
    rms_preserve: match output RMS to input RMS per sample
    """
    assert x.dim() == 2, f"Expected (B, T), got {tuple(x.shape)}"
    orig_device, orig_dtype = x.device, x.dtype
    
    # ---- 1) Use same device as input; cast to float32 for filtering ----
    x32 = x.detach().to(dtype=torch.float32)
    B, T = x32.shape

    # ---- 2) Sanitize edges (tolerate None) ----
    nyq = 0.5 * float(sr)
    # sensible defaults if None
    lo = 20.0 if (low_hz is None) else float(low_hz)
    hi = nyq * 0.95 if (high_hz is None) else float(high_hz)
    hi = min(hi, nyq * 0.95)
    lo = max(20.0, min(lo, hi - 1.0))

    # ---- 3) Disable autocast explicitly during filtering ----
    device_type = "cuda" if x32.is_cuda else "cpu"
    try:
        cm = torch.autocast(device_type=device_type, enabled=False)
    except TypeError:
        from contextlib import nullcontext
        cm = nullcontext()

    with cm:
        y = x32.clone()
        for i in range(B):
            yi = y[i].unsqueeze(0)  # (1, T)
            if lo > 0.0:
                yi = AF.highpass_biquad(yi, sample_rate=sr, cutoff_freq=lo)
            if hi < nyq:
                yi = AF.lowpass_biquad(yi, sample_rate=sr, cutoff_freq=hi)
            y[i] = yi.squeeze(0)

    # ---- 4) Dry/wet mix + optional RMS preserve (on CPU float32) ----
    out = dry * x32 + (1.0 - dry) * y

    if rms_preserve:
        eps = 1e-8
        rin  = torch.sqrt(torch.mean(x32**2, dim=1) + eps)   # (B,)
        rout = torch.sqrt(torch.mean(out**2, dim=1) + eps)   # (B,)
        scale = (rin / torch.clamp(rout, min=eps)).view(B, 1)
        out = out * scale

    # ---- 5) Return to original device/dtype ----
    return out.to(device=orig_device, dtype=orig_dtype)


# -------------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------------
class WaveformAugmenter:
    """SpeechBrain-friendly waveform augmentation orchestrator."""
    def __init__(self, hparams, device, dtype=torch.float32):
        # print("WaveformAugmenter: initializing with hparams", hparams)
        self.cfg = getattr(hparams, "augmentation", {}) or {}
        self.allow_combo = False
        self.sample_rate = int(getattr(hparams, "sample_rate", 16000))
        self.device, self.dtype = device, dtype

        # -------- discover RIR files (robust: explicit list, top-level dir, or nested dir) --------
        rir_files = []
        rir_dir = None
        # 1) explicit list (highest priority)
        if getattr(hparams, "rir_files", None):
            rir_files = list(hparams.rir_files)
        else:
            # Accept any of these:
            #  - augmentation["rir_dir"]
            #  - augmentation["rir"]["rir_dir"]
            #  - hparams.rir_folder (legacy)
            rir_dir = (
                self.cfg.get("rir_dir")
                or self.cfg.get("rir", {}).get("rir_dir")
                or getattr(hparams, "rir_folder", None)
            )
            if rir_dir:
                rir_dir = os.path.expanduser(str(rir_dir)).strip()
                if os.path.isdir(rir_dir):
                    # recursive + case-insensitive
                    pats = [
                        os.path.join(rir_dir, "**", "*.wav"),
                        os.path.join(rir_dir, "**", "*.WAV"),
                    ]
                    rir_files = sorted({p for pat in pats for p in glob.glob(pat, recursive=True)})

        # -------- read trim/onset from YAML (supports nested under augmentation.rir) --------
        rir_cfg = self.cfg.get("rir", {})  # YAML may provide RIR settings here
        trim_ms  = float(self.cfg.get("rir_trim_after_ms",  rir_cfg.get("rir_trim_after_ms",  80.0)))
        onset_db = float(self.cfg.get("rir_onset_db_drop",  rir_cfg.get("rir_onset_db_drop", 20.0)))

        # -------- construct RIRBank, only pass extended args if supported --------
        bank_kwargs = {}
        try:
            sig = inspect.signature(RIRBank.__init__)
            if "preload" in sig.parameters:
                bank_kwargs.update(dict(preload=True, max_files=256,
                                        onset_db_drop=onset_db, trim_after_ms=trim_ms))
        except Exception:
            # If introspection fails, fall back to minimal signature.
            pass

        self.rir_bank = RIRBank(rir_files, self.sample_rate, device, dtype, **bank_kwargs)

        # Optional logger (older SpeechBrain versions: may be None)
        self.logger = getattr(hparams, "train_logger", None)

        # Log one concise RIR discovery summary.
        try:
            msg = f"[AUG] RIR discovered: {len(rir_files)} file(s); trim={trim_ms}ms, onset_drop={onset_db}dB"
            if rir_files:
                msg += f"; e.g., {rir_files[0]}"
            elif rir_dir:
                msg += f"; rir_dir={rir_dir}"
            if self.logger is not None and hasattr(self.logger, "write"):
                self.logger.write(msg + "\n")
            else:
                print(msg, flush=True)
            if rir_dir and (not os.path.isdir(rir_dir)):
                warn = f"[AUG][RIR][WARN] rir_dir does not exist: {rir_dir}"
                if self.logger is not None and hasattr(self.logger, "write"):
                    self.logger.write(warn + "\n")
                else:
                    print(warn, flush=True)
            elif rir_dir and os.path.isdir(rir_dir) and not rir_files:
                warn = f"[AUG][RIR][WARN] rir_dir exists but no .wav files found: {rir_dir}"
                if self.logger is not None and hasattr(self.logger, "write"):
                    self.logger.write(warn + "\n")
                else:
                    print(warn, flush=True)
        except Exception:
            pass

    def update_config(self, new_cfg, allow_combo=False):
        """
        Update augmentation config dynamically (used by scheduler each epoch).
        """
        self.cfg = dict(new_cfg)
        self.allow_combo = bool(allow_combo)


    def __call__(self, wavs, lens=None, training=True):
        """Apply probabilistic waveform augmentations."""
        if not training or not self.cfg:
            return wavs

        wavs_dtype, wavs_device = wavs.dtype, wavs.device
        aug_applied = []  # record which ones were used this call
        # If combination is disabled, select at most ONE augmentation with p>0
        if not getattr(self, "allow_combo", False):
            active = [k for k,v in self.cfg.items() if float(v.get("p", 0.0)) > 0.0]
            if len(active) > 1:
                idx = torch.randint(0, len(active), (1,), device=wavs.device).item()
                chosen = active[idx]
                # Zero out all others
                for k in self.cfg.keys():
                    if k != chosen:
                        self.cfg[k]["p"] = 0.0

        # ---- SPEED perturbation ----
        sp_cfg = self.cfg.get("speed", {})
        if sp_cfg and (torch.rand(()) < float(sp_cfg.get("p", 0))):
            fmin, fmax = float(sp_cfg.get("min", 0.80)), float(sp_cfg.get("max", 1.20))
            factor = float(torch.empty(()).uniform_(fmin, fmax).item())
            wavs_bt, sb = _ensure_bt(wavs)
            wavs_bt = speed_perturb_batch(wavs_bt, sr=self.sample_rate, factor=factor, keep_len=True)
            wavs = _restore_shape(wavs_bt, sb)
            aug_applied.append(f"speed({factor:.3f})")

        # ---- RIR convolution ----
        rir_cfg = self.cfg.get("rir", {})
        if rir_cfg and (torch.rand(()) < float(rir_cfg.get("p", 0))):
            wavs = apply_rir_convolution(
                wavs, lens=lens, rir_bank=self.rir_bank, rir_scale=float(rir_cfg.get("scale", 0.3))
            )
            aug_applied.append("rir")

        # ---- BANDLIMIT ----
        bl_cfg = self.cfg.get("bandlimit", {})
        if bl_cfg and (torch.rand(()) < float(bl_cfg.get("p", 0))):
            wavs_bt, sb = _ensure_bt(wavs)
            wavs_bt = bandlimit_batch(
                wavs_bt, sr=self.sample_rate,
                low_hz=bl_cfg.get("low_hz", None),
                high_hz=bl_cfg.get("high_hz", None),
                dry=float(bl_cfg.get("dry", 0.2)),
                rms_preserve=True,
            )
            wavs = _restore_shape(wavs_bt, sb)
            aug_applied.append("bandlimit")

        # ---- NOISE ----
        n_cfg = self.cfg.get("noise", {})
        if n_cfg and (torch.rand(()) < float(n_cfg.get("p", 0))):
            wavs = add_noise_snr(
                wavs, lens=lens,
                snr_db_min=float(n_cfg.get("snr_min", 5)),
                snr_db_max=float(n_cfg.get("snr_max", 25)),
            )
            aug_applied.append("noise")

        # ---- TIME MASK ----
        tm_cfg = self.cfg.get("time_mask", {})
        if tm_cfg and (torch.rand(()) < float(tm_cfg.get("p", 0))):
            wavs = time_mask_waveform(
                wavs, lens=lens,
                max_mask_pct=float(tm_cfg.get("max_mask_pct", 0.08)),
                num_masks=int(tm_cfg.get("num_masks", 2)),
            )
            aug_applied.append("time_mask")

        # ---- Anti-clip ----
        wavs_bt, sb = _ensure_bt(wavs)
        scale = torch.clamp(1.0 / (wavs_bt.abs().amax(dim=1, keepdim=True) + 1e-8), max=1.0)
        wavs_bt = wavs_bt * scale
        wavs = _restore_shape(wavs_bt, sb)

        # ---- Restore dtype/device ----
        wavs = wavs.to(wavs_device, wavs_dtype)
        verbose = bool(self.cfg.get("verbose", False))

        # ---- Logging ----
        if verbose:
            if aug_applied:
                msg = "[AUG] applied: " + ", ".join(aug_applied)
                if self.logger is not None and hasattr(self.logger, "write"):
                    self.logger.write(msg + "\n")
                else:
                    print(msg, flush=True)
            else:
                if self.logger is not None and hasattr(self.logger, "write"):
                    self.logger.write("[AUG] none applied\n")
                else:
                    print("[AUG] none applied", flush=True)

        return wavs
