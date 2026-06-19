# =========================================================
# dataio prep
# =========================================================

import os
import math
import json
import subprocess
import torch
import numpy as np
import soundfile as sf
import speechbrain as sb

def dataio_prep(hparams):
    """Prepare datasets for VAD regression + categorical (sig, vad, y_idx, emo_dist)."""
    save_folder = hparams["save_folder"]
    os.makedirs(save_folder, exist_ok=True)

    # --------- helpers (local; minimal) ----------
    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    def _scale_vad(v, a, d, mode):
        # Keep your existing scaler behavior; extend minimally
        if mode == "neg1_1":
            def s(u): 
                val = (u - 4.0) / 2.0
                return max(-1.0, min(1.0, val))
            return torch.tensor([s(v), s(a), s(d)], dtype=torch.float32)
        elif mode == "zscore" and "vad_stats" in hparams:
            mu = hparams["vad_stats"]["mean"]  # [3]
            sd = hparams["vad_stats"]["std"]   # [3]
            return torch.tensor([(v - mu[0]) / (sd[0] + 1e-8),
                                 (a - mu[1]) / (sd[1] + 1e-8),
                                 (d - mu[2]) / (sd[2] + 1e-8)], dtype=torch.float32)
        else:
            # raw 1..7
            return torch.tensor([v, a, d], dtype=torch.float32)
            
    def _scale_vad_var(vv, va, vd, mode):
        """Scale VAD variances to match the target scaling used for V/A/D."""
        if mode == "neg1_1":
            # x'=(x-4)/2 => var' = var / 4
            return torch.tensor([vv / 4.0, va / 4.0, vd / 4.0], dtype=torch.float32)
        elif mode == "zscore" and "vad_stats" in hparams:
            sd = hparams["vad_stats"]["std"]
            return torch.tensor([
                vv / ((sd[0] + 1e-8) ** 2),
                va / ((sd[1] + 1e-8) ** 2),
                vd / ((sd[2] + 1e-8) ** 2),
            ], dtype=torch.float32)
        else:
            return torch.tensor([vv, va, vd], dtype=torch.float32)

    def _scale_vad_std(sv, sa, sd_, mode):
        """Scale VAD std-devs to match the target scaling used for V/A/D."""
        if mode == "neg1_1":
            # x'=(x-4)/2 => std' = std / 2
            return torch.tensor([sv / 2.0, sa / 2.0, sd_ / 2.0], dtype=torch.float32)
        elif mode == "zscore" and "vad_stats" in hparams:
            base_sd = hparams["vad_stats"]["std"]
            return torch.tensor([
                sv / (base_sd[0] + 1e-8),
                sa / (base_sd[1] + 1e-8),
                sd_ / (base_sd[2] + 1e-8),
            ], dtype=torch.float32)
        else:
            return torch.tensor([sv, sa, sd_], dtype=torch.float32)
    
    # Canonical emotion normalization map
    CANONICAL_MAP = {
        "neutral": "neu", "neutrality": "neu", "none":"neu", "calm":"neu",
        "happy": "hap", "happiness": "hap", "joy": "hap",
        "angry": "ang", "anger": "ang", "frustrated":"ang", "annoyed":"ang", 
        "sad": "sad", "sadness": "sad", "depressed": "sad", "concerned":"sad",
        "disgusted": "disgust", "disgusting": "disgust",
        "fear": "fear", "afraid": "fear", "scared": "fear",
        "contempt": "contempt", "scorn": "contempt",
        "surprise": "surprise", "surprised": "surprise",
    }
    ORDERED_CLASSES = ["neu", "hap", "sad", "ang", "fear", "disgust", "contempt", "surprise", "other"]

    def _dist_to_vec(dist, lab2ind, C):
        vec = torch.zeros(C, dtype=torch.float32)
        if isinstance(dist, dict) and dist:
            s = 0.0
            for k, p in dist.items():
                k = str(k).lower().strip()
                k = CANONICAL_MAP.get(k, k)
                if k in lab2ind:
                    val = float(p)
                    if val > 0:
                        vec[lab2ind[k]] = val
                        s += val
            if abs(s - 1.0) > 1e-4:
                vec /= (s + 1e-8)
            vec = torch.clamp(vec, 0.0, 1.0)
        return vec

    def _is_neutral_label(lbl):
        """Heuristic neutral detector; override via hparams['neutral_labels'] if provided."""
        if not lbl:
            return False
        lbl = str(lbl).strip().lower()
        if "neutral" in lbl or lbl in {"neu", "n", "none", "neutral", "calm"}:
            return True
        return False

    def label_to_onehot(emo, lab2ind, C):
        lab = (emo or "").lower().strip()
        lab = CANONICAL_MAP.get(lab, lab)
        if lab not in lab2ind and  _is_neutral_label(lab):
            lab = "neu"
        idx = lab2ind.get(lab, lab2ind.get("other", 0))
        vec = torch.zeros(C, dtype=torch.float32)
        vec[int(idx)] = 1.0
        return vec

    # --------- Canonical emotion class space ---------
    # Goal: align class space with prepared JSONs (supports 9-class MSP-Podcast).
    def _infer_classes_from_json(json_path):
        """Infer emotion set from a JSON annotation file."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            emos = set()
            entries = data.values() if isinstance(data, dict) else data if isinstance(data, list) else []
            for item in entries:
                if isinstance(item, dict) and "emo" in item:
                    emo = str(item["emo"]).lower().strip()
                    if emo:
                        emos.add(emo)
            return list(emos)
        except Exception:
            return []

    def _load_label_mapping(json_path):
        """Load class order from label_mapping.json (written by prepare script)."""
        try:
            lm_path = os.path.join(os.path.dirname(json_path), "label_mapping.json")
            if os.path.exists(lm_path):
                with open(lm_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                if isinstance(m, dict) and "class_order" in m:
                    return [str(c).lower().strip() for c in m["class_order"]]
        except Exception:
            pass
        return None

    ncls = hparams.get("out_n_neurons_cls", None)
    class_names = hparams.get("emo_classes", None)
    inferred_classes = []
    mapping_classes = None

    if not class_names:
        train_ann = hparams.get("train_annotation", None)
        if train_ann and os.path.exists(train_ann):
            mapping_classes = _load_label_mapping(train_ann)
            if not mapping_classes:
                inferred_classes = _infer_classes_from_json(train_ann)

    # If user explicitly provided class names, trust them.
    if isinstance(class_names, (list, tuple)) and len(class_names) > 0:
        canonical_classes = [str(x).lower().strip() for x in class_names]

    elif mapping_classes:
        canonical_classes = mapping_classes
        hparams["emo_classes"] = canonical_classes
        hparams["out_n_neurons_cls"] = len(canonical_classes)

    elif inferred_classes:
        # Order by preferred ORDERED_CLASSES, then any leftovers alphabetically
        ordered = [c for c in ORDERED_CLASSES if c in inferred_classes]
        leftovers = [c for c in inferred_classes if c not in ORDERED_CLASSES]
        canonical_classes = ordered + sorted(leftovers)
        hparams["emo_classes"] = canonical_classes
        hparams["out_n_neurons_cls"] = len(canonical_classes)

    # Explicit 4-class (major emotions only)
    elif (ncls == 4) or bool(hparams.get("major4_only", False)):
        canonical_classes = ["neu", "hap", "ang", "sad"]

    # Default legacy behavior (6-class) if nothing else is specified/inferred
    else:
        canonical_classes = ["neu", "hap", "ang", "sad", "disgust", "other"]

    hparams["emo_classes"] = canonical_classes

    # Create encoder using canonical mapping
    enc = sb.dataio.encoder.CategoricalEncoder()
    enc.update_from_iterable(canonical_classes)
    enc.expect_len(len(canonical_classes))
    hparams["label_encoder"] = enc
    # Default to hard labels unless explicitly overridden.
    hparams.setdefault("dist_mode", "hard")

    # --------- Pipelines ----------
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """
        Ultra-robust universal audio loader:
        - soundfile first (fastest, lossless)
        - librosa fallback (no torchcodec)
        - ffmpeg fallback (universal)
        - NEVER uses torchaudio.load for FLAC (Mac fails)
        """
        target_sr = 16000
        from pathlib import Path

        def _resample_linear(audio_np, orig_sr, new_sr):
            """Dependency-free 1D linear resampling fallback."""
            if int(orig_sr) == int(new_sr):
                return audio_np.astype(np.float32, copy=False)
            n_in = int(audio_np.shape[0])
            if n_in <= 1:
                return audio_np.astype(np.float32, copy=False)
            n_out = max(1, int(round(n_in * float(new_sr) / float(orig_sr))))
            x_in = np.linspace(0.0, 1.0, num=n_in, endpoint=False, dtype=np.float64)
            x_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
            return np.interp(x_out, x_in, audio_np.astype(np.float64)).astype(np.float32, copy=False)

        wav = Path(wav)
        if not wav.exists():
            # If JSON contains absolute paths from another machine, fall back to data_folder + basename
            wav = Path(hparams["data_folder"]) / wav.name
        wav = str(wav)
        # ----------------------------
        # 1) Try soundfile
        # ----------------------------
        try:
            audio, sr = sf.read(wav, always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != target_sr:
                audio = _resample_linear(np.asarray(audio), sr, target_sr)
            return torch.from_numpy(audio.astype("float32"))
        except Exception:
            pass

        # ----------------------------
        # 2) Try librosa (no torchcodec)
        # ----------------------------
        try:
            import librosa
            audio, sr = librosa.load(wav, sr=target_sr, mono=True)
            return torch.from_numpy(audio.astype("float32"))
        except Exception:
            pass

        # ----------------------------
        # 3) Ultimate fallback: ffmpeg-python
        # ----------------------------
        try:
            import ffmpeg
            if not hasattr(ffmpeg, "input"):
                raise RuntimeError("ffmpeg module is not ffmpeg-python (missing .input)")
            out, _ = (
                ffmpeg
                .input(wav)
                .output("pipe:", format="f32le", ac=1, ar="16000")
                .run(capture_stdout=True, capture_stderr=True)
            )
            audio = np.frombuffer(out, dtype=np.float32)
            return torch.from_numpy(audio)
        except Exception as e_ffpy:
            # 4) Final fallback: ffmpeg CLI via subprocess
            try:
                proc = subprocess.run(
                    [
                        "ffmpeg",
                        "-v", "error",
                        "-i", wav,
                        "-f", "f32le",
                        "-ac", "1",
                        "-ar", "16000",
                        "pipe:1",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                )
                audio = np.frombuffer(proc.stdout, dtype=np.float32)
                if audio.size == 0:
                    raise RuntimeError("ffmpeg CLI returned empty audio buffer")
                return torch.from_numpy(audio)
            except Exception as e_cli:
                raise RuntimeError(
                    f"❌ COMPLETE FAILURE loading {wav}: ffmpeg-python={e_ffpy}; ffmpeg-cli={e_cli}"
                )
        
    vad_mode = hparams.get("vad_target_scale", "neg1_1")

    # Accept valence/arousal/dominance keys from JSON
    @sb.utils.data_pipeline.takes("valence", "arousal", "dominance")
    @sb.utils.data_pipeline.provides("vad")
    def vad_pipeline(valence, arousal, dominance):
        v = _safe_float(valence)
        a = _safe_float(arousal)
        d = _safe_float(dominance)
        return _scale_vad(v, a, d, vad_mode)

    # --- Soft VAD targets (var/std + n_vad) from JSON ---
    @sb.utils.data_pipeline.takes("vad_var",)
    @sb.utils.data_pipeline.provides("vad_var_t")
    def vad_var_pipeline(vad_var):
        """Return VAD variance target as [3] tensor scaled consistently with `vad_mode`."""
        if isinstance(vad_var, dict) and len(vad_var) > 0:
            vv = _safe_float(vad_var.get("V", 0.0))
            va = _safe_float(vad_var.get("A", 0.0))
            vd = _safe_float(vad_var.get("D", 0.0))
            return _scale_vad_var(vv, va, vd, vad_mode)
        return torch.zeros(3, dtype=torch.float32)


    @sb.utils.data_pipeline.takes("vad_std",)
    @sb.utils.data_pipeline.provides("vad_std_t")
    def vad_std_pipeline(vad_std):
        """Return VAD std-dev target as [3] tensor scaled consistently with `vad_mode`."""
        if isinstance(vad_std, dict) and len(vad_std) > 0:
            sv = _safe_float(vad_std.get("V", 0.0))
            sa = _safe_float(vad_std.get("A", 0.0))
            sd_ = _safe_float(vad_std.get("D", 0.0))
            return _scale_vad_std(sv, sa, sd_, vad_mode)
        return torch.zeros(3, dtype=torch.float32)


    # @sb.utils.data_pipeline.takes("n_vad",)
    # @sb.utils.data_pipeline.provides("n_vad_t")
    # def n_vad_pipeline(n_vad):
    #     """Return number of worker VAD annotations as scalar tensor."""
    #     try:
    #         return torch.tensor(int(n_vad), dtype=torch.int64)
    #     except Exception:
    #         return torch.tensor(0, dtype=torch.int64)

    # --- Categorical hard label
    @sb.utils.data_pipeline.takes("emo")
    @sb.utils.data_pipeline.provides("y_idx")
    def label_pipeline(emo):
        y = enc.encode_label_torch(emo if emo is not None else "")
        return y.view(-1)


    # --- Categorical one-hot label (optional convenience; derived from y_idx)
    @sb.utils.data_pipeline.takes("y_idx")
    @sb.utils.data_pipeline.provides("emo_encoded")
    def emo_encoded_pipeline(y_idx):
        """Return [C] one-hot vector derived from y_idx.

        This is useful for code paths that expect `emo_encoded`, while keeping
        the JSONs minimal (hard JSONs only need `emo`).
        """
        C = len(enc.lab2ind)
        base = getattr(y_idx, "data", y_idx)
        y = torch.as_tensor(base).view(-1).long()
        vec = torch.zeros(C, dtype=torch.float32)
        if y.numel() > 0:
            yi = int(y[0].item())
            if 0 <= yi < C:
                vec[yi] = 1.0
        return vec


    # --- Dynamic categorical distribution pipeline (respects hparams.dist_mode at call time) ---
    @sb.utils.data_pipeline.takes("emo", "emo_dist_primary", "emo_dist_secondary", "emo_dist")
    @sb.utils.data_pipeline.provides("emo_vec")
    def dist_pipeline(emo, emo_dist_primary, emo_dist_secondary, emo_dist):
        """Return a [C] categorical distribution for the current mode.

        IMPORTANT: This reads `hparams['dist_mode']` at call time so label curriculum
        (hard -> primary/secondary/merged) works without rebuilding the dataset.

        Modes:
          - "hard": one-hot from `emo`
          - "primary": use `emo_dist_primary`
          - "secondary": use `emo_dist_secondary`
          - "merged": use `emo_dist`
        """
        lab2ind = enc.lab2ind
        C = len(lab2ind)

        mode = str(hparams.get("dist_mode", "hard")).lower()

        # Hard mode: strict one-hot
        if mode == "hard":
            return label_to_onehot(emo, lab2ind, C)

        # Soft modes: choose the right distribution dict
        if mode == "primary":
            dist = emo_dist_primary
        elif mode == "secondary":
            dist = emo_dist_secondary
        else:
            dist = emo_dist

        if isinstance(dist, dict) and len(dist) > 0:
            vec = _dist_to_vec(dist, lab2ind, C)
            if vec.sum() > 0:
                return vec / (vec.sum() + 1e-8)

        # Fallback: still return a valid one-hot
        return label_to_onehot(emo, lab2ind, C)

    # --- Entropy / confidence signals for curriculum scheduling (computed from emo_vec) ---
    @sb.utils.data_pipeline.takes("emo_vec")
    @sb.utils.data_pipeline.provides(
        "emo_entropy",
        "emo_entropy_norm",
        "emo_maxprob",
        "emo_margin",
        "ambiguity_signal",
    )
    def emo_conf_pipeline(emo_vec):
        """Compute entropy + confidence metrics from categorical distribution.

        Returns:
        - emo_entropy:        scalar entropy H(p)
        - emo_entropy_norm:   entropy normalized by log(C) in [0,1]
        - emo_maxprob:        max_i p_i
        - emo_margin:         top1(p) - top2(p)
        - ambiguity_signal:   single scalar in [0,1] (higher = more ambiguous)

        Notes:
        - ambiguity_signal is intentionally simple and easy to redefine later.
        - Current definition mixes normalized entropy + (1 - margin).
        """
        vec = getattr(emo_vec, "data", emo_vec)
        vec = vec.to(dtype=torch.float32)

        # Clamp and renormalize defensively
        vec = torch.clamp(vec, min=1e-8)
        vec = vec / vec.sum().clamp_min(1e-8)

        C = int(vec.numel())

        # Entropy
        ent = -(vec * torch.log(vec)).sum()
        ent_norm = ent / math.log(max(C, 2))
        ent_norm = torch.clamp(ent_norm, 0.0, 1.0)

        # Confidence metrics
        maxprob = torch.max(vec)
        if C >= 2:
            top2 = torch.topk(vec, k=2).values
            margin = top2[0] - top2[1]
        else:
            margin = torch.tensor(0.0, dtype=torch.float32)

        margin = torch.clamp(margin, 0.0, 1.0)

        # Single ambiguity scalar (easy to redefine later)
        # More ambiguous if entropy is high and margin is low.
        ambiguity = 0.5 * ent_norm + 0.5 * (1.0 - margin)
        ambiguity = torch.clamp(ambiguity, 0.0, 1.0)

        # Make scalars 1-D so SpeechBrain padding/collate doesn’t choke
        ent = ent.view(1)
        ent_norm = ent_norm.view(1)
        maxprob = maxprob.view(1)
        margin = margin.view(1)
        ambiguity = ambiguity.view(1)

        return ent, ent_norm, maxprob, margin, ambiguity

    @sb.utils.data_pipeline.takes("emo_dist")
    @sb.utils.data_pipeline.provides("emo_entropy_norm_merged")
    def merged_entropy_pipeline(emo_dist):
        """Entropy normalized to [0,1] from the merged annotation distribution."""
        C = len(enc.lab2ind)
        vec = _dist_to_vec(emo_dist, enc.lab2ind, C)
        if float(vec.sum().item()) <= 0:
            ent_norm = 0.0
        else:
            vec = vec / vec.sum().clamp_min(1e-8)
            ent = -(vec * torch.log(vec.clamp_min(1e-8))).sum()
            ent_norm = float((ent / math.log(max(C, 2))).item())
            ent_norm = max(0.0, min(1.0, ent_norm))
        return torch.tensor([ent_norm], dtype=torch.float32)
    
    # --------- Datasets ----------
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test":  hparams["test_annotation"],
    }

    dyn_items = [
        audio_pipeline,
        vad_pipeline,
        vad_var_pipeline,
        vad_std_pipeline,
        # n_vad_pipeline,
        label_pipeline,
        emo_encoded_pipeline,
        dist_pipeline,
        emo_conf_pipeline,
        merged_entropy_pipeline,
    ]
    out_keys  = [
        "id",
        "sig",
        "vad",
        "vad_var_t",
        "vad_std_t",
        # "n_vad_t",
        "y_idx",
        "emo_encoded",
        "emo_vec",
        "emo_entropy",
        "emo_entropy_norm",
        "emo_entropy_norm_merged",
        "emo_maxprob",
        "emo_margin",
        "ambiguity_signal",
    ]

    def _sb_compatible_json_path(name, json_path):
        """SpeechBrain reserves per-entry key 'id'; strip it in a local cache copy if present."""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return json_path

        # We only need to sanitize dict manifests with per-utterance dict values.
        if not isinstance(raw, dict):
            return json_path

        has_inner_id = False
        cleaned = {}
        for uttid, entry in raw.items():
            if isinstance(entry, dict):
                if "id" in entry:
                    has_inner_id = True
                    e = dict(entry)
                    e.pop("id", None)
                    cleaned[uttid] = e
                else:
                    cleaned[uttid] = entry
            else:
                cleaned[uttid] = entry

        if not has_inner_id:
            return json_path

        cache_dir = os.path.join(save_folder, "_sb_manifest_cache")
        os.makedirs(cache_dir, exist_ok=True)
        out_path = os.path.join(cache_dir, f"{name}.sb.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, sort_keys=True)
        return out_path

    for name, json_path in data_info.items():
        if json_path is None or not os.path.exists(json_path):
            continue

        load_path = _sb_compatible_json_path(name, json_path)
        datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=load_path,
            replacements={"data_root": hparams.get("data_folder", "")},
            dynamic_items=dyn_items,   
            output_keys=out_keys
        )

    return datasets
