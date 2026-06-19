#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_msp_podcast.py
Author: Zahra Omidi, CRSS-UTDallas
----------------------------------
Converts MSP-Podcast labels into train/valid/test JSONs for emotion recognition.
Supports both:
  • consensus-only mode  (from labels_consensus.csv)
  • soft-label mode      (from labels.txt with primary/secondary distributions)
"""

import os
import re
import json
import math
import numpy as np
from typing import Dict, List, Optional
from collections import Counter, defaultdict
from speechbrain.dataio.dataio import read_audio
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)
# Debug counters (for sanity-checking label normalization)
_DBG = Counter()
_NOAGREE_TOKENS = {"x", "noagree", "no-agree", "no_agree", "no agreement", "no-agreement", "no_agreement"}
SAMPLERATE = 16000

# ---- canonical mappings ----
# Canonical short labels used throughout this script.
# NOTE: This is intentionally *not* a merged space. Merging/grouping is handled via merge_map presets.
_ALLOWED = {"neu", "hap", "ang", "sad", "fear", "disgust", "contempt", "surprise", "other"}
# Preferred deterministic ordering (align with dataio_msp_podcast.py ORDERED_CLASSES)
_ORDERED_CLASSES = ["neu", "hap", "ang", "sad", "fear", "disgust", "contempt", "surprise", "other"]

_EMOTION_CANONICAL_MAP = {
    # Neutral
    "neutral": "neu", "neu": "neu", "neut": "neu", "n": "neu",

    # Happiness / Positive
    "happy": "hap", "happiness": "hap", "joy": "hap", "elated": "hap", "hap": "hap", "h": "hap",
    "excited": "hap", "amused": "hap",

    # Anger
    "angry": "ang", "anger": "ang", "annoyed": "ang", "hostile": "ang", "frustrated": "ang", "frustration": "ang", "ang": "ang", "a": "ang",

    # Sadness
    "sad": "sad", "sadness": "sad", "depressed": "sad", "disappointed": "sad", "concerned": "sad", "s": "sad",

    # Fear
    "fear": "fear", "afraid": "fear", "scared": "fear", "f": "fear",

    # Disgust
    "disgust": "disgust", "disgusted": "disgust", "revulsion": "disgust", "d": "disgust",

    # Contempt
    "contempt": "contempt", "scorn": "contempt", "c": "contempt",

    # Surprise
    "surprise": "surprise", "surprised": "surprise", "u": "surprise",

    # Other (includes no-agreement)
    "other": "other", "oth": "other", "o": "other",
    "x": "other", "noagree": "other", "no-agree": "other", "no_agree": "other",
}

_SPLIT_MAP = {
    "train": "train", "Train": "train", "TRAIN": "train",
    "dev": "valid", "devel": "valid", "development": "valid",
    "val": "valid", "valid": "valid", "Validation": "valid",
    "test": "test", "Test": "test", "TEST": "test",
    "Test1": "test",
}

_CODE2NAME = {
    "A": "angry", "S": "sad", "H": "happy", "U": "surprise",
    "F": "fear", "D": "disgust", "C": "contempt", "N": "neutral",
    "O": "other", "X": "noagree",
}

def load_vad_stats_from_labels_detailed(labels_detailed_json: str):
    """Load per-utterance VAD mean/var/std from MSP-Podcast labels_detailed.json.

    Expected structure:
      {"MSP-PODCAST_0001_0008.wav": {"WORKER...": {"EmoVal":4.0,"EmoAct":1.0,"EmoDom":1.0}, ...}, ...}

    Returns:
      dict[uttid] -> {"vad_mu":{V,A,D}, "vad_var":{V,A,D}, "vad_std":{V,A,D}, "n_vad": int}
      where uttid has no file extension.
    """
    with open(labels_detailed_json, "r", encoding="utf-8") as f:
        det = json.load(f)

    out = {}
    for wav_key, worker_map in det.items():
        if not isinstance(worker_map, dict):
            continue

        uttid = str(wav_key)
        if uttid.lower().endswith(".wav"):
            uttid = uttid[:-4]
        else:
            uttid = os.path.splitext(os.path.basename(uttid))[0]

        V_list, A_list, D_list = [], [], []
        for _, ann in worker_map.items():
            if not isinstance(ann, dict):
                continue
            v = ann.get("EmoVal", None)
            a = ann.get("EmoAct", None)
            d = ann.get("EmoDom", None)
            if v is None or a is None or d is None:
                continue
            try:
                V_list.append(float(v))
                A_list.append(float(a))
                D_list.append(float(d))
            except Exception:
                continue

        if len(V_list) == 0:
            continue

        v_arr = np.asarray(V_list, dtype=float)
        a_arr = np.asarray(A_list, dtype=float)
        d_arr = np.asarray(D_list, dtype=float)

        v_mu = float(v_arr.mean()); a_mu = float(a_arr.mean()); d_mu = float(d_arr.mean())
        v_var = float(v_arr.var(ddof=0)); a_var = float(a_arr.var(ddof=0)); d_var = float(d_arr.var(ddof=0))

        out[uttid] = {
            "vad_mu": {"V": v_mu, "A": a_mu, "D": d_mu},
            "vad_var": {"V": v_var, "A": a_var, "D": d_var},
            "vad_std": {"V": float(math.sqrt(v_var)), "A": float(math.sqrt(a_var)), "D": float(math.sqrt(d_var))},
            "n_vad": int(len(V_list)),
        }

    return out

# ---------------------------------------------------------------------
def _map_emotion(e, merge_map=None):
    if e is None:
        return None
    e_str = str(e).strip()
    if not e_str:
        return None
    low = e_str.lower()

    # Apply merge_map on the full token only
    if merge_map and low in merge_map:
        low = str(merge_map[low]).strip().lower()

    # Direct canonical mapping
    if low in _EMOTION_CANONICAL_MAP:
        mapped = _EMOTION_CANONICAL_MAP[low]
        # Count explicit no-agreement → other conversions
        if low in _NOAGREE_TOKENS and mapped == "other":
            _DBG["noagree_to_other"] += 1
        return mapped

    # Single-letter codes only (IMPORTANT: do NOT apply first-letter fallback to multi-char strings)
    if len(low) == 1:
        mapped = _EMOTION_CANONICAL_MAP.get(low)
        if low in _NOAGREE_TOKENS and mapped == "other":
            _DBG["noagree_to_other"] += 1
        return mapped

    return None

def _norm_label(s: str) -> str:
    """Normalize emotion label tokens (primary/secondary)."""
    if not s:
        return ""
    s = str(s).strip().lower()
    # Normalize any "other<sep>" prefix to "other-" (e.g., "Other:Confused" -> "other-confused")
    if re.match(r'(?i)^other\s*[-_:]', s):
        s = re.sub(r'(?i)^other\s*[-_:]+\s*', 'other-', s)
    return re.sub(r"\s+", " ", s).strip()

_SECONDARY_LABEL_SPLIT_PATTERN = r'(?:,|;|\||/|\+|&|\band\b)'
def _split_secondary_list(s: str, merge_map: dict | None = None, majors: set | None = None):
    """Parse a secondary-emotion string into canonical short labels.

    This function is intentionally *major-set aware* to support Interspeech ablations.

    Key behaviors:
    - Applies `merge_map` (synonym/grouping preset) if provided.
    - Supports compound tokens like `other-X` by optionally counting both `other` and `X`.
    - If `majors` is provided, returns ONLY labels in `majors`.
      • If `"other"` is not in majors, out-of-scope labels are dropped (NOT forced to `other`).
    - Deduplicates while preserving order.

    NOTE: Always pass `majors` from the current experiment class space (e.g., 9-class includes `other`).
    If `majors` is omitted, the function defaults to the full canonical set (`_ALLOWED`).
    """
    if not s:
        return []

    # Normalize merge_map keys to lowercase (values are kept as-is; may already be short labels).
    if merge_map is None:
        merge_map = {}
    else:
        merge_map = {str(k).lower(): v for k, v in merge_map.items()}

    # If majors is not provided, default to the full canonical set.
    # (Avoid silent filtering that would drop contempt/other in 9-class runs.)
    if majors is None:
        majors = _ALLOWED

    parts = re.split(_SECONDARY_LABEL_SPLIT_PATTERN, str(s), flags=re.IGNORECASE)
    toks: list[str] = []

    def _maybe_keep(lbl: str | None):
        """Append lbl if it is allowed by majors."""
        if not lbl:
            return
        if lbl in majors:
            toks.append(lbl)

    for p in parts:
        p = (p or "").strip()
        if not p:
            continue

        # Remove trailing weights like ":0.5", "-0.3", "(0.3)"
        p = re.sub(r'[:\-\u2013\u2014]\s*\d+(\.\d+)?$', '', p)
        p = re.sub(r'\(\s*\d+(\.\d+)?\s*\)$', '', p)

        # Allow single-letter codes if present
        if len(p) == 1 and p.upper() in _CODE2NAME:
            p = _CODE2NAME[p.upper()]

        p = _norm_label(p)
        if not p:
            continue

        # Handle "other-X" tokens.
        # If the current class space includes the coarse label "other", keep it as "other".
        # Otherwise, fall back to mapping the subtype (e.g., other-disappointed -> sad).
        if p.startswith("other-"):
            if majors and "other" in majors:
                toks.append("other")
                _DBG["secondary_other_hyphen_to_other"] += 1
                continue

            sub = p.split("-", 1)[1]
            sub_mapped = merge_map.get(sub.lower(), sub)
            short = _map_emotion(sub_mapped, merge_map)
            _maybe_keep(short)
            _DBG["secondary_other_hyphen_to_subtype"] += 1
            continue

        # Apply merge_map at token-level (synonyms/grouping presets)
        low = p.lower()
        if low in merge_map:
            low = merge_map[low]

        short = _map_emotion(low, merge_map)
        _maybe_keep(short)

    # Deduplicate preserving order
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _safe_float_or_none(x):
    try:
        v = float(x)
        return None if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return None

def _norm_dist_dict(d: dict):
    """Clamp negatives, drop non-numeric, renormalize to sum=1."""
    if not isinstance(d, dict) or not d:
        return {}
    dd = {}
    for k, v in d.items():
        k2 = _norm_label(k)
        v2 = _safe_float_or_none(v)
        if k2 and v2 is not None and v2 >= 0:
            dd[k2] = dd.get(k2, 0.0) + v2
    s = sum(dd.values())
    if s > 0:
        for k in list(dd.keys()):
            dd[k] /= s
    return dd

def _apply_group_map(dist: dict, group_map: dict):
    """Map fine labels to grouped classes, then renorm."""
    if not group_map:
        return dist
    out = defaultdict(float)
    for k, p in dist.items():
        key2 = group_map.get(k) or group_map.get(k.replace("other-", "")) or group_map.get(k.split("-")[0])
        out[key2 if key2 else k] += float(p)
    s = sum(out.values())
    if s > 0:
        for k in list(out.keys()):
            out[k] /= s
    return dict(out)


def _norm_split(x: str):
    xl = x.lower()
    return _SPLIT_MAP.get(xl) or ("test" if xl.startswith("test") else "valid" if "dev" in xl else None)


def _safe_duration(wav_path: str) -> float:
    try:
        sig = read_audio(wav_path)
        return float(sig.shape[0]) / SAMPLERATE
    except Exception as e:
        logger.warning(f"Failed to read {wav_path}: {e}")
        return 0.0

def _merge_dist(p_dist, s_dist, majors, w1=0.95, w2=0.05):
    out = {m: (w1 * p_dist.get(m,0.0) + w2 * s_dist.get(m,0.0)) for m in majors}
    s = sum(out.values())
    if s > 0: 
        for k in out: out[k] /= s
    return out

def _sanitize_dist(dist: dict, class_order: List[str], fallback_idx: Optional[int] = None, tol: float = 1e-3):
    """Ensure a distribution dict is non-negative, dense over class_order, and sums to 1.

    Returns:
      (dist_dict, dist_list)
    """
    clean = {}
    for c in class_order:
        v = dist.get(c, 0.0) if isinstance(dist, dict) else 0.0
        try:
            v = float(v)
        except Exception:
            v = 0.0
        if math.isnan(v) or v < 0.0:
            v = 0.0
        clean[c] = v
    s = sum(clean.values())
    if s <= 0.0 and fallback_idx is not None and 0 <= fallback_idx < len(class_order):
        clean = {c: 0.0 for c in class_order}
        clean[class_order[fallback_idx]] = 1.0
        s = 1.0
    if s > 0.0:
        for k in clean:
            clean[k] /= s
    # final safety clamp
    s2 = sum(clean.values())
    if abs(s2 - 1.0) > tol and s2 > 0:
        for k in clean:
            clean[k] /= s2
    lst = [clean[c] for c in class_order]
    return clean, lst

def _entropy_and_margin(prob_list: List[float]):
    vec = [max(0.0, float(x)) for x in prob_list]
    s = sum(vec)
    if s > 0:
        vec = [v / s for v in vec]
    # entropy
    ent = -sum(v * math.log(v + 1e-12) for v in vec if v > 0)
    sorted_v = sorted(vec, reverse=True)
    p1 = sorted_v[0] if sorted_v else 0.0
    p2 = sorted_v[1] if len(sorted_v) > 1 else 0.0
    margin = p1 - p2
    return ent, p1, margin

def _consensus_from_dist(dist, primary_counts, neu_min_primary=2, neu_min_prob=0.35, neu_margin=0.10):
    """
    Enforce that 'neu' only wins when it's clearly dominant.
    - at least neu_min_primary primary votes, AND
    - P(neu) >= neu_min_prob, AND
    - P(neu) >= next_best + neu_margin
    Otherwise pick the best non-neu.
    """
    # rank by prob
    # Deterministic ranking:
    # 1) higher probability
    # 2) prefer non-neutral in ties (avoid boundary over-neutralization)
    # 3) lexicographic fallback for full determinism
    ranked = sorted(
        dist.items(),
        key=lambda kv: (kv[1], 1 if kv[0] != "neu" else 0, kv[0]),
        reverse=True
    )
    top_lab, top_p = ranked[0]
    # if top is not neu, done
    if top_lab != "neu":
        return top_lab

    # neu guard
    # primary_counts is expected to be a mapping of counts (ints) but some callers
    # may pass fractional/probability-based values. Normalize to an integer
    # count for the neu-guard checks.
    neu_val = primary_counts.get("neu", 0)
    total_primary = sum(primary_counts.values()) if isinstance(primary_counts, dict) else 0
    if isinstance(neu_val, float) and total_primary > 0:
        neu_primary = int(round(neu_val * total_primary))
    else:
        try:
            neu_primary = int(neu_val)
        except Exception:
            neu_primary = 0
    # get second best
    second_lab, second_p = ranked[1] if len(ranked) > 1 else ("other", 0.0)

    if (neu_primary >= neu_min_primary) and (top_p >= neu_min_prob) and ((top_p - second_p) >= neu_margin):
        return "neu"
    # else: pick the best non-neu
    for lab, p in ranked:
        if lab != "neu":
            return lab
    return "other"

# ---------------------------------------------------------------------

def get_sample_split(labels_consensus_csv: str):
    """Read Split_Set column from consensus CSV."""
    import csv
    splits = {}
    with open(labels_consensus_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row.get("FileName", "").strip()
            split_raw = row.get("Split_Set", "").strip()
            if not fname or not split_raw:
                continue
            split = _norm_split(split_raw)
            if split:
                uttid = os.path.splitext(os.path.basename(fname))[0]
                splits[uttid] = split
    return splits


def parse_labels_txt(labels_txt_path: str, data_root: str,
                     w_primary: float = 0.9, w_secondary: float = 0.1, use_secondary: bool = False,
                     use_grouped: bool = False, group_map: dict = None, majors: set = None,
                     consensus_rule: str = "neutral_guard"):
    """Parse MSP-Podcast labels.txt including secondary emotions."""
    if not os.path.isfile(labels_txt_path):
        raise FileNotFoundError(labels_txt_path)

    with open(labels_txt_path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    entries, i, N = {}, 0, len(lines)
    while i < N:
        header = lines[i].strip(); i += 1
        if not header:
            continue
        m = re.match(r"^([^;]+);\s*([A-Z]);\s*A:([0-9.]+);\s*V:([0-9.]+);\s*D:([0-9.]+);", header)
        if not m:
            continue
        uttid = os.path.splitext(os.path.basename(m.group(1).strip()))[0]
        code = m.group(2).strip()
        A, V, D = float(m.group(3)), float(m.group(4)), float(m.group(5))
        prim_counts = Counter()
        sec_counts = Counter()

        workers = []
        while i < N and lines[i].strip():  # process each annotator's label block for the current utterance
            parts = [p.strip() for p in lines[i].split(";") if p.strip()]; i += 1
            if len(parts) >= 2:
                primary_raw = _norm_label(parts[1])

                # Handle MSP-Podcast "other-<subtype>" primary labels.
                # If the current class space includes "other", keep it as "other".
                # Otherwise, fall back to mapping the subtype (so smaller class presets don't silently drop labels).
                if primary_raw.startswith("other-"):
                    if majors and "other" in majors:
                        primary_raw = "other"
                        _DBG["primary_other_hyphen_to_other"] += 1
                    else:
                        primary_raw = primary_raw.split("-", 1)[1]
                        _DBG["primary_other_hyphen_to_subtype"] += 1

                if use_grouped and group_map:
                    mapped = group_map.get(primary_raw.lower(), primary_raw)
                    primary_short = _map_emotion(mapped, group_map)
                else:
                    primary_short = _map_emotion(primary_raw, None)

                # If mapping failed and we are in a class space that supports "other",
                # treat unknown primary tokens as "other" (do NOT drop them silently).
                if primary_short is None and majors and "other" in majors:
                    primary_short = "other"
                    _DBG["primary_unknown_to_other"] += 1

                # Enforce majors: drop out-of-scope labels unless majors explicitly includes 'other'.
                if primary_short and majors and primary_short not in majors:
                    if "other" in majors:
                        primary_short = "other"
                    else:
                        primary_short = None

                if primary_short:
                    prim_counts[primary_short] += 1

                # Always define secondaries to avoid UnboundLocalError
                secondaries = []
                if use_secondary and len(parts) >= 3 and not parts[2].lower().startswith(("a:", "v:", "d:")):
                    secondaries = _split_secondary_list(parts[2], group_map, majors=majors)
                    # Worker-balanced secondary counting: each worker contributes total mass 1.0
                    if secondaries:
                        w = 1.0 / float(len(secondaries))
                        for sec in secondaries:
                            if majors and sec in majors:
                                sec_counts[sec] += w

                workers.append({"primary": primary_short,
                                "secondary": secondaries})

        while i < N and not lines[i].strip():
            i += 1

        # Ensure majors is a list of canonical short labels (e.g., ['neu', ..., 'other'])
        majors = list(majors) if majors is not None else list(_ALLOWED)
        # build p_dist_raw directly from prim_counts
        p_dist_raw = {m: prim_counts.get(m, 0.0) for m in majors}
        sp = sum(p_dist_raw.values())
        if sp > 0:
            for k in p_dist_raw:
                p_dist_raw[k] /= sp

        # build s_dist_raw only if use_secondary
        if use_secondary:
            s_dist_raw = {m: sec_counts.get(m, 0.0) for m in majors}
            ss = sum(s_dist_raw.values())
            if ss > 0:
                for k in s_dist_raw:
                    s_dist_raw[k] /= ss
        else:
            s_dist_raw = {m: 0.0 for m in majors}

        p_counts = dict(prim_counts)

        if not use_secondary:
            s_dist_raw = {m: 0.0 for m in majors}
            merged = dict(p_dist_raw)
        else:
            merged = _merge_dist(p_dist_raw, s_dist_raw, majors, w1=w_primary, w2=w_secondary)

        if consensus_rule == "argmax":
            emo_consensus_from_dist = max(merged.items(), key=lambda kv: kv[1])[0] if merged else "other"
        else:
            emo_consensus_from_dist = _consensus_from_dist(
                merged, p_counts,
                neu_min_primary=2,
                neu_min_prob=0.35,
                neu_margin=0.10
            )

        wav_path = os.path.join(data_root, f"{uttid}.wav")
        if not os.path.exists(wav_path):
            continue
        length = _safe_duration(wav_path)
        if length <= 0.0:
            continue
        
        # VAD soft-label stats are attached later from labels_detailed.json.
        # Keep header-level values here as placeholders.
        v_mu = float(V); a_mu = float(A); d_mu = float(D)
        v_var = 0.0; a_var = 0.0; d_var = 0.0

        entries[uttid] = {
            "id": uttid,
            "wav": wav_path,
            "length": length,

            # Original VAD values from labels.txt header
            "V": V, "A": A, "D": D,

            # VAD soft-label stats
            "vad_mu": {"V": v_mu, "A": a_mu, "D": d_mu},
            "vad_var": {"V": v_var, "A": a_var, "D": d_var},
            "vad_std": {"V": float(math.sqrt(v_var)), "A": float(math.sqrt(a_var)), "D": float(math.sqrt(d_var))},
            "n_vad": None,
            "emo_code": code,
            "emo": emo_consensus_from_dist,
            "emo_dist_primary": p_dist_raw,
            "emo_dist_secondary": s_dist_raw,
            "emo_dist": merged,
        }
    return entries

# ---------------------------------------------------------------------

def read_labels(labels_dir: str, data_root: str, merge_map: Optional[Dict] = None, majors: Optional[set] = None):
    """Read consensus CSV and return split-wise label lists."""
    import csv
    splits = {"train": [], "valid": [], "test": []}
    labels_file = os.path.join(labels_dir, "labels_consensus.csv")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"{labels_file} not found.")

    missing = bad_rows = 0
    with open(labels_file, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            r = { (k or "").strip().lower(): (v or "").strip() for k, v in row.items() }
            fname, emo_raw = r.get("filename"), r.get("emoclass")
            val, act, dom, split_raw = r.get("emoval"), r.get("emoact"), r.get("emodom"), r.get("split_set")
            gender = r.get("gender", "")
            if not (fname and emo_raw and val and act and dom and split_raw):
                bad_rows += 1; continue
            emo = _map_emotion(emo_raw, merge_map)
            # Drop anything unmapped (e.g., no-agreement / unknown)
            if emo is None:
                continue

            if majors is not None and emo not in majors:
                continue
            if emo not in _ALLOWED:
                continue

            try:
                v, a, d = float(val), float(act), float(dom)
            except Exception:
                bad_rows += 1; continue
            split = _norm_split(split_raw)
            if not split:
                bad_rows += 1; continue
            wav_path = os.path.join(data_root, fname)
            if not os.path.exists(wav_path):
                wav_path = os.path.join(data_root, os.path.basename(fname))
                if not os.path.exists(wav_path):
                    missing += 1; continue
            g = gender[:1].upper() if gender else ""
            splits[split].append([wav_path, emo, g, v, a, d])

    if missing:
        logger.info(f"Skipped {missing} rows (audio not found).")
    if bad_rows:
        logger.info(f"Skipped {bad_rows} malformed rows.")
    return splits


def _summarize_split(name: str, items):
    # Support both list-of-lists (no secondary) and dict-of-entries (with secondary)
    if isinstance(items, dict):
        n = len(items)
        cnt = Counter(entry.get("emo") for entry in items.values())
        lgths = [d['length'] for d in items.values()]
    else:
        n = len(items)
        cnt = Counter(x[1] for x in items)  # [wav, emo, ...]
        lgths = [_safe_duration(x[0]) for x in items]
        
    logger.info(f"[{name}] n={n} | class_counts={dict(cnt)}")
    print(f"[{name}] total utts:", n)
    print(f"[{name}] min/max dur:",
          min(lgths) if lgths else 0.0,
          max(lgths) if lgths else 0.0)



def _write_json(items: Dict, json_path: str):
    """Write dict of utterance entries to JSON."""
    with open(json_path, "w", encoding="utf-8") as w:
        json.dump(items, w, indent=2, sort_keys=True)
    logger.info(f"{json_path} created (n={len(items)})")

def _finalize_split(split_name: str, items: Dict[str, Dict], class_order: List[str]):
    """Augment + validate each entry, enforcing deterministic schema."""
    lab2idx = {c: i for i, c in enumerate(class_order)}
    checked = {}
    ambiguous = 0
    ent_norm_denom = math.log(max(len(class_order), 2))

    for uttid in sorted(items.keys()):
        e = dict(items[uttid])  # shallow copy

        # Required fields
        wav = e.get("wav")
        if not wav or not os.path.exists(wav):
            raise FileNotFoundError(f"[{split_name}] audio missing for {uttid}: {wav}")

        length = float(e.get("length") or e.get("duration") or 0.0)
        if length <= 0.0 or math.isnan(length):
            length = _safe_duration(wav)
        if length <= 0.0 or math.isnan(length):
            raise ValueError(f"[{split_name}] invalid duration for {uttid}: {length}")

        emo = e.get("emo")
        if emo is None:
            raise ValueError(f"[{split_name}] missing primary label for {uttid}")
        emo = str(emo).strip().lower()
        if emo not in lab2idx:
            raise ValueError(f"[{split_name}] label '{emo}' not in class set {class_order} for {uttid}")
        y_idx = lab2idx[emo]

        # Distributions (ensure dense + normalized)
        prim_dict, prim_list = _sanitize_dist(e.get("emo_dist_primary", {}), class_order, fallback_idx=y_idx)
        sec_dict,  sec_list  = _sanitize_dist(e.get("emo_dist_secondary", {}), class_order, fallback_idx=None)
        merged_dict, merged_list = _sanitize_dist(e.get("emo_dist", prim_dict), class_order, fallback_idx=y_idx)

        ent, maxprob, margin = _entropy_and_margin(merged_list)
        if (ent / ent_norm_denom) > 0.90 or margin < 0.15:
            ambiguous += 1

        checked[uttid] = {
            **e,
            "id": uttid,
            "wav": wav,
            "length": length,  # alias for downstream tools
            "primary_label": emo,
            "y_idx": y_idx,
            "emo": emo,

            # Canonical dict fields (SpeechBrain uses these)
            "emo_dist_primary": prim_dict,
            "emo_dist_secondary": sec_dict,
            "emo_dist": merged_dict,

            # Explicit list fields (for reproducible downstream use)
            "primary_dist": prim_list,
            "secondary_dist": sec_list,
            "merged_dist": merged_list,

            # Ambiguity diagnostics
            "emo_entropy": ent,
            "emo_maxprob": maxprob,
            "emo_margin": margin,
        }
    logger.info(f"[{split_name}] ambiguous {ambiguous}/{len(checked)} (entropy>0.9 or margin<0.15)")
    return checked

# ---------------------------------------------------------------------
def prepare_msppodcast(data_root: str, labels_dir: str, output_dir: str,
                       majors=None, use_soft_labels=True, include_secondary_emos=False,
                       primary_weight=0.9, secondary_weight=0.1,
                       drop_onx=True, merge_map: Optional[Dict] = None):
    """Main entry to create MSP-Podcast train/valid/test JSONs."""
    os.makedirs(output_dir, exist_ok=True)

    majors_in = majors or _ALLOWED
    majors = sorted(set((_map_emotion(m, merge_map) or str(m).lower()) for m in majors_in))
    logger.info(f"Using major emotions (ordered): {majors}")

    vad_stats = {}
    labels_detailed_json = os.path.join(labels_dir, "labels_detailed.json")
    if not os.path.exists(labels_detailed_json):
        raise FileNotFoundError(labels_detailed_json)
    logger.info(f"Loading VAD stats from labels_detailed.json: {labels_detailed_json}")
    vad_stats = load_vad_stats_from_labels_detailed(labels_detailed_json)
    logger.info(f"Loaded VAD stats for {len(vad_stats)} utterances")

    merge_map = merge_map or {}
    # _merge_map_norm = {str(k).lower(): _map_emotion(v, merge_map) or str(v).lower() for k, v in merge_map.items()}
    logger.info(f"Using merge_map: {merge_map}")

    if use_soft_labels:
        labels_txt = os.path.join(labels_dir, "labels.txt")
        if not os.path.exists(labels_txt):
            raise FileNotFoundError(f"{labels_txt} not found. Disable --include_secondary_emos or provide the file.")
        if include_secondary_emos:
            # Full soft-label mode: primary + secondary + merged distributions
            logger.info("Including secondary emotions in output JSONs.")
            entries = parse_labels_txt(
                labels_txt, data_root,
                w_primary=primary_weight,
                w_secondary=secondary_weight,
                use_grouped=False,
                group_map=None,
                majors=majors,
                use_secondary=True,
            )

            utt_split_map = get_sample_split(os.path.join(labels_dir, "labels_consensus.csv"))
            splits = {"train": {}, "valid": {}, "test": {}}

            for uttid, split in utt_split_map.items():
                if uttid not in entries or split not in splits:
                    continue
                entry = entries[uttid]
                final_emo = entry["emo"]

                base_primary = entry.get("emo_dist_primary", {})
                base_secondary = entry.get("emo_dist_secondary", {})
                base_merged = entry.get("emo_dist", {})

                vs = vad_stats.get(uttid, {})

                splits[split][uttid] = {
                    "wav": entry["wav"],
                    "length": entry["length"],
                    "emo": final_emo,
                    "emo_dist_primary": base_primary,
                    "emo_dist_secondary": base_secondary,
                    "emo_dist": base_merged,

                    # Point VAD values (for backward compatibility)
                    "valence": float(entry["V"]),
                    "arousal": float(entry["A"]),
                    "dominance": float(entry["D"]),

                    # Soft-label VAD stats
                    "vad_mu": (vs["vad_mu"] if vs else {"V": float(entry["V"]), "A": float(entry["A"]), "D": float(entry["D"])}),
                    "vad_var": (vs["vad_var"] if vs else {"V": 0.0, "A": 0.0, "D": 0.0}),
                    "vad_std": (vs["vad_std"] if vs else {"V": 0.0, "A": 0.0, "D": 0.0}),
                    "n_vad": (vs.get("n_vad") if vs else None),
                }

        else:
            # labels.txt primary-only mode: keep worker primary distribution, ignore secondary by setting w_secondary=0.
            logger.info("Using labels.txt with worker PRIMARY distributions only (secondary ignored).")
            entries = parse_labels_txt(
                labels_txt, data_root,
                w_primary=primary_weight,
                w_secondary=0.0,
                use_grouped=False,
                group_map=None,
                majors=majors,
                use_secondary=False,
            )

            utt_split_map = get_sample_split(os.path.join(labels_dir, "labels_consensus.csv"))
            splits = {"train": {}, "valid": {}, "test": {}}

            for uttid, split in utt_split_map.items():
                if uttid not in entries or split not in splits:
                    continue
                entry = entries[uttid]
                final_emo = entry["emo"]

                vs = vad_stats.get(uttid)

                splits[split][uttid] = {
                    "wav": entry["wav"],
                    "length": entry["length"],
                    "emo": final_emo,
                    "emo_dist_primary": entry.get("emo_dist_primary", {}),
                    "emo_dist_secondary": {},
                    "emo_dist": entry.get("emo_dist_primary", {}),

                    # Point VAD values (for backward compatibility)
                    "valence": float(entry["V"]),
                    "arousal": float(entry["A"]),
                    "dominance": float(entry["D"]),

                    # Soft-label VAD stats
                    "vad_mu": (vs["vad_mu"] if vs else {"V": float(entry["V"]), "A": float(entry["A"]), "D": float(entry["D"])}),
                    "vad_var": (vs["vad_var"] if vs else {"V": 0.0, "A": 0.0, "D": 0.0}),
                    "vad_std": (vs["vad_std"] if vs else {"V": 0.0, "A": 0.0, "D": 0.0}),
                    "n_vad": (vs.get("n_vad") if vs else None),
                }
    else:
        splits = read_labels(labels_dir, data_root, merge_map, majors)
        for split in splits:
            splits[split] = [
                x for x in splits[split] if x[1] in majors
            ]

    # ------------------------------------------------------------------
    # Convert to dict keyed by uttid (if coming from consensus-only path)
    # ------------------------------------------------------------------
    if not isinstance(splits.get("train"), dict):
        new_splits = {"train": {}, "valid": {}, "test": {}}
        for split in ["train", "valid", "test"]:
            for x in splits[split]:
                uttid = os.path.splitext(os.path.basename(x[0]))[0]
                vs = vad_stats.get(uttid) if labels_detailed_json else None
                final_emo = x[1]
                v = float(x[3]); a = float(x[4]); d = float(x[5])
                new_splits[split][uttid] = {
                    "wav": x[0],
                    "length": _safe_duration(x[0]),
                    "emo": final_emo,
                    # Always include distribution keys for schema consistency (hard mode -> empty dicts)
                    "emo_dist_primary": {},
                    "emo_dist_secondary": {},
                    "emo_dist": {},
                    "valence": v,
                    "arousal": a,
                    "dominance": d,
                    "gender": x[2],

                    # Soft-label VAD stats (consensus CSV provides one V/A/D per utterance)
                    "vad_mu": (vs["vad_mu"] if vs else {"V": v, "A": a, "D": d}),
                    "vad_var": (vs["vad_var"] if vs else {"V": 0.0, "A": 0.0, "D": 0.0}),
                    "vad_std": (vs["vad_std"] if vs else {"V": 0.0, "A": 0.0, "D": 0.0}),
                    "n_vad": (vs.get("n_vad") if vs else None),
                }
        splits = new_splits

    # ------------------------------------------------------------------
    # Validate + enrich entries, then summarize and write JSONs
    # ------------------------------------------------------------------
    class_order = [c for c in _ORDERED_CLASSES if c in majors]
    for c in majors:
        if c not in class_order:
            class_order.append(c)
    lab2idx = {c: i for i, c in enumerate(class_order)}

    for split in ["train", "valid", "test"]:
        splits[split] = _finalize_split(split, splits.get(split, {}), class_order)

    _summarize_split("TRAIN", splits["train"])
    _summarize_split("VALID", splits["valid"])
    _summarize_split("TEST",  splits["test"])        

    logger.info(f"Writing JSONs to {output_dir}")
    # Persist label maps for reproducibility
    mapping_report = {
        "class_order": class_order,
        "lab2idx": lab2idx,
        "canonical_map": _EMOTION_CANONICAL_MAP,
        "merge_map": merge_map,
    }
    with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping_report, f, indent=2, sort_keys=True)

    # Split-level summary (counts + ambiguity)
    ent_norm_denom = math.log(max(len(class_order), 2))
    prep_summary = {
        "class_order": class_order,
        "split_stats": {},
    }
    for split in ["train", "valid", "test"]:
        cnt = Counter(entry.get("primary_label") for entry in splits[split].values())
        amb = sum(
            (entry["emo_entropy"] / ent_norm_denom) > 0.90 or entry["emo_margin"] < 0.15
            for entry in splits[split].values()
        )
        prep_summary["split_stats"][split] = {
            "num_utts": len(splits[split]),
            "class_counts": dict(cnt),
            "ambiguous": int(amb),
        }
    with open(os.path.join(output_dir, "prep_summary.json"), "w", encoding="utf-8") as f:
        json.dump(prep_summary, f, indent=2, sort_keys=True)

    for split in ["train", "valid", "test"]:
        _write_json(splits[split], os.path.join(output_dir, f"{split}.json"))

    logger.info("All done.")
    if _DBG:
        logger.info(f"[DEBUG] label-normalization counters: {dict(_DBG)}")
    if not any(len(v) for v in splits.values()):
        logger.warning("All splits are empty — check that labels.txt and consensus CSV match and merge_map covers all emotion variants.")


# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Prepare MSP-Podcast JSONs for training and evaluation."
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Directory containing the audio files (wav).")
    parser.add_argument("--labels_folder", type=str, required=True,
                        help="Path to folder with labels_consensus.csv and labels.txt.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write train/valid/test JSONs.")
    parser.add_argument("--include_secondary_emos", action="store_true",
                        help="Include primary/secondary distributions from labels.txt.")
    parser.add_argument("--primary_weight", type=float, default=0.95,
                        help="Weight for primary emotions.")
    parser.add_argument("--secondary_weight", type=float, default=0.05,
                        help="Weight for secondary emotions.")
    parser.add_argument("--class_preset", type=str, default="",
                        choices=["", "4", "6", "8", "9"],
                        help="Primary-only class preset: 4={neu,hap,ang,sad}, 6=+{disgust,surprise}, 8=+{fear,contempt}, 9=+{other}. No-agreement (X/noagree) is mapped to 'other'.")
    parser.add_argument("--merge_preset", type=str, default="paper_v1",
                        choices=["none", "minimal", "paper_v1", "paper_v1_nosur"],
                        help="Grouping strategy for the *_grouped fields. Canonical fields are always unmerged.")
    parser.add_argument("--consensus_only", action="store_true",
                    help="Use only labels_consensus.csv (no labels.txt).")

    args = parser.parse_args()
    use_soft_labels = not args.consensus_only

    # -----------------------------
    # Merge presets (grouping strategies)
    # -----------------------------
    # IMPORTANT: These affect ONLY the *_grouped fields. Canonical `emo` / `emo_dist*` are always unmerged.
    MERGE_PRESETS = {
        # No merging (grouped == canonical)
        "none": {},

        # Minimal synonym normalization only (does not collapse core emotions)
        "minimal": {
            "excited": "hap", "amused": "hap",
            "annoyed": "ang", "frustrated": "ang", "frustration": "ang",
            "disappointed": "sad", "depressed": "sad", "concerned": "sad",
        },

        # Paper v1 (aggressive): collapse contempt->disgust, and surprise->hap
        "paper_v1": {
            "excited": "hap", "amused": "hap",
            "annoyed": "ang", "frustrated": "ang", "frustration": "ang",
            "disappointed": "sad", "depressed": "sad", "concerned": "sad",
            "contempt": "disgust",
            "surprise": "hap",
        },

        # Alternative: keep surprise separate but collapse contempt->disgust
        "paper_v1_nosur": {
            "excited": "hap", "amused": "hap",
            "annoyed": "ang", "frustrated": "ang", "frustration": "ang",
            "disappointed": "sad", "depressed": "sad", "concerned": "sad",
            "contempt": "disgust",
        },
    }

    preset = getattr(args, "merge_preset", "paper_v1")
    merge_map_norm = MERGE_PRESETS.get(preset, {})

    # -----------------------------
    # Class presets for primary-only experiments
    # -----------------------------
    CLASS_PRESETS = {
        "": None,  # use full canonical set
        "4": ["neu", "hap", "ang", "sad"],
        "6": ["neu", "hap", "ang", "sad", "disgust", "surprise"],
        "8": ["neu", "hap", "ang", "sad", "disgust", "surprise", "fear", "contempt"],
        "9": ["neu", "hap", "ang", "sad", "disgust", "surprise", "fear", "contempt", "other"],
    }
    majors = CLASS_PRESETS.get(args.class_preset, None)

    logger.info(f"Preparing MSP-Podcast with data_root={args.data_root}, labels={args.labels_folder}, merge_preset={args.merge_preset}, class_preset={args.class_preset or 'full'}")

    prepare_msppodcast(
        data_root=os.path.abspath(args.data_root),
        labels_dir=os.path.abspath(args.labels_folder),
        output_dir=os.path.abspath(args.output_dir),
        majors=majors,
        include_secondary_emos=args.include_secondary_emos,
        primary_weight=args.primary_weight,
        secondary_weight=args.secondary_weight,
        merge_map=merge_map_norm,
        use_soft_labels=use_soft_labels,
    )
