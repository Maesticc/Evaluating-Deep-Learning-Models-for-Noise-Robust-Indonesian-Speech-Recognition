"""Audit speech and noise files and assign valid / review / invalid status.

Automatic checks only *flag* questionable files. Ambiguous cases go to "review" and are
resolved by listening; decisions are recorded in a CSV so the filtering stays auditable:

    data/processed/manifests/manual_review_decisions.csv
    columns: manifest (speech|noise), id (utt_id|noise_id), decision (valid|invalid), note

Usage:
    python -m src.audio_validation --target speech|noise|all
    python -m src.audio_validation --target speech --summary-only   # duration distribution
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import soundfile as sf

from src.utils import get_logger, load_config, manifest_path, project_path, save_json

log = get_logger("audio_validation")

DECISIONS_FILE = "manual_review_decisions.csv"
EPS = 1e-12


def signal_stats(audio: np.ndarray, clip_peak: float) -> dict:
    """Amplitude statistics used for the audit (audio: mono float32)."""
    if audio.size == 0:
        return {"num_samples": 0, "rms_dbfs": float("-inf"), "peak": 0.0, "clipped_fraction": 0.0}
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return {
        "num_samples": int(audio.size),
        "rms_dbfs": round(20 * np.log10(rms + EPS), 2),
        "peak": round(float(np.max(np.abs(audio))), 5),
        "clipped_fraction": round(float(np.mean(np.abs(audio) >= clip_peak)), 6),
    }


def inspect_file(path, clip_peak: float) -> dict:
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # unreadable / corrupt
        return {"readable": False, "read_error": f"{type(exc).__name__}: {exc}"[:200]}
    stats = signal_stats(audio.mean(axis=1), clip_peak)
    stats.update({"readable": True, "read_error": "", "actual_sample_rate": sr,
                  "actual_channels": audio.shape[1],
                  "actual_duration_sec": round(audio.shape[0] / sr, 4)})
    return stats


def _inspect_all(paths: list, clip_peak: float, workers: int) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda p: inspect_file(p, clip_peak), paths, chunksize=64))
    return pd.DataFrame(results)


def _apply_manual_decisions(df: pd.DataFrame, manifest: str, id_col: str, cfg: dict) -> pd.DataFrame:
    path = manifest_path(cfg, DECISIONS_FILE)
    if not path.exists():
        pd.DataFrame(columns=["manifest", "id", "decision", "note"]).to_csv(path, index=False)
        return df
    dec = pd.read_csv(path, keep_default_na=False)
    dec = dec[dec["manifest"] == manifest]
    bad = set(dec["decision"]) - {"valid", "invalid"}
    if bad:
        raise ValueError(f"Invalid decisions in {path}: {bad}")
    for _, row in dec.iterrows():
        mask = df[id_col] == row["id"]
        if not mask.any():
            log.warning("Manual decision for unknown %s id %s", manifest, row["id"])
            continue
        df.loc[mask, "quality_status"] = row["decision"]
        df.loc[mask, "manual_decision"] = f"{row['decision']}: {row['note']}"
    log.info("Applied %d manual %s decisions", len(dec), manifest)
    return df


def _assign(df: pd.DataFrame, mask: pd.Series, status: str, reason: str) -> None:
    """Flag rows; 'invalid' overrides 'review', which overrides 'valid'. Reasons accumulate."""
    rank = {"valid": 0, "review": 1, "invalid": 2}
    for idx in df.index[mask]:
        if rank[status] > rank[df.at[idx, "quality_status"]]:
            df.at[idx, "quality_status"] = status
        prev = df.at[idx, "exclusion_reason"]
        df.at[idx, "exclusion_reason"] = f"{prev};{reason}" if prev else reason


# --------------------------------------------------------------------------- speech
def audit_speech(cfg: dict, summary_only: bool = False) -> pd.DataFrame:
    a = cfg["speech_audit"]
    root = project_path(cfg["paths"]["speech_root"])
    df = pd.read_csv(manifest_path(cfg, "speech_manifest_raw.csv"), keep_default_na=False)
    df["exclusion_reason"] = ""
    df["quality_status"] = "valid"
    df["manual_decision"] = ""

    present = df.index[df["audio_path"].map(lambda p: (root / p).exists())]
    log.info("Inspecting %d / %d speech files", len(present), len(df))
    stats = _inspect_all([root / df.at[i, "audio_path"] for i in present], a["clipping_peak"], a["num_workers"])
    stats.index = present
    df = df.join(stats)

    durations = df["actual_duration_sec"].dropna()
    summary = {
        "n_rows": len(df), "n_inspected": int(len(present)),
        "duration_sec": durations.describe(percentiles=[.001, .01, .05, .5, .95, .99, .999]).round(3).to_dict(),
        "rms_dbfs": df["rms_dbfs"].replace(-np.inf, np.nan).dropna().describe(percentiles=[.001, .01, .05, .5]).round(2).to_dict(),
    }
    log.info("Duration summary (sec): %s", summary["duration_sec"])
    log.info("RMS summary (dBFS): %s", summary["rms_dbfs"])
    if summary_only:
        return df

    target_sr = cfg["audio"]["target_sample_rate"]
    readable = df["readable"].fillna(False).astype(bool)
    _assign(df, df["readable"].isna(), "invalid", "audio_not_found")
    _assign(df, df["readable"].notna() & ~readable, "invalid", "unreadable")
    _assign(df, readable & (df["num_samples"] == 0), "invalid", "empty_waveform")
    _assign(df, df["transcript"].astype(str).str.strip() == "", "invalid", "missing_transcript")
    _assign(df, readable & (df["peak"] == 0), "invalid", "all_zero_waveform")
    _assign(df, readable & (df["rms_dbfs"] < a["silence_rms_dbfs"]), "review", "near_silent")
    _assign(df, readable & (df["clipped_fraction"] >= a["clipping_fraction_review"]), "review", "possible_clipping")
    _assign(df, readable & (df["actual_sample_rate"] != target_sr), "review", "sample_rate_mismatch")
    meta_dur = pd.to_numeric(df["duration_sec"], errors="coerce")
    _assign(df, readable & ((meta_dur - df["actual_duration_sec"]).abs() > a["metadata_duration_tolerance_sec"]),
            "review", "duration_metadata_mismatch")
    if a.get("min_duration_sec") is not None:
        _assign(df, readable & (df["actual_duration_sec"] < a["min_duration_sec"]), "review", "too_short")
    if a.get("max_duration_sec") is not None:
        _assign(df, readable & (df["actual_duration_sec"] > a["max_duration_sec"]), "review", "too_long")
    else:
        log.warning("Duration thresholds are not set; inspect the summary and set them in config.yaml")

    df = _apply_manual_decisions(df, "speech", "utt_id", cfg)

    # Primary-evaluation eligibility is separate from audio quality.
    df["eligible_primary"] = (df["quality_status"] == "valid") & ~(
        df["is_synthetic"].astype(str).str.lower().eq("true") & bool(a["exclude_synthetic"]))

    out = manifest_path(cfg, "speech_manifest.csv")
    df.to_csv(out, index=False)
    summary["status_counts"] = df.groupby(["official_split", "quality_status"]).size().unstack(fill_value=0).to_dict()
    summary["eligible_primary_per_split"] = df[df["eligible_primary"]].groupby("official_split").size().to_dict()
    summary["exclusion_reasons"] = df["exclusion_reason"].str.split(";").explode().value_counts().drop("", errors="ignore").to_dict()
    save_json(summary, manifest_path(cfg, "speech_audit_summary.json"))
    log.info("Wrote %s\n%s", out, df.groupby(["official_split", "quality_status"]).size().to_string())
    return df


# --------------------------------------------------------------------------- noise
def audit_noise(cfg: dict) -> pd.DataFrame:
    a = cfg["noise_audit"]
    root = project_path(cfg["paths"]["noise_root"])
    df = pd.read_csv(manifest_path(cfg, "noise_manifest_raw.csv"), keep_default_na=False)
    df["exclusion_reason"] = ""
    df["quality_status"] = "valid"
    df["manual_decision"] = ""
    stats = _inspect_all([root / p for p in df["noise_path"]], 0.999, a["num_workers"])
    df = df.join(stats)

    readable = df["readable"].astype(bool)
    _assign(df, ~readable, "invalid", "unreadable")
    _assign(df, readable & (df["num_samples"] == 0), "invalid", "empty_waveform")
    _assign(df, df["noise_category"].isin(a["speech_like_categories"]), "invalid", "speech_or_babble")
    if not a.get("include_music", False):
        _assign(df, df["noise_category"].str.lower().str.contains("music"), "invalid", "music_out_of_scope")
    _assign(df, df["noise_category"].isin(a["review_categories"]), "review", "may_contain_background_speech")
    _assign(df, readable & (df["rms_dbfs"] < a["silence_rms_dbfs"]), "review", "near_silent")
    _assign(df, readable & (df["actual_sample_rate"] != cfg["audio"]["target_sample_rate"]),
            "review", "sample_rate_mismatch")

    df = _apply_manual_decisions(df, "noise", "noise_id", cfg)
    out = manifest_path(cfg, "noise_manifest.csv")
    df.to_csv(out, index=False)
    log.info("Wrote %s\n%s", out,
             df.groupby(["noise_pool", "noise_category", "quality_status"]).size().to_string())
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--target", choices=["speech", "noise", "all"], default="all")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.target in ("speech", "all"):
        audit_speech(cfg, summary_only=args.summary_only)
    if args.target in ("noise", "all") and not args.summary_only:
        audit_noise(cfg)


if __name__ == "__main__":
    main()
