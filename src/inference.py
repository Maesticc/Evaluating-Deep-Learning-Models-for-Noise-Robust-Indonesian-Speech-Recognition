"""Run every model on the same clean and noisy utterances and record predictions and timing.

Noisy waveforms are regenerated deterministically from the conditions manifest, so all
models receive identical inputs. Timing covers model inference only (waveform generation
excluded) and skips warm-up batches.

Usage:
    python -m src.inference --models whisper whisper_turbo wav2vec2 custom_cnn_bilstm_ctc [--pilot] [--split test]
"""

from __future__ import annotations

import argparse
import time
import traceback

import numpy as np
import pandas as pd
import torch

from src.model_loader import MODEL_NAMES, load_model
from src.noise_mixing import generate_waveform
from src.utils import (ensure_dir, environment_info, get_logger, load_config, manifest_path,
                       project_path, resolve_device, save_json, set_seed)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def run_model(cfg: dict, name: str, conditions: pd.DataFrame, tag: str, device: str) -> None:
    log = get_logger("inference", project_path(cfg["paths"]["logs_dir"]) / f"inference_{tag}.log")
    out_dir = ensure_dir(project_path(cfg["paths"]["predictions_dir"]))
    sr = cfg["audio"]["target_sample_rate"]
    batch_size = cfg["models"][name]["batch_size"]

    t_load = time.time()
    model = load_model(name, cfg, device)
    load_sec = time.time() - t_load
    if model.sample_rate != sr:
        raise ValueError(f"{name} expects {model.sample_rate} Hz but the pipeline produces {sr} Hz")
    log.info("Loaded %s in %.1fs: %s", name, load_sec, model.describe())
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Sort by duration inside each condition to reduce padding; order does not affect results.
    rows = conditions.sort_values(["condition", "duration_sec", "utt_id"]).reset_index(drop=True)
    batches = [rows.iloc[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    for b in batches[: cfg["inference"]["warmup_batches"]]:
        model.transcribe([generate_waveform(cfg, r) for _, r in b.iterrows()])

    records, errors = [], []
    for bi, b in enumerate(batches):
        waves = [generate_waveform(cfg, r) for _, r in b.iterrows()]
        try:
            _sync(device)
            t0 = time.perf_counter()
            hyps = model.transcribe(waves)
            _sync(device)
            elapsed = time.perf_counter() - t0
        except Exception as exc:  # recorded, not hidden
            errors.append({"batch": bi, "utt_ids": list(b["utt_id"]), "error": repr(exc),
                           "traceback": traceback.format_exc()})
            log.error("Batch %d failed: %r", bi, exc)
            hyps, elapsed = [None] * len(b), float("nan")
        audio_sec = sum(len(w) for w in waves) / sr
        for (_, r), w, h in zip(b.iterrows(), waves, hyps):
            records.append({
                "model": name, "utt_id": r["utt_id"], "condition": r["condition"],
                "realization": r["realization"], "target_snr_db": r["target_snr_db"],
                "noise_id": r["noise_id"], "noise_category": r["noise_category"],
                "speaker_id": r["speaker_id"], "category": r["category"],
                "reference": r["transcript"], "hypothesis": h, "audio_sec": len(w) / sr,
                # batch time apportioned by audio duration
                "inference_sec": elapsed * (len(w) / sr) / audio_sec if audio_sec else float("nan"),
            })
        if (bi + 1) % 50 == 0:
            log.info("%s: %d/%d batches", name, bi + 1, len(batches))

    pred = pd.DataFrame(records)
    pred_path = out_dir / f"predictions_{tag}_{name}.csv"
    pred.to_csv(pred_path, index=False)
    meta = {
        "model": model.describe(), "tag": tag, "device": device, "batch_size": batch_size,
        "n_rows": len(pred), "n_failed": int(pred["hypothesis"].isna().sum()),
        "warmup_batches": cfg["inference"]["warmup_batches"], "model_load_sec": round(load_sec, 2),
        "timing_method": "time.perf_counter around model.transcribe with CUDA synchronize; "
                         "waveform generation and warm-up excluded",
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else None,
        "environment": environment_info(), "errors": errors, "config_path": cfg["_config_path"],
    }
    save_json(meta, out_dir / f"run_{tag}_{name}.json")
    log.info("Wrote %s (%d rows, %d failed)", pred_path, len(pred), meta["n_failed"])

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--split", default="test")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Debug only: first N utterances")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = resolve_device(cfg["inference"]["device"])
    tag = f"{args.split}{'_pilot' if args.pilot else ''}"
    conditions = pd.read_csv(manifest_path(cfg, f"conditions_{tag}.csv"))
    conditions["noise_id"] = conditions["noise_id"].fillna("")
    conditions["noise_category"] = conditions["noise_category"].fillna("")
    if args.limit:
        keep = sorted(conditions["utt_id"].unique())[: args.limit]
        conditions = conditions[conditions["utt_id"].isin(keep)]
        tag += f"_limit{args.limit}"
    for name in args.models:
        run_model(cfg, name, conditions, tag, device)


if __name__ == "__main__":
    main()
