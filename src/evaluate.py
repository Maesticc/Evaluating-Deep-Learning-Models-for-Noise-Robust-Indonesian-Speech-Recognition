"""Compute WER, CER, WER degradation and RTFx; produce result tables and plots.

WER  = (S + D + I) / N_ref_words * 100      (corpus-level: errors summed over utterances)
CER  = (S + D + I) / N_ref_chars * 100      (spaces count as characters)
dWER = WER_noisy - WER_clean                 (percentage points)
rel  = (WER_noisy - WER_clean) / WER_clean * 100
RTFx = audio duration / inference time       (higher = faster than real time)

The same normalization (src.preprocessing.normalize_text) is applied to references and
hypotheses of every model. Confidence intervals: utterance-level bootstrap.

Usage:
    python -m src.evaluate [--pilot] [--split test] [--models custom_cnn_bilstm_ctc wav2vec2 whisper]
"""

from __future__ import annotations

import argparse

import jiwer
import numpy as np
import pandas as pd

from src.model_loader import MODEL_NAMES
from src.preprocessing import normalize_text
from src.utils import ensure_dir, get_logger, load_config, load_json, project_path, save_json

log = get_logger("evaluate")
CONDITION_ORDER = ["clean", "snr_20dB", "snr_10dB", "snr_5dB", "snr_0dB"]


def error_counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(word_errors, ref_words, char_errors, ref_chars) for one utterance."""
    hyp = hyp or ""
    w = jiwer.process_words(ref, hyp)
    c = jiwer.process_characters(ref, hyp)
    w_err = w.substitutions + w.deletions + w.insertions
    c_err = c.substitutions + c.deletions + c.insertions
    return w_err, w.hits + w.substitutions + w.deletions, c_err, c.hits + c.substitutions + c.deletions


def score_predictions(pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pred = pred.copy()
    pred["ref_norm"] = pred["reference"].map(lambda t: normalize_text(t, cfg))
    pred["hyp_norm"] = pred["hypothesis"].map(lambda t: normalize_text(t, cfg))
    empty = pred["ref_norm"] == ""
    if empty.any():
        log.warning("Dropping %d rows with empty normalized reference", int(empty.sum()))
        pred = pred[~empty]
    counts = [error_counts(r, h) for r, h in zip(pred["ref_norm"], pred["hyp_norm"])]
    pred[["word_errors", "ref_words", "char_errors", "ref_chars"]] = np.array(counts)
    return pred


def bootstrap_ci(err: np.ndarray, n: np.ndarray, resamples: int, level: float, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), size=(resamples, len(err)))
    rates = err[idx].sum(axis=1) / n[idx].sum(axis=1) * 100
    lo, hi = np.percentile(rates, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def summarize(scored: pd.DataFrame, keys: list[str], cfg: dict, with_ci: bool = True) -> pd.DataFrame:
    ecfg = cfg["evaluation"]
    rows = []
    for k, g in scored.groupby(keys, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        row = dict(zip(keys, k))
        row.update({
            "n_utterances": g["utt_id"].nunique(), "n_rows": len(g),
            "wer": g["word_errors"].sum() / g["ref_words"].sum() * 100,
            "cer": g["char_errors"].sum() / g["ref_chars"].sum() * 100,
            "utt_wer_std": float((g["word_errors"] / g["ref_words"] * 100).std()),
            "audio_sec": g["audio_sec"].sum(), "inference_sec": g["inference_sec"].sum(),
        })
        row["rtfx"] = row["audio_sec"] / row["inference_sec"] if row["inference_sec"] > 0 else np.nan
        if with_ci and len(g) > 1:
            seed = cfg["project"]["seed"]
            row["wer_ci_low"], row["wer_ci_high"] = bootstrap_ci(
                g["word_errors"].values, g["ref_words"].values, ecfg["bootstrap_resamples"], ecfg["confidence_level"], seed)
            row["cer_ci_low"], row["cer_ci_high"] = bootstrap_ci(
                g["char_errors"].values, g["ref_chars"].values, ecfg["bootstrap_resamples"], ecfg["confidence_level"], seed)
        rows.append(row)
    return pd.DataFrame(rows)


def add_degradation(table: pd.DataFrame, clean: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    """Join the clean WER/CER of the *same utterances* (matched on `on`) and compute degradation.

    Relative degradation is undefined when the clean WER is 0: it is set to NaN and flagged in
    `rel_wer_undefined` instead of producing inf. Absolute degradation is still reported.
    """
    ref = clean.set_index(on)[["wer", "cer"]].rename(columns={"wer": "wer_clean", "cer": "cer_clean"})
    t = table.join(ref, on=on)
    t["abs_wer_degradation"] = t["wer"] - t["wer_clean"]
    t["rel_wer_undefined"] = t["wer_clean"] == 0
    t["rel_wer_degradation"] = np.where(t["rel_wer_undefined"], np.nan,
                                        t["abs_wer_degradation"] / t["wer_clean"].replace(0, np.nan) * 100)
    t["abs_cer_degradation"] = t["cer"] - t["cer_clean"]
    if t["rel_wer_undefined"].any():
        log.warning("Clean WER is 0 for %d rows; relative WER degradation set to NaN (see rel_wer_undefined)",
                    int(t["rel_wer_undefined"].sum()))
    return t


def clean_by_noise_category(scored: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Clean WER/CER of the utterances that were paired with each noise category.

    Each utterance is mixed with one category, so categories cover different utterances. The
    per-category degradation must use the clean score of those same utterances, not the overall one.
    """
    pairing = (scored.loc[scored["condition"] != "clean", ["model", "utt_id", "noise_category"]]
               .drop_duplicates())
    clean_rows = (scored[scored["condition"] == "clean"].drop(columns="noise_category")
                  .merge(pairing, on=["model", "utt_id"]))
    return summarize(clean_rows, ["model", "noise_category"], cfg, with_ci=False)


def clean_mismatch_candidates(scored: pd.DataFrame, min_cer: float) -> pd.DataFrame:
    """Clean utterances whose CER is >= min_cer for every model: likely transcript/audio mismatches.

    One model failing is a model error; all models failing on clean audio points at the data.
    """
    clean = scored[scored["condition"] == "clean"].assign(cer=lambda d: d["char_errors"] / d["ref_chars"] * 100)
    table = clean.pivot_table(index="utt_id", columns="model", values="cer")
    flagged = table[(table >= min_cer).all(axis=1)]
    refs = clean.drop_duplicates("utt_id").set_index("utt_id")["reference"]
    return flagged.add_prefix("clean_cer_").join(refs).reset_index()


def runtime_table(rtfx: pd.DataFrame, pred_dir, tag: str) -> pd.DataFrame:
    """RTFx next to the hardware/runtime configuration recorded by src.inference (run_<tag>_<model>.json)."""
    rows = []
    for model in rtfx["model"]:
        path = pred_dir / f"run_{tag}_{model}.json"
        meta = load_json(path) if path.exists() else {}
        env, desc = meta.get("environment", {}), meta.get("model", {})
        rows.append({"model": model, "checkpoint": desc.get("checkpoint"), "dtype": desc.get("dtype"),
                     "device": meta.get("device"), "gpu": env.get("gpu"), "batch_size": meta.get("batch_size"),
                     "peak_gpu_memory_gb": meta.get("peak_gpu_memory_gb"), "torch": env.get("torch"),
                     "transformers": env.get("transformers")})
    return rtfx.merge(pd.DataFrame(rows), on="model", how="left")


def make_plots(by_cond: pd.DataFrame, by_cat: pd.DataFrame, plots_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    order = [c for c in CONDITION_ORDER if c in set(by_cond["condition"])]
    labels = [c.replace("snr_", "").replace("dB", " dB") for c in order]
    for metric in ("wer", "cer"):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, g in by_cond.groupby("model"):
            g = g.set_index("condition").loc[order]
            ax.plot(labels, g[metric], marker="o", label=model)
            if f"{metric}_ci_low" in g:
                ax.fill_between(labels, g[f"{metric}_ci_low"], g[f"{metric}_ci_high"], alpha=0.15)
        ax.set_xlabel("Condition (SNR)")
        ax.set_ylabel(f"{metric.upper()} (%)")
        ax.set_title(f"{metric.upper()} vs SNR")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric}_vs_snr.png", dpi=150)
        plt.close(fig)

    noisy = by_cond[by_cond["condition"] != "clean"]
    if not noisy.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, col, title in zip(axes, ["abs_wer_degradation", "rel_wer_degradation"],
                                  ["Absolute WER degradation (pp)", "Relative WER degradation (%)"]):
            sns.barplot(data=noisy, x="condition", y=col, hue="model", order=order[1:], ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Condition")
            ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / "wer_degradation.png", dpi=150)
        plt.close(fig)

    if not by_cat.empty:
        models = sorted(by_cat["model"].unique())
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5), squeeze=False)
        for ax, model in zip(axes[0], models):
            pivot = by_cat[by_cat["model"] == model].pivot(index="noise_category", columns="condition", values="wer")
            pivot = pivot[[c for c in order if c in pivot.columns]]
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds", ax=ax, cbar=False)
            ax.set_title(f"{model}: WER (%)")
        fig.tight_layout()
        fig.savefig(plots_dir / "wer_heatmap_noise_category.png", dpi=150)
        plt.close(fig)


def evaluate(cfg: dict, tag: str, models: list[str] | None = None) -> None:
    pred_dir = project_path(cfg["paths"]["predictions_dir"])
    frames = [pd.read_csv(pred_dir / f"predictions_{tag}_{m}.csv", keep_default_na=False)
              for m in (models or MODEL_NAMES) if (pred_dir / f"predictions_{tag}_{m}.csv").exists()]
    if not frames:
        raise FileNotFoundError(f"No predictions_{tag}_*.csv in {pred_dir}. Run src.inference first.")
    pred = pd.concat(frames, ignore_index=True)
    failed = pred["hypothesis"].eq("")
    log.info("Loaded %d predictions for models %s (%d empty hypotheses)", len(pred),
             sorted(pred["model"].unique()), int(failed.sum()))
    for col in ("audio_sec", "inference_sec"):
        pred[col] = pd.to_numeric(pred[col], errors="coerce")

    # Fair comparison: keep only utterance/condition pairs present for every model.
    key = ["utt_id", "condition", "realization"]
    common = pred.groupby(key)["model"].nunique()
    common = common[common == pred["model"].nunique()].index
    pred = pred.set_index(key).loc[common].reset_index()

    scored = score_predictions(pred, cfg)
    by_cond = summarize(scored, ["model", "condition"], cfg)
    clean = by_cond[by_cond["condition"] == "clean"]
    by_cond = add_degradation(by_cond, clean, ["model"])
    by_cond["condition"] = pd.Categorical(by_cond["condition"], CONDITION_ORDER, ordered=True)
    by_cond = by_cond.sort_values(["model", "condition"]).reset_index(drop=True)
    by_cond["condition"] = by_cond["condition"].astype(str)

    by_cat = summarize(scored[scored["condition"] != "clean"], ["model", "noise_category", "condition"], cfg,
                       with_ci=False)
    if not by_cat.empty:
        by_cat = add_degradation(by_cat, clean_by_noise_category(scored, cfg), ["model", "noise_category"])

    mismatch = clean_mismatch_candidates(scored, cfg["evaluation"]["mismatch_clean_cer"])
    if not mismatch.empty:
        log.warning("%d clean utterances have CER >= %.0f%% for every model (possible transcript/audio "
                    "mismatch, see clean_mismatch_candidates.csv)", len(mismatch),
                    cfg["evaluation"]["mismatch_clean_cer"])

    rank = by_cond.pivot(index="condition", columns="model", values="wer").rank(axis=1).astype(int)
    rtfx = summarize(scored, ["model"], cfg, with_ci=False)[["model", "audio_sec", "inference_sec", "rtfx"]]
    rtfx = runtime_table(rtfx, pred_dir, tag)

    out_dir = ensure_dir(pred_dir / f"results_{tag}")
    scored.to_csv(out_dir / "scored_utterances.csv", index=False)
    by_cond.to_csv(out_dir / "summary_by_model_condition.csv", index=False)
    by_cat.to_csv(out_dir / "summary_by_model_noise_category.csv", index=False)
    rank.to_csv(out_dir / "model_rank_by_condition.csv")
    rtfx.to_csv(out_dir / "rtfx.csv", index=False)
    mismatch.to_csv(out_dir / "clean_mismatch_candidates.csv", index=False)
    runs = {m: str(pred_dir / f"run_{tag}_{m}.json") for m in scored["model"].unique()}
    save_json({"tag": tag, "normalization": cfg.get("text_normalization"), "run_metadata": runs,
               "n_common_rows": len(common)}, out_dir / "evaluation_info.json")

    plots_dir = ensure_dir(project_path(cfg["paths"]["plots_dir"]) / tag)
    make_plots(by_cond, by_cat, plots_dir)

    cols = ["model", "condition", "n_utterances", "wer", "cer", "abs_wer_degradation", "rel_wer_degradation"]
    log.info("Results (%s):\n%s", tag, by_cond[cols].round(2).to_string(index=False))
    log.info("RTFx:\n%s", rtfx[["model", "rtfx", "gpu", "batch_size", "dtype"]].round(2).to_string(index=False))
    log.info("Tables in %s, plots in %s", out_dir, plots_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--tag", default=None, help="Override the predictions tag, e.g. test_pilot_limit5")
    parser.add_argument("--models", nargs="+", default=None, choices=MODEL_NAMES,
                        help="Models to compare (default: every model with predictions). The primary "
                             "comparison uses exactly one Whisper checkpoint.")
    args = parser.parse_args()
    evaluate(load_config(args.config), args.tag or f"{args.split}{'_pilot' if args.pilot else ''}", args.models)


if __name__ == "__main__":
    main()
