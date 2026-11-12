"""Train the custom CNN-BiLSTM-CTC model from scratch on the official training split.

Validation split: checkpoint selection and early stopping. The test split is never touched here.

Usage:
    python -m src.train_custom_model [--max-epochs N] [--max-train-utterances N]
"""

from __future__ import annotations

import argparse
import math
import time

import jiwer
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.custom_cnn_bilstm_ctc import CharVocab, CustomASR, pad_batch, save_checkpoint
from src.noise_mixing import load_speech
from src.preprocessing import normalize_text
from src.utils import (ensure_dir, environment_info, get_logger, load_config, manifest_path,
                       project_path, resolve_device, save_json, set_seed)


class SpeechDataset(Dataset):
    def __init__(self, cfg: dict, rows: pd.DataFrame, vocab: CharVocab):
        self.cfg, self.rows, self.vocab = cfg, rows.reset_index(drop=True), vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        wave = load_speech(self.cfg, row["audio_path"])
        text = normalize_text(row["transcript"], self.cfg)
        return wave, torch.tensor(self.vocab.encode(text), dtype=torch.long), text


def collate(batch):
    waves, labels, texts = zip(*batch)
    return list(waves), list(labels), list(texts)


def load_split(cfg: dict, split: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path(cfg, "speech_manifest.csv"), keep_default_na=False)
    df = df[(df["official_split"] == split) & df["eligible_primary"].astype(str).str.lower().eq("true")]
    return df.sort_values("utt_id").reset_index(drop=True)


@torch.no_grad()
def evaluate(model: CustomASR, loader: DataLoader, device: str) -> dict:
    model.eval()
    refs, hyps = [], []
    for waves, _, texts in loader:
        wave, lengths = pad_batch(waves, device)
        log_probs, out_lengths = model(wave, lengths)
        hyps += model.greedy_decode(log_probs, out_lengths)
        refs += texts
    return {"wer": jiwer.wer(refs, hyps), "cer": jiwer.cer(refs, hyps), "example": (refs[0], hyps[0])}


def train(cfg: dict, max_epochs: int | None = None, max_train_utt: int | None = None) -> None:
    mcfg = cfg["models"]["custom_cnn_bilstm_ctc"]
    tcfg = mcfg["training"]
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = resolve_device(cfg["inference"]["device"])
    out_dir = ensure_dir(project_path(mcfg["checkpoint"]).parent)
    log = get_logger("train_custom", project_path(cfg["paths"]["logs_dir"]) / "train_custom_model.log")

    train_df, val_df = load_split(cfg, "train"), load_split(cfg, "val")
    dur = pd.to_numeric(train_df["duration_sec"], errors="coerce")
    train_df = train_df[dur <= tcfg["max_train_duration_sec"]]
    if max_train_utt or tcfg.get("max_train_utterances"):
        n = max_train_utt or tcfg["max_train_utterances"]
        train_df = train_df.sample(n=min(n, len(train_df)), random_state=seed)
    if tcfg.get("max_val_utterances") and len(val_df) > tcfg["max_val_utterances"]:
        val_df = val_df.sample(n=tcfg["max_val_utterances"], random_state=seed)
    log.info("Train utterances: %d | val utterances: %d | device: %s", len(train_df), len(val_df), device)

    vocab = CharVocab.from_texts(normalize_text(t, cfg) for t in train_df["transcript"])
    log.info("Vocabulary (%d incl. blank): %s", len(vocab), "".join(vocab.itos[1:]))

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(SpeechDataset(cfg, train_df, vocab), batch_size=tcfg["batch_size"], shuffle=True,
                              collate_fn=collate, num_workers=tcfg["num_workers"], generator=gen,
                              persistent_workers=tcfg["num_workers"] > 0)
    val_loader = DataLoader(SpeechDataset(cfg, val_df, vocab), batch_size=tcfg["batch_size"], shuffle=False,
                            collate_fn=collate, num_workers=tcfg["num_workers"])

    model = CustomASR(vocab, mcfg["features"], mcfg["architecture"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %.2fM", n_params / 1e6)
    optim = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])
    epochs = max_epochs or tcfg["max_epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=tcfg["learning_rate"], epochs=epochs,
                                                steps_per_epoch=max(1, len(train_loader)), pct_start=0.1)
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    use_amp = bool(tcfg["amp"]) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    metric = tcfg["selection_metric"]
    best, patience, history = math.inf, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for step, (waves, labels, _) in enumerate(train_loader, 1):
            wave, lengths = pad_batch(waves, device)
            targets = torch.cat(labels).to(device)
            target_lengths = torch.tensor([len(l) for l in labels], device=device)
            with torch.autocast(device_type=device, enabled=use_amp):
                log_probs, out_lengths = model(wave, lengths)
            loss = ctc(log_probs.float().transpose(0, 1), targets, out_lengths, target_lengths)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optim)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            if step % 200 == 0:
                log.info("epoch %d step %d/%d loss %.4f", epoch, step, len(train_loader), np.mean(losses[-200:]))

        val = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_wer": val["wer"],
                  "val_cer": val["cer"], "epoch_sec": round(time.time() - t0, 1)}
        history.append(record)
        log.info("epoch %d | loss %.4f | val WER %.4f | val CER %.4f | ref=%r hyp=%r", epoch,
                 record["train_loss"], val["wer"], val["cer"], *val["example"])

        score = val[metric]
        if score < best:
            best, patience = score, 0
            save_checkpoint(out_dir / "best.pt", model, mcfg,
                            {"epoch": epoch, "val": {k: val[k] for k in ("wer", "cer")}, "seed": seed})
            log.info("New best val %s %.4f -> saved %s", metric, score, out_dir / "best.pt")
        else:
            patience += 1
            if patience >= tcfg["early_stopping_patience"]:
                log.info("Early stopping after %d epochs without improvement", patience)
                break
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    save_json({"best_val_" + metric: best, "epochs_run": len(history), "n_train": len(train_df),
               "n_val": len(val_df), "n_params": n_params, "config": mcfg, "environment": environment_info()},
              out_dir / "training_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--max-train-utterances", type=int, default=None)
    args = parser.parse_args()
    train(load_config(args.config), args.max_epochs, args.max_train_utterances)


if __name__ == "__main__":
    main()
