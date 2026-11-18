# Evaluating Deep Learning Models for Noise-Robust Indonesian Speech Recognition

Comparative study of Indonesian automatic speech recognition (ASR) under environmental noise:
**clean vs. noisy speech** at controlled signal-to-noise ratios (SNR).

| Model | Input path | Role |
|---|---|---|
| Custom CNN-BiLSTM-CTC | 80-bin log-Mel filterbank (frozen in `config/config.yaml`) | Trained from scratch on the official training split |
| [Wav2Vec2 XLSR Indonesian](https://huggingface.co/indonesian-nlp/wav2vec2-large-xlsr-indonesian) | Raw waveform via the checkpoint's processor | Pretrained, evaluated without fine-tuning |
| [Whisper Large-v3](https://huggingface.co/openai/whisper-large-v3) | Log-Mel via the checkpoint's processor, `language=indonesian`, `task=transcribe` | Pretrained, evaluated without fine-tuning |

Conditions: **Clean, 20 dB, 10 dB, 5 dB, 0 dB**. Metrics: WER, CER, absolute/relative WER degradation, RTFx.

> Status: implementation of the research plan (v5). Data audit and pilot mixing are done; no ASR results are
> reported until the full runs are complete.
> The local demo UI (`app/`) is intentionally deferred until the core pipeline is stable.

## Research questions

**Primary:** How does environmental noise at different SNR levels affect Indonesian ASR performance, and which evaluated model is most robust?

1. How large is the WER/CER increase from clean to noisy speech?
2. Does the ranking of models change across SNR levels?
3. Do different environmental noise categories affect the models differently?

## Datasets

| Dataset | Used for | Not used for |
|---|---|---|
| [Indonesian-ASR-11-Class-Dataset](https://huggingface.co/datasets/Atika88/Indonesian-ASR-11-Class-Dataset) (CC BY 4.0) | Indonesian speech, reference transcripts, speaker metadata, official speaker-disjoint `train/val/test` splits | Noise bank |
| [MS-SNSD](https://github.com/microsoft/MS-SNSD) (`noise_train/`, `noise_test/`) | Environmental (non-speech) noise for controlled mixing | Speech or Indonesian transcripts |

- `metadata/dataset_metadata_public.csv` is the source of truth for paths, transcripts, speakers and splits. Official splits are never reshuffled.
- The 132 synthetic repair files (`is_synthetic == True`) are excluded from the primary evaluation (`speech_audit.exclude_synthetic`).
- Speech-like noise categories (`Babble`, `NeighborSpeaking`, `AirportAnnouncements`) are excluded. In `noise_test` the neighbour recordings are named `Neighbor_*.wav`; they are mapped to `NeighborSpeaking` and excluded too. Ambience categories that may contain background talk (Cafe, Restaurant, Station, ...) stay in `review` and are not used; they only exist in `noise_train`.
- Music is out of scope (`noise_audit.include_music: false`); the study covers non-speech environmental noise only.
- Checkpoints were verified on Hugging Face before use: both accept 16 kHz mono input through their official processor, list Indonesian (`id`) as a supported language, and are openly licensed (`indonesian-nlp/wav2vec2-large-xlsr-indonesian` Apache-2.0, `openai/whisper-large-v3` Apache-2.0; the documented fallback `openai/whisper-large-v3-turbo` is MIT).
- Evaluation uses the `noise_test` pool. `noise_train` is reserved for the optional noise-augmentation experiment and for the validation development subset (same six categories). No noise file appears in both pools, and test noise is never used for a protocol or checkpoint decision.
- The corpus is prompted/read speech with a fixed sentence inventory (213 distinct sentences, and every test sentence also appears in the training split); conclusions are limited to this corpus and the selected noise conditions.

### Audit results

Audit run on all 104,500 speech files and 179 noise files (`speech_audit_summary.json`, `noise_manifest.csv`).

| Split | Speakers | Files | Invalid | Eligible for the primary experiment |
|---|---|---|---|---|
| train | 28 | 73,150 | 41 | 72,987 |
| val | 6 | 15,675 | 2 | 15,665 |
| test | 3 human (M4, M8, M12) + 1 synthetic | 15,675 | 0 | 15,673 |

- Every file is readable, non-empty, 16 kHz mono and has a transcript. Duration: median 4.5 s, 99.9th percentile 11.3 s, maximum 206 s.
- Files outside the audit thresholds were flagged `review` and resolved by the documented rules in `speech_audit.auto_review`
  instead of per-file listening:
  - *Clipping* (1,138 files, mild: median 0.25 % clipped samples, concentrated in a few recording sessions): accepted below 1 %, excluded at 1 % or more (32 files, all in train).
  - *Duration 1.5–15 s or speaking rate 5–30 characters/s* (26 files): the speaking rate is recomputed over the active-speech span. Between 3 and 30 characters/s the file is accepted (it only had extra silence); otherwise it is excluded (11 files: extra or wrong content, or truncated audio).
- `audit_flags` lists every automatic flag a file received. `exclusion_reason` is filled only for files that are not valid.
  Decisions in `manual_review_decisions.csv` override the rules.
- Noise used for evaluation: 31 `noise_test` files in 6 categories (AirConditioner 10, Munching 8, ShuttingDoor 8,
  CopyMachine 2, Typing 2, VacuumCleaner 1). Byte-identical noise files are detected: `noise_train/AirConditioner_2` and
  `noise_train/SqueakyChair_3` are the same recording with different labels, so both are excluded.

Raw data are **not** committed. See [`data/README.md`](data/README.md).

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate          # Windows (source .venv/bin/activate elsewhere)
# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python -m pytest -q tests                                  # unit tests, no data or GPU needed
```

Tested environment of the pilot runs (also written to `output/predictions/run_*.json`):

| Component | Version |
|---|---|
| OS / CPU | Windows 11, AMD64 (AMD Family 25) |
| GPU | NVIDIA GeForce RTX 3050 Laptop, 4.29 GB, CUDA 12.6 |
| Python | 3.12.16 |
| torch / torchaudio | 2.11.0+cu126 / 2.11.0+cu126 |
| transformers | 5.23.0 |
| numpy / pandas | 2.5.5 / 3.1.1 |
| librosa / soundfile | 1.0.1 / 0.14.0 |
| jiwer | not recorded in the pilot (fixed: versions now come from package metadata) |

`requirements.txt` holds minimum versions. Exact pins (`pip freeze > requirements-lock.txt`) are made on the
machine that runs the full evaluation, once that environment is confirmed.

## Repository structure

```
data/
  raw/indonesia_asr/        # HF dataset snapshot + extracted audio (not committed)
  raw/ms_snsd/              # sparse clone of MS-SNSD noise folders (not committed)
  processed/manifests/      # audit, split, pairing and SNR manifests
  processed/pilot_cache/    # optional pilot WAVs for listening (not committed)
notebooks/
  exploration.ipynb         # dataset audit and EDA
  experiments.ipynb         # small-scale experiments
src/
  data_loader.py            # download, metadata, audio loading, raw manifests
  audio_validation.py       # integrity, duration, silence/RMS, clipping, valid/review/invalid
  preprocessing.py          # audio standardization, deterministic text normalization
  noise_mixing.py           # noise selection, SNR mixing, achieved-SNR verification
  model_loader.py           # all models (incl. two Whisper checkpoints) behind one transcribe() interface
  custom_cnn_bilstm_ctc.py  # custom model architecture
  train_custom_model.py     # training on the official train split (val for selection)
  inference.py              # predictions, runtime and configuration per model
  evaluate.py               # WER, CER, degradation, RTFx, tables and plots
  utils.py                  # config, seeds, logging, environment info
config/config.yaml          # all experiment settings
output/{models,logs,predictions,plots}/
tests/                      # unit tests: normalization, SNR mixing, metrics, config consistency
app/                        # later phase: local demo UI
report/
```

## Pipeline

Run all commands from the repository root.

```bash
# 1. Download data
python -m src.data_loader download-speech                      # all categories (~134 h of audio)
#    or: python -m src.data_loader download-speech --categories Declarative   (quick start)
python -m src.data_loader download-noise

# 2. Build raw manifests (speech + noise) and check speaker-disjoint splits
python -m src.data_loader build-manifests

# 3. Audit. Inspect the duration distribution first, then set thresholds in config.yaml
python -m src.audio_validation --target speech --summary-only
python -m src.audio_validation --target all                    # ~13 min for the full corpus
python -m src.audio_validation --target noise-speech           # screen valid noise files for speech (Wav2Vec2)
#    Manual listening (plan 5): export the files to listen to, fill decision (valid/invalid) and note in
#    data/processed/listening/listening_sheet.csv, apply, then re-run the audit and commit the filled sheet.
python -m src.audio_validation --target listening-sheet
python -m src.audio_validation --target apply-listening
python -m src.audio_validation --target all
#    "review" speech files are resolved by speech_audit.auto_review. To override a decision, add a row to
#    data/processed/manifests/manual_review_decisions.csv  (manifest,id,decision,note) and re-run the audit.

# 4. Pilot: 20-50 test utterances, all SNRs
python -m src.noise_mixing build-pairs --split test --pilot
python -m src.noise_mixing export-pilot                         # WAVs for listening
python -m src.inference --pilot --models wav2vec2 whisper       # pretrained models first (pipeline/hardware check)
python -m src.evaluate --pilot --models wav2vec2 whisper
#    Whisper stays Large-v3 as planned. Only if the hardware cannot run it, follow
#    report/whisper_model_selection.md (plan 7.3; checked on a val development subset, never on test results).

# 5. Train the custom model (train split; val for early stopping/checkpoint selection)
python -m src.train_custom_model
#    then complete the pilot with all three systems on the same inputs (plan 11.5)
python -m src.inference --pilot --models custom_cnn_bilstm_ctc
python -m src.evaluate --pilot --models custom_cnn_bilstm_ctc wav2vec2 whisper

# 6. Freeze config.yaml, then run the full comparison
python -m src.noise_mixing build-pairs --split test
python -m src.inference
python -m src.evaluate
```

## Experimental protocol

**Noise mixing** (`src/noise_mixing.py`)
- `SNR(dB) = 10·log10(P_speech / P_noise)`. Speech power is measured over active-speech frames (25 ms / 10 ms frames within 40 dB of the loudest frame). Noise power is measured over the full added segment.
- The noise segment is cropped (or tiled if shorter) from a seeded offset. The same noise file and offset are reused for every SNR of an utterance.
- Noise categories are assigned round-robin so each category gets a similar number of utterances.
- Clipping protection (`mixing.gain_scope: utterance`): impulsive noise (Typing, CopyMachine, Munching) can push a mixture above full scale. One output gain per utterance, taken from its loudest mixture, is applied to speech and noise in all five conditions, clean included. The SNR is unchanged and the speech component is identical across conditions, so conditions differ only in noise level. In the pilot, 15 of 30 utterances were attenuated (median −0.5 dB, at most −17.9 dB). The gain is recorded per row.
- The achieved SNR is recalculated for every mixture and checked against a 0.1 dB tolerance.
- Mixing is deterministic and on the fly. The conditions manifest stores the noise file, offset, seed, target/achieved SNR, gain and clipping flags.

**Fair comparison**
- All models use the same test utterances, noise segments, SNR targets and normalization. `evaluate.py` scores only rows present for every model.
- The pretrained models are not fine-tuned in the primary experiment. The custom model uses only the training split. The validation split is used for checkpoint selection, and the test set is not used for tuning.
- Every noisy variant stays in its utterance's official split.
- Whisper is Large-v3 as planned. If the hardware requires a smaller checkpoint (plan 7.3), the change is checked on a validation development subset (val speech + `noise_train` noise), never on test results, and documented.

**Custom CNN-BiLSTM-CTC decisions** (frozen before training; `config.yaml` → `models.custom_cnn_bilstm_ctc`, `src/custom_cnn_bilstm_ctc.py`)
- Input: 80-bin log-Mel filterbank, 16 kHz, 25 ms window (`n_fft` 400), 10 ms hop, per-utterance normalization.
- Tokens: character vocabulary built from the normalized training transcripts; index 0 is the CTC blank.
- Acoustic model: two 3×3 Conv2d layers (32, 32 channels; stride (2, 2) then (2, 1), BatchNorm, GELU; strided convolution instead of pooling, so time is downsampled 2×), dropout 0.2, a linear projection to 256, then a 3-layer BiLSTM (256 hidden per direction), linear output.
- Padding: packed sequences with per-utterance output lengths. Decoding: CTC greedy.
- Training: AdamW (lr 5e-4, weight decay 1e-5), one-cycle schedule, batch 16, gradient clipping 5.0, mixed precision, at most 50 epochs, early stopping after 5 epochs without improvement in CER on a fixed-seed 2,000-utterance validation subset.

**Text normalization** (`preprocessing.normalize_text`, used for both references and hypotheses): NFKC, lowercase, `%` → `persen`, digits → Indonesian words (`2018` → `dua ribu delapan belas`), hyphens → spaces, accent stripping, punctuation removal, whitespace collapse. No stemming or lemmatization. Original transcripts are never modified.

**Timing:** `time.perf_counter` around model inference with CUDA synchronization. Warm-up batches and waveform generation are excluded. The hardware and software versions are written to `output/predictions/run_*.json`.

## Outputs

- `output/predictions/predictions_<tag>_<model>.csv`: per-utterance hypotheses and timing
- `output/predictions/results_<tag>/summary_by_model_condition.csv`: WER/CER with 95% bootstrap CI, ΔWER, relative degradation, RTFx
- `output/predictions/results_<tag>/rtfx.csv`: RTFx per model with checkpoint, precision, device, GPU, batch size, peak GPU memory and library versions
- `output/predictions/results_<tag>/accuracy_vs_speed.csv` and `output/plots/<tag>/accuracy_vs_speed.png`: WER per condition against RTFx (accuracy-versus-speed trade-off)
- `data/processed/manifests/noise_speech_check.csv`: noise files screened for speech (transcribed characters per second)
- `data/processed/listening/listening_sheet.csv`: files selected for manual listening with the reviewers' decisions (the copied WAVs are not committed)
- `output/predictions/results_<tag>/summary_by_model_noise_category.csv`
- `output/predictions/results_<tag>/model_rank_by_condition.csv`
- `output/predictions/results_<tag>/clean_mismatch_candidates.csv`: possible transcript/audio mismatches (reported, not excluded)
- `output/plots/<tag>/`: WER vs SNR, CER vs SNR, WER degradation, WER heatmap by noise category

## Results

Results template. Values are filled in from actual runs only.

| Model | Clean | 20 dB | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|---|
| Custom CNN-BiLSTM-CTC | WER/CER | WER/CER | WER/CER | WER/CER | WER/CER |
| Wav2Vec2 XLSR Indonesian | WER/CER | WER/CER | WER/CER | WER/CER | WER/CER |
| Whisper Large-v3 | WER/CER | WER/CER | WER/CER | WER/CER | WER/CER |

## Limitations

- This is a practical comparison between a custom model trained on this corpus and two pretrained models used without fine-tuning, not an architecture-only comparison with identical training exposure. Fine-tuning or noise-augmented training, if done later, is reported as a separate secondary experiment.
- The speech is prompted/read, and every test sentence also occurs in the training split. The custom model has therefore seen the test sentences (spoken by other speakers), while the pretrained models have not. These results do not measure open-vocabulary or spontaneous-speech performance.
- The test split has only three human speakers, all male (M4, M8, M12). Results may not transfer to female speakers.
- The noise conditions are limited to six audited MS-SNSD categories, one of which (VacuumCleaner) has a single recording.
- Listening (plan 5): until the listening sheet is filled, noise categories and speech review cases are resolved without listening. Speech-like noise is excluded by category, and speech review cases are resolved by the documented rules above. `--target listening-sheet` exports every evaluation noise file, 2 `noise_train` files per evaluation category, every rule-resolved duration/speaking-rate case and rejected clipping case, 10 accepted clipping cases and 10 eligible utterances per split; the decisions recorded there override the rules. If the sheet is not filled before the full run, this remains a deviation from the plan and is reported as such. Transcript/audio mismatches are checked automatically: `src.evaluate` lists clean utterances whose CER is at least `evaluation.mismatch_clean_cer` (30 %) for every model in `clean_mismatch_candidates.csv` (pilot: none). Speech inside the noise files is screened automatically with Wav2Vec2 (`python -m src.audio_validation --target noise-speech`, report in `noise_speech_check.csv`); suspected files are reviewed and excluded through `manual_review_decisions.csv`.
- Whisper Large-v3 fits on the 4 GB RTX 3050 only at batch size 1 (peak 3.36 GB) and runs slower than real time (pilot RTFx 0.85), about 121 GPU-hours for the full test set. The full run therefore needs a larger GPU for Large-v3; if none is available, the plan's fallback (a smaller checkpoint, plan 7.3) and its justification are recorded in [`report/whisper_model_selection.md`](report/whisper_model_selection.md).

## References

- Indonesian-ASR-11-Class-Dataset, DOI [10.57967/hf/10345](https://doi.org/10.57967/hf/10345)
- C. K. A. Reddy et al., "A Scalable Noisy Speech Dataset and Online Subjective Test Framework," Interspeech 2019 (MS-SNSD)
- [indonesian-nlp/wav2vec2-large-xlsr-indonesian](https://huggingface.co/indonesian-nlp/wav2vec2-large-xlsr-indonesian)
- [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
- [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- [JiWER](https://github.com/jitsi/jiwer)
