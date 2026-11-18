# Whisper checkpoint selection

Decision record for which Whisper checkpoint is used in the full comparison.

## Problem

The pilot (30 test utterances x 5 conditions = 150 rows, RTX 3050 Laptop 4 GB) ran
`openai/whisper-large-v3` in fp16 with batch size 1:

| Model | WER clean | WER 0 dB | RTFx | Peak GPU memory |
|---|---|---|---|---|
| Whisper large-v3 | 6.35 % | 21.40 % | 0.85 | 3.36 GB of 4.29 GB |
| Wav2Vec2 XLSR Indonesian | 30.10 % | 61.87 % | 22.28 | - |

Accuracy is not the problem; throughput is. The full test set is 15,673 eligible utterances x 5
conditions = 78,365 rows. At the pilot's mean duration (4.76 s) that is about 104 h of audio, so
large-v3 at RTFx 0.85 needs roughly **121 GPU-hours** on this laptop (Wav2Vec2: about 5 h).
Batch 4 does not fit in 4 GB (4.14 GB, spills into shared memory, 2-3x slower).

## Candidates

| Checkpoint | Indonesian accuracy in noise | Memory (fp16 weights) | Speed | Verdict |
|---|---|---|---|---|
| `whisper-large-v3` (1.55B) | Best measured (pilot above) | ~3.1 GB, batch 1 only on 4 GB | Slowest: 32 decoder layers | Reference |
| `whisper-large-v3-turbo` (809M) | Expected close to large-v3 (same encoder) | ~1.6 GB, batch 4-8 expected to fit | Expected several times faster: 4 decoder layers | **Primary candidate** |
| `whisper-medium` (769M) | Below large-v3 | ~1.5 GB | Slower than turbo: 24 decoder layers | Dominated by turbo |
| `whisper-small` (244M) | Clearly lower, worst at low SNR | < 1 GB | Fast | Not a fair "best Whisper" |
| `distil-large-v3` | - | - | - | English-only, not usable |
| faster-whisper / CTranslate2 | Near-identical to its source checkpoint | Lower (int8) | 2-4x faster | Different runtime than the other models; RTFx not comparable |
| Community Indonesian fine-tunes | Possibly higher | - | - | Violates "pretrained, no fine-tuning"; training data may overlap the corpus |

Rationale for turbo: the encoder, where most noise robustness comes from, is unchanged from
large-v3; only the decoder is pruned (32 to 4 layers) and fine-tuned. The corpus is read speech with a
fixed sentence inventory, so it relies little on the decoder's language modelling. This is a
**hypothesis**; the pilot below decides.

## Local vs. cloud

- **Hosted APIs** (OpenAI, Groq, ...): not used. The served model version is opaque and can change,
  decoding cannot be fully controlled, and network latency makes RTFx incomparable with the local models.
- **Cloud GPU** (Kaggle / Colab T4 16 GB): fallback for running large-v3 on the full set with the same
  code and checkpoint. The GPU is recorded in `run_*.json`; RTFx is then reported per hardware.
- **Local**: primary, so all models are timed on the same hardware.

## Decision rule

1. Run the pilot for turbo on the same 150 rows:
   `python -m src.inference --pilot --models whisper_turbo`, then `python -m src.evaluate --pilot`.
   `whisper` (large-v3) and `whisper_turbo` write separate prediction files, so the large-v3 pilot is kept.
2. If turbo's WER lies inside large-v3's 95 % bootstrap CI in every condition (0 dB in particular),
   use `whisper_turbo` for the full run and report large-v3 as a pilot-only reference.
3. Otherwise, run large-v3 on the full set on a cloud GPU.
4. Record the outcome and the measured turbo RTFx / peak memory in this file and in the README.

## Outcome

Pending: the turbo pilot has not been run yet.
