# Passive–Active Speaker Verification: Testing Protocols

A lightweight toolkit for running **passive** and **active** speaker‑verification (SV) experiments and evaluating them with standard metrics (EER, minDCF). Utilities are included for signal‑quality features (SNR/C50) and basic VAD to help you control test conditions.

---

## Why this exists

Reproducible SV evaluation depends on *how* you probe the system:

* **Passive testing**: score pre‑recorded trials with no interaction.
* **Active testing**: interactively collect or select trials (e.g., prompt for targeted phrases, re‑tries under different SNRs) before scoring.

This repo provides both flows so you can compare protocols under the same codebase and metrics.

---

## Features

* 🔁 **Two protocols**: `passive.py` and `active.py` pipelines.
* 🎯 **Verification utilities**: scoring helpers in `verification.py`.
* 📉 **Metrics**: Equal Error Rate (EER) via `compute_EER.py`, minimum DCF via `compute_min_dcf.py`.
* 🔊 **Signal helpers**: SNR/C50 computation + simple VAD in `snr_c50_vad.py`.

---

## Installation

This is a plain‑Python repo. Create an environment and install common audio/ML deps:

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install numpy scipy scikit-learn librosa soundfile tqdm matplotlib
```

> Tip: run `python <script>.py -h` to see the exact CLI flags each script supports.

---

## Data layout (example)

You can use any SV dataset with **enrollment** and **test** utterances (e.g., VoxCeleb, in‑house data). A minimal structure:

```
data/
  enroll/
    spk1/ utt1.wav, utt2.wav ...
    spk2/ ...
  test/
    spk1/ xxxxx.wav
    spk3/ ...
```

Prepare a **trials file** mapping enrollment ↔ test pairs if your protocol requires it (format up to your script arguments).

---

## Quickstart

### 1) Passive protocol

Scores fixed trials without interaction.

```bash
python passive.py \
  --enroll-dir data/enroll \
  --test-dir data/test \
  --trials trials.tsv \
  --out runs/passive/scores.tsv
```

### 2) Active protocol

Collect/curate trials interactively or by rule before scoring.

```bash
python active.py \
  --enroll-dir data/enroll \
  --test-dir data/test \
  --strategy prompt \
  --out runs/active/scores.tsv
```

> Use `-h` for the actual options supported by your scripts.

---

## Evaluate

Compute **EER**:

```bash
python compute_EER.py --scores runs/passive/scores.tsv --out runs/passive/eer.txt
```

Compute **minDCF** (set target prior & costs as needed):

```bash
python compute_min_dcf.py \
  --scores runs/passive/scores.tsv \
  --p-target 0.01 --c-miss 1 --c-fa 1 \
  --out runs/passive/min_dcf.txt
```

---

## Signal quality & VAD (optional)

Estimate SNR / C50 and run a simple VAD to filter frames:

```bash
python snr_c50_vad.py --in data/test --out features/test_quality.json
```

You can then condition your active strategy on these measures (e.g., re‑record if SNR < threshold).

---

## Reproducibility

* Set seeds (if the scripts expose `--seed`).
* Fix sample rate (e.g., 16 kHz) and channel layout (mono).
* Log exact CLI invocations to `runs/<...>/args.json`.

---

## Results (template)

| Protocol | EER (%) | minDCF\@p=0.01 | Notes             |
| -------: | ------: | -------------: | ----------------- |
|  Passive |       – |              – | baseline          |
|   Active |       – |              – | prompt‑then‑retry |

> Fill in after you run `compute_EER.py` / `compute_min_dcf.py`.

---

## Suggested repository structure

```
.
├── active.py                # Active‑testing pipeline
├── passive.py               # Passive‑testing pipeline
├── verification.py          # Core verification utilities
├── compute_EER.py           # Equal Error Rate calculator
├── compute_min_dcf.py       # minDCF calculator
├── snr_c50_vad.py           # SNR/C50 estimators + simple VAD
└── README.md
```

---

## Contributing

Issues and PRs welcome. Please include:

* dataset description,
* exact command lines,
* environment details (`python -V`, `pip freeze`).

---

## License

*Add a LICENSE file (MIT/Apache‑2.0/BSD‑3‑Clause are common).*

---

## Citation

If this toolkit supports a paper/report, add your BibTeX here.
