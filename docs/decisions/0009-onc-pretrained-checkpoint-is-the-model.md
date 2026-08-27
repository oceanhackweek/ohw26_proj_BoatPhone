# 0009. ONC's pretrained checkpoint is the vessel-presence model

Status: superseded by 0012
Date: 2026-08-27

**Superseded 2026-08-27 by `docs/decisions/0012-b0-model-viability-outcome.md`.** B0 found the
premise refuted: no CPU CNN baseline exists, the released checkpoint has no distinct Engine Noise
output, and it is not CPU-viable at the corpus scale needed. Body left otherwise intact below --
the VTUAD-rejection reasoning in "Context" is unaffected and still load-bearing. Three factual
errors are corrected inline where they occur, each pointing at 0012.

## Context

`docs/plans/accoutics_plan.md` §A8 planned to download VTUAD, fit a linear head on frozen
PANNs/VGGish embeddings, and measure a "transfer gap" from AIS-labelled commercial traffic to the
small recreational craft at Folger. Two findings killed that design.

**First, VTUAD does not support the claim.** It carries **no vessel size or length label** -- only
5-class AIS ship *type* crossed with 3 coarse range bins -- so the size-stratified transfer gap
§A8 promised cannot be computed at all. Its audio is **uncalibrated** raw PCM where Folger's is
calibrated dB re 1 uPa, so any absolute-level gap would be a calibration artefact indistinguishable
from domain shift. Its `background` clips silently contain non-AIS small craft, which is label
noise pointed directly at our target population. And it sits behind a **paid IEEE DataPort
Subscriber tier with no download API**, not the overnight scripted download §A8 assumed. Facts,
with sources and retrieval dates, in `docs/vtuad-facts.md`.

**Second, and more fundamental: no public AIS-trained acoustic model weights exist.** Verified
2026-08-27 across Hugging Face, Zenodo, GitHub releases, and the marine-acoustics ecosystem
(Ketos, PAMGuard, NOAA, Orcasound, MERIDIAN):

* Domingos et al. 2022 -- VTUAD's own author -- releases MIT training code
  ([`underwater_snd`](https://github.com/lucascesarfd/underwater_snd),
  [`onc_dataset`](https://github.com/lucascesarfd/onc_dataset)) but **no checkpoints**.
* Decrop et al., IEEE JSTARS 2025 (`10.1109/JSTARS.2025.3593779`) releases an open dataset
  (`10.14284/723`) but **no code and no weights**.
* Renaud et al. (`arXiv:2607.13840`) is **method-only** -- no data, no code.

So the "AIS-trained model" in the research question was always going to be one we built. That makes
a poor transfer result ambiguous between "AIS-trained methods do not transfer" and "we built a weak
model" -- exactly the confusion invariant 9 exists to prevent, and there was no in-domain reference
available to resolve it.

## Decision

**Use Ocean Networks Canada's own pretrained checkpoint as the vessel-presence model, and reproduce
its published performance before trusting it.**

* Source: [`OceanNetworksCanada/selfsupervision_anomalies_onc`](https://github.com/OceanNetworksCanada/selfsupervision_anomalies_onc),
  **MIT**, PyTorch. Checkpoints public on Hugging Face under `merileo/*`, MIT-tagged.
  **[CORRECTION, see 0012f: the repo's actual `LICENSE` file is BSD-3-Clause, not MIT; both are
  permissive and this did not change the decision.]**
* Signal: the **`Engine Noise`** class. Full label set: `Anomaly, Data Gap, Dropout, Engine Noise,
  Rain, Sensitivity, Tonal, Unknown Feature`.
  **[CORRECTION, see 0012b: the released checkpoint has a single-logit binary head; Engine Noise
  is pooled with seven other anomaly types into one "anomalous" output, not a distinct class.]**
* **Use the CPU CNN baseline** (`cnn_baseline/cnn_best.pt`) with `eval/evaluate_model.py`. The
  **[CORRECTION, see 0012a: no CNN baseline exists under any `merileo/*` repo -- live `HfApi`
  enumeration found none; this was a literature-sweep error, not a since-removed artefact.]**
  SSAMBA/Mamba path needs NVIDIA plus `mamba_ssm`; this environment has no GPU
  (`torch.cuda.is_available()` is False).
* **Reproduce first.** Before scoring any Folger data, run the checkpoint over ONC's public
  labelled eval set restricted to `ICLISTENHF1266` rows and reproduce the reported `Engine Noise`
  performance. If it does not reproduce, that is "the method is broken" -- say so and fall back.
* **Class ordering is read from `args.pkl` / the h5 `label_names`, never assumed** from the YAML.
  The model cards are a bare `license: mit` with no documented ordering.
* **VTUAD is not acquired** -- not v1, not v2, not via `onc_dataset` regeneration.

Why this model rather than the alternatives:

* **`ICLISTENHF1266` -- our exact device -- is in its training/eval device list**, with per-device
  figures in `result.csv` (reported P 0.96 / R 0.88 / AUC 0.99 on 149 samples; epoch-1 numbers,
  unverified until reproduced). A device-matched in-domain reference is strictly better than the
  cross-domain one VTUAD would have supplied, and it is free.
* **It consumes ONC spectrogram arrays, not WAV**, so `.fft.gz` is the native input class and the
  ~3.5 TB WAV pull that every other candidate implied does not arise.
* Its positive label is generic engine noise from BC coastal ONC sites, so **recreational traffic
  is inside the positive class by construction** -- the opposite of VTUAD's exclusion problem.

Considered and rejected: **UATR-CMoE** (released weights, classes include `Motorboat`/`Sailboat`,
but **no declared licence** and ShipsEar content is heavily sub-1 kHz); **PANNs/AudioSet zero-shot**
(class 304 is literally `Motorboat, speedboat`, free, but no published underwater performance
anywhere). Both remain on the bench as fallbacks.

## Consequences

* **A hard dependency on a third party's checkpoint.** If the weights disappear, pin the file hash
  and keep a local copy outside `data/`.
* **The model gives presence, not count, size or range.** Goals G2 and G3 still need the matchups
  (B7) and the physics baseline (B6); this decision does not deliver them.
* **The transfer-gap result is gone.** It was never in `Project_Source_of_Truth.txt` -- it lived
  only in the acoustics plan -- so no project goal is lost. Say so plainly rather than reporting a
  weaker version of it.
* **`Engine Noise` is an anomaly-detection label, not a vessel label.** It may fire on other
  engines (ONC's own ROV/ship operations) and may miss a drifting sailboat entirely. The B5
  band-level detector is the independent check on exactly this.
* **How to tell this was wrong:** the reproduction gate fails, or `Engine Noise` shows no
  separation against the B5 detector and against optical labels. Either sends us to B5 as primary,
  which needs no weights at all.

Supersedes `docs/plans/accoutics_plan.md` §A8. See `docs/plans/acoustics_plan_v2.md` §4 and §B0/B4.
