# 0012. B0 outcome: NO-GO on the pretrained-checkpoint path; B5 promoted to primary

Status: accepted
Date: 2026-08-27

## Context

Decision `0009` committed to Ocean Networks Canada's pretrained `selfsupervision_anomalies_onc`
checkpoint as the vessel-presence model, gated on `acoustics_plan_v2.md` §4/§B0: acquire the
artefacts, quantify the `.fft.gz`-vs-`.mat` shape gap, confirm `ICLISTENHF1266` in `result.csv`,
recover class ordering from `args.pkl`/h5 rather than assuming it, and run one CPU forward pass.
GO required all five. B0 ran end to end (`docs/derived/b0_external_provenance.json`, commit
`e1c1544`), followed by an adversarial re-check that re-derived every artefact-level claim from
the files themselves rather than trusting the ledger. Both runs agree: **NO-GO**, on two blockers
independently sufficient on their own, plus five subsidiary findings that correct 0009 without
changing the verdict.

## Decision

**Do not build on the ONC pretrained checkpoint. Close plan v2 §4/B2/B4. Promote B5 (band-level
SPL, no weights) from "B4's independent check" to the primary detector.**

### a. No CNN baseline exists

0009 and plan v2 §4 both specify `cnn_baseline/cnn_best.pt` run through `eval/evaluate_model.py`
as the CPU-viable path. Live enumeration of every `merileo/*` repo via `HfApi` (not a cached
listing) found no `cnn_baseline` directory, no `cnn_best.pt`, and no "CNN" anywhere in the cloned
source (`onc_ssamba` @ `e3aebfcbf199e2e7ef6aeb39b0a871ce4bb6e920`). Only SSAMBA/Vision-Mamba
`.pth` checkpoints exist. This was a **literature-sweep error in §4**, not a since-removed
artefact -- the repo and Hub state agree there was never a CNN baseline to acquire.

### b. The only released checkpoint is a single-logit binary detector, not a multi-class one

`mlp_head.1.weight` in the finetuned checkpoint (`ft-cls_best_checkpoint.pth`,
`merileo/finetune-amba-base-f16-t16-b16-lr1e-4-m300-custom-tr0.8-full_dataset_hydrophones-noexclude`,
sha256 `e474ddc5...`) has shape `(1, 768)` -- **one output logit**. It is a binary
normal-vs-anomalous detector. `Engine Noise` is one of eight labels pooled into the single
"anomalous" class (`Anomaly, Data Gap, Dropout, Engine Noise, Rain, Sensitivity, Tonal, Unknown
Feature`), recovered by cross-tabulating the eval h5's `labels` array against `label_strings`
since neither `args.pkl` nor the h5 carries an explicit `label_names` field. There is no separate
Engine Noise output to read. In the eval h5, only **31 of 1158** rows are Engine Noise, so the
positive class the model was actually trained to fire on is **~80% non-vessel** anomaly types.
Scoring this checkpoint and calling the result "Engine Noise" would silently report a mostly-not-a-
vessel signal as a vessel signal.

### c. Zero device-matched engine-noise validation is possible even in principle

The `merileo/onc-ssl-tutorial` eval h5, restricted to `ICLISTENHF1266` rows (247 of 1158), has
**zero** Engine Noise rows. Its only positives at our exact device are `Sensitivity` (20 rows) and
`Dropout` (4 rows) -- both instrument faults, not vessel signal. Even a working, correctly-headed
model could not be validated for engine noise at Folger from this dataset; the device-matched
reference 0009 relied on does not cover the label we need.

### d. The CPU forward pass is OOM-killed, and the mechanism was misdiagnosed once already

`torch.load(..., mmap=True)` succeeds; a strict `load_state_dict` is clean (92.66M params, no
missing/unexpected keys). The forward pass at the model's required sequence length (L=2501: 50x50
patches plus cls token, fixed by `pos_embed`) is OOM-killed, exit 137, even under a CPU shim
(`boatphone/onc_model_cpu.py`) that substitutes `mamba_ssm`'s own pure-PyTorch
`selective_scan_ref` for the CUDA-only fast path.

**The first-pass B0 ledger recorded this as "not a thread-buffer artefact" and implicitly framed
it against `free`'s reported 30 GB host RAM -- both wrong.** The adversarial recheck read
`/sys/fs/cgroup/memory.max` directly: this container is capped at **3,902,836,736 bytes (3.90 GB)
by cgroup v2**, not 30 GB, and `memory.events` showed `oom_kill=10` at the time. One Mamba layer
measured **1.18 GB RSS at L=2200**; roughly 1.1 GB was free at the moment of the OOM because seven
concurrent agent processes were holding ~2.8 GB. **Correction: this is "broken on this host"
(invariant 9), not an architectural impossibility.** A larger RAM allocation or a GPU host would
plausibly clear it. This correction is recorded here because the original framing, left
uncorrected, would have looked like stronger evidence against the model than it actually is.

**Raising the container's RAM cap does not reopen this path on its own** -- see the note on
resource provisioning below and the "how to tell this was wrong" clause.

### e. The blocker that makes (d) moot regardless of RAM: compute cost

`depth=24`, `bimamba_type=v2`, `if_bidirectional=True`: 24 layers, 2 directions per layer, each
running the pure-Python `selective_scan_ref` (the CUDA fast path is unavailable on CPU) at
L=2501. **[CORRECTION, 2026-08-27, pre-merge quality review]** This section previously said
"Measured at ~2 minutes/sample on 4 cores". That word was wrong: the forward pass **never
completed** -- it was OOM-killed at exit 137 (finding d) -- so no per-sample wall-clock was ever
observed end-to-end. The ~2 min/sample figure is an **extrapolation**: one Mamba layer's
`selective_scan_ref` was timed in isolation at sequence lengths up to L≈2200 (the pre-OOM regime
in finding d), and that per-layer, per-direction timing was scaled by 24 layers x 2 directions x
(2501/2200) to a per-sample estimate. No timing script or artefact for this scaling is checked
into the repo; the number should be reproduced with a standalone single-layer timing script before
being relied on further. The downstream arithmetic is unaffected by the relabelling (9,300 files x
~2 min ≈ **12.9 CPU-days**, restated as an extrapolation not a measurement) and the estimate is
probably conservative, since it ignores per-sample Python/tensor-allocation overhead outside the
scan itself, which would only add time. B3's target corpus is ~9,300 files. Scoring it once is
**~12 CPU-days (extrapolated, see above)** — independent of RAM, independent of the head-shape
problem in (b), and independent of what host it runs on unless a GPU (not just more CPU RAM) is
provisioned. **This is decisive on its own**, even bracketing (b), (c) and (d) entirely.

### f. Licence correction (not a decision input)

The repo's actual `LICENSE` file is **BSD-3-Clause** (Yuan Gong 2022, this is a fork of the
SSAMBA codebase), not MIT as 0009 stated. The Hugging Face model cards for the `merileo/*`
checkpoints do say `license: mit`. **Recorded for accuracy only.** Both licences are permissive
for this project's use; this fact did not move the GO/NO-GO call in either direction.

### g. Two provenance defects, preserved as reusable warnings

- **`result.csv`'s per-device attribution is unreliable past epoch 1.** The training script writes
  its per-device column header once, from a `defaultdict` keyed in first-appearance order over a
  *shuffled* validation loader. The header drifts at every resume boundary (epochs 14 and 27
  recur). Only the **epoch-1** row is name-attributable to a device; by epoch 63 (the pinned
  checkpoint's epoch) the `ICLISTENHF1266`-signature rows (`targets==25`) sit under a column
  labelled `ICLISTENHF1253_*`. **This is the invariant-4/structural-null-check success story**: the
  null check was run, it disagreed with the naive header reading, and that disagreement is what
  surfaced the defect rather than the naive P 0.96 / R 0.88 / AUC 0.99 (n=149) figure being taken
  at face value. That figure is confirmed **verbatim** against the Hub's `result.csv` -- bit-for-
  bit the same checkpoint (sha256 `e474ddc5...`) -- but its semantics are binary-anomalous @
  threshold 0.5, validation split, **epoch 1**, not the epoch-63 weights we hold.
- **Repo/checkpoint drift.** The pinned repo commit hardcodes `num_classes=label_dim` and the
  checkpoint's actual head is `(2, 768)` (before the further finetune to the `(1, 768)` binary
  head above); `create_model` reads `args.if_divide_out`, but `args.pkl` spells the field
  `if_devide_out`. **The pinned repo cannot rebuild the pinned checkpoint as shipped.** Anyone
  revisiting this path needs to reconcile the drift before trusting anything the repo's own
  `create_model` produces from these args.

### h. The model-survey rejection criterion was wrong, and is replaced

The alternative-model survey (plan v2 §2 "Dropped, and why", and 0009's Context) partly rejected
candidates -- Domingos/`underwater_snd`, Decrop et al., and by extension the whole
"frozen-encoder-plus-linear-head" family -- on **"no downloadable weights"**. That criterion is
**withdrawn**. It contradicted the survey's own finding, restated above, that weights in this
field are not portable: each published checkpoint is bonded to one hydrophone's calibration,
depth, sample rate and propagation environment, and this project's own experience with the
`merileo` checkpoint (findings b-e above) is a demonstration of exactly that bonding. A released
checkpoint from CATFISH, UATR-CMoE or anyone else would have needed retraining or fine-tuning on
our data regardless, so rejecting an MIT-licensed, weightless, reusable codebase for lacking the
one artefact we could not have used unmodified anyway was rejecting it for the wrong reason.

**Replacement criterion, which is checkable and does not depend on licence or weights at all:**

> Rejected because our continuous product must be computed from the `.fft.gz` surface, and no
> published vessel model's input geometry can be fed from it without a spectrogram remap that is a
> research project in its own right. Waveform-based models can be scored only on the ~25-30
> labelled windows -- exactly our evaluation set -- so they cannot produce the continuous estimate
> (G1). Secondarily, CATFISH is band-disqualified (discriminative content collapses to ~60-300 Hz
> at range) and, with UATR-CMoE and Conformer_UATR, is unlicensed -- a harder blocker than missing
> weights. MIT-licensed weightless code (the three Peeples models; Domingos/`underwater_snd`) was
> rejected for the wrong reason: it is reusable, and only corpus-access latency and the CPU budget
> rule it out this week.

**Corrections carried with the criterion change:**

* **PANNs's code licence is MIT, not Apache-2.0** (verified via the GitHub licence API); its
  weights are CC BY 4.0. MIT+CC BY is the most permissive combination found anywhere in the
  survey.
* **The Decrop dataset is at VLIZ, DOI `10.14284/723`, CC BY 4.0** -- **not** Zenodo. Zenodo record
  `12799031`, previously cited as its location, is only the ICUA2024 conference slide deck. Decrop
  is 27,524 x 10 s clips (~76 h), AIS-labelled including vessel type, activity, speed, MMSI, and,
  uniquely among the open corpora surveyed, **vessel-to-hydrophone distance**. It has no public
  code, only the dataset.
* **The WAV-wall arithmetic that makes the input-surface argument decisive:** the full local
  corpus is ~9,300 files x ~115 MB ≈ **1.07 TB**, against **~1.2 TB free** on this host, before any
  download time is counted. The `.fft.gz` path over the same corpus is ~50 CPU-minutes single-core
  (measured, alt-3). That asymmetry -- not licence, not weights -- is why every waveform-native
  model is confined to the ~25-30 labelled matchup windows and cannot produce a continuous
  estimate.
* **Unadjudicated band conflict, to be resolved by B5's band-design justification, not inherited
  from either paper:** CATFISH's own classifier config puts distant-vessel discriminative content
  at ~60-300 Hz, mostly below our 250 Hz floor. The May River recreational-vessel detector
  deliberately discards everything below 800 Hz, treating it as fish chorusing. Barkley Sound has
  both fish chorusing and small recreational outboards, so neither paper's band choice transfers
  uncritically; B5 must state and justify its own band against this disagreement rather than
  adopting one side by default. (It also partly rescues the 250 Hz floor: if May River is right
  for recreational craft specifically, the floor costs less than CATFISH's low-band emphasis would
  suggest, because CATFISH's low band is where the *commercial* traffic lives.)

Full detail and the reasoning that reached this replacement: run-phase ledger, `alt-2 DONE` and
`alt-4 DONE` entries.

### i. How to tell this was wrong

Revisit this path only if **both** of the following become true, not either alone:

1. ONC releases a checkpoint with a **multi-label head that includes Engine Noise as a distinct
   output** (finding b) — this also needs a device-matched validation set with nonzero Engine
   Noise rows at `ICLISTENHF1266` (finding c) to be usable, not just released.
2. **A GPU host** becomes available. This is a compute-cost blocker (finding e), not a RAM
   blocker: **raising this container's cgroup RAM cap alone does not clear it.** More RAM would
   only address finding (d), which was already the least load-bearing of the three primary
   findings (b, d, e) — (b) and (e) hold regardless of how much RAM this container gets.

## Consequences

* **§4 of `acoustics_plan_v2.md` is rewritten, not annotated** — see the plan surgery landed in
  the same change. B2 (spectrogram adapter) and B4 (reproduce-then-run) are struck; their time
  goes to B5 and B7.
* **B5 (band-level SPL, `boatphone/features.py`/`boatphone/ambient.py`) is now the primary
  detector**, not a cross-check. It needs no weights and is the method this community actually
  publishes.
* **The project loses a cross-check and a headline, not a goal.** G1, G2 and G3 all still resolve
  on the B5 path; G4 is untouched (it never depended on §4). The "two independent detectors
  agreeing" epistemic device from plan v2 §7 is replaced with band-split agreement (an independent
  detector on a disjoint band, e.g. 1-4 kHz vs 8-20 kHz, over the same windows), the far-field
  diagnostic, and the null checks — now explicitly non-optional rather than a nice-to-have.
* **0009 is superseded, not amended** — its VTUAD-rejection reasoning is unaffected and still
  load-bearing; only its three factual claims about the checkpoint (licence, CNN baseline
  existing, Engine Noise being a distinct output) are wrong, and each is corrected inline there
  with a pointer back to this record.
* **`boatphone/onc_model_cpu.py` is unused but not deleted** — its docstring is the surviving
  evidence for finding (d) and carries an `UNUSED -- see decision 0012` banner so the architecture
  audit does not (correctly, on its own terms) flag an unimported library module as drift.
* **B5's excess-over-ambient threshold and minimum-duration constraint become the single most
  consequential free parameter left in the project**, now that no pretrained decision boundary is
  involved. It needs its own decision record when chosen, not a default buried in `features.py`.

Supersedes the §4/B2/B4 portions of `docs/plans/acoustics_plan_v2.md`. Corrects (does not
supersede) `docs/decisions/0009-onc-pretrained-checkpoint-is-the-model.md`. Provenance for the
artefacts underlying every finding above: `docs/derived/b0_external_provenance.json`.
