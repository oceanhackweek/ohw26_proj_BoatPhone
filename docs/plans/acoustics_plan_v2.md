# BoatPhone -- Acoustics Workstream Plan v2

**Owner:** Isaac Guld
**Supersedes:** `accoutics_plan.md` (v1)
**Parent plan:** `docs/plans/proposed_plan_IG.md` (rev. 3)
**Sibling plans:** PlanetScope acquisition (Malachy), YOLO vessel detection (Neve)
**Day 1 = Thu 2026-08-27**

> v1 stays in the repo for its evidence sections only -- the 250 Hz bin-mapping derivation and the
> `.fft.gz` anomaly characterisation, which §3 below cites. Where the two plans disagree on
> anything else, **v2 wins**.

---

## 1. What this workstream is for

`Project_Source_of_Truth.txt` gives Isaac "hydrophone data and acoustic modelling." Four project
goals depend on it. Each row states what *done* means, so progress is checkable rather than
narrative.

| # | Source-of-truth goal | Done means | Priority |
|---|---|---|---|
| **G1** | Continuous estimate of vessel presence / count | A per-bin vessel-presence score across the covered archive, validated against optical labels, with a stated false-positive rate | **Must** |
| **G2** | Distinguish small (non-AIS) from large vessels | A measured statement about which size classes are separable acoustically -- **including "they are not"**, which is a valid result | **Must** |
| **G3** | Useful acoustic detection range | P(detect) vs range vs size class, with the optical label-noise floor carried through | **Must** |
| **G4** | Speed from pitch / frequency | Any measured statement at all | **Stretch** |

**The thing to avoid:** a clean vessel-to-energy correlation is exactly what a time-alignment bug
also produces (CLAUDE.md invariant 4). Every headline number below is paired with a null check.

---

## 2. Status

### Complete

- **A0 -- environment, credentials, deterministic gate.** ONC token loading with redaction,
  `boatphone/paths.py`, `.gitignore` hardening, and `scripts/checks.py` as the check harness
  (there is no pytest in this environment). Merged at `0a9fdfc`.
- **A1 -- uptime calendar.** `boatphone/onc_client.py` (paging-complete ONC archive listing,
  deployment assignment, gap summarisation), `boatphone/config.py`,
  `scripts/build_uptime_calendar.py`, and `docs/derived/hydrophone_gaps.md` -- **deliverable O1,
  shipped to Malachy**. Decisions 0007 and 0008. Full seasonal scan 2020-2026.

### The finding that reshapes everything else

**`ICLISTENHF1266`'s deployment ended 2026-03-14T21:37:20Z. The entire 2026 season has no
hydrophone data.** Today is 2026-08-27.

Consequences, which are project-wide and not acoustics-only:

1. **All matchups must use ARCHIVE PlanetScope imagery, 2020-2025 seasons.** New tasking produces
   scenes with no acoustic counterpart at all. **Tell Malachy before any order is placed** -- this
   is the same class of unrecoverable quota waste that O1 was built to prevent.
2. Usable in-season span is 2020-05-01 -> 2025-10-01, minus the four outages already enumerated in
   `hydrophone_gaps.md` (largest: 9.68 days, Aug-Sep 2023).
3. October-April was never scanned and is claimed neither way. If archive imagery turns out to be
   plentiful in the shoulder months, extending the scan is cheap and worth doing.

Note the availability figure is an **upper bound**: a bin counts as available if any listed file
overlaps it, and each file is modelled as 300 s of coverage, so dropouts shorter than ~10 min
cannot appear. B3's pull is expected to report lower uptime; that disagreement is this model, not
a bug.

### Dropped, and why

- **VTUAD and the transfer experiment (v1 §A8).** Verified during planning: VTUAD carries **no
  vessel size or length label**, so v1's headline size-stratified transfer gap cannot be computed
  at all; its audio is **uncalibrated** where ours is calibrated, so an absolute-level gap would be
  a calibration artefact indistinguishable from domain shift; and it sits behind a **paid IEEE
  DataPort Subscriber tier with no download API**, not the "overnight scripted download" v1
  assumed. Superseded by §4, which supplies a *device-matched* reference for free. Facts recorded
  in `docs/vtuad-facts.md`.
- **"Frozen PANNs/VGGish + linear head" as the model (v1 §A8).** **No public AIS-trained acoustic
  weights exist.** Domingos releases MIT training code but no checkpoints; Decrop et al. (JSTARS
  2025) release an open dataset but no code or weights; Renaud et al. (`arXiv:2607.13840`) is
  method-only. Any such model would therefore have been ours, which makes a poor transfer result
  ambiguous between "the method fails" and "we built it badly" -- the ambiguity invariant 9 exists
  to prevent.
- **v1's framing of >=250 Hz as a general limitation.** Small planing craft radiate peak energy at
  roughly 1-10 kHz (cavitation broadband); it is *large* ships that carry the diagnostic blade-rate
  tonals at 10-100 Hz. The floor costs us large-ship detection, **not** small-boat detection --
  and small craft are the target population. v1 half-stated this in §A5 and contradicted it
  elsewhere.

### Carried forward

v1's A2 (`.fft.gz` gate), A3 (calibration), A4 (bulk acquisition), A5 (features), A6 (ambient and
detectability) and A9 (matchups and split hygiene), renumbered below. `boatphone/models.py` from
the A8 worktree survives intact: its `assert_band_matched`, `band_limit` and `assert_comparable`
are 40/40 green and mutation-tested against eight deliberate sabotages.

---

## 3. Settled facts -- do not re-derive

| Fact | Source |
|---|---|
| **250 Hz per bin** (1024-pt FFT at 256 kHz, 512 bins over 0-128 kHz). Bin 1 = 250 Hz | v1 §"What changed since rev. 3" -- three independent confirmations |
| Calibration file covers **10 Hz - 51.2 kHz only** = bins 0-205. Bins 206-417 are uncalibratable | the file's own header |
| Decidecade bands resolved only **above ~2.2 kHz**; no hybrid-millidecade compliance | 250 Hz bins; a decidecade band at *f* is ~0.23*f* wide |
| **~42 dB unexplained shape difference** across 250 Hz -> 2 kHz survives the remapping | v1 §A2 -- still open, gated in B1 |
| `.fft.gz` is **0.29 MB per 5-min file**; WAV for the same span is ~3.5 TB | v1 §A4 |
| ONC 400s of the "not deployed during range" form are **measured zeros** | decision 0007 |
| An empty listing over a deployed span is a **measured zero**, not a failure | decision 0008 |
| Study window **2020-02-18** (calibration validity) -> **2026-03-14** (deployment end) | calibration file; `hydrophone_gaps.md` |

---

## 4. The model: ONC's own pretrained checkpoint

Ocean Networks Canada released
[`selfsupervision_anomalies_onc`](https://github.com/OceanNetworksCanada/selfsupervision_anomalies_onc)
(MIT, PyTorch) with public checkpoints on Hugging Face under `merileo/*`. It fits unusually well:

- **`Engine Noise` is an output class** -- vessel presence, directly. Full label set: `Anomaly,
  Data Gap, Dropout, Engine Noise, Rain, Sensitivity, Tonal, Unknown Feature`.
- **`ICLISTENHF1266` -- our device -- is in its training/eval device list**, with per-device figures
  broken out in `result.csv`: reported P 0.96 / R 0.88 / AUC 0.99 on 149 samples. **These are
  epoch-1 numbers and are unverified until we reproduce them.**
- **It consumes ONC spectrogram arrays, not WAV.** So `.fft.gz` is the native input class, and the
  expensive WAV pull that every other candidate required does not arise.
- The positive label is generic engine noise from BC coastal ONC sites, so **recreational traffic
  sits inside the positive class by construction** -- the opposite of VTUAD's exclusion problem.
- Public labelled eval data exists (`merileo/different_locations_...`), so the model can be
  validated before being trusted on our data.

**Environment constraint, checked:** there is no GPU here (`torch.cuda.is_available()` is False, no
`nvidia-smi`). The SSAMBA/Mamba path needs NVIDIA plus `mamba_ssm`, so we use the **CPU CNN
baseline** (`cnn_baseline/cnn_best.pt`) with `eval/evaluate_model.py`.

**Considered and not chosen, kept as fallbacks:**

- **UATR-CMoE** -- released 135 MB checkpoint whose classes include `Motorboat` and `Sailboat`,
  the only released model naming our target population. But **no declared licence**, no feature
  extraction script, and ShipsEar's discriminative content is heavily sub-1 kHz, where our floor
  bites hardest.
- **PANNs / AudioSet zero-shot** -- AudioSet class 304 is literally `Motorboat, speedboat` (plus
  `Ship`, `Sailboat`, `Boat`), so any AudioSet-trained model emits a vessel score for free. But
  **nobody has published how these perform on underwater audio**, and the nearest benchmark warns
  the embeddings encode recording conditions more than vessel identity. A one-hour experiment with
  a real chance of failing, not a plan.

---

## 5. Segments

In dependency order. Each names its module, its gate, and its fallback.

### B0 -- Model viability gate *(Day 1, half a day -- this segment can kill §4)*

**Do this before any other new work.** Everything in §4 comes from a literature sweep and is
unverified first-hand.

- **Input compatibility -- the real risk.** `onc_ssamba/utilities/spectrogram_utils.py` expects an
  ONC `.mat` spectrogram `[F, T]` of shape `[854, 1000]`. Ours is `.fft.gz`, 1200 x 512. Read one
  of each; compare shape, frequency axis, time axis, units and scaling. **Quantify the gap
  precisely** -- it decides whether B2 is an afternoon or the whole segment.
- Confirm `ICLISTENHF1266` really appears in `result.csv`, and what those per-device numbers mean.
- Confirm the CNN baseline loads and runs on CPU under torch 2.12, and that its dependencies exist
  here -- **check `cv2` in particular** (`spectrogram_utils` calls `cv2.resize`; CLAUDE.md does not
  list it as available). Report anything installed, per CLAUDE.md.
- Read class ordering from `args.pkl` or the h5 `label_names`. **Never assume the YAML order** --
  the model cards are a bare `license: mit` with no documented ordering.

**Fallback if this fails:** the band-level SPL route (B5), which needs no weights at all and is the
method this community actually publishes. **Decided here, on Day 1 -- not Day 4.**

### B1 -- The `.fft.gz` gate *(Day 1-2, timeboxed to one day)*

`boatphone/fft_io.py`. Unchanged in substance from v1 §A2, and still the project's named blocker --
now doubly important, because the model in §4 consumes this exact surface.

1. **Re-run the magnitude regression under the 250 Hz mapping.** Product bin *b* against WAV bin
   *2b*, bins 1-205 only, censoring-aware (drop values at 0 and 86 -- regressing over clipped
   values biases the slope toward 1 spuriously), like estimator against like estimator
   (median-of-dB vs median-of-dB, never median vs Welch mean).
2. **Resolve time alignment empirically.** Cross-correlate on the **38 kHz echosounder transient**,
   which is narrowband, intermittent and high-contrast -- worth far more than broadband ambient.
   Do not assume the nominal 3.971 s offset between WAV and FFT file starts.
3. **Run the gate on a loud window, not the supplied sample.** The sample is quiet ambient (WAV
   broadband std 0.6 dB over 5 minutes); **any r from it is uninformative in either direction.**
   Source a loud window by ranking B3's corpus on 1-10 kHz broadband level.
4. **Characterise the ~42 dB low-frequency anomaly.** Test: high-pass in product generation;
   per-bin noise-floor subtraction; a different averaging convention in the low bins. In parallel,
   **email ONC for the product definition on Day 1 regardless of progress** -- it is a
   documentation question and the answer may take days.

**Pass:** slope ~ 1 within stated tolerance over bins 1-205; intercept characterised; convention
identified (power / PSD-per-Hz / spectral level; window-energy corrected; one-sided x2). Record as
documented constants and re-run whenever the reader changes.

**Fallback, decided now and not on Day 4:** proceed on relative and shape-based features --
spectral slope, band ratios, level *changes* through closest point of approach. G1, G2 and G3 all
survive this. Only the physics baseline (B6) is genuinely lost.

### B2 -- Spectrogram adapter *(Day 2, gated on B0 and B1)*

`boatphone/onc_spectrogram.py`. Bridges `.fft.gz` to the model's expected input.

- Read to `[F, T]` with an **explicit named frequency axis in Hz and time axis in `t_utc_s`**
  (decision 0002 -- name the frame at every boundary, never a bare `t`).
- Reproduce their preprocessing exactly: percentile clip -> `log` -> min-max to [0,1] -> resize to
  512 x 512, with `dataset_mean 51.506817` and `dataset_std 13.638703`.
- **Assert** band and shape rather than silently resampling. Reuse `assert_band_matched`,
  `band_limit` and `assert_comparable` already in `boatphone/models.py` -- do not write new ones.
- No `fillna` or interpolation across a real recording gap; print any dropped frames with counts
  (invariant 5).

### B3 -- Bulk acquisition *(Day 1 evening, overnight)*

`boatphone/onc_client.py`, extending A1's listing code, which already handles paging and retry.

**Changed from v1:** target the **2024 and 2025 seasons first**. v1 said "the two most recent
summers", which now resolves to the dead 2026 and a partial 2025. Roughly 2.7 GB, 1.5-3 h at 1-2
files/sec. Extend backwards toward 2020 if throughput allows (~8 GB for the full span).

- Window: PlanetScope crosses 09:30-11:30 local, padded to **09:15-11:45 `America/Vancouver`**,
  computed per date with `zoneinfo` -- **never a hardcoded UTC offset**, which silently loses the
  shoulder months across a DST boundary.
- Cache by content hash; resume on interrupt; exponential back-off on rate limits. A pull this size
  **will** hit a transient failure -- make resumption the default path, not the exception.
- **Log every requested-but-absent file.** That log refines O1 with measured rather than inferred
  availability. Where the listing and the pull disagree, **the pull wins**; reissue O1 to Malachy
  with a note if any orderable date changes.

What this buys beyond the corpus itself: the loud window B1 needs, on Day 1; ambient statistics
with real N rather than the ~25 windows that carry optical labels; time-of-day-matched negatives
for free; and a hedge if the optical yield disappoints.

**Sampling conditionality, to be stated wherever a statistic from this corpus appears:** it is a
~10:30-local window, so it supports diurnal claims not at all, and seasonal claims only within that
hour band.

### B4 -- Reproduce, then run the model *(Day 2-3)*

**Reproduce first. This is the competence gate.** Run the CPU checkpoint over ONC's public labelled
h5, **restricted to `ICLISTENHF1266` rows**, and reproduce the reported `Engine Noise` performance
*before* touching Folger data.

This is what makes every downstream number interpretable, and it is precisely the role VTUAD was
supposed to play -- now free and device-matched. If it cannot be reproduced, say so plainly: that
is "the method is broken", not "the method found nothing" (invariant 9), and they need different
follow-ups.

**Then score** `Engine Noise` across the covered bins of the B3 corpus. Emit to `data/derived/`
with a record of what produced it -- checkpoint hash, n, device, date.

**Null checks, before believing anything** (invariant 4): shuffle the labels; shift the time base
by an hour; score a known-quiet period. **Report that these were run and what they showed**, pass
or fail.

### B5 -- Physical baseline detector *(Day 3)*

`boatphone/features.py`, `boatphone/ambient.py`. **Not optional.** It is B4's independent check,
B0's fallback, and the method the passive-acoustics community actually publishes.

- Decidecade band levels -- **above 2.2 kHz only**; below that, raw-bin levels **labelled as such**,
  not as standards-compliant band levels. Via PyPAM / MBARI `pbp` / MHKiT, the last of which
  explicitly supports icListen. All three operate on PSD, so **this runs on `.fft.gz` without ever
  touching WAV**.
- Rolling percentile ambient baseline by season, time-of-day and band; excess-over-ambient
  threshold with a minimum-duration constraint.
- **Compute on the time-frequency surface, not one averaged spectrum.** 1200 frames at 0.25 s
  across a +/-15 min window gives level-versus-time through closest point of approach: the CPA
  time, the slope of rise and fall, and the width of the peak jointly constrain range and speed in
  a way a single spectrum cannot. This is the most valuable structure in the data and the basis of
  standard single-hydrophone ranging.
- **Far-field diagnostic:** windows with zero in-AOI detections but elevated broadband level are
  direct evidence of out-of-AOI traffic. It falls out of the `n_vessels == 0` negatives already
  collected, and it is the AIS-free way to bound how often the far field contaminates labels.

Two independent detectors agreeing is a far stronger result than either alone. Disagreeing is
itself a finding worth reporting.

### B6 -- Calibration and physics *(Day 3-4, gated on B1)*

`boatphone/calibrate.py`, `boatphone/priors.py`, `boatphone/propagation.py`.

- Parse the sensitivity file (3 header lines, 5120 rows, 10 Hz -> 51,200 Hz at 10 Hz spacing,
  dB re Counts^2/uPa^2). Interpolate onto the 250 Hz grid; subtract; return dB re 1 uPa^2/Hz.
- **Emit a band-validity mask.** Bins 0-205 calibratable; bins 206-417 are not covered by the file
  we hold. Every downstream consumer respects the mask and violations raise.
- **Request the 256 kHz sensitivity curve from ONC** in the same message as B1's product-definition
  question.
- Sanity-check quiet-period levels against Wenz ambient curves for the sea state -- expect
  agreement within a few dB.
- SL from JOMOPANS-ECHO (+/-6 dB, validated in BC through the ECHO programme) plus the small-vessel
  extension; TL via Bellhop with GEBCO/CHS bathymetry and SSP from ONC's co-located Folger CTD.

**Two gates, and this is the first segment to cut if the week compresses.** Neither B4 nor B5
depends on absolute calibration -- the model consumes normalised spectrograms and B5 works on
excess-over-ambient. And `arlpy` is not installed, with Bellhop needing an acoustics-toolbox FORTRAN
binary that `pip install arlpy` does not supply (`docs/environment-audit.md`).

**Deliver a posterior with stated width, not a point inversion.** A single hydrophone does not
jointly determine range, size, hull class, speed, count and the local correction term. Put priors
on class, speed and count; sample over range and size; report posterior width. The error budget up
front: +/-6 dB SL uncertainty against a ~15-18 dB/decade TL slope is roughly a **factor of ~2 in
range**, before class, speed and count uncertainty.

### B7 -- Matchups and split hygiene *(Day 4, gated on I3)*

`boatphone/geo.py`, `boatphone/matchup.py`, `boatphone/splits.py`.

- Range and bearing from the hydrophone via `pyproj`. Join each detection to its +/-15 min acoustic
  window.
- Emit `matchups.parquet` with **`overpass_id` on every row** and **AIS columns present but
  nullable**, so an optional AIS stage drops in later against an unchanged schema.
- **Dominance stratification:** `R2 > 2*R1` -> acoustically dominated, the MVP subset (larger than
  `n_vessels == 1` would be); genuine mixtures -> nearest-vessel range plus count;
  `n_vessels == 0` -> negatives.
- **`splits.py` is the single source of splits** -- overpass-grouped, time-blocked CV, with a hard
  assertion that no `overpass_id` appears in both train and test.

**The effective sample size is the number of usable overpasses (order 20-30), not the number of
detections (100-400).** A scene with 8 vessels gives 8 labels on *one* acoustic mixture -- eight
labels on one measurement. For anything acoustic -- sea state, tide, SSP, ambient, propagation,
hydrophone state -- the unit of analysis is the overpass. Report uncertainty clustered by overpass,
and **state both counts alongside every metric**: "n = 214 detections across 23 overpasses."

### B8 -- Results *(Day 5)*

Detectability curve (G3), size separability including a negative result (G2), continuous presence
estimate (G1), model-versus-physics agreement, and the matchup dataset. Final notebooks, README
workflow section, presentation.

**Every model number carries its overpass count. Every population statistic carries the
sampling-conditionality caveat** -- PlanetScope is sun-synchronous at ~10:30 local and usable scenes
are cloud-free summer days, precisely the conditions that maximise recreational traffic. No
population statistic here generalises to Folger Passage at large.

---

## 6. Interfaces

These are contracts. Each names the artefact, the owner, the schema, and what breaks without it.

### IN

| # | From | Artefact | When | Notes |
|---|---|---|---|---|
| **I1** | Malachy | `data/interim/overpasses.csv` -- `overpass_id`, `acquired_utc`, `scene_id`, `cloud_fraction`, `aoi_wkt`, `ordered` | Day 3 | `acquired_utc` **must be UTC and must be the acquisition instant**, not the order or publish time. A systematic timestamp error propagates directly and silently into every range label. Agree the convention in writing on Day 1. |
| **I2** | Malachy | Hydrophone coordinates, AOI centre and radius | Day 1 | |
| **I3** | Neve | `data/interim/detections.parquet` -- `overpass_id`, `detection_id`, `lat`, `lon`, `length_px`, `length_m_est`, `size_class` (FAO bin), `confidence` | Day 3 | Blocks B7 only |
| **I4** | Neve | Measured detector P/R on the hand-labelled set, per size class | Day 4 | The acoustic detectability curve cannot be cleaner than the optical labels that define it |

### OUT

| # | To | Artefact | When |
|---|---|---|---|
| **O1** | Malachy | `docs/derived/hydrophone_gaps.md` + `data/derived/hydrophone_uptime.csv` | **Delivered.** Reissue after B3's pull refines it |
| **O1b** | Malachy | **"2026 has no hydrophone data -- order ARCHIVE imagery, 2020-2025 seasons only"** | **Day 1, urgent** |
| **O2** | Both | Acoustic band support + calibration validity note | Day 2 |
| **O3** | Joint | `data/processed/acoustic_features.parquet` | Day 3 |
| **O4** | Joint | `data/processed/matchups.parquet` | Day 4 |
| **O5** | Write-up | Detectability curve, model results, physics baseline | Day 5 |

**B1-B6 are deliberately independent of I1 and I3.** Only B7 blocks, and B3's speculative pull
means it starts the moment `overpasses.csv` lands, with no download latency.

---

## 7. Verification

There is no pytest, no lint and no type check (CLAUDE.md). What exists is `scripts/checks.py` and
re-executing a notebook top-to-bottom in a fresh kernel.

| What | How | Pass criterion |
|---|---|---|
| **Synthetic control** | Tone of known level, frequency and time pushed through `.fft.gz` -> model input | Lands in the right bin and the right `t_utc_s` (invariant 3) |
| FFT reader -- structure | Row/bin counts; zero-column periodicity | 1200 x 512; zeros at col 0 and 419-511 |
| FFT reader -- mapping | 38 kHz echosounder line; anti-alias shoulder | Line at bin 152 +/- 1; shoulder onset ~ bin 408 |
| FFT reader -- magnitude | Product bin *b* vs WAV bin *2b*, bins 1-205, censoring-aware, loud window | Slope ~ 1; intercept characterised; convention documented |
| Time alignment | Cross-correlate on the 38 kHz transient | Lag resolved to < 1 frame (0.25 s) |
| Calibration coverage | Assert every absolute-level output falls in bins 1-205 | Violations raise, enforced in `calibrate.py` |
| Calibration accuracy | Quiet-period levels vs Wenz curves for the sea state | Within a few dB |
| Band claims | Assert decidecade bands only requested above 2.2 kHz | Enforced in `features.py` |
| **Model reproduction** | Our `Engine Noise` metrics vs ONC's `result.csv`, HF1266 rows only | Within stated tolerance, **with n reported** |
| **Null checks** | Shuffled labels; +1 h time shift; known-quiet period | Reported explicitly, pass or fail |
| **Detector agreement** | ONC model vs B5 band-level detector over the same windows | Agreement rate reported; disagreements characterised, not hidden |
| Split hygiene | Assert no `overpass_id` in both train and test | Zero overlap, enforced in `splits.py` |
| Deployment stability | Re-run the gate on one file from each deployment 2020-2026 | Consistent, or per-deployment constants recorded |
| Repo hygiene | `git log --stat` | No `data/` paths (inv. 2), no notebook outputs (inv. 7), no tokens |
| Reproducibility | Fresh clone + environment manifest, run the notebooks end to end | Works from an ONC token alone, no AIS required |

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| **`.fft.gz` is not the array the model expects** | B0 is a half-day gate that settles this before anything is built on it. Fallback is B5, which needs no weights. |
| **ONC's reported numbers do not reproduce** | Report it as a broken method (invariant 9) and fall through to B5. Do not proceed on an unvalidated model. |
| **B1's low-frequency anomaly never resolves** | Relative and shape-based features, decided Day 1. G1-G3 survive; only B6 is lost. |
| **No GPU in this environment** | CPU CNN baseline; confirmed as a supported path in B0. |
| **`arlpy` / Bellhop unavailable** | B6 is the first segment to cut, and nothing else depends on it. |
| **No 256 kHz calibration curve from ONC** | Restrict absolute levels to <= 51.2 kHz (bins 1-205). Costs the 51-104 kHz band, which matters little for vessel noise. |
| **Archive imagery scarce for 2020-2025** | Flagged Day 1 via O1b so Malachy can check yield before spending quota. If scarce, B4 and B5 still deliver a continuous presence estimate with no optical labels; only G2 and G3 need matchups. |
| **Effective N too small for any ML claim** | B5 and B6 need ~0 training data and produce a result regardless. |
| **I1 timestamps in local or publish time** | Agree the convention in writing Day 1; assert UTC and sanity-check against solar time (~10:30 local overpass). |
| **B3 throughput worse than 1 file/sec** | Pull the newest season first and work backwards, so the corpus is always useful at whatever depth it reaches. |

---

## 9. Assumptions

- **Timeline is the OHW hackathon week**, Day 1 = Thu 2026-08-27.
- **This stream owns the full acoustic stack** -- acquisition through physics through the acoustic
  half of the modelling. Optical acquisition and detection are owned elsewhere.
- **`matchup.py` sits on the acoustics side**, since the join is against acoustic windows on the
  acoustic time base.
- **PlanetScope overpasses fall within 09:30-11:30 local** for the whole period of interest, per
  the team. B3's speculative pull rests on this; verify against the first `overpasses.csv` rather
  than trusting it.
- **AIS stays optional.** Nothing above depends on it. If access lands it adds the blind-spot
  fraction, range-label validation, and a within-site control.

---

## 10. Out of scope

Acoustic Doppler speed estimation beyond a single measured statement (G4 is a stretch goal, and the
JASA finding that planing hulls show weak speed-SL dependence makes it harder than it looks); the
Folger Pinnacle hydrophone (build for Folger Deep alone, with the hydrophone as a config parameter,
and add Pinnacle only if Day 3 lands early); a second observatory site; pre-2020 deployments; bulk
WAV/FLAC acquisition; sub-250 Hz features as a pipeline component; training any model from scratch;
blind source separation; VTUAD v1 or v2; the ONC-AIS/CPA labelling pivot.
