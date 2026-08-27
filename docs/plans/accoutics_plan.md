# BoatPhone — Acoustics Workstream Plan

**Owner:** Isaac Guld
**Parent plan:** `docs/plans/proposed_plan_IG.md` (rev. 3)
**Sibling plans (other owners):** PlanetScope acquisition (Malachy), YOLO vessel detection (Neve)

---

## Context

`Project_Source_of_Truth.txt` assigns Isaac "downloading hydrophone data, and exploring hydrophone models." This plan is the executable expansion of that slice — everything on the acoustic side of BoatPhone, from ONC acquisition through to the acoustic half of the modelling — written as a subplan of `proposed_plan_IG.md` rev. 3.

The project estimates vessel **count, size class, and range** at Folger Passage from a single hydrophone, using PlanetScope detections as ground-truth labels for the small recreational vessels that never broadcast AIS. The acoustics stream owns the signal side of that: it must deliver calibrated, correctly-mapped acoustic features on a defensible time base, plus the physics baseline and the transfer experiment. The optical streams deliver the labels.

**The repo is greenfield.** There is no Python source of any kind — every module below is new code.

### What changed since rev. 3 — the bin mapping is wrong

Rev. 3 records the `.fft.gz` magnitude validation as failing (slope 1.37, r 0.70, an unexplained +45 dB low-frequency rise) and names it the project's only true blocker. Empirical work during planning found a concrete cause for a large part of that failure, and it changes the plan.

**Rev. 3 assumes 125 Hz per bin (1024-pt FFT at 128 kHz). The evidence says 250 Hz per bin (1024-pt FFT at 256 kHz, 512 bins spanning 0–128 kHz).** Three independent confirmations:

1. **A 38 kHz echosounder line.** The WAV has a strong intermittent narrowband source at 37.5–38.5 kHz (per-bin temporal std 12.3 dB). The product carries the same transient at bins 146–158, peaking at bin 152 — `152 × 250 Hz = 38.0 kHz`. The product has **no** variance at bins 295–312 (std 1.1 dB), where a 125 Hz/bin mapping would put it. A physical narrowband source pins the mapping directly; no inference required.
2. **The anti-alias shoulder is at 102.4 kHz, not 51.2 kHz.** The median profile decays smoothly 10 → 0 across bins 404–417 in *both* sample files. Roll-off onset ≈ bin 408 → `408 × 250 Hz = 102.0 kHz ≈ 0.4 × 256 kHz = 102.4 kHz`. Rev. 3 tried to match the zero-run onset to 0.4 × 128 kHz and got 417 vs the true 419 — close enough to look convincing, but the 250 Hz mapping fits far better.
3. **No discontinuity at 51.2 kHz.** The profile runs smoothly through bin 205 (`19,18,18,18,17,17,17,17,17,17,17,16,16,16`). If the product were a 128 kHz stream it would have to end there.

**Four consequences, all of which reshape the work:**

- **The rev. 3 magnitude regression compared product bin *b* against WAV bin *b*** — i.e. product@2*f* against WAV@*f*, sliding along a sloping spectrum. That mismatch alone can generate a spurious slope well away from 1. **Re-running the regression with the corrected mapping is the single highest-value first action**, and it may substantially or wholly resolve the "blocker."
- **The supplied calibration file does not span the product's band.** It stops at 51.2 kHz (`0.4 × 128 kHz`, stated in its own header) — that is bin 205 of 512. **Bins 206–417 (51.4–104 kHz) are uncalibratable from the file we hold.** Either request the 256 kHz sensitivity curve from ONC, or restrict all absolute-level work to ≤51.2 kHz. Rev. 3 did not identify this.
- **Sub-250 Hz structure is unavailable, not sub-125 Hz.** Bin 1 is 250 Hz. The low-frequency situation is *worse* than rev. 3 states, and the 40–60 Hz band the ranging literature favours (Jang et al. 2025) is further out of reach than recorded.
- **The WAV and the `.fft.gz` are different acquisition streams** (128 kHz vs 256 kHz). Cross-validation between them is only meaningful over 0–51.2 kHz, and their time bases must be aligned empirically rather than assumed — consistent with the observed 1.25 s lag and weak (r ≈ 0.64) broadband correlation.

**What is still genuinely unresolved:** the low-frequency anomaly survives the remapping. Product median at bin 1 (250 Hz) is 16 against a WAV level of 70.7 dB; at bin 8 (2 kHz) it is 60 against 72.4 dB. That is still a ~42 dB *shape* difference across 250 Hz → 2 kHz that no constant offset explains. A high-pass or noise-floor subtraction in ONC's product generation is the leading hypothesis. **This remains the blocker for absolute low-frequency levels and is the subject of the A2 gate below.**

*Caveat on all correlation figures quoted here: the sample is quiet ambient (WAV broadband std 0.6 dB over 5 minutes). Correlation tests have almost no dynamic range to bite on. Re-run the gate on a window containing a vessel transit before drawing conclusions from any r value.*

---

## Interfaces with the other workstreams

These are contracts. Each names the artefact, the owner, the schema, and what breaks without it.

### IN — required from the PlanetScope stream (Malachy)

| # | Artefact | When | Schema |
|---|---|---|---|
| **I1** | `data/interim/overpasses.csv` | Day 3 (blocking for A9 only) | `overpass_id` (str, stable), `acquired_utc` (ISO 8601, UTC, sub-second if available), `scene_id`, `cloud_fraction`, `aoi_wkt`, `ordered` (bool) |
| **I2** | Hydrophone coordinates + AOI centre | Day 1 | lat, lon, depth for Folger Deep; AOI disk centre and radius |

`acquired_utc` **must be UTC and must be the acquisition instant, not the order or publish time** — a systematic timestamp error propagates directly into every range label as a mis-association, and it is silent. Agree the timezone convention in writing on Day 1.

**I1 is no longer blocking for acquisition.** Because PlanetScope crosses this location between 09:30 and 11:30 local across the whole period of interest, the acoustic stream pre-downloads that daily window speculatively (A4 below) without knowing which days were ordered. When `overpasses.csv` arrives, matching it against an already-downloaded corpus is an index lookup. This removes the single largest schedule dependency in the workstream.

### IN — required from the YOLO stream (Neve)

| # | Artefact | When | Schema |
|---|---|---|---|
| **I3** | `data/interim/detections.parquet` | **Day 3** (blocking for A9) | `overpass_id`, `detection_id`, `lat`, `lon`, `length_px`, `length_m_est`, `size_class` (FAO bin), `confidence` |
| **I4** | Measured detector P/R on the hand-labelled set | Day 4 | precision, recall, per size class — needed to state the detectability curve's label-noise floor |

Without **I3** there are no labels; A9 and everything downstream stall. The acoustic stream can, however, complete A1–A8 without it — that independence is deliberate.

### OUT — delivered by the acoustics stream

| # | Artefact | To | When | Why it matters |
|---|---|---|---|---|
| **O1** | `data/interim/hydrophone_uptime.csv` — Folger Deep data availability, 2020–2026, at 5-min resolution | **Planet stream** | **Day 1 — highest priority outbound** | Planet quota is capped near 30 orders/month and **spent orders are unrecoverable**. Ordering a scene over a hydrophone gap wastes one permanently. This must land *before* any orders are placed. |
| **O2** | Acoustic band support + calibration validity note | Both | Day 2 | Tells the others which claims the acoustics can actually back |
| **O3** | `data/processed/acoustic_features.parquet` | Joint modelling | Day 3 | The feature table |
| **O4** | `data/processed/matchups.parquet` | Joint | Day 4 | The dataset deliverable |
| **O5** | Detectability curve, physics baseline, transfer result | Write-up | Day 5 | The headline results |

**O1 is the most important thing this workstream produces in the first 24 hours**, and it is easy to overlook because it is an input to someone else's task rather than an acoustic result. Flag it to Malachy immediately.

---

## Modules

All under a new importable package `boatphone/` — not notebook-only code. `notebooks/A1`–`A6` are thin wrappers.

```
boatphone/
  onc_client.py    # auth, location/device discovery, uptime scan, product download w/ retry+cache
  fft_io.py        # .fft.gz reader: 1200 x 512, mapping resolved by evidence, censoring-aware
  calibrate.py     # sensitivity interpolation -> dB re 1 uPa^2/Hz; band-validity mask
  features.py      # band levels, spectral shape/slope, DEMON, cavitation band, HF roll-off
  ambient.py       # Wenz comparison, percentile ambient baseline, far-field diagnostic
  priors.py        # JOMOPANS-ECHO SL(L, v, class, f) + small-vessel extension
  propagation.py   # arlpy/Bellhop TL, GEBCO bathymetry, ONC Folger CTD SSP
  geo.py           # range + bearing from hydrophone (pyproj)
  matchup.py       # detections x acoustic windows -> parquet (overpass_id, nullable AIS)
  vtuad.py         # VTUAD loader + band-matching to Folger support
  models.py        # forward posterior, embeddings + linear head, transfer evaluation
  splits.py        # overpass-grouped, time-blocked CV — the single source of splits
```

---

## Workstream

### A0 — Environment and credentials *(Day 1, first hour)*

- **ONC API token.** The API returns 401 without one; nothing acquisitional works until this exists. Register at `data.oceannetworks.ca`. Store in `.env`, gitignore it.
- **Install:** `arlpy` (Bellhop TL), `pypam` or `mbari-pbp` (ISO 18405 conventions), `soundfile` (FLAC). Create the missing `environment.yml` — the repo has **no dependency manifest at all**, which makes the "fresh clone reproduces" verification criterion currently unmeetable.
- **Already present and sufficient:** `onc` 2.6.0, `scipy` 1.18, `numpy` 2.5, `torch` 2.12, `sklearn` 1.9, `pyproj`, `pyarrow`, `xarray`.
- **`soundfile` is not needed for the WAV.** `scipy.io.wavfile.read` handles the 24-bit file directly (verified — returns int32, fs 128000, 38.4 M samples). Only FLAC needs it.
- **Housekeeping:** `search78910238.zip` (72 MB) is fully redundant — it contains only the calibration txt and the same WAV already extracted. Delete it and extend `.gitignore` to `data/raw/`, `data/interim/`, `*.flac`, `*.zip`.

### A1 — Hydrophone uptime calendar *(Day 1 — outbound blocker O1)*

`boatphone/onc_client.py`.

- Discover Folger location codes at runtime via `onc.getLocations()` — do not hardcode. Device is `ICLISTENHF1266`, ONC deviceId 23235, Folger Deep.
- Query archive file listings for the `.fft.gz` product across 2020-02-18 → 2026, restricted to May–Sep. **Listings are cheap — run this first, before the A4 bulk pull**, both to size the download and to deliver O1 within hours rather than after an overnight job.
- Emit `hydrophone_uptime.csv` at 5-min resolution: `start_utc`, `end_utc`, `available` (bool), `deployment_id`.
- **Refine from A4's absent-file log once the bulk pull completes.** Listings state what ONC believes exists; the pull establishes what actually downloads. Where they disagree, the pull wins — reissue O1 with a note to Malachy if the corrected calendar changes any orderable date.
- **Ship to Malachy the same day** with a plain-language summary: "these date ranges have no hydrophone data — do not spend Planet quota on them."
- Record deployment boundaries. Rev. 3 flags that `.fft.gz` layout and semantics may differ across deployments; the A2 gate must be re-run per deployment.

**Start bound is 2020-02-18**, the calibration file's validity date for this device. Earlier Folger deployments are different devices with different sensitivities.

### A2 — The `.fft.gz` gate *(Day 1 — the blocker)*

`boatphone/fft_io.py`. **Timeboxed to one day, with a defined fallback.** Do not let this consume the week.

**Step 1 — re-run the magnitude regression under the 250 Hz mapping.** Highest expected value per unit effort in the entire plan. Compare product bin *b* against WAV bin *2b*, censoring-aware (drop values at 0 and 86 — regressing over clipped values biases slope toward 1 spuriously), like estimator against like estimator (median-of-dB vs median-of-dB, never median vs Welch mean). Restrict to **bins 1–205 only** — above 51.2 kHz the WAV has no content and the calibration has no coverage.

**Step 2 — resolve time alignment empirically.** Do not assume the nominal 3.971 s offset between the WAV start (`000000.029Z`) and the FFT file start (`000004.000Z`). Cross-correlate broadband level over lag and take the argmax. Use the **38 kHz echosounder transient as the alignment fiducial** — it is narrowband, intermittent and high-contrast, worth far more than broadband ambient for this. Note its event times in the sample cluster at `.75 s` phase within the 0.25 s grid, which is itself a clue about frame boundaries worth chasing.

**Step 3 — run the gate on a loud window, not this one.** The supplied sample is quiet ambient (WAV broadband std 0.6 dB); correlation is close to meaningless on it. **Any r value from the quiet sample should be treated as uninformative, in either direction.**

Source the loud window from **A4's corpus**: rank downloaded windows by broadband level in the 1–10 kHz band, take the top handful, and order the matching WAV or FLAC for just those (~50 MB each as FLAC). Re-run structure, mapping and magnitude there. This is the dependency that makes A4 worth starting Day 1 evening rather than Day 2 — the gate needs its output. If the overnight pull has not finished by the time A2 resumes, run the ranking against whatever partial corpus exists; a few hundred windows is ample to find a transit.

**Step 4 — characterise the low-frequency anomaly.** The ~42 dB shape difference across 250 Hz → 2 kHz survives the remapping. Test: (a) high-pass in product generation; (b) per-bin noise-floor subtraction; (c) a different averaging convention in the low bins. In parallel, **email ONC for the product definition** — this is a documentation question and the answer may take days, so ask on Day 1 regardless of progress.

**Pass criteria:** slope ≈ 1 within stated tolerance over bins 1–205; intercept characterised; convention identified (power / PSD-per-Hz / spectral level; window-energy corrected; one-sided ×2). Record as documented constants in `calibrate.py` and re-run whenever the reader changes.

**Fallback, decided Day 1 and not Day 4:** if unresolved by end of Day 1, proceed on **relative and shape-based features** — spectral slope, band ratios, level *changes* through closest point of approach — and defer every absolute-dB deliverable. The detectability curve and the transfer experiment both survive this fallback; only the physics baseline is genuinely gated on absolute levels.

### A3 — Calibration *(Day 1–2)*

`boatphone/calibrate.py`.

- Parse the 3-header-line, comma-delimited sensitivity file (5120 rows, 10 Hz → 51,200 Hz at 10 Hz spacing, dB re Counts²/µPa²).
- Interpolate onto the product's **250 Hz** grid; subtract; return dB re 1 µPa²/Hz.
- **Emit a band-validity mask.** Bins 0–205 calibratable; **bins 206–417 are not covered by the file we hold.** Every downstream consumer must respect this mask, and no absolute-level claim may be made above 51.2 kHz until ONC supplies the 256 kHz curve.
- **Request that curve from ONC** in the same message as the product-definition question.
- Sanity-check quiet-period levels against Wenz ambient curves for the sea state — expect agreement within a few dB.

### A4 — Speculative bulk acquisition *(Day 1 evening, overnight — NOT gated on I1)*

`boatphone/onc_client.py`. **This is the structural change that decouples the acoustics stream from the optical schedule.**

PlanetScope crosses Folger Passage between **09:30 and 11:30 local** for the entire period of interest. Rather than waiting on `overpasses.csv` and pulling `acquired ± 15 min` per scene, download that daily window wholesale, for every day in the archive, and index it. Every scene the Planet stream eventually orders is then already on disk.

**This is affordable only because `.fft.gz` is tiny — 0.29 MB per 5-min file:**

| Coverage | Files | Volume |
|---|---|---|
| One day, 2 h window (+ 15 min padding each side) | 30 | ~8.7 MB |
| One summer (May–Sep, 153 d) | ~4,600 | **~1.3 GB** |
| Two most recent summers | ~9,200 | **~2.7 GB** |
| All summers 2020–2026 | ~27,500 | **~8 GB** |
| *Same window as WAV, for comparison* | *—* | *~3.5 TB — not an option* |

**Recommendation: pull the two most recent summers on Day 1 (~2.7 GB), then extend to the full 2020–2026 span if throughput allows.** Disk is not the constraint; **per-file request throughput is.** At 1–2 files/sec, two summers is 1.5–3 hours and the full span is 4–8 hours. Start it Day 1 evening and let it run overnight alongside the VTUAD download.

**Timezone — get this right or the whole pull misses.** ONC's API is UTC-only. 09:30–11:30 local is **16:30–18:30 UTC in PDT** (UTC−7) and 17:30–19:30 UTC in PST (UTC−8). Compute the local window per date with `zoneinfo` (`America/Vancouver`) rather than hardcoding an offset — a fixed −7 silently loses the shoulder months if the span is ever widened past DST boundaries. Pad to **09:15–11:45 local** so a ±15 min matchup window around any overpass in the range is fully covered.

Implementation notes:
- Cache by content hash; resume on interrupt; exponential back-off on rate limits. A 27,500-file pull **will** hit a transient failure — make resumption the default path, not the exception.
- Log every requested-but-absent file. **That log is the uptime calendar (O1)** — measured availability, not inferred from listings. See A1.
- **Scoped WAV/FLAC experiment, not a second pipeline:** pull tens of windows across the dominated subset and measure whether sub-250 Hz features add anything over the ≥250 Hz set. Default expectation is that they do not and we stay on `.fft.gz`. Only a large measured gain justifies reopening the acquisition strategy — and that conversation must re-price the whole project, since it forfeits the few-MB-per-scene advantage.

**Four things this buys beyond schedule independence:**

1. **A loud window for the A2 gate, on Day 1.** The supplied sample is quiet ambient and near-useless for validation. Scanning a corpus of thousands of windows for high broadband level yields vessel transits immediately — which is exactly what A2 step 3 needs and would otherwise wait on.
2. **Ambient statistics with real N.** Seasonal and diurnal ambient percentiles (A6) computed over thousands of windows rather than the ~25 that carry optical labels.
3. **Negatives for free.** Every downloaded window with no ordered scene is a candidate control window for the far-field diagnostic, already time-of-day matched by construction.
4. **A hedge against optical shortfall.** If the Planet quota yields fewer usable overpasses than hoped, the acoustic corpus still supports the ambient and detectability work standalone.

**Constraint to respect:** this window is ~10:30 local, so the corpus inherits the same sampling conditionality as the optical data — mid-morning only. It supports diurnal claims not at all, and seasonal claims only within that hour band. Say so wherever a statistic derived from it appears.

### A5 — Features *(Day 2–3)*

`boatphone/features.py`. Compute **only on bins the product and the calibration both support.**

- Broadband and band levels; spectral shape and slope; high-frequency roll-off; cavitation-band energy; DEMON envelope on higher-frequency carriers.
- **Decidecade bands are resolved only above ~2.2 kHz** under the corrected mapping — a decidecade band at *f* has bandwidth ≈ 0.23·*f*, and two 250 Hz bins require *f* ≳ 2.2 kHz. This is double rev. 3's stated 1.1 kHz threshold. Below that, report **raw-bin levels and label them as such** — not standards-compliant band levels.
- Use PyPAM for decidecade bands above 2.2 kHz and for its calibration/PSD conventions. **Do not claim hybrid-millidecade compliance** — that scheme needs 1 Hz resolution below 455 Hz, which this product cannot supply.
- **Compute on the time–frequency surface, not one averaged spectrum.** 1200 frames at 0.25 s across a ±15 min window gives level-versus-time through CPA: the CPA time, the slope of rise and fall, and the width of the peak jointly constrain range and speed in a way a single spectrum cannot. This is the basis of standard single-hydrophone ranging and is the most valuable structure in the data.
- Emit `acoustic_features.parquet` keyed by `window_id`, carrying `overpass_id`.

**Small recreational craft are the favourable case for this band restriction** — outboards and planing hulls put substantial energy into the hundreds of Hz and kHz. The loss of sub-250 Hz hurts large-ship signatures more than the target population. State this explicitly rather than hoping it goes unnoticed.

### A6 — Ambient and detectability *(Day 3)*

`boatphone/ambient.py`. **This is source-of-truth goal 3 and it is a must-have deliverable.**

- Percentile ambient baseline by season, time-of-day and band.
- **Detectability curve: P(detect) vs range vs size class**, from the matchups.
- **Far-field diagnostic:** windows with zero in-AOI detections but elevated broadband level are direct evidence of out-of-AOI traffic. Falls out of the `n_vessels == 0` negatives already collected, and bounds how often the far field contaminates labels. This is the AIS-free substitute for accounting for vessels beyond 5.5 km.
- Carry the detector's measured P/R (**I4**) through as the label-noise floor — the acoustic detectability curve cannot be cleaner than the optical labels that define it.

### A7 — Physics baseline *(Day 3–4, gated on A2 magnitude)*

`boatphone/priors.py`, `boatphone/propagation.py`, `boatphone/models.py`.

- **SL:** JOMOPANS-ECHO (MacGillivray & de Jong 2021, ±6 dB, validated in BC through the ECHO programme) plus the small-vessel extension (JMSE 14:561, 25 vessels <12 m by hull class; Lagrois et al. 2023, St. Lawrence recreational craft).
- **TL:** Bellhop via `arlpy.uwapm`, GEBCO/CHS bathymetry, SSP from ONC's co-located Folger Passage CTD.
- **Forward model:** `RL(f) = SL(L, v, class, f) − TL(R, f)`, extended to mixtures:
  `RL_total(f) = 10·log10( Σᵢ 10^((SL(Lᵢ,vᵢ,f) − TL(Rᵢ,f))/10) ) ⊕ ambient(f)`
- **Deliver a posterior, not an inversion.** A single hydrophone does not uniquely determine range, size, hull class, speed, vessel count, heading and the local correction term simultaneously. Put priors on class, speed and count; sample or optimise over range and size; **report posterior width**. Then use the matchups — the cases where range and approximate size are *known* — to establish empirically which variables are identifiable.
- **Error budget stated up front:** ±6 dB SL uncertainty against a ~15–18 dB/decade TL slope maps to roughly a **factor of ~2 in range**, before class, speed and count uncertainty.
- Evaluate on supported, calibrated bands only (≤51.2 kHz).

**If A2's magnitude gate failed, run the shape-based fallback instead** and say so plainly in the write-up.

### A8 — VTUAD and the transfer experiment *(Day 1 download, Day 4 run)*

`boatphone/vtuad.py`, `boatphone/models.py`.

- **Start the VTUAD download on Day 1** — IEEE DataPort, large, run it overnight. It is on the critical path via band-matching.
- **Day 1 metadata check: sample rate, band, label type, licence, size.** The inclusion/exclusion-zone design suggests distance *bins* or scenario classes rather than continuous range, which changes what the transfer experiment can measure at all. Find out before building around an assumption.
- **Band-matching is critical path.** VTUAD and any pretrained embedding assume a sample rate and band that differ from the ≥250 Hz Folger support. Without band-limiting both sides to the common support, the measured "transfer gap" is a preprocessing artefact, not a domain shift. Enforce it with an assertion in `models.py`.
- Frozen PANNs/VGGish embeddings, linear head only. **Do not fine-tune beyond a linear head and do not train a CNN from scratch** at this effective N.
- **Name the three-way confound in the write-up regardless:** VTUAD → Folger varies population (commercial vs recreational), site (Strait of Georgia vs Folger) and time/instrument (2017 vs 2020–2026) simultaneously, so measured degradation cannot be attributed to the population axis — which is the entire claim. AIS-free mitigation: **stratify the gap by size class**; a gap that grows monotonically as vessels get smaller is evidence for the population axis that site shift does not readily explain. Report VTUAD's own held-out performance as the in-domain reference and state plainly that site and population shift are not separated without a within-site control.

### A9 — Matchups and split hygiene *(Day 4, gated on I3)*

`boatphone/geo.py`, `boatphone/matchup.py`, `boatphone/splits.py`.

- Range and bearing from the hydrophone via `pyproj`.
- Join each detection to its ±15 min acoustic window; emit `matchups.parquet` with **`overpass_id` on every row** and **AIS columns present but nullable**, so the optional AIS stage drops in later against an unchanged schema.
- **Dominance stratification:** `R₂ > 2·R₁` → acoustically dominated, the MVP subset (larger than `n_vessels == 1` would be); genuine mixtures → nearest-vessel range plus count; `n_vessels == 0` → negatives.
- **`splits.py` is the single source of splits.** Overpass-grouped, time-blocked CV, with a hard assertion that no `overpass_id` appears in both train and test.

**The effective sample size is the number of usable overpasses (order 20–30), not the number of detections (100–400).** A scene with 8 vessels gives 8 labels on *one* acoustic mixture — eight labels on one measurement. For anything acoustic (sea state, tide, SSP, ambient, propagation, hydrophone state) the unit of analysis is the overpass. Report uncertainty clustered by overpass, and **state both counts alongside every metric**: "n = 214 detections across 23 overpasses."

### A10 — Results *(Day 5)*

Detectability curve, physics baseline, transfer gap, matchup dataset. Final notebooks, README workflow section, presentation. **Every model number carries its overpass count; every population statistic carries the sampling-conditionality caveat** — PlanetScope is sun-synchronous at ~10:30 local and usable scenes are cloud-free summer days, precisely the conditions that maximise recreational traffic, so no population statistic here generalises to Folger Passage at large.

---

## Verification

| What | How | Pass criterion |
|---|---|---|
| Uptime calendar | Cross-check listing-derived calendar against A4's absent-file log | Disagreements enumerated; pull treated as authoritative |
| A4 window coverage | Assert every downloaded file's UTC timestamp maps to 09:15–11:45 `America/Vancouver` | 100%, across a DST boundary |
| A4 completeness | Downloaded file count vs listing count for the same span | Every gap explained by the uptime calendar |
| FFT reader — structure | Row/bin counts; zero-column periodicity | 1200 × 512; zeros at col 0 and 419–511 |
| **FFT reader — mapping** | **38 kHz echosounder line position; anti-alias shoulder position** | **Line at bin 152 ± 1; shoulder onset ≈ bin 408. Confirms 250 Hz/bin** |
| **FFT reader — magnitude** | **Regress product bin *b* vs WAV bin *2b*, bins 1–205, censoring-aware, like estimators, on a loud window** | **Slope ≈ 1; intercept characterised; convention documented** |
| Time alignment | Cross-correlate on the 38 kHz transient | Lag resolved to < 1 frame (0.25 s) |
| Calibration coverage | Assert every absolute-level output falls in bins 1–205 | Enforced in `calibrate.py`; violations raise |
| Calibration accuracy | Quiet-period levels vs Wenz curves for the sea state | Within a few dB |
| Band claims | Assert decidecade bands only requested above 2.2 kHz | Enforced in `features.py` |
| Deployment stability | Re-run the full gate on one file from each deployment 2020–2026 | Mapping and convention consistent, or per-deployment constants recorded |
| SL/TL priors | Modelled RL vs measured RL, known-size vessels, supported bands | Bias < 6 dB, no range trend |
| Physics posterior | Residuals vs range, size, frequency; posterior width reported | No systematic structure; identifiable variables stated |
| Band-matching | Assert VTUAD and Folger share a common band before comparison | Enforced in `models.py` |
| Split hygiene | Assert no `overpass_id` in both train and test | Zero overlap, enforced in `splits.py` |
| Transfer result | VTUAD-trained model on satellite-labelled vessels, overpass-grouped | Gap by size class, with effective N, clustered uncertainty, confound named |
| Reproducibility | Fresh clone + `environment.yml`, run `notebooks/A1`→`A6` | End to end from an ONC token, no AIS required |

---

## Risks

| Risk | Mitigation |
|---|---|
| **A2 low-frequency anomaly never resolves** | Relative/shape features fallback, decided Day 1. Detectability curve and transfer result both survive it; only the physics baseline is lost. |
| **No 256 kHz calibration curve from ONC** | Restrict absolute levels to ≤51.2 kHz (bins 1–205). Costs the 51–104 kHz band, which matters little for vessel noise. |
| **I1 timestamps in local time or publish time** | Agree the convention in writing Day 1; assert UTC and sanity-check against solar time (~10:30 local overpass). |
| **I1 or I3 arrive late** | A1–A8 are deliberately independent of both. Only A9 blocks, and A4's speculative pull means it starts the moment `overpasses.csv` lands with no download latency. |
| **Planet quota spent on hydrophone gaps** | O1 delivered Day 1 from listings, before any orders are placed; refined from A4's absent-file log. |
| **A4 throughput worse than 1 file/sec** | Pull newest summer first and work backwards, so the corpus is always useful at whatever depth it reaches. Two summers is the target, one is sufficient for the gate and ambient work. |
| **Overpasses fall outside 09:15–11:45 local** | Verify against the first `overpasses.csv` delivered; the window is stated as covering the whole period of interest, but confirm rather than assume. Any stragglers can be pulled individually — it is a few MB. |
| **`.fft.gz` semantics vary across deployments** | Per-deployment gate re-run; per-deployment constants if needed. |
| **Effective N too small for any ML claim** | Physics baseline (A7) needs ~0 training data and produces a result regardless. |

## Assumptions made

- **Timeline is the OHW hackathon week**, Day 1 = Thu Aug 27, matching the parent plan's five-day structure.
- **This stream owns the full acoustic stack** — acquisition through physics through the acoustic half of the ML — with optical acquisition and detection owned elsewhere.
- **`matchup.py` sits on the acoustics side** as the consumer of `detections.parquet`, since the join is against acoustic windows on the acoustic time base.
- **PlanetScope overpasses fall within 09:30–11:30 local for the entire period of interest**, per the team. A4's speculative pull is built on this; it is verified against the first `overpasses.csv` rather than taken on trust.
- **AIS stays an optional stage.** Nothing above depends on it. If access lands, it adds the blind-spot fraction, range-label validation and the within-site control.

## Out of scope

Acoustic Doppler speed estimation (source-of-truth goal 4 — and the JASA finding that planing hulls show weak speed–SL dependence makes it harder than it looks); a second observatory site; the Folger Pinnacle hydrophone (build for Folger Deep alone, with the hydrophone as a config parameter, and add Pinnacle only if Day 3 lands early); training any CNN from scratch; blind source separation; pre-2020 deployments; bulk WAV/FLAC acquisition; sub-250 Hz features as a pipeline component.
