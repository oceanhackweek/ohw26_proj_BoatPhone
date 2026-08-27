# BoatPhone — One-Week Build Plan (rev. 3)

## Context

`docs/plans/Project_Source_of_Truth.txt` sets out to estimate vessel **count, size class, and range** from a Folger Passage hydrophone, targeting the small recreational vessels that never broadcast AIS. `references/Isaac_Reaserch.txt` establishes the research gap precisely: the optical and acoustic communities each solve half this problem, and **no published work uses optical satellite detections to generate distance ground truth for acoustic ML training of non-AIS vessels**. That gap is the contribution.

The approach inverts the usual pipeline: filter PlanetScope first (cloud, then vessel detection), then pull only the matching hydrophone windows. This collapses the acoustic download from terabytes to a few MB per scene.

Repo is currently a bare OHW template — a README, an empty `final_notebooks/project_result_1.ipynb`, two zero-byte contributor notebooks, and a 21-byte placeholder `data/data.csv`. Everything below is new code.

**Two constraints shape everything below.**

1. On an E&R Basic Planet account the matchup dataset is bounded by quota at roughly **20–30 satellite overpasses**, containing perhaps 100–400 vessel detections. Those detections are *not* independent acoustic samples (see "Effective sample size"). The plan is built around never learning ship acoustics from scratch.
2. **Local AIS access is unconfirmed.** AIS is therefore staged as an optional validation and enrichment layer. Nothing on the critical path may depend on it (see "Staging AIS").

*Rev. 3 revises rev. 1 in response to reviewer feedback and to empirical checks against the sample data. A changelog is at the end. The most consequential finding: the `.fft.gz` magnitude validation does not currently pass, which makes it the project's only true blocker.*

---

## Framing

The project is organised around the question the satellite data uniquely answers, not around the range estimator.

**Primary research question**
*Do models trained on AIS-labelled underwater-acoustic corpora transfer to the small-vessel population that satellite imagery reveals at Folger Passage?*

Note the careful wording. Without local AIS you cannot verify that the satellite-detected population is *non-AIS* — you know only that these are satellite-detected small vessels. Their AIS-carriage status is an **inference from size class** (small recreational craft in Canadian waters largely do not carry AIS; Class B is voluntary below the carriage thresholds), not a measurement. Carry that as a **stated assumption**, and upgrade it to a measurement in the optional AIS stage if access lands. The transfer gap is a real result either way; the stronger "AIS blind spot" phrasing is held back until it can be evidenced.

**Dataset contribution**
PlanetScope + hydrophone cross-modal matchups at Folger Passage, with AIS columns present but nullable.

**Primary outputs (satellite labels + acoustics only)**
1. Range-dependent acoustic detectability of small vessels — P(detect) vs range vs size class.
2. Transfer-performance gap: AIS-corpus-trained model → satellite-labelled small-vessel population.
3. The matchup dataset itself.

**Supporting roles**
- **Physics forward model** — baseline and interpretation tool, not a headline claim.
- **AIS enrichment** — optional stage, gated on access.
- **Domain adaptation** — stretch goal.
- **Acoustic range/size inference** — secondary experimental goal. The project must not depend on it succeeding.

This ordering makes the week hard to fail: a negative transfer result is as publishable as a positive one, and both the dataset and the detectability curve stand alone.

---

## What the sample data actually is

Decoded from `data/Folger Deep Hydrophone Data Sample/` (device `ICLISTENHF1266`, ONC deviceId 23235, Folger Deep):

| Product | Size / 5 min | Per day | Role |
|---|---|---|---|
| `.wav` 128 kHz, 24-bit, mono | 115 MB | ~33 GB | Format validation + a small targeted subset |
| `.flac` | 50 MB | ~14 GB | Lossless alternative for the targeted subset |
| `.mp3` | 4.8 MB | ~1.4 GB | Fallback only (lossy above ~16 kHz) |
| **`.fft.gz`** | **0.29 MB** | **~84 MB** | **Bulk input** |

### Layout — inferred and self-consistent, but the values are not understood

614,400 newline-delimited integers = **1200 rows × 512 bins**. Zero-runs sit at indices 417–511 of every row, which fixes row length at 512. Under a forward, linear bin mapping:
- 0.25 s per row (1200 rows over 300 s)
- **125 Hz per bin** (1024-pt FFT at 128 kHz, bins 0 → Nyquist 64 kHz), i.e. bin centres at 0, 125, 250, 375 Hz …
- 417 × 125 Hz = 52.1 kHz ≈ the ISO 7605 anti-alias limit of 0.4·fs = 51.2 kHz. **This correspondence is the strongest evidence for the layout** — reversing the bin order breaks it.
- Values are integers 0–86 (87 distinct), consistent with 1 dB quantisation, and **censored at 0**: bin 0 sits at 0.0 with std 0.16 across all 1200 rows.

**But the value semantics are unresolved and currently contradicted by the WAV.** Measured on the sample files:

- **The calibration is flat.** Sensitivity is −39.7 ± 0.4 dB from 125 Hz to 20 kHz, rolling off only below ~30 Hz (−58.4 dB at 10 Hz). Instrument sensitivity explains no low-frequency shape in the product.
- **The WAV is physically sensible.** Welch PSD (1024-pt, Hann, full 300 s): −114.6 dB at 125 Hz, −116.1 at 1 kHz, −125.8 at 5 kHz, −145.1 at 50 kHz. Flat 125 Hz → 1 kHz (−1.5 dB), then a smooth monotonic decline. Normal ambient.
- **The `.fft.gz` product is not.** Median-over-time profile: bin 0 = 0, then 16, 34, 43, 50, 55, 58, 59, 60, 61 — a **+45 dB rise** from 125 Hz to ~1 kHz, plateauing near bin 9, declining to ~10 by 50 kHz. A 45 dB climb toward 1 kHz is neither ambient ocean noise nor the instrument.
- **The magnitude regression fails.** Product dB against WAV-derived dB, excluding values censored at 0 and 86:

| Band | n | slope | intercept | r |
|---|---|---|---|---|
| 125 Hz – 52 kHz | 415 | **1.370** | 18.00 | 0.695 |
| 1 – 50 kHz | 392 | **1.452** | 17.60 | 0.669 |
| 5 – 50 kHz | 360 | **1.006** | 18.24 | 0.503 |

Slope reaches ≈1 only above 5 kHz, and even there r = 0.50. Median-of-dB versus mean-of-power is a ~1–2 dB roughly constant bias and cannot account for a 45 dB shape difference.

**Treat the layout as a hypothesis and the levels as unknown until the Day 1 gate passes.** Every absolute-level deliverable rests on this.

**Calibration file** (`...PreDeploymentCalibration.txt`): 5120 rows, 10 Hz → 51,200 Hz at 10 Hz spacing, dB re Counts²/µPa². Interpolate onto the product's frequency grid and subtract for absolute dB re 1 µPa²/Hz. **Valid from 18-Feb-2020 onward, this device only** — so restrict to 2020–2026 rather than back to 2014. Earlier Folger deployments are different devices with different sensitivities. Six summers on one calibration removes a whole class of silent dB bugs.

---

## Frequency support — what the bulk product can and cannot do

**Sub-125 Hz structure is unavailable.** Under the inferred layout there is no bin between DC and 125 Hz, so low-frequency vessel tonals — including the 40–60 Hz band the shallow-water ranging literature favours — cannot be resolved from the bulk product by any amount of processing.

One nuance, since the absolute claim is slightly too strong: bin 0 spans 0–62.5 Hz and bin 1 spans 62.5–187.5 Hz, so 40–60 Hz *energy* is present as a coarse aggregate even though 40–60 Hz *structure* is not. In this sample bin 0 is pinned at 0 (censored, DC-blocked), so the aggregate looks unusable — confirm empirically rather than assuming either way.

**The stronger reason to avoid the low bins is that their semantics are broken**, not merely coarse: the unexplained +45 dB rise across 125 Hz–1 kHz makes them the least trustworthy part of the product.

Consequences to design around:

- **Track B computes only features supported at ≥125 Hz**, and preferentially above 1 kHz until the gate resolves the low end. Broadband and band levels, spectral shape and slope, high-frequency roll-off, cavitation-band energy, DEMON envelope on higher-frequency carriers.
- **Third-octave / decidecade bands are only properly resolved above ~1 kHz.** A decidecade band centred at *f* has bandwidth ≈0.23·f; two 125 Hz bins require *f* ≳ 1.1 kHz. Between 125 Hz and ~1 kHz you have a handful of raw bins, not resolved bands. Report those as raw-bin levels, not standards-compliant band levels.
- **PyPAM hybrid millidecade is not directly applicable.** The standard hybrid scheme uses 1 Hz resolution below ~455 Hz, which this product cannot supply. Use PyPAM for ≥1 kHz decidecade bands and for its calibration/PSD conventions; do not claim hybrid-millidecade compliance across the full band.
- **Small recreational craft are the favourable case.** Outboards and planing hulls put substantial energy well above 125 Hz — into the hundreds of Hz and kHz — so the loss hurts large-ship low-frequency signatures more than the target population. Say so explicitly rather than hoping nobody notices.

**Separately assess whether a small WAV/FLAC subset justifies sub-125 Hz features.** A scoped experiment, not a pipeline change: pull WAV/FLAC for order tens of windows spanning the dominated-matchup subset, compute the <125 Hz features the bulk product cannot, and measure whether they add anything over the ≥125 Hz set. Only a large gain would justify discussing a wider WAV pull — and that discussion must re-price the whole acquisition strategy, since it forfeits the few-MB-per-scene advantage that makes this project tractable. Default expectation: stay on `.fft.gz`.

---

## Planet API access — what is actually possible

**Can we extract arbitrary small windows? Technically yes, economically no.** The [Orders API clip tool](https://docs.planet.com/develop/apis/orders/tools/) accepts arbitrary GeoJSON polygons/multipolygons down to **1 m²** (no holes, no overlapping multipolygons, ≤1,500 vertices). You are *not* limited to Planet-determined tiles.

**But E&R Basic bills under "preferred clipping":**

> charge = **max(100 km², clipped area) per intersecting scene**
> ([minimum quota under Basic E&R](https://support.planet.com/hc/en-us/articles/4425683189533-What-is-the-Minimum-Quota-Charged-Under-Basic-E-R-Plan), [Orders API FAQ](https://support.planet.com/hc/en-us/articles/360011216618-Orders-API-FAQ))

A 1 km² clip and a 100 km² clip cost the same, which inverts the optimisation:

| Design | Charge/overpass | Orders per 3,000 km² month |
|---|---|---|
| Small stratified boxes @ 13.5 km² | 100 km² | 30 — **small boxes buy nothing** |
| Full 10 km disk (314 km²) | 314 km² | ~9 |
| **Disk r = 5.5 km (95 km²)** | **100 km²** | **~30** |

PSScene footprints are ~32.5 × 19.6 km (SuperDove), ~25 × 11.5 km (Dove-Classic); an AOI straddling two scenes is billed twice because the minimum applies **per intersecting scene**. Keep the "single scene fully contains the AOI" filter. E&R PlanetScope also carries a 30-day delay, irrelevant for archive work.

**Order a full disk of radius ~5.5 km (95 km²) centred on Folger Deep.** It sits just under the 100 km² floor, so the area is effectively free — anything smaller is wasted headroom you are billed for anyway. And a *full* disk means every vessel inside 5.5 km is observed; a 10 km sector would leave three-quarters of the acoustic near-field unseen, and an unobserved close vessel corrupts a label far worse than a distant one.

**~30 is a quota ceiling, not a yield forecast.** State it as **"maximum ≈30 single-scene orders per month before other losses."** Usable overpasses are cut by scene geometry (single-scene containment), cloud, sun glint, seasonal hydrophone uptime and gaps, and vessel detectability. Do not hard-code 30 as an expected N anywhere; carry the realised count as a measured quantity reported at the Day 3 checkpoint.

Two optimisations for `planet.py`:
- **Search is free.** Filter to overpasses where a *single* PSScene fully contains the AOI. Straight 2× on effective quota.
- **Tile streaming does not consume download quota** ([FAQ](https://support.planet.com/hc/en-us/articles/16457647004701-Planet-Tile-Streaming-FAQ)). It is an excellent **human triage** mechanism for deciding which cloud-free overpasses deserve quota. Whether it supports *automated* detection of 6–12 m vessels is unverified and depends on the image representation, bit depth, band set and effective resolution the tile service actually returns. **Day 1 empirical check**: pull tiles for a scene with known vessels and compare against the ordered GeoTIFF. If tiles fail, fall back to human triage plus deliberate inclusion of some empty scenes — selection bias on negatives costs more than the quota saved.

---

## Staging AIS — what it changes, and the one claim it costs

Local ONC AIS access is unconfirmed, so **AIS is an optional stage attempted only after the satellite-labelled pipeline works.**

**What runs without any AIS — the primary path:**
- Satellite-derived range and size-class labels joined to Folger acoustic windows (`matchups.parquet`).
- **Acoustic detectability curve**: P(detect) vs range vs size class. Answers source-of-truth goal 3 directly.
- **Physics baseline** validated against satellite range labels.
- **Transfer experiment.** VTUAD is an *external, already-AIS-labelled* corpus, so training on it and testing on the satellite-labelled Folger population requires **no local AIS**. The headline ML result is unaffected by the staging.

**What is contingent on access — the AIS stage:**
1. Blind-spot fraction (AIS-matched vs dark) — converts the carriage assumption into a measurement.
2. Geospatial/temporal validation of the range labels (satellite vs AIS range).
3. Far-field confound accounting for vessels beyond the 5.5 km AOI.
4. The within-site control group (see "Transfer confound").

**Costs of the staging, stated rather than hidden:**
- The satellite-vs-AIS range check was the strongest end-to-end validation available. Without it, the range-label geometry chain is validated only by internal consistency (Planet's stated geolocation accuracy, ONC hydrophone coordinates from metadata), not against an independent source. Flag as a limitation.
- "Non-AIS" becomes an assumption rather than a measurement, as set out in the Framing section.

**AIS-free fallbacks worth building anyway:**
- *Far-field diagnostic.* Windows with zero in-AOI detections but elevated broadband level are direct evidence of out-of-AOI traffic. This falls out of the `n_vessels == 0` negatives already being collected, and bounds how often the far field contaminates labels.
- *In-domain reference.* VTUAD's own held-out set serves as the reference point for the transfer gap. Weaker than a within-site control — it does not separate site shift from population shift — and the write-up must say so.

Keep the AIS columns in `matchups.parquet` **present but nullable**, so the enrichment step drops in later against an unchanged schema.

---

## Effective sample size — the honest accounting

**100–400 vessel detections is not 100–400 independent acoustic samples.**

The acoustic observation is one mixed spectrum per time window. If a scene contains 8 vessels, those 8 detections annotate a *single* acoustic mixture — eight labels on one measurement, not eight measurements. For anything driven by the acoustic side (sea state, tide, sound-speed profile, ambient, propagation, hydrophone state), the effective sample size is closer to **the number of usable overpasses, order 20–30**, than to the number of detections.

So any claim that "300 samples is ample for fitting a linear head" is withdrawn.

**The fix is clustering, not collapsing to N = 25.** The unit of analysis differs by question — vessel-level for optical statistics (detection P/R, size distribution), overpass-level for anything acoustic. Asserting a flat N = 25 everywhere would discard real vessel-level precision. Rules, non-negotiable:

- **All detections and all acoustic windows from one Planet overpass stay in the same train/test fold.** Grouped CV with `overpass_id` as the group. Random splits at the detection level leak almost by construction.
- **Temporal blocking on top of that** — hold out whole months or a season, since adjacent windows and adjacent days share conditions.
- **Report uncertainty clustered by overpass** (cluster-robust or block bootstrap over overpasses), not per-detection standard errors.
- **State both counts alongside every metric:** "n = 214 detections across 23 overpasses."
- Prefer few parameters. At n ≈ 20–30 acoustic units, a linear head on frozen embeddings is near the ceiling of what is defensible, and even that needs grouped CV.

---

## Is the dataset enough? — the tiered answer

**Trained from scratch: no.** Continuous range and size regression from raw spectrograms, on tens of independent acoustic observations across six years of varying conditions, would not generalise. Do not attempt it and do not promise it.

**But almost none of this needs to be learned from your data.** The acoustics are published and the AIS-labelled corpora exist. The matchups are worth far more as *evaluation and calibration* than as training data.

### Tier 1 — Physics with published priors (needs ~0 training data)

Source level as a function of length, speed and class is a solved, open-access, regionally-validated problem:

- **[MacGillivray & de Jong 2021](https://doi.org/10.3390/jmse9040369)** (JOMOPANS-ECHO) — SL spectrum in decidecade bands vs frequency, speed, length, ship type. Validated against the Vancouver Fraser Port Authority **ECHO programme — BC waters**. Stated uncertainty ±6 dB.
- **[Estimation of Source Levels of Small Vessels](https://www.mdpi.com/2077-1312/14/6/561)** (JMSE 14:561) — 25 vessels <12 m, 5 sites, by hull class (planing / semi-displacement / displacement); evaluates JOMOPANS-ECHO against them and proposes modifications for recreational craft. Exactly the population MacGillivray does not cover.
- **[Lagrois et al. 2023](https://doi.org/10.3390/s23031674)** (Sensors 23:1674) — monopole source levels of small recreational vessels, St. Lawrence Estuary. Canadian, open access.
- **[JASA 156:2077](https://pubs.aip.org/asa/jasa/article-abstract/156/4/2077/3314966/Speed-dependence-sources-and-directivity-of-small)** — speed dependence and directivity of small-vessel noise. Planing hulls' SL rises with speed until planing, then flattens, so speed is weakly identifiable in exactly the regime recreational craft operate in.

Transmission loss from **Bellhop via [`arlpy.uwapm`](https://arlpy.readthedocs.io/en/latest/uwapm.html)**, with real bathymetry (GEBCO / Canadian Hydrographic Service) and a sound-speed profile from ONC's **co-located Folger Passage CTD**. Evaluate only on bands the product supports.

**Identifiability — the corrected claim.** The forward model is

```
RL(f) = SL(L, v, class, f) − TL(R, f)
```

It is tempting to say this leaves "only a small number of local correction terms to fit" and that one can "invert for range and length." That holds only during scenes where Planet already supplies vessel positions and approximate sizes. For **acoustic-only** inference the unknowns include range, size, hull/source class, speed, number of contributing vessels, heading/directivity, and the local propagation/source-level correction. A single hydrophone does not uniquely determine all of them.

So Tier 1's goal is:

> **Estimate a probabilistic range/size posterior under explicit priors on vessel class and speed, and determine empirically which variables are identifiable.**

Put priors on class, speed and count; sample or optimise the posterior over range and size; report posterior width, not a point estimate dressed as an inversion. Then use the matchups — the cases where range and approximate size are *known* — to establish what is actually recoverable and when.

**Run this on the time–frequency surface, not a single averaged spectrum.** The product gives 1200 frames at 0.25 s per file, and the matchup window is ±15 min. During a transit you get level-versus-time through closest point of approach: CPA time, the slope of the rise and fall, and the width of the peak jointly constrain range and speed in a way one spectrum cannot — the basis of standard single-hydrophone ranging. This does not make the problem determined (class, count and directivity remain confounded), but analysing a single averaged spectrum would understate what is recoverable.

Honest error budget: ±6 dB SL uncertainty against a ~15–18 dB/decade TL slope maps to roughly a **factor of ~2 in range**, before class/speed/count uncertainty. State this up front.

### Tier 2 — Transfer learning (the ML story)

- **Pre-train on [VTUAD](https://dx.doi.org/10.21227/msg0-ag12)** (Domingos et al.) — ONC, Strait of Georgia, icListen hydrophone at 147 m, AIS-labelled, June–Nov 2017. Built around **distance-based scenarios with inclusion/exclusion zones**, admitting a clip only when a single vessel is in the inclusion zone and none in the exclusion zone. Same operator, same province, same instrument family as Folger's icListen HF, clean single-source labels, range structure built in. As close to an in-domain pre-training corpus as exists.
- **Then linear-probe on the PlanetScope matchups** — a small head on frozen embeddings, fit and evaluated under overpass-grouped CV. [arXiv 2601.08358](https://arxiv.org/html/2601.08358) shows linear probing on frozen pretrained audio embeddings works for underwater target recognition; [arXiv 2409.13878](https://arxiv.org/pdf/2409.13878) (MIT Lincoln Lab) covers cross-domain transfer with pretrained models.
- **Band-match both sides before comparing — critical path.** VTUAD and any pretrained audio embedding assume a sample rate and band that differ from the ≥125 Hz product support. Without band-limiting both sides to the common support, the measured "transfer gap" is a preprocessing artefact rather than a domain shift. Make it an explicit step in `models.py`. Check VTUAD's sample rate, band and **label type** on Day 1 — the inclusion/exclusion-zone design suggests distance *bins* or scenario classes rather than continuous range, which changes what the transfer experiment can measure at all.
- **Wout et al. 2025** (already in the references) is the closest published analogue: a CNN on 116 days from two North Sea stations classifying **nearest-vessel** distance bins — independent validation of the nearest-vessel target formulation adopted below.

Fine-tuning beyond a linear head is a stretch goal at this effective N, not a plan step.

### Tier 3 — The transfer result, and its confound

Applying a VTUAD-trained model to the satellite-labelled Folger population varies three things at once:

1. **Population** — large commercial vs small recreational (the axis of interest);
2. **Site** — Strait of Georgia vs Folger Passage (different bathymetry, propagation, ambient);
3. **Time/instrument** — 2017 vs 2020–2026, different deployment.

A measured degradation therefore cannot be attributed to the population axis, which is the entire claim. **Name this confound in the write-up regardless** — it is the difference between a defensible result and an uninterpretable one.

**Clean fix (needs the AIS stage):** use Folger's AIS-visible vessels as a within-site control. VTUAD → Folger-AIS isolates site/instrument shift; Folger-AIS → Folger-small isolates population shift within-site. The second comparison is the publishable one, and it is nearly free once AIS matchups exist.

**AIS-free mitigations, on the primary path:**
- **Stratify the transfer gap by size class.** A gap that grows monotonically as vessels get smaller is evidence for the population axis that site shift does not readily explain.
- Report VTUAD's own held-out performance as the in-domain reference, and state plainly that site and population shift are not separated without the control.

---

## What we reuse instead of building

| Need | Reuse | Saves |
|---|---|---|
| Source level vs length/speed | JOMOPANS-ECHO + small-vessel extensions (above) | The entire SL fit |
| Transmission loss | `arlpy` + Bellhop; GEBCO bathymetry; ONC Folger CTD for SSP | Hand-rolled TL |
| Calibrated band levels (≥1 kHz) | **PyPAM** (ISO 18405 PSD conventions, decidecade bands), [`mbari-org/pbp`](https://github.com/mbari-org/pbp) | Standards-compliant feature code |
| Acoustic pre-training corpus | **VTUAD** (external, AIS+range labels) | Months of labelling |
| Audio embeddings | PANNs / VGGish, frozen, band-matched | Training a CNN we cannot afford |
| **Vessel detection in imagery** | **Pretrained YOLO from Mäyrä et al. 2025** — [GitHub](https://github.com/mayrajeo/ship-detection), [HF weights](https://huggingface.co/mayrajeo/marine-vessel-yolo), also `torchgeo/yolo11s_marine_vessel_detection`. 8,768 recreational-vessel annotations, F1 ≈ 0.86 / mAP50 ≈ 0.89 on Sentinel-2 | A day of detector work |
| AIS for the region (optional stage) | ONC AIS receiver data (same API, co-located) | Third-party AIS sourcing |

**Do not assume downsampling is the right domain-shift fix.** Mäyrä's weights are trained on **Sentinel-2 at 10 m**; PlanetScope is 3 m. Resampling to 10 m moves the imagery statistically closer to the training domain but destroys exactly the resolution PlanetScope was bought for. **Test empirically on the first scene:**

1. **Native 3 m PlanetScope** fed directly;
2. **Resampled to 10 m**;
3. **Blurred/resampled to approximate Sentinel-2's point-spread characteristics** (PSF-matched, not just decimated).

Score against **50–100 hand-labelled targets** — a couple of hours that settles the question with data instead of assumption.

**Frame the experiment as scale-and-radiometry matching, not GSD alone.** What actually matters is apparent object size in pixels relative to the scales the detector was trained on, plus radiometric match. Two levers the three treatments do not isolate: **tiling / detector input size**, which changes apparent scale without touching resolution; and **band-centre and reflectance-scaling differences** between Sentinel-2 and PlanetScope, which can matter as much as resolution for a detector keyed on water/hull contrast. Whatever wins for *detection*, return to native 3 m for morphometry.

Keep a classical NIR top-hat detector as fallback if transfer disappoints across all treatments; water is near-black in NIR and hulls plus wakes are bright, so it is ~100 lines with `cv2` (installed).

---

## Label quality — what regression can and cannot deliver

Range supports continuous regression; size does not.

- **Range labels are excellent.** PlanetScope geolocation is <10 m RMSE against ranges of 500–5,500 m — 0.2–2% label error. Continuous range regression is well-posed. (Note the staging caveat: without AIS this rests on Planet's stated accuracy rather than independent verification.)
- **Length labels are at the resolution floor.** At 3 m GSD a **6 m hull is ~2 pixels** and a **12 m hull ~4 pixels**, before wake contamination, sub-pixel mixing and orientation effects. Planet-derived length is not a trustworthy continuous ground-truth variable at the small end — which is the entire target population.

**So the primary optical label is coarse vessel size class.** Report the FAO bins (0–12 / 12–24 / >24 m) the source-of-truth asked for, as a confusion matrix.

**But the FAO bins alone are degenerate here.** The target population sits almost entirely inside 0–12 m, so a 3-class confusion matrix will show high accuracy and no discrimination where the science is. Add an **exploratory finer split inside 0–12 m** (e.g. <6 / 6–12), stated up front as sitting at the resolution floor. **Continuous length stays exploratory**: measure it, report its distribution and its disagreement with class, but build no headline metric on continuous length RMSE. Where a continuous number is needed downstream (e.g. the SL prior), pass a size *posterior* with honest width rather than a point estimate.

**The multi-vessel mixture is the central methodological problem** — and the place to make it the method rather than a filter. At 95 km² over summer Barkley Sound, N > 1 is the common case, so discarding mixtures would throw away most of an already small dataset. Model all vessels at once:

```
RL_total(f) = 10·log10( Σᵢ 10^((SL(Lᵢ, vᵢ, f) − TL(Rᵢ, f)) / 10) ) ⊕ ambient(f)
```

with `SL` from JOMOPANS-ECHO and `TL` from Bellhop. Note this is one *observation* constraining many vessel parameters — it adds terms to the model, not independent samples.

Then stratify:
- **Acoustically dominated** (`R₂ > 2·R₁`) → nearest vessel dominates, a comparatively clean single-source target even with several boats present. **The MVP subset**, and larger than `n_vessels == 1` would be.
- **Genuine mixtures** → nearest-vessel range plus count (matching Wout et al.).
- `n_vessels == 0` → negatives for detection, ambient baselining, and the far-field diagnostic.

**Confound to state:** vessels beyond 5.5 km are outside the AOI but still audible. With AIS, account for them directly — out-of-AOI traffic skews large and AIS-carrying. Without AIS, use the level-based far-field diagnostic above to bound the contamination, and state the residual as a limitation.

**Sampling conditionality.** PlanetScope is sun-synchronous, ~10:30 local, and usable scenes are cloud-free summer days — precisely the conditions that maximise recreational traffic. Any population statistic derived from these scenes (size distribution, vessel density, and the blind-spot fraction if the AIS stage runs) is conditional on "cloud-free summer mid-morning," not general to Folger Passage, and carries wide overpass-clustered uncertainty at 20–30 overpasses. State this wherever such a number appears.

---

## Deliverables, in priority order

**Must-have (no AIS required)**
1. Validated, calibrated `.fft.gz` reader — structure *and* magnitude. Nothing works without it.
2. Optical detections with measured P/R and a chosen inference treatment.
3. `matchups.parquet` — satellite detections ↔ acoustic windows, with `overpass_id` and nullable AIS columns.
4. **Acoustic detectability curve** — P(detect) vs range vs size class (source-of-truth goal 3).
5. A simple **physics baseline** — forward model + posterior, with the error budget and identifiability findings.

**Very strong result**

6. **VTUAD-trained model tested on the satellite-labelled small-vessel population** — the transfer gap, band-matched, size-stratified, with the confound named.

**Contingent on AIS access**

7. Blind-spot fraction — converts the AIS-carriage assumption into a measurement.
8. Geospatial/temporal validation of the range labels against AIS.
9. Within-site control group separating site shift from population shift.

**Stretch**

10. Fine-tuning / domain adaptation.
11. Joint range + size posterior inversion with a full identifiability analysis.
12. Optical speed from inter-band offsets.

---

## Day-by-day

### Day 1 — De-risk and gate

**Blockers first, in the first hour.** ONC API token (the API returns 401 without one, confirmed). Planet API key, **remaining quota, and whether the account bills Preferred or Premium**. Start the VTUAD download (IEEE DataPort) — large, run it overnight. Ask ONC about local AIS availability, but do not wait on the answer.

- `boatphone/fft_io.py` — reader for the 1200×512 layout, plus ONC download via the installed `onc` 2.6.0 client. Discover Folger location codes at runtime with `onc.getLocations()` rather than hardcoding.
- `boatphone/calibrate.py` — interpolate the 10 Hz sensitivities onto the product grid, subtract, return dB re 1 µPa²/Hz.

- **THE BLOCKING GATE — structure, magnitude, and mapping.** This is the project's only true blocker and it **does not currently pass** (slope 1.37, r 0.70 full band; +45 dB low-frequency discrepancy against the WAV). Correlation alone is insufficient: a spectrogram can correlate at r > 0.99 while carrying the wrong dB offset, scaling, PSD normalisation, FFT normalisation, window-energy correction or calibration convention, and a silent 20–30 dB error would wreck the physics while producing beautiful spectrograms. Require all three:
  - **Structure** — bin and row counts and spacing; correlation against a spectrogram recomputed from the supplied `.wav` (`scipy.signal.spectrogram`, 1024-pt, 128 kHz) over the overlapping window. Require **r > 0.95**. The two FFT files are 5 min apart (`235504`, `000004`) and the WAV starts `000000.029`, so a clean overlap exists.
  - **Mapping** — do not inherit the inferred bin↔frequency mapping. Search candidate FFT lengths and orderings and select the mapping that maximises agreement with WAV spectrograms. The anti-alias correspondence at index 417 is the current best evidence, not proof.
  - **Magnitude** — regress decoded dB against WAV-derived dB across bins and frames. Require **slope ≈ 1** within a stated tolerance, characterise the intercept, and identify the convention: power, PSD (per Hz), spectral level, mean-square amplitude; window energy corrected; one-sided ×2. **Handle censoring at 0 and 86** — regressing over clipped values biases slope toward 1 spuriously. **Compare like estimators** (median-of-dB vs median-of-dB, not median vs Welch mean). Record the resolved convention and intercept in `calibrate.py` as documented constants and re-run whenever the reader changes.
  - **Absolute received levels are blocked until this passes**, which blocks the physics baseline. **Fallback if unresolved within a day:** proceed on relative/shape-based features, defer the physics baseline, and escalate to ONC about the product definition. Decide this on Day 1, not on Day 4.

- `boatphone/planet.py` — Data API search over the 95 km² disk, 2020–2026, May–Sep, cloud filter, single-scene-containment filter, quota projection expressed as a *ceiling*. **Search only, no orders yet.**
- **Tile-streaming check.** Pull streamed tiles for one scene with known vessels; inspect band set, bit depth, effective resolution and rendering. Decide whether tiles can support automated triage or whether triage stays human-in-the-loop. Record the answer either way.
- **VTUAD metadata check** — sample rate, band, label type. Feeds the band-matching step, which is on the critical path.
- **Go/no-go:** if confirmed quota buys fewer than ~20 overpasses, add Sentinel-2 via `earth-search.aws.element84.com` (reachable, tested) as a parallel coarser track — and note Mäyrä's weights are *native* to Sentinel-2, so that track needs no domain shift. It loses the small end of the 0–12 m class, so treat it as a supplement, never a replacement.

### Day 2 — Three tracks in parallel

**Track A — optical.** Stand up Mäyrä's pretrained YOLO. Hand-label 50–100 targets in a pilot scene, then score the inference treatments (native 3 m / 10 m resample / PSF-matched, plus tiling and radiometric scaling as levers) and pick one on measured P/R. Calibrate pixel-length → metres at native 3 m; bin to FAO classes as the primary label, with the sub-12 m split as exploratory.

**Track B — acoustic features.** `boatphone/features.py`: **features supported at ≥125 Hz**, preferentially above 1 kHz until the gate resolves the low end — broadband and band levels, spectral shape and slope, high-frequency roll-off, cavitation-band energy, DEMON envelope on higher-frequency carriers. Use PyPAM for decidecade bands above ~1 kHz and for its calibration/PSD conventions; below that report raw-bin levels and say so. **Separately**, pull a small WAV/FLAC subset (tens of windows) and assess whether sub-125 Hz features add anything. An experiment with a decision, not a second pipeline.

**Track C — priors.** Implement JOMOPANS-ECHO `SL(L, v, class)` plus the small-vessel extension; stand up Bellhop through `arlpy` with GEBCO bathymetry and an ONC Folger CTD profile. Evaluate on supported bands only. Sanity-check modelled TL against the ONC ambient record.

### Day 3 — The matchup dataset

- `boatphone/geo.py` — range and bearing from the hydrophone (`pyproj`, installed).
- `boatphone/matchup.py` — join each detection to the ±15 min FFT window around the scene's `acquired` timestamp (~1.7 MB/scene). Emit `data/processed/matchups.parquet` with the dominance stratification, **an `overpass_id` column on every row** (the CV grouping key every downstream split depends on), and **nullable AIS columns** so the optional stage drops in later against an unchanged schema.
- Build the **far-field diagnostic** from the `n_vessels == 0` windows: zero in-AOI detections plus elevated broadband level indicates out-of-AOI traffic.
- **Checkpoint:** report usable overpasses and dominated matchups. Those two numbers — not the detection count — decide Day 4.
- **Optional AIS stage, only if access landed:** pull ONC AIS for the same windows; flag each detection AIS-matched or dark; compute the blind-spot fraction; and run the **geospatial and temporal validation of the satellite-derived range labels** — for vessels that *do* carry AIS, compare satellite-derived range against AIS-derived range. Agreement to tens of metres validates **Planet geolocation, time matching, hydrophone coordinates and AIS association** — the geometry and timing chain. It does **not** validate the detector on small vessels, because AIS-visible vessels are a biased subset (larger, faster, differently rendered); detector performance comes from Track A's hand-labelling. Keep those two claims distinct.

### Day 4 — Baseline and the transfer result

Deliberately narrow. Physics inversion, embeddings, ML regression, transfer evaluation and cross-validation do not fit in one day together.

- **Physics baseline (must-have).** Forward model with published priors; range/size **posteriors** under explicit class and speed priors on the dominated subset, computed over the time–frequency surface. Report the error budget and which variables proved identifiable. The safety net — produces a result regardless of N. *Gated on the Day 1 magnitude check; if that failed, run the shape-based fallback instead.*
- **Transfer result (the target).** Apply the VTUAD-trained model to the satellite-labelled population, band-matched, and quantify degradation against its in-domain held-out performance. **Stratify the gap by size class.** **Overpass-grouped, time-blocked splits only.** Report effective N (overpasses) with overpass-clustered uncertainty, and name the three-way confound.
- **If AIS landed:** add the within-site control (VTUAD → Folger-AIS → Folger-small).
- **Only if the above land early:** a linear head on frozen embeddings under the same grouped CV, compared against the physics baseline.
- Do **not** train a CNN from scratch, and do not fine-tune beyond a linear head at this N.

### Day 5 — Results and write-up

Detectability curve, transfer gap, matchup dataset, `final_notebooks/`, README, presentation. Every model number carries its overpass count; every population statistic carries the sampling-conditionality caveat.

---

## Files

```
boatphone/                 # importable package, not notebook-only code
  fft_io.py                # 1200×512 .fft.gz reader + ONC download
  calibrate.py             # sensitivities → product grid → dB re 1 µPa²/Hz
                           #   + documented magnitude convention from the Day 1 gate
  features.py              # ≥125 Hz band levels, decidecade ≥1 kHz, DEMON, spectral shape
  priors.py                # JOMOPANS-ECHO SL model + small-vessel extension
  propagation.py           # arlpy/Bellhop TL, GEBCO bathymetry, ONC CTD SSP
  planet.py                # search / quota accounting / clip order / download
  detect.py                # pretrained YOLO (scale+radiometry treatments) + NIR top-hat fallback
  geo.py                   # range + bearing from hydrophone
  matchup.py               # detections ↔ acoustic windows → parquet (overpass_id; AIS nullable)
  models.py                # forward posterior, band-matching, embeddings + head, transfer eval
  splits.py                # overpass-grouped, time-blocked CV — single source of splits
notebooks/01..06           # one per stage, thin wrappers over the package
data/processed/matchups.parquet
```

`.gitignore` already excludes the sample WAV; extend to `data/raw/` and `data/interim/`.

**Install:** `rasterio`, `pystac-client`, `planet`, `arlpy`, `ultralytics`, `pypam` (or `mbari-pbp`), `huggingface_hub`. Already present: `onc` 2.6.0, `cv2` 5.0.0, `torch` 2.12, `sklearn` 1.9, `geopandas`, `shapely`, `pyproj`, `pyarrow`, `scipy`, `xarray`.

---

## Explicitly out of scope

Speed from acoustic Doppler (goal 4) — and note the JASA finding that planing hulls have weak speed–SL dependence, so this is harder than it looks. Also: a second observatory site; training any detector or CNN from scratch; blind source separation; pre-2020 deployments; bulk WAV/FLAC acquisition; sub-125 Hz features as a pipeline component (they remain a scoped Day 2 experiment).

**One stretch worth flagging as genuinely novel:** PlanetScope bands are acquired from different focal-plane strips with sub-second offsets, so moving vessels show band-to-band displacement — the effect used for ship velocity in Sentinel-2 ([Heiselberg 2019](https://www.mdpi.com/1424-8220/19/13/2873), ~2.6 s offsets). If the Dove per-band offset can be pinned down, that yields **vessel speed from a single scene**, giving goal 4 an optical ground truth. Verify the offset magnitude before committing time.

---

## Verification

| What | How | Pass criterion |
|---|---|---|
| FFT reader — structure | Bin/row counts and spacing; correlate recomputed WAV spectrogram vs decoded `.fft.gz` | 1200 rows @0.25 s, 512 bins; r > 0.95 |
| FFT reader — mapping | Search candidate FFT lengths/orderings; select best agreement with WAV | Mapping selected on evidence, not inherited |
| **FFT reader — magnitude** | **Regress decoded dB vs WAV dB, censoring-aware, like estimators** | **Slope ≈ 1; intercept characterised; convention stated. CURRENTLY FAILS (1.37 / r 0.70)** |
| Calibration | Quiet-period levels vs Wenz ambient curves for the sea state | Within a few dB |
| Planet tile streaming | Streamed tiles vs ordered GeoTIFF on a scene with known vessels | Suitability for automated triage decided empirically, recorded either way |
| Detector treatment | 50–100 hand-labelled targets across scale/radiometry treatments | P/R per treatment; one chosen on measured performance |
| Vessel detection | Same hand-labelled set | P/R reported; size-class accuracy measured |
| SL/TL priors | Modelled RL vs measured RL for vessels of known size, supported bands only | Bias < 6 dB, no range trend |
| Physics posterior | Residuals vs range, size, frequency; posterior width reported | No systematic structure; identifiable variables stated |
| Split hygiene | Assert no `overpass_id` appears in both train and test | Zero overlap, enforced in `splits.py` |
| Band-matching | Assert VTUAD and Folger features share a common band before comparison | Enforced in `models.py`; gap is not a preprocessing artefact |
| Transfer result | VTUAD-trained model on satellite-labelled vessels, overpass-grouped | Gap reported by size class, with effective N and clustered uncertainty; confound named |
| Size model | Same split | FAO 3-class confusion matrix primary; sub-12 m split and continuous length exploratory |
| *Range-label geometry (AIS stage)* | *Satellite- vs AIS-derived range, AIS-visible subset* | *Agreement to tens of metres — validates geolocation/timing/AIS association only* |
| Reproducibility | Fresh clone, run `notebooks/01`→`06` | End to end from API tokens, with no AIS required |

---

## Open items for the team

- **The `.fft.gz` magnitude discrepancy.** Highest priority. Is a high-pass or shaping applied in ONC's product generation? Ask ONC for the product definition in parallel with the Day 1 gate.
- **Planet quota remaining, and Preferred vs Premium billing.** Preferred means a 100 km² floor and a ceiling of ~30 orders/month; Premium bills actual area and would raise N substantially.
- **Whether local ONC AIS is accessible at all, and by when.** Determines whether the contingent deliverables run. Not a blocker.
- **VTUAD sample rate, band, label type, licence and download size** — now critical path via band-matching. Start the download Day 1.
- Folger Deep hydrophone uptime across 2020–2026 summers — ONC deployments have gaps, and uptime directly caps usable overpasses.
- Whether ONC's Folger Passage CTD has usable SSP coverage for the target periods.
- Whether `.fft.gz` layout *and* semantics are stable across 2020–2026 deployments — re-run the Day 1 gate on a file from each deployment.
- Whether to add the Folger Pinnacle hydrophone (23 m) as a second receiver. Two receivers relieve some of the single-hydrophone identifiability problem the references dwell on (Byun et al. 2026), but roughly double the wrangling. **Recommend building for Folger Deep alone with the hydrophone as a config parameter, adding Pinnacle only if Day 3 lands early.**

---

## Changelog — rev. 1 → rev. 3

Rev. 2 was an intermediate draft; this consolidates.

**From reviewer feedback:**
1. **40–60 Hz features removed.** The product has 125 Hz bins; sub-125 Hz structure is unrecoverable. Track B is now "features supported at ≥125 Hz," with sub-125 Hz relegated to a scoped small-subset WAV experiment. Added the consequences for decidecade bands (<~1 kHz unresolved) and PyPAM hybrid millidecade (not applicable). *Qualified:* bins 0–1 do carry sub-125 Hz energy as a coarse aggregate, and the stronger reason to avoid the low bins is their unresolved semantics.
2. **Identifiability claim narrowed.** "Invert for range and length" → "estimate a probabilistic range/size posterior under explicit priors, and determine empirically which variables are identifiable." *Qualified:* the observable is a time–frequency surface with CPA structure, not a single spectrum, so the analysis runs on the surface.
3. **Effective sample size corrected.** N is ~20–30 overpasses for acoustic analyses, not 100–400 detections. "300 samples is ample" withdrawn. Overpass-grouped CV mandatory, enforced in `splits.py`; uncertainty clustered by overpass. *Qualified:* the fix is clustering, not a flat N = 25 everywhere.
4. **Quota framing.** ~30 stated as a monthly ceiling before cloud, glint, geometry, uptime and detectability losses — not an expected yield.
5. **Tile streaming.** Retained as human triage; automated-detection suitability is a Day 1 empirical check.
6. **Detector domain shift.** No longer commits to 10 m downsampling; treatments scored against 50–100 hand-labelled targets. *Extended:* framed as scale-and-radiometry matching, adding tiling/input-size and band/reflectance differences as levers.
7. **Length claims weakened.** Primary optical label is size class. *Extended:* FAO bins alone are degenerate for a population concentrated in 0–12 m, so a sub-12 m exploratory split is added.
8. **FFT gate strengthened** to structure *and* magnitude. *Escalated:* the gate **currently fails** on the sample data, so it is now the project's only true blocker, extended to re-derive the bin↔frequency mapping, handle censoring at 0 and 86, compare like estimators, and carry an explicit fallback.
9. **AIS-range check renamed and rescoped** to geospatial/temporal validation of the range labels; it does not validate small-vessel detection.
10. **Day 4 simplified** to physics baseline + transfer result; deliverables re-ranked into must-have / very strong / contingent / stretch.

**From the AIS staging decision:**
11. **AIS demoted** from required input to an optional stage gated on access. Primary path runs on satellite labels plus acoustics. AIS columns in `matchups.parquet` are present but nullable. Costs stated: "non-AIS" becomes an assumption rather than a measurement, and the strongest end-to-end range-label validation leaves the critical path. AIS-free fallbacks added (level-based far-field diagnostic; VTUAD held-out as in-domain reference).
12. **Primary question restated** in measurable terms — transfer to the satellite-revealed small-vessel population, with AIS carriage as a stated assumption.

**Additional issues identified:**
13. **Transfer confound named.** VTUAD → Folger varies population, site and time/instrument simultaneously, so degradation cannot be attributed to the population axis. Clean fix needs the AIS within-site control; AIS-free mitigation is size-stratifying the gap.
14. **Band-matching required** between VTUAD and the ≥125 Hz features, or the measured gap is a preprocessing artefact. Now critical path.
15. **Sampling conditionality.** Sun-synchronous ~10:30 local, cloud-free summer scenes — all population statistics are conditional on those conditions, not general to Folger Passage.
