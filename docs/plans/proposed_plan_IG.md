# BoatPhone — One-Week Build Plan

## Context

`docs/plans/Project_Source_of_Truth.txt` sets out to estimate vessel **count, size class, and range** from a Folger Passage hydrophone, targeting the small recreational vessels that never broadcast AIS. `references/Isaac_Reaserch.txt` establishes the research gap precisely: the optical and acoustic communities each solve half this problem, and **no published work uses optical satellite detections to generate distance ground truth for acoustic ML training of non-AIS vessels**. That gap is the contribution.

The approach inverts the usual pipeline: filter PlanetScope first (cloud, then vessel detection), then pull only the matching hydrophone windows. This collapses the acoustic download from terabytes to a few MB per scene.

Repo is currently a bare OHW template — a README, an empty `final_notebooks/project_result_1.ipynb`, two zero-byte contributor notebooks, and a 21-byte placeholder `data/data.csv`. Everything below is new code.

**The central design constraint:** on an E&R Basic Planet account the matchup dataset will be roughly **100–400 vessel detections**, mostly multi-vessel. That is far too small to learn ship acoustics from scratch, and the plan is built around not having to.

---

## What the sample data actually is

Decoded from `data/Folger Deep Hydrophone Data Sample/` (device `ICLISTENHF1266`, ONC deviceId 23235, Folger Deep):

| Product | Size / 5 min | Per day | Role |
|---|---|---|---|
| `.wav` 128 kHz, 24-bit, mono | 115 MB | ~33 GB | Format validation + spot analysis only |
| `.flac` | 50 MB | ~14 GB | Not needed |
| `.mp3` | 4.8 MB | ~1.4 GB | Fallback only (lossy above ~16 kHz) |
| **`.fft.gz`** | **0.29 MB** | **~84 MB** | **Bulk input** |

**`.fft.gz` layout — verified, not assumed.** 614,400 newline-delimited integers = **1200 rows × 512 bins**. Zero-runs sit at indices 417–511 of every row (confirmed by zero-position gap analysis), which fixes row length at 512. So:
- 0.25 s per row (1200 rows over 300 s)
- 125 Hz per bin (1024-pt FFT at 128 kHz, bins 0 → Nyquist 64 kHz)
- Real content in bins 1–416 (~52 kHz), rolling off to the integer floor at the ISO 7605 anti-alias limit of 0.4·fs = 51.2 kHz
- Values are quantised dB (observed 0–86), uncalibrated

**Calibration** (`...PreDeploymentCalibration.txt`): 5120 rows, 10 Hz → 51,200 Hz at 10 Hz spacing, dB re Counts²/µPa². Interpolate onto the 125 Hz grid and subtract for absolute dB re 1 µPa²/Hz. **Valid from 18-Feb-2020 onward, this device only** — so restrict to 2020–2026 rather than back to 2014. Earlier Folger deployments are different devices with different sensitivities. Six summers on one calibration removes a whole class of silent dB bugs.

---

## Planet API access — what is actually possible

**Can we extract arbitrary small windows? Technically yes, economically no.** The [Orders API clip tool](https://docs.planet.com/develop/apis/orders/tools/) accepts arbitrary GeoJSON polygons/multipolygons down to **1 m²** (no holes, no overlapping multipolygons, ≤1,500 vertices). You are *not* limited to Planet-determined tiles.

**But E&R Basic bills under "preferred clipping":**

> charge = **max(100 km², clipped area) per intersecting scene**

A 1 km² clip and a 100 km² clip cost the same, which inverts the optimisation:

| Design | Charge/overpass | Overpasses per 3,000 km² month |
|---|---|---|
| Small stratified boxes @ 13.5 km² | 100 km² | 30 — **small boxes buy nothing** |
| Full 10 km disk (314 km²) | 314 km² | ~9 |
| **Disk r = 5.5 km (95 km²)** | **100 km²** | **~30** |

PSScene footprints are ~32.5 × 19.6 km (SuperDove), ~25 × 11.5 km (Dove-Classic); an AOI straddling two scenes is billed twice. E&R PlanetScope also carries a 30-day delay, irrelevant for archive work.

**Order a full disk of radius ~5.5 km (95 km²) centred on Folger Deep.** It sits just under the 100 km² floor, so the area is effectively free — anything smaller is wasted headroom you are billed for anyway. And a *full* disk means every vessel inside 5.5 km is observed; a 10 km sector would leave three-quarters of the acoustic near-field unseen, and an unobserved close vessel corrupts a label far worse than a distant one.

Two optimisations for `planet.py`:
- **Search is free.** Filter to overpasses where a *single* PSScene fully contains the AOI. Straight 2× on effective quota.
- **Tile streaming does not consume download quota** ([FAQ](https://support.planet.com/hc/en-us/articles/16457647004701-Planet-Tile-Streaming-FAQ)). Triage which cloud-free overpasses contain vessels before spending quota — but deliberately order some empty scenes, or selection bias destroys your negatives.

**Realistic yield: 20–30 overpasses per month of quota ≈ 100–400 vessel detections.**

Worth one email on Day 1: ask ONC/UBC whether an institutional **Premium** plan exists. Premium bills actual clipped area with no floor, under which small stratified boxes become optimal and N rises ~20×. Keep the AOI in config so switching is free.

---

## Is 100–400 samples enough? — the honest answer

**Trained from scratch: no.** Continuous range + length regression from raw spectrograms, on a few hundred mixture-contaminated samples, across six years of varying sea state, tide and sound-speed profile, would produce a model that barely beats a two-parameter physics fit and would not generalise. Do not attempt it, and do not promise it.

**But almost none of this needs to be learned from your data.** The acoustics are already published, and the AIS-labeled training corpora already exist. Your 300 matchups are worth far more as *validation and calibration* than as training data. Three tiers:

### Tier 1 — Physics with published priors (needs ~0 training data)

Source level as a function of length, speed and class is a solved, open-access, regionally-validated problem:

- **[MacGillivray & de Jong 2021](https://doi.org/10.3390/jmse9040369)** (JOMOPANS-ECHO) — SL spectrum in decidecade bands vs frequency, speed, length, AIS ship type. Validated against the Vancouver Fraser Port Authority **ECHO programme — BC waters**. Stated uncertainty ±6 dB.
- **[Estimation of Source Levels of Small Vessels](https://www.mdpi.com/2077-1312/14/6/561)** (JMSE 14:561) — 25 vessels, 5 sites, by hull class (planing / semi-displacement / displacement), and **explicitly extends JOMOPANS-ECHO downward to small vessels**. This is exactly the recreational population MacGillivray does not cover.
- **[Lagrois et al. 2023](https://doi.org/10.3390/s23031674)** (Sensors 23:1674) — monopole source levels of small recreational vessels, St. Lawrence Estuary. Canadian, open access.
- **[JASA 156:2077](https://pubs.aip.org/asa/jasa/article-abstract/156/4/2077/3314966/Speed-dependence-sources-and-directivity-of-small)** — speed dependence and directivity of small-vessel noise. Key caveat: planing hulls' SL rises with speed until planing, then flattens, so speed is weakly identifiable in exactly the regime recreational craft operate in.

Transmission loss from **Bellhop via [`arlpy.uwapm`](https://arlpy.readthedocs.io/en/latest/uwapm.html)**, with real bathymetry (GEBCO / Canadian Hydrographic Service) and a sound-speed profile from ONC's **co-located Folger Passage CTD**.

With SL and TL both supplied externally, **range and length are inverted, not learned** — and the matchups become the validation set. Validating a physical model needs tens of samples, not thousands. This tier is near-certain to produce a result.

Honest error budget: ±6 dB SL uncertainty against a ~15–18 dB/decade TL slope maps to roughly a **factor of ~2 in range**. State this up front.

### Tier 2 — Transfer learning (the ML story)

- **Pre-train on [VTUAD](https://dx.doi.org/10.21227/msg0-ag12)** (Domingos et al.) — ONC, Strait of Georgia, icListen hydrophone at 147 m, AIS-labeled, June–Nov 2017. Critically it is built around **distance-based scenarios with inclusion/exclusion zones**, admitting a clip only when a single vessel is in the inclusion zone and no other in the exclusion zone. Same operator, same province, same instrument family as Folger's icListen HF, clean single-source labels, and range structure already built in. This is as close to an in-domain pre-training corpus as exists.
- **Then linear-probe or lightly fine-tune on the PlanetScope matchups.** [arXiv 2601.08358](https://arxiv.org/html/2601.08358) shows linear probing on frozen pretrained audio embeddings works for underwater target recognition; [arXiv 2409.13878](https://arxiv.org/pdf/2409.13878) (MIT Lincoln Lab) covers cross-domain transfer with pretrained models. Frozen PANNs or VGGish embeddings plus a small head is the right recipe at n≈300.
- **Wout et al. 2025** (already in the references) is the closest published analogue: a CNN on 116 days from two North Sea stations classifying **nearest-vessel** distance bins — which independently validates the nearest-vessel target formulation adopted below.

300 samples is ample for fitting a linear head or recalibrating a pretrained model. It is not enough to train a CNN.

### Tier 3 — What the satellite data uniquely buys (the actual contribution)

The novelty is not a better range model — it is **labels for vessels AIS cannot see**. So spend the hard-won matchups on questions only they can answer:

1. **Quantify the AIS blind spot acoustically.** What fraction of acoustically-detectable vessels near Folger carry no AIS? O'Hara et al. 2022 report up to 72% non-AIS in BC from shore-based optical; nobody has measured it against a hydrophone.
2. **Test whether AIS-trained models transfer to non-AIS vessels.** Train on VTUAD (large commercial ships, all AIS), apply to satellite-detected small craft, measure the degradation. This is the gap the reference doc identifies, it is publishable whichever way it comes out, and it needs ~100 samples, not thousands.
3. **Domain-adapt** the AIS-trained model onto the small-vessel population and show the improvement.

Framing the dataset as an *evaluation and calibration* set rather than a training set is what makes 100–400 samples sufficient rather than fatal.

---

## What we reuse instead of building

| Need | Reuse | Saves |
|---|---|---|
| Source level vs length/speed | JOMOPANS-ECHO + small-vessel extensions (above) | The entire SL fit |
| Transmission loss | `arlpy` + Bellhop; GEBCO bathymetry; ONC Folger CTD for SSP | Hand-rolled TL |
| Calibrated band levels | **PyPAM** (ISO 18405 PSD, hybrid millidecade bands), [`mbari-org/pbp`](https://github.com/mbari-org/pbp) | Standards-compliant feature code |
| Acoustic pre-training corpus | **VTUAD** (ONC, AIS+range labels) | Months of labelling |
| Audio embeddings | PANNs / VGGish, frozen | Training a CNN we cannot afford |
| **Vessel detection in imagery** | **Pretrained YOLO from Mäyrä et al. 2025** — [GitHub](https://github.com/mayrajeo/ship-detection), [HF weights](https://huggingface.co/mayrajeo/marine-vessel-yolo), also `torchgeo/yolo11s_marine_vessel_detection`. 8,768 recreational-vessel annotations, P 0.863 / R 0.841 / mAP 0.890 | A day of detector work |
| AIS for the region | ONC AIS receiver data (same API, co-located) | Third-party AIS sourcing |

Mäyrä's weights are trained on **Sentinel-2 at 10 m**, not PlanetScope at 3 m. Handle the domain shift by **downsampling PlanetScope to 10 m for detection, then returning to native 3 m for length measurement** — pretrained detection plus full-resolution morphometry. Keep a classical NIR top-hat detector as the fallback if transfer disappoints; water is near-black in NIR and hulls plus wakes are bright, so it is ~100 lines with `cv2` (installed).

---

## Label quality — what regression can and cannot deliver

Range supports continuous regression; length does not.

- **Range labels are excellent.** PlanetScope geolocation is <10 m RMSE against ranges of 500–5,500 m — 0.2–2% label error. Continuous range regression is well-posed.
- **Length labels are floor-limited.** At 3 m GSD a hull carries ~±1 px at each end plus wake contamination, so σ ≈ 3–5 m. On a 0–12 m vessel that is 25–40%. **Report continuous length RMSE *and* the FAO 0–12 / 12–24 / >24 m confusion matrix** — the band metric is what the label noise supports, and it is what the source-of-truth asked for.

**The multi-vessel mixture is the central methodological problem** — and the place to make it the method rather than a filter. At 95 km² over summer Barkley Sound, N > 1 is the common case, so discarding mixtures would throw away most of an already small dataset. Instead invert a forward model over all vessels at once:

```
RL_total(f) = 10·log10( Σᵢ 10^((SL(Lᵢ, vᵢ, f) − TL(Rᵢ, f)) / 10) ) ⊕ ambient(f)
```

with `SL` from JOMOPANS-ECHO and `TL` from Bellhop, leaving only a small number of local correction terms to fit. Every vessel in every scene contributes a constraint.

Then stratify for the supervised model:
- **Acoustically dominated** (`R₂ > 2·R₁`) → nearest vessel dominates, so a clean single-source target even with several boats present. **This is the MVP subset**, and far larger than `n_vessels == 1` would be.
- **Genuine mixtures** → nearest-vessel range plus count (matching Wout et al.).
- `n_vessels == 0` → negatives for detection and ambient baselining.

**Confound to state in the write-up:** vessels beyond 5.5 km are outside the AOI but still audible. Use ONC AIS to account for them — out-of-AOI traffic skews large and AIS-carrying, precisely what AIS covers well. Satellite handles small dark vessels in the near field; AIS handles large vessels in the far field. This makes AIS a required input, not an optional extra.

---

## Deliverables, in priority order

1. Validated `.fft.gz` reader + calibrator (nothing works without it)
2. `matchups.parquet` — the cross-modal dataset, itself the novel contribution
3. **Acoustic measurement of the AIS blind spot** at Folger Passage
4. Physics inversion for range + length, validated on matchups, with an honest error budget
5. **VTUAD-trained model tested on non-AIS satellite-detected vessels** — the gap-filling result
6. Acoustic detection-range curve: P(detect) vs range vs vessel size (source-of-truth goal 3)

---

## Day-by-day

### Day 1 — De-risk and gate

**Blockers first, in the first hour.** ONC API token (the API returns 401 without one, confirmed). Planet API key, **remaining quota, and whether the account bills Preferred or Premium**. Start the VTUAD download (IEEE DataPort) — it is large and should run overnight.

- `boatphone/fft_io.py` — reader for the 1200×512 layout, plus ONC download via the installed `onc` 2.6.0 client. Discover Folger location codes at runtime with `onc.getLocations()` rather than hardcoding.
- `boatphone/calibrate.py` — interpolate the 10 Hz sensitivities onto the 125 Hz grid, subtract, return dB re 1 µPa²/Hz.
- **Validation gate:** recompute a spectrogram from the supplied `.wav` (`scipy.signal.spectrogram`, 1024-pt, 128 kHz) and correlate against the decoded `.fft.gz` over the overlapping window. The two FFT files are 5 min apart (`235504`, `000004`) and the WAV starts `000000.029`, so a clean overlap exists. Require r > 0.95.
- `boatphone/planet.py` — Data API search over the 95 km² disk, 2020–2026, May–Sep, cloud filter, single-scene-containment filter, quota projection. **Search only, no orders yet.**
- **Go/no-go:** if confirmed quota buys fewer than ~20 overpasses, add Sentinel-2 via `earth-search.aws.element84.com` (reachable, tested) as a parallel coarser track — and note that Mäyrä's weights are *native* to Sentinel-2, so that track needs no domain shift at all. It loses the small end of the FAO 0–12 m class, so treat it as a supplement, never a replacement.

### Day 2 — Three tracks in parallel

**Track A — optical.** Stand up Mäyrä's pretrained YOLO; validate on ~100 hand-labelled vessels from a pilot scene; calibrate pixel-length → metres at native 3 m.

**Track B — acoustic features.** `boatphone/features.py` on top of PyPAM where possible: decade and 1/3-octave band levels, hybrid millidecade spectra, tonal structure, DEMON envelope. The references make **40–60 Hz** the strongest band for shallow-water ranging — weight feature design accordingly.

**Track C — priors.** Implement JOMOPANS-ECHO `SL(L, v, class)` plus the small-vessel extension; stand up Bellhop through `arlpy` with GEBCO bathymetry and an ONC Folger CTD profile. Sanity-check modelled TL against the ONC ambient record.

### Day 3 — The matchup dataset

- `boatphone/geo.py` — range and bearing from the hydrophone (`pyproj`, installed).
- `boatphone/matchup.py` — join each detection to the ±15 min FFT window around the scene's `acquired` timestamp (~1.7 MB/scene). Emit `data/processed/matchups.parquet` with the dominance stratification.
- Pull ONC AIS for the same windows; flag every detection as AIS-matched or dark. **This produces deliverable 3.**
- **The strongest validation in this plan:** for vessels that *do* carry AIS, compare satellite-derived range against AIS-derived range. Agreement to tens of metres validates the whole labelling method using the cases where both truths exist — and the residual population is exactly the dark vessels.
- **Checkpoint:** count dominated matchups. That number decides Day 4.

### Day 4 — Models, in tier order

- **Tier 1, physics inversion.** Fit the small set of local correction terms in the forward model; invert for range and length. Report the error budget honestly. This is the safety net and will produce a result regardless of N.
- **Tier 2, transfer.** Frozen embeddings (VTUAD-trained or PANNs) + `sklearn.ensemble.HistGradientBoostingRegressor` / linear head on the dominated subset. **Split by time, never randomly** — adjacent windows leak badly; hold out whole months or a year.
- **Tier 3, the gap result.** Apply the VTUAD/AIS-trained model to dark vessels and quantify the degradation.
- Do **not** train a CNN from scratch at this N.

### Day 5 — Results and write-up

Detection-range curve, dark-vessel fraction, `final_notebooks/`, README, presentation.

---

## Files

```
boatphone/                 # importable package, not notebook-only code
  fft_io.py                # 1200×512 .fft.gz reader + ONC download
  calibrate.py             # 10 Hz sensitivities → 125 Hz grid → dB re 1 µPa²/Hz
  features.py              # PyPAM band levels, millidecade, DEMON, tonals
  priors.py                # JOMOPANS-ECHO SL model + small-vessel extension
  propagation.py           # arlpy/Bellhop TL, GEBCO bathymetry, ONC CTD SSP
  planet.py                # search / quota accounting / clip order / download
  detect.py                # pretrained YOLO + NIR top-hat fallback
  geo.py                   # range + bearing from hydrophone
  matchup.py               # detections ↔ acoustic windows ↔ AIS → parquet
  models.py                # forward inversion, embeddings + head, transfer eval
notebooks/01..06           # one per stage, thin wrappers over the package
data/processed/matchups.parquet
```

`.gitignore` already excludes the sample WAV; extend to `data/raw/` and `data/interim/`.

**Install:** `rasterio`, `pystac-client`, `planet`, `arlpy`, `ultralytics`, `pypam` (or `mbari-pbp`), `huggingface_hub`. Already present: `onc` 2.6.0, `cv2` 5.0.0, `torch` 2.12, `sklearn` 1.9, `geopandas`, `shapely`, `pyproj`, `pyarrow`, `scipy`, `xarray`.

---

## Explicitly out of scope

Speed from acoustic Doppler (goal 4) — and note the JASA finding that planing hulls have weak speed–SL dependence, so this is harder than it looks. Also: a second observatory site; training any detector or CNN from scratch; blind source separation; pre-2020 deployments.

**One stretch worth flagging as genuinely novel:** PlanetScope bands are acquired from different focal-plane strips with sub-second offsets, so moving vessels show band-to-band displacement — the effect used for ship velocity in Sentinel-2 ([Heiselberg 2019](https://www.mdpi.com/1424-8220/19/13/2873), ~2.6 s offsets). If the Dove per-band offset can be pinned down, that yields **vessel speed from a single scene**, giving goal 4 an optical ground truth. Verify the offset magnitude before committing time.

---

## Verification

| What | How | Pass criterion |
|---|---|---|
| FFT reader | Recompute spectrogram from supplied `.wav`, correlate vs decoded `.fft.gz` | r > 0.95 |
| Calibration | Quiet-period levels vs Wenz ambient curves for the sea state | Within a few dB |
| SL/TL priors | Modelled RL vs measured RL for AIS ships of known length and speed | Bias < 6 dB, no range trend |
| Vessel detection | ~100 hand-labelled vessels | P/R reported; length bias measured |
| **Satellite range labels** | **Satellite-derived vs AIS-derived range, AIS-visible subset** | **Agreement to tens of metres** |
| Physics inversion | Residuals vs range, length, frequency | No systematic structure |
| Range model | Time-based held-out split | Beats the physics inversion; both reported |
| Length model | Same split | Continuous RMSE **and** FAO 3-class confusion matrix |
| Reproducibility | Fresh clone, run `notebooks/01`→`06` | End to end from two API tokens |

---

## Open items for the team

- **Planet quota remaining, and Preferred vs Premium billing.** Preferred means a 100 km² floor and ~30 overpasses; Premium bills actual area and would raise N ~20×. Worth more than any other Day 1 task.
- VTUAD licence and download size; start it Day 1.
- Folger Deep hydrophone uptime across 2020–2026 summers — ONC deployments have gaps.
- Whether ONC's Folger Passage CTD has usable SSP coverage for the target periods.
- Whether to add the Folger Pinnacle hydrophone (23 m) as a second receiver. Two receivers resolve the single-hydrophone ambiguity the references dwell on (Byun et al. 2026), but roughly double the wrangling. **Recommend building for Folger Deep alone with the hydrophone as a config parameter, adding Pinnacle only if Day 3 lands early.**
