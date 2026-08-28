# Hydrophone visualization & processing methods, evaluated against the ONC `.fft` product

**Audience:** whoever owns the B5 band-level detector and the `.fft` processing.
**Status:** supplementary suggestions. Nothing here is a decision; nothing here has been merged.
**Written:** 2026-08-27, against commit `827e63e`.

> **Correction, 2026-08-27, after this document's first draft.** An earlier version led with a
> warning that B5's fixed +10 dB threshold might not transfer across seasons, on the basis of an
> 8.3-8.5 count seasonal spread in the 1-10 kHz band. **That has now been tested at scale and it
> does not hold.** Over 1,790 windows (2/date, all 895 dates) the seasonal spread of the band
> median is **2.0 counts** and of the ambient P10 baseline **2.5 counts** — comfortably below the
> threshold. The 8.3-8.5 figure was sampling noise off 8-30 files/season. §1.4 now carries the
> large-sample numbers, and the threshold-transfer concern is **withdrawn**.

Every measurement below is mine, this session, against `data/raw/onc/overpass_window_corpus/` and
`data/Folger Deep Hydrophone Data Sample/`, with its sample size stated.

This is a survey of what passive acoustic monitoring (PAM) normally does, filtered hard through
what *this* product can actually support. The anti-recommendations in §5 matter as much as the
recommendations: several standard PAM methods assume calibrated data, sub-250 Hz resolution, or
censored levels, and on this product they return plausible wrong numbers rather than errors.

---

## §0 · The three constraints that decide everything below

**(a) The level scale is an unknown, possibly non-linear, monotone transform of power.**
ONC's product owner states the `.fft` product is a proprietary Ocean Sonics format, "essentially
uncalibrated unless you have all the metadata from Ocean Sonics, which I don't believe ONC
currently publishes", with undocumented filtering applied
(`references/ONC_communication.txt`). We do not know counts-per-dB, and we do not know that the
mapping is even affine.

> **Consequence:** only *rank*, *shape*, and *difference* statistics are safe. Anything that sums
> power — a band level, a decidecade level, an SEL — is unavailable until the scale is pinned.
> §3.1 is the one route to pinning it.

**(b) 250 Hz bins and 0.25 s frames.** Nothing below 250 Hz exists (bin 0 is DC). The frame series
is 4 Hz, so its Nyquist is **2 Hz** — no modulation analysis above that.

**(c) The corpus is a 2.5 h/day strip, not a continuous record.** Measured UTC hour histogram over
all 26,666 windows: `{16: 8030, 17: 10699, 18: 8027}` and nothing else. The project's "continuous
vessel-presence estimate" is, on this corpus, honestly a **daily 150-minute** estimate.

**Environment.** Present: `numpy`, `scipy`, `xarray`, `pandas`, `matplotlib`, `netCDF4`, `dask`,
`zarr` 3.3, `h5py`, `cmocean`, `seaborn`, `statsmodels`, `joblib`, `torch`, `sklearn`, `cv2`.
Absent: `librosa`, `soundfile`, `obspy`, `rasterio`, `pytest`, `lifelines`.

---

## §1 · The data surface, in numbers

### 1.1 Corpus inventory — and a double-count

| | |
|---|---|
| `.fft.gz` files | **26,666** |
| plain `.fft` files | **90** |
| `.sha256` sidecars | 26,756 |
| **unique acoustic windows** | **26,666** |

All 90 plain-`.fft` stems are *also* present as `.gz`, and **10 of 10 tested pairs have
byte-identical decompressed payloads**. They are duplicates, on three dates: 2025-07-15,
2025-07-16, 2025-08-12.

> **Action:** `overpasses.corpus_file_index()` walks the directory and includes both containers, so
> it double-counts 90 windows. Key the index on `(device, start_utc)`, prefer `.fft.gz`, and log
> the drops. Until then, any percentile, histogram, or LTSA built on that index is 0.34%
> duplicated, concentrated on three dates.

26,666 windows x 300 s = **2,222 hours = 92.6 days** of acoustic time across 895 in-season dates.

### 1.2 The canonical bin map — publish once, import everywhere

Bin *k* is *k* x 250 Hz under the centre convention, which remains **open** (decision 0013);
carry the +/-125 Hz on both edges of every band.

| Purpose | Hz | Bins | n | Sample median level |
|---|---|---|---|---|
| "ship proxy" (nearest reachable to ONC's ~100 Hz) | 250-1,000 | 1-4 | 4 | 9-23 |
| **small craft (primary)** | 1,000-10,000 | 4-40 | 37 | 36-45 |
| mid | 10,000-30,000 | 40-120 | 81 | 20-24 |
| rain signature | 13,000-25,000 | 52-100 | 49 | — |
| echosounder hump | 35,000-40,500 | 140-162 | 23 | +27-29 over local background |
| calibration-file ceiling | <= 51,000 | <= 204 | — | — |
| **instrument control band** | 51,250-102,000 | 205-408 | 204 | **~5, every year** |
| anti-alias skirt | 102,000-106,000 | 409-424 | 16 | ~0 |
| nominal dead band | > 106,000 | 425-511 | 87 | ~0, but see 1.3 |

### 1.3 Six repo-held constants that do not hold on the corpus

Each was tuned on the two 2026 sample fixtures. Each fails on the corpus. **This is the section
most worth acting on** — these constants are load-bearing for checks and for decision records.

| Claim | Where | Measured on the corpus |
|---|---|---|
| "18.7% of cells sit at the floor" = left-censoring | `config.py` ~709, `fft_io.censoring_report` docstring, decision 0015 | 19.9% over all 512 bins — but **bins 409-511 alone are 95.8% zero**. In-band (1-408) the floor fraction is **0.00-0.55%**; in 1-10 kHz it is ~0%. *(16 files + 6/yr)* |
| bin 0 is "structurally near-zero"; `FFT_DC_COL_MAX_LEVEL = 5`, `MAX_NONZERO_FRACTION = 0.02` | decision 0014, `config.py:457-458` | **max 60; mean nonzero fraction 0.232; 20 of 25 random files violate the bounds** |
| bins 425-511 are "hard exact zero" | decision 0014, `config.py:446` | **15 of 25 random files carry nonzero cells there** (1-35 cells/file) |
| the ~37.65 kHz hump is persistent, "present even in quiet windows" | decision 0014 / plan notes | **A co-located instrument with a duty cycle, not a soundscape feature.** OFF continuously **2021-05-01 -> 2022-07-17**, ON from **2022-07-18**; exactly one transition in 2022, and the switch-off falls in the winter service gap. Short outages of 1-5 dates in 2023-25. *(1,790 windows, all 895 dates)* |
| `FFT_LEVEL_CEILING = 86` | `config.py:594` | Falsified by decision 0026 but unchanged, so `censoring_report`'s `n_at_ceiling` is a meaningless count |
| "Folger levels are calibrated dB re 1 uPa" | `models.py:22-23`, `config.py:170-171` | Stale VTUAD-era framing. True of the FLAC/WAV; **false of the `.fft` product** per ONC |

> The censoring one is the most consequential. **The headline "18.7% censored" is the anti-alias
> dead band, not censoring.** In the analysis band there is effectively none. Everything built on
> a censoring premise — decision 0015's threshold machinery, the median-over-mean rationale in
> `config.py`, any plan to use Kaplan-Meier or Tobit — rests on a measurement of the wrong cells.
> (Median-over-mean is still right, for the different reason given in §5.2.)

### 1.4 A cross-year regime shift that threatens every threshold

Per-year median level, 8 files/year, in four bands:

Median level per season, 8 files/year:

| Year | 250 Hz-1 k | 1-10 k | 10-30 k | 51-102 k (control) | hump excess* |
|---|---|---|---|---|---|
| 2020 | 14.5 | 37.0 | 21.0 | 5.0 | 16.6 |
| 2021 | 16.0 | 37.2 | 21.8 | 5.0 | **-0.8** |
| 2022 | 9.0 | 36.2 | 20.0 | 5.0 | **4.3** |
| 2023 | 23.5 | 44.5 | 23.5 | 5.0 | 17.6 |
| 2024 | 19.5 | 42.5 | 21.8 | 5.5 | 16.0 |
| 2025 | 13.0 | 38.5 | 20.5 | 5.0 | 15.8 |

\* mean over bins 140-162 minus the 0014 background bins, median across 30 windows/season.

The primary 1-10 kHz band swings **8.3 counts** across seasons; the ship-proxy band swings **14.5**.
Recomputed at 30 files/season in the companion notebook, the 1-10 kHz seasonal spread is **8.5**.
That is the same order as a vessel-pass excess. Meanwhile the 51-102 kHz control band is pinned at
~5 in every year, and `data/derived/deployments.csv` shows **one single deployment**
(`FGPD:2020-03-08 -> 2026-03-14`) spans the entire corpus — so a deployment boundary does not
explain it.

**TESTED AT SCALE — the apparent swing was sampling noise.** Over **1,790 windows** (2 per date,
all 895 dates), the same statistics are nearly flat:

| Year | n | 1-10 kHz P10 | 1-10 kHz median | 1-10 kHz P90 | 250 Hz-1 k median |
|---|---|---|---|---|---|
| 2020 | 302 | 32.0 | 42.0 | 56.0 | 19.0 |
| 2021 | 306 | 31.0 | 42.0 | 54.5 | 19.0 |
| 2022 | 298 | 31.0 | 42.0 | 56.6 | 19.0 |
| 2023 | 272 | 31.0 | 42.0 | 55.0 | 19.0 |
| 2024 | 306 | 34.0 | 44.0 | 54.0 | 20.0 |
| 2025 | 306 | 31.0 | 42.5 | 55.0 | 18.0 |

**Seasonal spread: 2.0 counts (median), 3.0 (P10), 2.5 (ambient baseline).** Not 8.5.

> **The threshold-transfer concern is withdrawn.** A fixed +10 dB excess is safe against a 2-2.5
> count baseline drift. §3.3 (whitening) and §3.4 (the control band) remain worth doing on their
> own merits — they cost little and they make drift *visible* — but they are no longer urgent.
>
> **The methodological lesson is the durable part.** A per-season median computed from 8 files, off
> a distribution with P10 = 31 and P90 = 56, carries several counts of sampling error. The
> difference between seasons looked like a signal and was noise. Any seasonal or interannual claim
> on this corpus needs n in the hundreds per season — it is affordable (~37 core-minutes for the
> whole corpus), so there is no reason to run it on eight files.

### 1.5 What has already been settled, and what this document adds

Commit `329b7cd` (B5 viability gate, verdict GO) and `827e63e` (per-overpass review set) landed
while this was being written. Credit where it is due, and no duplicated effort:

**Already done — this document does not re-litigate any of it.**
- In-band floor censoring measured at **0.00%**, and the median reduction correctly justified from
  it. That independently confirms §1.3's censoring correction; the two of us got there separately.
- A frame-shuffle null with 43 events against 0 — a real null, correctly applied.
- Synthetic 50 dB injection recovered at 50.0 dB, correct time and duration, invisible out of band.
- Per-overpass figures over 18 windows (13 full, 5 partial) — the same 13/5/12 split derived
  independently here. **The companion notebook's Figure 9 duplicates that work**; it is kept only
  because it carries a different null (§7) and because the notebook should stand alone.

**What is genuinely still open, in priority order.**
1. **The fixed +10 dB threshold vs the 8.5-count seasonal spread** (§1.4). The one item that could
   change a result.
2. **The 18.7% censoring figure is still asserted in five places** — `fft_io.py:52`, `fft_io.py:523`,
   `config.py:569`, `config.py:709`, and decision 0015's reasoning — even though 0.00% has now been
   measured in-band. The analysis is right; the documentation and the decision record still carry
   the wrong premise, and decision 0015's threshold machinery is built on it.
3. **`corpus_file_index` still includes both containers** and so double-counts 90 windows (§1.1).
   Its docstring frames the 90 as a manifest-vs-disk disagreement; they are duplicate *windows*.
4. **Decision 0014's bin-0 and dead-band bounds** still fail on most corpus files (§1.3).
5. **`models.py:22-23` and `config.py:170-171`** still assert Folger levels are calibrated dB re
   1 µPa (§1.3).

*(One earlier finding is now resolved: `features.band_level_series` reduced with `np.mean` while
labelling the result `median_of_product_db`. It reduces with `np.median` as of `329b7cd`. The
companion notebook's integration cell checks this every run.)*

---

## §2 · Tier 0 — build these once; everything else reads them

Measured decode cost: **0.084 s/file**, **1.23 MB** per file as `int16(1200, 512)`. A full-corpus
pass is therefore **~37 core-minutes**. (Quoted in core-minutes deliberately: this project runs
across several JupyterHub instances with different CPU and RAM, so wall-clock and memory caps are
not project facts. Write every pass to stream.)

**T0.1 · Deduped corpus index · DO.** Per §1.1.
*Null check:* assert `len(index) == 26_666`, and exactly 30 windows on each of the three affected
dates.

**T0.2 · Exact per-bin integer histogram cache · DO.**
Levels are small integers, so the histogram is **exact** — no bin-width choice, no quantile
interpolation, no approximation anywhere downstream. `uint32[409 bins x 128 levels]` = 209 KB per
accumulation unit; accumulate per (year, month) = 30 units = 6.3 MB. This single artefact feeds
every percentile spectrum and every SPD plot in §4 for free, forever.
*Null check:* histogram total must equal `n_files x 1200 x 409` exactly; recompute one month's
percentiles from raw files and compare.

**T0.3 · Per-file reduced spectra · DO.**
Per file, over its 1200 frames, per bin: median, L05, L50, L95, max.
`float32[26666 x 5 x 512]` ~ **273 MB**. This is the LTSA backbone.
Do **not** cache full resolution: `int16` over bins 0-408 is ~26 GB.

**T0.4 · Per-frame band levels for the 4 configured bands · DO.**
`float32[26666 x 1200]` ~ **128 MB per band**. Feeds event segmentation and the stamp panels.

Write all four to one `zarr` store under `data/derived/` with a provenance sidecar, matching the
existing `pull_overpass_corpus.provenance.json` pattern (invariant 2).

---

## §3 · Tier 1 — the highest-leverage methods

### 3.1 · Matched WAV<->product empirical response function · **DO — highest leverage in this document**

**A matched pair already exists on disk and appears not to have been noticed.**

| | Coverage (UTC) |
|---|---|
| `ICLISTENHF1266_20260313T000000.029Z_...T000500.029Z.wav` | 00:00:00.029 -> 00:05:00.029 |
| `ICLISTENHF1266_20260313T000004.000Z.fft.gz` | 00:00:04.000 -> 00:05:04.000 |
| **overlap** | **296.029 s of the same acoustic scene** |

And the analysis parameters fall out exactly: **a 512-point FFT at 128 kHz gives exactly 250.0 Hz
bins**, so WAV bins 0-255 map **1:1** onto product bins 0-255 (0-63.75 kHz). A 0.25 s product
frame is **62** non-overlapping 512-point FFTs (32,000 samples / 512 = 62.5).

What the regression of product counts against WAV dB, per bin, buys you:

1. **The counts->dB scale** — the missing quantity from §0(a). This is what unblocks band levels,
   decidecade bands, and every energy-summing statistic.
2. **The shape of the low-frequency deficit**, and critically **whether it is a fixed invertible
   filter or level-dependent** (i.e. AGC/compression). If the slope varies with level, no fixed
   transfer function exists and that is itself a first-class finding.
3. **An independent read on centre-vs-edge** (decision 0013) via the cross-spectrum of the two
   time-frequency surfaces — currently the only open route, since ONC has confirmed they have no
   documentation to settle it.
4. **A check on the 4.971 s frame offset** between the two files.

The WAV is also the *calibratable* one: ONC states the sensitivity file applies to the audio, and
that curve is flat within a few dB over 250 Hz-51.2 kHz. So this is the only path to
approximately-calibrated product levels.

> **Implementation gotcha:** the WAV is **24-bit**. `scipy.io.wavfile` cannot read it and there is
> no `soundfile` or `librosa` in this environment. Use the stdlib `wave` module and unpack 3-byte
> frames to `int32` by hand. (Confirmed: `wave` opens it; `sampwidth = 3`.)

**Measured on that one window** (companion notebook, Figure 8): a cross-bin regression of product
counts on WAV dB gives a slope of **0.52 counts/dB** and **r = 0.744 (r² = 0.55)**.

That r² is the finding. **The product spectrum is not a simple rescaling of the WAV spectrum over
the same 296 s** — which is direct evidence for ONC's "there is sometimes filtering applied", and
it means no single scalar counts/dB will do. Note this cuts against reading the B5 gate's
withdrawal of B1's "slope ≈ 1" criterion as *"the comparison cannot be run"*: it can be run, it
just does not return a clean slope, and the shape of the residual is informative.

*Note what this is not:* one window, and a **cross-bin** regression conflates spectral shape with
the counts→dB scale. A usable transfer function needs many matched windows spanning a range of
levels, fitted **per bin**. That is the work this figure argues for, not work it completes.
*Fails when:* the slope varies with level (=> AGC => no fixed transfer exists).
*Null check:* fit on the first half of the window, predict the second half.

### 3.2 · Rank / monotone-invariant statistics as the house default · **DO**

This is the correct general answer to "the scale is unknown", and it is not currently the project's
default. Under an unknown monotone transform, **only order statistics are invariant.** So:

- rank-based excess (a frame's percentile within a matched reference distribution) instead of a
  count difference;
- Mann-Whitney / AUC for "is this window louder than that one";
- Spearman rather than Pearson for any optical<->acoustic association.

*Fails when:* someone reports a "dB difference" derived from ranks.
*Null check — cheap and decisive:* apply an arbitrary monotone warp (`x -> x**1.3`) to the levels
and confirm every reported statistic is **bit-identical**. Any statistic that moves was depending
on the scale you do not have.

### 3.3 · Per-bin rolling-quantile whitening · **DO, before any threshold**

Per bin, estimate a floor as a rolling low quantile over N days *at the same minute-of-window*,
and score excess against that rather than against a global constant. An unknown and drifting
frequency response then largely cancels in the difference, which is exactly the failure mode ONC
warned about, and exactly what §1.4 measures.

Candidate windows: 7 / 21 / 61 days. Within-window baselines can keep using
`features.ambient_baseline_product_db` (10th percentile).

*Fails when:* the rolling window is shorter than event duration — the detector subtracts the vessel.
*Null check:* inject a synthetic broadband bump of known amplitude and duration into a real window
and confirm recovery as a function of window length. The repo already has the idiom
(`fft_io.assert_tone_at`, `check_b0_2a_synthetic_tone_frame_shuffle_null_is_rejected`).

### 3.4 · The 51-102 kHz control band as a free instrument reference · **DO, as a MEDIAN only**

Bins 205-408 sit at ~5 counts in every year (§1.4) and no small vessel will dominate them, so a
detector-band-minus-control-band contrast rejects instrument drift for free. Any cross-year figure
that moves *together with* the control band is showing you the instrument, not the ocean.

> **Caveat, measured.** That band contains the strongest narrowband feature in the whole spectrum
> — a **continuous line at 81.25 kHz, +17 counts** — plus lines at 78.5, 91.5 and 100 kHz (§3.4a).
> Its *median* over 204 bins is robust to them, which is why it reads a flat ~5 every season. A
> **mean or energy sum over the same band would be dominated by self-noise.** Use the median.

### 3.4a · The 20.3 kHz harmonic family is system self-noise — and the 81.25 kHz line is the gain reference · **DO**

Narrowband lines survive the six-season median at 20.25, 61.0, 78.5, **81.25**, 91.5 and 100 kHz.
They are **not** sound in the water, and the discriminator is duty cycle:

| Line | duty (fraction of frames present) | reading |
|---|---|---|
| 81.25 kHz | **0.87-0.98, every season** | continuous |
| 78.5 kHz | 0.64-0.98 | continuous |
| 100 kHz | 0.67-1.00 | continuous |
| 20.25 kHz | 0.55-0.89 | mostly continuous |
| 61 kHz | 0.17-0.34 | intermittent — the one plausible transmitter |
| 37.5 kHz (echosounder) | 0.44-0.51 on / 0.13 off | pulsed, and absent 2021 to mid-2022 |

A fundamental at **20.3125 kHz** predicts bins 81, 244, 325 and the lines land on those bins
exactly (20.31 / 60.94 / 81.25 kHz); the n=2 harmonic at 40.6 kHz is masked by the echosounder
hump. A non-round fundamental with exact integer harmonics, running continuously through six
seasons and indifferent to the echosounder duty cycle, is a **clock or switching-supply artefact in
the acquisition chain** — not a source in the ocean.

**Use it.** A continuously present, internally generated line is a *better* gain reference than the
echosounder, which turns out to be intermittent (§1.3): it tracks the electronics gain directly and
it never switches off. Track the 81.25 kHz line level per window as a first-class series; a step in
it is a chain change, and it is the cleanest instrument-drift monitor this product offers.

*Fails when:* the line is read as an acoustic detection, or its band edge is placed inside another
line — run §4's per-bin MAD map before choosing any band edge near it.
*Null check:* the line's level must be uncorrelated with the vessel-band event rate. If it tracks
traffic, it is not what this section claims it is.

**What these lines are NOT, and why the reasoning generalises.** Vessel radiated noise is broadband
and produces no stable narrow line at 80 kHz. Harbour porpoise NBHF clicks peak near 130 kHz —
above this product's 102 kHz ceiling — and delphinid clicks are broadband transients that a median
over 1,200 frames erases. Active instruments (echosounders, ADCPs, acoustic modems, fish tags)
**pulse**; anything at duty ~1.0 has excluded itself. The 61 kHz line is the only one whose duty
cycle leaves it a candidate for a real transmitter, and it matches no obvious standard (Innovasea
tags are 69 kHz; AZFP's second channel is 67 or 125 kHz) — record it as unidentified, not guessed.

### 3.5 · Hysteresis segmentation, and ATL vs fixed-percentile as a *pair* · **DO**

Single thresholds give a per-frame boolean. **Events** — count, duration, CPA time — are what a
vessel-count estimator actually needs, so segment with a dual threshold (enter high, exit low,
minimum duration).

Numbers here: 0.25 s frames => 30 s = 120 frames; a 6 m/s craft transiting a ~500 m detection
radius is ~160 s ~ 640 frames.

Report **two** detectors, because they fail in opposite directions:
- **Adaptive Threshold Level** (fixed offset above the running quantile of §3.3) is blind to a slow
  ambient rise that *is* vessel traffic;
- **fixed-percentile** (top q% of a matched reference) forces a constant detection rate by
  construction — which is itself the diagnostic that it is not really a detector.

*Fails when:* median event duration ~ the minimum-duration constant — you are measuring the constant.
*Null check:* the event-count-vs-threshold curve must be smooth. A knee at the constant is the tell.
Also require stability under a +/-1 bin shift of the band edges (the open decision-0013 question).

### 3.6 · Echosounder ping cadence as a per-file time-base verifier · **DO**

**Measured.** Autocorrelation of the mean level across bins 140-162, per frame:

| File | ACF(1) | ACF(2) | ACF(3) | **ACF(4)** | ACF(5) | ACF(8) |
|---|---|---|---|---|---|---|
| 2020-05-01T16:17:03 | -0.16 | -0.67 | -0.16 | **+1.00** | -0.16 | +0.99 |
| 2023-06-19T18:05:23 | -0.09 | -0.79 | -0.09 | **+1.00** | -0.09 | +1.00 |
| 2025-09-08T18:15:05 | -0.16 | -0.63 | -0.16 | **+0.96** | -0.16 | +0.96 |
| 2021-05-17T16:53:08 *(ping absent)* | +0.45 | +0.35 | +0.31 | +0.33 | +0.32 | +0.32 |

A period of exactly 4 frames = a **1.000 Hz ping on a 4.000 Hz frame grid**. This **independently
confirms the 0.25 s frame duration** — on a product whose only time source is its own filename.
Further, ping phase is predictable across consecutive files, so a **phase jump at a file seam is a
filename-timestamp error**, which is otherwise undetectable.

Degrade honestly: in 2021-22 the ping is absent, so the answer is "not checkable", never "checked".

> **Corollary hazard:** the ping occupies ~1 frame in 4 within bins 140-162. Any statistic spanning
> those bins is 25%-duty-cycle contaminated. **Mask by phase; do not discard the band.**
> Reassuringly, the ping does *not* leak into the primary band: detrended ACF(4) over bins 4-40
> measures -0.06 to +0.06 across four files.

### 3.7 · File-seam continuity check · **DO — it gates §3.3**

30 files/day tile 150 minutes, but they start off-grid and the start time drifts by year
(`161703` in 2020, `161505` in 2025). Plot the level discontinuity across each 5-minute seam.

*Fails when:* seam jumps exceed within-file variance — which would mean the product renormalizes
per file (AGC-like), and **every cross-file baseline in §3.3 would be invalid.** Cheap to run, and
it must run before whitening is trusted.

---

## §4 · Visualization that earns its keep

**V1 · Long-term spectral average (LTSA) · DO.** x = date (895 columns), y = bins 1-408,
value = daily median of per-file medians, straight from T0.3. Plus a within-day drilldown
(30 columns x 408). Label the x-axis **"16:15-18:45 UTC only"** so nobody reads it as continuous.
*Null check:* shuffle dates within a month — seasonal banding must vanish; per-bin instrument lines
must not.

**V2 · Percentile exceedance spectra (L01/L05/L50/L95/L99) · DO.** Exact from T0.2.
**State the convention explicitly:** L95 = the level exceeded 95% of the time = the *5th*
percentile. This inversion is silent and common.
*Fails when:* L95 tracks L50 with no separation — that means a homogeneous window population, i.e.
**no events, not a broken method** (invariant 9).
*Null check:* L95 of the control band must sit within a few counts of the floor in every year.

**V3 · Spectral probability density (SPD) · DO — and it is *better* here than usual.**
Merchant et al. 2013's standard first look: a 2-D density of frequency x level, with percentile
spectra overlaid. Integer levels make the density **exact** at 1-count resolution. It is a *shape*
diagnostic, so it is immune to an unknown constant offset — an unusually good fit for this product.
*Never label the y-axis "SPL" or "dB re 1 uPa".*
*Fails when:* apparent bimodality in bins 140-162 is really the echosounder duty cycle.
*Null check:* compute with ping frames masked and unmasked; the bimodality must be attributable.

**V4 · Data-quality tracks · DO, high priority.** Five stacked tracks over the 895 dates:
files present vs 30 expected; in-band floor fraction; nonzero cell count in 425-511; max in-band
level; echosounder-present flag. **This is where §1.3 becomes visible instead of inherited.**
*Fails when:* a track is flat at exactly an assertion bound — you are plotting the constant.

**V5 · Per-overpass stamp panels · DO, but scoped.** Spectrogram + band traces (primary, proxy,
control) + a vertical marker at scene time + a quality strip.
**Hard scope:** 13 full + 5 partial = **18 panels maximum**, because the corpus stops at 18:45 UTC
while the measured gate2 overpass spread is 18:17-19:49 UTC.
**Every caption must say "scene catalogued", not "vessel present"** — no imagery is ordered and no
optical detections exist yet.
*Null check:* render the same panel time-shifted +/-1 h and date-shuffled. If those look equally
interesting, the eye is the detector, and the eye is not a detector.

**V6 · Seasonal / interannual distributions · DO.** Box or violin of daily band percentiles by
(year x month). Always draw the control band alongside (§3.4).

**V7 · Minute-of-window x date heatmap · DO — this replaces the diel plot.** 0-150 minutes on one
axis, date on the other. Honest about coverage, and still shows within-window structure and
file-seam artefacts.

---

## §5 · Anti-recommendations — do NOT do these here

Each of these produces a finite, plausible, publishable-looking number. That is what makes them
dangerous.

**5.1 · Censoring-aware estimators (Kaplan-Meier, Tobit, censored likelihood) · DO NOT.**
The premise is falsified (§1.3): in-band censoring is ~0.05%, not 18.7%. A Tobit fit at 0.05%
censoring returns essentially the OLS answer with inflated standard errors and a
"censoring-corrected" label that makes the number look *more* careful than it is.
*Keep* `censoring_report` as a **guard** that prints the fraction — a future winter or night pull
could genuinely be censored, and then you would want to know.

**5.2 · Arithmetic mean of `level_product_db` across bins or frames · DO NOT.**
Averaging decibels is wrong even when calibrated; here you cannot convert to power to average
correctly either, because the scale is unknown. A median across bins is the right *robust central
tendency* — but see 5.3: **it is not a band level**, and calling it one over-claims.

**5.3 · Decidecade (third-octave) band levels, even above 2.2 kHz · DO NOT.**
Decision 0010 permits them above 2.2 kHz on *resolution* grounds. But a decidecade level is a
**power sum over bins**, which needs the counts->dB scale. So they are blocked by **two independent
reasons** — resolution below 2.2 kHz, *and* the missing scale everywhere. If §3.1 succeeds, only
the first remains. This is a stronger statement than 0010 as currently written.

**5.4 · Hybrid millidecade bands (the MANTA/NCEI soundscape standard) · DO NOT.**
Requires ~1 Hz resolution below 455 Hz. The product has 250 Hz bins. Not approximable.

**5.5 · DEMON / cavitation envelope analysis · DO NOT — and quote the number.**
The frame series is 4 Hz, so its Nyquist is 2 Hz, so detectable shaft rates are below 120 rpm.
A small outboard runs 2,000-6,000 rpm with 3 blades => a 100-300 Hz blade rate. **Off by two orders
of magnitude.** This is absent, not degraded.

**5.6 · LOFAR / sub-250 Hz tonal tracking — and ONC's own "~100 Hz" advice · DO NOT.**
Bin 0 is DC, so 100 Hz is not representable at any band width. `features.assert_band_representable`
already refuses this correctly and its error message is good; the point here is that the advice
will be **quoted back as authority** by someone who has not read the bin map. Say plainly that it
is unusable, and note the second reason: bins 1-4 also sit inside the low-frequency deficit, so
even the 250 Hz-1 kHz proxy band is *degraded*, not merely truncated.

**5.7 · Acoustic indices — ACI, NDSI, ADI, acoustic entropy · DO NOT. Highest-risk item here.**
- **ACI** is defined on *amplitude*, as a sum of relative differences between adjacent frames over
  their sum. Its value changes under any monotone warp — it fails §3.2's invariance test by
  construction.
- **NDSI**'s biophony/anthropophony split (1-2 kHz vs 2-8 kHz) is a terrestrial soundscape
  construct with no validation at 250 Hz resolution in a 128 kHz band, and here the "anthropophony"
  band is precisely the one the deficit degrades.
- The marine literature already reports them as unreliable underwater, with NDSI misclassifying
  noise sources in noisy settings and ACI confounded by loud biological activity.

They are dangerous **because they always produce output**: a finite, seasonally-varying number that
looks like a result.

**5.8 · Diel (hour-of-day x day-of-year) analyses · DO NOT.** No coverage (§0(c)). Use V7.

**5.9 · Lloyd-mirror striations for range/CPA · DO NOT as framed; speculative box only.**
The repo has **no hydrophone depth or coordinate constant** — that is interface I2, still owed.
With z_r ~ 100 m (unverified) and z_s ~ 1 m, null spacing is dF ~ c*r/(2*z_s*z_r) ~ 7.5*r Hz:
**3 bins at 100 m, 15 bins at 500 m.** Marginally resolvable at close range only, on 250 Hz bins,
0.25 s frames, behind an unknown filter. It is a research project, not a method. The numbers are
recorded here so nobody has to re-derive them to reach the same conclusion.

**5.10 · A global fixed threshold in product counts across years · DO NOT.** See §1.4.

**5.11 · `corpus_file_index()` for population statistics before deduping · DO NOT.** See §1.1.

---

## §6 · What only the WAV / `.mat` products buy

ONC's own recommendation was to use the audio, or to generate `.mat` spectrogram products through
Oceans 3.0. Concretely, that route unlocks:

- **Everything below 250 Hz** — blade rate, the 50-150 Hz cavitation peak, and the 0.125 kHz
  third-octave band where Hermannsen et al. (2019) measured recreational vessels elevating noise by
  47-51 dB. This is the band where the project's target signal is strongest and the product is blind.
- **Calibrated dB re 1 uPa**, via the pre-deployment calibration file (10 Hz-51.2 kHz).
- **DEMON**, hybrid millidecade, decidecade, SEL — every energy-summing statistic.
- **Arbitrary time-frequency resolution**, chosen for the question rather than inherited.

**Cost, measured from the local sample:** 115 MB per 5 minutes => ~1.4 GB/hour. The 18 covered
overpass windows at +/-15 min is ~9 hours ~ **13 GB** — entirely feasible **for the matchup subset**.
The full corpus at this rate would be ~3 TB, which is not.

> The honest framing: **`.fft` for the 6-year survey, WAV for the matchup subset.** They answer
> different questions and only the WAV can validate the other.

---

## §7 · Cross-cutting conventions

**Four nulls accompany every result** (invariant 4):
1. shift the time base by +/-1 h;
2. shuffle labels across windows;
3. compare against a matched-hour quiet-period control;
4. apply a monotone warp and confirm the statistic is unchanged (§3.2).

**Every figure states**, in the caption or on the axis: UTC; the statistic and how it reduced; the
bin range; the +/-125 Hz axis uncertainty; and `level_product_db` — **never "SPL", never
"dB re 1 uPa"**.

**And distinguish the two outcomes** (invariant 9): "the method found nothing" and "the method is
broken" need different follow-ups, and only the second is a bug. Several checks above — V2's L95/L50
separation especially — exist precisely to tell them apart.

---

## §8 · Companion notebook

`contributor_folders/isaac/hydrophone_methods_gallery.ipynb` renders eleven of the figures argued
for here, on 180 corpus windows (30/season), and re-derives every measured number in this document.
It executes top-to-bottom in a fresh kernel in ~2 minutes. Its final cell is the only place it
calls into `boatphone.features`, and it checks that module's band-level series against a local
computation on every run.

---

## §9 · Sources

- Merchant, N. D. et al. (2013). Spectral probability density as a tool for ambient noise analysis.
  *JASA* 133(4): EL262-267. <https://asa.scitation.org/doi/10.1121/1.4794934>
- Hermannsen, L. et al. (2019). Recreational vessels without AIS dominate anthropogenic noise
  contributions to a shallow water soundscape. *Scientific Reports* 9: 15477.
  <https://www.nature.com/articles/s41598-019-51222-9>
- Merchant, N. D. et al. (2015). Measuring acoustic habitats. *Methods Ecol. Evol.* (PAMGuide).
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC4413749/>
- Miksis-Olds, J. et al. (2021). Ocean sound analysis software for making ambient noise trends
  accessible (MANTA). *Front. Mar. Sci.* 8: 703650.
  <https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.703650/full>
- Triton soundscape metrics (LTSA, decidecade, hybrid millidecade).
  <https://github.com/MarineBioAcousticsRC/Triton/wiki/Soundscape-Metrics>
- Wladichuk, J. et al. — small-vessel source levels from controlled measurements, incl. BC coastal
  sites. See also <https://www.mdpi.com/2077-1312/14/6/561>
- Marine acoustic-index caveats (ACI/NDSI underwater): <https://doi.org/10.3390/urbansci9040129>
- `references/ONC_communication.txt` — Austen Sorochak (ONC) on the `.fft` product, in-repo.

---

## Appendix · Verification ledger

| Claim | Status | To settle |
|---|---|---|
| 26,666 unique windows; 90 duplicates | **verified** (all stems; 10/10 payload hashes) | — |
| in-band floor fraction 0.00-0.55%, not 18.7% | **verified** (16 files + 6/yr) | full pass to publish one number (~37 core-min) |
| bin 0 max 60, nonzero fraction 0.232, 20/25 violate 0014 | **verified** (25 files) | full pass for the corpus max |
| bins 425-511 nonzero in 15/25 files | **verified** (25 files) | full pass for per-file counts |
| echosounder absent: 100% of 2021, 50% of 2022, ~3% otherwise | **verified** (30 files/season) | targeted scan to bound the 2022 outage dates |
| 1.000 Hz ping / 0.25 s frame confirmed | **verified** (4 files, incl. one negative control) | scan ~100 files across years |
| ping does not leak into 1-10 kHz | **verified** (4 files, detrended) | wider sample |
| WAV<->product overlap 296.029 s, 250.0 Hz bin match | **verified** (headers read directly) | — |
| seasonal swing in 1-10 kHz is **2.0 counts**, not 8.3-8.5 | **verified** (1,790 windows, all 895 dates) | — the 8.3-8.5 figure was small-sample noise and is withdrawn |
| no COVID-era signal in the vessel bands | **verified** (1,790 windows): event rate (>=+10 over ambient) 12.6 / 11.7 / 11.8 / 13.7 / 11.3 / 12.0 % for 2020-25; 2020->2021 is -7.4%, inside year-to-year scatter | — |
| narrowband lines at 20.25/61/78.5/81.25/91.5/100 kHz are system self-noise | **verified** (duty 0.87-1.00 for the continuous ones, all six seasons; harmonics of ~20.3125 kHz land on bins 81/244/325 exactly) | identify the clock with ONC; confirm the 61 kHz line's source |
| the 37.65 kHz feature is an instrument, not traffic | **verified**: 1.000 Hz PRF (ACF lag-4 = 1.00 on / 0.03 off); present in ~100% of windows when on; single step transition 2021-05-01 -> 2022-07-17 OFF, 2022-07-18 ON; switch-off falls in the winter service gap | identify the instrument with ONC |
| WAV<->product cross-bin slope 0.52 counts/dB, r² 0.55 | **verified**, one window | many matched windows, fitted per bin |
| corpus max level | **unverified** (112 claimed; 115 reported elsewhere) | full pass |
| centre-vs-edge axis convention | **open** (decision 0013) | §3.1, or a two-bin census — ONC has no documentation |
| hydrophone depth / coordinates | **absent from the repo** | interface I2, owed by the optical side |
