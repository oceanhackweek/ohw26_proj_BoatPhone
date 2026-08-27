# 0027. The `.fft.gz` product IS viable for B5 band-level detection -- GO

Status: accepted
Date: 2026-08-27
Scope: `boatphone/features.py`, `boatphone/overpasses.py`, `scripts/run_b5_gate.py`,
`boatphone/config.py` (B5 band block), `docs/plans/acoustics_plan_v2.md` SS5 B1/B5/B6
Related: decision 0010 (analysis band), 0012 (B0 NO-GO), 0013 (axis convention open),
0014 (two ceilings), 0015 (censoring-aware thresholds), 0026 (86 is not a ceiling),
`references/ONC_communication.txt`
Supersedes in part: `acoustics_plan_v2.md` SS5 B1's pass criterion (see "What this closes")

## Context

ONC's product owner answered our questions about the `.fft.gz` product
(`references/ONC_communication.txt`). Three statements matter:

1. the product is a proprietary Ocean Sonics format, "essentially uncalibrated unless you have
   all the metadata from Ocean Sonics, which I don't believe ONC currently publishes";
2. filtering is applied that ONC cannot document ("I believe there is sometimes filtering
   applied to the fft files for some reason");
3. the 256 kHz sensitivity curve they could supply "wouldn't be applicable to those files".

That forecloses absolute calibration on this product and, with it, `acoustics_plan_v2.md` SS5
B1's pass criterion -- a magnitude regression of product bin *b* against WAV bin *2b* with
slope ~ 1. ONC's own recommendation was to proceed relatively ("the absolute calibrated level
isn't necessary for that to work") or to switch to Oceans 3.0 spectrogram `.mat` data products,
which are calibrated and small but carry processing latency.

The question this record answers is therefore narrow and practical: **is the `.fft.gz` product
usable as a vessel-presence surface at all, or must the project order `.mat` products?** It was
run before any further acquisition, because the answer decides what to acquire.

## What was run

`scripts/run_b5_gate.py`, over the 13 Planet gate2 scenes whose +-15 min acoustic window the
existing 6.8 GB corpus fully covers (of 30 scenes: 13 full, 5 partial, 12 with no coverage --
see decision on the overpass window, pending). Two bands, from `config`:
`FFT_B5_SMALL_CRAFT_BAND_HZ` (1-10 kHz) and `FFT_B5_SHIP_PROXY_BAND_HZ` (250-1000 Hz).
Wall clock: **42 s** for the full gate.

## Finding: GO

| Evidence | 1-10 kHz | 250-1000 Hz |
|---|---|---|
| windows with >= 1 event (>= 10 dB excess, >= 20 s duration) | 10 / 13 | 11 / 13 |
| events per window, real | 3.31 | 3.69 |
| events per window, **frame-shuffled null** | **0.00** | **0.00** |
| total events real / shuffled | 43 / **0** | 48 / **0** |
| event duration, median / max | 67 s / 897 s | 70 s / 1358 s |
| peak excess over ambient, median / max | 45 / 81 dB | 42.5 / 89 dB |
| max in-band floor-censored fraction | **0.00%** | 24.45% |

1. **The surface reduces without degeneracy.** In the 1-10 kHz band, **zero** in-band cells sit
   at the censoring floor across all 13 windows, so the median reduction
   (`FFT_BAND_LEVEL_STATISTIC`) is unbiased there and decision 0015's precondition is met with
   room to spare. The 250-1000 Hz band reaches 24.45% -- still under the 50% at which a median
   breaks, but visibly degraded, exactly as the ~42 dB low-frequency anomaly predicts.
2. **Closest-point-of-approach structure is present and is the dominant feature.** Events are
   contiguous runs of tens of seconds to minutes, not impulses. Traced at 10 s resolution, the
   strongest window (`20220625_182034_44_2440`) shows a symmetric rise-and-fall from 42 to 104
   and back to 31 over ~4 minutes -- the level-versus-time signature `acoustics_plan_v2.md` SS5
   B5 is built on, visible directly in the product.
3. **The synthetic-tone proof passes exactly** (CLAUDE.md invariant 3). A broadband in-band
   injection of 50 dB over ambient is recovered at 50.0 dB, at the correct time, with the
   correct 50 s duration; the same injection placed out of band yields zero events.
4. **The frame-shuffle null is rejected in all 26 window-band combinations**, 0 events against
   43 and 48. Shuffling preserves the level distribution exactly and destroys only time
   ordering, so this is direct evidence the statistic measures temporal structure rather than a
   level artefact.

**Verdict: GO. B5 proceeds on `.fft.gz`. Oceans 3.0 `.mat` products are NOT needed for
vessel presence,** and the latency ONC warned about is not on the critical path. `.mat` remains
the only route to absolute calibrated levels if B6 is ever wanted (see "What stays open").

## What this closes

* **`acoustics_plan_v2.md` SS5 B1's pass criterion is withdrawn, not failed.** No slope-~1
  regression against WAV can be run, because ONC states the product is uncalibrated and
  filtered. B1's own stated fallback -- "proceed on relative and shape-based features" -- is
  hereby the decided path, and the plan's note that "G1, G2 and G3 all survive this" is upheld
  by the evidence above. The ~42 dB low-frequency anomaly is **explained** (undocumented
  filtering), not resolved.
* **The B1 WAV second pass is moot for calibration.** It was required only to give the
  magnitude regression something to regress against.
* **B6's absolute-calibration branch is dead on the fft path.** Every bin is uncalibrated, not
  merely bins 205-417 -- decision 0010's "calibrated support, bins 1-205" describes the
  instrument, not this product.
* **Decision 0026 is extended.** `FFT_LEVEL_CEILING = 86` was already falsified for the corpus
  at 112.0 from 90 probe files; the gate observes **111** and **98** as per-window maxima across
  a different, six-season sample, consistent with 0026 and further evidence that 86 is not a
  clip point. No change to the constant.

## What stays open, and is NOT resolved by this record

* **No detection performance is measured, and none may be quoted.** There are no vessel labels:
  no Planet imagery has been ordered (`planet_folger_HANDOFF.md`: "Nothing ordered",
  `CONFIRM_ORDER = False`) and the optical detector has produced no detections. A "positive"
  window here means only that a cloud-free scene was **catalogued** at that instant. **Goal G1's
  "validated against optical labels, with a stated false-positive rate" remains blocked on the
  optical arm.** The events counted above are candidate vessel passes, not confirmed ones.
* **The time-shift null has no discriminating power on real data here, and its passing is not
  evidence.** Real and time-shifted event rates match closely (3.31 vs 4.46 in the primary
  band). That is the expected result, not a bug: with no label to misalign, an hour earlier on
  the same cloud-free summer afternoon carries just as many boats. It discriminates only against
  synthetic ground truth, where it is asserted by
  `check_b5_6_time_shifted_window_does_not_carry_the_event`. Do not cite the real-data
  time-shift figures as a passed null.
* **ONC's ~100 Hz recommendation is not implementable** and was not implemented. 100 Hz falls
  in bin 0 (DC); bin 1 at 250 Hz is the product's floor. `FFT_B5_SHIP_PROXY_BAND_HZ`
  (250-1000 Hz) is a *proxy*, not that band, and `features.assert_band_representable` raises on
  any request below 250 Hz rather than silently clipping.
* **The centre-vs-edge axis question (decision 0013) is now harder to close, not closed.** Its
  cheapest resolution route was "ask ONC for the product definition"; ONC's reply is that the
  documentation does not exist. Route (c) needed B1 to pin counts to dB, which this record
  forecloses. Only the scale-free two-bin-split centroid census over the corpus remains, and
  `FFT_AXIS_OFFSET_UNCERTAINTY_HZ = 125.0` is carried by every band edge in the meantime.
* **The CATFISH vs May River band disagreement is not yet adjudicated.** Both bands fire at
  similar rates; the discriminating evidence is the censoring asymmetry (0.00% vs 24.45%),
  which favours the small-craft band on data quality, not on which band carries more vessels.

## Consequences

* B5 is unblocked and is the project's vessel-presence detector, as decision 0012 established.
* The targeted top-up pull for the 17 uncovered/partial scenes, and the overpass-window
  correction that caused them, are now worth doing -- the product they would feed is proven.
* Every figure and number derived from this path must carry: the relative/uncalibrated unit
  (`level_product_db`, never dB re 1 uPa), the overpass count as the unit of analysis, and the
  sampling conditionality in `config.PLANET_SAMPLING_CONDITIONALITY_STATEMENT`.

---

## Amendment, 2026-08-27: the seasonal-threshold challenge, tested

`references/hydrophone-methods-brief.md` §1.4/§1.5 raised the one objection to this record that
could have changed its verdict, and named it as such: the gate scores events at a **fixed +10 dB
excess**, while the 1-10 kHz band median moves ~8.5 counts across seasons — the same order as a
vessel-pass excess — and the gate's 13 windows do not span the seasonal range. If that shift
reached the detector, the same physical vessel would clear the threshold in one season and miss it
in another, and the event rate would carry a seasonal artefact indistinguishable from a trend in
traffic.

**Tested, at the brief's own suggested design: 240 windows, 40 per season, 2020-2025, deduplicated
index.**

| Year | median baseline | events/window | median peak excess |
|---|---|---|---|
| 2020 | 38.5 | 0.07 | 17.0 |
| 2021 | 39.0 | 0.12 | 17.5 |
| 2022 | 38.0 | **0.62** | 22.0 |
| 2023 | 36.0 | 0.10 | 18.0 |
| 2024 | 35.0 | 0.05 | 18.0 |
| 2025 | 36.5 | 0.12 | 17.0 |

Correlation between a window's own baseline and its event count: **r = -0.021** (rank r = -0.037),
n = 240.

**The objection does not hold for this detector, for a reason the brief could not see from the
outside: the threshold is not absolute.** `band_excess` measures +10 dB over *that window's own*
10th-percentile baseline, recomputed per window, so a seasonal shift in ambient is subtracted
before the threshold is applied. That is what the per-window baseline is for, and the near-zero
correlation is the direct evidence that it works. At 40 files/season the seasonal spread of the
median baseline is also **4.0 counts**, not 8.5 — consistent with the brief's own caveat that its
8-30 files/season mixed seasonal variability with sampling noise.

**What WOULD break it, and is not ruled out:** a change in the level scale's *gain* (counts per
dB). A per-window baseline removes an additive offset, not a multiplicative one, so under a gain
change "+10 counts" would mean different physical dB in different seasons. The evidence against
that is the brief's own §1.4: the 51-102 kHz control band sits at ~5.0 in **every** year, and a
gain change would have moved it too. Weak evidence, since that band sits near the floor where a
gain change is hardest to see, and it should be revisited if absolute levels ever matter.

**An unexplained finding this test surfaced, recorded rather than buried:** 2022's event rate is
**0.62 events/window against 0.05-0.12 in every other season** — a five- to twelvefold difference
that does NOT track the baseline (2022's baseline is mid-range). It is not a threshold artefact,
and this gate cannot say whether it is real traffic, a propagation regime, or an instrument
change. It is a flag for whoever builds the continuous estimate, not a result.

Also fixed as a direct result of the brief (§1.1, §1.5 item 3): `overpasses.corpus_file_index`
double-counted 90 windows present in both containers. Verified independently -- 90 duplicate start
times, 12/12 tested pairs byte-identical, on 2025-07-15/16 and 2025-08-12. The index now
deduplicates by start time preferring `.fft.gz`, reports the drops via
`corpus_index_duplicates()`, and is pinned by
`check_b5_10_corpus_index_deduplicates_windows_present_in_both_containers`. The 13/5/12 coverage
split is unchanged (no scene falls on the affected dates), but any corpus-wide statistic built on
the old index was 0.34% duplicated.

---

## Amendment 2, 2026-08-27: the population pass settles the seasonal question

The first amendment tested the seasonal-threshold objection on 240 windows and found the
per-window baseline absorbed it. The full-population pass now measures the underlying quantity
directly, over **all 26,665 readable windows** rather than the 180-window sample the objection
was raised from.

Per-season mean band level, ~4,500 windows per season:

| Year | n | control (51-102 kHz) | small craft (1-10 kHz) | craft - control |
|---|---|---|---|---|
| 2020 | 4,513 | 6.59 | 43.03 | 36.44 |
| 2021 | 4,574 | 6.70 | 42.79 | 36.09 |
| 2022 | 4,422 | 6.59 | 43.27 | 36.68 |
| 2023 | 4,026 | 6.10 | 42.26 | 36.16 |
| 2024 | 4,576 | 6.77 | 44.23 | 37.46 |
| 2025 | 4,554 | 6.35 | 42.23 | 35.88 |

**Seasonal spread: control 0.67 counts, small craft 1.99 counts, and the two differenced
1.57 counts.**

**The 8.5-count seasonal spread was sampling error.** It measured 8.5 at 8-30 files/season
(methods brief §1.4), 4.0 at 40 files/season (amendment 1), and **1.99 at ~4,500 files/season**.
The effect shrinks monotonically with sample size, which is the signature of noise rather than
of a real regime shift. Against a +10-count threshold the true seasonal variation is about 20%
of one threshold — and the per-window baseline subtracts it before the threshold is applied
anyway, so it never reaches the detector at all. **The objection is closed on both counts.**

The control band earns its keep here for the first time: it moves 0.67 counts across six seasons
while the craft band moves 1.99, so most of the craft-band variation is *not* instrumental. 2024
is the high season in both bands (6.77 and 44.23), so part of its elevation is the instrument and
part is not — which is exactly the discrimination this band was added to provide, and which no
amount of staring at the craft band alone could have supplied.

**Two further population findings, recorded because a sample could not have established either:**

* **The ~37.6 kHz echosounder is seasonally absent, at population scale.** P3a shows 2021 flat
  across 33-40 kHz where every other season peaks near 33 counts. This confirms the methods
  brief's LTSA observation (absent 2021-2022) on 26,665 windows rather than 180, inside a single
  deployment. It does not touch the 1-10 kHz detection band.
* **One corpus file is empty, and the pull recorded it as a success.**
  `ICLISTENHF1266_20220825T184149.000Z.fft.gz` has `bytes_downloaded: 0`, `http_status: 200`,
  and sha256 `e3b0c442...` — the hash of the empty string. ONC served an empty body with a 200,
  and `status` reads `"downloaded"`. It is the ONLY such file in the corpus (checked: exactly one
  row with zero bytes, exactly one file under 100 bytes on disk). So the corpus holds **26,665
  usable windows, not 26,666**, and the manifest's `present` count is off by one in the direction
  of optimism. This is a third category alongside decision 0007's measured zeros and 0023's
  Error-96 absences: a *listed, served, empty* file. The pull's absent/present accounting should
  distinguish it, and that belongs with the manifest work in
  `docs/plans/b3-close-out-offline.md`.

Artefacts: `data/derived/population/<run>/` — `population.npz` (per-season exact integer
histograms and per-window median spectra), `windows.json`, `seasonal_ambient.npz` (per-season
L95, the ambient a detector can subtract without a window subtracting its own vessel away),
and figures P1 (LTSA), P2 (SPD), P3 (seasonal spectra and ambient). Produced by
`scripts/build_population_set.py` and `scripts/plot_population_set.py`.

---

## Amendment 3, 2026-08-27: the second baseline, and a correction to amendment 2's reading

The population ambient is now wired into the review set as a **second baseline** beside the
per-window one (`features.load_seasonal_ambient_counts`,
`features.band_baseline_from_per_bin_ambient`). Each window reports how far its own quiet floor
sits above its season's, per band.

**Measured across all 19 review windows, 1-10 kHz:** baseline shift ranges **+0.0 to +28.0
counts**, median **+8.0**. Most windows in this set were elevated above their season's quiet
floor for their entire duration, by an amount comparable to or larger than the +10-count event
threshold — and the per-window baseline subtracts exactly that away before the threshold is
applied. The five windows elevated by >= 15 counts average **3.00 events**, against **5.86** for
the rest, and the three most elevated windows produce **zero events each**. The correlation
across all 19 is weak (r = -0.234, n = 19) so this is a flag rather than a measurement, but the
mechanism is not in doubt: a detector that defines "quiet" from the window it is scoring cannot
see a window that was never quiet.

### The correction

**Amendment 2's far-field reading of the labelled window is withdrawn as unsupported.**

That window (`20210601_191849_05_2412`, optically confirmed no vessels within 10 km2) sits
+12.0 counts above its season's floor in 1-10 kHz and +18.5 in 250-1000 Hz. The earlier
interpretation offered was far-field contamination — a vessel outside the ~1.78 km reviewed
radius, audible but not visible — supported by the elevation being stronger at low frequency.

**The control band does not support that, and it is the channel that was added to adjudicate
exactly this.** Across the 19 windows, the control-band shift and the small-craft shift correlate
at **r = +0.847**. The control band spans 51-102 kHz, where no small vessel puts meaningful
energy, so a window elevated there is elevated BROADBAND — sea state, flow noise or instrument
condition, not a vessel. And the labelled window sits **on that trend, not above it**: its control
shift of +2.0 against a craft shift of +12.0 is what the relationship predicts. There is no
vessel-specific excess left to explain once the broadband component is accounted for.

So the honest statement is: **that window was broadband-elevated throughout, in a way shared with
most windows in this set, and nothing in the acoustics singles it out as containing a
missed vessel.** The far-field diagnostic `acoustics_plan_v2` §5 B5 asks for remains a real and
worthwhile check; this is not yet an instance of it.

Two caveats that survive independently, and either could still overturn this:

* The window is **off-strip** (19:03 UTC against the corpus's 16:00-18:59), so its comparison to
  a season ambient measured at a different hour confounds time-of-day with source. Every
  off-strip window now carries this warning on the figure and in `summary.json`. The labelled
  windows are off-strip by construction, so this will recur.
* **n = 1 label.** Nothing here is a rate.

### What this changes about the detector

Nothing yet, and deliberately. Both baselines are reported and neither replaces the other: the
per-window baseline is blind to a window elevated throughout, the population baseline assumes the
window belongs to its population and is wrong for off-strip windows. Switching the detector onto
the population baseline would trade a known blind spot for an unquantified bias, and decision 0015
forbids retuning the threshold more than once on the overpass set. The disagreement between them
is a **reported diagnostic**, and the right moment to act on it is when labels exist to say which
of the two was right.
