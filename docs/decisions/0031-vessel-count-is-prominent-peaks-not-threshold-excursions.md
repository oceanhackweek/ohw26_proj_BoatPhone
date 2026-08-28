---
title: Count vessels as prominent PEAKS of the smoothed band level, not threshold excursions
number: 0031
status: accepted
date: 2026-08-28
owner: Isaac Guld
related:
  - 0030-time-shift-null-does-not-collapse.md
  - 0015-censoring-aware-thresholds-for-b5.md
  - 0027-fft-product-is-viable-for-b5-band-level-detection.md
---

# AMENDMENT 2026-08-28 -- a duplication bug moved every number here

**A bug found after this record was first written changed all of its figures.**
The top-up pull was re-run for all 30 scenes and re-fetched windows the bulk
corpus already held, into a second directory. `corpus_file_index` deduplicates
WITHIN a container, and callers concatenated two of its results -- so **17 of 29
labelled overpasses had every file present twice**.

Duplicated timestamps do not merely double-count: they HALVE the real-time span
of any fixed-frame kernel. The 45 s smooth became 22.5 s and the 180 s minimum
separation became 90 s, **on 17 of 29 windows and not the other 12** -- a
silently inconsistent estimator, not a biased one. Fixed by
`overpasses.analysis_file_index()`, which dedupes across containers; 98
duplicates dropped.

Corrected figures at +/-15 min, deduplicated:

| estimator | predicted | ratio | within +/-1 | r | r_partial |
|---|---|---|---|---|---|
| `peak_prominence` (prom = T/2) | 48 | 1.09 | 90% | +0.611 | +0.609 |
| **`estimate_vessel_count` (prom = T)** | **40** | **0.91** | **90%** | **+0.590** | **+0.581** |
| band-split union (post-hoc, below) | 43 | 0.98 | 86% | +0.603 | -- |
| merged events, 10 s | 55 | 1.25 | 66% | +0.477 | +0.463 |
| presence only | 24 | 0.55 | 72% | +0.262 | +0.230 |

The two pre-declared peak variants are now a TIE (|ratio-1| = 0.09 either way,
both 90% within +/-1). Production stays on the stricter one rather than churning
for a coin-flip.

## The loudness worry is CLEARED

This record originally warned that part of the correlation was a loudness effect,
because the frame-shuffled count still correlated +0.38 with the manual count.
That was the wrong diagnostic: a frame shuffle preserves each window's level
DISTRIBUTION, so it cannot probe skill arising from BETWEEN-window level
differences. The right test is partial correlation against window median level.

**`r_partial` = +0.581 against `r` = +0.590.** Removing the linear effect of
window loudness costs 0.009 of correlation. The estimator is not reading
loudness. (Diagnostic suggested by a Fable methods consult, 2026-08-28.)

## Concurrent vessels: diagnosed, partially fixed, NOT adopted

The two remaining misses were both concurrency failures, exactly as predicted:
two or more vessels present at once are ONE peak in a single band level, by
construction.

* `20240801_1928` -- 4 counted, 1 predicted. Measured: 1-4 kHz has **three**
  peaks (-6.8, -1.0, +6.0 min) and 4-10 kHz has **two** (-10.2, +5.9 min), but
  the 1-10 kHz median smears them into one.
* `20250630_1938` -- 2 counted, 0 predicted. Same shape.

A band-split union (peaks counted per sub-band, deduplicated in time at 120 s,
bands qualifying only if their whitened range exceeds 5 counts) recovers **4 and
2 exactly** and scores ratio 0.98, r = +0.603. It is **NOT adopted**: its four
parameters were chosen from a post-hoc sweep of 36 configurations, which is
precisely the fitting this record's pre-declaration exists to avoid, and it does
not beat the pre-declared estimator on MAE or within-1. It is the strongest
candidate for the next labelled batch to arbitrate.

**A negative result worth keeping:** 2-band and 4-band splits give IDENTICAL
numbers across every configuration tested, because the 10-25 and 25-51 kHz bands
never pass the qualification gate -- measured whitened range 1.6 and 0.9 counts
on the 4-vessel window. Absorption-driven spectral separation by range, the
mechanism that would let high bands distinguish a near vessel from a far one,
**does not operate at the ranges in this data**. Only the 1-4 / 4-10 kHz split
carries information.

Also closed by the same consult: **DEMON / envelope demodulation of blade rate is
not merely hard here, it is impossible.** The 0.25 s frame gives a 4 Hz envelope
sampling rate and so a 2 Hz Nyquist, against blade rates of 50-150 Hz. Do not
spend time on it.

# Context

`find_events` counts runs above an excess-over-ambient threshold. A vessel's
level fluctuates as it passes, so **one passage opens many runs**: 298 raw events
against 44 vessels counted by eye. Merging adjacent fragments across a 10 s gap
cut that to 135 but left the correlation with the manual count at **r = +0.20**,
so fragmentation was not the whole story -- the estimator was counting the wrong
kind of thing, not merely counting it badly.

29 human vessel counts (`data/validation/planet_scope_validations.csv`) made it
possible to test alternatives instead of arguing about them.

# Decision

**`run_b5_gate.estimate_vessel_count` is the vessel-count estimator.** Prominent
local maxima of the band level smoothed over 45 s, requiring a peak to stand a
full `EVENT_EXCESS_THRESHOLD_COUNTS` above its *surroundings* (prominence, not
height) and to sit at least 180 s from the previous peak.

A passage rises to closest point of approach and falls exactly once, so it
contributes exactly one peak however much it wobbles on the way. **Prominence is
what does the work**: it is what separates "one vessel whose level wobbles" from
"two vessels", and it is why this beats merging, which can only join fragments
that are already adjacent in time.

## Measured, on the clipped +/-15 min window

| estimator | predicted | ratio | within +/-1 | r |
|---|---|---|---|---|
| **`estimate_vessel_count`** | **42** | **0.95** | **27/29 (93%)** | **+0.613** |
| merged events, 10 s gap | 58 | 1.32 | 59% | +0.341 |
| raw events | 119 | 2.70 | 48% | +0.362 |
| presence only (0 or 1) | 24 | 0.55 | 72% | +0.262 |

Against 44 vessels over 29 labelled instants. **Replicates independently on the
`ship_proxy` band** (250 Hz-1 kHz, disjoint from the 1-10 kHz band above): ratio
1.07, r = +0.598. Two independent frequency bands agreeing is the strongest
evidence here that the improvement is not an artefact of one band's noise.

## Chosen from a pre-declared field

`scripts/compare_event_grouping.py` declares **8 estimators x 4 window widths**,
written down before any was scored, and reports all 32. At n = 29 that is the
only available defence against picking a winner by its score. It is not a strong
one, and **this is a proof of concept, not a validated model** -- the user's
stated bar was "within 70% of the validation numbers", which 7 of the 32
configurations clear.

The peak-based family (`merged_120s`, `slope_zero_cross`, `peak_prominence`,
`peak_prominence_strict`) clusters at r = 0.55-0.61 while threshold-counting sits
at 0.34-0.36 and presence-only at 0.26. **The family effect, not the single
winner, is the result.**

# The window is +/-15 min, and shorter is worse

Tested at +/-15, 10, 5 and 2.5 min. Shortening degrades everything: at +/-5 min
the best ratio is 0.43 and the estimator badly undercounts. Vessels audible in a
30-minute window are simply not all within 5 minutes of the overpass instant.
**The window stays at `OVERPASS_MATCH_HALF_WINDOW_S`.**

Consequence found while wiring this in: `concat_window` returns whole 5-minute
FILES, so a 1800 s window arrived as up to 2100 s of samples and every statistic
was computed over ~17% more time than its label claimed. `build_events_table` now
**clips to the window** and prints the dropped frame count. Unclipped, this
estimator reads 51 against 44 instead of 42 -- the clip is not cosmetic.

# What this does NOT establish

* **The time-shift null still does not collapse.** An hour-shifted window yields
  37 against the real window's 42. The estimator counts passages well; it does
  **not** establish that they are the *overpass's* vessels. Decision 0030 stands
  and is unaffected -- most likely because midday Barkley Sound has traffic all
  afternoon, so -1 h is not a negative control.
* **The frame-shuffled null still correlates r = +0.38 with the manual count**,
  even though shuffling collapses the count itself from 42 to 9. Part of the
  +0.61 is therefore a LOUDNESS effect that survives destroying time order, not
  passage structure. The temporal contribution is the gap between the two, not
  the whole of +0.61.
* **Two vessels within 180 s of each other count as one**, by construction.
* **Not calibrated.** Levels remain uncalibrated counts, never dB re 1 uPa.

# How to tell this was wrong

More labels. 29 instants with 6 negatives cannot separate a real estimator from
a lucky one, and the two remaining misses (`20240801_1928`, 4 counted -> 1
predicted; `20250630_1938`, 2 -> 0) are unexplained. If the ratio moves far from
1.0 on a second batch of labelled scenes, this was fitted, not measured.
