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
