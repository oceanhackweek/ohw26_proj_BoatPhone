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
