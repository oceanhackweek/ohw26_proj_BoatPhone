---
title: The time-shift null is UNRESOLVED -- confounded by unequal coverage on the two sides
number: 0030
status: accepted
date: 2026-08-27 (amended 2026-08-28)
owner: Isaac Guld
supersedes: none
related:
  - 0027-fft-product-is-viable-for-b5-band-level-detection.md
  - 0015-censoring-aware-thresholds-for-b5.md
  - 0020-pull-refines-only-the-overpass-window.md
  - ../plans/acoustics_plan_v2.md
---

# Context

After the targeted top-up pull closed the coverage gap (all 30 gate-2 scenes now
have acoustic data; see "Measurement" below), `scripts/build_events_table.py`
scored every overpass window in all four bands with both nulls attached to every
row. This is the first time BOTH nulls have been run across the FULL scene set
rather than the 13 fully-covered windows the B5 gate scored.

The two nulls disagreed, and that disagreement was this record's original
finding. A coverage defect found the next day invalidated the measurement behind
it; see the amendment in Measurement below.

# Measurement

**AMENDED 2026-08-28 after the coverage defect below was fixed. The original
figures are kept at the bottom of this section because the amendment CHANGES THE
HEADLINE, and a reader who only saw the new numbers would not know the old ones
had been cited.**

## The coverage defect that invalidated the first measurement

`window_coverage` counts files by INTERVAL OVERLAP (decision 0020), but ONC's
archive listing selects by file START TIME. The file straddling a window's start
therefore overlaps the window while starting before it, and was excluded from
every listing -- losing up to `FFT_FILE_SECONDS` (300 s) at the HEAD of every
window, including windows that scored as "full". Verified live: listing
`2021-08-12T19:03:32 -> 19:33:32` returns 6 files; padding the start by 300 s
returns 7, the extra being `ICLISTENHF1266_20210812T190301.000Z`, which closes a
269 s head gap. Fixed in `scripts/pull_labelled_windows.py` by padding the
listing start by one file duration. After the re-pull: **30 full, 0 partial,
0 zero.**

## Current figures (full coverage, 30/30 windows)

| band | detection? | events | frame-shuffled | time-shifted (-1 h) |
|---|---|---|---|---|
| `small_craft` | yes | **146** | 0 | 101 |
| `ship_proxy` | yes | **152** | 0 | 85 |
| `rain` | diagnostic | 27 | 0 | 21 |
| `control` | diagnostic | 15 | 0 | 3 |

Detection bands together: **n = 298 events across 30 overpasses**, against **0**
under the frame-shuffle null.

## The real-vs-shifted comparison is CONFOUNDED and is not yet a result

The shifted windows were NOT re-pulled and still carry the head-gap defect. Real
windows now average **100%** coverage; shifted windows average **90%**, with 9 of
30 between 37% and 76%. The null therefore has less data and fewer opportunities
to fire, which biases the comparison toward "real exceeds shifted".

Normalised by covered time, the raw counts become:

| band | real | shifted | ratio |
|---|---|---|---|
| `small_craft` | 9.73 events/h over 15.0 h | 7.48 events/h over 13.5 h | **1.30x** |
| `ship_proxy` | 10.13 events/h over 15.0 h | 6.29 events/h over 13.5 h | **1.61x** |

**Do not cite these ratios as the overpass effect.** Rate-normalising corrects
for exposure but NOT for which windows were lost: the shifted windows missing
coverage are not a random sample, and a head gap removes the earliest part of a
window specifically. The comparison becomes meaningful only when the shifted
windows are pulled to the same completeness as the real ones. **That pull has not
been done, and until it is, the honest statement is that the time-shift null is
UNRESOLVED, not that it collapsed.**

## Superseded original figures (partial coverage -- do not cite)

Measured before the head-gap fix, at 19 full / 11 partial / 0 zero: small_craft
89 real vs 101 shifted, ship_proxy 92 vs 85, total **181** events against 0
frame-shuffled. These were the basis of this record's original conclusion that
the time-shift null "does not collapse" at parity. That conclusion rested on a
real-vs-shifted comparison in which BOTH sides were undercounted by the same
defect, and it is withdrawn -- not because it was refuted, but because the
measurement behind it was not sound.

# This is "the method found nothing", not "the method is broken" (invariant 9)

The distinction matters and the two demand opposite follow-ups.

The most likely reading is **physical, not a defect**: Barkley Sound has
recreational traffic throughout the midday hours, so an hour-shifted window is
not a negative control for vessel presence at all -- it is a second, equally
vessel-populated sample. A null that is not actually null cannot collapse, and
its failure to collapse is then a statement about the null's construction, not
about the detector.

**That reading is untested, and it must not be assumed.** The competing reading
is that the threshold is low enough to fire on ambient variability that is
present all day, in which case a large share of the 298 events are not vessels at
all. The frame-shuffle result does NOT discriminate between these: broadband
weather and flow noise are also genuine temporal structure and also survive a
shuffle.

**Do not resolve this by moving the threshold.** Decision 0015 permits tuning on
the overpass set exactly once, and it has been spent. Tuning until the time-shift
null collapses would be fitting the detector to make a null behave, which
manufactures the very result invariant 4 warns about.

# How to tell which reading is right

In increasing cost:

1. **Re-pull the shifted windows to parity.** Cheapest by far, and it is a
   PREREQUISITE for the other two rather than an alternative: until both sides of
   the comparison have the same coverage, no amount of further evidence
   interprets it. `scripts/pull_labelled_windows.py` now pads correctly; the
   shifted windows need their own pull into a separate landing zone, since they
   are not overpass windows and must not enter either the corpus or the top-up
   zone.
2. **A genuinely quiet control.** Score a night or winter window, where
   recreational traffic is near zero and the hour-shift confound does not apply.
   If the event rate drops sharply, the -1 h result is the null's construction;
   if it does not, the threshold is firing on ambient.
3. ~~**More human labels.**~~ **DONE 2026-08-28, and it corroborates the
   pessimistic reading.** Isaac visually reviewed the scenes and produced
   `data/validation/planet_scope_validations.csv` -- 29 labelled overpass
   instants, 44 vessels, and crucially **6 zero-vessel windows**, the labelled
   negatives this record said could not be had at n = 1. Result: correlation
   between manual vessel count and merged event count is **r = +0.20**; the
   detector fired on 18/23 occupied windows and on **4/6 empty ones**.

   This is a SECOND, independent comparison agreeing with the time-shift null:
   event count carries little information about vessel presence. It is not
   explained by a bug -- the frame-shuffle null still collapses to 0, so the
   events are real temporal structure; they are simply not vessel-count
   structure.

   **Do not read 4/6 as a false-positive rate.** Six negatives is far too few,
   and the two instruments do not measure the same thing: the manual count is
   what is visible INSIDE THE IMAGE FOOTPRINT at ONE INSTANT, while the acoustic
   window is +/-15 min and the hydrophone's range is unmeasured (G3) and likely
   exceeds the footprint. An empty scene can be acoustically occupied with no
   error by either instrument -- the `OpticalLabel.area_km2` caveat, now
   load-bearing rather than theoretical.
4. **Shape, not count.** A vessel pass has CPA structure -- rise, peak, fall over
   tens of seconds to minutes. `_plot_band_detail`'s slope and centroid series
   already carry it. A statistic on event SHAPE may separate the overpass window
   from an arbitrary hour where a bare count does not.

# What does not change

* Decision 0027's GO verdict on the `.fft.gz` product. That was a verdict on
  whether the product can support band-level detection at all, and the
  frame-shuffle result strengthens it.
* The detector, the threshold, and the band definitions. Nothing here is a reason
  to edit them, and 0015 forbids re-tuning.
* The claim that levels are uncalibrated counts, never dB re 1 uPa.

# Where this is recorded in the artefacts

`data/derived/events/<run_id>/events_by_window.csv` carries
`n_events_frame_shuffled` and `n_events_time_shifted` on EVERY row, so the count
cannot be read without them, and `events_by_window_summary.md` prints the table
above. A blank in a null column means that null was not evaluable, which is not
the same as zero and must never be summed as one.
