# `data/validation/planet_scope_validations.csv` -- human vessel counts

**Tracked in git, unlike most of `data/`.** Same reasoning as `data/labels/`:
hand-made, tiny, and **not reproducible from anything else in the repo**. If it is
lost, someone has to look at the imagery again. `data/raw/`, `data/interim/`,
`data/derived/` and `data/processed/` are ignored because they regenerate; this
does not.

(This file lives in `docs/` rather than beside the CSV because the raw-data
immutability hook, decision 0001, refuses writes under `data/`.)

| column | meaning |
|---|---|
| `timestamp` | `YYYYMMDD_HHMM` -- the first 13 characters of a PlanetScope `scene_id` |
| `vessel count` | vessels counted BY EYE in the scene |

29 rows, 44 vessels, **6 zero-vessel windows**. Reviewer: Isaac Guld, 2026-08-28.

## The join is minute-resolution and NOT one-to-one

`scene_id`s carry seconds; this file keys on minutes. Two scenes in the gate-2 set
share a prefix -- `20210616_183111_05_2448` and `20210616_183140_94_2436`, 29 s
apart, adjacent frames of the same overpass. **One label legitimately covers both.**

Consequence: **do not sum `vessel count` across scene rows.** The corpus total is
over the 29 distinct labelled instants, not the 30 scene rows, or those 2 vessels
are counted twice. `build_events_table.load_manual_vessel_counts()` carries
`manual_label_shared_by` on every affected row for exactly this reason, and
asserts BOTH directions of the join -- an unmatched label or an unlabelled scene
raises rather than being silently dropped.

## What this count is, and what it is not

It is **what a human can see inside the image footprint at one instant.**

It is **not** a statement about what the hydrophone could hear. The acoustic window
is +/-15 min and the hydrophone's detection range is unmeasured -- that is goal G3,
still open -- and very likely exceeds the image footprint. So:

* A **zero-vessel** scene is NOT an acoustic negative. A boat out of frame, or
  arriving ten minutes later, occupies the acoustic window with no error by either
  instrument.
* A **vessel-present** scene does not guarantee an audible source. A visible vessel
  may be anchored, drifting, or too distant to hear.

This is the `boatphone.overpasses.OpticalLabel.area_km2` caveat, and here it is
load-bearing rather than theoretical: **this file records no reviewed area**, so the
region examined is not stated and cannot be recovered. Treat the counts as
scene-footprint counts and nothing narrower.

## Measured against the acoustics

29 labelled instants against the B5 band-level detector
(`scripts/build_events_table.py`): correlation between vessel count and merged event
count **r = +0.20**; the detector fired on 18/23 occupied windows and on 4/6 empty
ones. **4/6 is not a false-positive rate** -- six negatives support no rate, and the
footprint/window mismatch above means the two instruments are not observing the same
region or the same instant. See
`docs/decisions/0030-time-shift-null-does-not-collapse.md`.

## Related

* `data/labels/optical_vessel_labels.csv` -- the earlier single label, which DOES
  carry `area_km2`. Kept separate: different schema, stricter contract, not a subset.
* `contributor_folders/malachymcc/planet_folger/detections.csv` -- automated optical
  detection CANDIDATES (32,877 rows, ~47% flagged `transient`). Not vessel counts,
  and not validated by this file.
