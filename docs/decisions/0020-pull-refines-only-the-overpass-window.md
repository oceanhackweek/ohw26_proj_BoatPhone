# 0020. The B3 pull refines only the overpass window, not the whole calendar

Status: accepted
Date: 2026-08-27
Amends: [0007](0007-onc-400s-that-are-measured-zeros.md),
[0008](0008-empty-listing-is-a-measured-zero.md)
Scope: segment D (refining `data/derived/hydrophone_uptime.csv` from B3's pull), not yet
implemented
Source: `claude/run-phase/run-phase-milestone1-b3-bulk-acquisition.handoff.md` (B3 segment D,
"Open items carried from review", item 5d)

## Context

Decisions 0007 and 0008 state, without a scope qualifier, that where a listing and a positive ONC
response disagree, the more direct evidence wins -- "pull wins" is the informal shorthand carried
forward into B3's plan for segment D, which refines the year-scale uptime calendar
(`hydrophone_uptime.csv`, built from a full-calendar *listing*) using B3's *pull* of actual FFT
products.

The trap, flagged in integration review before segment D was implemented: B3's pull only ever
requests the 2.5-hour PlanetScope overpass window (09:15-11:45 `America/Vancouver`, decision 0017)
per in-season date. Against a full day of 5-minute uptime bins, that window covers roughly 10% of
the calendar's bins. "Pull wins" stated without qualification reads as though the *entire*
`hydrophone_uptime.csv` calendar has now been upgraded from listing-derived to pull-verified. It
has not -- only the ~10% of bins that fall inside an overpass window on an in-season date have any
pull evidence at all. Presenting a window-only refinement as a whole-calendar measurement is
exactly the kind of plausible-looking wrong number CLAUDE.md invariant 5 exists to prevent, and it
would land directly in a figure downstream (uptime is what decides where PlanetScope quota is
spent).

## Decision

**This record scopes decisions 0007 and 0008; it does not reverse them.** Within the window B3
actually pulled, pull evidence still wins over listing evidence, exactly as 0007/0008 state. The
qualifier this record adds is where that precedence applies: only to bins inside a covered
overpass window, never asserted about the calendar as a whole.

Two concrete requirements on segment D's output, both from the handoff and neither yet
implemented:

1. **A refined uptime product must carry a per-bin flag distinguishing MEASURED (covered by the
   B3 pull) from LISTED (uptime known only from the year-scale listing, as before).** A consumer
   reading the refined CSV must be able to tell, per bin, which evidentiary basis it rests on.
2. **The manifest must enumerate the covered UTC spans** -- the actual per-date overpass windows
   the pull touched -- so a downstream reader can reconstruct the ~10% coverage claim without
   re-deriving it from the raw pull.

A window-only refinement must never be presented as, or silently treated as, a whole-calendar
measurement.

This is the governing constraint on segment D's implementation and its checks. Per the handoff,
segment D's check suite must additionally assert: listing-available + pull-absent flips a bin to
unavailable; the reverse (listing-absent, pull-present) **raises** rather than silently upgrading
a bin (a listing saying "no file" while the pull found one is a coverage disagreement worth
surfacing, not smoothing over); refined bins are a strict subset of window bins; unrefined bins
are byte-identical to the A1 CSV; and total availability after refinement is **<=** availability
before refinement (a higher number after refining is a sign the join is broken, not that uptime
improved).

## The join contract, added post-review, 2026-08-27

Segment D's join of the pull's manifest onto `data/derived/hydrophone_uptime.csv` **must be an
INTERVAL-OVERLAP join on UTC time, not an equality join on any field.**

The reason is a real mismatch between the two sides, confirmed against the completed manifest
(`data/derived/pull_overpass_corpus.manifest.json`, 918 dates, 26,666 files):

* The A1 uptime CSV's `start_utc`/`end_utc` are epoch-aligned edges of the fixed 300 s (5-minute)
  grid decision 0020's own context describes.
* The manifest's per-file span fields (currently named `bin_start_utc`/`bin_end_utc` in
  `scripts/pull_overpass_corpus.py`'s `_bin_identity`) are **not** epoch-aligned -- they are derived
  from the real archive filename, and real file starts jitter against the grid. Confirmed example
  from the manifest: `ICLISTENHF1266_20250501T161623.000Z.fft` yields `bin_start_utc =
  2025-05-01T16:16:23+00:00`, 83 seconds off the nearest 300 s grid edge, not exactly on one.

An equality join between these two time fields matches essentially nothing -- not zero, but close
enough to zero to be indistinguishable from "the pull found almost no data" rather than "the join
key is wrong." That reads as low coverage instead of as a broken join, which is precisely the
confusion invariant 9 exists to catch: a plausible-looking wrong number where the honest answer is
"this measurement didn't run." The correct join matches a manifest file's `[bin_start_utc,
bin_end_utc)` interval against the grid bin(s) it overlaps, not against a grid edge it happens to
equal.

**Field rename in progress:** to stop this exact collision at its source -- two same-named-looking
fields (`start_utc`/`end_utc` vs. `bin_start_utc`/`bin_end_utc`) that look like they should be
joined on equality but encode different alignment guarantees -- the manifest's per-file span
fields are being renamed to `file_start_utc`/`file_end_utc`. As of this correction the code still
uses `bin_start_utc`/`bin_end_utc` (see `scripts/pull_overpass_corpus.py` `_bin_identity`); the
rename is planned, not yet landed, and segment D's implementation should not assume the new names
are present without checking.

**Join key to the corpus on disk:** segment D (or any consumer resolving a manifest row to the
actual file on disk) must join on `files[].path`, **not** `filename`. The wire name recorded in
`filename` ends `.fft` for every row (that is the name ONC reports and downloads under), but the
file actually on disk ends `.fft.gz` for every compressed row (decision 0024) -- only the 90
live-probe files predating 0024 are plain-text `.fft` on disk. `path` in each manifest row already
carries the correct on-disk name; `filename` does not, and using it to open a file will fail for
the entire 26,666-file bulk-pull population.

## Consequences

* Any code or figure that reads `hydrophone_uptime.csv` after segment D must check the per-bin
  MEASURED/LISTED flag before treating a bin's availability as pull-verified, not just
  listing-derived.
* Decisions 0007 and 0008 remain correct as stated; this record does not change their content,
  only clarifies that their "pull wins" precedence is scoped to the bins the pull actually
  touched.
* Ambiguity not resolved by this record, carried from the handoff as-is: the exact mechanism for
  "enumerate the covered UTC spans" in the manifest (e.g. one row per date-window vs. a merged
  span list) is left to segment D's implementation; this record only requires that such an
  enumeration exist and be sufficient to reconstruct the coverage fraction.
* Segment D must use an interval-overlap join per the section above, not equality, and must
  resolve on-disk files via `path`, not `filename`, per the same section.
