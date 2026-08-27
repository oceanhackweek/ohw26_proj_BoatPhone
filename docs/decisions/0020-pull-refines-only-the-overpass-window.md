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
