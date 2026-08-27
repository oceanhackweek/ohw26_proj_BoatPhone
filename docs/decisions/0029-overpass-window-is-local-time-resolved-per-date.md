# 0017. The overpass window is local time, resolved per date

Status: accepted
Date: 2026-08-27
Scope: `boatphone/config.py`, `boatphone/acquire.py`, `scripts/pull_overpass_corpus.py`
Source: `claude/run-phase/run-phase-milestone1-b3-bulk-acquisition.handoff.md` (B3 segment A,
"Contracts already settled")

## Context

B3 needs a per-day window over which to pull FFT products so they line up with the PlanetScope
overpass. The window is defined as clock time at the hydrophone's shore location, not as a fixed
UTC span: PlanetScope overpasses at a stable local time of day, and the study is asking "what did
the ocean sound like during the overpass," not "what did it sound like during this fixed UTC
interval."

The naive implementation -- compute the UTC pair once (from one date, or from a hand-picked
offset) and reuse it for every date in a six-season pull -- is wrong for exactly half the year.
`America/Vancouver` observes PST (UTC-8) and PDT (UTC-7), and the DST transition dates are not
fixed calendar days. A single cached UTC pair is correct only until the next transition, silently
wrong for every date on the other side of it, and gives no error to notice by.

This is precisely the class of bug CLAUDE.md invariant 3 and decision 0002 exist to prevent, and
segment A of B3 was written specifically to close it.

## Decision

The overpass window is defined once, in local civil time: `09:15-11:45 America/Vancouver`, as
named constants in `boatphone/config.py`. It is resolved to a UTC pair **per date**, using
`zoneinfo`, in `boatphone/acquire.py` -- never cached or computed once for a run. Every date in
the corpus pull (2020-2025 in-season) gets its own local-to-UTC resolution at the point it is
used.

`check_b3c_2` pins this against a real transition: the 2024-03-09 -> 2024-03-10 PST -> PDT
boundary, where the resolved UTC window for the two adjacent dates must differ by exactly one
hour.

**This is a cross-team convention, not an acoustics-internal one.** Malachy's Planet imagery
ordering must resolve the overpass window the same way -- same local times, same per-date
`zoneinfo` resolution -- or the optical-acoustic matchup joins two different definitions of "the
overpass window" and the join silently produces a wrong answer (CLAUDE.md invariant 6). As of this
record the convention lives only in `boatphone/config.py`/`boatphone/acquire.py` and in this
record; it has not been confirmed as mirrored on the Planet-acquisition side.

## Consequences

* Any future code that touches the overpass window must resolve it per date via `zoneinfo`
  against the constants in `boatphone/config.py`. Introducing a second UTC-window computation
  (e.g. a hardcoded offset, or a once-per-run cache) reopens the DST bug this record exists to
  document.
* `check_b3c_2` is the regression guard; it must keep pinning a real DST-boundary date pair, not a
  synthetic one, so a `zoneinfo` misconfiguration or tzdata mismatch is caught against the actual
  calendar.
* Open question, not settled by this record: whether Malachy's Planet ordering pipeline currently
  uses the same local-time definition and per-date resolution. This needs to be confirmed
  cross-team before B3's corpus and Planet's imagery are treated as covering the same instants.
