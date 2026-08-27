# 0011. Level comparability is declared at every pipeline boundary

Status: accepted
Date: 2026-08-27

## Context

Decision 0002 requires the time base, sample rate and units to be stated at every boundary. Working
on the VTUAD comparison exposed a gap it does not cover: **whether two sets of levels are
comparable at all.**

Folger `.fft.gz` levels are calibrated, dB re 1 uPa, once the sensitivity file is applied. VTUAD
audio is uncalibrated raw PCM in arbitrary counts, possibly per-segment normalised. ONC's
pretrained checkpoint (decision 0009) consumes min-max normalised spectrograms, so its inputs carry
no absolute level either.

Band-matching two such sources makes their *frequency axes* comparable and does nothing about their
*level* axes. An unknown fixed gain offset between two domains produces a difference that looks
exactly like a domain shift, a detection-threshold difference, or a real acoustic finding. Nothing
raises. You get a plausible number, which is the failure mode CLAUDE.md invariant 5 is about.

This is not hypothetical: it is precisely the artefact that would have contaminated the VTUAD
transfer gap, and it was not identified anywhere in the original plan.

## Decision

**Any function that compares, joins or transfers levels between two sources takes an explicit
declaration of each source's calibration state and feature kind. There are no defaults.**

Implemented in `boatphone/models.py` as the `CalibrationState` and `FeatureKind` enums, with
`assert_comparable(...)` as the boundary check. Both arguments are **mandatory** -- a caller that
has not thought about it cannot silently get the permissive case.

The rules `assert_comparable` enforces:

* **Calibrated vs uncalibrated may never be compared on an absolute-level feature.** Raises.
* **Level-invariant features are always comparable** across calibration states -- spectral slope,
  band ratios, shape descriptors, level *changes* through closest point of approach, and any
  per-clip normalised representation. These carry no absolute level, so no gain offset survives.
* **Band-matching is necessary but not sufficient.** `assert_band_matched` and `assert_comparable`
  are separate checks and both must pass before a cross-source comparison.

**Corollary for cross-domain work:** use **level-invariant features only**, unless both sides are
calibrated against a known sensitivity. Per-clip level normalisation before any embedding is the
default, not an option.

## Consequences

* **Every new comparison function costs two extra arguments.** That is the point -- the cost is the
  moment of thought, and it is small next to a wrong headline number.
* **Some comparisons are simply refused.** An absolute-level comparison between Folger and any
  uncalibrated corpus raises rather than returning a value. If that blocks something we want, the
  fix is a level-invariant feature, never a suppressed check.
* **Absolute-level work stays inside bins 1-205** (decision 0010) and inside a single calibrated
  source.
* **The ONC checkpoint's normalised input is safe by construction** -- its scores are not levels, so
  no calibration claim attaches to them. Do not report an `Engine Noise` score in dB.
* **How to tell this was wrong:** if `assert_comparable` fires often on comparisons that turn out to
  be legitimate, the enum granularity is too coarse and should be refined -- but loosen it by adding
  a state, never by adding a default.

`boatphone/models.py` carries 40 checks covering this, mutation-tested against eight deliberate
sabotages (disabled guards, silent clipping, no-op `band_limit`), each of which produced a failure.

Extends `0002-time-alignment-and-units.md`. Related: `0009`, `0010`.
