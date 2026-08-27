# 0015. B5's thresholds must be censoring-aware, and its free parameters are tuned off the overpasses

Status: accepted
Date: 2026-08-27

## Context

Decision `0012` promoted B5 (band-level SPL, no weights) from cross-check to primary detector
after the pretrained-model path was closed. B5's input is `levels_db`, the product's own
uncalibrated integer scale, and B1a established that the **lower end** of this scale is censored:
18.7% of cells sit exactly at the 0 floor, far too dense a pile-up to be ordinary quantisation. The
**upper end (86) is treated as an assumed, not confirmed, ceiling** -- see the correction below. A
detector built on censored data that ignores the censoring will silently convert "off-scale" into
"small" or "large" rather than "unknown", exactly at the extremes where the signal of interest -- a
vessel close-passing the hydrophone -- lives; that risk is real regardless of whether the upper
end turns out to be a hard clip.

## Decision

**Levels are on the product's own uncalibrated integer scale, observed 0-86 in the local sample.
The floor (0) is confirmed censoring; the ceiling (86) is an assumption. B5 must still treat both
ends as missing data with a known direction, not as ordinary measurement noise:**

1. **Upper "censoring" is an assumption, not a confirmed measurement.** Three cells sit at 86 in
   bins 140-165 of fixture `...000004` (two frames), on a **quiet ambient** window with no known
   vessel event. **[CORRECTION, 2026-08-27, pre-merge quality review]** This was originally stated
   as measured, confirmed censoring; it is not. The count tail argues against a hard clip: 84/85/86
   counts are 8/3/3 on this fixture, and the *other* fixture tops out at 84 with zero cells at 85
   or 86 -- a smooth decay, not the pile-up a real ceiling clip produces (contrast the floor, whose
   18.7%-of-cells pile-up is unambiguous). Ten quiet minutes cannot distinguish "the scale
   ceilings at 86" from "the loudest thing observed happened to reach 86". Kept as a conservative
   working assumption (`FFT_LEVEL_CEILING = 86`) -- treating a window that reaches 86 as a
   possible lower bound is the safe direction even if the true ceiling turns out higher -- with a
   named resolution path: a loud window from B3's corpus pull (a close vessel pass) settles it
   either way. A close vessel pass, if the assumption holds, will clip far harder than this quiet
   window did; any band level computed from a window touching 86 should still be treated as a
   **possible lower bound**, not a confirmed measurement, until the assumption is resolved.
2. **B5 reports per-window counts at the 0 floor and the 86 ceiling alongside every band level.**
   `fft_io.censoring_report()` supplies these counts; a band level without its accompanying
   censoring counts is incomplete.
3. **Any threshold or regression built over these levels must be censoring-aware.** In particular,
   a **rolling-percentile ambient baseline computed over censored values is biased toward the
   censoring point, and that bias moves with ambient conditions** -- louder ambient periods push
   more cells toward the ceiling, quieter periods push more toward the floor. This means the bias
   is **confounded with exactly the thing B5 is trying to detect** (elevated levels from vessel
   presence), not an independent nuisance that averages out.
4. **B5's excess-over-ambient threshold and minimum-duration constant are the single most
   consequential free parameter left in the project** (per the ledger, following the NO-GO on the
   pretrained-model path in decision 0012 -- with no learned decision boundary anywhere in the
   pipeline, this hand-set threshold now carries the entire precision/recall trade-off). They must
   be **tuned on published practice and the null test, not on the optical overpasses.** The
   project has only ~20-30 optical overpasses; if the threshold is fit to them, there is no
   remaining independent data to evaluate it against, and the overpass-based accuracy figure
   becomes a training-set number reported as a test-set number.

## Evidence

- Censoring counts: 3 cells at the ceiling of 86 on a quiet window (bins 140-165, fixture
  `...000004`); B1a's synthetic-data mutation check confirmed `censoring_report()` returns 0/0 on
  genuinely uncensored data, so the fixture counts are measurements, not a stuck counter.
- `alt-3` (toolkit survey) independently converged on the same requirement from the literature
  side: COMPASS MPA's published construct (RL > 97 dB re 1 uPa in the 2 kHz decidecade, with
  ambient at least 6 dB below) is the same excess-over-ambient-with-margin design B5 uses,
  supporting "anchor on published practice" as achievable rather than aspirational.
- Boats and sea state both follow the diurnal cycle, so a naive threshold tuned only on daytime
  overpasses would be confounded with time-of-day regardless of the censoring issue -- this is why
  the null test (a one-hour time shift) is required in addition to, not instead of, the
  censoring-aware design.

## Alternatives rejected

- **Fill or clip-correct censored cells (e.g. treat 86 as "some large unknown value" via
  extrapolation).** Rejected under invariant 5 (errors surface): a censored cell's true value is
  unknown by construction; extrapolating it manufactures a number where none exists.
- **Tune the threshold directly on the ~20-30 optical overpasses via cross-validation within that
  set.** Rejected: the unit of independence here is the *overpass*, not the sample, and with only
  ~20-30 of them there is no realistic way to both fit and evaluate a threshold without leakage.
  The overpasses are needed once, as a final independent read, not as a tuning set.
- **Ignore censoring and treat `levels_db` as an ordinary bounded measurement.** Rejected: the bias
  it introduces is directional and moves with ambient noise, i.e. it is confounded with the signal
  of interest, not a symmetric nuisance a large-sample argument would average away.

## Consequences

* Every B5 band-level output must be paired with its floor/ceiling counts; a band level reported
  without them is not spec-compliant.
* The rolling-percentile ambient baseline (planned for B5) must either exclude censored cells from
  its percentile computation or explicitly bound the resulting bias direction and cite it -- silently
  computing a percentile over `[0, 86]`-clipped values is disallowed.
* The threshold and minimum-duration constant get tuned against published reference levels (e.g.
  the COMPASS MPA construct) and the null test (label shuffle, one-hour time shift, quiet-period
  check per invariant 4), not against the overpass set. The overpass set is reserved for a single
  final evaluation pass.
* This threshold and duration constant should be named in `boatphone/config.py` with their source
  cited (published value, or the null-test result that set them), consistent with invariant 6 --
  they are exactly the kind of magic number this project has committed to not burying.

## How this would show up as wrong

If the final overpass-based read shows the threshold badly miscalibrated (e.g. detecting on every
overpass regardless of vessel presence, or missing all of them), that is evidence the
published-practice anchor does not transfer to this hydrophone/deployment, and the threshold needs
re-derivation -- but that finding must come from the reserved final read, not from iterating the
threshold against the overpasses themselves, or the finding cannot be trusted.

Related: `docs/decisions/0012-b0-model-viability-outcome.md` (why B5 is now primary, and the
consequence that names this parameter); `docs/decisions/0014-two-spectral-ceilings-and-the-408-exclusion.md`
(the top-of-band exclusion this threshold must also respect); `docs/plans/acoustics_plan_v2.md` §3.3.
