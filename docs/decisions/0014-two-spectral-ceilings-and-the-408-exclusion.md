# 0014. Two spectral ceilings, kept apart, and nothing above bin 408 enters B5

Status: accepted
Date: 2026-08-27

## Context

The product's 512 columns are not uniformly usable. B0-2a's original single assertion --
"structural zero above bin 419" -- was wrong at both ends: it understated the true hard-zero
region and mislabelled a filter skirt as a boundary. B1a re-measured all 512 columns over both
sample fixtures (2,400 frames, 1,228,800 cells) and found **three** distinct regions, and the
project needs two separate ceilings, not one, because "we have data" and "we can put a dB number
on it" are different claims.

## Decision

**Two ceilings, kept apart in `boatphone/config.py`:**

- **Calibrated range: bins 1-204 (250 Hz - 51,000 Hz)** -- `FFT_CALIBRATED_BIN_RANGE`,
  `FFT_B5_CALIBRATED_CEILING_BIN = 204`. The supplied sensitivity/calibration file's own header
  states 10 Hz - 51,200 Hz; bins 1-204 are the bins WHOLLY INSIDE that span. Bin 0 (0 Hz, below
  the file's 10 Hz floor) and bin 205 (51,250 Hz, above its 51,200 Hz ceiling) are excluded rather
  than admitted by extrapolating the sensitivity curve -- see the correction note below. Above
  bin 204, no dB re 1 uPa claim is defensible from what we hold.
- **Uncalibrated / relative ceiling: bin 408 (~102 kHz)** -- `FFT_B5_RELATIVE_CEILING_BIN`, set at
  `FFT_ROLLOFF_ONSET_BIN`. **Nothing above bin 408 enters any B5 statistic, calibrated or not.**
  Bins 419-424 (the anti-alias roll-off skirt) are excluded even though they are formally "below"
  the hard-zero boundary at 425, and bins 425-511 are excluded as a hard structural zero.

### The measured three-region split (both fixtures, 2,400 frames)

| Region | Constant | Measured | Assertion |
|---|---|---|---|
| cols 425-511 | `FFT_STRUCTURAL_ZERO_COLS_HIGH` | 0 of 208,800 cells nonzero, on **both** fixtures | hard exact zero; any nonzero raises (reader/format failure) |
| cols 419-424 | `FFT_ROLLOFF_TAIL_COLS` | 74/7,200 and 68/7,200 nonzero, max 6 and 5 | bounded (max<=10, mean<=0.05) and per-bin mean over bins 405-424 non-increasing |
| col 0 | `FFT_DC_COL` | 14/1,200 and 8/1,200 nonzero, max 3 and 2, occupancy 1.17%/0.67% | bounded (max<=5, nonzero fraction<=0.02) -- **not** exactly zero |

The previous single constant, `(419, 511)`, was **off by six columns**. Bins 419-424 are not a
separate region at all -- they are the tail of one continuous anti-alias filter skirt whose
per-bin mean runs smoothly from ~9.8 at bin 405 through 0.28 at bin 418 to ~0.001 at bin 424, with
no discontinuity at 419. The monotonicity assertion over bins 405-424 is the one that actually
catches a reader bug: a mis-strided or wrapped row moves a bin mean by order 1, which a bare
cell-count bound would not notice. (One correction carried into `config.py`'s comment: the tail is
not *strictly* monotonic -- on fixture `...000004`, bin 423 has mean 0.0000 and bin 424 has mean
0.0008, i.e. one count in one frame out of 1200. At the far end of the skirt the mean is quantised
to multiples of 1/1200 and its ordering there is integer noise, not physics, so the monotonicity
assertion carries an explicit tolerance, `FFT_ROLLOFF_MONOTONIC_TOL_LEVEL = 1/1200`, three orders
of magnitude below the order-1 step a real stride bug produces.)

### Why bin 408, not bin 425, is where B5 stops (the decisive reason is the third)

1. **Bins 409-424 are instrument response, not ocean.** This is the anti-alias filter's own
   roll-off, not a measurement of anything outside the hydrophone.
2. **The shoulder onset itself is structural, not evidence of a signal.** `FFT_ROLLOFF_ONSET_BIN`
   sits at `0.4 * 1024 = 409.6` for *any* sample rate, so its position confirms only that the
   reader has the right stride, not that anything real lives just below it.
3. **409-418 is real filter skirt, not censoring; 419-424 is where floor-censoring takes over.**
   Re-measured at-floor fractions, split at the actual boundary: bins 409-424 as a whole are only
   ~49.35%/49.19% at the 0 floor -- roughly half nonzero, i.e. a real, gradually decaying filter
   skirt, not a censored region. Bins 419-424 alone are ~99.06%/98.97% at floor, and 419-511
   together are 99.94%/99.93% at floor (the figure this section previously misattributed to
   409-424 as a whole). So the floor-censoring argument is decisive for 419-424 and above, but it
   is NOT what excludes 409-418 -- that exclusion rests on reasons 1 and 2 (instrument response,
   structural roll-off onset), not on censoring. Averaging 419 and above converts "we cannot
   measure this" into a number that looks like a measurement; averaging 409-418 would instead
   average real (if uninterpretable) filter-skirt signal, which reasons 1-2 already forbid on
   their own.

## Alternatives rejected

- **Keep the single `(419, 511)` structural-zero constant.** Rejected: measured wrong at both
  ends -- it treats 419-424 as zero when up to 6 counts per cell occur there, and (separately) it
  understates how hard the true zero at 425-511 is (0 of 208,800 cells, not "almost zero").
- **Use bin 425 (the hard-zero boundary) as B5's ceiling, since that is where data is literally
  absent.** Rejected: 419-424 technically carries nonzero data and is ~99% floor-censored
  instrument roll-off, not signal, and including it biases any band mean that spans it. But that
  floor-censoring argument alone would only justify stopping at 419, not 408; reaching back to
  408 rests on 409-418 being real filter skirt (reasons 1-2), which floor-censoring does not
  cover -- see the correction above.
- **Use the calibrated ceiling (205) as the single ceiling for all B5 work.** Rejected: B5's
  band-split / far-field / null-check triad (decision 0012's replacement for detector agreement)
  needs relative, uncalibrated levels above 51.2 kHz for cross-checks that do not require absolute
  dB; collapsing to one ceiling would discard that entirely rather than just marking it uncalibrated.

## Consequences

* `boatphone/config.py` carries both ceilings as separate named constants
  (`FFT_B5_CALIBRATED_CEILING_BIN = 204`, `FFT_B5_RELATIVE_CEILING_BIN = 408`), plus
  `FFT_STRUCTURAL_ZERO_COLS_HIGH = (425, 511)`, `FFT_ROLLOFF_TAIL_COLS = (419, 424)`,
  `FFT_ROLLOFF_ONSET_BIN = 408`, `FFT_DC_COL = 0`, and `FFT_ROLLOFF_MONOTONIC_TOL_LEVEL`.
* Any B5 band level whose support extends above bin 408 is a bug, not a design choice; the
  boundary must be enforced before the first band level is produced, or the first B5 output
  silently averages censored roll-off cells.
* `fft_io.structural_zero_report()` surfaces the per-region counts so downstream code can assert
  against them rather than re-deriving the split.
* A mutation check (column-major read, +1-bin roll, hump shuffle) confirmed these assertions catch
  real reader bugs: a column-major misread produces 84,690 nonzero cells in the hard-zero block and
  fails monotonicity (caught twice); a +1-bin roll produces exactly 1 nonzero cell in 425-511,
  caught by the exact-zero check.
* **The DC-column and roll-off bounds (`FFT_DC_COL`, `FFT_ROLLOFF_TAIL_COLS`) are tuned on two
  CONTIGUOUS 5-minute fixtures from ONE hour of ONE deployment**, not two independent samples --
  the fixtures are back-to-back (`...235504` ends exactly where `...000004` begins, gap 0.0 s).
  `boatphone/config.py`'s comments state these were "measured on both local fixtures" without
  saying the fixtures are adjacent windows from the same hour; the "how this would show up as
  wrong" note above addresses a *different* scenario (a future differently-configured product) and
  does not cover ordinary within-deployment variability across seasons or sea states. Treat the
  numeric bounds (max counts, mean thresholds) as a description of one hour of one deployment with
  roughly 2x headroom, not a population estimate, until a wider sample is pulled.
* **`FFT_ECHOSOUNDER_ABS_CENTRE_HZ` is pinned in config but never auto-asserted against the WAV.**
  The 37,650 Hz figure was measured absolutely on the 128 kHz sample WAV (B1a), which is
  documented, but no `check_b0_2a_*` re-reads that WAV to re-verify it -- the check suite is
  data-optional and the WAV read is ~115 MB, so this number is currently pinned-and-documented but
  not continuously re-verified. Add a WAV-backed check when B1b next touches the WAV file anyway.

## How this would show up as wrong

If a future higher-bit-depth or differently-configured product shows genuine nonzero energy above
bin 408 that is not roll-off (e.g. a different anti-alias design, or recalibration extending past
51.2 kHz), this ceiling needs revisiting -- the 408 boundary is specific to this ICLISTEN HF1266
product's filter design, not a universal fact about the instrument.

Related: `docs/decisions/0010-analysis-band-and-small-craft.md` (calibrated support, defined
independently); `docs/decisions/0013-fft-axis-convention-is-an-open-assumption.md` (bin 408's
absolute frequency inherits the +-125 Hz uncertainty); `docs/plans/acoustics_plan_v2.md` §3.3.

## [CORRECTION, 2026-08-27, integration review-2]

The original "decisive" reason 3 cited "99.94% of cells in bins 419-424 sit at the 0 floor" as
the justification for the whole 409-424 exclusion. That figure was measured over bins **419-511**,
not 409-424, and the two are not interchangeable: re-measured at-floor fractions are
409-424 = 49.35%/49.19% (roughly half nonzero -- a real, gradually decaying filter skirt, not a
censored region), 419-424 alone = 99.06%/98.97%, and 419-511 = 99.94%/99.93%. The document was
internally inconsistent -- it prints the counts (74/7,200 = 98.97%, not 99.94%) that contradict
its own headline number two paragraphs later. `FFT_ROLLOFF_ONSET_BIN = 408` is unaffected and
stands on reasons 1 and 2 (instrument response; the `0.4 * 1024` structural argument), neither of
which depended on the mis-scoped percentage. Corrected text is inline above (reason 3 and the
"Alternatives rejected" bin-425 entry); no constant changed. Also touched for the same error:
`boatphone/config.py:502`, `docs/plans/acoustics_plan_v2.md` §3.3, `scripts/checks.py` (the
`check_b0_2a_*` docstring/comment near line 6130).

## [CORRECTION, 2026-08-27, integration review-2 [MEDIUM 2]]

`assert_calibratable` (`boatphone/fft_io.py`) gated on `FFT_CALIBRATED_BIN_RANGE = (0, 205)`,
i.e. `calibrated_band_hz() == (0.0, 51250.0)` Hz. That admits DC-10 Hz at the bottom (bin 0 is the
near-zero DC column, inside the unexplained 42 dB low-frequency deficit) and extrapolates 50 Hz
past the calibration file's own stated 51,200 Hz top -- up to 175 Hz once 0013's +-125 Hz axis
uncertainty is carried. This is the function a cross-team caller (the optical side) would use to
ask "can I state a dB here?", so its permissiveness matters beyond acoustics.

**Chosen fix: narrow, not document-and-keep.** `FFT_CALIBRATED_BIN_RANGE` is now `(1, 204)` --
250 Hz to 51,000 Hz, wholly inside the file's documented 10 Hz - 51,200 Hz span, no extrapolation
at either edge. `FFT_B5_CALIBRATED_CEILING_BIN` moves from 205 to 204. Chosen over the
keep-and-document-extrapolation alternative because the call site needs no caveat this way, and a
cross-team caller with no context on the axis-uncertainty discussion in 0013 should not have to
reason about it to get a safe answer. No assertion was weakened: `assert_calibratable` still
rejects any band reaching outside the (now smaller) support, and the checks that exercised it
(`scripts/checks.py` `check_b0_2a_calibratable_band_matches_bin_range_and_assert_calibratable_rejects_beyond`
and its independent hand-transcribed `B0_2A_CALIBRATED_BIN_RANGE`) were updated to the same
`(1, 204)` and continue to pass.
