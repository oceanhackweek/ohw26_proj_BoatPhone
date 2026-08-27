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

- **Calibrated ceiling: bin 205 (51.2 kHz)** -- `FFT_B5_CALIBRATED_CEILING_BIN`. The supplied
  sensitivity/calibration file's own header stops at 51.2 kHz; above this, no dB re 1 uPa claim is
  defensible from what we hold.
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
3. **Decisive: the region is floor-censored.** 99.94% of cells in bins 419-424 sit at the 0 floor
   (from the 74/7,200 and 68/7,200 nonzero counts above). Any mean computed over this region is a
   censoring artefact, biased upward by an unboundable amount, because the true values below the
   floor are unknown by construction. Averaging it converts "we cannot measure this" into a number
   that looks like a measurement. This reasoning is what pushes the exclusion boundary down to 408
   rather than the structural hard-zero boundary at 425 -- a naive reader would keep using
   419-424 right up until the true zero region and get a systematically biased mean for it.

## Alternatives rejected

- **Keep the single `(419, 511)` structural-zero constant.** Rejected: measured wrong at both
  ends -- it treats 419-424 as zero when up to 6 counts per cell occur there, and (separately) it
  understates how hard the true zero at 425-511 is (0 of 208,800 cells, not "almost zero").
- **Use bin 425 (the hard-zero boundary) as B5's ceiling, since that is where data is literally
  absent.** Rejected: 419-424 technically carries nonzero data, but it is 99.94% floor-censored
  instrument roll-off, not signal, and including it biases any band mean that spans it. Reason 3
  above is why the exclusion has to reach back to 408, not just to 425.
- **Use the calibrated ceiling (205) as the single ceiling for all B5 work.** Rejected: B5's
  band-split / far-field / null-check triad (decision 0012's replacement for detector agreement)
  needs relative, uncalibrated levels above 51.2 kHz for cross-checks that do not require absolute
  dB; collapsing to one ceiling would discard that entirely rather than just marking it uncalibrated.

## Consequences

* `boatphone/config.py` carries both ceilings as separate named constants
  (`FFT_B5_CALIBRATED_CEILING_BIN = 205`, `FFT_B5_RELATIVE_CEILING_BIN = 408`), plus
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

## How this would show up as wrong

If a future higher-bit-depth or differently-configured product shows genuine nonzero energy above
bin 408 that is not roll-off (e.g. a different anti-alias design, or recalibration extending past
51.2 kHz), this ceiling needs revisiting -- the 408 boundary is specific to this ICLISTEN HF1266
product's filter design, not a universal fact about the instrument.

Related: `docs/decisions/0010-analysis-band-and-small-craft.md` (calibrated support, defined
independently); `docs/decisions/0013-fft-axis-convention-is-an-open-assumption.md` (bin 408's
absolute frequency inherits the +-125 Hz uncertainty); `docs/plans/acoustics_plan_v2.md` §3.3.
