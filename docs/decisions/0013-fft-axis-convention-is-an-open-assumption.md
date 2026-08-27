# 0013. The FFT axis convention is an assumption, not a settled fact, worth +-125 Hz

Status: accepted
Date: 2026-08-27

## Context

Every band edge in the project -- calibration, the analysis band, the echosounder landmark, B5's
band levels -- is defined against `boatphone/fft_io.py`'s frequency axis. B1a (2026-08-27) set out
to adjudicate whether the product's 512 columns are bin **centres** (`f_k = k * 250 Hz`) or bin
**edges** (bin *k* spans `[k*250, (k+1)*250)`, effective centre `(k+0.5)*250 Hz`). It did not
settle it. This record exists so that "we pinned `centre`" is never mistaken for "we know it is
`centre`" -- the two look identical in the code and are not identical in truth.

## Decision

**`FFT_AXIS_CONVENTION = "centre"` stays pinned in `boatphone/config.py` -- the reader must be
deterministic -- but it is recorded here as a named assumption, not a confirmed fact, carrying
`FFT_AXIS_OFFSET_UNCERTAINTY_HZ = 125.0` (one-sided, toward higher frequency). Every band-edge
consumer widens its support by this uncertainty rather than treating the axis as exact.**

### The evidence, both ways

*For edge*: a narrow, reproducible spur straddles bins 399/400, with a power ratio
bin400/bin399 of **0.931** and **0.990** on two fixtures five minutes apart -- a stable ~5 dB
excess, the signature of a real tone sitting almost exactly halfway between two bin **centres**.
Under the edge convention that midpoint is the round **100.000 kHz**; under centre it is
**99.875 kHz**.

*But that is a roundness prior, not a measurement.* Of six narrowband lines found in the product
(the 399/400 spur plus five others at 366.48, 314.9, 131-134, 171-175, and 184-187 kHz-equivalent),
the other five are round under **neither** convention. There is no plausible 100 kHz source at
Folger: the AZFP on site runs 38/67/125/200/455/769 kHz, and the ADCPs run 300/600/1200 kHz. A
100 kHz artefact from a switching-supply or internal clock is equally consistent with the data and
need not be acoustically round at all.

*Structural argument for centre*: a 1024-point real FFT produces 513 bins (DC through Nyquist). A
512-column product is most naturally that set with the Nyquist bin dropped, i.e. `k = 0..511` at
centres `k * 250 Hz` -- which makes column 0 the DC bin and explains its near-exact zero (measured
max 3 counts, 1.17% nonzero occupancy, mean 0.016 vs. column 1's mean 16.4) far more economically
than positing a high-pass filter that happens to land exactly on DC.

*Structural argument against centre (for edge)*: 512 bins of 250 Hz tile `[0, 128000)` exactly
under the edge convention (a natural filterbank layout), and the ~42 dB low-frequency deficit
across 250 Hz -> 2 kHz shows a low-frequency filter **does** exist in this signal chain -- so
positing a further high-pass explanation for column 0 is not implausible on its face.

*The WAV cross-check is circular on B1 and cannot settle it.* The 128 kHz sample WAV gives an
independent absolute frequency reference, but converting its measured centroid into an implied
bin offset requires assuming a counts-to-dB scale that B1 has not yet established. Sweeping that
assumed scale swings the answer across both hypotheses:

| Assumed dB/count | Implied offset (bins) |
|---|---|
| 0.25 | -0.10 |
| 0.5 | +0.07 |
| 1.0 | +0.37 |
| 1.5 | +0.60 |
| 2.0 | +0.76 |
| 3.0 | +0.89 |

A non-overlapping fixture gives +0.167, a further +-0.2 bin of drift on top. A global multi-band
shift fit is unidentifiable (residual 0.29-0.33 dB, a flat objective, nonsense results on smooth
bands) and is recorded only because it was the obvious next attempt and it failed informatively.

*The echosounder hump is also not evidence.* Under edge the measured centroid reads 37.70 kHz;
under centre, 37.58 kHz. Which is closer to the true 37,650 +- 150 Hz source centre flips with the
choice of background model and dB scale, so it cannot discriminate the +-125 Hz question -- only
the much coarser question of a 2x mapping error, where it is decisive (it discriminates by ~150
bins).

## Alternatives rejected

- **Pin to edge on the roundness prior alone.** Rejected: a single coincidence out of six lines,
  none of which round under either convention, is not evidence strong enough to move a project-wide
  constant, and doing so would remove the visible uncertainty the code now carries deliberately.
- **Leave the axis unpinned / raise on every band-edge call.** Rejected: the reader must return a
  deterministic axis for `fft_io` to be usable at all; the uncertainty is carried as an explicit
  widening parameter instead of blocking every consumer.
- **Bake the 125 Hz widening into `models.band_limit` unconditionally.** Rejected -- see
  Consequences.

## Consequences

* Every band-edge consumer widens by `FFT_AXIS_OFFSET_UNCERTAINTY_HZ`. `boatphone/models.py`'s
  `band_limit` takes `axis_offset_uncertainty_hz` as an explicit keyword, defaulting to **0.0**
  rather than 125.0. This is deliberate, not an oversight: baking 125 Hz in unconditionally would
  wrongly widen WAV spectra, whose `rfftfreq` axis is exact by construction and carries no such
  uncertainty, and would silently loosen the existing A8b out-of-band checks that assume an exact
  axis. `boatphone/fft_io.py`'s `band_limit_product` supplies the 125 Hz on the product path only.
* No check in `scripts/checks.py` may assert a bin position tighter than +-1 bin until this
  question is settled.
* `check_b0_2a_axis_uncertainty_is_carried` asserts both that the constant is positive and that
  `band_limit` actually widens support when given it, so a later edit cannot silently drop the
  open question.
* **Known residual weakness, recorded rather than hidden**: this is enforced by a single guard
  check. A direct call to `models.band_limit` on product-derived data at a *new* call site, without
  passing `axis_offset_uncertainty_hz`, silently gets the narrow (0.0-uncertainty) behaviour. The
  guard catches deletion of the constant; it does not catch misuse at a new call site. Anyone
  adding a new product-band consumer must pass the uncertainty explicitly.

## How to resolve, in order

1. **Email ONC for the product's frequency-axis definition.** One sentence settles this for free;
   already queued as a B1 action item regardless of this record.
2. **A scale-free two-bin-split census over the B3 corpus.** Histogram the sub-bin centroid
   fraction of every narrow (<=2-bin) spectral line found across the corpus. Clustering at
   half-integer fractions means edge; clustering at integer fractions means centre. This uses only
   the two-bin power ratio, so it needs no counts-to-dB scale and is not circular the way the WAV
   check is -- it is one pass over data B3 pulls anyway.
3. **Re-run the WAV centroid comparison once B1 pins counts-to-dB.** At a plausible ~1 dB/count
   scale the implied offset already reads +0.37 +- 0.15 bins, a real ~2-sigma lean toward edge, but
   it is not usable as evidence until the scale itself is established independently.

## How this would show up as wrong

If the two-bin-split census clusters cleanly at half-integers, or ONC confirms edge-binning, this
record should be superseded: `FFT_AXIS_CONVENTION` moves to `"edge"`, every named frequency in
`docs/plans/acoustics_plan_v2.md` §3 shifts up by 125 Hz, and the default in `band_limit` should be
revisited (an exact edge axis would remove, not just narrow, the uncertainty on the product path).

Related: `docs/decisions/0010-analysis-band-and-small-craft.md` (the band this axis feeds);
`docs/plans/acoustics_plan_v2.md` §3.2 (fuller evidence table, restated here with the ledger's
actual numbers).
