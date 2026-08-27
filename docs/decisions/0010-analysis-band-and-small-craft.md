# 0010. The analysis band, and why >=250 Hz is not a limitation for small craft

Status: accepted
Date: 2026-08-27

## Context

Two claims about frequency have been repeated inconsistently across the plans, and both decide
what the project can honestly say.

**The band itself.** `proposed_plan_IG.md` rev. 3 assumed 125 Hz per bin. The evidence says
**250 Hz per bin** (1024-pt FFT at 256 kHz, 512 bins over 0-128 kHz), confirmed three independent
ways in `accoutics_plan.md` §"What changed since rev. 3": a 38 kHz echosounder line landing at bin
152, the anti-alias shoulder at bin ~408 = 102 kHz, and no discontinuity at bin 205. So **bin 1 is
250 Hz** and there is no sub-250 Hz structure available at all. Separately, the supplied
sensitivity file stops at **51.2 kHz = bin 205**, so bins 206-417 are uncalibratable from what we
hold. `accoutics_plan.md` §A8 then wrote "<= 51.2 kHz" as if it were the *analysis* band, conflating
the calibration ceiling with the usable band.

**What the floor costs.** Every plan so far has treated ">= 250 Hz" as a straightforward
limitation, and `proposed_plan_IG.md` explicitly laments losing the 40-60 Hz band the ranging
literature favours. §A5 half-contradicted this and the contradiction was never resolved. It matters
because the project's stated target is **small recreational craft** (source-of-truth goals 2 and
4), and a team that believes its instrument is blind to its target will misjudge every result it
gets.

## Decision

**1. The analysis band is 250 Hz -- 128 kHz, with absolute-level claims restricted to 250 Hz --
51.2 kHz (bins 1-205).** These are two different limits and are named separately:

* *Support*: what the product carries. Bins 1-417; above ~bin 418 is anti-alias roll-off.
* *Calibrated support*: where dB re 1 uPa is defensible. Bins 1-205 only.
* `calibrate.py` emits a band-validity mask; every downstream consumer respects it, and a request
  for an absolute level outside bins 1-205 **raises** rather than returning a number.

Where a common band with an external corpus is needed, compute it rather than assuming one; for
32 kHz-sampled sources that is 250 Hz -- 11.652 kHz. Enforced by `assert_band_matched` in
`boatphone/models.py`.

**2. Record, and state in the write-up, that the 250 Hz floor does not impair small-craft
detection.** Small planing hulls and outboards radiate peak energy at roughly **1-10 kHz**
(cavitation broadband) -- well inside our support. It is **large ships** whose diagnostic
blade-rate tonals sit at **10-100 Hz**, below the floor.

So the floor costs us **large-vessel** characterisation, not small-boat detection. It happens to
cut against the population we care least about.

**3. Decidecade bands are requested only above ~2.2 kHz.** A decidecade band at *f* is ~0.23*f*
wide, and two 250 Hz bins require *f* >~ 2.2 kHz. Below that, report **raw-bin levels and label
them as such** -- not as standards-compliant band levels. **Do not claim hybrid-millidecade
compliance** at all: that scheme needs 1 Hz resolution below 455 Hz, which this product cannot
supply. Enforced in `features.py`.

## Consequences

* **Large-vessel work is genuinely degraded**, and any statement about cargo/tanker traffic must
  say so. This is a real cost, not a reframing -- it is simply not the cost the plans assumed.
* **The ranging literature's preferred 40-60 Hz band is unreachable.** Single-hydrophone ranging
  here leans on the time-frequency structure through closest point of approach (rise/fall slope,
  peak width, CPA time) rather than on low-frequency tonals.
* **No absolute-level claim above 51.2 kHz** until ONC supplies the 256 kHz sensitivity curve. That
  request goes out with the product-definition question.
* **The ~42 dB unexplained shape difference across 250 Hz -> 2 kHz is not resolved by this record.**
  It survives the 250 Hz remapping and remains the open gate (v2 §B1). If it turns out to be a
  high-pass in ONC's product generation, part of the low band may be recoverable and this record
  should be revisited.
* **How to tell this was wrong:** if measured small-craft detections concentrate below 250 Hz in
  the WAV comparison, or if the detectability curve for the 0-12 m FAO class is markedly worse than
  for larger classes at equal range, the premise in (2) is wrong and the band cost is worse than
  stated.

Supersedes the "<= 51.2 kHz" framing in `accoutics_plan.md` §A8 and the sub-125 Hz framing in
`proposed_plan_IG.md` rev. 3. Related: `0002-time-alignment-and-units.md`.
