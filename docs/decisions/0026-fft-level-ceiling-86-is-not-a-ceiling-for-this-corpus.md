# 0026. FFT_LEVEL_CEILING = 86 is falsified for the 2025 in-season overpass corpus

Status: accepted
Date: 2026-08-27
Amends: none directly -- corrects the applicability of decision 0014's discussion of 86 and of
`boatphone/config.py`'s `FFT_LEVEL_CEILING` comment, without changing either
Scope: `boatphone/config.py` (`FFT_LEVEL_CEILING`), `boatphone/fft_io.py` (`censoring_report`,
`n_at_ceiling`), decision 0014, decision 0015 (B5 censoring-aware thresholds)
Source: `boatphone.fft_io.read_fft_gz` run over all 90 real files pulled by the live probe,
`data/raw/onc/overpass_window_corpus/`, 2025 in-season dates, device ICLISTENHF1266

## Context

`config.FFT_LEVEL_CEILING = 86` is documented in `boatphone/config.py` as an ASSUMPTION, not a
measurement -- the comment there already states it was derived from the max observed on two quiet
5-minute fixtures and explicitly flags itself as unconfirmed. Decision 0014 inherits this and
treats 86 as "possibly a hard clip," carried into `censoring_report`'s `n_at_ceiling` count, which
feeds decision 0015's censoring-aware thresholds for segment B5.

## Finding

Reading all 90 real corpus files pulled by the live probe with `boatphone.fft_io.read_fft_gz`:

* Every file reads to shape **(1200, 512)**, dtype **float64** -- consistent with the format
  already established.
* The value range across the corpus is **0.0 to 112.0**.
* **93.0 was already exceeded within the first five files** -- this is not a rare tail event
  reached only deep into the corpus.

A single file's range of 0.0-92.0 (the basis the earlier assumption generalised from) was
previously generalised to describe the corpus. **That generalisation was wrong.**

## Decision

Record the falsification plainly: **`config.FFT_LEVEL_CEILING = 86` is not a ceiling for this
corpus, and decision 0014's treatment of 86 as "possibly a hard clip" is falsified for it.** Values
up to 112.0 occur, well above 86, and occur early rather than as a rare extreme.

This record does NOT change `FFT_LEVEL_CEILING` in `config.py`, and does NOT resolve what 86 was
measuring in the first place. It states the falsification and its consequences so nothing
downstream keeps treating 86 as a ceiling without knowing it has been checked and failed.

## Consequences

* Anything computing `n_at_ceiling` (via `boatphone.fft_io.censoring_report`) against the pulled
  corpus will report a large and meaningless count -- meaningless because it is counting cells at
  a threshold that is not actually a scale limit for this corpus, not counting genuine
  floor/ceiling-censored cells the way the floor side of `censoring_report` does.
* Decision 0015's censoring-aware thresholds for segment B5 were built assuming 86 was at least a
  plausible ceiling. Since 86 is not a ceiling for the corpus B5 will actually run against, **any
  censoring-aware threshold from 0015 must be reviewed against this finding before it is trusted**
  -- this record does not perform that review, it flags that the review is required.
* `boatphone/config.py`'s `FFT_LEVEL_CEILING` comment and decision 0014's ceiling discussion should
  be read alongside this record, not as settled, until `FFT_LEVEL_CEILING` (or its use in
  `censoring_report`) is itself revisited.

## Scope -- what this record does and does not cover

This observation covers **the 2025 in-season overpass windows on device ICLISTENHF1266 only** (90
files, 3 dates, from the live probe). It says nothing about:

* whether 86 was a genuine ceiling for the earlier hand-delivered sample (the two quiet 5-minute
  fixtures the original 86 figure came from) -- that measurement stands on its own two files and
  is not contradicted, only shown not to generalise;
* whether 86 holds for other years or other device deployments in the eventual full corpus.

Do not overstate this into a claim about the whole archive.

## Open question (not answered here)

Is 86 a genuine clip point in some deployments or years and not others (e.g. a gain-setting or
calibration change between the sample fixtures and the 2025 in-season corpus), or was 86 never a
real ceiling at all -- just the loudest value two quiet windows happened to reach, as decision
0014's own correction already anticipated as a possibility? This record does not decide between
these; resolving it needs either provenance on why the two fixtures and the 90-file corpus might
differ (deployment/gain history) or a wider sample across years to see whether 86-ish pile-up
recurs anywhere. Marked here as requiring follow-up before B5 censoring-aware thresholds are
trusted for the real corpus.

## How this would show up as wrong

If a future wider pull across years/deployments shows values reliably piling up at some ceiling
distinct from 112 (the current corpus's observed max), that ceiling -- not 86 -- would be the
candidate to re-examine; 112 is simply the maximum observed so far in the 90 files checked, not
asserted here as a new ceiling.

Related: `docs/decisions/0014-two-spectral-ceilings-and-the-408-exclusion.md` (86's original
context and its own "not established the way the floor is" caveat); `docs/decisions/0015-censoring-aware-thresholds-for-b5.md`
(the downstream consumer this record requires be reviewed); `boatphone/config.py`
(`FFT_LEVEL_CEILING`, unchanged by this record).
