# 0002. Time base, sample rate, and units are UTC-first and stated

Status: accepted
Date: 2026-08-26

## Context

This project's core operation is a **join between two instruments on time**: hydrophone records
from Barkley Sound and PlanetScope scenes overhead. A satellite pass is seconds long; the
acoustic signature of a vessel is minutes long. The join tolerance is small enough that an hour
of timezone error, or a few seconds of clock convention error, silently destroys it.

The failure mode is what makes this worth a record. A time-base error does not throw. It produces
a *plausible* result: a weak-but-present correlation, or a clean correlation between the wrong
things. Nothing in the pipeline flags it, no reviewer reading the code sees it, and the number
looks like an answer. The same is true of the neighbouring conventions:

- **Sample rate** assumed rather than read: every frequency in the spectrum is scaled wrong, and
  the spectrogram still looks like a spectrogram.
- **Calibration**: the ICLISTEN hydrophone ships a pre-deployment calibration file. Levels
  compared across instruments or across deployments without it are not comparable, but they are
  still numbers.
- **FFT scaling and windowing**: one-sided vs two-sided, and window normalization, move levels by
  fixed dB offsets. A threshold tuned on one convention is wrong on the other.
- **Position and range**: degrees mistaken for metres, or a range in the wrong units, produces a
  detection-range estimate off by orders of magnitude that still plots.

Every one of these is cheap to prevent and expensive to discover -- typically discovered by the
result being wrong in a way nobody can explain, on the last day.

## Decision

1. **UTC end to end.** Every timestamp in code, in derived files, and in plots is UTC and
   timezone-aware. Naive datetimes are not passed across a function boundary. Local time appears
   only in a final human-facing label, and is labelled as such.
2. **Read the sample rate from the file.** Never hardcode it, never infer it from a filename.
   If a file's rate differs from what the analysis expects, handle it explicitly or reject it --
   never resample silently.
3. **Apply calibration before comparing levels** across instruments, deployments, or time, and
   state the reference (`dB re 1 uPa`) wherever a level appears.
4. **State the convention at every boundary.** A function taking or returning a time, frequency,
   level, or position says which convention in its signature or a one-line docstring. `t` is not
   a time base; `t_utc_s` is. Same for `level_db_re_1upa`, `freq_hz`, `lat_deg`.
5. **Prove it with synthetic ground truth, before anything depends on it.** A tone of known
   amplitude, frequency, and onset time goes through the real pipeline and comes back at the
   right level, the right frequency, and the right time -- and a deliberately mis-offset input
   *fails* the check. In `/run-phase` this is its own gated segment, and it comes first.
6. **Be suspicious of a result that works.** Before believing a correlation, check it against a
   null: shuffle the labels, shift the time base by an hour, or use a period with no detections,
   and confirm the signal disappears. Report that you did.

## Consequences

- The first segment of the acoustic pipeline is a calibration-and-timing gate that produces no
  scientific result. That is the point; it is the cheapest hour in the project.
- Some code is more verbose (`t_utc_s` rather than `t`). Worth it.
- Reviewers have something concrete to check, rather than reading for vibes.
- If a result is later found to be wrong, the null checks and the synthetic gate narrow *where*
  it went wrong, instead of casting doubt on everything.
