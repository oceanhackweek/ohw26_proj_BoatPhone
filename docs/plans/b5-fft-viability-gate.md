---
title: B5 viability gate -- is the .fft.gz product usable?
type: plan
status: done
milestone: "1"
owner: Isaac Guld
related:
  - ../decisions/0010-analysis-band-and-small-craft.md
  - ../decisions/0012-b0-model-viability-outcome.md
  - ../decisions/0013-fft-axis-convention-is-an-open-assumption.md
  - ../decisions/0014-two-spectral-ceilings-and-the-408-exclusion.md
  - ../decisions/0015-censoring-aware-thresholds-for-b5.md
  - ../decisions/0026-fft-level-ceiling-86-is-not-a-ceiling-for-this-corpus.md
  - ../../references/ONC_communication.txt
  - acoustics_plan_v2.md
source: ~/.claude/plans/a-previous-agent-left-typed-hollerith.md
---

# PLAN 1 of 2 — `.fft.gz` detector viability gate (RUN NOW)

> **Companion plan:** `b3-close-out-offline.md` (in this directory) — the B3 manifest/correctness close-out, to run
> **tonight**, after this one. The two are independent: this plan reads the corpus directly off
> disk and takes each file's start time from its filename (`onc_client.parse_file_coverage`, per
> `fft_io`'s existing contract); it never opens the manifest that the other plan rewrites.
> **Do not interleave them** — the other plan rewrites artefacts while these files are being read.

## Context

Three things landed at once and they change the acoustics plan's near-term ordering.

**1. ONC answered (`references/ONC_communication.txt`).** The reply closes two open items in
`docs/plans/acoustics_plan_v2.md` — negatively — and opens a new path:

- The `.fft.gz` product is a proprietary Ocean Sonics black box with filtering sometimes applied,
  and is "essentially uncalibrated unless you have all the metadata from Ocean Sonics, which I
  don't believe ONC currently publishes." So **B1's pass criterion (§5: magnitude regression of
  product bin *b* against WAV bin *2b*, slope ≈ 1) can no longer pass on its own terms**, and the
  ~42 dB low-frequency anomaly (§3, "still open, gated in B1") is now *explained* rather than open.
  B1's own stated fallback — relative and shape-based features, under which G1/G2/G3 all survive —
  becomes the decided path.
- The 256 kHz sensitivity curve exists but ONC states it "wouldn't be applicable to those files."
  So §3's bins-1-204-calibratable / 205-417-not split collapses: for `.fft.gz`, **every** bin is
  uncalibrated. B6's absolute-calibration branch is dead on the fft path.
- §3.2's open centre-vs-edge binning question loses resolution route (a) — "ask ONC, one sentence
  settles it" — because the documentation does not exist. Route (c) needed B1 pinning counts to dB,
  which the above kills. Only route (b), the scale-free two-bin-split centroid census over the
  corpus, survives. `FFT_AXIS_OFFSET_UNCERTAINTY_HZ = 125.0` is carried indefinitely.
- ONC independently endorses B5's design ("look at the sound level at ~100 Hz… the absolute
  calibrated level isn't necessary for that to work", and change the band for smaller vessels).
- **New third path:** Oceans 3.0 server-side spectrogram `.mat` data products — calibrated, small,
  but with processing latency. The plan never considered this. It is the fallback if the fft
  product proves unusable, and the latency is why this gate must run early.

**2. The Planet timestamps falsify B3's window assumption.** `acoustics_plan_v2.md` §9 assumed
PlanetScope overpasses fall 09:30–11:30 local and flagged it: *"B3's speculative pull rests on
this; verify against the first overpasses.csv."* Verified now against
`contributor_folders/malachymcc/planet_folger/gate2_survivors.csv` (30 scenes, column `acquired`,
tz-aware UTC): they land at **18:17–19:49 UTC = 11:17–12:49 local**, bimodal (a ~18:2x cluster and
a ~19:4x cluster, tracking the PS2.SD / PSB.SD constellation split). The corpus was pulled for
16:15–18:45 UTC. Measured coverage of each scene's ±15 min window against the files on disk:

| Coverage | Scenes |
|---|---|
| Full | **13** |
| Partial | **5** |
| **Zero** | **12** |

The corpus hour histogram is `{16: 8030, 17: 10699, 18: 8027}` — it stops before the second
cluster entirely.

Also worth stating plainly: **no imagery has actually landed.** `planet_folger_HANDOFF.md:5` reads
"Nothing ordered", `CONFIRM_ORDER = False`, and there are zero image files in the repo. What we
have is a 30-scene candidate list with trustworthy UTC timestamps — which is exactly what this
plan needs, but it is not pixels, and Neve's detector has produced no detections yet.

**3. The corpus is complete and untouched:** 26,666 `.fft.gz` + 90 plain `.fft`, 6.8 GB, 918/918
dates, `run_complete: true`, 0 errors.

**Decision taken (user, this session): do not pull anything yet.** Test a simple detector on the
overpasses we already cover. That test decides whether the `.fft.gz` product is usable at all, or
whether we must switch to Oceans 3.0 `.mat` products — and that answer, not a window fix, is what
should drive any further acquisition.

---

## Two things carried out of the deferred plan, because deferring them has a real cost

- **Tell Malachy today that `config.py:633-634` is wrong.** This is a message, not an edit — the
  constant change itself belongs to the companion plan. If he orders archive imagery against the
  `09:15–11:45` window in the meantime, that is unrecoverable quota spend — the same failure class
  deliverable O1 was built to prevent, and the source of truth already carries "MALACHY, THIS ONE IS
  FOR YOU: ORDER ARCHIVE IMAGERY ONLY, 2020-2025 SEASONS." Send the measured window
  (18:17–19:49 UTC / 11:17–12:49 local, bimodal) and the 13/5/12 coverage table.
- **The check suite is RED and that blocks new checks.** `check_b3c_12` fails partway through
  `scripts/checks.py`, and the 8 checks after it never run. Any detector check appended to the end
  of the file will silently not run either. Until the companion plan lands, either register the new
  checks **before** the failure point or run them selectively — and report which, so a green line in
  the log is not mistaken for a full suite. **Do not fix `check_b3c_12` here**; it belongs to the
  companion plan and has a subtlety documented there.

---

## The work

A minimal B5-style detector on the 13 fully-covered overpasses, whose purpose is to answer one
question: *is the `.fft.gz` product usable as a vessel-presence surface, or must we order
Oceans 3.0 `.mat` products?*

Build on what exists — do not write a new reader. `boatphone/fft_io.py` already provides
`read_fft_gz`, `frequency_axis_hz`, `time_axis_utc_s`, `band_limit_product`,
`censoring_report`, `structural_zero_report`; `boatphone/models.py` provides `assert_band_matched`,
`band_limit`, `assert_comparable` (40/40 green, mutation-tested).

New module `boatphone/features.py` (the plan's §5 B5 home), plus a notebook in
`contributor_folders/` for the figures.

**Inputs.** The 18 scenes with any coverage; the 13 with full ±15 min windows are the primary set.
Negatives come from the existing corpus at the same clock hour on other dates — abundant.

**Constraints inherited, all non-negotiable:**
- Nothing above `FFT_B5_RELATIVE_CEILING_BIN = 408` enters any statistic (decision 0014).
- Every band level is reported **next to** `censoring_report()` output; a window with ceiling hits
  is a lower bound, not a measurement (decision 0015). Note decision 0026 says the 86 ceiling is
  not actually binding for this corpus — confirm that holds on loud windows, which is precisely
  where it would first fail.
- Every band edge widened by `FFT_AXIS_OFFSET_UNCERTAINTY_HZ = 125.0` via `band_limit_product`
  (§3.2); no band edge placed inside a narrow feature.
- **Relative only.** No absolute dB re 1 µPa is claimed anywhere — ONC's reply forecloses it for
  this product. Levels are excess-over-ambient in the product's own uncalibrated integer scale, and
  must be labelled as such at every boundary.
- Below 2.2 kHz these are raw-bin levels, **not** standards-compliant decidecade bands — label them
  so (§5 B5).

**Method.** Band-excess-over-rolling-ambient on the time-frequency surface, not one averaged
spectrum — 1200 frames × 0.25 s across the window gives level-versus-time through closest point of
approach, which is the structure that matters. Bands to try, justified rather than inherited: ONC
suggests ~100 Hz for ships, but §"Dropped, and why" argues small planing craft radiate 1–10 kHz
broadband cavitation and small craft are the target population — and the ~42 dB LF anomaly plus the
250 Hz bin floor degrade the low band anyway. Test both; report both. The CATFISH (60–300 Hz) vs
May River (discard <800 Hz) disagreement noted in §4 is unadjudicated and this is where it gets
adjudicated on our own data.

**Null checks, run and reported — not optional (invariant 4).** A clean vessel-to-energy
correlation is what a time-alignment bug also produces:
- shift the acoustic time base by ±1 h and re-score;
- shuffle the overpass labels across windows;
- score quiet windows with no scene at all.

If the shifted or shuffled version scores as well as the aligned one, the result is a bug, and
say so.

**The gate's verdict, stated in one of three forms (invariant 9):**
- **GO** — the product separates vessel-present from vessel-absent windows above the nulls. Proceed
  with B5 on `.fft.gz`; schedule the targeted top-up pull (~210 files, ~60 MB, minutes) for the
  16 uncovered/partial scenes, then the window widening.
- **NO-GO, product is broken** — filtering or censoring destroys the signal. Switch to Oceans 3.0
  `.mat` products; the latency ONC warns about makes ordering them urgent, so this verdict must be
  reached early.
- **NULL RESULT, method found nothing** — the product is fine but 13 overpasses at ~11:20 local do
  not carry enough vessel signal to separate. Different follow-up: more overpasses, not a different
  product.

Record the verdict as a decision record and reflect it in `docs/plans/acoustics_plan_v2.md` and
`docs/plans/Project_Source_of_Truth.txt` the same day.

**Honest caveat that must ship with any number from this gate:** n = 13 overpasses, from a corpus
sampled at one clock hour, over cloud-free summer days — and, per §7, the unit of analysis is the
overpass, not the detection. There are no optical detections yet, so "vessel present" here means
"a gate-2 scene exists", not "a vessel was seen". That is a weaker label than B7 assumes and every
statement must say so.

## Out of scope for this plan

- Anything in the companion plan: the manifest migration, `check_b3c_12`, the `config.py` window
  and coordinate constants, the 90 orphan `.fft` files.
- The targeted top-up pull and the window widening — both gated on this gate's verdict.
- The WAV second pass for B1's magnitude regression — largely moot now that ONC has foreclosed
  `.fft.gz` calibration; revisit only on GO, and only if B6 is still wanted.

## Verification

1. New checks for the detector, authored to fail first, built on a synthetic tone of known level,
   frequency and time (invariant 3 / the test-author standard) — including a check that asserts the
   time-shifted null is *rejected*, mirroring the existing
   `check_b0_2a_synthetic_tone_frame_shuffle_null_is_rejected`.
2. Run those checks either before `check_b3c_12`'s failure point or selectively, and **state which**.
   A full-suite green is the companion plan's exit criterion and must not be claimed here.
3. Re-execute the notebook top-to-bottom in a fresh kernel — the strongest verification available
   here (CLAUDE.md; there is no pytest, ruff, or type check).
4. Clear notebook outputs before committing unless the output is the point.

Branch is `milestone1/b3-bulk-acquisition`, currently RED with a DO-NOT-MERGE commit at `1937b6e`.
It stays red through this plan; the companion plan greens it. Nothing merges until then.


---

## Outcomes (2026-08-27)

**Verdict: GO.** Full detail in `../decisions/0027-fft-product-is-viable-for-b5-band-level-detection.md`.

What landed:

* `boatphone/features.py` -- band levels and excess-over-ambient on the time-frequency surface;
  the module decision 0010 SS3 named but that did not exist. Enforces the 250 Hz floor, the
  bin-408 relative ceiling (0014), the carried +-125 Hz axis uncertainty (0013), and reports
  in-band floor censoring beside every level (0015).
* `boatphone/overpasses.py` -- the join from Malachy's `gate2_survivors.csv` to the corpus, by
  interval overlap (0020), refusing any acquisition stamp without a UTC offset (0002).
* `scripts/run_b5_gate.py` -- the gate. Runs in **42 s** over the 13 fully-covered windows.
* `boatphone/config.py` -- the B5 band block and the overpass-matchup constants.
* Nine `check_b5_*` checks in `scripts/checks.py`, mutation-tested against five deliberate
  sabotages (median->mean, dropped frequency floor, ignored band limit, naive-timestamp
  coercion, start-containment join); each was caught by its intended check.

Headline numbers: 43 events across 13 windows in the 1-10 kHz band against **0** under the
frame-shuffle null; 0.00% in-band floor censoring; synthetic 50 dB injection recovered at
50.0 dB at the right time and duration, and invisible when placed out of band.

Corrections to this plan as written:

* coverage is **13 full / 5 partial / 12 zero**, not 14/4/12 -- the earlier figure counted files
  rather than measuring interval overlap, and the 2020-08-06 window has 6 files but only 92%
  coverage.
* the plan's worry that a check appended after `check_b3c_12` would silently not run was
  unnecessary: `scripts/checks.py --all` runs past failures. Suite result **178 passed,
  1 failed, 4 skipped, 0 not run** -- the 1 failure is the pre-existing `check_b3c_12`, which
  the companion plan owns.
* ONC's ~100 Hz band proved **unimplementable** (bin 0 is DC), which the plan did not anticipate.
  `FFT_B5_SHIP_PROXY_BAND_HZ` at 250-1000 Hz is a labelled proxy, and a sub-250 Hz request now
  raises.
* the real-data time-shift null turned out to have **no discriminating power** absent labels,
  and its result must not be cited as a passed null. See 0027 "What stays open".
