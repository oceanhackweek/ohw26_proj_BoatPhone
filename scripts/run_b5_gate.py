#!/usr/bin/env python3
"""B5 viability gate: is the `.fft.gz` product usable as a vessel-presence surface?

Runnable entry point. DEFINES NOTHING SHARED (CLAUDE.md invariant 6): bands come
from `boatphone.config`, the reduction from `boatphone.features`, the scene/corpus
join from `boatphone.overpasses`, the reader from `boatphone.fft_io`.

WHAT THIS GATE CAN AND CANNOT DECIDE -- read before quoting any number from it.

It answers ONE question: does the ONC `.fft.gz` product carry usable
vessel-presence structure, or must the project switch to Oceans 3.0 spectrogram
`.mat` data products (references/ONC_communication.txt)? That question is about
the PRODUCT.

It does NOT and cannot measure detection performance, because THERE ARE NO
VESSEL LABELS YET. No Planet imagery has been ordered
(planet_folger_HANDOFF.md: "Nothing ordered", CONFIRM_ORDER = False) and the
optical detector has produced no detections. A "positive" window here means only
"a cloud-free scene was CATALOGUED at this instant" -- not that a vessel was
seen in it -- and a "negative" window means only "no scene was catalogued", which
is unlabelled, NOT vessel-free. So:

  * no P, R, AUC or false-positive rate is computed here, and none may be
    quoted from this gate;
  * goal G1's "validated against optical labels, with a stated false-positive
    rate" (acoustics_plan_v2 SS1) remains blocked on the optical arm;
  * a scene-vs-no-scene difference is NOT expected and its absence is NOT a
    negative result about the product (CLAUDE.md invariant 9).

What the gate rests on instead is evidence that does not need labels:
  A. the surface reads and reduces without degeneracy, and in-band censoring is
     low enough for the median reduction to be unbiased (decision 0015);
  B. band level versus time shows closest-point-of-approach STRUCTURE -- a
     contiguous rise and fall -- not isolated single-frame spikes;
  C. a SYNTHETIC tone of known level, frequency and time is recovered at that
     frequency and time (CLAUDE.md invariant 3), proving the pipeline's
     time-frequency geometry rather than assuming it;
  D. the nulls behave: a time-shifted window and a frame-shuffled surface do not
     reproduce the structure (CLAUDE.md invariant 4).

Usage:  python3 scripts/run_b5_gate.py [--json OUT.json] [--max-windows N]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import sys

import numpy as np

# scripts/ is the entry-point layer; the repo root must be importable for the
# boatphone/ library. Same idiom as the other scripts here.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scipy import signal as scipy_signal

from boatphone import config, features, fft_io
from boatphone import overpasses as ov

# How long a run of consecutive above-threshold frames must last to count as a
# CPA-shaped event rather than a spike. A vessel passing a fixed hydrophone
# rises and falls over tens of seconds to minutes; a single 0.25 s frame is an
# impulse (a click, a bit error, an echosounder ping) and must not count.
EVENT_MIN_DURATION_S = 20.0
# Excess over the window's own ambient baseline that opens an event, in the
# product's own dB-like units. Chosen ABOVE the frame-to-frame spread of quiet
# ambient, not tuned on any label -- there are no labels to tune on, and
# decision 0015 forbids tuning thresholds more than once on the overpass set.
EVENT_EXCESS_THRESHOLD_COUNTS = 10.0


def _frames_per_event_min(frame_seconds: float) -> int:
    return max(1, int(round(EVENT_MIN_DURATION_S / frame_seconds)))


def concat_window(paths, band_hz):
    """Read a window's files and return one time-ordered band-level series.

    Files are concatenated in TIME order, not directory order, and the frame
    times are carried through so nothing is positional. Nothing is resampled,
    padded or gap-filled: a window with an interior gap simply has fewer frames
    (CLAUDE.md invariant 5).
    """
    series = []
    for path in sorted(paths):
        product = fft_io.read_fft_gz(path)
        series.append(features.band_level_series(product, band_hz))
    if not series:
        raise ValueError("no files in window")
    t = np.concatenate([s.t_utc_s for s in series])
    level = np.concatenate([s.level_counts for s in series])
    order = np.argsort(t)
    n_at_floor = sum(s.censoring["n_at_floor"] for s in series)
    n_cells = sum(s.censoring["n_cells"] for s in series)
    return {
        "t_utc_s": t[order],
        "level_counts": level[order],
        "n_bins_in_band": series[0].n_bins_in_band,
        "fraction_in_band_at_floor": n_at_floor / n_cells if n_cells else float("nan"),
        "max_cell_level": max(float(np.max(s.level_counts)) for s in series),
        "decidecade_resolvable": series[0].decidecade_resolvable,
    }


def find_events(t_utc_s, level_counts, *, frame_seconds=None,
                threshold_counts=EVENT_EXCESS_THRESHOLD_COUNTS):
    """Contiguous runs of excess-over-ambient lasting at least the minimum duration.

    The DURATION constraint is the whole point (acoustics_plan_v2 SS5 B5:
    "excess-over-ambient threshold with a minimum-duration constraint"). Without
    it this counts impulses, and the ~38 kHz echosounder alone would fire it
    every few seconds -- which is exactly why an event count with no duration
    floor is not evidence of anything.
    """
    if frame_seconds is None:
        frame_seconds = config.FFT_FRAME_SECONDS
    baseline = features.ambient_baseline_counts(level_counts)
    excess = np.asarray(level_counts, dtype=float) - baseline
    hot = excess >= threshold_counts
    min_frames = _frames_per_event_min(frame_seconds)

    events = []
    start = None
    for i, flag in enumerate(hot):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_frames:
                events.append((start, i))
            start = None
    if start is not None and len(hot) - start >= min_frames:
        events.append((start, len(hot)))

    out = []
    for lo, hi in events:
        seg = excess[lo:hi]
        peak = int(np.argmax(seg))
        out.append({
            "t_start_utc_s": float(t_utc_s[lo]),
            "t_peak_utc_s": float(t_utc_s[lo + peak]),
            "duration_s": float((hi - lo) * frame_seconds),
            "peak_excess_counts": float(seg[peak]),
        })
    return {"baseline_counts": float(baseline), "events": out,
            "peak_excess_counts": float(np.max(excess)),
            "threshold_counts": float(threshold_counts),
            "min_duration_s": EVENT_MIN_DURATION_S}


# How large a below-threshold dip may be bridged when merging fragments of one
# elevated period. NOT a detector parameter and NOT a threshold: `find_events`
# runs unchanged and its raw count is always reported alongside the merged one.
# This is an AGGREGATION rule applied afterwards, so decision 0015's "tune the
# threshold on the overpasses once" is not spent by it.
#
# THE VALUE IS A CONVENTION, NOT A MEASUREMENT, and it is consequential.
# On the single densest window (20230719_182055_22_2449) the merged count IS flat
# across the parameter -- ship_proxy gives 3 at every gap from 5 s to 120 s. That
# plateau does NOT hold across the corpus: the 30-window detection-band total
# runs 160 / 141 / 135 / 117 / 108 / 101 / 88 at gaps of 0 / 5 / 10 / 20 / 30 /
# 60 / 120 s -- a 45% decline with no flat region. An earlier version of this
# comment claimed the plateau corpus-wide, generalising from one window; that was
# wrong and is corrected here rather than deleted.
#
# So 10 s is defensible (it is short against the minutes-long timescale of a
# passage and long against the 0.25 s frame) but it is not picked out by the
# data, and the merged count must never be quoted without the gap that produced
# it. build_events_table.py prints the full sweep for exactly this reason.
EVENT_MERGE_GAP_S = 10.0


def merge_events(events, *, max_gap_s=EVENT_MERGE_GAP_S):
    """Join events separated by a gap no larger than ``max_gap_s``.

    WHY THIS EXISTS. A vessel passing a fixed hydrophone does not hold a steady
    level: it fluctuates, and each dip below the threshold ENDS one event and
    STARTS another. The raw count therefore measures threshold chatter as much as
    it measures passages -- 16 raw events in the window above are visibly one
    sustained elevated period plus a couple of separate ones.

    MERGED ON TIME ALONE, deliberately not on amplitude similarity. A real
    closest-point-of-approach RISES and FALLS, so fragments of one genuine pass
    have systematically DIFFERENT peak amplitudes (measured here: 31 -> 36.5 ->
    31 -> 23.5 counts across one passage). An amplitude-similarity gate would
    refuse to merge exactly the fragments that belong to one pass, and would
    preferentially merge flat sources like weather -- selecting against the CPA
    shape the detector exists to find.

    A merged event is a DISTINCT ELEVATED PERIOD. It is still not a vessel: two
    boats overlapping in time merge into one, and the count-to-vessel mapping
    remains unmeasured (decision 0030).

    THE CALLER MUST APPLY THIS TO THE NULLS TOO. Merging real events while
    leaving a null unmerged manufactures a difference out of bookkeeping alone:
    the real count falls and the null does not.
    """
    if not events:
        return []
    ordered = sorted(events, key=lambda e: e["t_start_utc_s"])
    out = [dict(ordered[0])]
    for ev in ordered[1:]:
        last = out[-1]
        last_end = last["t_start_utc_s"] + last["duration_s"]
        if ev["t_start_utc_s"] - last_end <= max_gap_s:
            end = max(last_end, ev["t_start_utc_s"] + ev["duration_s"])
            # The merged peak is the LARGEST of the fragments' peaks, not the
            # first and not a mean: the peak of a passage is its closest
            # approach, and averaging fragments would erase it.
            if ev["peak_excess_counts"] > last["peak_excess_counts"]:
                last["t_peak_utc_s"] = ev["t_peak_utc_s"]
                last["peak_excess_counts"] = ev["peak_excess_counts"]
            last["duration_s"] = end - last["t_start_utc_s"]
            last["n_fragments"] = last.get("n_fragments", 1) + 1
        else:
            out.append(dict(ev))
    for ev in out:
        ev.setdefault("n_fragments", 1)
    return out


# --- Vessel-count estimator (decision 0031) ---------------------------------
#
# Smoothing applied before peak-picking, seconds. Levels are integer-quantised,
# so an unsmoothed trace differentiated over 0.25 s steps is quantisation noise.
# Matches build_review_set.SLOPE_SMOOTHING_SECONDS.
VESSEL_COUNT_SMOOTH_S = 45.0
# Two passages closer together than this are not separable on a single
# hydrophone's level trace and count as one. Generous on purpose: the failure
# this estimator exists to fix is OVERcounting.
VESSEL_COUNT_MIN_SEPARATION_S = 180.0


def estimate_vessel_count(t_utc_s, level_counts, *, frame_seconds=None):
    """Count PASSAGES, not threshold excursions. The B5 vessel-count estimator.

    `find_events` counts runs above a threshold, and a vessel's level fluctuates,
    so one passage opens many runs -- 298 raw events against 44 vessels counted
    by eye. This counts prominent local maxima of the smoothed band level
    instead: a passage rises to closest point of approach and falls exactly once,
    so it contributes exactly one peak however much it wobbles on the way.

    PROMINENCE, not height, is what does the work. A bump counts only if it rises
    a full EVENT_EXCESS_THRESHOLD_COUNTS above the surrounding trace -- not
    merely above the window baseline. That is what separates "one vessel whose
    level wobbles" from "two vessels", and it is why this beats simple merging.

    MEASURED against 29 human-counted overpasses (decision 0031): 42 predicted
    against 44 counted, 27/29 windows within +-1, r = +0.61 -- against r = +0.20
    for merged event counts and r = +0.26 for presence-only. Replicates
    independently on the ship_proxy band (ratio 1.07, r = +0.60).

    CHOSEN FROM A PRE-DECLARED FIELD of 8 estimators x 4 window widths
    (`scripts/compare_event_grouping.py`), all of them reported. At n = 29 that
    is the only available defence against picking a winner by its score, and it
    is not a strong one: this is a proof of concept, not a validated model.

    NOT a vessel identity. Two vessels within
    VESSEL_COUNT_MIN_SEPARATION_S of each other still count as one, and the
    time-shift null does NOT collapse (decision 0030) -- an hour-shifted window
    yields nearly as many. This counts passages well; it does not establish that
    they are the overpass's vessels.
    """
    if frame_seconds is None:
        frame_seconds = config.FFT_FRAME_SECONDS
    level = np.asarray(level_counts, dtype=float)
    n = max(1, int(round(VESSEL_COUNT_SMOOTH_S / frame_seconds)))
    if n % 2 == 0:
        n += 1
    if n >= len(level):
        return 0
    return len(vessel_peak_times(t_utc_s, level_counts,
                                 frame_seconds=frame_seconds))


def vessel_peak_times(t_utc_s, level_counts, *, frame_seconds=None):
    """The CPA instants `estimate_vessel_count` counts, as absolute UTC seconds.

    Exposed so a figure can MARK the peaks the count is made of. A labelled
    figure that showed a number without showing which features produced it would
    be unreviewable -- the reader could not tell a correct count from a
    coincidence, which is the whole purpose of putting the figure next to the
    imagery.
    """
    if frame_seconds is None:
        frame_seconds = config.FFT_FRAME_SECONDS
    level = np.asarray(level_counts, dtype=float)
    t = np.asarray(t_utc_s, dtype=float)
    n = max(1, int(round(VESSEL_COUNT_SMOOTH_S / frame_seconds)))
    if n % 2 == 0:
        n += 1
    if n >= len(level):
        return []
    smooth = np.convolve(level, np.ones(n) / n, mode="same")
    baseline = features.ambient_baseline_counts(level)
    peaks, props = scipy_signal.find_peaks(
        smooth,
        height=baseline + EVENT_EXCESS_THRESHOLD_COUNTS,
        prominence=EVENT_EXCESS_THRESHOLD_COUNTS,
        distance=max(1, int(round(VESSEL_COUNT_MIN_SEPARATION_S / frame_seconds))),
    )
    return [{"t_peak_utc_s": float(t[i]),
             "prominence_counts": float(props["prominences"][k]),
             "level_counts": float(smooth[i])}
            for k, i in enumerate(peaks)]


# --- Test C: synthetic tone ------------------------------------------------

def synthetic_tone_test(band_hz):
    """Inject a tone of KNOWN level, frequency and time; require it back.

    Built on a synthetic surface rather than real data so the ground truth is
    exact (CLAUDE.md invariant 3). Proves three things the rest of the gate
    otherwise assumes: the frequency axis puts a tone in the band that contains
    its frequency, the time axis puts the peak at the frame it was injected in,
    and the band reduction actually responds to in-band energy.

    Also runs the NEGATIVE control: the same tone placed OUTSIDE the band must
    NOT produce an event. A detector that fires on both is band-blind, and a
    positive-only test would never notice.
    """
    n_frames, n_bins = config.FFT_N_FRAMES, config.FFT_N_BINS
    freq_hz = fft_io.frequency_axis_hz(n_bins=n_bins)
    frame_s = config.FFT_FRAME_SECONDS
    t_utc_s = np.arange(n_frames, dtype=float) * frame_s

    in_band_bin = int(np.argmin(np.abs(freq_hz - 5000.0)))     # inside 1-10 kHz
    out_band_bin = int(np.argmin(np.abs(freq_hz - 30000.0)))   # outside it
    tone_lo, tone_hi = 400, 600                                # frames, 50 s long
    quiet, loud = 20.0, 70.0

    results = {}
    for label, tone_bin in (("in_band", in_band_bin), ("out_of_band", out_band_bin)):
        surface = np.full((n_frames, n_bins), quiet, dtype=float)
        surface[tone_lo:tone_hi, tone_bin] = loud
        kept_freq, _ = fft_io.band_limit_product(freq_hz, surface[0], band_hz)
        in_band = np.isin(freq_hz, kept_freq)
        # Reduce exactly as band_level_series does, so this tests THAT reduction.
        level = np.median(surface[:, in_band], axis=1)
        found = find_events(t_utc_s, level)
        results[label] = {
            "tone_freq_hz": float(freq_hz[tone_bin]),
            "tone_frames": [tone_lo, tone_hi],
            "n_events": len(found["events"]),
            "peak_excess_counts": found["peak_excess_counts"],
            "t_peak_utc_s": found["events"][0]["t_peak_utc_s"] if found["events"] else None,
        }

    # A single loud bin among ~37 does not move a MEDIAN, and that is correct
    # behaviour, not a bug: the median is chosen precisely so one bin cannot
    # dominate a band level. Report what actually happened rather than asserting
    # a recovery the estimator is designed not to give.
    results["n_bins_in_band"] = int(in_band.sum())
    results["note"] = (
        "A one-bin tone is deliberately NOT recoverable by a median over "
        f"{int(in_band.sum())} bins; broadband vessel energy is. The discriminating "
        "assertion is the band separation below, not the peak level."
    )

    # Broadband injection: the signal class the detector is actually for.
    surface = np.full((n_frames, n_bins), quiet, dtype=float)
    kept_freq, _ = fft_io.band_limit_product(freq_hz, surface[0], band_hz)
    in_band = np.isin(freq_hz, kept_freq)
    surface[tone_lo:tone_hi, in_band] = loud
    level = np.median(surface[:, in_band], axis=1)
    found = find_events(t_utc_s, level)
    ok_time = bool(found["events"]) and (
        t_utc_s[tone_lo] <= found["events"][0]["t_peak_utc_s"] < t_utc_s[tone_hi]
    )
    results["broadband_in_band"] = {
        "n_events": len(found["events"]),
        "peak_excess_counts": found["peak_excess_counts"],
        "expected_excess_product_db": loud - quiet,
        "peak_time_inside_injection": ok_time,
        "duration_s": found["events"][0]["duration_s"] if found["events"] else None,
        "expected_duration_s": (tone_hi - tone_lo) * frame_s,
    }

    # Same broadband injection placed entirely OUTSIDE the band must be invisible.
    surface = np.full((n_frames, n_bins), quiet, dtype=float)
    out_mask = (freq_hz >= 20000.0) & (freq_hz <= 30000.0)
    surface[tone_lo:tone_hi, out_mask] = loud
    level = np.median(surface[:, in_band], axis=1)
    results["broadband_out_of_band"] = {
        "n_events": len(find_events(t_utc_s, level)["events"]),
        "expected_n_events": 0,
    }
    return results


# --- Test D: nulls ---------------------------------------------------------

def frame_shuffle_null(t_utc_s, level_counts, *, seed=0):
    """Shuffle frames in time. Event STRUCTURE must collapse; level stats must not.

    The sharpest available null for "is this a pass or is it processing?"
    Shuffling preserves the level distribution exactly and destroys only the
    time ordering, so any statistic that survives it was never about temporal
    structure. A CPA event cannot survive; a mis-scaled axis or a stuck bin can.
    """
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(level_counts, dtype=float).copy()
    rng.shuffle(shuffled)
    return find_events(t_utc_s, shuffled)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="write the full result set here")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="limit the number of overpass windows scored (smoke runs)")
    parser.add_argument("--shift-hours", type=float, default=-1.0,
                        help="offset for the time-shift null (default -1 h; the "
                             "corpus does not extend +1 h past the overpasses)")
    args = parser.parse_args(argv)

    overpass_list = ov.load_gate2_overpasses()
    index = ov.corpus_file_index()
    coverages = [ov.window_coverage(o, index) for o in overpass_list]
    summary = ov.coverage_summary(coverages)
    full = [c for c in coverages if c.is_full]
    if args.max_windows:
        full = full[: args.max_windows]

    print("=" * 78)
    print("B5 VIABILITY GATE -- is the .fft.gz product usable?")
    print("=" * 78)
    print(f"scenes in gate2 list        : {summary['n_overpasses']}")
    print(f"  full acoustic coverage    : {summary['n_full']}   <- the primary set")
    print(f"  partial coverage          : {summary['n_partial']}")
    print(f"  NO coverage               : {summary['n_zero']}")
    print(f"corpus files indexed        : {len(index)}")
    print("\nNO VESSEL LABELS EXIST. 'positive' == a scene was catalogued, not that a")
    print("vessel was seen. No detection metric is computed or may be quoted.\n")

    bands = {
        "small_craft_1-10kHz": config.FFT_B5_SMALL_CRAFT_BAND_HZ,
        "ship_proxy_250-1000Hz": config.FFT_B5_SHIP_PROXY_BAND_HZ,
    }

    results = {"coverage": summary, "bands": {}, "n_corpus_files": len(index)}

    for band_name, band_hz in bands.items():
        print("-" * 78)
        print(f"BAND {band_name}  = {band_hz} Hz")
        print("-" * 78)
        rows = []
        print(f"{'scene':<26} {'base':>6} {'peak':>7} {'events':>6} {'longest':>8} "
              f"{'floor%':>7} {'shufEv':>7} {'shiftEv':>8}")
        for cov in full:
            w = concat_window(cov.paths, band_hz)
            found = find_events(w["t_utc_s"], w["level_counts"])
            shuffled = frame_shuffle_null(w["t_utc_s"], w["level_counts"])

            # Time-shift null: score a REAL window an hour away on the same date.
            shifted = _dt.timedelta(hours=args.shift_hours)
            shifted_op = ov.Overpass(
                scene_id=cov.overpass.scene_id + "_SHIFTED",
                acquired_utc=cov.overpass.acquired_utc + shifted,
                clear_percent=None, instrument=None,
            )
            shift_cov = ov.window_coverage(shifted_op, index)
            if shift_cov.n_files:
                sw = concat_window(shift_cov.paths, band_hz)
                shift_found = find_events(sw["t_utc_s"], sw["level_counts"])
                shift_n = len(shift_found["events"])
            else:
                shift_found, shift_n = None, None

            longest = max((e["duration_s"] for e in found["events"]), default=0.0)
            rows.append({
                "scene_id": cov.overpass.scene_id,
                "acquired_utc": cov.overpass.acquired_utc.isoformat(),
                "n_bins_in_band": w["n_bins_in_band"],
                "fraction_in_band_at_floor": w["fraction_in_band_at_floor"],
                "max_cell_level": w["max_cell_level"],
                "decidecade_resolvable": w["decidecade_resolvable"],
                "detected": found,
                "null_frame_shuffle": shuffled,
                "null_time_shift_hours": args.shift_hours,
                "null_time_shift": shift_found,
            })
            print(f"{cov.overpass.scene_id:<26} {found['baseline_counts']:6.1f} "
                  f"{found['peak_excess_counts']:7.1f} {len(found['events']):6d} "
                  f"{longest:8.1f} {w['fraction_in_band_at_floor']*100:6.2f}% "
                  f"{len(shuffled['events']):7d} "
                  f"{'n/a' if shift_n is None else shift_n:>8}")

        n_ev = [len(r["detected"]["events"]) for r in rows]
        n_sh = [len(r["null_frame_shuffle"]["events"]) for r in rows]
        n_ts = [len(r["null_time_shift"]["events"]) for r in rows
                if r["null_time_shift"] is not None]
        floor = [r["fraction_in_band_at_floor"] for r in rows]
        print(f"\n  events/window   real {np.mean(n_ev):.2f}   "
              f"frame-shuffled {np.mean(n_sh):.2f}   "
              f"time-shifted {np.mean(n_ts):.2f} (n={len(n_ts)})")
        print(f"  max in-band floor-censored fraction: {max(floor)*100:.2f}% "
              f"(median reduction unbiased below 50%)")
        results["bands"][band_name] = {
            "band_hz": list(band_hz),
            "windows": rows,
            "mean_events_real": float(np.mean(n_ev)),
            "mean_events_frame_shuffled": float(np.mean(n_sh)),
            "mean_events_time_shifted": float(np.mean(n_ts)) if n_ts else None,
            "max_fraction_in_band_at_floor": float(max(floor)),
        }

    print("=" * 78)
    print("SYNTHETIC TONE (known level, frequency, time) -- invariant 3")
    print("=" * 78)
    tone = synthetic_tone_test(config.FFT_B5_SMALL_CRAFT_BAND_HZ)
    bb = tone["broadband_in_band"]
    print(f"  bins in band                : {tone['n_bins_in_band']}")
    print(f"  broadband in-band  events   : {bb['n_events']} "
          f"(peak {bb['peak_excess_counts']:.1f} counts, expected "
          f"{bb['expected_excess_product_db']:.1f})")
    print(f"  peak lands inside injection : {bb['peak_time_inside_injection']}")
    print(f"  duration {bb['duration_s']} s, expected {bb['expected_duration_s']} s")
    print(f"  broadband OUT-of-band events: {tone['broadband_out_of_band']['n_events']} "
          f"(expected {tone['broadband_out_of_band']['expected_n_events']})")
    print(f"  single-bin tone note        : {tone['note']}")
    results["synthetic_tone"] = tone

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1, default=str)
        print(f"\nfull results -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
