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
EVENT_EXCESS_THRESHOLD_DB = 10.0


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
    level = np.concatenate([s.level_product_db for s in series])
    order = np.argsort(t)
    n_at_floor = sum(s.censoring["n_at_floor"] for s in series)
    n_cells = sum(s.censoring["n_cells"] for s in series)
    return {
        "t_utc_s": t[order],
        "level_product_db": level[order],
        "n_bins_in_band": series[0].n_bins_in_band,
        "fraction_in_band_at_floor": n_at_floor / n_cells if n_cells else float("nan"),
        "max_cell_level": max(float(np.max(s.level_product_db)) for s in series),
        "decidecade_resolvable": series[0].decidecade_resolvable,
    }


def find_events(t_utc_s, level_product_db, *, frame_seconds=None,
                threshold_db=EVENT_EXCESS_THRESHOLD_DB):
    """Contiguous runs of excess-over-ambient lasting at least the minimum duration.

    The DURATION constraint is the whole point (acoustics_plan_v2 SS5 B5:
    "excess-over-ambient threshold with a minimum-duration constraint"). Without
    it this counts impulses, and the ~38 kHz echosounder alone would fire it
    every few seconds -- which is exactly why an event count with no duration
    floor is not evidence of anything.
    """
    if frame_seconds is None:
        frame_seconds = config.FFT_FRAME_SECONDS
    baseline = features.ambient_baseline_product_db(level_product_db)
    excess = np.asarray(level_product_db, dtype=float) - baseline
    hot = excess >= threshold_db
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
            "peak_excess_product_db": float(seg[peak]),
        })
    return {"baseline_product_db": float(baseline), "events": out,
            "peak_excess_product_db": float(np.max(excess)),
            "threshold_db": float(threshold_db),
            "min_duration_s": EVENT_MIN_DURATION_S}


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
            "peak_excess_product_db": found["peak_excess_product_db"],
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
        "peak_excess_product_db": found["peak_excess_product_db"],
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

def frame_shuffle_null(t_utc_s, level_product_db, *, seed=0):
    """Shuffle frames in time. Event STRUCTURE must collapse; level stats must not.

    The sharpest available null for "is this a pass or is it processing?"
    Shuffling preserves the level distribution exactly and destroys only the
    time ordering, so any statistic that survives it was never about temporal
    structure. A CPA event cannot survive; a mis-scaled axis or a stuck bin can.
    """
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(level_product_db, dtype=float).copy()
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
            found = find_events(w["t_utc_s"], w["level_product_db"])
            shuffled = frame_shuffle_null(w["t_utc_s"], w["level_product_db"])

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
                shift_found = find_events(sw["t_utc_s"], sw["level_product_db"])
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
            print(f"{cov.overpass.scene_id:<26} {found['baseline_product_db']:6.1f} "
                  f"{found['peak_excess_product_db']:7.1f} {len(found['events']):6d} "
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
          f"(peak {bb['peak_excess_product_db']:.1f} dB, expected "
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
