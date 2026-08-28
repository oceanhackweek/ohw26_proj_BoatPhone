#!/usr/bin/env python3
"""Compare candidate vessel-COUNT estimators against the 29 manual vessel counts.

Runnable entry point. Answers one question: which way of turning a band-level
time series into a NUMBER OF VESSELS comes closest to what a human counted?

WHY THIS EXISTS. `find_events` counts excursions above a threshold. A vessel's
level fluctuates, so one passage opens several excursions -- 298 raw events
against 44 counted vessels. Merging adjacent fragments (decision 0030 work) cut
that to 135 but left the correlation with the manual count at r = +0.20, so
fragmentation was not the whole story. These estimators are the alternatives.

THE METHODS ARE PRE-DECLARED. Every estimator in METHODS below was written down
BEFORE any of them was scored, and every one is reported whatever it does. This
is the only defence available at n = 29 with 6 negatives: with that little data,
choosing an estimator by looking at its score and then quoting that score is
circular. The winner here is EXPLORATORY -- the honest claim is its rank in a
pre-declared field, not its accuracy.

WHY NO ZERO-CROSSING RATE OF THE WAVEFORM. There is no waveform. `.fft.gz` is
averaged magnitude with no phase (decision 0027), so ZCR in the audio sense is
not computable. `slope_zero_cross` below is the analogue that IS computable and
is the one asked for: a passage rises to closest approach and falls, so the
smoothed d(level)/dt crosses zero DOWNWARD once per passage. Counting those
crossings counts passages rather than threshold excursions.

UNITS: level is the product's own uncalibrated integer `counts`, never dB re
1 uPa. Times are absolute UTC.

Usage:  python3 scripts/compare_event_grouping.py [--band small_craft] [--json OUT]
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import pathlib
import sys

import numpy as np
from scipy import signal as _sig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from boatphone import config, features
from boatphone import overpasses as ov
from boatphone import paths

import build_events_table as bet
import run_b5_gate

# Half-windows to evaluate, seconds. The corpus was pulled at +/-15 min; the
# question is whether a TIGHTER window around the acquisition instant tracks the
# manual count better, since the manual count is an instantaneous observation and
# a 30-minute window can contain traffic the image never saw.
HALF_WINDOWS_S = (900, 600, 300, 150)

# Smoothing for the level trace before slope/peak methods, seconds. Levels are
# integer-quantised, so differentiating a lightly-smoothed series over 0.25 s
# steps returns quantisation noise and nothing else. 45 s matches
# build_review_set.SLOPE_SMOOTHING_SECONDS -- well below the minutes-long
# timescale of a passage and well above the quantisation step.
SLOPE_SMOOTH_S = 45.0

# Minimum separation between two counted passages, seconds. Two vessels closer
# together than this are not separable by a single hydrophone's level trace and
# are counted as one. Deliberately generous: this estimator is trying to STOP
# overcounting.
MIN_SEPARATION_S = 180.0


def _smooth(x, seconds, frame_seconds):
    n = max(1, int(round(seconds / frame_seconds)))
    if n % 2 == 0:
        n += 1
    if n >= len(x):
        return np.full_like(np.asarray(x, dtype=float), float(np.median(x)))
    kernel = np.ones(n) / n
    return np.convolve(np.asarray(x, dtype=float), kernel, mode="same")


# --- the pre-declared estimators -------------------------------------------
# Each takes (t_utc_s, level_counts, frame_seconds) and returns an integer count.

def m_raw_events(t, level, fs):
    """Baseline: threshold excursions, unmerged. Known to overcount."""
    return len(run_b5_gate.find_events(t, level, frame_seconds=fs)["events"])


def m_merged_10s(t, level, fs):
    """Current production rule: fragments joined across gaps <= 10 s."""
    ev = run_b5_gate.find_events(t, level, frame_seconds=fs)["events"]
    return len(run_b5_gate.merge_events(ev, max_gap_s=10.0))


def m_merged_120s(t, level, fs):
    """The same rule with a much larger gap -- how far does merging alone go?"""
    ev = run_b5_gate.find_events(t, level, frame_seconds=fs)["events"]
    return len(run_b5_gate.merge_events(ev, max_gap_s=120.0))


def m_slope_zero_cross(t, level, fs):
    """Downward zero-crossings of the smoothed slope, inside elevated regions.

    THE METHOD ASKED FOR. A passage rises to closest point of approach and falls
    again, so d(level)/dt crosses zero from positive to negative exactly once per
    passage. Counting those crossings counts PASSAGES, where a threshold counts
    EXCURSIONS -- and a fluctuating vessel produces many excursions but only one
    rise-and-fall.

    Restricted to samples above the event threshold so that zero-crossings of
    quiet ambient wander are not counted: without that restriction this counts
    the noise floor's own texture.
    """
    smooth = _smooth(level, SLOPE_SMOOTH_S, fs)
    baseline = features.ambient_baseline_counts(level)
    hot = smooth >= baseline + run_b5_gate.EVENT_EXCESS_THRESHOLD_COUNTS
    slope = np.gradient(smooth, fs)
    crossings = np.where((slope[:-1] > 0) & (slope[1:] <= 0) & hot[:-1])[0]
    if len(crossings) == 0:
        return 0
    # Collapse crossings closer together than MIN_SEPARATION_S: a single passage
    # with a double-humped peak would otherwise count twice.
    keep = [crossings[0]]
    for c in crossings[1:]:
        if (t[c] - t[keep[-1]]) >= MIN_SEPARATION_S:
            keep.append(c)
    return len(keep)


def m_peak_prominence(t, level, fs):
    """Prominent local maxima of the smoothed level.

    Same intuition as the slope method but stated in the language `scipy` uses,
    and with an explicit PROMINENCE requirement: a bump only counts if it rises
    the threshold amount above the surrounding trace, not merely above the
    window baseline. That distinguishes a genuine approach from a ripple riding
    on an already-elevated period.
    """
    smooth = _smooth(level, SLOPE_SMOOTH_S, fs)
    baseline = features.ambient_baseline_counts(level)
    peaks, _ = _sig.find_peaks(
        smooth,
        height=baseline + run_b5_gate.EVENT_EXCESS_THRESHOLD_COUNTS,
        prominence=run_b5_gate.EVENT_EXCESS_THRESHOLD_COUNTS / 2.0,
        distance=max(1, int(round(MIN_SEPARATION_S / fs))),
    )
    return len(peaks)


def m_peak_prominence_strict(t, level, fs):
    """As above, but a peak must stand a FULL threshold above its surroundings.

    The stricter prominence is the single knob most likely to separate "one
    vessel whose level wobbles" from "two vessels".
    """
    smooth = _smooth(level, SLOPE_SMOOTH_S, fs)
    baseline = features.ambient_baseline_counts(level)
    peaks, _ = _sig.find_peaks(
        smooth,
        height=baseline + run_b5_gate.EVENT_EXCESS_THRESHOLD_COUNTS,
        prominence=run_b5_gate.EVENT_EXCESS_THRESHOLD_COUNTS,
        distance=max(1, int(round(MIN_SEPARATION_S / fs))),
    )
    return len(peaks)


def m_elevated_periods(t, level, fs):
    """Contiguous elevated periods of the SMOOTHED trace -- no peak structure.

    The null estimator for the peak methods: if smoothing alone explains the
    improvement, this matches them and the peak logic earns nothing.
    """
    smooth = _smooth(level, SLOPE_SMOOTH_S, fs)
    baseline = features.ambient_baseline_counts(level)
    hot = smooth >= baseline + run_b5_gate.EVENT_EXCESS_THRESHOLD_COUNTS
    if not hot.any():
        return 0
    edges = np.diff(hot.astype(int))
    n = int((edges == 1).sum()) + (1 if hot[0] else 0)
    return n


def m_any_elevation(t, level, fs):
    """Presence only: 1 if the window is ever elevated, else 0.

    The floor. Any estimator that cannot beat "was anything there at all" is not
    counting vessels, it is detecting presence -- and at a mean of 1.5 vessels
    per window this is a genuinely competitive baseline, which is the point.
    """
    return 1 if m_elevated_periods(t, level, fs) > 0 else 0


METHODS = {
    "raw_events": m_raw_events,
    "merged_10s": m_merged_10s,
    "merged_120s": m_merged_120s,
    "slope_zero_cross": m_slope_zero_cross,
    "peak_prominence": m_peak_prominence,
    "peak_prominence_strict": m_peak_prominence_strict,
    "elevated_periods": m_elevated_periods,
    "any_elevation": m_any_elevation,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--band", default="small_craft",
                    choices=["small_craft", "ship_proxy"])
    ap.add_argument("--json", help="write the full result set here")
    args = ap.parse_args(argv)

    band_hz = {"small_craft": config.FFT_B5_SMALL_CRAFT_BAND_HZ,
               "ship_proxy": config.FFT_B5_SHIP_PROXY_BAND_HZ}[args.band]

    manual, _src = bet.load_manual_vessel_counts()
    overpass_list = ov.load_gate2_overpasses()
    index = ov.corpus_file_index()
    if paths.ONC_LABELLED_WINDOW_DIR.is_dir():
        index = sorted(index + ov.corpus_file_index(paths.ONC_LABELLED_WINDOW_DIR),
                       key=lambda r: r[0])

    # ONE ROW PER LABELLED INSTANT. The 20210616 label covers two adjacent
    # frames; scoring both would weight that overpass twice.
    seen, targets = set(), []
    for op in overpass_list:
        key = op.scene_id[:13]
        if key in seen or op.scene_id not in manual:
            continue
        seen.add(key)
        targets.append(op)
    print(f"band {args.band} = {band_hz} Hz;  {len(targets)} labelled instants")

    fs = config.FFT_FRAME_SECONDS
    results = {}
    truth = [manual[op.scene_id][0] for op in targets]

    for half_s in HALF_WINDOWS_S:
        series = []
        for op in targets:
            cov = ov.window_coverage(op, index, half_window_s=half_s)
            if not cov.n_files:
                series.append(None)
                continue
            w = run_b5_gate.concat_window(cov.paths, band_hz)
            t, lvl = w["t_utc_s"], w["level_counts"]
            # concat_window returns whole FILES; clip to the requested window so
            # a shorter half-window is actually shorter, not just fewer files.
            lo, hi = op.window_utc(half_s)
            keep = (t >= lo.timestamp()) & (t <= hi.timestamp())
            series.append((t[keep], lvl[keep]))
        for name, fn in METHODS.items():
            pred = [fn(s[0], s[1], fs) if s is not None else None for s in series]
            results[(half_s, name)] = pred

    def score(pred):
        pairs = [(p, m) for p, m in zip(pred, truth) if p is not None]
        if not pairs:
            return None
        p_, m_ = zip(*pairs)
        total_p, total_m = sum(p_), sum(m_)
        mae = float(np.mean([abs(a - b) for a, b in zip(p_, m_)]))
        within1 = sum(1 for a, b in zip(p_, m_) if abs(a - b) <= 1) / len(p_)
        r = (float(np.corrcoef(p_, m_)[0, 1])
             if len(set(p_)) > 1 and len(set(m_)) > 1 else float("nan"))
        # "Within 70% of the validation numbers": the predicted TOTAL relative to
        # the counted total. 1.00 is exact; >1 overcounts, <1 undercounts.
        ratio = total_p / total_m if total_m else float("nan")
        return {"total_pred": total_p, "total_manual": total_m, "ratio": ratio,
                "mae": mae, "within_1": within1, "r": r, "n": len(p_)}

    print(f"\nmanual total = {sum(truth)} vessels over {len(truth)} instants "
          f"(mean {np.mean(truth):.2f}/window)\n")
    header = (f"{'half-window':>11} {'method':<24} {'pred':>5} {'ratio':>6} "
              f"{'MAE':>5} {'<=1':>5} {'r':>7}")
    print(header)
    print("-" * len(header))
    table = []
    for half_s in HALF_WINDOWS_S:
        for name in METHODS:
            sc = score(results[(half_s, name)])
            if sc is None:
                continue
            table.append(dict(half_window_s=half_s, method=name, **sc))
            print(f"{half_s // 60:>9} min {name:<24} {sc['total_pred']:>5} "
                  f"{sc['ratio']:>6.2f} {sc['mae']:>5.2f} "
                  f"{sc['within_1']:>5.0%} {sc['r']:>+7.3f}")

    # The stated stopping rule: predicted total within 70% of the counted total,
    # i.e. |ratio - 1| <= 0.30. Reported for every method, not just the winner.
    ok = [row for row in table if abs(row["ratio"] - 1.0) <= 0.30]
    print(f"\nwithin 70% of the manual total (|ratio-1| <= 0.30): {len(ok)} of "
          f"{len(table)} configurations")
    for row in sorted(ok, key=lambda r: abs(r["ratio"] - 1.0))[:8]:
        print(f"   {row['half_window_s']//60:>2} min  {row['method']:<24} "
              f"ratio {row['ratio']:.2f}  MAE {row['mae']:.2f}  "
              f"within1 {row['within_1']:.0%}  r {row['r']:+.3f}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(
            {"band": args.band, "band_hz": list(band_hz),
             "manual_total": sum(truth), "n_instants": len(truth),
             "half_windows_s": list(HALF_WINDOWS_S),
             "slope_smooth_s": SLOPE_SMOOTH_S,
             "min_separation_s": MIN_SEPARATION_S,
             "methods_pre_declared": list(METHODS),
             "table": table}, indent=1), encoding="utf-8")
        print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
