#!/usr/bin/env python3
"""Acoustic event counts for every PlanetScope overpass, with their nulls beside them.

Runnable entry point. DEFINES NOTHING SHARED (CLAUDE.md invariant 6): the bands
come from `boatphone.config`, the band reduction from `boatphone.features`, the
detector from `scripts.run_b5_gate` (`concat_window` + `find_events`), and the
scene/corpus join from `boatphone.overpasses`. There is exactly ONE definition of
"event" in this repo and it is `run_b5_gate.find_events`; this script counts what
that function returns and invents no rule of its own.

WHY THE COLUMN IS `n_events` AND NOT `n_vessels`. The detector finds runs of
excess over the window's own ambient baseline lasting at least
`EVENT_MIN_DURATION_S`. That is not a vessel identity. One boat lingering through
closest point of approach can open several events; two boats overlapping merge
into one; a rain squall is broadband and transient on the same timescale. Mapping
events to vessels is goal G1/G3 work that needs labels. 29 overpasses now carry a
manual vessel count (`data/validation/planet_scope_validations.csv`), and the
mapping MEASURED against them is weak: r = +0.20 between manual vessel count and
merged event count. Naming the column `n_vessels` would assert a relationship the
data does not support -- `manual_vessel_count` is the only vessel count here.

WHY THE NULLS RIDE ON THE SAME ROW. CLAUDE.md invariant 4: a clean
vessel-to-energy correlation is exactly what a time-alignment bug also produces.
Two nulls travel with every count so it can never be read alone --

  * `n_events_frame_shuffled` -- the same frames in shuffled time order. Preserves
    the level distribution EXACTLY and destroys only temporal structure, so a
    count that survives it was never about a passage.
  * `n_events_time_shifted` -- a real window `--shift-hours` away on the same
    date. Controls for "this hydrophone is just noisy on this day".

An empty cell in a null column means the null could not be evaluated (no corpus
coverage at the shifted time), which is NOT the same as zero and is not counted
as one.

WHY THE OPTICAL COLUMNS SAY `candidates`. `detections.csv` from the optical arm
holds 32,877 detection CANDIDATES over 26 scenes, of which ~15,490 carry
`transient=1` and one single scene carries 8,257 rows. It has not been filtered
to vessels. Both a raw and a non-transient count are reported and neither is a
vessel count.

UNITS, stated at every boundary (decision 0002 SS4): levels are the product's own
UNCALIBRATED integer scale, `counts`, never dB re 1 uPa -- the matched
WAV/product pair measures ~0.52 counts/dB WITH CURVATURE, so no fixed conversion
exists (decision 0027). Times are absolute UTC.

Usage:  python3 scripts/build_events_table.py [--out-dir DIR] [--shift-hours H]
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as _dt
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from boatphone import config, features
from boatphone import overpasses as ov
from boatphone import paths
from boatphone.paths import DERIVED_DIR, ensure_dir

import run_b5_gate

# The optical arm's output. Read-only here; this script never writes to it.
OPTICAL_DETECTIONS_RELPATH = (
    "contributor_folders/malachymcc/planet_folger/detections.csv")

# Detection bands make events; diagnostic bands explain them. Both are tabulated,
# and the `is_detection_band` column says which is which, because a rain-band
# event is evidence ABOUT a small-craft event and not a second detection to add
# to it.
DETECTION_BANDS = ("small_craft", "ship_proxy")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=pathlib.Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_optical_candidate_counts():
    """Per-scene candidate counts from the optical arm, or an empty dict.

    Returns ``(counts, source)`` where counts maps scene_id -> (n_all,
    n_non_transient). Missing file is reported and returns empty rather than
    raising: the acoustic table is meaningful without the optical join, and a
    hard failure here would make the acoustic deliverable hostage to another
    workstream's file (CLAUDE.md: fail clearly when input is absent).
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    csv_path = root / OPTICAL_DETECTIONS_RELPATH
    if not csv_path.is_file():
        print(f"optical detections absent ({OPTICAL_DETECTIONS_RELPATH}) -- "
              "optical columns will be blank, which is not zero")
        return {}, None
    n_all = collections.Counter()
    n_solid = collections.Counter()
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scene = (row.get("scene_id") or "").strip()
            if not scene:
                continue
            n_all[scene] += 1
            # `transient=1` marks a candidate the optical arm itself flags as
            # not persistent across years. Kept as a SECOND column rather than
            # used to filter, since neither number has been validated against a
            # vessel label.
            if (row.get("transient") or "").strip() not in ("1", "true", "True"):
                n_solid[scene] += 1
    counts = {s: (n_all[s], n_solid[s]) for s in n_all}
    print(f"optical candidates: {sum(n_all.values()):,} rows over {len(counts)} scenes")
    return counts, OPTICAL_DETECTIONS_RELPATH


# Isaac's manual vessel counts from visual review of the PlanetScope scenes.
# The ONLY count in this project that is a VESSEL count rather than a candidate
# count or an event count.
MANUAL_VALIDATION_RELPATH = "data/validation/planet_scope_validations.csv"


def load_manual_vessel_counts():
    """Manual per-scene vessel counts, keyed by scene_id. ``({}, None)`` if absent.

    JOINED ON A TIMESTAMP PREFIX, and that is the fragile part. The file keys on
    ``YYYYMMDD_HHMM``, which is the first 13 characters of a PlanetScope
    ``scene_id`` -- minute resolution, where scene ids carry seconds. Two scenes
    in this set share a prefix (``20210616_183111_05_2448`` and
    ``20210616_183140_94_2436``, 29 s apart: adjacent frames of one overpass), so
    one label legitimately covers both.

    That sharing is RECORDED on every affected row (``manual_label_shared_by``)
    because it breaks summation: adding the manual column down all 30 scene rows
    counts those vessels twice. The corpus total is over the 29 distinct labelled
    instants, not the 30 scene rows.

    Both directions of the join are asserted. An unmatched label or an unlabelled
    scene RAISES rather than being dropped -- a silently unmatched label would
    look exactly like a scene the reviewer chose not to count.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    csv_path = root / MANUAL_VALIDATION_RELPATH
    if not csv_path.is_file():
        print(f"manual validations absent ({MANUAL_VALIDATION_RELPATH}) -- "
              "manual column will be blank, which is NOT zero vessels")
        return {}, None

    by_prefix = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row_no, row in enumerate(csv.DictReader(handle), start=2):
            # The header carries spaces after the commas; normalise both sides
            # rather than depending on the exact spelling of a hand-made file.
            row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
            key = row.get("timestamp", "")
            if not key:
                continue
            count = row.get("vessel count", "")
            if count == "":
                raise ValueError(
                    f"{csv_path.name} line {row_no}: empty vessel count for "
                    f"{key!r}. A blank is not a zero -- state the count or "
                    "remove the row.")
            by_prefix[key] = int(count)

    scenes = [o.scene_id for o in ov.load_gate2_overpasses()]
    shared = {}
    for scene in scenes:
        shared.setdefault(scene[:13], []).append(scene)

    unmatched = sorted(k for k in by_prefix if k not in shared)
    unlabelled = sorted(s for s in scenes if s[:13] not in by_prefix)
    if unmatched or unlabelled:
        raise ValueError(
            f"{csv_path.name} does not join cleanly to "
            f"{config.PLANET_GATE2_SURVIVORS_RELPATH}: "
            f"{len(unmatched)} label(s) match no scene ({unmatched}); "
            f"{len(unlabelled)} scene(s) have no label ({unlabelled}).")

    counts = {}
    for prefix, count in by_prefix.items():
        for scene in shared[prefix]:
            counts[scene] = (count, len(shared[prefix]))
    n_zero = sum(1 for c in by_prefix.values() if c == 0)
    print(f"manual vessel counts: {len(by_prefix)} labelled instants over "
          f"{len(counts)} scenes, {sum(by_prefix.values())} vessels, "
          f"{n_zero} labelled ZERO-vessel windows")
    return counts, MANUAL_VALIDATION_RELPATH


def count_readable(window_paths):
    """How many of a window's files actually carry frames.

    ONC served one file in the corpus as an empty body under HTTP 200
    (`ICLISTENHF1266_20220825T184149.000Z.fft.gz`, sha256 of the empty string,
    manifest status `downloaded`). It is a THIRD category alongside decision
    0007's measured zeros and 0023's absences -- listed, served, empty -- and a
    window containing it holds fewer frames than its file count implies. Counted
    explicitly so the shortfall shows up as a number rather than as a quietly
    shorter series.
    """
    readable = 0
    for path in window_paths:
        try:
            if path.stat().st_size > 0:
                readable += 1
        except OSError:
            pass
    return readable


def band_row(cov, band_name, band_hz, index, shift_hours, optical, manual):
    """One (scene, band) row: the count, its two nulls, and its caveats."""
    scene = cov.overpass
    is_detection = band_name in DETECTION_BANDS
    lo_hz, hi_hz = band_hz
    unc = config.FFT_AXIS_OFFSET_UNCERTAINTY_HZ

    row = {
        "scene_id": scene.scene_id,
        "overpass_id": scene.scene_id,          # B7 schema join key
        "acquired_utc": scene.acquired_utc.isoformat(),
        "coverage": "full" if cov.is_full else ("partial" if cov.n_files else "NONE"),
        "covered_fraction": round(cov.covered_fraction, 4),
        "n_window_files": cov.n_files,
        "n_window_files_readable": "",
        "band": band_name,
        "band_lo_hz": lo_hz,
        "band_hi_hz": hi_hz,
        # The axis convention is an OPEN assumption (decision 0013): the band
        # edges are known only to within this, so it travels with them.
        "band_edge_uncertainty_hz": unc,
        "is_detection_band": int(is_detection),
        "n_events": "",
        "n_events_merged": "",
        # The vessel-COUNT estimator (decision 0031). Prominent peaks of the
        # smoothed level: one passage contributes one peak however much it
        # wobbles. This is the column to compare against manual_vessel_count.
        "n_vessels_est": "",
        "peak_excess_counts_max": "",
        "total_event_duration_s": "",
        "baseline_counts": "",
        "n_events_frame_shuffled": "",
        "n_events_frame_shuffled_merged": "",
        "n_events_time_shifted": "",
        "n_events_time_shifted_merged": "",
        "merge_gap_s": run_b5_gate.EVENT_MERGE_GAP_S,
        "time_shift_hours": shift_hours,
        "max_fraction_in_band_at_floor": "",
        "n_optical_candidates": "",
        "n_optical_candidates_non_transient": "",
        # The one true vessel count in this project: human visual review.
        "manual_vessel_count": "",
        "manual_label_shared_by": "",
        "note": "",
    }
    if scene.scene_id in manual:
        count, n_sharing = manual[scene.scene_id]
        row["manual_vessel_count"] = count
        # >1 means this label is shared with another scene; summing the column
        # down the rows would count those vessels more than once.
        row["manual_label_shared_by"] = n_sharing
    if scene.scene_id in optical:
        row["n_optical_candidates"], row["n_optical_candidates_non_transient"] = \
            optical[scene.scene_id]

    if cov.n_files == 0:
        # An absence, not a zero. The detector was never run here, so writing 0
        # into n_events would create a silent negative out of no measurement.
        row["note"] = "NO ACOUSTIC DATA -- window not covered by the corpus"
        return row

    row["n_window_files_readable"] = count_readable(cov.paths)

    try:
        w = run_b5_gate.concat_window(cov.paths, band_hz)
    except (ValueError, features.UnrepresentableBandError) as exc:
        # Recorded, never swallowed (invariant 5). An unrepresentable band is a
        # statement about the band, not a detection of nothing.
        row["note"] = f"BAND NOT SCORED: {type(exc).__name__}: {exc}"
        return row

    # CLIP TO THE WINDOW. concat_window returns whole 5-minute FILES, so a
    # +/-15 min (1800 s) window arrives as up to 2100 s of samples -- the file
    # straddling each edge overhangs it. Every statistic below would then be
    # computed over ~17% more time than the window it is labelled with, and the
    # vessel-count estimator in particular was calibrated on the clipped window
    # (decision 0031): unclipped it reads 51 against 44 counted instead of 42.
    win_lo, win_hi = scene.window_utc(config.OVERPASS_MATCH_HALF_WINDOW_S)
    t_all = np.asarray(w["t_utc_s"], dtype=float)
    keep = (t_all >= win_lo.timestamp()) & (t_all <= win_hi.timestamp())
    n_dropped = int((~keep).sum())
    if n_dropped:
        # Printed, never silent (CLAUDE.md invariant 5).
        row["note"] = (row["note"] + " | " if row["note"] else "") + \
            f"clipped {n_dropped} frame(s) outside the window"
    w = dict(w, t_utc_s=t_all[keep],
             level_counts=np.asarray(w["level_counts"], dtype=float)[keep])

    found = run_b5_gate.find_events(w["t_utc_s"], w["level_counts"])
    events = found["events"]
    # Sensitivity: the merged count at every gap in the sweep, so the summary can
    # show whether the answer sits on a plateau or on a slope.
    row["_merge_sweep"] = {gap: len(run_b5_gate.merge_events(events, max_gap_s=gap))
                           for gap in MERGE_GAP_SWEEP_S}
    row["n_events"] = len(events)
    # Merged: fragments of ONE elevated period joined across dips no longer than
    # EVENT_MERGE_GAP_S. Both counts are reported -- the raw one measures
    # threshold chatter as much as passages, the merged one is the number of
    # distinct elevated periods, and neither is a vessel count.
    row["n_events_merged"] = len(run_b5_gate.merge_events(events))
    row["n_vessels_est"] = run_b5_gate.estimate_vessel_count(
        w["t_utc_s"], w["level_counts"])
    row["baseline_counts"] = round(found["baseline_counts"], 2)
    row["peak_excess_counts_max"] = (
        round(max(e["peak_excess_counts"] for e in events), 2) if events else "")
    row["total_event_duration_s"] = (
        round(sum(e["duration_s"] for e in events), 2) if events else 0.0)
    row["max_fraction_in_band_at_floor"] = round(w["fraction_in_band_at_floor"], 5)

    shuffled = run_b5_gate.frame_shuffle_null(w["t_utc_s"], w["level_counts"])
    row["n_events_frame_shuffled"] = len(shuffled["events"])
    row["n_events_frame_shuffled_merged"] = len(
        run_b5_gate.merge_events(shuffled["events"]))

    shifted_op = ov.Overpass(
        scene_id=scene.scene_id + "_SHIFTED",
        acquired_utc=scene.acquired_utc + _dt.timedelta(hours=shift_hours),
        clear_percent=None, instrument=None,
    )
    shift_cov = ov.window_coverage(shifted_op, index)
    if shift_cov.n_files:
        try:
            sw = run_b5_gate.concat_window(shift_cov.paths, band_hz)
            shift_events = run_b5_gate.find_events(
                sw["t_utc_s"], sw["level_counts"])["events"]
            row["n_events_time_shifted"] = len(shift_events)
            # MERGED WITH THE SAME RULE. Merging the real side only would drop
            # the real count while the null held, inventing an effect from
            # bookkeeping (see run_b5_gate.merge_events).
            row["n_events_time_shifted_merged"] = len(
                run_b5_gate.merge_events(shift_events))
        except (ValueError, features.UnrepresentableBandError) as exc:
            row["note"] = (row["note"] + " | " if row["note"] else "") + \
                f"time-shift null not scored: {type(exc).__name__}"
    else:
        # Blank, not zero: the corpus does not reach the shifted window, so the
        # null was not evaluated. Reading this as "the null found nothing" would
        # be the strongest possible version of the wrong conclusion.
        row["note"] = (row["note"] + " | " if row["note"] else "") + \
            "time-shift null NOT EVALUATED (no corpus coverage at the shifted time)"

    if cov.n_files and not cov.is_full:
        row["note"] = (row["note"] + " | " if row["note"] else "") + \
            "PARTIAL coverage -- a lower bound on this window's event count"
    return row


# Gaps the sensitivity table sweeps. Spans more than a decade so a reader can
# see whether the merged count sits on a plateau or on a slope.
MERGE_GAP_SWEEP_S = (0, 5, 10, 20, 30, 60, 120)


def write_summary_md(path, rows, summary, shift_hours, optical_source, run_id,
                     wide=None, sweep=None):
    """The aggregate the table is for, with every caveat that qualifies it."""
    scored = [r for r in rows if r["n_events"] != ""]
    sweep = sweep or {}
    det = [r for r in scored if r["is_detection_band"]]
    n_overpasses_scored = len({r["scene_id"] for r in scored})

    lines = [
        f"# Acoustic events across all overpass windows -- {run_id}",
        "",
        "Produced by `scripts/build_events_table.py`. Per-row detail in "
        "`events_by_window.csv`.",
        "",
        "## Totals by band",
        "",
        f"Counts shown as `raw -> merged` (merge gap "
        f"{run_b5_gate.EVENT_MERGE_GAP_S:.0f} s). **The merge rule is applied "
        "identically to the real events and to BOTH nulls** -- merging only the "
        "real side would drop that count while the nulls held, inventing a "
        "difference out of bookkeeping.",
        "",
        "| band | detection? | windows scored | events | frame-shuffled | "
        f"time-shifted ({shift_hours:+g} h) |",
        "|---|---|---|---|---|---|",
    ]
    for band in sorted({r["band"] for r in scored}):
        br = [r for r in scored if r["band"] == band]
        n_ev = sum(r["n_events"] for r in br)
        n_sh = sum(r["n_events_frame_shuffled"] for r in br
                   if r["n_events_frame_shuffled"] != "")
        ts = [r["n_events_time_shifted"] for r in br
              if r["n_events_time_shifted"] != ""]
        n_evm = sum(r["n_events_merged"] for r in br)
        n_shm = sum(r["n_events_frame_shuffled_merged"] for r in br
                    if r["n_events_frame_shuffled_merged"] != "")
        tsm = [r["n_events_time_shifted_merged"] for r in br
               if r["n_events_time_shifted_merged"] != ""]
        lines.append(
            f"| `{band}` | {'yes' if br[0]['is_detection_band'] else 'diagnostic'} | "
            f"{len(br)} | {n_ev} -> **{n_evm}** | {n_sh} -> {n_shm} | "
            f"{sum(ts)} -> {sum(tsm)} (n={len(ts)} windows) |")

    total_det_merged = sum(r["n_events_merged"] for r in det)
    total_det = sum(r["n_events"] for r in det)
    total_sh = sum(r["n_events_frame_shuffled"] for r in det
                   if r["n_events_frame_shuffled"] != "")
    lines += [
        "",
        f"**Detection bands together: n = {total_det} raw events -> "
        f"**{total_det_merged} merged** across {n_overpasses_scored} "
        f"overpasses**, against {total_sh} under the frame-shuffle null.",
        "",
        "## Against the manual vessel counts",
        "",
    ]
    # ONE ROW PER LABELLED INSTANT, not per scene: the shared 20210616 label
    # covers two adjacent frames and would otherwise be counted twice.
    seen, uniq = set(), []
    for w in (wide or []):
        if w.get("manual_vessel_count") in ("", None):
            continue
        key = w["scene_id"][:13]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(w)
    if not uniq:
        lines.append("No manual vessel counts available.")
    else:
        man = [int(w["manual_vessel_count"]) for w in uniq]
        mrg = [int(w["total_detection_events_merged"] or 0) for w in uniq]
        est = [int(w.get("n_small_craft_vessels_est") or 0) for w in uniq]
        r_est = (float(np.corrcoef(est, man)[0, 1])
                 if len(set(man)) > 1 and len(set(est)) > 1 else float("nan"))
        tp = sum(1 for a, b in zip(man, mrg) if a > 0 and b > 0)
        fn = sum(1 for a, b in zip(man, mrg) if a > 0 and b == 0)
        fp = sum(1 for a, b in zip(man, mrg) if a == 0 and b > 0)
        tn = sum(1 for a, b in zip(man, mrg) if a == 0 and b == 0)
        r = float(np.corrcoef(man, mrg)[0, 1]) if len(set(man)) > 1 else float("nan")
        lines += [
            f"n = {len(uniq)} labelled overpass instants, {sum(man)} vessels "
            f"counted by eye, {tn + fp} of them ZERO-vessel windows.",
            "",
            "| | detector fired | detector silent |",
            "|---|---|---|",
            f"| **vessel(s) present** ({tp + fn}) | {tp} | {fn} |",
            f"| **no vessel** ({fp + tn}) | {fp} | {tn} |",
            "",
            f"* **`n_vessels_est` (decision 0031, the estimator to use): "
            f"{sum(est)} predicted against {sum(man)} counted, "
            f"r = {r_est:+.3f}**, "
            f"{sum(1 for a, b in zip(est, man) if abs(a - b) <= 1)}/{len(est)} "
            "windows within +/-1.",
            f"* Merged EVENT count, for contrast: {sum(mrg)} against "
            f"{sum(man)}, r = {r:+.3f}. Event counts are excursions, not "
            "passages -- this is the gap decision 0031 closes.",
            f"* Fired on {tp}/{tp + fn} windows with a vessel; fired on "
            f"{fp}/{fp + tn} windows with none.",
            "",
            "**READ THIS BEFORE CALLING THE SECOND NUMBER A FALSE-POSITIVE "
            f"RATE.** It rests on {fp + tn} negative windows, so its uncertainty "
            "is enormous. More importantly the two instruments do not measure "
            "the same thing: the manual count is what a human can see INSIDE THE "
            "IMAGE FOOTPRINT at ONE INSTANT, while the acoustic window is "
            "+/-15 min and the hydrophone's range is unmeasured (goal G3) and "
            "probably exceeds the footprint. A vessel-free scene can therefore "
            "be acoustically occupied by a boat that was out of frame, or "
            "arrived ten minutes later, with no error by either instrument. "
            "The same asymmetry cuts the other way for the silent-but-occupied "
            "windows: a vessel visible in the image may be drifting, anchored, "
            "or too distant to hear.",
            "",
            f"**What the near-zero correlation does say** is that merged event "
            "count carries little information about how many vessels were "
            "visible. That is consistent with the time-shift null (decision "
            "0030), which also found event counts to be roughly as common away "
            "from the overpass as at it. Two independent comparisons now point "
            "the same way, and neither is explained by a coding error: the "
            "frame-shuffle null still collapses to zero, so the events are real "
            "temporal structure. They are simply not, on this evidence, "
            "vessel-count structure.",
            "",
        ]
    lines += [
        "## Merge-gap sensitivity",
        "",
        "How the merged total depends on the one free parameter. **There is no "
        "plateau corpus-wide** -- the total declines steadily across the sweep, "
        "so the gap is a consequential convention and not a value the data picks "
        "out. Individual dense windows DO plateau (20230719_182055_22_2449 gives "
        "3 ship_proxy events at every gap from 5 s to 120 s), but that does not "
        "generalise. **Never quote a merged count without the gap that produced "
        "it.**",
        "",
        "| merge gap (s) | " + " | ".join(f"`{b}`" for b in DETECTION_BANDS)
        + " | total |",
        "|---|" + "---|" * (len(DETECTION_BANDS) + 1),
    ]
    for gap in MERGE_GAP_SWEEP_S:
        per = []
        for b in DETECTION_BANDS:
            per.append(sum(sweep.get((r["scene_id"], b, gap), 0) for r in scored
                           if r["band"] == b))
        mark = "  <- used" if gap == run_b5_gate.EVENT_MERGE_GAP_S else ""
        lines.append(f"| {gap:g}{mark} | " + " | ".join(str(v) for v in per)
                     + f" | **{sum(per)}** |")
    lines += [
        "",
        "## Per window",
        "",
        "One row per overpass. `total` sums the DETECTION bands only -- the rain "
        "and control bands are diagnostic and explain a detection rather than "
        "adding one.",
        "",
        f"**raw / merged**: a vessel's level fluctuates, and every dip below the "
        f"+{run_b5_gate.EVENT_EXCESS_THRESHOLD_COUNTS:.0f}-count threshold ends "
        f"one event and starts another, so the RAW count measures threshold "
        f"chatter as much as it measures passages. MERGED joins fragments "
        f"separated by no more than {run_b5_gate.EVENT_MERGE_GAP_S:.0f} s "
        "(`run_b5_gate.merge_events`) and counts DISTINCT ELEVATED PERIODS. "
        "Merged on time alone, never on amplitude similarity -- a real closest "
        "approach rises and falls, so fragments of one pass have systematically "
        "different peaks and an amplitude gate would refuse to merge them. "
        "Neither count is a vessel count.",
        "",
        "| scene | acquired (UTC) | **manual vessels** | **est. vessels "
        "(small_craft)** | small_craft raw/merged | ship_proxy raw/merged | "
        "total merged | optical cand. |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for w in (wide or []):
        def _c(key):
            v = w.get(key, "")
            return "--" if v == "" or v is None else str(v)
        lines.append(
            f"| `{w['scene_id']}` | {w['acquired_utc'][:16].replace('T', ' ')} | "
            f"**{_c('manual_vessel_count')}**"
            + ("*" if str(w.get('manual_label_shared_by') or 1) != "1" else "")
            + " | "
            f"**{_c('n_small_craft_vessels_est')}** | "
            f"{_c('n_small_craft')} / {_c('n_small_craft_merged')} | "
            f"{_c('n_ship_proxy')} / {_c('n_ship_proxy_merged')} | "
            f"{_c('total_detection_events_merged')} | "
            f"{_c('n_optical_candidates')} |")
    lines += [
        "",
        "`--` means not scored (no coverage), which is NOT zero. `*` marks a "
        "manual label SHARED with another scene (same minute, adjacent frames) "
        "-- do not sum that column down the rows, it would double-count.",
        "",
        "## Coverage",
        "",
        f"* {summary['n_full']} overpasses fully covered, {summary['n_partial']} "
        f"partial, {summary['n_zero']} with no acoustic data at all "
        f"(of {summary['n_overpasses']} scenes).",
        "* Uncovered windows appear in the CSV with `coverage = NONE` and a BLANK "
        "`n_events`. Blank is an absence; it is not zero and must never be summed "
        "as one.",
        "* Partial windows carry a lower bound, flagged in `note`.",
        "",
        "## What these numbers are not",
        "",
        "* **Not vessel counts.** `n_events` counts excess-over-ambient excursions "
        f"lasting at least {run_b5_gate.EVENT_MIN_DURATION_S:.0f} s "
        f"(`run_b5_gate.find_events`, threshold "
        f"+{run_b5_gate.EVENT_EXCESS_THRESHOLD_COUNTS:.0f} counts). One vessel can "
        "open several events and two vessels can merge into one. No "
        "event-to-vessel mapping has been measured.",
        "* **Not a false-positive rate.** 29 overpasses now carry a manual "
        f"vessel count (`{MANUAL_VALIDATION_RELPATH}`) including 6 zero-vessel "
        "windows -- see the comparison section above -- but 6 negatives will not "
        "support a rate, and the image footprint and the acoustic window do not "
        "cover the same region or the same instant.",
        "* **Not calibrated.** Levels are the product's own integer counts, never "
        "dB re 1 uPa; the measured counts/dB relation is ~0.52 with curvature, so "
        "no fixed conversion exists (decision 0027).",
        "* **Not a vessel count on the optical side either.** "
        f"`n_optical_candidates` comes from `{optical_source}` unfiltered "
        "(candidates, ~47% flagged `transient`); "
        "`n_optical_candidates_non_transient` drops the flagged ones and is still "
        "not a validated vessel count.",
        "* **Band edges are uncertain by "
        f"{config.FFT_AXIS_OFFSET_UNCERTAINTY_HZ:.0f} Hz** -- the FFT axis "
        "convention is an open assumption (decision 0013).",
        "",
        "## Sampling conditionality",
        "",
        config.PLANET_SAMPLING_CONDITIONALITY_STATEMENT,
        "",
        "The unit of analysis is the **overpass**, not the event: a window with "
        "eight events is one measurement of one ocean state, not eight.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", help="defaults to data/derived/events/<run_id>")
    parser.add_argument("--shift-hours", type=float, default=-1.0,
                        help="offset for the time-shift null (default -1 h; the "
                             "corpus does not extend +1 h past the overpasses)")
    args = parser.parse_args(argv)

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ensure_dir(pathlib.Path(args.out_dir) if args.out_dir
                         else DERIVED_DIR / "events" / run_id)

    overpass_list = ov.load_gate2_overpasses()
    # BOTH landing zones, exactly as build_review_set.py does: the bulk corpus
    # holds the 09:15-11:45 local strip and the top-up zone holds windows pulled
    # for a specific scene, which are off-strip by construction.
    index = ov.corpus_file_index()
    if paths.ONC_LABELLED_WINDOW_DIR.is_dir():
        extra = ov.corpus_file_index(paths.ONC_LABELLED_WINDOW_DIR)
        index = sorted(index + extra, key=lambda row: row[0])
        print(f"corpus {len(index) - len(extra):,} + labelled top-up {len(extra)} windows")
    coverages = [ov.window_coverage(o, index) for o in overpass_list]
    summary = ov.coverage_summary(coverages)
    print(f"coverage: {summary['n_full']} full, {summary['n_partial']} partial, "
          f"{summary['n_zero']} zero, of {summary['n_overpasses']}")

    optical, optical_source = load_optical_candidate_counts()
    manual, manual_source = load_manual_vessel_counts()

    bands = {
        "small_craft": config.FFT_B5_SMALL_CRAFT_BAND_HZ,
        "ship_proxy": config.FFT_B5_SHIP_PROXY_BAND_HZ,
        "rain": config.FFT_RAIN_BAND_HZ,
        "control": config.FFT_CONTROL_BAND_HZ,
    }

    rows = []
    for cov in coverages:
        for band_name, band_hz in bands.items():
            rows.append(band_row(cov, band_name, band_hz, index,
                                 args.shift_hours, optical, manual))
        det = [r for r in rows[-len(bands):] if r["is_detection_band"]]
        n_ev = sum(r["n_events"] for r in det if r["n_events"] != "")
        state = ("no data" if cov.n_files == 0
                 else f"{n_ev} event(s) in detection bands")
        print(f"  {cov.overpass.scene_id}  {cov.n_files:2d} files  {state}")

    table = out_dir / "events_by_window.csv"
    fieldnames = [k for k in rows[0] if not k.startswith("_")]
    with open(table, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # WIDE VIEW: one row per window, one column per band. The long table above
    # is the one to compute from (it carries the nulls, the censoring and the
    # per-band caveats); this is the one to read. `total_detection_events` sums
    # the DETECTION bands only -- adding the diagnostic rain and control bands
    # would count a weather explanation as a second detection.
    wide_path = out_dir / "events_by_window_wide.csv"
    by_scene = {}
    for r in rows:
        w = by_scene.setdefault(r["scene_id"], {
            "scene_id": r["scene_id"],
            "acquired_utc": r["acquired_utc"],
            "coverage": r["coverage"],
            "n_optical_candidates": r["n_optical_candidates"],
            "manual_vessel_count": r["manual_vessel_count"],
            "manual_label_shared_by": r["manual_label_shared_by"],
        })
        w[f"n_{r['band']}"] = r["n_events"]
        w[f"n_{r['band']}_merged"] = r["n_events_merged"]
        w[f"n_{r['band']}_vessels_est"] = r["n_vessels_est"]
        w[f"n_{r['band']}_frame_shuffled"] = r["n_events_frame_shuffled"]
    for w in by_scene.values():
        det = [w.get(f"n_{b}") for b in DETECTION_BANDS]
        # Blank if ANY detection band went unscored: a total over a partial set
        # of bands is not a total, and writing one would understate silently.
        w["total_detection_events"] = (
            sum(det) if all(v != "" and v is not None for v in det) else "")
        detm = [w.get(f"n_{b}_merged") for b in DETECTION_BANDS]
        w["total_detection_events_merged"] = (
            sum(detm) if all(v != "" and v is not None for v in detm) else "")
        w["total_detection_events_frame_shuffled"] = sum(
            w.get(f"n_{b}_frame_shuffled") or 0 for b in DETECTION_BANDS
            if w.get(f"n_{b}_frame_shuffled") != "")
    wide_fields = (["scene_id", "acquired_utc", "coverage"]
                   + [x for b in DETECTION_BANDS
                      for x in (f"n_{b}", f"n_{b}_merged", f"n_{b}_vessels_est")]
                   + ["total_detection_events", "total_detection_events_merged",
                      "total_detection_events_frame_shuffled"]
                   + [f"n_{b}" for b in bands if b not in DETECTION_BANDS]
                   + ["n_optical_candidates", "manual_vessel_count",
                      "manual_label_shared_by"])
    with open(wide_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=wide_fields,
                                extrasaction="ignore")
        writer.writeheader()
        for scene in sorted(by_scene):
            writer.writerow(by_scene[scene])

    sweep = {}
    for r in rows:
        for gap, n in (r.pop("_merge_sweep", {}) or {}).items():
            sweep[(r["scene_id"], r["band"], gap)] = n

    write_summary_md(out_dir / "events_by_window_summary.md", rows, summary,
                     args.shift_hours, optical_source or "(absent)", run_id,
                     wide=[by_scene[k] for k in sorted(by_scene)], sweep=sweep)

    (out_dir / "provenance.json").write_text(json.dumps({
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "produced_by": "scripts/build_events_table.py",
        "corpus_dirs": [str(ov.ONC_OVERPASS_CORPUS_DIR),
                        str(paths.ONC_LABELLED_WINDOW_DIR)],
        "n_corpus_files_indexed": len(index),
        "scene_list": config.PLANET_GATE2_SURVIVORS_RELPATH,
        "optical_detections": optical_source,
        "manual_validations": manual_source,
        "coverage": summary,          # INCLUDING the scene-id lists
        "bands_hz": {k: list(v) for k, v in bands.items()},
        "detection_bands": list(DETECTION_BANDS),
        "event_threshold_counts": run_b5_gate.EVENT_EXCESS_THRESHOLD_COUNTS,
        "event_min_duration_s": run_b5_gate.EVENT_MIN_DURATION_S,
        "event_definition": "scripts/run_b5_gate.find_events -- the ONE definition",
        "null_time_shift_hours": args.shift_hours,
        "band_level_statistic": config.FFT_BAND_LEVEL_STATISTIC,
        "axis_convention": config.FFT_AXIS_CONVENTION,
        "axis_offset_uncertainty_hz": config.FFT_AXIS_OFFSET_UNCERTAINTY_HZ,
        "level_units": "product COUNTS, uncalibrated -- not dB re 1 uPa",
        "sampling_conditionality": config.PLANET_SAMPLING_CONDITIONALITY_STATEMENT,
        "blank_is_not_zero": (
            "A blank n_events means the window was not covered and the detector "
            "never ran. A blank null column means that null was not evaluable. "
            "Neither is a measured zero."),
    }, indent=1, default=str), encoding="utf-8")

    scored = [r for r in rows if r["n_events"] != "" and r["is_detection_band"]]
    n_ev = sum(r["n_events"] for r in scored)
    n_sh = sum(r["n_events_frame_shuffled"] for r in scored
               if r["n_events_frame_shuffled"] != "")
    print(f"\nn = {n_ev} events across {len({r['scene_id'] for r in scored})} "
          f"overpasses (detection bands), {n_sh} under the frame-shuffle null")
    print(f"table   -> {table}")
    print(f"summary -> {out_dir / 'events_by_window_summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
