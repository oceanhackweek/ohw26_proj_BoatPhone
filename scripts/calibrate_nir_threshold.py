#!/usr/bin/env python3
"""Sweep Detector B's threshold two ways over a whole batch, and measure the sweep.

Runnable ENTRY POINT. Every constant and all detection logic live in
`boatphone/optical.py`; this file defines nothing another notebook could disagree
with (CLAUDE.md invariant 6).

WHY THIS EXISTS. Neve measured that ocean NIR has a heavy tail, so `n_mad` is
close to a dead knob: 6 sigma -> 10 sigma only halves the surviving pixel count
where a Gaussian would drop it fourteen orders of magnitude. Sweeping N harder
cannot converge because there is no value to converge on. The proposed
replacement is a fixed CANDIDATE-PIXEL BUDGET per km2 of water, read off the
empirical distribution -- and a proposal about which threshold travels better
between scenes has to be settled by running both over the whole batch, not on
one scene.

WHAT IT MEASURES, per scene and per setting:
  * the cut itself   -- threshold, water median, sigma_MAD, and where the cut
    landed in sigmas (`n_mad_effective`). At a FIXED budget the spread of
    n_mad_effective across scenes IS the amount of scene-to-scene variation a
    fixed `n_mad` is blind to, and vice versa.
  * the workload     -- candidate pixels and surviving detections per scene.
  * the persistence split -- detections whose ~50 m cell is occupied in more than
    one YEAR are fixed objects (rock, islet, beacon), not boats. This is the one
    signal here that assumes nothing about the brightness distribution, so it is
    the closest thing to ground truth this batch has. It is scored ACROSS the
    whole batch, which is why the sweep has to be batch-wide.

WHAT IT DOES NOT MEASURE, and no threshold sweep can: recall against real
vessels. There is no truth set. `n_persistent` is a floor on real fixed objects
recovered, not a count of boats found, and plan section 6.2 still forbids ranking
on counts. The section 8 eyeball census remains the gate.

Usage:
    python3 scripts/calibrate_nir_threshold.py \\
        '/path/to/downloads/**/PSScene/*_AnalyticMS_SR_clip.tif' \\
        --out data/derived/nir_threshold_calibration.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from boatphone import optical as opt                      # noqa: E402
from export_detections import load_scene, radiometry_of    # noqa: E402

# One scene reader, not a fourth: export_detections.load_scene already handles
# SR/TOA radiometry, the band-order assertion and the lon/lat callable. It does
# not read UDM2, so the clear mask is read here -- the only raster work this
# file does itself.

SCENE_ID_RE = re.compile(r"(\d{8}_\d{6}_\d{2}(?:_\w+)?)_3B_AnalyticMS")


def scene_id_of(path: str) -> str:
    m = SCENE_ID_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(
            f"cannot read a scene id out of {os.path.basename(path)}. Expected a "
            "PlanetScope <YYYYMMDD_HHMMSS_..>_3B_AnalyticMS[_SR]_clip.tif name; "
            "renaming the file breaks the year the persistence filter needs.")
    return m.group(1)


def clear_mask(scene_path: str, shape: tuple[int, int]):
    """UDM2 band 1 == 1 means clear. Missing UDM2 is reported, never assumed clear."""
    import rasterio
    udm2 = re.sub(r"_3B_AnalyticMS(_SR)?_clip\.tif$", "_3B_udm2_clip.tif", scene_path)
    if not os.path.exists(udm2):
        return None
    with rasterio.open(udm2) as u:
        clear = u.read(1) == 1
    if clear.shape != shape:
        raise ValueError(
            f"{os.path.basename(udm2)} is {clear.shape}, scene is {shape}. A "
            "resampled or mismatched UDM2 would mask the wrong pixels.")
    return clear


def settings_from(budgets, n_mads):
    """The sweep, as (kind, value) pairs. Ordered cheap-first so a run killed
    part-way still leaves the tight budgets measured."""
    return ([("quantile", b) for b in budgets] + [("mad", n) for n in n_mads])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+",
                    help="analytic .tif paths or globs (SR or TOA)")
    ap.add_argument("--radiometry", choices=("auto", "sr", "toa"), default="auto")
    ap.add_argument("--budgets", default="5,10,20,40,80,160,320",
                    help="quantile-mode candidate-pixel budgets, px per km2 of water")
    ap.add_argument("--n-mad", default="6,10",
                    help="mad-mode sigma multiples. 6 is what the delivered batch ran.")
    ap.add_argument("--out", default="data/derived/nir_threshold_calibration.csv")
    ap.add_argument("--limit", type=int, default=0, help="first N scenes only (smoke test)")
    args = ap.parse_args()

    paths: list[str] = []
    for pat in args.scenes:
        paths.extend(sorted(glob.glob(pat, recursive=True)) if any(c in pat for c in "*?[")
                     else [pat])
    paths = sorted(set(paths))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        return print("No scenes matched.") or 2

    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    n_mads = [float(x) for x in args.n_mad.split(",") if x.strip()]
    settings = settings_from(budgets, n_mads)

    rows: list[dict] = []
    # (kind, value) -> list of (scene_id, year, cell) for the batch-wide persistence pass
    dets: dict[tuple, list] = defaultdict(list)

    for i, path in enumerate(paths, 1):
        sid = scene_id_of(path)
        year = sid[:4]
        sc = load_scene(path, args.radiometry)
        clear = clear_mask(path, sc["shape"])
        water, mrep = opt.water_mask(sc["green"], sc["nir"], sc["valid"], clear,
                                     res_m=sc["res_m"])
        print(f"[{i:>2}/{len(paths)}] {sid}  {radiometry_of(path, args.radiometry)}  "
              f"water {water.mean():5.1%}  udm2 {'yes' if clear is not None else 'MISSING'}",
              flush=True)
        if not water.any():
            # Not a scene with no boats in it -- a scene with no water. Say which.
            print(f"     SKIPPED: water mask is empty ({mrep})", flush=True)
            continue

        for kind, value in settings:
            kw = ({"threshold_mode": "quantile", "budget_px_per_km2": value}
                  if kind == "quantile" else {"threshold_mode": "mad", "n_mad": value})
            blobs, d = opt.detect_nir_blobs(sc["nir"], water, sc["res_m"], **kw)
            lon, lat = (sc["to_lonlat"]([b.col for b in blobs], [b.row for b in blobs])
                        if blobs else (np.empty(0), np.empty(0)))
            for x, y in zip(lon, lat):
                dets[(kind, value)].append(
                    (sid, year, (round(float(y) / opt.PERSISTENCE_CELL_DLAT),
                                 round(float(x) / opt.PERSISTENCE_CELL_DLON))))
            rows.append({
                "scene_id": sid, "year": year,
                "radiometry": radiometry_of(path, args.radiometry),
                "mode": kind, "setting": value,
                "water_km2": round(d["water_km2"], 3),
                "water_px": d["water_px"],
                "threshold_reflectance": round(d["threshold_reflectance"], 6),
                "water_median": round(d["water_median"], 6),
                "sigma_mad": round(d["sigma_mad"], 6),
                "n_mad_effective": round(d["n_mad_effective"], 3),
                "budget_px": d["budget_px"],
                "budget_selected_px": d["budget_selected_px"] or "",
                "candidate_px": d["candidate_px"],
                "raw_components": d["raw"],
                "kept": d["kept"],
                "rej_too_few_px": d["too_few_px"],
                "rej_sub_pixel_length": d["sub_pixel_length"],
                "rej_implausible_length": d["implausible_length"],
                "rej_too_elongated": d["too_elongated"],
                "rej_too_sparse": d["too_sparse"],
            })
            print(f"       {kind:>8} {value:>6g}  thr {d['threshold_reflectance']:.5f}  "
                  f"N_eff {d['n_mad_effective']:5.2f}  cand {d['candidate_px']:>7,}  "
                  f"kept {d['kept']:>6,}", flush=True)

    # ------------------------------------------------------------ persistence
    # Batch-wide: a cell is PERSISTENT if it is occupied in more than one year.
    # Scored separately per setting, because the setting decides which detections
    # exist to be gridded in the first place.
    for key, recs in dets.items():
        years_at = defaultdict(set)
        for _, year, cell in recs:
            years_at[cell].add(year)
        n_persistent = defaultdict(int)
        for sid, _, cell in recs:
            if len(years_at[cell]) > 1:
                n_persistent[sid] += 1
        persistent_cells = sum(1 for c, ys in years_at.items() if len(ys) > 1)
        for r in rows:
            if (r["mode"], r["setting"]) == key:
                r["n_persistent"] = n_persistent.get(r["scene_id"], 0)
                r["n_transient"] = r["kept"] - r["n_persistent"]
                r["batch_persistent_cells"] = persistent_cells

    fields = list(rows[0].keys())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    opt.write_csv(args.out, rows, fields)
    print(f"\n{len(rows)} rows ({len(paths)} scenes x {len(settings)} settings) -> {args.out}")

    # -------------------------------------------------------------- summary
    # The comparison the whole run is for: at each setting, how far apart are the
    # scenes? A threshold that travels well between scenes has a low spread in
    # detections per km2; one that does not is measuring sea state.
    print(f"\n{'mode':>8} {'setting':>8} {'det/km2 median':>15} {'p10..p90':>18} "
          f"{'spread':>8} {'persist.cells':>14} {'%persistent':>12}")
    for kind, value in settings:
        sel = [r for r in rows if (r["mode"], r["setting"]) == (kind, value)]
        if not sel:
            continue
        per_km2 = np.array([r["kept"] / r["water_km2"] for r in sel])
        lo, med, hi = np.percentile(per_km2, [10, 50, 90])
        tot_keep = sum(r["kept"] for r in sel)
        tot_pers = sum(r["n_persistent"] for r in sel)
        print(f"{kind:>8} {value:>8g} {med:>15.2f} {lo:>8.2f}..{hi:<8.2f} "
              f"{hi / max(lo, 1e-9):>7.1f}x {sel[0]['batch_persistent_cells']:>14,} "
              f"{100 * tot_pers / max(tot_keep, 1):>11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
