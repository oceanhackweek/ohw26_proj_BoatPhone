#!/usr/bin/env python3
"""Detector A over the 26 delivered TOA scenes -> detections_a_toa.csv.

Uses ortho_analytic_4b (TOP-OF-ATMOSPHERE) rendered through the FIXED L1C TCI
scale. Measured on 20200730: TOA water lands at DN 50.0 against S2 L1C TCI's
~50-100, where surface reflectance sat at 13 and Planet's `visual` render at 1 --
and all three wakes visible in that scene are detected at conf 0.51-0.63, having
been missed entirely by both other renderings. See the source of truth.

Radiometry helpers are imported from detector_a_toa_test.py rather than copied:
ortho_analytic_4b is TOA RADIANCE and needs the per-band reflectanceCoefficient
from the scene XML before the TCI recipe applies, while ortho_analytic_4b_sr does
not. One definition of that asymmetry, not two (CLAUDE.md invariant 6).
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from boatphone import optical, vessel_detect as vd
from detector_a_toa_test import reflectance_coefficients, tci_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--toa-dir", required=True)
    ap.add_argument("--sr-dir", default=None,
                    help="If given, apply optical.water_mask (plan section 6.5). The mask "
                         "is built from the _sr product ON PURPOSE: LAND_NIR_MIN is an "
                         "ABSOLUTE reflectance threshold (0.05) calibrated on surface "
                         "reflectance in decision 0017, and TOA reflectance is "
                         "systematically higher because path radiance is still in it. "
                         "Same scene, same grid, same clip, so the mask transfers "
                         "directly -- and there stays ONE water-mask definition.")
    ap.add_argument("--slice", type=int, default=640)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--out", default="detections_a_toa.csv")
    ap.add_argument("--preview-dir", default=None)
    args = ap.parse_args()

    toa = {p.name.split("_3B_")[0]: p
           for p in Path(args.toa_dir).rglob("*_AnalyticMS_clip.tif")}
    sr = ({p.name.split("_3B_")[0]: p
           for p in Path(args.sr_dir).rglob("*_AnalyticMS_SR_clip.tif")}
          if args.sr_dir else {})
    model, _ = vd.load_model("cpu")
    rows = list(csv.DictReader(open(args.manifest)))
    dets = []

    for i, r in enumerate(rows, 1):
        sid = r["id"]
        path = toa.get(sid)
        if path is None:
            print(f"[{i:>2}/{len(rows)}] {sid}  MISSING TOA clip -- skipped"); continue
        xml = str(path).replace("_AnalyticMS_clip.tif", "_AnalyticMS_metadata_clip.xml")
        rgb, valid, meta = tci_rgb(path, reflectance_coefficients(xml))
        udm2 = str(path).replace("_3B_AnalyticMS_clip.tif", "_3B_udm2_clip.tif")
        clear = vd.read_udm2_clear(udm2, valid.shape)
        keep, wfrac = clear, float("nan")
        if sr.get(sid) is not None:
            import rasterio
            with rasterio.open(sr[sid]) as ss:
                srr = ss.read().astype(np.float32) / optical.PLANET_QUANT
            water, _rep = optical.water_mask(srr[1], srr[3], valid, clear,
                                             res_m=meta["res_x"])
            keep = water                      # water_mask already folds in `clear`
            wfrac = float(water.mean())
        boxes = vd.predict_tiled(model, rgb, valid, args.slice, args.imgsz,
                                 args.conf, 0.2, 0.5, bgr=True)
        recs = vd.boxes_to_records(boxes, meta, sid, r["acq_time_utc"], valid, keep,
                                   optical.HYDROPHONE_LONLAT, 0.0, 150.0, 2.5)
        for d in recs:
            d["size_class"] = optical.size_class(d["length_class_m"])
            d["detector"] = "A"
            d.setdefault("nir_snr", "")
        dets.extend(recs)
        wdn = float(np.median(rgb[valid])) if valid.any() else float("nan")
        print(f"[{i:>2}/{len(rows)}] {sid}  waterDN {wdn:5.1f}  water {wfrac:5.1%}  "
              f"raw {len(boxes):>3}  kept {len(recs):>3}", flush=True)
        if args.preview_dir:
            Path(args.preview_dir).mkdir(parents=True, exist_ok=True)
            vd.save_preview(rgb, boxes, f"{args.preview_dir}/{sid}.jpg")

    # Persistence: a detection recurring in the same ~50 m cell across different
    # YEARS is a fixed object. The residual TOA false positives cluster on islands
    # and shoreline, which is exactly what this removes without tuning anything.
    from collections import defaultdict
    DLAT, DLON = 0.00045, 0.00068
    years = defaultdict(set)
    for d in dets:
        d["cell"] = (round(d["lat"] / DLAT), round(d["lon"] / DLON))
        years[d["cell"]].add(d["acq_time_utc"][:4])
    for d in dets:
        d["n_years_at_cell"] = len(years[d.pop("cell")])
        d["transient"] = int(d["n_years_at_cell"] == 1)

    fields = list(optical.DETECTIONS_FIELDS) + ["size_class", "n_years_at_cell", "transient"]
    optical.write_csv(args.out, dets, fields)
    t = sum(d["transient"] for d in dets)
    print(f"\n{len(dets)} detections over {len(rows)} scenes -> {args.out}")
    print(f"{t} transient ({t/max(len(dets),1):.0%}), {len(dets)-t} in multi-year cells")


if __name__ == "__main__":
    main()
