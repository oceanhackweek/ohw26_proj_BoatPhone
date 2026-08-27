#!/usr/bin/env python3
"""Image chips for the windows where the hull selector is ambiguous.

Detector A locates a moving vessel by its wake; Detector B measures the hull. When
more than one B blob falls inside A's window, something has to decide which is the
boat. Brightness picks wake foam (foam is bright in NIR); compactness picks 2-pixel
specks (a 2 px blob fills its own rectangle, so fill == 1.0 by construction). Both
are degenerate at opposite ends of the size range.

Median window holds ONE candidate, so this is not an algorithm problem -- it is a
handful of human decisions. This renders those windows so a person can settle them,
and writes the candidate table alongside so the verdict can be recorded per blob.

Chips are TOA RGB under the same fixed L1C TCI scale the detector saw, upscaled with
NEAREST so a 2 px hull stays 2 px and is not smoothed into something it is not.
"""
import argparse, csv, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from boatphone import optical
from detector_a_toa_test import reflectance_coefficients, tci_rgb

HULL_MARGIN_M = 30.0
CHIP_HALF_M = 300.0      # 600 m across: a 147 m wake plus context
UPSCALE = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--toa-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-aspect", type=float, default=2.0)
    ap.add_argument("--only-ambiguous", action="store_true",
                    help="Only windows with >1 candidate, i.e. where a choice exists")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    A = [r for r in csv.DictReader(open(args.a))
         if float(r["aspect_ratio"]) >= args.min_aspect]
    by_scene = defaultdict(list)
    for r in csv.DictReader(open(args.b)):
        by_scene[r["scene_id"]].append(r)
    toa = {p.name.split("_3B_")[0]: p
           for p in Path(args.toa_dir).rglob("*_AnalyticMS_clip.tif")}

    rows, made = [], 0
    for a in A:
        sid = a["scene_id"]
        alon, alat = float(a["lon"]), float(a["lat"])
        radius = 0.5 * max(float(a["bbox_w_m"]), float(a["bbox_h_m"])) + HULL_MARGIN_M
        mlat = math.radians(alat)
        cand = []
        for c in by_scene.get(sid, []):
            dx = (float(c["lon"]) - alon) * 111_320.0 * math.cos(mlat)
            dy = (float(c["lat"]) - alat) * 110_540.0
            d = math.hypot(dx, dy)
            if d <= radius:
                cand.append((d, c))
        if args.only_ambiguous and len(cand) < 2:
            continue
        cand.sort()

        with rasterio.open(toa[sid]) as src:
            xml = str(toa[sid]).replace("_AnalyticMS_clip.tif",
                                        "_AnalyticMS_metadata_clip.xml")
            rgb, _valid, _m = tci_rgb(toa[sid], reflectance_coefficients(xml))
            res = abs(src.transform.a)
            def to_rc(lon, lat):
                xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
                r, c = rasterio.transform.rowcol(src.transform, xs[0], ys[0])
                return int(r), int(c)
            r0, c0 = to_rc(alon, alat)

        half = int(CHIP_HALF_M / res)
        rlo, rhi = max(0, r0 - half), min(rgb.shape[0], r0 + half)
        clo, chi = max(0, c0 - half), min(rgb.shape[1], c0 + half)
        chip = rgb[rlo:rhi, clo:chi]
        im = Image.fromarray(chip).resize(
            (chip.shape[1] * UPSCALE, chip.shape[0] * UPSCALE), Image.NEAREST)
        d = ImageDraw.Draw(im)

        # A's wake box, in yellow
        wpx = float(a["bbox_w_m"]) / res / 2.0; hpx = float(a["bbox_h_m"]) / res / 2.0
        ax = (c0 - clo) * UPSCALE; ay = (r0 - rlo) * UPSCALE
        d.rectangle([ax - wpx * UPSCALE, ay - hpx * UPSCALE,
                     ax + wpx * UPSCALE, ay + hpx * UPSCALE], outline=(255, 220, 0), width=2)
        d.text((4, 4), f"{sid}  wake {float(a['length_class_m']):.0f} m  "
                       f"conf {float(a['confidence']):.2f}  range {float(a['range_km']):.2f} km",
               fill=(255, 220, 0))

        for k, (dist, c) in enumerate(cand, 1):
            br, bc = to_rc(float(c["lon"]), float(c["lat"]))
            bx = (bc - clo) * UPSCALE; by = (br - rlo) * UPSCALE
            d.ellipse([bx - 9, by - 9, bx + 9, by + 9], outline=(255, 60, 60), width=2)
            d.text((bx + 11, by - 6), str(k), fill=(255, 60, 60))
            rows.append({"chip": f"{sid}_{made:02d}.png", "scene_id": sid,
                         "candidate": k, "offset_m": round(dist, 1),
                         "length_m": c["length_class_m"], "width_m": c["width_m"],
                         "fill": c["fill"], "n_px": c["n_px"], "nir_snr": c["nir_snr"],
                         "size_class": c["size_class"], "verdict": ""})
        im.save(out / f"{sid}_{made:02d}.png")
        made += 1

    with open(out / "candidates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{made} chips -> {out}")
    print(f"{len(rows)} candidate blobs listed -> {out}/candidates.csv  (fill in `verdict`)")


if __name__ == "__main__":
    main()
