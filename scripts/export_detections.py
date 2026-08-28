#!/usr/bin/env python3
"""Export Detector A's bounding boxes to one CSV per image.

For each scene: run the frozen pretrained YOLO, and write every box it returns as
a row carrying the four corner coordinates, an identifier, the range and bearing
to the Folger Deep hydrophone, and the confidence score.

WHAT THIS IS FOR. `detections.csv` (plan section 10) carries centroids and is frozen
for the acoustics join. That is the right shape for a join and the wrong shape for
a human: to judge a detection you need to see the BOX, because the box is the
evidence and the point is only its summary. This file is the QC and hand-off
artefact, kept deliberately separate so the frozen schema does not grow columns.

NOTHING HERE FILTERS. No size cut, no water mask, no Detector B. Every box the
model returns becomes a row. The NIR columns are annotation on those rows and
never remove one -- `--no-nir` writes the same rows without them, which the
self-test asserts.

Usage:
    python3 scripts/export_detections.py data/raw/*.tif --out-dir data/derived/detections
    python3 scripts/export_detections.py <scene.tif> --slice 640 --imgsz 640 --rgb

Two caveats the CSV header repeats, because they change how the numbers read:

  * The boxes are LOOSE. On 20200730_192942_71_1059 they run 7-14 px for vessels
    that should be 4-8 px at 3 m GSD, with only ~10% of the enclosed pixels
    NIR-bright. A box bounds a region containing the target; it does not outline
    a hull.
  * `length_m` is BIASED LOW for a rotated vessel. An axis-aligned box measures
    L|cos t| + W|sin t|, so a 24 x 4 m hull at 45 degrees reads 19.8 m -- short by
    17.5%, which pushes 12-24 m vessels into 0-12 m. That is the bin truncation
    decision 0016 exists to prevent. Where Detector B sees the same vessel, its
    minAreaRect length is unbiased and should be preferred.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boatphone import optical as opt                      # noqa: E402
from boatphone.vessel_detect import (                     # noqa: E402
    load_model, predict_tiled, parse_scene_meta,
)

HEADER_NOTE = (
    "# Detector A boxes, one row per model detection, nothing filtered.\n"
    "# length_m is biased LOW for rotated vessels (axis-aligned box); prefer\n"
    "# Detector B's minAreaRect length where available. See decision 0016.\n"
    "# possible_false_positive means UNSUPPORTED BY NIR, not 'not a boat':\n"
    "# a dark hull on dark water produces exactly this. See decision 0017.\n"
    "# bearing_deg_from_hydrophone is the hydrophone->detection azimuth,\n"
    "# NOT the vessel's heading -- an axis-aligned box carries no heading.\n"
)


def reflectance_coefficients(xml_path):
    """Per-band DN -> TOA reflectance factors from Planet's scene metadata.

    ortho_analytic_4b is TOA *RADIANCE* in DN, NOT reflectance. The TCI recipe
    below assumes reflectance x PLANET_QUANT, so applying it to radiance saturates
    the scene -- measured on 20200730: water at DN 236 of 255, and detections that
    mean nothing. The XML carries <ps:reflectanceCoefficient> per band (~2e-5).
    ortho_analytic_4b_sr needs no coefficient because it IS already reflectance x
    PLANET_QUANT. That asymmetry between the two products is the whole trap.
    """
    txt = Path(xml_path).read_text()
    bands = re.findall(r"<ps:bandNumber>(\d+)</ps:bandNumber>", txt)
    coefs = re.findall(r"<ps:reflectanceCoefficient>([0-9.eE+-]+)</ps:reflectanceCoefficient>", txt)
    if len(bands) != len(coefs) or len(coefs) < 4:
        raise ValueError(f"{xml_path}: {len(bands)} bandNumber, {len(coefs)} "
                         f"reflectanceCoefficient -- cannot convert radiance to reflectance")
    return {int(b): float(c) for b, c in zip(bands, coefs)}


def radiometry_of(path, override="auto"):
    """'sr' or 'toa', from Planet's canonical filenames unless overridden.

    ortho_analytic_4b_sr -> *_AnalyticMS_SR_clip.tif
    ortho_analytic_4b    -> *_AnalyticMS_clip.tif
    Both bundles ship an identically named *_AnalyticMS_metadata_clip.xml, so the
    raster name is the only reliable discriminator.
    """
    if override != "auto":
        return override
    return "sr" if "_AnalyticMS_SR" in os.path.basename(path) else "toa"


def load_scene(path, radiometry="auto"):
    """Planet analytic 4-band (SR or TOA) -> the arrays and callables optical.py needs.

    WHICH PRODUCT TO PREFER: the model was trained on Sentinel-2 L1C, which is
    TOP-OF-ATMOSPHERE. Over water most of the TOA signal is atmospheric path
    radiance, so surface reflectance removes the bulk of what it learned. Measured
    on 20200730 at slice 640 -> imgsz 640, conf 0.05: water DN 50 and 46 detections
    on TOA against DN 13 and 17 on SR, and only TOA boxes the three wakes visible
    in the scene. Feed this TOA where you have it.
    """
    import rasterio
    from rasterio.transform import xy as rio_xy
    from rasterio.warp import transform as rio_transform

    with rasterio.open(path) as src:
        if src.count < 4:
            raise ValueError(
                f"{os.path.basename(path)} has {src.count} bands; this expects the "
                f"4-band analytic_sr product (blue, green, red, nir). A `visual` "
                f"asset will not do -- it carries no NIR.")
        if src.descriptions and src.descriptions[:4] != (None,) * 4:
            expected = ("blue", "green", "red", "nir")
            if tuple(d.lower() if d else "" for d in src.descriptions[:4]) != expected:
                raise ValueError(
                    f"band descriptions are {src.descriptions[:4]}, expected {expected}. "
                    f"Reading band 4 as NIR would be wrong; SuperDove's 8-band product "
                    f"puts NIR at band 8.")
        arr = src.read().astype(np.float32)
        tf, crs, res = src.transform, src.crs, abs(src.transform.a)
        shape = (src.height, src.width)

    mode = radiometry_of(path, radiometry)
    if mode == "toa":
        xml = re.sub(r"_AnalyticMS_clip\.tif$", "_AnalyticMS_metadata_clip.xml", path)
        if not os.path.exists(xml):
            raise ValueError(
                f"{os.path.basename(path)} looks like TOA (ortho_analytic_4b) but "
                f"{os.path.basename(xml)} is missing. The per-band reflectance "
                f"coefficients live there and radiance cannot be converted without "
                f"them. Pass --radiometry sr if this really is a surface-reflectance "
                f"file under a non-standard name.")
        coefs = reflectance_coefficients(xml)
        bands = [np.clip(arr[i] * coefs[i + 1], 0, None) for i in range(4)]
    else:
        bands = [np.clip((arr[i] + opt.PLANET_SR_OFFSET) / opt.PLANET_QUANT, 0, None)
                 for i in range(4)]
    valid = np.logical_and.reduce([arr[i] > 0 for i in range(4)])
    for band in bands:
        band[~valid] = 0.0
    blue, green, red, nir = bands

    def to_xy(cols, rows):
        x, y = rio_xy(tf, list(rows), list(cols), offset="center")
        return np.asarray(x), np.asarray(y)

    def to_lonlat(cols, rows):
        x, y = to_xy(cols, rows)
        lon, lat = rio_transform(crs, "EPSG:4326", list(x), list(y))
        return np.asarray(lon), np.asarray(lat)

    rgb8 = np.ascontiguousarray(np.clip(
        np.stack([red, green, blue], -1) * opt.PLANET_QUANT / opt.TCI_DIVISOR,
        0, 255).astype(np.uint8))
    # `green` is returned as float reflectance, not just folded into rgb8: the
    # water mask needs NDWI, and rgb8 is 8-bit and clipped, so re-deriving green
    # from it would quantise the mask. Purely additive -- Detector A ignores it.
    return dict(rgb8=rgb8, nir=nir, green=green, valid=valid, res_m=res, shape=shape,
                crs=crs, to_lonlat=to_lonlat, to_xy=to_xy)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenes", nargs="+",
                    help="Planet analytic 4-band GeoTIFFs, SR or TOA (globs ok)")
    ap.add_argument("--radiometry", choices=("auto", "sr", "toa"), default="auto",
                    help="auto reads Planet's filename: *_AnalyticMS_SR_clip.tif is "
                         "surface reflectance, *_AnalyticMS_clip.tif is TOA radiance "
                         "and is converted with the XML reflectance coefficients. "
                         "PREFER TOA: the model was trained on Sentinel-2 L1C, which "
                         "is top-of-atmosphere.")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--slice", type=int, default=opt.SLICE_PX)
    ap.add_argument("--imgsz", type=int, default=opt.IMGSZ)
    ap.add_argument("--conf", type=float, default=opt.CONF)
    ap.add_argument("--iou", type=float, default=opt.NMS_IOU)
    ap.add_argument("--overlap", type=float, default=opt.TILE_OVERLAP)
    ap.add_argument("--rgb", action="store_true",
                    help="feed RGB instead of BGR (see plan section 6.1 -- an inverted "
                         "channel order is a silent accuracy killer, and the two "
                         "orders currently DISAGREE on this data)")
    ap.add_argument("--no-nir", action="store_true",
                    help="omit the NIR annotation columns; same rows either way")
    ap.add_argument("--device", default="cpu",
                    help="'cpu' (default) or 'cuda'. A full 100 km2 scene at 3 m "
                         "takes ~40 s on CPU at slice 320, ~20 s at slice 640, so "
                         "a GPU is not needed for a 26-scene batch")
    args = ap.parse_args()

    paths = []
    for pat in args.scenes:
        paths.extend(sorted(glob.glob(pat)) or
                     ([pat] if os.path.exists(pat) else []))
    if not paths:
        sys.exit("No scenes matched.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {opt.YOLO_HF_REPO}/{opt.YOLO_HF_FILE} ...")
    model, weights = load_model(args.device)

    fields = (opt.DETECTION_CORNER_FIELDS_NO_NIR if args.no_nir
              else opt.DETECTION_CORNER_FIELDS)
    config = {"slice_px": args.slice, "imgsz": args.imgsz,
              "conf_threshold": args.conf,
              "channel_order": "RGB" if args.rgb else "BGR",
              "weights_file": os.path.basename(weights)}

    for path in paths:
        # Per scene, not once: with --radiometry auto the mode comes from each
        # filename, so a mixed SR/TOA batch resolves differently file to file.
        config["radiometry"] = radiometry_of(path, args.radiometry)
        scene = load_scene(path, args.radiometry)
        scene_id, acq = parse_scene_meta(path)
        boxes = predict_tiled(model, scene["rgb8"], scene["valid"], args.slice,
                              args.imgsz, args.conf, args.overlap, args.iou,
                              bgr=not args.rgb)
        rows, thr = opt.boxes_to_corner_records(
            boxes, scene_id, acq, scene["res_m"], scene["to_lonlat"],
            to_xy=scene["to_xy"], shape=scene["shape"],
            nir=None if args.no_nir else scene["nir"],
            valid=scene["valid"], config=config)
        if args.no_nir:
            rows = [{k: v for k, v in r.items() if k in fields} for r in rows]

        out = out_dir / f"{scene_id}_detections.csv"
        with open(out, "w", newline="") as fh:
            fh.write(HEADER_NOTE)
            fh.write(f"# scene CRS {scene['crs']}; _x/_y are in it, _lon/_lat are EPSG:4326\n")
            fh.write(f"# hydrophone {opt.HYDROPHONE_LONLAT} -- {opt.HYDROPHONE_SOURCE}\n")
        with open(out, "a", newline="") as fh:
            import csv as _csv
            w = _csv.DictWriter(fh, fieldnames=list(fields))
            w.writeheader()
            w.writerows(rows)

        flagged = sum(1 for r in rows if r.get("possible_false_positive") == 1)
        extra = "" if args.no_nir else (
            f", {flagged} flagged as NIR-unsupported (advisory), "
            f"threshold rho {thr:.4f}")
        print(f"  {scene_id}: {len(rows)} detections -> {out}{extra}")


if __name__ == "__main__":
    main()
