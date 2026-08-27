#!/usr/bin/env python3
"""Detector B over the delivered PlanetScope batch -> detections.csv + scenes.csv.

Runnable ENTRY POINT (CLAUDE.md invariant 6): all constants and all detection
logic live in `boatphone/optical.py`; nothing is defined here that another
notebook could disagree with.

Why Detector B and not Detector A: B screens NIR by contrast and needs no model
weights, no AGPL decision and no RGB transform. Detector A on this batch found
four shoreline blobs and missed three visible wakes (see the source of truth),
and its RGB input path is still unresolved. B is the path that runs today.

WHAT THIS EMITS, and what it does NOT:
  * acq_time_utc  -- the scene acquisition instant, per detection. A PlanetScope
    scene is ONE snapshot, so every detection in a scene shares its timestamp.
  * range_km      -- great-circle distance from the detection centroid to the
    hydrophone, computed by optical.range_km against config's coordinate.
  * length_m / size_class -- from the blob's minimum-area rectangle, which is
    orientation-free, so it does NOT carry Detector A's axis-aligned bias.
  * orientation   -- NOT EMITTED. detect_nir_blobs computes a minAreaRect but
    Blob does not expose its angle, and re-deriving it here would put a second
    measurement of the same quantity in a second file. One field on Blob fixes it.
  * speed         -- NOT EMITTED AND NOT ESTIMABLE from a single snapshot. Do not
    infer it from length or wake_flag. The plan's Friday analysis uses wake
    presence (moving vs not), not a speed value.
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boatphone import optical


def load_scene(sr_path, udm2_path):
    """Read reflectance bands + the clear mask. Asserts the band count (optical.py
    warns that a 4-band assumption against an 8-band file reads RedEdge as NIR)."""
    with rasterio.open(sr_path) as s:
        assert s.count == 4, f"{sr_path}: expected 4 bands, got {s.count}"
        a = s.read().astype(np.float32) / optical.PLANET_QUANT
        tr, crs, res_m = s.transform, s.crs, abs(s.transform.a)
    blue, green, red, nir = a
    valid = a.sum(0) > 0

    clear = None
    if udm2_path and Path(udm2_path).exists():
        with rasterio.open(udm2_path) as u:
            clear = u.read(1) == 1

    def to_lonlat(cols, rows):
        xs, ys = rasterio.transform.xy(tr, list(rows), list(cols))
        lon, lat = rio_transform(crs, "EPSG:4326", xs, ys)
        return np.asarray(lon), np.asarray(lat)

    return green, nir, valid, clear, res_m, to_lonlat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--downloads", required=True)
    ap.add_argument("--n-mad", type=float, default=6.0,
                    help="NIR threshold in MADs. Neve measured N=3 ~125 false "
                         "candidates/scene; N=6 clears them and still recovers 6 m.")
    ap.add_argument("--out-detections", default="detections.csv")
    ap.add_argument("--out-scenes", default="scenes.csv")
    ap.add_argument("--out-queue", default="review_queue.csv")
    ap.add_argument("--queue-snr", type=float, default=20.0,
                    help="UNCALIBRATED ordering cut for the review queue only; it "
                         "does not filter detections.csv and is not a vessel test.")
    args = ap.parse_args()

    D = Path(args.downloads)
    rows = list(csv.DictReader(open(args.manifest)))
    dets, scenes = [], []

    for i, r in enumerate(rows, 1):
        green, nir, valid, clear, res_m, to_lonlat = load_scene(
            D / r["sr_file"], D / r["udm2_file"])
        water, mrep = optical.water_mask(green, nir, valid, clear, res_m=res_m)
        blobs, stats = optical.detect_nir_blobs(nir, water, res_m, n_mad=args.n_mad)
        recs = optical.blobs_to_records(blobs, r["id"], r["acq_time_utc"], to_lonlat)
        for d in recs:
            d["size_class"] = optical.size_class(d["length_class_m"])
        dets.extend(recs)
        scenes.append(optical.classify_scene(recs, r["id"], r["acq_time_utc"]))
        print(f"[{i:>2}/{len(rows)}] {r['id']}  water {water.mean():5.1%}  "
              f"land_comp {mrep.land_components:>3}  blobs {len(blobs):>3}  "
              f"{'' if blobs else '(none)'}", flush=True)

    # ---------------------------------------------------------- persistence
    # A detection recurring at the same coordinates across DIFFERENT YEARS is a
    # fixed object -- rock, islet, beacon -- not a boat. This substitutes for the
    # national rock layers (`vesikivi`, `vesikivikko`) that mayrajeo/ship-detection
    # filters against and we do not have. It is principled in a way a brightness
    # threshold is not: nothing here is tuned to make the counts look plausible.
    # Cells are ~50 m, wide enough to absorb inter-scene georeferencing jitter.
    DLAT, DLON = 0.00045, 0.00068
    from collections import defaultdict
    years = defaultdict(set)
    for d in dets:
        d["cell"] = (round(d["lat"] / DLAT), round(d["lon"] / DLON))
        years[d["cell"]].add(d["acq_time_utc"][:4])
    for d in dets:
        d["n_years_at_cell"] = len(years[d.pop("cell")])
        d["transient"] = int(d["n_years_at_cell"] == 1)

    fields = list(optical.DETECTIONS_FIELDS) + ["size_class", "n_years_at_cell",
                                                "transient"]
    optical.write_csv(args.out_detections, dets, fields)

    # ---------------------------------------------------------- review queue
    # NOT a vessel list. The SNR cut here is UNCALIBRATED -- there is no truth set
    # yet, and plan section 6.2 forbids ranking on counts. Its only job is to make
    # the section 8 eyeball census finishable today: transient candidates ordered
    # by NIR SNR, strongest first. Verify top-down and stop where it turns to noise;
    # where you stop IS the measurement, and it is what calibrates the threshold.
    queue = sorted((d for d in dets if d["transient"] and d["nir_snr"] >= args.queue_snr),
                   key=lambda d: -d["nir_snr"])
    optical.write_csv(args.out_queue, queue, fields)
    print(f"{len(queue)} review candidates (transient, snr >= {args.queue_snr}) "
          f"-> {args.out_queue}")
    optical.write_csv(args.out_scenes, scenes, optical.SCENES_FIELDS)
    print(f"\n{len(dets)} detections over {len(rows)} scenes -> {args.out_detections}")
    print(f"{len(scenes)} scene rows -> {args.out_scenes}")


if __name__ == "__main__":
    main()
