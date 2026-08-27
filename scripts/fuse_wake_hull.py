#!/usr/bin/env python3
"""Measure the hull with Detector B in a window around each Detector A wake.

WHY THIS EXISTS RATHER THAN optical.fuse(). fuse() implements a SYMMETRIC
two-detector agreement test with a fixed tolerance, and appends every unmatched B
row. Three of its premises do not hold here:

  * Its comment claims A's box is the better extent estimate because the model
    "was trained to bound hulls". Measured: A's elongated detections are 70-150 m
    with aspect 2.0-2.5 -- those are WAKES. A gives LOCATION, not size.
  * A fixed tolerance cannot work when the hull sits at one END of a wake box
    whose length varies 30-150 m. The radius has to scale with A's box.
  * Appending unmatched B rows re-imports the 1,265-per-scene noise population
    that windowing exists to avoid.

So the contract here is ASYMMETRIC and deliberate:
    Detector A (TOA RGB)  -> WHERE a moving vessel is, and that it is moving
    Detector B (NIR)      -> HOW BIG its hull is, at that location
Each detector is used only for what it measures well. B is never asked to search,
which is why its noise problem does not follow it here.

This should be reconciled with optical.fuse() rather than living alongside it
forever -- flagged to Neve, not done unilaterally in her file.
"""
import argparse, csv, math
from collections import defaultdict

import numpy as np

HULL_MARGIN_M = 30.0   # slack beyond half A's box: georeferencing jitter + wake head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Detector A detections (TOA)")
    ap.add_argument("--b", required=True, help="Detector B detections (NIR)")
    ap.add_argument("--out", default="fused_wake_hull.csv")
    ap.add_argument("--min-aspect", type=float, default=0.0,
                    help="Restrict to elongated A rows (>=2.0 isolates wakes)")
    args = ap.parse_args()

    A = [r for r in csv.DictReader(open(args.a))
         if float(r["aspect_ratio"]) >= args.min_aspect]
    B = list(csv.DictReader(open(args.b)))
    by_scene = defaultdict(list)
    for r in B:
        by_scene[r["scene_id"]].append(r)

    # Local metric frame per scene. Equirectangular about the scene's own centre is
    # accurate to well under a metre over 10 km, which is far below the ~50 m radii
    # in play -- and it lets the match run in numpy instead of 1.1M geodesic calls.
    out, matched = [], 0
    for a in A:
        cand = by_scene.get(a["scene_id"], [])
        alon, alat = float(a["lon"]), float(a["lat"])
        radius = 0.5 * max(float(a["bbox_w_m"]), float(a["bbox_h_m"])) + HULL_MARGIN_M
        row = dict(a)
        row["hull_search_radius_m"] = round(radius, 1)
        if cand:
            mlat = math.radians(alat)
            blon = np.array([float(c["lon"]) for c in cand])
            blat = np.array([float(c["lat"]) for c in cand])
            dx = (blon - alon) * 111_320.0 * math.cos(mlat)
            dy = (blat - alat) * 110_540.0
            d = np.hypot(dx, dy)
            inside = np.flatnonzero(d <= radius)
        else:
            inside = np.array([], dtype=int)

        row["n_hull_candidates"] = int(inside.size)
        if inside.size:
            # Brightest NIR blob in the window is the hull: a vessel is the bright
            # object on dark water, which is Detector B's entire premise.
            best = max(inside, key=lambda j: float(cand[j]["nir_snr"]))
            h = cand[best]
            row["hull_length_m"] = float(h["length_class_m"])
            row["hull_aspect"] = float(h["aspect_ratio"])
            row["hull_size_class"] = h["size_class"]
            row["hull_nir_snr"] = float(h["nir_snr"])
            row["hull_offset_m"] = round(float(d[best]), 1)
            row["detector"] = "AB"
            matched += 1
        else:
            for k in ("hull_length_m", "hull_aspect", "hull_size_class",
                      "hull_nir_snr", "hull_offset_m"):
                row[k] = ""
            row["detector"] = "A"
        out.append(row)

    fields = list(out[0].keys()) if out else []
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(out)
    print(f"A rows in (aspect >= {args.min_aspect}): {len(A)}")
    print(f"  with a hull measured : {matched} ({matched/max(len(A),1):.0%})")
    print(f"  A-only, no hull      : {len(A)-matched}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
