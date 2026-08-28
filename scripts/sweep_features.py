#!/usr/bin/env python3
"""Attach independent per-box evidence to the cached Detector A boxes.

Two features, neither of which the model could have used, both computed from the
_sr product on the same grid:

  nir_pct     the box's peak NIR as an EMPIRICAL QUANTILE of the scene's own water
              NIR distribution. Quantile, not sigma-multiple, on purpose: ocean NIR
              is not Gaussian (see the source of truth, "why Detector B
              over-predicts"), so a MAD threshold means something different on every
              scene. A quantile means the same thing on all of them by construction.
  shore_m     distance from the box centre to the nearest non-water pixel. Shoreline
              is the dominant false positive in an archipelago, and the water mask's
              fixed erosion is a blunt version of this.

boatphone.optical.nir_support_for_box documents NIR evidence as ADVISORY, NEVER A
FILTER. This script deliberately makes it available AS a filter, because the reason
for that rule -- no way to tell whether it costs recall -- no longer holds: we now
have a human count per scene and can measure the cost. Whether it is used is decided
by that measurement, in sweep_score.py, not here.
"""
import argparse, sys
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from boatphone import optical
from sweep_cache import scene_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--geoms", nargs="+", default=["s640_i640_bgr", "s320_i640_bgr"])
    ap.add_argument("--assets", nargs="+", default=["toa", "sr"])
    args = ap.parse_args()

    import cv2
    cache = Path(args.cache_dir)
    allp = scene_paths()

    for i, (sid, paths) in enumerate(sorted(allp.items()), 1):
        m = np.load(cache / f"masks_{sid}.npz")
        water = m["water"]
        with rasterio.open(paths["sr"]) as s:
            nir = s.read(4).astype(np.float32) / optical.PLANET_QUANT
            res_m = abs(s.transform.a)
        wv = nir[water]
        if wv.size == 0:
            print(f"! {sid}: empty water mask, skipped", file=sys.stderr); continue
        order = np.sort(wv)
        # Distance (in metres) from every pixel to the nearest NON-water pixel.
        shore = cv2.distanceTransform(water.astype(np.uint8), cv2.DIST_L2, 5) * res_m

        for asset in args.assets:
            for geom in args.geoms:
                f = cache / f"boxes_{asset}_{sid}_{geom}.npz"
                if not f.exists():
                    continue
                z = dict(np.load(f, allow_pickle=False))
                if "nir_pct" in z:
                    continue
                b = z["boxes"]
                n = len(b)
                nir_peak = np.zeros(n, np.float32)
                shore_m = np.zeros(n, np.float32)
                for k in range(n):
                    x1, y1, x2, y2 = b[k]
                    c0, c1 = max(int(np.floor(x1)), 0), min(int(np.ceil(x2)) + 1, nir.shape[1])
                    r0, r1 = max(int(np.floor(y1)), 0), min(int(np.ceil(y2)) + 1, nir.shape[0])
                    sub = nir[r0:r1, c0:c1]
                    nir_peak[k] = float(sub.max()) if sub.size else 0.0
                    ri = min(max(int(round((y1 + y2) / 2)), 0), nir.shape[0] - 1)
                    ci = min(max(int(round((x1 + x2) / 2)), 0), nir.shape[1] - 1)
                    shore_m[k] = shore[ri, ci]
                # Empirical quantile of the peak within THIS scene's water pixels.
                z["nir_pct"] = (np.searchsorted(order, nir_peak).astype(np.float32)
                                / max(order.size, 1))
                z["nir_peak"] = nir_peak
                z["shore_m"] = shore_m
                z["water_median"] = np.float32(np.median(wv))
                np.savez_compressed(f, **z)
        print(f"[{i:>2}/{len(allp)}] {sid}  water med {np.median(wv):.4f}", flush=True)


if __name__ == "__main__":
    main()
