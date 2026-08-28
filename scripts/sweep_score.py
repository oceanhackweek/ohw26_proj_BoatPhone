#!/usr/bin/env python3
"""Sweep Detector A's post-hoc filters over the cached boxes and score against truth.

Truth is data/validation/planet_scope_validations.csv -- a human count of vessels
per scene, keyed by the YYYYMMDD_HHMM prefix of the scene id. It is a COUNT, not a
set of positions, so this scores counts and nothing more: a config can hit the right
number for the wrong reasons, and that is what the previews exist to check.

No inference happens here. Everything swept (confidence, length bounds, aspect,
water/clear mask, persistence) is a filter applied to boxes cached by sweep_cache.py.
"""
import argparse, csv, itertools, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rasterio.transform import Affine, xy as rio_xy
from rasterio.warp import transform as rio_transform

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from boatphone import optical
from boatphone.paths import REPO_ROOT
from sweep_cache import RAW_CONF

TRUTH_CSV = REPO_ROOT / "data" / "validation" / "planet_scope_validations.csv"

# Persistence cell, ~50 m at this latitude. Source: scripts/detect_vessels_toa.py,
# which introduced the same rule -- one definition of the cell size, not two.
DLAT, DLON = 0.00045, 0.00068


def load_truth():
    """{'YYYYMMDD_HHMM': int} -- the human vessel count per acquisition."""
    truth = {}
    with open(TRUTH_CSV) as fh:
        for row in csv.DictReader(fh):
            row = {k.strip(): v.strip() for k, v in row.items() if k}
            truth[row["timestamp"]] = int(row["vessel count"])
    return truth


def load_cache(cache_dir, asset, geom):
    """Cached boxes for one (asset, geometry), georeferenced once, per scene.

    Returns {scene_id: dict of per-box arrays}. Georeferencing is done here, once,
    rather than inside the sweep loop -- it does not depend on any swept parameter.
    """
    cache_dir = Path(cache_dir)
    out = {}
    for f in sorted(cache_dir.glob(f"boxes_{asset}_*_{geom}.npz")):
        sid = f.name[len(f"boxes_{asset}_"):-len(f"_{geom}.npz")]
        z = np.load(f, allow_pickle=False)
        boxes, scores = z["boxes"], z["scores"]
        res_x, res_y = float(z["res_x"]), float(z["res_y"])
        m = np.load(cache_dir / f"masks_{sid}.npz")
        water, clear, valid = m["water"], m["clear"], m["valid"]

        # Independent evidence attached by sweep_features.py. Absent until that
        # has been run; the filters that use them are then no-ops rather than a
        # KeyError, which would hide the fact that they never ran.
        extra = {k: z[k] for k in ("nir_pct", "shore_m", "nir_peak") if k in z}
        rec = {"boxes": boxes, "scores": scores, "res_x": res_x, "res_y": res_y,
               "transform": Affine(*z["transform"][:6]), "crs": str(z["crs"]),
               "shape": valid.shape, **extra}
        if len(boxes) == 0:
            rec.update({k: np.zeros(0) for k in
                        ("w_m", "h_m", "length", "aspect", "lon", "lat", "range_km")})
            rec.update({k: np.zeros(0, bool) for k in ("in_valid", "in_clear", "in_water")})
            rec.update({k: np.zeros(0) for k in ("nir_pct", "shore_m", "nir_peak")})
            out[sid] = rec
            continue

        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        ci = np.clip(np.round(cx).astype(int), 0, valid.shape[1] - 1)
        ri = np.clip(np.round(cy).astype(int), 0, valid.shape[0] - 1)
        w_m = (boxes[:, 2] - boxes[:, 0]) * res_x
        h_m = (boxes[:, 3] - boxes[:, 1]) * res_y
        length = np.maximum(w_m, h_m)
        short = np.maximum(np.minimum(w_m, h_m), 1e-6)
        xs, ys = rio_xy(rec["transform"], ri.tolist(), ci.tolist(), offset="center")
        lon, lat = rio_transform(rec["crs"], "EPSG:4326", xs, ys)
        lon, lat = np.asarray(lon), np.asarray(lat)
        hx, hy = optical.HYDROPHONE_LONLAT
        rng = np.array([optical.range_km(a, b) for a, b in zip(lon, lat)])

        rec.update({"w_m": w_m, "h_m": h_m, "length": length, "aspect": length / short,
                    "lon": lon, "lat": lat, "range_km": rng,
                    "in_valid": valid[ri, ci], "in_clear": clear[ri, ci],
                    "in_water": water[ri, ci]})
        out[sid] = rec
    return out


def select(rec, conf, mask, min_len, max_len, max_aspect, nir_pct=0.0, shore_m=0.0):
    """Boolean keep-mask over one scene's cached boxes for one filter setting.

    nir_pct / shore_m are the independent-evidence gates from sweep_features.py:
    keep only boxes whose peak NIR sits above that quantile of the scene's own
    water, and whose centre is at least that far from the nearest non-water pixel.
    Both default to off.
    """
    if len(rec["scores"]) == 0:
        return np.zeros(0, bool)
    keep = (rec["scores"] >= conf) & rec["in_valid"]
    if mask == "clear":
        keep &= rec["in_clear"]
    elif mask == "water":
        keep &= rec["in_water"]
    keep &= (rec["length"] >= min_len) & (rec["length"] <= max_len)
    keep &= rec["aspect"] <= max_aspect
    if nir_pct > 0:
        if "nir_pct" not in rec:
            raise KeyError("--nir-pct was given but the cache has no NIR features; "
                           "run scripts/sweep_features.py first")
        keep &= rec["nir_pct"] >= nir_pct
    if shore_m > 0:
        if "shore_m" not in rec:
            raise KeyError("--shore-m was given but the cache has no shore feature; "
                           "run scripts/sweep_features.py first")
        keep &= rec["shore_m"] >= shore_m
    return keep


def apply_persistence(cache, keeps, min_years):
    """Drop boxes in a cell occupied in >= min_years distinct YEARS.

    A vessel does not return to the same 50 m cell across years; a rock does. Makes
    no distributional assumption, which is why it survives when thresholds do not.
    0 disables it.
    """
    if not min_years:
        return keeps
    years = defaultdict(set)
    for sid, rec in cache.items():
        k = keeps[sid]
        if not k.any():
            continue
        cells = list(zip(np.round(rec["lat"][k] / DLAT).astype(int),
                         np.round(rec["lon"][k] / DLON).astype(int)))
        for c in cells:
            years[c].add(sid[:4])
    out = {}
    for sid, rec in cache.items():
        k = keeps[sid].copy()
        if k.any():
            idx = np.flatnonzero(k)
            cells = zip(np.round(rec["lat"][idx] / DLAT).astype(int),
                        np.round(rec["lon"][idx] / DLON).astype(int))
            for j, c in zip(idx, cells):
                if len(years[c]) >= min_years:
                    k[j] = False
        out[sid] = k
    return out


def fuse_counts(caches, keeps, mode, match_m=30.0):
    """Per-scene counts from the TWO asset types combined.

    The two renderings are the SAME acquisition -- same water, same boats, same
    instant -- so a detection present in both is corroborated by an independent
    radiometry, and one present in only one is not. Two boxes are the same object
    when their centroids fall within `match_m` (30 m = 10 px at 3 m GSD).

      both   intersection -- precision tool; drops anything only one asset saw
      any    union        -- recall tool; counts each object once
    """
    from itertools import count as _count
    a, b = "toa", "sr"
    out = {}
    for sid in caches[a]:
        if sid not in caches[b]:
            continue
        pts = {}
        for k in (a, b):
            rec, keep = caches[k][sid], keeps[k][sid]
            idx = np.flatnonzero(keep)
            pts[k] = np.column_stack([rec["lon"][idx], rec["lat"][idx]]) if idx.size \
                else np.zeros((0, 2))
        # metres-per-degree at this latitude; the AOI is ~10 km so a local flat
        # approximation is exact to well under one pixel.
        lat0 = float(np.mean(np.concatenate([pts[a][:, 1], pts[b][:, 1]]))) \
            if len(pts[a]) or len(pts[b]) else 48.81
        mx = 111320.0 * math.cos(math.radians(lat0))
        my = 110540.0
        A = pts[a] * np.array([mx, my]); B = pts[b] * np.array([mx, my])
        if len(A) == 0 or len(B) == 0:
            out[sid] = 0 if mode == "both" else len(A) + len(B)
            continue
        d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
        # Greedy mutual nearest-neighbour pairing under the distance gate: one
        # box may corroborate at most one other, or a cluster would over-count.
        pairs, used_a, used_b = 0, set(), set()
        for i, j in sorted(((i, j) for i in range(len(A)) for j in range(len(B))
                            if d[i, j] <= match_m), key=lambda ij: d[ij]):
            if i in used_a or j in used_b:
                continue
            used_a.add(i); used_b.add(j); pairs += 1
        out[sid] = pairs if mode == "both" else len(A) + len(B) - pairs
    return out


def score(counts, truth):
    """Count-accuracy of a config. Scenes with no truth entry are excluded.

    Two numbers, because they fail differently:
      batch  -- 1 - |sum(pred) - sum(truth)| / sum(truth). Cancels over/under
                between scenes, so a config can look perfect while being wrong twice.
      scene  -- mean per-scene 1 - |pred-truth|/max(truth,1), floored at 0. Does not
                cancel. This is the honest one, and the one to report.
    """
    pairs = [(counts.get(sid, 0), truth[k])
             for sid, k in ((s, s[:13]) for s in counts) if k in truth]
    if not pairs:
        return None
    pred = np.array([p for p, _ in pairs], float)
    true = np.array([t for _, t in pairs], float)
    scene_acc = np.clip(1 - np.abs(pred - true) / np.maximum(true, 1), 0, 1)
    return {
        "n_scenes": len(pairs),
        "pred_total": int(pred.sum()), "truth_total": int(true.sum()),
        "batch_acc": float(1 - abs(pred.sum() - true.sum()) / max(true.sum(), 1)),
        "scene_acc": float(scene_acc.mean()),
        "mae": float(np.abs(pred - true).mean()),
        # Total absolute error against total truth. Unlike batch_acc a miss in one
        # scene cannot be paid for by a false positive in another, and unlike
        # scene_acc it is not dominated by whether a truth-1 scene landed exactly.
        "err_acc": float(max(0.0, 1 - np.abs(pred - true).sum() / max(true.sum(), 1))),
        "exact": float((pred == true).mean()),
        # Does the count TRACK the truth, independently of its scale? A detector can
        # have a good MAE by predicting the mean everywhere; r says whether it is
        # actually responding to vessels. Invariant 9: this is what separates "the
        # method found nothing" from "the method is broken".
        "r": (float(np.corrcoef(pred, true)[0, 1])
              if pred.std() > 0 and true.std() > 0 else float("nan")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--assets", nargs="+", default=["toa", "sr", "both", "any"],
                    help="toa / sr, or the two fused: 'both' = intersection "
                         "(precision), 'any' = union (recall)")
    ap.add_argument("--match-m", type=float, default=30.0,
                    help="centroid distance at which a TOA and an SR box are the "
                         "same object, for the fused modes (30 m = 10 px)")
    ap.add_argument("--geom", default="s640_i640_bgr")
    ap.add_argument("--conf", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--mask", nargs="+", default=["none", "clear", "water"])
    ap.add_argument("--min-len", type=float, nargs="+", default=[0.0, 6.0, 10.0])
    ap.add_argument("--max-len", type=float, nargs="+", default=[40.0, 80.0, 150.0])
    ap.add_argument("--max-aspect", type=float, nargs="+", default=[99.0, 4.0, 2.5])
    ap.add_argument("--persist", type=int, nargs="+", default=[0, 2, 3])
    ap.add_argument("--nir-pct", type=float, nargs="+", default=[0.0],
                    help="keep boxes whose peak NIR is above this quantile of the "
                         "scene's own water NIR (0 = off)")
    ap.add_argument("--shore-m", type=float, nargs="+", default=[0.0],
                    help="keep boxes at least this many metres from non-water (0 = off)")
    ap.add_argument("--conf2", type=float, nargs="+", default=None,
                    help="SR-side confidences for the fused modes; default = same "
                         "as --conf")
    ap.add_argument("--rank", default="scene_acc",
                    choices=["scene_acc", "batch_acc", "err_acc", "exact", "r"])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default=None, help="write the full ranked table as JSON")
    args = ap.parse_args()

    bad = [c for c in args.conf if c < RAW_CONF]
    if bad:
        sys.exit(f"conf {bad} is below the cache floor RAW_CONF={RAW_CONF}; "
                 f"those boxes were never stored, so the sweep would be reporting "
                 f"a filter it did not run. Rebuild the cache with a lower floor.")

    truth = load_truth()
    caches = {}
    for asset in ("toa", "sr"):
        c = load_cache(args.cache_dir, asset, args.geom)
        if c:
            caches[asset] = c
            print(f"{asset}: {len(c)} scenes, "
                  f"{sum(len(r['scores']) for r in c.values())} raw boxes", file=sys.stderr)
    if not caches:
        sys.exit(f"no cache for geom={args.geom} in {args.cache_dir}")

    results = []
    for conf, mask, mnl, mxl, mxa, npct, shm in itertools.product(
            args.conf, args.mask, args.min_len, args.max_len, args.max_aspect,
            args.nir_pct, args.shore_m):
        if mnl >= mxl:
            continue
        # The two renderings are not confidence-calibrated alike -- TOA water sits
        # ~4x brighter than SR -- so the fused modes tune a conf per asset. conf2
        # is the SR threshold; for single-asset rows it is ignored (guarded below).
        for conf2 in (args.conf2 or [conf]):
            keeps = {}
            for a, c in caches.items():
                cf = conf if a == "toa" else conf2
                keeps[a] = {sid: select(rec, cf, mask, mnl, mxl, mxa, npct, shm)
                            for sid, rec in c.items()}
            for pers in args.persist:
                k2 = {a: apply_persistence(caches[a], k, pers) for a, k in keeps.items()}
                for asset in args.assets:
                    if asset in ("both", "any"):
                        if len(caches) < 2:
                            continue
                        counts = fuse_counts(caches, k2, asset, args.match_m)
                    elif asset in k2:
                        # a single-asset row does not depend on the other asset's
                        # threshold; emit it once, not once per conf2.
                        if args.conf2 and conf2 != (args.conf2 or [conf])[0]:
                            continue
                        counts = {sid: int(k.sum()) for sid, k in k2[asset].items()}
                    else:
                        continue
                    sc = score(counts, truth)
                    if sc is None:
                        continue
                    sc.update({"asset": asset, "geom": args.geom,
                               "conf": conf if asset != "sr" else conf2,
                               "conf_toa": conf, "conf_sr": conf2,
                               "mask": mask, "min_len": mnl, "max_len": mxl,
                               "max_aspect": mxa, "persist": pers,
                               "nir_pct": npct, "shore_m": shm})
                    results.append(sc)

    import math as _m
    results.sort(key=lambda r: (-(r[args.rank] if not _m.isnan(r[args.rank] if isinstance(r[args.rank], float) else 0) else -9), r["mae"]))
    hdr = (f"{'asset':>5} {'cTOA':>5} {'cSR':>5} {'mask':>6} {'minL':>5} {'maxL':>5} {'maxAR':>6} "
           f"{'pers':>4} {'nirQ':>5} {'shore':>5} | {'pred':>5} {'truth':>5} {'scene':>6} {'batch':>6} "
           f"{'err':>6} {'MAE':>5} {'exact':>5} {'r':>6}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in results[:args.top]:
        print(f"{r['asset']:>5} {r['conf_toa']:>5.2f} {r['conf_sr']:>5.2f} "
              f"{r['mask']:>6} {r['min_len']:>5.0f} "
              f"{r['max_len']:>5.0f} {r['max_aspect']:>6.1f} {r['persist']:>4} "
              f"{r['nir_pct']:>5.3f} {r['shore_m']:>5.0f} | "
              f"{r['pred_total']:>5} {r['truth_total']:>5} {r['scene_acc']:>6.3f} "
              f"{r['batch_acc']:>6.3f} {r['err_acc']:>6.3f} {r['mae']:>5.2f} "
              f"{r['exact']:>5.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))
        print(f"\n{len(results)} configs -> {args.out}")


if __name__ == "__main__":
    main()
