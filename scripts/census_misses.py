#!/usr/bin/env python3
"""Chip the boxes the AGREEMENT RULE threw away, for the recall half of the census.

WHY. The frozen detector keeps a box only when TOA and SR both see it within 30 m
(scripts/detect_vessels_final.py). That requirement is what makes the precision
tolerable, and it is also the only place a real vessel can be lost -- a box that one
rendering found and the other did not is DISCARDED, whether it was a whitecap or a
boat. Against the human counts the detector is 41 to 42 on the total but that total is
a near-cancellation: 12 surplus detections against 13 missed vessels, exact on only
8 of 26 scenes. This script renders the discarded single-asset boxes so the misses can
be looked at rather than inferred.

NO INFERENCE. It reads the box cache written by scripts/sweep_cache.py at RAW_CONF and
re-applies the frozen confidences and the same greedy matcher, so what it calls
"unmatched" is exactly what detect_vessels_final.py dropped -- not a reimplementation
of the filter.

WHAT A VERDICT HERE MEANS. `vessel` on an unmatched box is a RECALL loss caused by the
agreement rule and an argument for relaxing it; `not_vessel` is the rule working as
intended. Both are useful and the second is the more likely.

RANKING AND THE CAP. Unmatched boxes are plentiful (one scene has 18 TOA-only), so
they are ranked by confidence and capped by --top. The number dropped is printed and
written to the index, because a silent cap reads as "we looked at everything".
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from boatphone import optical                                     # noqa: E402
from detector_a_toa_test import reflectance_coefficients, tci_rgb   # noqa: E402
from sweep_cache import scene_paths, RAW_CONF                       # noqa: E402
import detect_vessels_final as dvf                                  # noqa: E402
from census_chips import sheet                                      # noqa: E402

INDEX_FIELDS = ["scene_id", "acq_time_utc", "asset", "panel_row", "chip_file",
                "rank_in_asset", "n_unmatched_in_asset", "conf", "range_km",
                "bearing_deg", "length_m", "nir_x_water", "px_cx", "px_cy",
                "truth_count", "detected_count", "verdict", "note"]


def cached(cache_dir, asset, sid):
    f = Path(cache_dir) / f"boxes_{asset}_{sid}_s{dvf.SLICE}_i{dvf.IMGSZ}_bgr.npz"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} missing -- build it with:\n"
            f"  python scripts/sweep_cache.py --cache-dir {cache_dir} "
            f"--assets toa sr --slice {dvf.SLICE} --imgsz {dvf.IMGSZ} --order bgr")
    z = np.load(f, allow_pickle=False)
    return z["boxes"], z["scores"], float(z["res_x"]), float(z["res_y"])


def keep(boxes, scores, conf, res_x, res_y):
    """Frozen post-filters: confidence, then the implausibility bound on box length."""
    if len(scores) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    w = (boxes[:, 2] - boxes[:, 0]) * res_x
    h = (boxes[:, 3] - boxes[:, 1]) * res_y
    length = np.maximum(w, h)
    ok = (scores >= conf) & (length >= dvf.MIN_LENGTH_M) & (length <= dvf.MAX_LENGTH_M)
    return boxes[ok], scores[ok]


def truth_counts(path):
    out = {}
    with open(path) as f:
        for r in csv.reader(f):
            if not r or r[0].strip().lower().startswith("timestamp"):
                continue
            out[r[0].strip()] = int(r[1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True, help="written by scripts/sweep_cache.py")
    ap.add_argument("--out-dir", default="data/derived/census/misses")
    ap.add_argument("--validation", default="data/validation/planet_scope_validations.csv")
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--top", type=int, default=6, help="highest-confidence unmatched per asset")
    ap.add_argument("--only-shortfall", action="store_true",
                    help="restrict to scenes where the human count exceeds ours")
    args = ap.parse_args()

    if RAW_CONF > min(dvf.CONF_TOA, dvf.CONF_SR):
        sys.exit(f"cache floor {RAW_CONF} is above a frozen confidence -- the cache "
                 f"cannot reproduce the shipped filter")

    truth = truth_counts(args.validation)
    scenes_csv = Path("data/derived/detections_final/scenes_final.csv")
    detected = {}
    if scenes_csv.exists():
        with scenes_csv.open() as f:
            detected = {r["scene_id"]: int(r["n_vessels"]) for r in csv.DictReader(f)}

    allp = scene_paths()
    sids = args.scenes or sorted(allp)
    if args.only_shortfall:
        sel = [s for s in sids
               if truth.get(s[:13]) is not None and detected.get(s, 0) < truth[s[:13]]]
        print(f"shortfall scenes: {len(sel)} of {len(sids)}")
        sids = sel

    out = Path(args.out_dir)
    index, dropped_total = [], 0
    for n, sid in enumerate(sids, 1):
        p = allp[sid]
        bt, st, res_x, res_y = cached(args.cache_dir, "toa", sid)
        bs, ss, _, _ = cached(args.cache_dir, "sr", sid)
        bt, st = keep(bt, st, dvf.CONF_TOA, res_x, res_y)
        bs, ss = keep(bs, ss, dvf.CONF_SR, res_x, res_y)
        ct, lt, wt, ht = dvf.centres([(b, s) for b, s in zip(bt, st)], res_x, res_y)
        cs, ls, ws, hs = dvf.centres([(b, s) for b, s in zip(bs, ss)], res_x, res_y)
        pairs = dvf.match(ct, cs, res_x)
        used_t = {i for i, _, _ in pairs}
        used_s = {j for _, j, _ in pairs}

        cand = []
        for asset, boxes, scores, cen, length, used in (
                ("toa_only", bt, st, ct, lt, used_t), ("sr_only", bs, ss, cs, ls, used_s)):
            idx = [i for i in range(len(scores)) if i not in used]
            idx.sort(key=lambda i: -scores[i])
            if len(idx) > args.top:
                dropped_total += len(idx) - args.top
                print(f"    {sid} {asset}: showing top {args.top} of {len(idx)} unmatched",
                      flush=True)
            for rank, i in enumerate(idx[:args.top], 1):
                cand.append((asset, i, rank, len(idx), boxes[i], float(scores[i]),
                             cen[i], float(length[i])))
        if not cand:
            print(f"[{n}] {sid}: no unmatched boxes", flush=True)
            continue

        xml = str(p["toa"]).replace("_AnalyticMS_clip.tif", "_AnalyticMS_metadata_clip.xml")
        rgb_toa, _, meta = tci_rgb(p["toa"], reflectance_coefficients(xml))
        rgb_sr, _, _ = tci_rgb(p["sr"], None)
        with rasterio.open(p["sr"]) as src:
            nir = src.read(4).astype(np.float32) / optical.PLANET_QUANT
        acq = dvf.vd.parse_scene_meta(str(p["toa"]))[1]
        med = float(np.median(nir[nir > 0])) if (nir > 0).any() else float("nan")

        rows = []
        for k, (asset, i, rank, n_un, box, score, cen, length) in enumerate(cand, 1):
            x1, y1, x2, y2 = [float(v) for v in box]
            xs, ys = rasterio.transform.xy(meta["transform"], [cen[1]], [cen[0]],
                                           offset="center")
            lon, lat = rasterio.warp.transform(meta["crs"], "EPSG:4326", xs, ys)
            lon, lat = float(lon[0]), float(lat[0])
            sub = nir[max(int(y1), 0):int(y2) + 1, max(int(x1), 0):int(x2) + 1]
            peak = float(sub.max()) if sub.size else float("nan")
            rows.append({
                "scene_id": sid, "acq_time_utc": acq, "asset": asset,
                "panel_row": k, "rank_in_asset": rank, "n_unmatched_in_asset": n_un,
                "conf": round(score, 4),
                "range_km": round(optical.range_km(lon, lat), 3),
                "bearing_deg": round(optical.bearing_deg(lon, lat), 1),
                "length_m": round(length, 1),
                "nir_x_water": round(peak / med, 1) if med > 0 else "",
                "px_x1": x1, "px_y1": y1, "px_x2": x2, "px_y2": y2,
                "px_cx": float(cen[0]), "px_cy": float(cen[1]),
                # sheet() captions on these keys; asset and rank ride in conf_toa/conf_sr
                "conf_toa": f"{score:.4f} ({asset})", "conf_sr": f"rank {rank}/{n_un}",
                "det_index": f"{asset}#{rank}",
            })
        chip = out / f"{sid}.png"
        sheet(sid, rows, rgb_toa, rgb_sr, nir, chip)
        t, dcount = truth.get(sid[:13]), detected.get(sid)
        print(f"[{n}] {sid}: {len(rows)} unmatched shown  (truth {t}, detected {dcount})"
              f" -> {chip}", flush=True)
        for r in rows:
            index.append({**{f: r.get(f, "") for f in INDEX_FIELDS},
                          "chip_file": chip.name, "truth_count": t,
                          "detected_count": dcount, "verdict": "", "note": "",
                          "px_cx": round(r["px_cx"], 1), "px_cy": round(r["px_cy"], 1)})

    optical.write_csv(out / "misses_index.csv", index, INDEX_FIELDS)
    print(f"\n{len(index)} unmatched boxes shown over "
          f"{len(set(r['scene_id'] for r in index))} scenes; "
          f"{dropped_total} further unmatched boxes NOT shown (--top {args.top})"
          f"\n  -> {out}/\n  -> {out/'misses_index.csv'}  (verdict is EMPTY)")


if __name__ == "__main__":
    main()
