#!/usr/bin/env python3
"""The frozen vessel detector: TOA-and-SR agreement, calibrated against human counts.

CONFIG AND WHERE IT CAME FROM. Every value below was selected by scripts/sweep_*.py
against data/validation/planet_scope_validations.csv, and validated in
scripts/sweep_validate.py against a label-shuffled null and a split-half holdout.
The grid deliberately has only TWO free knobs (one confidence per asset): a larger
grid scored no better on held-out scenes and overfit measurably more.

    tile 640 -> imgsz 640, BGR      native scale; 320->640 upscaling scored WORSE
    conf 0.45 on TOA                TOA water sits ~4x brighter than SR, so the two
    conf 0.075 on SR                renderings need different thresholds
    agreement within 30 m           the two are the SAME acquisition, so a box in
                                    only one of them is uncorroborated
    length <= 40 m                  implausibility bound, not a science cut

MEASURED: 41 detections against 42 human-counted vessels over 26 scenes (97.6% on
total count), r = 0.675 per scene, 77% of scenes within +-1, MAE 0.96.

LENGTH IS THE BOX, NOT THE HULL. Verified by eye on chips at native resolution: the
vessels are 2-3 px of bright pixels while the model's box is 8-12 px, so length_m
lands at 23-37 m for craft that are plainly much smaller. Every row therefore reports
size_class ">24m" or "12-24m" and NONE reports the 0-12 m FAO class -- that is an
artefact of the box, not a statement about the fleet. Use length_m for the
implausibility cut it was tuned for and NOT as a size measurement; the same
axis-aligned bias is recorded in the source of truth against Detector A.

WHAT THIS IS NOT. Per-scene counts are right within +-1 four times in five but exact
only ~31% of the time, and the truth set is a count, not positions -- so a scene can
carry the right number for the wrong reasons. Treat per-scene output as a candidate
list to eyeball, not a vessel census. See the source of truth.
"""
import argparse, math, sys
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from boatphone import optical, vessel_detect as vd
from detector_a_toa_test import reflectance_coefficients, tci_rgb
from sweep_cache import scene_paths

# --- the frozen configuration (see the module docstring for provenance) ---
SLICE, IMGSZ, BGR = 640, 640, True
CONF_TOA, CONF_SR = 0.45, 0.075
MATCH_M = 30.0
MAX_LENGTH_M, MIN_LENGTH_M = 40.0, 0.0
OVERLAP, NMS_IOU = 0.2, 0.5

FIELDS = ["scene_id", "acq_time_utc", "lon", "lat", "range_km", "bearing_deg",
          "bbox_w_m", "bbox_h_m", "length_m", "size_class", "aspect_ratio",
          "conf_toa", "conf_sr", "match_dist_m", "nir_peak", "nir_x_water",
          "px_x1", "px_y1", "px_x2", "px_y2"]
SCENE_FIELDS = ["scene_id", "acq_time_utc", "n_vessels", "n_toa_only", "n_sr_only",
                "r_nearest_km", "r_dominant_km", "water_median_nir", "detector"]


def detect(model, rgb, valid, conf):
    """Tiled inference -> [(box, score)] in scene pixel coordinates."""
    return vd.predict_tiled(model, rgb, valid, SLICE, IMGSZ, conf,
                            OVERLAP, NMS_IOU, bgr=BGR)


def centres(dets, res_x, res_y):
    """(centres_px, lengths_m, w_m, h_m) for a list of boxes."""
    if not dets:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0), np.zeros(0)
    b = np.array([d[0] for d in dets], float)
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    w_m = (b[:, 2] - b[:, 0]) * res_x
    h_m = (b[:, 3] - b[:, 1]) * res_y
    return np.column_stack([cx, cy]), np.maximum(w_m, h_m), w_m, h_m


def match(pa, pb, res_m):
    """Greedy mutual-nearest pairing of TOA and SR centres under MATCH_M.

    One box corroborates at most one other; without that a cluster of boxes in one
    rendering would each claim the same vessel in the other and inflate the count.
    """
    if len(pa) == 0 or len(pb) == 0:
        return []
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1) * res_m
    pairs, ua, ub = [], set(), set()
    for i, j in sorted(((i, j) for i in range(len(pa)) for j in range(len(pb))
                        if d[i, j] <= MATCH_M), key=lambda ij: d[ij]):
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); pairs.append((i, j, float(d[i, j])))
    return pairs


def draw(rgb, rows, path, title):
    """Annotated preview: box per vessel, labelled with its range to the hydrophone."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(rgb.copy()).convert("RGB")
    d = ImageDraw.Draw(img)
    for r in rows:
        x1, y1, x2, y2 = r["px_x1"], r["px_y1"], r["px_x2"], r["px_y2"]
        pad = 10
        d.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], outline=(255, 60, 60), width=3)
        d.text((x1 - pad, y2 + pad + 3), f"{r['range_km']:.2f} km", fill=(255, 220, 60))
    d.text((8, 8), title, fill=(255, 255, 255))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--preview-dir", default=None)
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    allp = scene_paths()
    sids = args.scenes or sorted(allp)
    model, _ = vd.load_model("cpu")
    hx, hy = optical.HYDROPHONE_LONLAT
    print(f"hydrophone {hx:.6f}, {hy:.6f}  ({optical.HYDROPHONE_SOURCE.splitlines()[0]})")

    rows, scene_rows = [], []
    for i, sid in enumerate(sids, 1):
        p = allp[sid]
        xml = str(p["toa"]).replace("_AnalyticMS_clip.tif", "_AnalyticMS_metadata_clip.xml")
        rgb_t, valid_t, meta = tci_rgb(p["toa"], reflectance_coefficients(xml))
        rgb_s, valid_s, _ = tci_rgb(p["sr"], None)
        res_x, res_y = meta["res_x"], meta["res_y"]
        _, acq = vd.parse_scene_meta(str(p["toa"]))

        with rasterio.open(p["sr"]) as src:
            green = src.read(2).astype(np.float32) / optical.PLANET_QUANT
            nir = src.read(4).astype(np.float32) / optical.PLANET_QUANT
        clear = vd.read_udm2_clear(str(p["udm2"]), valid_t.shape)
        # Water mask is built from _sr because LAND_NIR_MIN is an absolute SURFACE
        # reflectance threshold (decision 0017). It is reported, not applied: the
        # sweep measured it as inert here (the AOI is ~92% water), so applying it
        # would only suggest a filter that is not doing anything.
        water, _rep = optical.water_mask(green, nir, valid_s, clear, res_m=res_x)
        water_med = float(np.median(nir[water])) if water.any() else float("nan")

        dt = [d for d in detect(model, rgb_t, valid_t, CONF_TOA)]
        ds = [d for d in detect(model, rgb_s, valid_s, CONF_SR)]
        ct, lt, wt, ht = centres(dt, res_x, res_y)
        cs, ls, _, _ = centres(ds, res_x, res_y)
        okt = (lt >= MIN_LENGTH_M) & (lt <= MAX_LENGTH_M)
        oks = (ls >= MIN_LENGTH_M) & (ls <= MAX_LENGTH_M)
        it, is_ = np.flatnonzero(okt), np.flatnonzero(oks)
        pairs = match(ct[it], cs[is_], res_x)

        srows = []
        for a, b, dist in pairs:
            k = int(it[a])
            x1, y1, x2, y2 = dt[k][0]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            xs, ys = rasterio.transform.xy(meta["transform"], [cy], [cx], offset="center")
            lon, lat = rasterio.warp.transform(meta["crs"], "EPSG:4326", xs, ys)
            lon, lat = float(lon[0]), float(lat[0])
            length = float(lt[k]); short = max(min(wt[k], ht[k]), 1e-6)
            sub = nir[max(int(y1), 0):int(y2) + 1, max(int(x1), 0):int(x2) + 1]
            peak = float(sub.max()) if sub.size else float("nan")
            srows.append({
                "scene_id": sid, "acq_time_utc": acq,
                "lon": round(lon, 6), "lat": round(lat, 6),
                "range_km": round(optical.range_km(lon, lat), 3),
                "bearing_deg": round(optical.bearing_deg(lon, lat), 1),
                "bbox_w_m": round(float(wt[k]), 1), "bbox_h_m": round(float(ht[k]), 1),
                "length_m": round(length, 1),
                "size_class": optical.size_class(length),
                "aspect_ratio": round(length / short, 2),
                "conf_toa": round(float(dt[k][1]), 4),
                "conf_sr": round(float(ds[int(is_[b])][1]), 4),
                "match_dist_m": round(dist, 1),
                "nir_peak": round(peak, 4),
                "nir_x_water": (round(peak / water_med, 1) if water_med > 0 else ""),
                "px_x1": round(x1, 1), "px_y1": round(y1, 1),
                "px_x2": round(x2, 1), "px_y2": round(y2, 1),
            })
        rows.extend(srows)
        rr = sorted(r["range_km"] for r in srows)
        w = [r ** (-optical.NOMINAL_SPREADING_K / 10) for r in rr if r > 0]
        scene_rows.append({
            "scene_id": sid, "acq_time_utc": acq, "n_vessels": len(srows),
            "n_toa_only": int(len(it) - len(pairs)), "n_sr_only": int(len(is_) - len(pairs)),
            "r_nearest_km": rr[0] if rr else "",
            "r_dominant_km": (rr[int(np.argmax(w))] if w else ""),
            "water_median_nir": round(water_med, 5), "detector": "A_toa AND A_sr",
        })
        print(f"[{i:>2}/{len(sids)}] {sid}  TOA {len(it):>3} SR {len(is_):>3} "
              f"-> {len(srows)} vessel(s)  ranges {[f'{r:.1f}' for r in rr]}", flush=True)
        if args.preview_dir:
            draw(rgb_t, srows, f"{args.preview_dir}/{sid}.jpg",
                 f"{sid}  {acq}  {len(srows)} vessel(s)")

    optical.write_csv(out / "detections_final.csv", rows, FIELDS)
    optical.write_csv(out / "scenes_final.csv", scene_rows, SCENE_FIELDS)
    print(f"\n{len(rows)} vessels over {len(sids)} scenes")
    print(f"  -> {out/'detections_final.csv'}\n  -> {out/'scenes_final.csv'}")
    if args.preview_dir:
        print(f"  -> {args.preview_dir}/  (boxes labelled with range to the hydrophone)")


if __name__ == "__main__":
    main()
