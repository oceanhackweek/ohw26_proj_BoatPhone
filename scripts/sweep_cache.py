#!/usr/bin/env python3
"""Cache raw YOLO boxes + per-scene masks so the parameter sweep costs no inference.

The expensive thing here is CPU inference (~40 tiles x 26 scenes per geometry).
Confidence, length bounds and the water/UDM2 masks are all POST-hoc filters, so
this runs the model ONCE per (asset, scene, geometry) at a deliberately floor-level
confidence and stores every box it returns. `sweep_score.py` then sweeps the cheap
knobs over the cache for free.

RAW_CONF is the floor: no swept `conf` may go below it, or the sweep would be
reporting a filter it never actually ran. Guarded in sweep_score.py.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from boatphone import optical, vessel_detect as vd
from detector_a_toa_test import reflectance_coefficients, tci_rgb

RAW_CONF = 0.02          # floor; every swept conf must be >= this
DOWNLOADS = Path("/home/jovyan/shared-public/boatphone_shared/planet_folger/downloads")
TOA_DIR = DOWNLOADS / "folger_202608_toa"
SR_DIR = DOWNLOADS / "folger_202608"


def scene_paths():
    """{scene_id: {'toa':..., 'sr':..., 'udm2':...}} for scenes present in BOTH sets."""
    toa = {p.name.split("_3B_")[0]: p for p in TOA_DIR.rglob("*_AnalyticMS_clip.tif")}
    sr = {p.name.split("_3B_")[0]: p for p in SR_DIR.rglob("*_AnalyticMS_SR_clip.tif")}
    out = {}
    for sid in sorted(set(toa) & set(sr)):
        out[sid] = {"toa": toa[sid], "sr": sr[sid],
                    "udm2": Path(str(toa[sid]).replace("_3B_AnalyticMS_clip.tif",
                                                       "_3B_udm2_clip.tif"))}
    missing = (set(toa) ^ set(sr))
    if missing:
        print(f"! present in only one asset set, skipped: {sorted(missing)}", file=sys.stderr)
    return out


def render(sid, paths, asset):
    """8-bit TCI RGB + validity + geo meta for one asset type of one scene."""
    if asset == "toa":
        p = paths["toa"]
        xml = str(p).replace("_AnalyticMS_clip.tif", "_AnalyticMS_metadata_clip.xml")
        return tci_rgb(p, reflectance_coefficients(xml))
    if asset == "sr":
        return tci_rgb(paths["sr"], None)
    raise ValueError(asset)


def scene_masks(sid, paths, cache_dir):
    """Water mask (NDWI+erosion, folds in UDM2 clear) from the SR product, cached.

    Built from _sr ON PURPOSE: LAND_NIR_MIN is an absolute surface-reflectance
    threshold (decision 0017). Same grid and clip as TOA, so the mask transfers.
    """
    f = cache_dir / f"masks_{sid}.npz"
    if f.exists():
        z = np.load(f)
        return z["water"], z["clear"], z["valid"]
    with rasterio.open(paths["sr"]) as s:
        srr = s.read().astype(np.float32) / optical.PLANET_QUANT
        res_m = abs(s.transform.a)
    valid = (srr.sum(0) > 0)
    clear = vd.read_udm2_clear(str(paths["udm2"]), valid.shape)
    if clear is None:
        raise FileNotFoundError(f"{sid}: no usable UDM2 mask at {paths['udm2']}")
    water, _rep = optical.water_mask(srr[1], srr[3], valid, clear, res_m=res_m)
    np.savez_compressed(f, water=water, clear=clear, valid=valid)
    return water, clear, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--assets", nargs="+", default=["toa", "sr"])
    ap.add_argument("--slice", type=int, nargs="+", default=[640])
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640])
    ap.add_argument("--order", nargs="+", default=["bgr"], choices=["bgr", "rgb"])
    ap.add_argument("--scenes", nargs="+", default=None, help="scene ids (default: all)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(args.threads)

    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    allp = scene_paths()
    sids = args.scenes or sorted(allp)
    print(f"{len(sids)} scenes, assets {args.assets}, "
          f"slice {args.slice} x imgsz {args.imgsz} x {args.order}", flush=True)

    model, _ = vd.load_model("cpu")
    t_all = time.time()
    for i, sid in enumerate(sids, 1):
        paths = allp[sid]
        water, clear, valid = scene_masks(sid, paths, cache_dir)
        for asset in args.assets:
            rgb = vmask = meta = None
            for sl in args.slice:
                for im in args.imgsz:
                    for order in args.order:
                        f = cache_dir / f"boxes_{asset}_{sid}_s{sl}_i{im}_{order}.npz"
                        if f.exists():
                            continue
                        if rgb is None:
                            rgb, vmask, meta = render(sid, paths, asset)
                        t0 = time.time()
                        dets = vd.predict_tiled(model, rgb, vmask, sl, im, RAW_CONF,
                                                0.2, 0.5, bgr=(order == "bgr"))
                        boxes = np.array([b for b, _ in dets], np.float32).reshape(-1, 4)
                        scores = np.array([s for _, s in dets], np.float32)
                        np.savez_compressed(
                            f, boxes=boxes, scores=scores,
                            res_x=meta["res_x"], res_y=meta["res_y"],
                            transform=np.array(meta["transform"]).reshape(-1),
                            crs=str(meta["crs"]),
                            water_dn=float(np.median(rgb[vmask])) if vmask.any() else np.nan)
                        print(f"[{i:>2}/{len(sids)}] {sid} {asset} s{sl}/i{im}/{order}: "
                              f"{len(scores):>5} raw  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\ndone in {(time.time()-t_all)/60:.1f} min -> {cache_dir}")


if __name__ == "__main__":
    main()
