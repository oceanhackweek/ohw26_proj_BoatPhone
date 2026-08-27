#!/usr/bin/env python3
"""
Small-vessel detection in PlanetScope `ortho_visual` imagery.

Uses the pretrained mayrajeo/marine-vessel-yolo model (yolo11s_tci.pt), trained on
8,768 hand-annotated recreational vessels in Sentinel-2 L1C TCI. PlanetScope
ortho_visual is the same radiometric domain -- 8-bit colour-corrected RGB -- so the
model runs with no band math and no atmospheric correction.

Emits `detections.csv` with the agreed team schema:
    scene_id, acq_time_utc, lon, lat, bbox_w_m, bbox_h_m, length_class_m,
    confidence, aspect_ratio, wake_flag, range_km

Install:
    pip install ultralytics huggingface_hub rasterio numpy pillow
    pip install sahi          # optional; a built-in tiler is used if absent

Usage:
    # single scene
    python vessel_detect.py --scenes data/*_Visual_clip.tif \
        --hydrophone -125.28,48.81 --out detections.csv --preview-dir previews/

    # hyperparameter sweep (see module docstring section "Sweep")
    python vessel_detect.py --scenes data/one_scene.tif --sweep

NOTE ON TILING: a 5 km AOI at 3 m is ~3300 px. YOLO's default imgsz=640 would
downsample it ~5x and annihilate a 2-3 px boat. We ALWAYS tile and never resize the
whole scene. Tile size and imgsz are the two knobs that set apparent object scale at
network input -- sweep them (--sweep) before trusting any count.
"""

import argparse
import csv
import glob
import math
import os
import re
import sys

import numpy as np
import rasterio
from rasterio.transform import xy as rio_xy
from rasterio.warp import transform as rio_transform

HF_REPO = "mayrajeo/marine-vessel-yolo"
HF_FILE = "yolo11s_tci.pt"


# --------------------------------------------------------------------------- model


def load_model(device="cpu"):
    """Download the weights once (cached) and return an ultralytics YOLO model.

    Do NOT use the model card's `YOLOv8.from_pretrained(...)` snippet -- that class
    is not present in current ultralytics. Plain `YOLO(path)` is correct.
    """
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    weights = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
    model = YOLO(weights)
    model.to(device)
    return model, weights


# ----------------------------------------------------------------------------- io


def parse_scene_meta(path):
    """Planet delivery filenames look like 20230715_190458_12_2439_3B_Visual_clip.tif.

    The leading YYYYMMDD_HHMMSS is the acquisition time in UTC.
    """
    base = os.path.basename(path)
    m = re.search(r"(\d{8})_(\d{6})", base)
    if m:
        d, t = m.group(1), m.group(2)
        acq = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}Z"
    else:
        acq = ""
    scene_id = re.split(r"_3B_|_Visual|\.tif", base)[0]
    return scene_id, acq


def read_visual(path):
    """Read an ortho_visual GeoTIFF as HWC uint8 RGB plus a validity mask.

    Clipped Planet deliveries are nodata (0) outside the AOI; a 4th band, when
    present, is alpha. Both become part of `valid`.
    """
    with rasterio.open(path) as src:
        n = src.count
        rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
        if rgb.dtype != np.uint8:
            # Defensive: an analytic asset was passed by mistake. Stretch per-band
            # to 8-bit so the run still produces something, but warn loudly.
            print(f"  ! {os.path.basename(path)} is {rgb.dtype}, not uint8 -- this "
                  f"is probably an analytic asset. The model wants ortho_visual.",
                  file=sys.stderr)
            out = np.zeros(rgb.shape, np.uint8)
            for b in range(3):
                band = rgb[..., b].astype(np.float32)
                nz = band[band > 0]
                lo, hi = np.percentile(nz, [2, 98]) if nz.size else (0.0, 1.0)
                out[..., b] = np.clip((band - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
            rgb = out

        if n >= 4:
            valid = src.read(4) > 0
        else:
            valid = np.any(rgb > 0, axis=-1)

        return {
            "rgb": np.ascontiguousarray(rgb),
            "valid": valid,
            "transform": src.transform,
            "crs": src.crs,
            "res_x": abs(src.transform.a),
            "res_y": abs(src.transform.e),
        }


def read_udm2_clear(udm2_path, shape):
    """UDM2 band 1 == 1 means 'clear'. Returns a boolean mask, or None."""
    if not udm2_path or not os.path.exists(udm2_path):
        return None
    with rasterio.open(udm2_path) as src:
        clear = src.read(1) == 1
    if clear.shape != shape:
        print(f"  ! UDM2 shape {clear.shape} != scene {shape}; ignoring mask",
              file=sys.stderr)
        return None
    return clear


def find_udm2(scene_path):
    """Planet delivers the mask beside the scene, with _udm2 in place of _Visual."""
    for pat in ("_3B_Visual", "_Visual", "_visual"):
        if pat in scene_path:
            for repl in ("_3B_udm2", "_udm2"):
                cand = scene_path.replace(pat, repl)
                if os.path.exists(cand):
                    return cand
    return None


# ------------------------------------------------------------------------ tiling


def iter_tiles(h, w, size, overlap=0.2):
    """Top-left origins of overlapping square tiles covering an h x w image."""
    step = max(1, int(round(size * (1.0 - overlap))))

    def axis(extent):
        if extent <= size:
            return [0]
        pos = list(range(0, extent - size + 1, step))
        if pos[-1] != extent - size:
            pos.append(extent - size)
        return pos

    return [(x, y) for y in axis(h) for x in axis(w)]


def nms(boxes, scores, iou_thr=0.5):
    """Plain IoU non-maximum suppression; merges duplicates from tile overlap."""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)[::-1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou < iou_thr]
    return keep


# --------------------------------------------------------------------- inference


def predict_tiled(model, rgb, valid, slice_size, imgsz, conf, overlap=0.2,
                  iou_thr=0.5, bgr=True):
    """Tile, run the model on each tile, offset boxes back to scene pixels, NMS.

    CHANNEL ORDER: ultralytics treats a numpy array as BGR and flips it internally,
    so we hand it `rgb[..., ::-1]` by default. If the smoke test detects almost
    nothing, run --sweep, which tries both orders -- an inverted channel order is a
    silent accuracy killer, not an error.
    """
    h, w = rgb.shape[:2]
    boxes, scores = [], []

    for x0, y0 in iter_tiles(h, w, slice_size, overlap):
        tile_valid = valid[y0:y0 + slice_size, x0:x0 + slice_size]
        if not tile_valid.any():
            continue  # entirely outside the clipped AOI
        tile = rgb[y0:y0 + slice_size, x0:x0 + slice_size]
        arr = tile[..., ::-1] if bgr else tile
        res = model.predict(np.ascontiguousarray(arr), imgsz=imgsz, conf=conf,
                            verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy()
        cf = res.boxes.conf.cpu().numpy()
        xyxy[:, [0, 2]] += x0
        xyxy[:, [1, 3]] += y0
        boxes.extend(xyxy.tolist())
        scores.extend(cf.tolist())

    keep = nms(boxes, scores, iou_thr)
    return [(boxes[i], scores[i]) for i in keep]


def predict_sahi(weights, rgb, slice_size, imgsz, conf, overlap=0.2, device="cpu"):
    """SAHI sliced inference. SAHI's ultralytics wrapper expects RGB (it flips to
    BGR itself), which is the opposite of raw ultralytics -- hence no flip here."""
    from sahi.predict import get_sliced_prediction
    from sahi import AutoDetectionModel

    last = None
    for mtype in ("ultralytics", "yolov8"):  # name changed across SAHI versions
        try:
            det = AutoDetectionModel.from_pretrained(
                model_type=mtype, model_path=weights,
                confidence_threshold=conf, device=device, image_size=imgsz)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
    else:
        raise RuntimeError(f"SAHI could not load the model: {last}")

    result = get_sliced_prediction(
        rgb, det,
        slice_height=slice_size, slice_width=slice_size,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        verbose=0)
    out = []
    for p in result.object_prediction_list:
        b = p.bbox
        out.append(([b.minx, b.miny, b.maxx, b.maxy], float(p.score.value)))
    return out


# --------------------------------------------------------------------- geometry


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def boxes_to_records(dets, meta, scene_id, acq, valid, clear, hydrophone,
                     min_len, max_len, wake_ar):
    """Georeference boxes, convert size to metres, apply the small-vessel filter."""
    rows, cols, recs = [], [], []
    for (x1, y1, x2, y2), score in dets:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ci, ri = int(round(cx)), int(round(cy))
        if not (0 <= ri < valid.shape[0] and 0 <= ci < valid.shape[1]):
            continue
        if not valid[ri, ci]:
            continue                       # nodata outside the clipped AOI
        if clear is not None and not clear[ri, ci]:
            continue                       # cloud / shadow / haze per UDM2

        w_m = (x2 - x1) * meta["res_x"]
        h_m = (y2 - y1) * meta["res_y"]
        length = max(w_m, h_m)
        if not (min_len <= length <= max_len):
            continue                       # the small-vessel filter

        rows.append(cy)
        cols.append(cx)
        short = max(min(w_m, h_m), 1e-6)
        recs.append({
            "scene_id": scene_id,
            "acq_time_utc": acq,
            "bbox_w_m": round(w_m, 1),
            "bbox_h_m": round(h_m, 1),
            "length_class_m": round(length, 1),
            "confidence": round(float(score), 4),
            "aspect_ratio": round(length / short, 2),
        })

    if not recs:
        return []

    xs, ys = rio_xy(meta["transform"], rows, cols, offset="center")
    lons, lats = rio_transform(meta["crs"], "EPSG:4326", xs, ys)
    for rec, lon, lat in zip(recs, lons, lats):
        rec["lon"] = round(lon, 6)
        rec["lat"] = round(lat, 6)
        # Crude proxy: a moving boat's wake elongates the box. Only a moving boat
        # radiates strongly, so this feeds the tier-D analysis -- but VALIDATE IT
        # BY EYE on the previews before using it as anything but a flag.
        rec["wake_flag"] = int(rec["aspect_ratio"] >= wake_ar)
        rec["range_km"] = (round(haversine_km(hydrophone[0], hydrophone[1], lon, lat), 3)
                           if hydrophone else "")
    return recs


# ---------------------------------------------------------------------- previews


def save_preview(rgb, dets, path, max_side=2000):
    """Draw boxes for eyeball QC. Glint and wakes are the model's documented false
    positives -- look at these before trusting any count."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(rgb)
    scale = min(1.0, max_side / max(img.size))
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), score in dets:
        pad = 6
        draw.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad],
                       outline=(255, 40, 40), width=3)
        draw.text((x1 - pad, y2 + pad + 2), f"{score:.2f}", fill=(255, 200, 40))
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)


# -------------------------------------------------------------------------- main


FIELDS = ["scene_id", "acq_time_utc", "lon", "lat", "bbox_w_m", "bbox_h_m",
          "length_class_m", "confidence", "aspect_ratio", "wake_flag", "range_km"]


def detect_scene(path, model, weights, args, hydrophone, use_sahi):
    meta = read_visual(path)
    scene_id, acq = parse_scene_meta(path)
    clear = read_udm2_clear(args.udm2 or find_udm2(path), meta["valid"].shape)

    if use_sahi:
        dets = predict_sahi(weights, meta["rgb"], args.slice, args.imgsz,
                            args.conf, args.overlap, args.device)
    else:
        dets = predict_tiled(model, meta["rgb"], meta["valid"], args.slice,
                             args.imgsz, args.conf, args.overlap, args.iou,
                             bgr=not args.rgb)

    recs = boxes_to_records(dets, meta, scene_id, acq, meta["valid"], clear,
                            hydrophone, args.min_length, args.max_length,
                            args.wake_aspect)

    if args.preview_dir:
        # Draw ALL raw boxes, not just the filtered ones -- the point of QC is to
        # see what the model fired on, including the glint and wakes we discard.
        save_preview(meta["rgb"], dets,
                     os.path.join(args.preview_dir, f"{scene_id}.jpg"))
    return recs, len(dets)


def run_sweep(path, model, args):
    """Grid over slice size x imgsz x channel order on ONE scene.

    This is the entire GSD-adaptation strategy: the model's constraint is not
    metres, it is how many pixels a vessel spans at network input. Pick the cell
    with the most plausible detections, confirm on the preview, then freeze it.
    """
    meta = read_visual(path)
    if args.sweep_crop:
        n = args.sweep_crop
        h, w = meta["rgb"].shape[:2]
        y0, x0 = max(0, (h - n) // 2), max(0, (w - n) // 2)
        meta["rgb"] = meta["rgb"][y0:y0 + n, x0:x0 + n]
        meta["valid"] = meta["valid"][y0:y0 + n, x0:x0 + n]
        print(f"\n(sweeping a {n}px centre crop; --sweep-crop 0 for the full scene)")
    print(f"\nSweep on {os.path.basename(path)}  "
          f"({meta['rgb'].shape[1]}x{meta['rgb'].shape[0]} px @ "
          f"{meta['res_x']:.1f} m)\n")
    print(f"{'slice':>6} {'imgsz':>6} {'order':>6} {'raw':>6} {'small':>6}")
    print("-" * 34)
    for slice_size in (256, 320, 512):
        for imgsz in (640, 1024):
            for bgr in (True, False):
                dets = predict_tiled(model, meta["rgb"], meta["valid"], slice_size,
                                     imgsz, args.conf, args.overlap, args.iou,
                                     bgr=bgr)
                small = sum(1 for (x1, y1, x2, y2), _ in dets
                            if max((x2 - x1) * meta["res_x"],
                                   (y2 - y1) * meta["res_y"]) <= args.max_length)
                print(f"{slice_size:>6} {imgsz:>6} {'BGR' if bgr else 'RGB':>6} "
                      f"{len(dets):>6} {small:>6}")
    print("\nPick the best cell, then rerun without --sweep using "
          "--slice/--imgsz (add --rgb if RGB won).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", required=True,
                    help="ortho_visual GeoTIFFs (globs allowed)")
    ap.add_argument("--out", default="detections.csv")
    ap.add_argument("--hydrophone", default=None,
                    help="lon,lat of the hydrophone -- pull from ONC device "
                         "metadata, do not guess. Omit to leave range_km blank.")
    ap.add_argument("--udm2", default=None,
                    help="explicit UDM2 path (default: auto-find beside the scene)")
    ap.add_argument("--slice", type=int, default=320, help="tile size in pixels")
    ap.add_argument("--imgsz", type=int, default=640, help="network input size")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--min-length", type=float, default=0.0)
    ap.add_argument("--max-length", type=float, default=20.0,
                    help="small-vessel filter: keep detections at or under this "
                         "many metres (default 20)")
    ap.add_argument("--wake-aspect", type=float, default=2.5,
                    help="aspect ratio above which wake_flag is set")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rgb", action="store_true",
                    help="feed RGB instead of BGR to ultralytics (see --sweep)")
    ap.add_argument("--sahi", action="store_true", help="use SAHI instead of the "
                                                        "built-in tiler")
    ap.add_argument("--preview-dir", default=None,
                    help="write annotated JPEGs here for eyeball QC")
    ap.add_argument("--sweep", action="store_true",
                    help="grid slice x imgsz x channel order on the first scene")
    ap.add_argument("--sweep-crop", type=int, default=1500,
                    help="sweep a centre crop of this many px (0 = full scene). "
                         "12 configs on a full 3300px scene is ~30 min on CPU.")
    args = ap.parse_args()

    paths = []
    for pat in args.scenes:
        paths.extend(sorted(glob.glob(pat)) or ([pat] if os.path.exists(pat) else []))
    if not paths:
        sys.exit("No scenes matched.")

    hydrophone = None
    if args.hydrophone:
        lon, lat = (float(v) for v in args.hydrophone.split(","))
        hydrophone = (lon, lat)
    else:
        print("! No --hydrophone given; range_km will be blank.", file=sys.stderr)

    print(f"Loading {HF_REPO}/{HF_FILE} ...")
    model, weights = load_model(args.device)

    if args.sweep:
        run_sweep(paths[0], model, args)
        return

    all_recs, total_raw = [], 0
    for i, p in enumerate(paths, 1):
        recs, raw = detect_scene(p, model, weights, args, hydrophone, args.sahi)
        total_raw += raw
        all_recs.extend(recs)
        print(f"[{i}/{len(paths)}] {os.path.basename(p)}: "
              f"{raw} raw -> {len(recs)} small vessels")

    with open(args.out, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=FIELDS)
        wtr.writeheader()
        wtr.writerows(all_recs)

    print(f"\n{len(all_recs)} detections from {total_raw} raw across "
          f"{len(paths)} scenes -> {args.out}")
    if args.preview_dir:
        print(f"QC the previews in {args.preview_dir}/ -- glint and wakes are the "
              f"documented false positives.")


if __name__ == "__main__":
    main()
