#!/usr/bin/env python3
"""Chip every frozen-detector detection for the box-level census.

WHY THIS EXISTS. The detector in scripts/detect_vessels_final.py is calibrated against
a COUNT per scene (data/validation/planet_scope_validations.csv), so a scene can carry
the right number of vessels for the wrong reasons. The source of truth records the
open gate plainly: a count that matches can still be the wrong object, and 17 of the
26 scenes have had no box-level human pass at all. This script renders the evidence
that pass needs -- one panel row per detection, at native and magnified scale -- and
emits a verdict sheet to fill in.

WHAT EACH ROW SHOWS, and why four panels rather than one:
    TOA context   300 m around the box, the rendering the box was found in
    SR  context   the same ground at the same instant, the corroborating rendering
    NIR context   band 4, window-stretched -- water is dark, a hull/wake is bright
    TOA tight     75 m at ~20x, where a 2-3 px object is actually resolvable
    NIR tight     the same 75 m in band 4 -- the strongest single piece of evidence,
                  because a hull is a handful of bright pixels against dark water
A vessel should be a compact bright object in all four, sitting in open water. A rock
is bright in NIR too but repeats across years; a whitecap is bright in one rendering
and not the other; a shoreline artefact sits against land.

READING THE NUMBERS IN THE CAPTION. nir_x_water is the box's peak NIR over the scene's
median water NIR -- a contrast ratio, not a detection statistic. length_m is THE BOX,
not the hull: the model draws 8-12 px around a 2-3 px object, so every length lands in
23-37 m. Do not read it as a size (see the detector's module docstring).

THIS SCRIPT MAKES NO JUDGEMENT. It writes `verdict` and `note` columns empty. Filling
them in is the census.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from boatphone import optical                                    # noqa: E402
from detector_a_toa_test import reflectance_coefficients, tci_rgb  # noqa: E402
from sweep_cache import scene_paths                               # noqa: E402

CTX_PX, CTX_ZOOM = 100, 5      # 300 m at 3 m GSD, drawn at 500 px
TIGHT_PX, TIGHT_ZOOM = 25, 20  # 75 m, drawn at 500 px
GUTTER, CAPTION_H, MARGIN = 10, 46, 12
INDEX_FIELDS = ["det_index", "scene_id", "acq_time_utc", "panel_row", "chip_file",
                "range_km", "bearing_deg", "conf_toa", "conf_sr", "length_m",
                "nir_x_water", "px_cx", "px_cy", "verdict", "note"]


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:              # Pillow < 10.1: fixed-size bitmap font
        return ImageFont.load_default()


def crop(arr, cx, cy, half):
    """Window of side 2*half+1 centred on (cx, cy), zero-padded at the scene edge.

    Padding rather than shifting the window: a detection near the edge must stay
    centred in its chip or the box overlay lands in the wrong place.
    """
    h, w = arr.shape[:2]
    y0, y1 = int(round(cy)) - half, int(round(cy)) + half + 1
    x0, x1 = int(round(cx)) - half, int(round(cx)) + half + 1
    out = np.zeros((y1 - y0, x1 - x0) + arr.shape[2:], arr.dtype)
    sy0, sx0 = max(y0, 0), max(x0, 0)
    sy1, sx1 = min(y1, h), min(x1, w)
    if sy1 > sy0 and sx1 > sx0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return out, x0, y0


def stretch(win):
    """Per-window percentile stretch to 8-bit grey.

    Window-local ON PURPOSE: a scene-wide stretch is set by land and glint elsewhere
    in the 100 km2 clip and leaves a 2 px hull indistinguishable from the water.
    """
    finite = win[np.isfinite(win)]
    if finite.size == 0:
        return np.zeros(win.shape, np.uint8)
    lo, hi = np.percentile(finite, 2), np.percentile(finite, 99.5)
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    g = np.clip((win - lo) / (hi - lo), 0, 1) * 255
    return np.stack([g.astype(np.uint8)] * 3, -1)


def panel(rgb, cx, cy, box, half, zoom, label, ticks=False, prepared=None):
    """One magnified panel with the detection box drawn on it.

    `prepared` is (win, x0, y0) for a panel whose window was cropped and rendered
    upstream -- the NIR panel, whose stretch has to happen on the window itself.
    Passing the stretched window back through crop() would re-crop it at scene
    coordinates and yield a black panel.
    """
    win, x0, y0 = prepared if prepared is not None else crop(rgb, cx, cy, half)
    img = Image.fromarray(win).resize((win.shape[1] * zoom, win.shape[0] * zoom),
                                      Image.NEAREST)
    d = ImageDraw.Draw(img)
    bx1, by1, bx2, by2 = [(v - o) * zoom for v, o in
                          zip(box, (x0, y0, x0, y0))]
    if ticks:
        # Corner ticks, not a closed rectangle: at 20x the box edge would otherwise
        # sit on top of the 2-3 px object it is meant to point at.
        t = max(int(6 * zoom / 5), 8)
        for x, y, dx, dy in ((bx1, by1, 1, 1), (bx2, by1, -1, 1),
                             (bx1, by2, 1, -1), (bx2, by2, -1, -1)):
            d.line([(x, y), (x + dx * t, y)], fill=(255, 60, 60), width=2)
            d.line([(x, y), (x, y + dy * t)], fill=(255, 60, 60), width=2)
    else:
        d.rectangle([bx1, by1, bx2, by2], outline=(255, 60, 60), width=2)
    d.rectangle([0, 0, img.width - 1, img.height - 1], outline=(90, 90, 90), width=1)
    d.text((6, 6), label, fill=(255, 220, 60), font=_font(15))
    return img


def sheet(scene_id, rows, rgb_toa, rgb_sr, nir, out_path):
    """One PNG for a scene: a four-panel row per detection, captioned."""
    pw = CTX_PX * CTX_ZOOM
    width = MARGIN * 2 + pw * 5 + GUTTER * 4
    row_h = CAPTION_H + pw + GUTTER
    img = Image.new("RGB", (width, MARGIN * 2 + 26 + row_h * len(rows)), (18, 18, 22))
    d = ImageDraw.Draw(img)
    d.text((MARGIN, MARGIN), f"{scene_id}   {rows[0]['acq_time_utc']}   "
           f"{len(rows)} detection(s)   panels: TOA 300 m | SR 300 m | NIR 300 m | "
           f"TOA 75 m @20x | NIR 75 m @20x", fill=(210, 210, 220), font=_font(16))
    y = MARGIN + 26
    for r in rows:
        cx, cy = r["px_cx"], r["px_cy"]
        box = (r["px_x1"], r["px_y1"], r["px_x2"], r["px_y2"])
        nw, nx0, ny0 = crop(nir, cx, cy, CTX_PX // 2)
        nir_ctx = (stretch(nw), nx0, ny0)
        nt, tx0, ty0 = crop(nir, cx, cy, TIGHT_PX // 2)
        nir_tight = (stretch(nt), tx0, ty0)
        cap = (f"[{r['panel_row']}] det_index {r['det_index']}   "
               f"range {r['range_km']} km   bearing {r['bearing_deg']} deg   "
               f"conf TOA {r['conf_toa']} / SR {r['conf_sr']}   "
               f"box length {r['length_m']} m (BOX, not hull)   "
               f"NIR peak x water {r['nir_x_water']}   px ({int(cx)}, {int(cy)})")
        d.text((MARGIN, y + 12), cap, fill=(190, 200, 215), font=_font(15))
        for k, p in enumerate((
                panel(rgb_toa, cx, cy, box, CTX_PX // 2, CTX_ZOOM, "TOA 300 m"),
                panel(rgb_sr, cx, cy, box, CTX_PX // 2, CTX_ZOOM, "SR 300 m"),
                panel(None, cx, cy, box, CTX_PX // 2, CTX_ZOOM,
                      "NIR 300 m (window-stretched)", prepared=nir_ctx),
                panel(rgb_toa, cx, cy, box, TIGHT_PX // 2, TIGHT_ZOOM,
                      "TOA 75 m @20x", ticks=True),
                panel(None, cx, cy, box, TIGHT_PX // 2, TIGHT_ZOOM,
                      "NIR 75 m @20x", ticks=True, prepared=nir_tight))):
            img.paste(p, (MARGIN + k * (pw + GUTTER), y + CAPTION_H))
        y += row_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return img.size


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detections", default="data/derived/detections_final/detections_final.csv")
    ap.add_argument("--out-dir", default="data/derived/census")
    ap.add_argument("--scenes", nargs="+", default=None)
    args = ap.parse_args()

    det_path = Path(args.detections)
    if not det_path.exists():
        sys.exit(f"no detections file at {det_path} -- run scripts/detect_vessels_final.py first")
    with det_path.open() as f:
        det = list(csv.DictReader(f))
    if not det:
        sys.exit(f"{det_path} has no rows")

    allp = scene_paths()
    by_scene = {}
    for i, r in enumerate(det):
        by_scene.setdefault(r["scene_id"], []).append((i, r))
    wanted = args.scenes or sorted(by_scene)
    absent = [s for s in wanted if s not in allp]
    if absent:
        # Invariant 5: name what is missing rather than skipping it quietly.
        print(f"! imagery absent for {len(absent)} scene(s), not chipped: {absent}",
              file=sys.stderr)

    out = Path(args.out_dir)
    index = []
    for n, sid in enumerate([s for s in wanted if s in allp], 1):
        p = allp[sid]
        xml = str(p["toa"]).replace("_AnalyticMS_clip.tif", "_AnalyticMS_metadata_clip.xml")
        rgb_toa, _, _ = tci_rgb(p["toa"], reflectance_coefficients(xml))
        rgb_sr, _, _ = tci_rgb(p["sr"], None)
        with rasterio.open(p["sr"]) as src:
            nir = src.read(4).astype(np.float32) / optical.PLANET_QUANT

        rows = []
        for k, (i, r) in enumerate(by_scene[sid], 1):
            px = {f: float(r[f]) for f in ("px_x1", "px_y1", "px_x2", "px_y2")}
            rows.append({**r, **px, "det_index": i, "panel_row": k,
                         "px_cx": (px["px_x1"] + px["px_x2"]) / 2,
                         "px_cy": (px["px_y1"] + px["px_y2"]) / 2})
        chip = out / f"{sid}.png"
        size = sheet(sid, rows, rgb_toa, rgb_sr, nir, chip)
        print(f"[{n}] {sid}: {len(rows)} row(s) -> {chip} {size}", flush=True)
        for r in rows:
            index.append({**{f: r.get(f, "") for f in INDEX_FIELDS},
                          "chip_file": chip.name, "verdict": "", "note": "",
                          "px_cx": round(r["px_cx"], 1), "px_cy": round(r["px_cy"], 1)})

    optical.write_csv(out / "census_index.csv", index, INDEX_FIELDS)
    print(f"\n{len(index)} detections over {len(set(r['scene_id'] for r in index))} scenes"
          f"\n  -> {out}/  (one PNG per scene)"
          f"\n  -> {out/'census_index.csv'}  (verdict and note are EMPTY -- fill them in)")


if __name__ == "__main__":
    main()
