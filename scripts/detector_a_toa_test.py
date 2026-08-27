#!/usr/bin/env python3
"""Three-way test of Detector A's RGB input on ONE acquisition.

The hypothesis: yolo11s_tci.pt was trained on Sentinel-2 L1C, which is
TOP-OF-ATMOSPHERE, and we fed it SURFACE reflectance. Over water, atmospheric
path radiance is most of the TOA signal, so `_sr` removes the bulk of what the
model learned. S2 L1C TCI water sits near DN 50-100; ours measured DN 8-36.

Three renderings of the SAME acquisition -- same water, same boats, same instant:

    visual   Planet's 8-bit colour-corrected render      (already measured: 4 dets)
    sr       ortho_analytic_4b_sr  + FIXED TCI scale
    toa      ortho_analytic_4b     + FIXED TCI scale

CONTROLLED COMPARISON. sr and toa get the IDENTICAL transform, so any difference
between them is the PRODUCT, not the transform. The earlier sr run used
read_visual()'s 2-98% percentile fallback and is therefore not comparable to
either -- it confounded product with transform, which is why it is re-run here.

THE SCALE IS THE L1C TCI RECIPE, NOT A STRETCH (plan section 5.1):
    DN8 = clip(reflectance * 10000 / 10, 0, 255) = clip(DN16 / 10, 0, 255)
Fixed and global, so a scene's radiometry never depends on its own content.
Matches boatphone.optical.TCI_DIVISOR, which defines the same constant but is
currently only applied to synthetic fixtures.
"""
import argparse, re, sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from boatphone import optical, vessel_detect as vd


def reflectance_coefficients(xml_path):
    """Per-band DN -> TOA reflectance factors from Planet's scene metadata.

    ortho_analytic_4b is TOA *RADIANCE* in DN, NOT reflectance. Applying the TCI
    recipe to it directly saturates the image (measured: water at DN 236 of 255).
    The XML carries <ps:reflectanceCoefficient> per band, ~2e-5, which is the only
    correct route from this product to the reflectance the TCI recipe expects.
    ortho_analytic_4b_sr is ALREADY reflectance x PLANET_QUANT and needs no
    coefficient -- that asymmetry is the whole reason this function exists.
    """
    txt = Path(xml_path).read_text()
    bands = re.findall(r"<ps:bandNumber>(\d+)</ps:bandNumber>", txt)
    coefs = re.findall(r"<ps:reflectanceCoefficient>([0-9.eE+-]+)</ps:reflectanceCoefficient>", txt)
    assert len(bands) == len(coefs) >= 4, f"{xml_path}: {len(bands)} bands, {len(coefs)} coefs"
    return {int(b): float(c) for b, c in zip(bands, coefs)}


def tci_rgb(path, coefs=None):
    """Analytic 4-band -> 8-bit RGB by the FIXED L1C TCI scale.

    coefs=None  -> input is _sr, already reflectance * PLANET_QUANT
    coefs given -> input is TOA radiance DN; multiply to reflectance first
    Either way the final step is identical: DN8 = clip(reflectance * 1000, 0, 255),
    which is L1C TCI's reflectance * 10000 / 10.
    """
    with rasterio.open(path) as s:
        assert s.count == 4, f"{path}: expected 4 bands, got {s.count}"
        a = s.read().astype(np.float32)
        meta = {"res_x": abs(s.transform.a), "res_y": abs(s.transform.e),
                "transform": s.transform, "crs": s.crs}
    if coefs is None:
        refl = a / optical.PLANET_QUANT
    else:
        refl = np.stack([a[i] * coefs[i + 1] for i in range(4)])
    scale = optical.PLANET_QUANT / optical.TCI_DIVISOR          # = 1000
    rgb = np.clip(refl[[2, 1, 0]] * scale, 0, 255).astype(np.uint8)
    rgb = np.ascontiguousarray(rgb.transpose(1, 2, 0))
    return rgb, (a.sum(0) > 0), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toa", required=True)
    ap.add_argument("--sr", required=True)
    ap.add_argument("--visual", default=None)
    ap.add_argument("--udm2", default=None)
    ap.add_argument("--slice", type=int, default=640)   # Neve's measured peak, ratio 1.0
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--preview-dir", default=None)
    args = ap.parse_args()

    model, _weights = vd.load_model("cpu")   # returns (model, weights_path)
    scene_id, acq = vd.parse_scene_meta(args.toa)
    clear_path = args.udm2          # read per-run, once the scene shape is known

    runs = [("toa", args.toa), ("sr", args.sr)]
    if args.visual:
        runs.append(("visual", args.visual))

    print(f"\nscene {scene_id}   slice {args.slice} -> imgsz {args.imgsz}   conf {args.conf}\n")
    print(f"{'input':>7} {'water DN':>9} {'p99 DN':>7} {'raw':>5} {'kept':>5}  lengths (m)")
    print("-" * 74)
    for name, path in runs:
        if name == "visual":
            m = vd.read_visual(path)
            rgb, valid, meta = m["rgb"], m["valid"], m
        else:
            coefs = None
            if name == "toa":
                xml = str(path).replace("_AnalyticMS_clip.tif",
                                        "_AnalyticMS_metadata_clip.xml")
                coefs = reflectance_coefficients(xml)
            rgb, valid, meta = tci_rgb(path, coefs)
        c = vd.read_udm2_clear(clear_path, valid.shape) if clear_path else None
        dets = vd.predict_tiled(model, rgb, valid, args.slice, args.imgsz,
                                args.conf, 0.2, 0.5, bgr=True)
        recs = vd.boxes_to_records(dets, meta, scene_id, acq, valid, c,
                                   optical.HYDROPHONE_LONLAT,
                                   0.0, 150.0, 2.5)
        lens = sorted(round(r["length_class_m"]) for r in recs)
        med = float(np.median(rgb[valid])) if valid.any() else float("nan")
        p99 = float(np.percentile(rgb[valid], 99)) if valid.any() else float("nan")
        print(f"{name:>7} {med:9.1f} {p99:7.1f} {len(dets):5d} {len(recs):5d}  {lens[:12]}")
        if args.preview_dir:
            Path(args.preview_dir).mkdir(parents=True, exist_ok=True)
            vd.save_preview(rgb, dets, f"{args.preview_dir}/{scene_id}_{name}.jpg")
    print("\nS2 L1C TCI water for reference: DN ~50-100. If toa lands there and sr does "
          "not,\nthe product was the problem, not the transform.")


if __name__ == "__main__":
    main()
