#!/usr/bin/env python3
"""One board holding every review window's panel, in time order. Acoustic or optical.

Runnable entry point. DEFINES NOTHING SHARED (CLAUDE.md invariant 6): the panels are
files `scripts/build_review_set.py` and `scripts/attach_planet_previews.py` already
wrote into each window folder, and the vessel counts come from the events table. This
script composes; it computes nothing.

ONE SCRIPT, TWO BOARDS, and that is the point. `--panel` names which file to take from
each window folder, so the acoustic board and the optical board are the same grid in
the same order with the same captions, and a panel in cell 17 of one is the same
overpass as cell 17 of the other. Two scripts would drift.

    --panel denoised_labeled.png              the acoustic board
    --panel planet_scene/scene_preview.jpg    the optical board

WHAT IT IS FOR. The review set puts one overpass per folder, which is right for
judging a single window against a single image and wrong for seeing the corpus. Side
by side, the panels answer questions no individual sheet can: whether the quiet
windows cluster in a season, whether the estimator's misses look different from its
hits, whether anything about the 2025 scenes stands out.

READING ORDER is chronological, left to right then top to bottom, so a row is roughly
a season.

THE PANEL COUNT IS 30, NOT 29, AND THAT IS NOT AN ERROR. There are 29 labelled
INSTANTS and 30 SCENES: 20210616_183111 and 20210616_183140 are 29 seconds apart, so
one manual vessel count covers both. Both scenes have their own acoustic window and
their own figure, so both are drawn, and the pair is annotated `label shared`. Dropping
one would silently discard a real acoustic window; merging them would invent a count.

UNITS, at the boundary (decision 0002 §4): the panels are in the product's own
uncalibrated integer `counts`, never dB re 1 µPa. Times are absolute UTC.

Usage:  python3 scripts/build_review_contact_sheet.py [--review-dir DIR]
                                                      [--events-dir DIR]
                                                      [--panel NAME] [--out PATH]
                                                      [--dpi N] [--title TEXT]
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from boatphone import config
from boatphone.paths import DERIVED_DIR, REPO_ROOT

# Panel file drawn from each window folder. Source: the filenames build_review_set.py
# writes; `denoised_labeled` is the seasonal-ambient-subtracted view with the detected
# events annotated, which is the one that reads at contact-sheet size.
DEFAULT_PANEL_NAME = "denoised_labeled.png"

# What each known panel IS, said on the board itself. A contact sheet outlives the
# session that made it, and a grid of spectrograms with no unit on it is exactly the
# kind of figure that gets quoted as decibels two months later.
PANEL_DESCRIPTIONS = {
    "denoised_labeled.png": (
        "Denoised band spectrogram -- ACOUSTIC",
        "seasonal ambient removed; levels are {level_unit}; window is "
        "+/-{half_window_min} min around the acquisition instant",
    ),
    "planet_scene/scene_preview.jpg": (
        "PlanetScope RGB tile preview -- OPTICAL",
        "search-time preview at ~3.15 m/px, 8-bit and JPEG-compressed: LOOK, do not "
        "measure; the image is ONE INSTANT, the acoustic window is "
        "+/-{half_window_min} min around it",
    ),
}

# Columns in the grid. Source: 30 panels divide evenly by 6 (5 full rows), and the
# panels are portrait, so a wide grid keeps the sheet close to square.
GRID_COLUMNS = 6

# Inches per panel and output resolution. Chosen so the axis labels inside each
# panel survive the downscale; at 150 dpi they do not.
PANEL_WIDTH_IN = 4.6
PANEL_HEIGHT_IN = 5.0
DEFAULT_DPI = 180

# The low-resolution companion. The full sheet is ~16 MB, which is right for zooming
# into and wrong for embedding in a notebook, so the preview exists to show the LAYOUT
# at a few hundred kB while the notebook links the real one for reading. JPEG, not PNG:
# a PNG of this figure is still ~4 MB, which is bigger than the notebook it goes in.
# The preview is for orientation ONLY -- never read a level off it.
PREVIEW_LONG_EDGE_PX = 1600
PREVIEW_JPEG_QUALITY = 80


def _latest_run(parent: pathlib.Path) -> pathlib.Path:
    """The most recent timestamped run directory under `parent`.

    Runs are never overwritten, so "latest" is a real choice and is REPORTED rather
    than assumed: the caller prints which one was used and the provenance records it.
    """
    if not parent.is_dir():
        raise FileNotFoundError(
            f"{parent} does not exist -- run the script that writes it first")
    runs = sorted(p for p in parent.glob("*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"{parent} holds no run directories")
    return runs[-1]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        # A missing commit is recorded as unknown, never as a plausible-looking value.
        return "unknown"


def _counts_by_scene(events_dir: pathlib.Path) -> dict[str, dict[str, str]]:
    """Manual and estimated vessel counts, keyed by scene id.

    An ABSENT events table is not fatal -- the sheet still draws, with the counts
    omitted -- but it is reported, because a caption silently missing its numbers
    reads as "there were none".
    """
    wide = events_dir / "events_by_window_wide.csv"
    if not wide.is_file():
        print(f"NOTE: {wide} absent; panels will carry no vessel counts")
        return {}
    with wide.open() as fh:
        return {row["scene_id"]: row for row in csv.DictReader(fh)}


def _window_dirs(review_dir: pathlib.Path, panel_name: str):
    """Window folders holding the requested panel, in CHRONOLOGICAL order.

    Ordering is on the scene id's own timestamp prefix (`YYYYMMDD_HHMMSS`), which is
    the acquisition instant -- verified equal to the `acquired` column for 30/30
    scenes, 0 mismatches beyond 1.5 s. Folders are named `<scene_id>__<coverage>`.
    """
    windows_root = review_dir / "windows"
    if not windows_root.is_dir():
        raise FileNotFoundError(f"{windows_root} does not exist")

    found, missing = [], []
    for folder in windows_root.glob("*"):
        if not folder.is_dir():
            continue
        scene_id, _, coverage = folder.name.partition("__")
        panel = folder / panel_name
        if not panel.is_file():
            missing.append(folder.name)
            continue
        acquired = _dt.datetime.strptime(scene_id[:15], "%Y%m%d_%H%M%S").replace(
            tzinfo=_dt.timezone.utc)
        found.append((acquired, scene_id, coverage or "unknown", panel))

    if missing:
        # Invariant 5: a dropped panel is named and counted, never silently skipped.
        print(f"{len(missing)} window folder(s) hold no {panel_name} and are NOT drawn:")
        for name in sorted(missing):
            print("   ", name)
    return sorted(found), missing


def build(review_dir, events_dir, panel_name, out_path, dpi, title=None):
    panels, missing = _window_dirs(review_dir, panel_name)
    if not panels:
        raise SystemExit(f"no {panel_name} found under {review_dir}/windows")
    counts = _counts_by_scene(events_dir)

    n = len(panels)
    n_cols = min(GRID_COLUMNS, n)
    n_rows = -(-n // n_cols)          # ceiling division
    print(f"{n} panels -> {n_rows} rows x {n_cols} columns, "
          f"chronological left to right, top to bottom")

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * PANEL_WIDTH_IN, n_rows * PANEL_HEIGHT_IN + 1.2),
        squeeze=False,
    )

    shared_label_scenes = {s for s, row in counts.items()
                           if (row.get("manual_label_shared_by") or "1") != "1"}

    for slot, ax in enumerate(axes.flat):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if slot >= n:
            ax.set_visible(False)         # trailing empty cells, drawn as nothing
            continue

        acquired, scene_id, coverage, panel = panels[slot]
        ax.imshow(mpimg.imread(panel))

        row = counts.get(scene_id, {})
        manual = row.get("manual_vessel_count") or "-"
        estimated = row.get("n_small_craft_vessels_est") or "-"
        note = "  [label shared]" if scene_id in shared_label_scenes else ""
        partial = "" if coverage == "full" else f"  [{coverage} coverage]"
        ax.set_title(
            f"{slot + 1}. {acquired:%Y-%m-%d %H:%M} UTC{note}{partial}\n"
            f"{scene_id}\n"
            f"vessels: {manual} counted by eye, {estimated} estimated from sound",
            fontsize=8, linespacing=1.35,
        )

    # What the panels are, and in what unit. Unknown panels get an honest generic
    # line rather than a borrowed one from whichever panel was hardcoded first.
    headline, detail = PANEL_DESCRIPTIONS.get(
        panel_name, (panel_name, "panel content not described in PANEL_DESCRIPTIONS"))
    detail = detail.format(
        level_unit=config.FFT_LEVEL_UNIT,
        half_window_min=config.OVERPASS_MATCH_HALF_WINDOW_S // 60,
    )
    fig.suptitle(
        f"{title or headline} -- every PlanetScope overpass window at Folger Deep "
        f"({n} panels, chronological: left to right, top to bottom)\n"
        f"{detail}; review run {review_dir.name}",
        fontsize=13, y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)

    provenance = {
        "produced_by": "scripts/build_denoised_contact_sheet.py",
        "git_commit": _git_commit(),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "review_run": str(review_dir),
        "events_run": str(events_dir),
        "panel_name": panel_name,
        "n_panels": n,
        "n_windows_without_panel": len(missing),
        "windows_without_panel": sorted(missing),
        "grid_rows": n_rows,
        "grid_columns": n_cols,
        "dpi": dpi,
        "level_unit": config.FFT_LEVEL_UNIT,
        "reading_order": "chronological by acquisition instant, left to right, top to bottom",
    }
    prov_path = out_path.with_suffix(".provenance.json")
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")

    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path}  ({size_mb:.1f} MB, {dpi} dpi)")
    print(f"wrote {prov_path}")

    preview_path = _write_preview(out_path)
    if preview_path is not None:
        print(f"wrote {preview_path}  "
              f"({preview_path.stat().st_size / 1e6:.1f} MB, low resolution)")
    return out_path


def _write_preview(full_path: pathlib.Path):
    """A downscaled companion beside the full sheet, or None if Pillow is absent.

    Absence is REPORTED, not swallowed: a missing preview means a notebook cell finds
    nothing to show, and "no preview" is a different situation from "no sheet".
    """
    try:
        from PIL import Image
    except ImportError:
        print("NOTE: Pillow absent, so no downscaled preview was written "
              "(the full-resolution sheet above is unaffected)")
        return None
    preview_path = full_path.with_name(full_path.stem + "_preview.jpg")
    with Image.open(full_path) as im:
        im = im.convert("RGB")
        scale = min(1.0, PREVIEW_LONG_EDGE_PX / max(im.size))
        if scale < 1.0:
            im = im.resize((round(im.width * scale), round(im.height * scale)),
                           Image.LANCZOS)
        im.save(preview_path, quality=PREVIEW_JPEG_QUALITY)
    return preview_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--review-dir", default=None,
                    help="a review run directory; defaults to the most recent")
    ap.add_argument("--events-dir", default=None,
                    help="an events run directory; defaults to the most recent")
    ap.add_argument("--panel", default=DEFAULT_PANEL_NAME,
                    help=f"panel filename inside each window folder "
                         f"(default {DEFAULT_PANEL_NAME})")
    ap.add_argument("--out", default=None,
                    help="output PNG; defaults to <review run>/<panel stem>_contact_sheet.png")
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument("--title", default=None,
                    help="override the headline; the unit/caveat line is kept either way")
    args = ap.parse_args(argv)

    review_dir = (pathlib.Path(args.review_dir) if args.review_dir
                  else _latest_run(DERIVED_DIR / "review"))
    events_dir = (pathlib.Path(args.events_dir) if args.events_dir
                  else _latest_run(DERIVED_DIR / "events"))
    print(f"review run: {review_dir}")
    print(f"events run: {events_dir}")

    stem = pathlib.Path(args.panel).stem
    out_path = (pathlib.Path(args.out) if args.out
                else review_dir / f"{stem}_contact_sheet.png")
    print(f"panel:      {args.panel}")
    print(f"output:     {out_path}")
    build(review_dir, events_dir, args.panel, out_path, args.dpi, args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
