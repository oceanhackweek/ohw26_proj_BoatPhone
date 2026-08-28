#!/usr/bin/env python3
"""Fill each review window's empty `planet_scene/` slot with its PlanetScope preview.

Runnable entry point. DEFINES NOTHING SHARED (CLAUDE.md invariant 6): the review
folders come from `scripts/build_review_set.py`, the previews come from the shared
delivery, and the join key is the scene id both sides already carry.

WHAT THIS CLOSES. `build_review_set.py` writes one folder per overpass holding the
acoustic evidence and an EMPTY `planet_scene/` -- the slot the optical arm fills, so a
human can put sound and image side by side. This copies the search-time RGB tile
preview into that slot.

A PREVIEW IS FOR LOOKING, NOT FOR MEASURING, and the copy says so in its own
provenance file. The tile previews are 8-bit, resampled to ~3.15 m/px and JPEG
compressed, so nothing radiometric survives them: no threshold, no reflectance and no
detection may be computed from one. Measurement runs on the analytic bundles
(`ortho_analytic_4b` / `_sr`) instead -- see the delivery README named below.

THE PREVIEW SET IS A SUPERSET OF THE DELIVERED IMAGERY. 30 previews exist against 26
ordered scenes, because previews cost tile quota rather than imagery quota and were
built for the whole gate-2 survivor set. Three of the four extras are validation
timestamps we hold manual vessel counts for but no delivered scene, so for those the
preview is the ONLY picture that exists. That is a reason to attach previews, and it
is also why a window whose preview is missing is NAMED here rather than skipped.

THE FILE IS COPIED, NOT SYMLINKED, on purpose: a review run is meant to survive being
moved or archived, and a symlink into another user's tree does not. The cost is one
extra copy of ~140 MB under `data/derived/`, which is gitignored.

LICENCE, because this moves licensed imagery: the source lives in `shared-public`,
which every account on this hub can read. Keep it inside the hackweek and do not
republish it. This script only copies within the hub.

Usage:
    python3 scripts/attach_planet_previews.py [--review-dir DIR] [--preview-dir DIR]
                                              [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from boatphone.paths import DERIVED_DIR, REPO_ROOT

# The shared Planet delivery. Source: shared-public/boatphone_shared/planet_folger/README.md
# (Malachy, 2026-08-28) -- tile_previews/ sits beside downloads/ because it is a
# search-stage product, not a Planet order delivery.
DEFAULT_PREVIEW_DIR = pathlib.Path(
    "/home/jovyan/shared-public/boatphone_shared/planet_folger/tile_previews")

# Name the copy lands under, the SAME in every window folder. Deliberate: a constant
# filename is what lets one contact-sheet script address the optical panel and the
# acoustic panel identically. The original filename is recorded in the provenance.
PANEL_FILENAME = "scene_preview.jpg"

# Carried into every provenance file so the caveat travels with the copy rather than
# living only in a README the reader may never open.
PREVIEW_CAVEAT = (
    "Search-time RGB tile mosaic, 3584x3584 px at ~3.15 m/px. 8-bit, resampled and "
    "JPEG-compressed: NOTHING RADIOMETRIC SURVIVES IT. Use it to look; use the "
    "ortho_analytic_4b / _sr bundles to measure."
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _latest_run(parent: pathlib.Path) -> pathlib.Path:
    if not parent.is_dir():
        raise FileNotFoundError(f"{parent} does not exist")
    runs = sorted(p for p in parent.glob("*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"{parent} holds no run directories")
    return runs[-1]


def attach(review_dir: pathlib.Path, preview_dir: pathlib.Path,
           dry_run: bool = False, force: bool = False) -> dict:
    windows_root = review_dir / "windows"
    if not windows_root.is_dir():
        raise FileNotFoundError(f"{windows_root} does not exist")
    if not preview_dir.is_dir():
        raise FileNotFoundError(
            f"{preview_dir} does not exist -- is the shared delivery mounted?")

    windows = sorted(p for p in windows_root.glob("*") if p.is_dir())
    copied, already, missing = [], [], []

    for folder in windows:
        scene_id, _, _coverage = folder.name.partition("__")
        source = preview_dir / f"{scene_id}.jpg"
        if not source.is_file():
            # Invariant 5: an absent preview is named, never quietly left as an empty
            # slot that looks the same as a slot nobody tried to fill.
            missing.append(scene_id)
            continue

        slot = folder / "planet_scene"
        target = slot / PANEL_FILENAME
        source_sha = _sha256(source)

        if target.is_file() and not force:
            if _sha256(target) == source_sha:
                already.append(scene_id)
                continue
            # A DIFFERENT file already sits in the slot. Do not overwrite silently:
            # somebody put it there deliberately, and this script does not know why.
            raise SystemExit(
                f"{target} exists and differs from {source}. Re-run with --force to "
                "replace it, once you know what the existing file is.")

        if dry_run:
            copied.append(scene_id)
            continue

        slot.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        (slot / "provenance.json").write_text(json.dumps({
            "produced_by": "scripts/attach_planet_previews.py",
            "git_commit": _git_commit(),
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "scene_id": scene_id,
            "source_path": str(source),
            "source_filename": source.name,
            "source_sha256": source_sha,
            "source_bytes": source.stat().st_size,
            "product": "PlanetScope search-time RGB tile preview",
            "caveat": PREVIEW_CAVEAT,
            "licence_note": (
                "Licensed Planet imagery, copied from shared-public on the OHW hub. "
                "Keep inside the hackweek; do not republish."),
        }, indent=2) + "\n")
        copied.append(scene_id)

    verb = "would copy" if dry_run else "copied"
    print(f"{verb} {len(copied)} preview(s) into {review_dir}/windows/*/planet_scene/")
    if already:
        print(f"{len(already)} already present and identical, left alone")
    if missing:
        print(f"{len(missing)} window(s) have NO preview in {preview_dir} and are still empty:")
        for scene_id in missing:
            print("   ", scene_id)
    print(f"{len(windows)} window folder(s) examined")
    return {"copied": copied, "already": already, "missing": missing,
            "n_windows": len(windows)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--review-dir", default=None,
                    help="a review run directory; defaults to the most recent")
    ap.add_argument("--preview-dir", default=str(DEFAULT_PREVIEW_DIR))
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would be copied; write nothing")
    ap.add_argument("--force", action="store_true",
                    help="replace a DIFFERENT file already occupying the slot")
    args = ap.parse_args(argv)

    review_dir = (pathlib.Path(args.review_dir) if args.review_dir
                  else _latest_run(DERIVED_DIR / "review"))
    print(f"review run:  {review_dir}")
    print(f"preview dir: {args.preview_dir}")
    attach(review_dir, pathlib.Path(args.preview_dir), args.dry_run, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
