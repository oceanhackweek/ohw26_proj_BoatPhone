#!/usr/bin/env python3
"""Pull the acoustic window for a NAMED scene, outside the corpus's time strip.

Runnable entry point. DEFINES NOTHING SHARED (CLAUDE.md invariant 6): the window
half-width, device and landing zone come from `boatphone.config`/`paths`; the
listing and download from `boatphone.onc_client`; the scene times from
`boatphone.overpasses`.

WHY THIS EXISTS. The bulk pull targeted 09:15-11:45 America/Vancouver, from an
assumption `acoustics_plan_v2.md` SS9 flagged as unverified. The real PlanetScope
overpasses at Folger fall 18:17-19:49 UTC (11:17-12:49 local), so 12 of the 30
gate2 scenes have NO acoustic data and 5 more are partial. When a scene acquires
an optical LABEL, its acoustic window suddenly matters far more than the 26,666
unlabelled ones -- and for the off-strip scenes there is nothing on disk to look
at. This pulls exactly those windows and nothing else.

IT LANDS IN A SEPARATE DIRECTORY (`paths.ONC_LABELLED_WINDOW_DIR`), never in the
corpus. `ONC_OVERPASS_CORPUS_DIR` means "every in-season date, 09:15-11:45 local,
nothing else", and `config.PLANET_SAMPLING_CONDITIONALITY_STATEMENT` is written
against that definition. Off-strip files dropped in there would falsify every
population statistic over the directory silently -- no error, just a wrong
number with a correct-looking caption.

Usage:
    python3 scripts/pull_labelled_windows.py --scene 20210601_191849_05_2412
    python3 scripts/pull_labelled_windows.py --scene ID --dry-run

`--dry-run` lists what WOULD be fetched, makes no request and reads no
credential.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from boatphone import config, credentials, onc_client, paths
from boatphone import overpasses as ov


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True,
                              cwd=paths.REPO_ROOT).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", action="append", required=True,
                        help="gate2 scene id; repeatable")
    parser.add_argument("--dest-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    dest_dir = pathlib.Path(args.dest_dir) if args.dest_dir else paths.ONC_LABELLED_WINDOW_DIR

    by_id = {o.scene_id: o for o in ov.load_gate2_overpasses()}
    wanted = []
    for sid in args.scene:
        if sid not in by_id:
            raise SystemExit(
                f"scene {sid!r} is not in {config.PLANET_GATE2_SURVIVORS_RELPATH}. "
                f"Known ids include: {sorted(by_id)[:3]} ... ({len(by_id)} total)"
            )
        wanted.append(by_id[sid])

    print(f"landing zone: {dest_dir}")
    print(f"window half-width: {config.OVERPASS_MATCH_HALF_WINDOW_S} s "
          f"(config.OVERPASS_MATCH_HALF_WINDOW_S)\n")

    if args.dry_run:
        for op in wanted:
            lo, hi = op.window_utc()
            print(f"{op.scene_id}  acquired {op.acquired_utc.isoformat()}")
            print(f"  would list {config.DEVICE_CODE} over {lo.isoformat()} -> {hi.isoformat()}")
        print("\n--dry-run: no request made, no credential read")
        return 0

    client = credentials.get_onc_client()
    locations = onc_client.discover_folger_locations(client)
    transport = onc_client.RequestsArchiveTransport(client)
    paths.ensure_dir(dest_dir)

    records, absences = [], []
    for op in wanted:
        lo, hi = op.window_utc()
        # PAD THE LISTING START BY ONE FILE DURATION. ONC's listing selects on
        # file START TIME, but window_coverage counts by INTERVAL OVERLAP
        # (decision 0020): the file straddling the window start overlaps the
        # window while starting before it, so an unpadded listing silently omits
        # it and loses up to FFT_FILE_SECONDS at the HEAD of every window.
        # Measured 2026-08-28 on 20210812_191832_32_2406: the unpadded listing
        # returns 6 files and leaves a 269 s head gap; padded it returns 7,
        # including ICLISTENHF1266_20210812T190301.000Z, which closes it.
        # The end needs no pad -- a file starting before `hi` is already listed,
        # and its tail simply overhangs, which overlap-counting handles.
        listing_lo = lo - _dt.timedelta(seconds=config.FFT_FILE_SECONDS)
        print(f"{op.scene_id}  {lo.isoformat()} -> {hi.isoformat()}")
        print(f"  listing from {listing_lo.isoformat()} "
              f"(-{config.FFT_FILE_SECONDS}s head pad, decision 0020 overlap)")
        filenames, n_empty = onc_client.list_fft_files(
            client, locations, listing_lo, hi, allow_empty=True)
        if not filenames:
            # An empty listing over a deployed span is a MEASURED ZERO, not a
            # failure (decisions 0008, 0028). Recorded, never retried into a lie.
            print("  EMPTY LISTING -- measured zero, not an error")
            absences.append({"scene_id": op.scene_id, "reason": "empty_listing",
                             "window_utc": [lo.isoformat(), hi.isoformat()]})
            continue
        print(f"  {len(filenames)} file(s) listed")
        for name in filenames:
            rec = onc_client.download_archive_file(
                client, name, dest_dir=dest_dir, transport=transport)
            # DownloadRecord is a plain class, not a dict and not a dataclass.
            # Its fields are listed explicitly so a field added upstream shows up
            # as a missing key here rather than silently vanishing from the
            # manifest.
            row = {f: getattr(rec, f) for f in (
                "filename", "status", "path", "sha256", "bytes_downloaded",
                "http_status", "attempts", "total_size_known", "message")}
            row["path"] = str(row["path"]) if row["path"] else None
            row["scene_id"] = op.scene_id
            records.append(row)
            where = pathlib.Path(rec.path).name if rec.path else "(nothing on disk)"
            print(f"    {rec.status:>13}  {where}")

    manifest = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "produced_by": "scripts/pull_labelled_windows.py",
        "scenes": [o.scene_id for o in wanted],
        "dest_dir": str(dest_dir),
        "half_window_s": config.OVERPASS_MATCH_HALF_WINDOW_S,
        "device_code": config.DEVICE_CODE,
        "files": records,
        "absences": absences,
        "why_separate_from_corpus": (
            "These windows lie OUTSIDE the 09:15-11:45 America/Vancouver strip that "
            "defines ONC_OVERPASS_CORPUS_DIR. Mixing them in would silently falsify "
            "every population statistic over that directory and every caption citing "
            "config.PLANET_SAMPLING_CONDITIONALITY_STATEMENT."
        ),
    }
    out = dest_dir / "pull_labelled_windows.manifest.json"
    out.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(f"\n{len(records)} file(s), {len(absences)} absence(s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
