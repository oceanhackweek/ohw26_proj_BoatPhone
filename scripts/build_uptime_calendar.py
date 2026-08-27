#!/usr/bin/env python3
"""Build the A1 hydrophone uptime calendar (deliverable O1) from ONC's archive listing.

Runnable entry point. It DEFINES NOTHING SHARED (CLAUDE.md invariant 6): every
constant comes from `boatphone.config` / `boatphone.paths`, and the calendar
itself is built by `boatphone.onc_client`, not reimplemented here.

Three artifacts land in the output directory (default `paths.DERIVED_DIR`):

* `hydrophone_uptime.csv`  -- one row per in-season `config.BIN_SECONDS` bin,
  columns `onc_client.UPTIME_CSV_HEADER`, timestamps tz-aware UTC isoformat.
* `deployments.csv`        -- ONC deployment metadata, half-open `[start, end)`.
* `hydrophone_uptime.provenance.json` -- what produced the numbers.

Time base: UTC end to end (decision 0002). `--start`/`--end` are parsed as UTC
and a naive string is read as UTC explicitly, never as local time.

**`source` is `"listing"`, and that is load-bearing (A1 decision D3).** This
calendar records ONC's *belief* that a file exists; it is not proof that the file
downloads. A4's pull refines it and, where the two disagree, **the pull wins**.
The provenance carries that distinction so a later, pull-refined calendar is
distinguishable from this one by inspection rather than by memory.

Usage:

    python3 scripts/build_uptime_calendar.py [--start ISO] [--end ISO]
                                             [--out-dir DIR] [--dry-run]

Exit status is 0 on success and non-zero on any error; errors surface with the
span or path that caused them (invariant 5).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone

# scripts/ is a directory of entry points, not a package, so make the repo root
# importable when this file is run directly (`python3 scripts/build_uptime_calendar.py`).
# Path manipulation only -- no network, no directory creation, no parsing at import.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from boatphone import config, paths
from boatphone import onc_client
from boatphone.credentials import get_onc_client


def parse_utc(value):
    """An ISO-8601 string -> tz-aware UTC datetime. A naive string is read AS UTC.

    Stated rather than assumed (decision 0002): this tool has no local time base,
    so `2024-05-01` means `2024-05-01T00:00:00+00:00` and a string carrying a
    non-zero offset is converted to UTC by its own stated offset, never by the
    machine's timezone.
    """
    if isinstance(value, datetime):
        moment = value
    else:
        try:
            moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"{value!r} is not an ISO-8601 UTC datetime (e.g. 2024-05-01 or "
                f"2024-05-01T00:00:00Z): {exc}"
            ) from exc
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.replace(tzinfo=timezone.utc) - moment.utcoffset()


def build_parser() -> argparse.ArgumentParser:
    """The CLI. Defaults come from `boatphone/`, so there is one study window."""
    parser = argparse.ArgumentParser(
        prog="build_uptime_calendar.py",
        description=(
            "Build the hydrophone uptime calendar (O1) from the ONC archive listing. "
            "Availability is ONC's belief a file exists, not proof it downloads: "
            "A4's pull refines it and the pull wins."
        ),
    )
    parser.add_argument(
        "--start", type=parse_utc, default=config.STUDY_START_UTC,
        help="inclusive UTC start of the span to scan (default: config.STUDY_START_UTC, "
             f"{config.STUDY_START_UTC.isoformat()})",
    )
    parser.add_argument(
        "--end", type=parse_utc, default=config.STUDY_END_UTC,
        help="exclusive UTC end of the span to scan (default: config.STUDY_END_UTC, "
             f"{config.STUDY_END_UTC.isoformat()})",
    )
    parser.add_argument(
        "--out-dir", dest="out_dir", default=paths.DERIVED_DIR,
        help=f"directory to write the three artifacts into (default: {paths.DERIVED_DIR}). "
             "A path under data/ but outside data/derived/ is refused (decision 0001).",
    )
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="print what would be scanned and written, then exit: no network request, "
             "no credential needed, and nothing written",
    )
    return parser


def resolve_out_dir(value) -> pathlib.Path:
    """Resolve `value` to an absolute output directory, refusing acquisition paths.

    Raises `ValueError` when the resolved path lands under `paths.DATA_DIR` but
    NOT under `paths.DERIVED_DIR` -- an acquisition directory, the raw landing
    zone, or a `..` traversal out of `derived/`. `data/` is immutable
    (docs/decisions/0001, hook-enforced), so the refusal happens HERE, before any
    directory is created and before any listing request is made: a run that dies
    half-written has already done the damage.

    A path entirely outside `data/` is allowed and returned as-is, which is what
    makes this entry point drivable from a check without touching `data/`.
    Creates nothing.
    """
    resolved = pathlib.Path(value).expanduser().resolve()
    data_dir = paths.DATA_DIR.resolve()
    derived_dir = paths.DERIVED_DIR.resolve()
    under_data = resolved == data_dir or data_dir in resolved.parents
    under_derived = resolved == derived_dir or derived_dir in resolved.parents
    if under_data and not under_derived:
        raise ValueError(
            f"refusing to write to {resolved}: it is under {data_dir}, which holds "
            f"IMMUTABLE acquisitions (docs/decisions/0001-raw-data-immutability.md), and "
            f"not under {derived_dir}. Pass --out-dir {derived_dir} or a path outside data/."
        )
    return resolved


def artifact_paths(out_dir) -> dict:
    """The three O1 artifact paths under `out_dir`, keyed for the writers.

    Basenames come from `boatphone.paths` because the optical workstream opens
    these files by name. Every value is a direct child of `resolve_out_dir(out_dir)`;
    nothing else is written.
    """
    resolved = resolve_out_dir(out_dir)
    return {
        "uptime_csv": resolved / paths.UPTIME_CSV_NAME,
        "deployments_csv": resolved / paths.DEPLOYMENTS_CSV_NAME,
        "provenance_json": resolved / paths.UPTIME_PROVENANCE_JSON_NAME,
    }


def _git_commit() -> str:
    """The commit this run was made from, or a stated non-empty reason it is unknown."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(paths.REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unknown (git unavailable: {exc})"
    if completed.returncode != 0:
        return f"unknown (git rev-parse exited {completed.returncode}: {completed.stderr.strip()})"
    return completed.stdout.strip() or "unknown (git rev-parse printed nothing)"


def _onc_client_version() -> str:
    """Version of the `onc` package that made the requests, or a stated reason it is unknown."""
    import importlib.metadata as metadata
    try:
        return metadata.version("onc")
    except metadata.PackageNotFoundError:
        return "unknown (the `onc` distribution is not installed in this environment)"


def build_provenance(*, start_utc, end_utc, location_codes, deployments, rows) -> dict:
    """Provenance sidecar for the calendar: every fact needed to re-derive it.

    Times are tz-aware UTC isoformat strings. `source` is always `"listing"`
    (decision D3): ONC said a file exists, which is not the same claim as a
    successful download -- A4's pull refines this artefact and the pull wins.
    """
    available = sum(1 for row in rows if row[2])
    return {
        "script": str(pathlib.Path(__file__).resolve().relative_to(paths.REPO_ROOT)),
        "git_commit": _git_commit(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device_code": config.DEVICE_CODE,
        "device_id": config.DEVICE_ID,
        "location_codes": list(location_codes),
        "product_extension": config.PRODUCT_EXTENSION,
        "archive_extension": config.ARCHIVE_EXTENSION,
        "date_start_utc": parse_utc(start_utc).isoformat(),
        "date_end_utc": parse_utc(end_utc).isoformat(),
        "bin_seconds": config.BIN_SECONDS,
        "season_months_utc": list(config.SEASON_MONTHS_UTC),
        "source": "listing",  # D3 -- see the docstring; NOT proof of a download
        "onc_version": _onc_client_version(),
        # Beyond the required facts: cheap summary numbers, so a reader can tell
        # at a glance whether the file they hold is the one described here.
        "n_rows": len(rows),
        "n_available_bins": available,
        "n_deployments": len(deployments),
    }


def write_deployments_csv(deployments, path) -> None:
    """Write `get_deployments()` tuples to `path`. Times are tz-aware UTC isoformat."""
    path = pathlib.Path(path)
    if not deployments:
        raise ValueError(
            f"refusing to write an empty deployments table to {path}; 'no deployment "
            "metadata' and 'metadata not fetched' are different claims (invariant 9)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(onc_client.DEPLOYMENTS_CSV_HEADER))
        for dep_id, begin_utc, end_utc in deployments:
            writer.writerow([dep_id, begin_utc.isoformat(), end_utc.isoformat()])
    print(f"write_deployments_csv: {len(deployments)} deployment(s) -> {path}")


# `build_uptime_calendar` PRINTS its empty-chunk count rather than returning it.
# Rather than re-running the listing to recover the number, its output is captured,
# echoed verbatim (nothing is hidden) and this pattern reads the count back out.
# If ONC's wording or the library's summary line ever changes, the count is
# reported as unavailable instead of being invented.
_EMPTY_CHUNKS_PATTERN = re.compile(r"empty chunks:\s*(\d+)")


def _build_calendar_capturing_empty_chunks(client, start_utc, end_utc):
    """`build_uptime_calendar(...)` -> (rows, empty_chunks_or_None), output echoed verbatim."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        rows = onc_client.build_uptime_calendar(client, start_utc, end_utc)
    captured = buffer.getvalue()
    sys.stdout.write(captured)
    match = _EMPTY_CHUNKS_PATTERN.search(captured)
    return rows, (int(match.group(1)) if match else None)


def main(argv=None) -> int:
    """Build and write the calendar. Returns 0 on success, non-zero on error."""
    args = build_parser().parse_args(argv)
    start_utc = parse_utc(args.start)
    end_utc = parse_utc(args.end)
    if end_utc <= start_utc:
        print(
            f"ERROR: --start {start_utc.isoformat()} is not before --end {end_utc.isoformat()}",
            file=sys.stderr,
        )
        return 2

    # Refuse a forbidden out-dir BEFORE any work: no mkdir, no request (invariant 2).
    try:
        out_dir = resolve_out_dir(args.out_dir)
        targets = artifact_paths(out_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"build_uptime_calendar.py: {config.DEVICE_CODE} "
        f"{start_utc.isoformat()} -> {end_utc.isoformat()} (UTC, half-open), "
        f"{config.BIN_SECONDS} s bins, season months {list(config.SEASON_MONTHS_UTC)} (UTC); "
        f"out-dir {out_dir}"
    )
    for key, path in targets.items():
        print(f"  would write {key}: {path}")

    if args.dry_run:
        print("--dry-run: no ONC request made, no credential read, nothing written.")
        return 0

    client = get_onc_client()
    rows, empty_chunks = _build_calendar_capturing_empty_chunks(client, start_utc, end_utc)
    codes = onc_client.discover_folger_locations(client)
    deployments = onc_client.get_deployments(client)

    bins = [(row[0], row[1]) for row in rows]
    available = [row[2] for row in rows]
    gaps = onc_client.summarise_gaps(bins, available)

    paths.ensure_dir(out_dir)
    onc_client.write_uptime_calendar_csv(rows, targets["uptime_csv"])
    write_deployments_csv(deployments, targets["deployments_csv"])
    provenance = build_provenance(
        start_utc=start_utc, end_utc=end_utc, location_codes=codes,
        deployments=deployments, rows=rows,
    )
    with open(targets["provenance_json"], "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"provenance -> {targets['provenance_json']}")

    n_available = sum(1 for flag in available if flag)
    print(
        "SUMMARY: "
        f"{len(rows)} row(s) written; "
        f"available fraction {n_available / len(rows):.4f} ({n_available}/{len(rows)}); "
        f"{len(gaps)} gap run(s); "
        f"empty chunks: {empty_chunks if empty_chunks is not None else 'unavailable'}; "
        f"location codes discovered: {list(codes)}; "
        f"{len(deployments)} deployment(s). "
        "source=listing -- ONC's belief a file exists; A4's pull refines this and wins."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
