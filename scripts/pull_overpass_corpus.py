#!/usr/bin/env python3
"""B3-C: the bulk `.fft.gz` pull driver and absent-file log.

Runnable entry point. It DEFINES NOTHING SHARED (CLAUDE.md invariant 6): the
overpass window comes from `boatphone.acquire.overpass_window_utc` (B3-A), the
listing call from `boatphone.onc_client.list_fft_files` / discovery (A1), and
the download from `boatphone.onc_client.download_archive_file` (B3-B). This
file only orders the dates, wires the three together, and writes the manifest.

Scope (`docs/plans/acoustics_plan_v2.md` SS5 "B3 -- Bulk acquisition"):
every in-season (`config.SEASON_MONTHS_UTC`, May-Sep) LOCAL calendar date,
year-blocks ordered **2025, 2024, ..., 2020 (descending)** -- a risks-table
mitigation: if throughput drops below 1 file/sec and the run is stopped
early, the corpus stays useful at whatever depth it reached, newest season
first.

**The overpass window is recomputed INSIDE the date loop, every date**, via
`boatphone.acquire.overpass_window_utc`. It is never hoisted out and reused:
the true UTC window shifts by an hour across the March/November DST
transition (decision 0002), so a value computed once and cached would be
silently wrong on one side of that transition.

**Resume is the unconditional default.** There is no `--resume` flag: a bare
re-run of this script asks `download_archive_file` for the same files, which
recognises a complete file already at its final path and returns a "cached"
record without a single network byte requested. Nothing here needs its own
resume bookkeeping -- B3-B's cache-hit guarantee IS the resume behaviour.

Manifest content (`requested == present + absent` EXACTLY, no third bucket):
a real 404 ("absent"), a decision-0007 measured zero, and a decision-0016
fully-empty overpass window all land in `absent_log`, distinguished only by
`reason`. `requested` therefore counts retrieval units -- listed filenames plus
empty windows -- so the equality stays a statement about coverage (0016). The manifest is never written
without its provenance sidecar landing first -- if the sidecar write fails,
`run()` raises and neither file is left half-written and mistaken for
complete.

Usage:

    python3 scripts/pull_overpass_corpus.py [--dest-dir DIR] [--manifest-dir DIR]
                                             [--start-year YYYY] [--end-year YYYY]
                                             [--dry-run]

`--dry-run` prints the plan (dates, window, out-dirs) with no network request
and no credential read, and does nothing else. Without `--dry-run` this reads
an ONC credential (`boatphone.credentials.get_onc_client`) and performs the
real pull over `RequestsArchiveTransport`
(`boatphone.onc_client.RequestsArchiveTransport`) -- a deliberate,
human-triggered operational step: `python3 scripts/pull_overpass_corpus.py` with real
`--dest-dir`/`--manifest-dir` values.
"""

from __future__ import annotations

import argparse
import calendar
import json
import pathlib
import subprocess
import sys
import time
from datetime import date, datetime, timezone

# scripts/ is a directory of entry points, not a package, so make the repo root
# importable when this file is run directly. Path manipulation only -- no
# network, no directory creation, no parsing at import (side-effect free, per
# the B3-C ASSUMED INTERFACE comment in scripts/checks.py).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from boatphone import config, paths
from boatphone import acquire
from boatphone import onc_client
from boatphone.credentials import get_onc_client

# Basenames of the two output files under `manifest_dir`. Different names, so
# check_b3c_5 can sabotage the provenance path alone without guessing it.
MANIFEST_BASENAME = "pull_overpass_corpus.manifest.json"
PROVENANCE_BASENAME = "pull_overpass_corpus.provenance.json"

# Every invocation writes its OWN copy of both files under
# `<manifest_dir>/runs/<run id>/`, and ALSO refreshes a stable "latest" copy at
# `<manifest_dir>/<basename>` for consumers that just want the most recent run.
# The fixed basenames alone would overwrite: the plan explicitly contemplates a
# staged pull plus segment F's second pass, so a second run into the same
# manifest_dir is the EXPECTED case, and the first run's absent-file log --
# which is a B3 deliverable, not a byproduct -- must survive it.
#
# The run id is a UTC timestamp WITH MICROSECONDS, and directory creation is
# detect-and-suffix (`mkdir` without exist_ok, retry with `-1`, `-2`, ...), so
# uniqueness is guaranteed rather than probabilistic. A whole-second stamp
# collides when two runs start inside the same second, which is exactly what a
# scripted staged pull does.
RUNS_SUBDIR = "runs"

# Default year span for iter_overpass_dates(): descending, newest season first,
# regardless of which bound is named start/end. The VALUES live in
# boatphone.config (CLAUDE.md invariant 6 -- one definition of the corpus span,
# in the library, not here); these names are local aliases only, so nothing
# shared is defined in scripts/.
_DEFAULT_START_YEAR = config.CORPUS_PULL_START_YEAR
_DEFAULT_END_YEAR = config.CORPUS_PULL_END_YEAR

__all__ = [
    "MANIFEST_BASENAME", "PROVENANCE_BASENAME", "RUNS_SUBDIR",
    "iter_overpass_dates", "build_parser", "run", "main",
]


def iter_overpass_dates(start_year: int = _DEFAULT_START_YEAR,
                         end_year: int = _DEFAULT_END_YEAR):
    """Every in-season LOCAL calendar date, newest season first.

    Yields `datetime.date` objects -- LOCAL calendar dates in
    `config.PLANET_OVERPASS_TZ_NAME` (America/Vancouver), the same convention
    `boatphone.acquire.overpass_window_utc` expects as input. Year-blocks run
    descending from `max(start_year, end_year)` down to `min(start_year,
    end_year)` (risks-table mitigation: an interrupted run leaves the newest
    season complete). Within a year, only months in `config.SEASON_MONTHS_UTC`
    (May-Sep) are yielded, in calendar order.

    **Frame note -- a LOCAL filter applied with a UTC-defined constant.**
    `config.SEASON_MONTHS_UTC` is documented there as evaluated on the **UTC**
    month of a bin start. Here it filters **local** (America/Vancouver) calendar
    dates. That is sound for THIS window and only for this window: the overpass
    window is 09:15-11:45 local, which maps to 16:15-18:45 UTC (PDT) or
    17:15-19:45 UTC (PST) -- always the same calendar day in UTC as locally, so
    a local month edge and the UTC month edge of every window it selects
    coincide. The set of dates is identical either way.

    It would BREAK if the overpass window moved past ~16:00 local (17:00 with
    PST's UTC-8), because the window would then cross midnight UTC and a local
    date at a month edge would select a window in the neighbouring UTC month.
    If `config.PLANET_OVERPASS_WINDOW_END_LOCAL` ever moves that late, this
    filter must be re-derived, not just re-run. The constant is NOT relabelled:
    it means what config says it means; this function documents why applying it
    in a different frame is safe here.
    """
    lo, hi = sorted((start_year, end_year))
    for year in range(hi, lo - 1, -1):
        for month in config.SEASON_MONTHS_UTC:
            n_days = calendar.monthrange(year, month)[1]
            for day in range(1, n_days + 1):
                yield date(year, month, day)


def build_parser() -> argparse.ArgumentParser:
    """The CLI. No `--resume`/`--continue` flag anywhere: resume is unconditional."""
    parser = argparse.ArgumentParser(
        prog="pull_overpass_corpus.py",
        description=(
            "Bulk-pull the .fft.gz files covering every PlanetScope overpass window "
            "in-season, 2025 backwards to 2020. A bare re-run resumes unconditionally: "
            "there is no --resume flag to forget."
        ),
    )
    parser.add_argument(
        "--dest-dir", dest="dest_dir", default=paths.ONC_OVERPASS_CORPUS_DIR,
        help=f"landing zone for downloaded .{config.ARCHIVE_EXTENSION} files "
             f"(default: {paths.ONC_OVERPASS_CORPUS_DIR} -- a subdirectory of "
             f"{paths.ONC_RAW_DIR} that marks this corpus as OVERPASS-WINDOW-ONLY on "
             "disk, decision 0020, so a consumer globbing the landing zone can tell it "
             "from a whole-day pull without opening the manifest). Passed straight "
             "through to boatphone.onc_client.download_archive_file. Resolve it for "
             "reading with boatphone.acquire.resolve_corpus_files -- the ONE glob.",
    )
    parser.add_argument(
        "--manifest-dir", dest="manifest_dir", default=paths.DERIVED_DIR,
        help=f"directory to write the manifest + provenance sidecar into "
             f"(default: {paths.DERIVED_DIR})",
    )
    parser.add_argument(
        "--start-year", dest="start_year", type=int, default=_DEFAULT_START_YEAR,
        help=f"earliest year in scope (default: {_DEFAULT_START_YEAR})",
    )
    parser.add_argument(
        "--end-year", dest="end_year", type=int, default=_DEFAULT_END_YEAR,
        help=f"latest year in scope, pulled first (default: {_DEFAULT_END_YEAR})",
    )
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="print the date range and window, then exit: no network request, "
             "no credential read, nothing written",
    )
    return parser


def _git(*args) -> tuple:
    """Run one git command in the repo. Returns `(ok, text_or_reason)`.

    Never raises: provenance must record a stated reason it does not know
    something rather than fail the run that produced the data.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(paths.REPO_ROOT), *args],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"git unavailable: {exc}"
    if completed.returncode != 0:
        return False, (
            f"git {' '.join(args)} exited {completed.returncode}: {completed.stderr.strip()}"
        )
    return True, completed.stdout


def _git_commit() -> str:
    """The commit this run was made from, WITH a dirty-tree marker if it is dirty.

    A bare `rev-parse HEAD` asserts an identity git cannot vouch for: if the
    working tree had uncommitted changes, the code that produced this manifest
    is NOT the code at that commit, and a future reader who checks the commit
    out gets something else. `git status --porcelain` is consulted (tracked
    files only -- untracked files do not change what ran) and the answer is
    stated in the string itself, so the marker cannot be dropped by a consumer
    that reads only `git_commit`.

    Returns a stated non-empty reason when the commit or the dirty state is
    unknown; "unknown" is never silently rendered as clean.
    """
    ok, out = _git("rev-parse", "HEAD")
    if not ok:
        return f"unknown ({out})"
    commit = out.strip()
    if not commit:
        return "unknown (git rev-parse printed nothing)"
    ok, status = _git("status", "--porcelain", "--untracked-files=no")
    if not ok:
        return f"{commit} (dirty state UNKNOWN: {status})"
    n_dirty = len([line for line in status.splitlines() if line.strip()])
    if n_dirty:
        return (
            f"{commit}-dirty ({n_dirty} tracked file(s) modified in the working tree at "
            "run time; the code that produced this manifest is NOT exactly this commit)"
        )
    return f"{commit} (clean working tree)"


# The reason string recorded in absent_log for each of the FOUR ways a listed
# file (or a whole window) can end up NOT downloaded -- see _ABSENT_DEFINITION,
# which states the same four to a reader of the provenance sidecar. Decision
# 0007: all of them are "absent", never a third bucket -- these strings are what
# makes them distinguishable in the log, and segment D keys on `reason`.
#
# `download_archive_file` returns status "absent" for TWO DIFFERENT HTTP
# answers, so this side must not collapse them: a plain 404, and a 400 carrying
# ONC's Error 96 "file does not exist" (decision 0023, `_FILE_DOES_NOT_EXIST_
# MARKERS` in onc_client). A row that says "ONC returned 404" while carrying
# `http_status: 400` contradicts itself, and the contradiction lands in the
# artefact segment D reads. They are chosen on `record.http_status`, not on
# status alone.
_REASON_ABSENT_404 = "absent: ONC returned 404 for this listed filename"
_REASON_ABSENT_400_NO_SUCH_FILE = (
    "absent (decision 0023): ONC returned HTTP 400 Error 96 -- no archive file of this "
    "name exists. A definitive negative answer, not a 404 and not retried"
)
_REASON_ABSENT_UNSTATED_STATUS = (
    "absent: download_archive_file reported the file absent but recorded no HTTP status, "
    "so WHICH definitive negative ONC gave (404, or 400 Error 96) is UNKNOWN for this row"
)
_REASON_EMPTY_WINDOW = (
    "measured_zero (decision 0016): ONC listed zero files for this whole overpass "
    "window -- the window itself is the measured zero, not any named file"
)
_REASON_MEASURED_ZERO = (
    "measured_zero (decision 0007): ONC states no data can exist for this filename "
    "(no device deployed in the window, or the window has not elapsed)"
)
# A listed file whose download FAILED outright -- an exhausted retry budget, an
# unexpected status, a hash mismatch, a transfer whose length the server never
# stated. It goes in the SAME absent bucket as a 404 (requested == present +
# absent must stay exactly true; a third bucket would break the coverage
# arithmetic check_b3c_3 pins), but its row also carries an `error` field with
# the message, so "ONC has nothing here" and "we could not get what ONC has"
# stay different findings (CLAUDE.md invariant 9). Its presence is what keeps
# ONE bad file from killing a six-season overnight run.
_REASON_DOWNLOAD_ERROR = (
    "error: the file was listed by ONC but could not be downloaded (see this row's "
    "'error' field) -- NOT a statement that ONC has no data for it"
)

# What "absent" MEANS in this manifest, written into provenance verbatim so a
# future reader does not have to reconstruct it from three reason strings. It is
# a statement about the arithmetic (requested == present + absent) as much as
# about any one row.
_ABSENT_DEFINITION = (
    "'absent' = a retrieval unit that is counted in `requested` but produced no file on "
    "disk. requested == present + absent EXACTLY, with no third bucket. A retrieval unit "
    "is one listed filename, OR one whole overpass window that ONC listed no file for at "
    "all (decision 0016; that row carries filename=null). Four kinds land in absent_log, "
    "distinguished ONLY by the row's `reason`: (1) ONC gave a definitive negative for a "
    "listed filename -- either HTTP 404, or HTTP 400 Error 96 'file does not exist' "
    "(decision 0023); those are two different `reason` strings, because a row saying '404' "
    "while carrying http_status 400 would contradict itself; (2) decision 0007 -- ONC states no data CAN exist (no device deployed, or "
    "the window has not elapsed); (3) decision 0016 -- the whole window listed zero files; "
    "(4) the file was listed but the download FAILED (retry budget exhausted, unexpected "
    "status, hash mismatch, unstated length -- decision 0019); those rows carry an `error` "
    "field and are counted in `errors`. Kind (4) is NOT a statement that ONC has no data "
    "(CLAUDE.md invariant 9). 'absent' NEVER means 'the ocean was quiet' -- it is a "
    "statement about retrieval, not about acoustics."
)

# How many dates may be pulled between manifest flushes. The manifest is
# rewritten whole each flush, so 1 would rewrite a growing multi-MB JSON ~900
# times over a full run; the trade bought here is that a hard kill (SIGKILL,
# a lost node) loses at most this many dates' worth of absent-log rows, while
# any ORDINARY abort -- an exception, KeyboardInterrupt -- loses none, because
# run() also persists from a `finally`. Downloaded files themselves are never
# at risk: they are on disk and resume finds them.
_MANIFEST_FLUSH_EVERY_N_DATES = 10


def _unpack_listing(result, *, local_date):
    """Normalise a `listing_fn` return into `(filenames, empty_chunks)`.

    ONE shape is accepted: `(filenames, empty_chunks)`, which is what
    `boatphone.onc_client.list_fft_files` actually returns and what `main()`
    injects. A bare-list branch used to live here for an early fake that
    returned only filenames; every B3-C fake was since widened to the real
    2-tuple (verified by reading each `listing_fn` in `scripts/checks.py`), so
    that branch was unreachable and has been deleted rather than left as a
    second, untested contract for the same call.

    `empty_chunks` may still be `None` from the caller -- an UNKNOWN chunk
    count, kept distinct from a measured 0 (invariant 9) -- but that is the
    tuple's second element, not a separate return shape.

    Anything else raises rather than being iterated -- iterating the 2-tuple as
    if it were a filename list is exactly the bug this function exists to stop
    (it bound `filename` to a list, then to an int, and crashed on the first
    real date).
    """
    if isinstance(result, tuple):
        if len(result) != 2:
            raise TypeError(
                f"listing_fn returned a {len(result)}-tuple for {local_date.isoformat()}; "
                "the only tuple contract is (filenames, empty_chunks)"
            )
        filenames, empty_chunks = result
        return list(filenames), empty_chunks
    raise TypeError(
        f"listing_fn returned {type(result).__name__} for {local_date.isoformat()}; "
        "the only accepted contract is a (filenames, empty_chunks) tuple, as returned "
        "by boatphone.onc_client.list_fft_files"
    )


def _file_span(filename):
    """`{"file_start_utc": ..., "file_end_utc": ...}` for a NAMED archive file.

    Goes through `onc_client.parse_file_coverage`, which its own docstring calls
    "the ONE place a filename becomes a time". Recording it at write time means
    a consumer READS the span, it does not recompute one; two paths to the same
    fact drift (CLAUDE.md invariant 3).

    **These are FILE SPANS, and deliberately NOT called `bin_*`.** In this
    codebase `bin_start_utc`/`bin_end_utc` mean a cell of the fixed-width
    `config.BIN_SECONDS` grid whose edges are integer multiples of
    `BIN_SECONDS` since the UTC epoch (`config.py` L12-14; the real grid cells
    come out of `onc_client.season_bins_utc` under exactly those two names,
    `onc_client.py` L1158-1161). A file start is NOT on that grid --
    `parse_file_coverage` documents observed file-start seconds of ":36",
    ":04", with 1-2 s jitter between consecutive files. Using one name for both
    would make an arbitrary file span look equality-joinable against a real
    grid cell.

    Consequently **segment D's join onto the 5-minute bin grid must be an
    INTERVAL-OVERLAP join** -- does `[file_start_utc, file_end_utc)` intersect
    `[bin_start_utc, bin_end_utc)`? -- and never an equality join on the `*_utc`
    fields. An equality join would match almost nothing and would read as "low
    coverage" rather than as the naming bug it is.

    Values are ISO-8601 strings with an explicit UTC offset, so the manifest
    states its own time base rather than relying on a reader's assumption
    (decision 0002).

    A `FilenameParseError` here PROPAGATES. It means ONC listed a name this
    project cannot read as one of its own device's files -- a contract problem,
    not a data problem, and papering over it would put a row with no time in a
    log whose whole purpose is time (invariant 5).
    """
    file_start_utc, file_end_utc = onc_client.parse_file_coverage(filename)
    return {
        "file_start_utc": file_start_utc.isoformat(),
        "file_end_utc": file_end_utc.isoformat(),
    }


def run(*, dates=None, dest_dir, manifest_dir, listing_fn, transport,
        client=None, sleep=time.sleep) -> dict:
    """Pull every listed file for `dates` (or `iter_overpass_dates()`), write a
    manifest + provenance sidecar under `manifest_dir`, return the manifest dict.

    `dest_dir`: B3-B's landing zone, passed straight through to
    `download_archive_file` (never created independently of it).
    `manifest_dir`: written to directly by this function via `paths.ensure_dir`.
    `listing_fn(client, location_codes, start_utc, end_utc)`: A1's listing call,
    injectable so no real network happens under test. It may return either
    `(filenames, empty_chunks)` -- what `onc_client.list_fft_files` really
    returns -- or a bare list of filenames; see `_unpack_listing`. An
    `onc_client.EmptyListingError` from it is caught PER DATE and recorded as a
    measured zero (decision 0016), never allowed to abort a multi-season run.
    `transport`: B3-B's injectable network layer, passed straight through to
    `download_archive_file`.
    `client`: an ONC client, or `None`. When `None`, location discovery is
    skipped and `listing_fn` is called with `location_codes=None` -- the shape
    these checks use to drive `listing_fn` without a real client. A real
    invocation passes a real `client`, from which Folger's location codes are
    discovered (never hardcoded, per `onc_client.discover_folger_locations`).
    `sleep`: passed straight through to `download_archive_file`'s backoff.

    Resume is unconditional: `download_archive_file` recognises a complete file
    already at its final path and returns a "cached" record without touching
    `transport`, so re-running this with identical arguments re-requests zero
    bytes for anything already finished.

    `requested == present + absent` EXACTLY -- a real 404, a decision-0007
    measured zero, and a decision-0016 fully-empty overpass window all land in
    the SAME absent bucket, distinguished only by `absent_log[i]["reason"]`.
    `requested` counts RETRIEVAL UNITS: one per listed filename, plus one per
    window ONC listed no file for at all (whose row carries `"filename": None`).
    Without that, an empty morning would appear in neither bucket and the
    coverage arithmetic would quietly stop describing coverage.

    Writes the provenance sidecar BEFORE the manifest; if the sidecar write
    fails, this raises and the manifest is never written (no manifest without
    provenance).

    **A failure does not destroy the record.** Two containments, because an
    overnight multi-season run fails in the middle, not at the start:

    * a `DownloadError` on one listed file is CONTAINED -- it becomes an
      `absent_log` row carrying `reason` `_REASON_DOWNLOAD_ERROR` and the
      message under `error`, counted in `manifest["errors"]`, and the run goes
      on. It stays inside the absent bucket so `requested == present + absent`
      remains exactly true, and it is never confused with ONC saying there is
      no data (invariant 9). Anything else -- a local `OSError`, a broken
      contract, `KeyboardInterrupt` -- still propagates;
    * the manifest and provenance are persisted every
      `_MANIFEST_FLUSH_EVERY_N_DATES` dates AND from a `finally`, so an aborted
      run leaves a truthful PARTIAL manifest rather than nothing at all. Such a
      manifest carries `"run_complete": false` and `"n_dates_completed"`, so a
      partial record can never be read as a complete one.
    """
    date_list = list(dates) if dates is not None else list(iter_overpass_dates())
    dest_dir = pathlib.Path(dest_dir)
    manifest_dir = pathlib.Path(manifest_dir)

    location_codes = onc_client.discover_folger_locations(client) if client is not None else None

    requested = 0
    present = 0
    errors = 0
    files = []
    absent_log = []
    empty_windows = 0
    empty_chunks_total = 0
    empty_chunks_unreported = 0
    dates_done = 0
    run_complete = False

    def snapshot():
        """The manifest dict as of RIGHT NOW -- valid mid-run, not only at the end."""
        return {
            "requested": requested,
            "present": present,
            "absent": len(absent_log),
            "errors": errors,
            "files": files,
            "absent_log": absent_log,
            "n_dates": len(date_list),
            # Whether the date loop ran to completion. A manifest written from
            # the `finally` after a failure is TRUTHFUL but PARTIAL, and a
            # consumer that cannot tell the two apart would read an aborted
            # run's coverage as the whole answer (invariant 9).
            "run_complete": run_complete,
            "n_dates_completed": dates_done,
            # Decision-0008 measured-zero signal, carried through from
            # list_fft_files instead of discarded at the call site. `*_unreported`
            # counts dates whose listing_fn did not report a chunk count at all --
            # an unknown, kept distinct from a measured 0 (invariant 9).
            "empty_windows": empty_windows,
            "empty_chunks_total": empty_chunks_total,
            "empty_chunks_unreported_dates": empty_chunks_unreported,
            "sampling_conditionality": config.PLANET_SAMPLING_CONDITIONALITY_STATEMENT,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }

    run_dir_holder = {}

    def run_output_dir():
        """This invocation's own output directory, created once, never reused.

        `<manifest_dir>/runs/<UTC stamp with microseconds>`, with a
        detect-and-suffix retry so two runs starting inside the same
        microsecond still get two directories. Uniqueness is GUARANTEED here,
        not probabilistic: `mkdir` without `exist_ok` is the check, and the
        loop is the resolution. A whole-second stamp would collide on a
        scripted staged pull, which is precisely the case this exists for.
        """
        if "path" not in run_dir_holder:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            base = paths.ensure_dir(manifest_dir / RUNS_SUBDIR)
            candidate = base / stamp
            suffix = 0
            while True:
                try:
                    candidate.mkdir()
                except FileExistsError:
                    suffix += 1
                    candidate = base / f"{stamp}-{suffix}"
                else:
                    break
            run_dir_holder["path"] = candidate
        return run_dir_holder["path"]

    def write_json(path, payload):
        """Write one JSON file. Not caught anywhere: a failed write must surface."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def persist(manifest):
        """Write provenance THEN manifest. No manifest without provenance.

        Called on every flush and from the `finally`, so an aborted run still
        leaves a truthful partial record of what was requested and what ONC
        said was absent -- the B3 deliverable that a six-hour failure used to
        destroy for the entire run. Both provenance writes are FIRST and
        neither is caught: if either fails, this raises and no manifest lands
        (check_b3c_5).

        Each file is written TWICE: once under this invocation's own
        `runs/<run id>/` directory, which no later run can overwrite, and once
        at the stable `<manifest_dir>/<basename>` path, which is deliberately
        overwritten so a consumer has a fixed "latest run" location to open.
        The durable record is the run directory; the fixed path is a pointer.
        """
        paths.ensure_dir(manifest_dir)
        run_dir = run_output_dir()

        # The ONC deployment that served the bytes. Read off the transport (its
        # `base_url` property is public precisely so provenance can state it);
        # a transport that does not expose one -- a test double -- yields a
        # stated non-empty UNKNOWN rather than a plausible-looking default, so
        # nobody later reads a guess as a measurement (invariant 9).
        # NEVER a token: `base_url` carries none and no credential is read here.
        base_url = getattr(transport, "base_url", None) or getattr(client, "baseUrl", None)
        if base_url:
            endpoint = f"{base_url}api/archivefile/download"
        else:
            endpoint = (
                f"unknown (transport {type(transport).__name__} exposes no base_url and no "
                "ONC client was supplied; this manifest cannot state which ONC deployment "
                "served the bytes)"
            )

        provenance = {
            "script": str(pathlib.Path(__file__).resolve().relative_to(paths.REPO_ROOT)),
            "git_commit": _git_commit(),
            "generated_utc": manifest["generated_utc"],
            "device_code": config.DEVICE_CODE,
            # Open item 1: WHICH ONC deployment served these bytes. Without it
            # a future reader cannot tell a production pull from a staging one.
            "onc_endpoint": endpoint,
            "onc_base_url": base_url or endpoint,
            # Where the bytes landed. The default is the overpass-window-only
            # subdirectory (decision 0020); recording the actual value means a
            # --dest-dir override is not invisible.
            "dest_dir": str(pathlib.Path(dest_dir).resolve()),
            "manifest_dir": str(manifest_dir.resolve()),
            "run_output_dir": str(run_dir.resolve()),
            # What "absent" means, verbatim, next to the counts it governs.
            "absent_definition": _ABSENT_DEFINITION,
            "location_codes": list(location_codes) if location_codes is not None else None,
            "archive_extension": config.ARCHIVE_EXTENSION,
            "overpass_window_start_local": config.PLANET_OVERPASS_WINDOW_START_LOCAL.isoformat(),
            "overpass_window_end_local": config.PLANET_OVERPASS_WINDOW_END_LOCAL.isoformat(),
            "overpass_tz_name": config.PLANET_OVERPASS_TZ_NAME,
            # Scope of the corpus this run belongs to: which months, which
            # years. A partial manifest cannot state its own coverage without
            # them, and both are config constants, never restated here
            # (invariant 6).
            "season_months_utc": list(config.SEASON_MONTHS_UTC),
            # Invariant 3 -- the frame label must be TRUE at every boundary, and
            # this manifest is a cross-team boundary. The constant's name says
            # UTC and config defines it as evaluated on the UTC month of a bin
            # start; THIS run applied it to LOCAL (America/Vancouver) calendar
            # dates. For this window the two select an identical date set, but a
            # consumer must be told that from the record, not from a docstring
            # in a script they will not open (iter_overpass_dates has the full
            # derivation and the condition under which it stops holding).
            "season_months_frame_note": (
                "season_months_utc is named for the UTC month of a bin start "
                "(config.SEASON_MONTHS_UTC), but this run APPLIED it to LOCAL "
                f"{config.PLANET_OVERPASS_TZ_NAME} calendar dates. The two select the SAME "
                "set of dates for this overpass window only, because the window "
                f"({config.PLANET_OVERPASS_WINDOW_START_LOCAL}-"
                f"{config.PLANET_OVERPASS_WINDOW_END_LOCAL} local) never crosses midnight "
                "UTC, so every local month edge coincides with the UTC month edge of the "
                "windows it selects. If the window end ever moves past ~16:00 local the "
                "filter must be re-derived; see iter_overpass_dates in "
                "scripts/pull_overpass_corpus.py"
            ),
            "corpus_pull_year_span": (
                f"{config.CORPUS_PULL_START_YEAR}..{config.CORPUS_PULL_END_YEAR} "
                "(config.CORPUS_PULL_START_YEAR..CORPUS_PULL_END_YEAR; the full corpus "
                "scope, which this run's own date list may be a subset of -- see "
                "first_date/last_date/n_dates_requested)"
            ),
            "start_year": config.CORPUS_PULL_START_YEAR,
            "end_year": config.CORPUS_PULL_END_YEAR,
            "n_dates_requested": len(date_list),
            # NOT an interval. The date list is deliberately DESCENDING (newest
            # season first, so an interrupted run leaves the newest season
            # complete), so "first"/"last" would read as an inverted interval:
            # first_date 2025-05-01 with last_date 2020-09-30. These are named
            # for their position in the PULL ORDER instead, and the order is
            # stated in the record so a reader never has to infer it. The
            # earliest/latest calendar bounds are given separately.
            "date_order": (
                "DESCENDING by year (newest season pulled first); within a year, "
                "calendar order. See pull_order_first_date/pull_order_last_date"
            ),
            "pull_order_first_date": date_list[0].isoformat() if date_list else None,
            "pull_order_last_date": date_list[-1].isoformat() if date_list else None,
            "earliest_date": min(date_list).isoformat() if date_list else None,
            "latest_date": max(date_list).isoformat() if date_list else None,
            "requested": manifest["requested"],
            "present": manifest["present"],
            "absent": manifest["absent"],
            "errors": manifest["errors"],
            "run_complete": manifest["run_complete"],
            "n_dates_completed": manifest["n_dates_completed"],
            "empty_windows": manifest["empty_windows"],
            "empty_chunks_total": manifest["empty_chunks_total"],
            "empty_chunks_unreported_dates": manifest["empty_chunks_unreported_dates"],
            "sampling_conditionality": config.PLANET_SAMPLING_CONDITIONALITY_STATEMENT,
            # Decision 0018, placed HERE rather than in a doc a reader of this
            # file will not open, because the per-file `sha256` values in the
            # manifest look like an integrity guarantee and are a weaker one.
            "sha256_checksum_caveat": (
                "Decision 0018: ONC serves NO server-side checksum for archive downloads "
                "(no ETag, no Content-MD5; confirmed by live probe). Every `sha256` in the "
                "manifest's `files` rows is a LOCAL hash computed over the bytes this "
                "machine received, stored in a sidecar and re-verified on cache hits. It "
                "catches LOCAL corruption -- a bit-flip, a truncated write, a partial file "
                "from a crashed run mistaken for complete. It does NOT and cannot catch "
                "SERVER-side truncation: a short body would be hashed as-is and would match "
                "itself forever. The independent guard against that is decision 0019 -- "
                "download_archive_file REFUSES to promote a transfer whose total length the "
                "server never stated -- whose outcome is the `total_size_known` field on "
                "each file row. ONC did not vouch for these bytes; only this machine did."
            ),
        }
        # Provenance first, BOTH copies, before any manifest byte is written:
        # "no manifest without provenance" has to hold for the stable path too,
        # or a failed sidecar write could still leave a stale-provenance
        # manifest pair at the location consumers actually open.
        run_provenance_path = write_json(run_dir / PROVENANCE_BASENAME, provenance)
        provenance_path = write_json(manifest_dir / PROVENANCE_BASENAME, provenance)
        run_manifest_path = write_json(run_dir / MANIFEST_BASENAME, manifest)
        manifest_path = write_json(manifest_dir / MANIFEST_BASENAME, manifest)
        return (manifest_path, provenance_path, run_manifest_path, run_provenance_path)

    try:
        for local_date in date_list:
            start_utc, end_utc = acquire.overpass_window_utc(local_date)
            try:
                listing = listing_fn(client, location_codes, start_utc, end_utc)
            except onc_client.EmptyListingError as exc:
                # Decision 0016 (amending 0008): 0008 raises on an empty listing
                # because at YEAR scale nothing-returned is evidence of a broken
                # request. This span is 2.5 h, where an empty listing is the
                # ordinary case -- one outage morning -- and letting it kill a
                # six-season overnight run trades a real corpus for a false alarm.
                # `main()` passes allow_empty=True so this branch is not the normal
                # path; it stays here so a listing_fn that has NOT opted in still
                # yields the measured-zero row instead of aborting the run.
                filenames, empty_chunks = [], None
                print(
                    f"pull_overpass_corpus.run: {local_date.isoformat()} listed zero file(s) "
                    f"for {start_utc.isoformat()} -> {end_utc.isoformat()}: recorded as a "
                    f"measured zero (decision 0016). Listing said: {exc}"
                )
            else:
                filenames, empty_chunks = _unpack_listing(listing, local_date=local_date)

            if empty_chunks is None:
                empty_chunks_unreported += 1
            else:
                empty_chunks_total += int(empty_chunks)

            if not filenames:
                # The window is a retrieval unit that ONC answered with "no file".
                # It is counted in `requested` so `requested == present + absent`
                # stays a true statement about COVERAGE and not merely about named
                # files -- otherwise a fully-empty morning appears in neither
                # bucket and the run looks like it never asked.
                empty_windows += 1
                requested += 1
                absent_log.append({
                    "filename": None,
                    "reason": _REASON_EMPTY_WINDOW,
                    # EXPLICIT nulls, not omitted keys. This row has no
                    # filename, so `parse_file_coverage` -- the only sanctioned
                    # way to turn a name into a time -- has nothing to work
                    # from, and inventing a file span from the window edges
                    # would be exactly the second derivation path the named
                    # rows avoid. Writing the keys as null says "not derivable
                    # here" out loud; omitting them would leave a consumer
                    # unable to tell that from an implementation slip
                    # (invariant 9). The window edges below still bound this
                    # measured zero: it is the WINDOW that is the zero, not any
                    # named file (decision 0016).
                    "file_start_utc": None,
                    "file_end_utc": None,
                    "local_date": local_date.isoformat(),
                    "window_start_utc": start_utc.isoformat(),
                    "window_end_utc": end_utc.isoformat(),
                    "http_status": None,
                    "attempts": 0,
                    "empty_chunks": empty_chunks,
                })
            else:
                for filename in filenames:
                    requested += 1
                    # PER-FILE CONTAINMENT. A DownloadError is one file's
                    # problem: an exhausted retry budget, an unexpected status,
                    # a hash mismatch, a body of unstated length. Letting it out
                    # of this loop aborted an entire multi-season run on one bad
                    # file. It is recorded, not swallowed -- the row lands in
                    # absent_log with its own reason and the message, and
                    # `errors` counts it separately from a real 404 so "ONC has
                    # nothing" never gets confused with "we failed to fetch it"
                    # (invariant 9). Narrow and typed: only DownloadError. An
                    # OSError on the local disk, a bug here, or a
                    # KeyboardInterrupt still propagates -- and now leaves a
                    # truthful partial manifest behind via the `finally`.
                    record = None
                    download_error = None
                    try:
                        record = onc_client.download_archive_file(
                            client=client, filename=filename, dest_dir=dest_dir,
                            transport=transport, sleep=sleep,
                        )
                    except onc_client.DownloadError as exc:
                        download_error = str(exc)
                    if download_error is not None:
                        errors += 1
                        print(
                            f"pull_overpass_corpus.run: {filename} FAILED to download "
                            f"({download_error}); recorded as an error row and the run "
                            "continues"
                        )
                        absent_log.append({
                            "filename": filename,
                            "reason": _REASON_DOWNLOAD_ERROR,
                            "error": download_error,
                            # See _file_span. An error row is still a NAMED
                            # file with a knowable span, and segment D must be
                            # able to overlap it onto the grid to record that
                            # those bins are unresolved rather than empty.
                            **_file_span(filename),
                            "local_date": local_date.isoformat(),
                            "window_start_utc": start_utc.isoformat(),
                            "window_end_utc": end_utc.isoformat(),
                            "http_status": None,
                            "attempts": None,
                        })
                        continue
                    if record.status in ("downloaded", "cached"):
                        present += 1
                        files.append({
                            "filename": record.filename,
                            "status": record.status,
                            "local_date": local_date.isoformat(),
                            "window_start_utc": start_utc.isoformat(),
                            "window_end_utc": end_utc.isoformat(),
                            # The file's own time span from
                            # parse_file_coverage -- the ONE place a filename
                            # becomes a time. Segment D needs it to flag the
                            # bins it OVERLAPS as measured (decision 0020)
                            # without re-deriving the span a second way. It is
                            # a file span, not a bin edge: see _file_span.
                            **_file_span(record.filename),
                            "bytes_downloaded": record.bytes_downloaded,
                            "sha256": record.sha256,
                            # Whether the server stated the file's total length,
                            # so a consumer can tell a length-checked file from
                            # one only its local hash vouches for (B3-B refuses
                            # to promote an unknown-length transfer, so this is
                            # True on every current row -- it is carried anyway
                            # so the guarantee is visible in the artefact rather
                            # than only in a docstring).
                            "total_size_known": getattr(record, "total_size_known", None),
                            "path": str(record.path) if record.path is not None else None,
                            # The ON-DISK basename, a SEPARATE field from the
                            # wire `filename` above. Decision 0024 compresses on
                            # write, so a wire name `X.fft` lands as `X.fft.gz`
                            # and the two names differ for every compressed row.
                            # A consumer joining these rows to
                            # `boatphone.acquire.resolve_corpus_files()` BY NAME
                            # must join on THIS field; joining on `filename`
                            # matches only the plain-`.fft` probe files and
                            # drops the rest, which reads as low coverage rather
                            # than as the naming defect it is (invariant 9).
                            # Taken from the record's own path -- the one place
                            # the local name is decided (onc_client's
                            # compress-on-write block) -- never re-derived by
                            # appending a suffix here.
                            "disk_basename": (
                                pathlib.Path(record.path).name
                                if record.path is not None else None
                            ),
                            "http_status": record.http_status,
                            "attempts": record.attempts,
                        })
                    elif record.status in ("absent", "measured_zero"):
                        # Three distinguishable causes, not two. See the
                        # _REASON_ABSENT_* comment: status "absent" covers both
                        # a 404 and a 400/Error-96, and naming the wrong one
                        # would put a row in the log that contradicts its own
                        # http_status field.
                        if record.status == "measured_zero":
                            reason = _REASON_MEASURED_ZERO
                        elif record.http_status == 404:
                            reason = _REASON_ABSENT_404
                        elif record.http_status == 400:
                            reason = _REASON_ABSENT_400_NO_SUCH_FILE
                        else:
                            reason = _REASON_ABSENT_UNSTATED_STATUS
                        absent_log.append({
                            "filename": record.filename,
                            "reason": reason,
                            # See _file_span: the row carries its own file
                            # span so segment D's interval-overlap join never
                            # re-derives it.
                            **_file_span(record.filename),
                            "local_date": local_date.isoformat(),
                            "window_start_utc": start_utc.isoformat(),
                            "window_end_utc": end_utc.isoformat(),
                            "http_status": record.http_status,
                            "attempts": record.attempts,
                        })
                    else:
                        # download_archive_file's own contract is exactly these four
                        # statuses or a raise; anything else here is a broken contract,
                        # not a data problem, and must surface (invariant 5).
                        raise RuntimeError(
                            f"download_archive_file({filename!r}) returned unrecognised status "
                            f"{record.status!r}"
                        )

            dates_done += 1
            if dates_done % _MANIFEST_FLUSH_EVERY_N_DATES == 0:
                persist(snapshot())
        run_complete = True
    finally:
        # Runs on the way out whether the loop finished, raised, or was
        # interrupted. On the failure path this is the whole point of the
        # blocker fix: the absent-file log for everything already attempted
        # survives, flagged `run_complete: false`, instead of being lost for
        # the entire run. If THIS write fails the exception propagates -- and
        # if provenance is what failed, no manifest lands (check_b3c_5).
        manifest = snapshot()
        (manifest_path, provenance_path,
         run_manifest_path, run_provenance_path) = persist(manifest)

    absent = manifest["absent"]
    print(
        f"pull_overpass_corpus.run: {len(date_list)} date(s), {requested} file(s) listed, "
        f"{present} present, {absent} absent (dropped from the pull: {absent}); "
        f"{errors} of those absent are download ERRORS, not ONC saying no data; "
        f"{empty_windows} window(s) with zero files listed (measured zeros, decision 0016); "
        f"{empty_chunks_total} empty listing chunk(s) reported over "
        f"{len(date_list) - empty_chunks_unreported} date(s) "
        f"({empty_chunks_unreported} date(s) whose listing reported no chunk count)"
    )

    print(
        f"pull_overpass_corpus.run: this run's durable record -> {run_manifest_path} "
        f"(provenance {run_provenance_path}); latest-run pointers (OVERWRITTEN by the next "
        f"run) -> {manifest_path}, {provenance_path}"
    )
    return manifest


def _listing_fn_allow_empty(client, location_codes, start_utc, end_utc):
    """`list_fft_files` with decision 0016's short-span premise made explicit.

    The 2.5 h overpass window is a span where zero files is the ordinary case,
    so the opt-in is stated once, here, at the wiring point -- not hidden in a
    try/except around the call.
    """
    return onc_client.list_fft_files(
        client, location_codes, start_utc, end_utc, allow_empty=True
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    date_list = list(iter_overpass_dates(args.start_year, args.end_year))
    print(
        f"pull_overpass_corpus.py: {len(date_list)} in-season date(s), "
        f"{args.end_year} -> {args.start_year} (descending), "
        f"season months {list(config.SEASON_MONTHS_UTC)} (filtered on the LOCAL "
        f"{config.PLANET_OVERPASS_TZ_NAME} calendar date; identical to the UTC-evaluated "
        f"set for this window -- see iter_overpass_dates), "
        f"overpass window {config.PLANET_OVERPASS_WINDOW_START_LOCAL}-"
        f"{config.PLANET_OVERPASS_WINDOW_END_LOCAL} {config.PLANET_OVERPASS_TZ_NAME}; "
        f"dest-dir {args.dest_dir}; manifest-dir {args.manifest_dir}"
    )
    print(f"  {config.PLANET_SAMPLING_CONDITIONALITY_STATEMENT}")

    if args.dry_run:
        print("--dry-run: no ONC request made, no credential read, nothing written.")
        return 0

    # Real transport: a plain authenticated GET on archivefile/download,
    # established from the installed `onc` package source (see
    # boatphone.onc_client.RequestsArchiveTransport's module comment). This is
    # BUILDING the transport, not running a bulk pull -- the actual pull below
    # still walks every in-season date 2020-2025 and is a deliberate,
    # human-triggered operational step (module docstring), not something this
    # coding pass performs.
    client = get_onc_client()
    transport = onc_client.RequestsArchiveTransport(client)

    manifest = run(
        dates=date_list, dest_dir=args.dest_dir, manifest_dir=args.manifest_dir,
        listing_fn=_listing_fn_allow_empty, transport=transport, client=client,
    )
    print(
        f"SUMMARY: requested={manifest['requested']} present={manifest['present']} "
        f"absent={manifest['absent']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
