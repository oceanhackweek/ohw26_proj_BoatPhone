"""Planet scene acquisition instants, and the acoustic files that cover them.

The join between Malachy's optical scene list and Isaac's `.fft.gz` corpus. It
is a LIBRARY module (CLAUDE.md invariant 6): it resolves times to files and
states coverage, and it decides nothing about detection.

TIME BASE, stated at the boundary (decision 0002): scene instants come from the
`acquired` column of the gate2 CSV, which is ISO-8601 with an explicit `+00:00`
offset. A row whose stamp carries no offset RAISES rather than being assumed
UTC -- that assumption is the one decision 0002 exists to forbid, and it is the
single most likely way this project produces a confident wrong answer.

COVERAGE IS MEASURED, NEVER INFERRED. `window_coverage` reports what is actually
on disk for a window, and a window with no files is a MEASURED ZERO (decisions
0008, 0016) -- an honest statement that the corpus does not cover that overpass,
not a failure and not something to fill in.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as _dt
import pathlib

from .config import (
    FFT_FILE_SECONDS,
    OVERPASS_MATCH_HALF_WINDOW_S,
    PLANET_GATE2_SURVIVORS_RELPATH,
)
from .onc_client import parse_file_coverage
from .paths import ONC_OVERPASS_CORPUS_DIR, REPO_ROOT

__all__ = [
    "Overpass",
    "WindowCoverage",
    "load_gate2_overpasses",
    "corpus_file_index",
    "window_coverage",
    "coverage_summary",
]


@dataclasses.dataclass(frozen=True)
class Overpass:
    """One Planet scene: its identity and its acquisition instant.

    ``acquired_utc`` is the TRUE acquisition instant, tz-aware UTC. It is the
    join key to the acoustic corpus and to the optical detection schema's
    ``acq_time_utc``.
    """

    scene_id: str
    acquired_utc: _dt.datetime
    clear_percent: float | None
    instrument: str | None

    def window_utc(self, half_window_s: int = OVERPASS_MATCH_HALF_WINDOW_S
                   ) -> tuple[_dt.datetime, _dt.datetime]:
        """The half-open acoustic window `[start, end)` centred on acquisition."""
        delta = _dt.timedelta(seconds=int(half_window_s))
        return (self.acquired_utc - delta, self.acquired_utc + delta)


def load_gate2_overpasses(csv_path=None) -> list[Overpass]:
    """Read the gate2 scene list into `Overpass` records, sorted by time.

    Raises
    ------
    FileNotFoundError
        if the scene list is absent. It is Malachy's output and lives in his
        contributor folder; a clear failure here is the intended behaviour when
        an input has not landed (CLAUDE.md, "write code that fails clearly when
        its input is absent rather than obscurely").
    ValueError
        if a row's `acquired` stamp carries no UTC offset, or the column is
        missing entirely.
    """
    if csv_path is None:
        csv_path = REPO_ROOT / PLANET_GATE2_SURVIVORS_RELPATH
    csv_path = pathlib.Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"no Planet gate2 scene list at {csv_path}. It is produced by "
            "contributor_folders/malachymcc/planet_folger_search_order_download.ipynb; "
            "without it there are no optical labels to match against."
        )

    out: list[Overpass] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "acquired" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path.name}: no 'acquired' column; columns are "
                f"{reader.fieldnames}. That column is the acquisition instant and "
                "there is no substitute for it."
            )
        for row_no, row in enumerate(reader, start=2):
            raw = (row.get("acquired") or "").strip()
            if not raw:
                raise ValueError(f"{csv_path.name} line {row_no}: empty 'acquired'")
            stamp = _dt.datetime.fromisoformat(raw)
            if stamp.tzinfo is None or stamp.utcoffset() is None:
                raise ValueError(
                    f"{csv_path.name} line {row_no}: 'acquired' = {raw!r} states no "
                    "UTC offset. Refusing to assume UTC -- decision 0002 forbids "
                    "exactly this assumption, and a one-hour error here is "
                    "invisible in every plot and fatal in the join."
                )
            clear = (row.get("clear_percent") or "").strip()
            out.append(Overpass(
                scene_id=(row.get("id") or "").strip(),
                acquired_utc=stamp.astimezone(_dt.timezone.utc),
                clear_percent=float(clear) if clear else None,
                instrument=(row.get("instrument") or "").strip() or None,
            ))
    out.sort(key=lambda o: o.acquired_utc)
    return out


def corpus_file_index(corpus_dir=None) -> list[tuple[_dt.datetime, _dt.datetime, pathlib.Path]]:
    """Every corpus product file as `(start_utc, end_utc, path)`, time-sorted.

    Built from FILENAMES via `onc_client.parse_file_coverage` -- the one place a
    filename becomes a time -- and NOT from the pull manifest. That is
    deliberate: the manifest is mid-migration and disagrees with the corpus by
    90 files, while the files themselves are the thing being analysed. Where the
    two disagree, the disk wins.

    Both containers are included. The corpus is permanently MIXED (`.fft.gz`
    from the bulk pull, plain `.fft` from the live probe -- decisions 0022,
    0024, 0025) and `fft_io.read_fft_gz` sniffs the container rather than
    trusting the name, so both are equally readable here. `.sha256` sidecars are
    skipped.
    """
    if corpus_dir is None:
        corpus_dir = ONC_OVERPASS_CORPUS_DIR
    corpus_dir = pathlib.Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(
            f"no acoustic corpus directory at {corpus_dir}; it is produced by "
            "scripts/pull_overpass_corpus.py"
        )
    index = []
    for path in corpus_dir.iterdir():
        if path.suffix == ".sha256" or not path.is_file():
            continue
        start_utc, end_utc = parse_file_coverage(path.name)
        index.append((start_utc, end_utc, path))
    index.sort(key=lambda row: row[0])
    return index


@dataclasses.dataclass(frozen=True)
class WindowCoverage:
    """What the corpus actually holds for one overpass window.

    ``is_full`` means the files present cover the whole window with no interior
    gap. ``covered_seconds`` counts only the parts of files that fall INSIDE the
    window, so a file straddling an edge contributes only its overlap -- a
    coverage figure that counted the whole file would overstate.
    """

    overpass: Overpass
    window_start_utc: _dt.datetime
    window_end_utc: _dt.datetime
    paths: tuple[pathlib.Path, ...]
    covered_seconds: float
    window_seconds: float
    is_full: bool

    @property
    def n_files(self) -> int:
        return len(self.paths)

    @property
    def covered_fraction(self) -> float:
        return self.covered_seconds / self.window_seconds if self.window_seconds else 0.0


def window_coverage(overpass: Overpass, index, *,
                    half_window_s: int = OVERPASS_MATCH_HALF_WINDOW_S) -> WindowCoverage:
    """Measure the corpus's coverage of one overpass's acoustic window.

    Uses INTERVAL OVERLAP, not start-time containment (decision 0020): a file
    starting just before the window and running into it does cover part of it,
    and a containment test would drop it and understate coverage at both edges.

    An empty result is a measured zero, not an error -- see the module
    docstring.
    """
    win_start, win_end = overpass.window_utc(half_window_s)
    window_seconds = (win_end - win_start).total_seconds()

    covered = 0.0
    paths = []
    prev_end = win_start
    for start_utc, end_utc, path in index:
        if end_utc <= win_start or start_utc >= win_end:
            continue
        paths.append(path)
        lo = max(start_utc, win_start)
        hi = min(end_utc, win_end)
        # Overlapping files would double-count; clamp to what is not already
        # covered rather than summing raw overlaps.
        lo = max(lo, prev_end)
        if hi > lo:
            covered += (hi - lo).total_seconds()
            prev_end = hi

    # Tolerance of one frame-free second: file starts carry 1-2 s of jitter and
    # are not bin-aligned (parse_file_coverage), so demanding exact tiling would
    # call a fully-covered window partial on rounding alone.
    is_full = covered >= window_seconds - 1.0
    return WindowCoverage(
        overpass=overpass,
        window_start_utc=win_start,
        window_end_utc=win_end,
        paths=tuple(paths),
        covered_seconds=covered,
        window_seconds=window_seconds,
        is_full=is_full,
    )


def coverage_summary(coverages) -> dict:
    """Counts of full / partial / zero coverage across a set of windows.

    The three categories are reported SEPARATELY and never collapsed: a zero is
    "the corpus does not reach this overpass" and a partial is "it reaches part
    of it". Averaging them into one coverage percentage would hide that 12 of
    the 30 scenes are not sampled at all.
    """
    coverages = list(coverages)
    full = [c for c in coverages if c.is_full]
    zero = [c for c in coverages if c.n_files == 0]
    partial = [c for c in coverages if c.n_files and not c.is_full]
    return {
        "n_overpasses": len(coverages),
        "n_full": len(full),
        "n_partial": len(partial),
        "n_zero": len(zero),
        "full_scene_ids": [c.overpass.scene_id for c in full],
        "partial_scene_ids": [c.overpass.scene_id for c in partial],
        "zero_scene_ids": [c.overpass.scene_id for c in zero],
        "file_seconds": float(FFT_FILE_SECONDS),
    }
