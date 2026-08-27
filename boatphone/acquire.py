"""Acquisition-window derivation for the PlanetScope-matched hydrophone pull (B3).

TIME CONVENTION AT THIS BOUNDARY (CLAUDE.md invariant 3, decision 0002):

* **In**: a `datetime.date` -- a LOCAL CALENDAR DATE in the
  `config.PLANET_OVERPASS_TZ_NAME` zone (`America/Vancouver`). It is a date, not
  an instant: "2024-11-03 in Vancouver" is what a Planet overpass schedule is
  expressed against. A naive or aware `datetime` is REJECTED, not coerced --
  passing one means the caller has confused a date with an instant, and
  silently taking its `.date()` would hide that.
* **Out**: a `(start_utc, end_utc)` pair of tz-aware `datetime`s with a ZERO UTC
  offset. Everything downstream (ONC listing, the FFT bin grid, the
  optical-acoustic join) is UTC, so the conversion happens exactly once, here.
* **The offset is never a literal.** It is looked up per date via `zoneinfo`,
  because `America/Vancouver` is UTC-8 in winter and UTC-7 in summer and the
  2024/2025 seasons plus their shoulder months straddle both transitions. A
  hardcoded offset produces a window that is silently one hour wrong on one
  side of March/November -- the kind of error that yields a plausible number
  rather than a failure.

The window edges themselves live in `boatphone/config.py`
(`PLANET_OVERPASS_WINDOW_START_LOCAL` / `_END_LOCAL` / `PLANET_OVERPASS_TZ_NAME`)
and are not restated here (invariant 6).
"""

import pathlib
from datetime import date as _date, datetime, timezone
from zoneinfo import ZoneInfo

from . import config, paths

__all__ = ["overpass_window_utc", "resolve_corpus_files"]


def overpass_window_utc(local_date: _date) -> tuple[datetime, datetime]:
    """Return the (start_utc, end_utc) of the PlanetScope overpass window for a
    local calendar date in `config.PLANET_OVERPASS_TZ_NAME`.

    Parameters
    ----------
    local_date : datetime.date
        A LOCAL calendar date (America/Vancouver). Must be a plain `date`; a
        `datetime` -- naive or aware -- raises `TypeError`.

    Returns
    -------
    (start_utc, end_utc) : tuple[datetime, datetime]
        Tz-aware UTC datetimes, `start_utc < end_utc`.

    Raises
    ------
    TypeError
        If `local_date` is not a `datetime.date`, or is a `datetime`.
    ValueError
        If the configured local window does not order start before end, or if
        the local wall-clock times do not exist / are ambiguous on this date
        (a DST gap or fold overlapping the window).
    """
    if isinstance(local_date, datetime):
        raise TypeError(
            "overpass_window_utc() takes a datetime.date (a LOCAL calendar date in "
            f"{config.PLANET_OVERPASS_TZ_NAME}), not a datetime; got {local_date!r}. "
            "A datetime is an instant and carries a time of day that this function "
            "would have to discard -- pass `.date()` explicitly at the call site if "
            "that is genuinely what you mean."
        )
    if not isinstance(local_date, _date):
        raise TypeError(
            "overpass_window_utc() takes a datetime.date, got "
            f"{type(local_date).__name__}: {local_date!r}"
        )

    start_local_time = config.PLANET_OVERPASS_WINDOW_START_LOCAL
    end_local_time = config.PLANET_OVERPASS_WINDOW_END_LOCAL
    if not start_local_time < end_local_time:
        raise ValueError(
            "config.PLANET_OVERPASS_WINDOW_START_LOCAL "
            f"({start_local_time}) is not before "
            f"config.PLANET_OVERPASS_WINDOW_END_LOCAL ({end_local_time})"
        )

    tz = ZoneInfo(config.PLANET_OVERPASS_TZ_NAME)
    start_local = datetime.combine(local_date, start_local_time, tzinfo=tz)
    end_local = datetime.combine(local_date, end_local_time, tzinfo=tz)

    # Both DST transitions in this zone occur at 02:00 local, well outside a
    # 09:15-11:45 window, so neither edge should ever land in a gap or a fold.
    # Assert it rather than assume it: if the zone rules ever move, a nonexistent
    # or ambiguous local time must surface as an error, not be resolved silently
    # to whichever branch zoneinfo defaults to (CLAUDE.md invariant 5).
    for label, aware_local in (("start", start_local), ("end", end_local)):
        _reject_nonexistent_or_ambiguous(label, aware_local, tz)

    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    return start_utc, end_utc


def _reject_nonexistent_or_ambiguous(label: str, aware_local: datetime, tz: ZoneInfo) -> None:
    """Raise if `aware_local` falls in a DST gap (nonexistent wall clock) or a
    DST fold (ambiguous wall clock) in `tz`.

    Nonexistent: the same wall clock resolves to different UTC instants under
    fold=0 and fold=1 AND round-tripping through UTC does not reproduce the
    wall clock. Ambiguous: the two folds differ but the wall clock survives the
    round trip.
    """
    fold0 = aware_local.replace(fold=0)
    fold1 = aware_local.replace(fold=1)
    if fold0.utcoffset() == fold1.utcoffset():
        return  # unambiguous and existent
    round_tripped = fold0.astimezone(timezone.utc).astimezone(tz)
    if round_tripped.replace(tzinfo=None, fold=0) != aware_local.replace(tzinfo=None, fold=0):
        raise ValueError(
            f"overpass window {label} {aware_local.replace(tzinfo=None)} does not exist on "
            f"{aware_local.date()} in {tz.key}: it falls in a DST spring-forward gap"
        )
    raise ValueError(
        f"overpass window {label} {aware_local.replace(tzinfo=None)} is AMBIGUOUS on "
        f"{aware_local.date()} in {tz.key}: it occurs twice across a DST fall-back fold, "
        "so it maps to two different UTC instants"
    )


def resolve_corpus_files(corpus_dir=None):
    """The ONE way to get "the B3 overpass-window corpus files" off disk.

    Exists so B5, segment E, and every notebook share a single glob instead of
    three (CLAUDE.md invariant 6). Two facts it centralises, both of which are
    silent-wrong-answer generators if a caller re-derives them:

    * **Extension.** The pulled corpus lands as `*.<config.ARCHIVE_EXTENSION>`
      -- `.fft` -- because that is the name the ONC archive index registers and
      serves, even though the bytes inside are gzip. `config.PRODUCT_EXTENSION`
      (`.fft.gz`) is the HAND-DELIVERED sample's name only. A notebook globbing
      `*.fft.gz` finds ZERO files in the pulled corpus and reports "no data"
      when the truth is "wrong glob" (invariant 9). This function matches the
      archive extension and deliberately does NOT match the product extension.
    * **Location.** The default is `paths.ONC_OVERPASS_CORPUS_DIR`, the
      subdirectory that marks the corpus as overpass-window-only on disk
      (decision 0020) rather than only inside a manifest.

    It **raises** rather than returning `[]` when the corpus location does not
    exist or holds no matching file (invariant 5). An empty list here would
    reach a detector as "zero vessels detected" over a corpus that was never
    pulled -- the exact confusion between "the method found nothing" and "the
    method is broken" that invariant 9 forbids.

    Parameters
    ----------
    corpus_dir : path-like or None
        Directory to resolve. `None` means `paths.ONC_OVERPASS_CORPUS_DIR`.

    Returns
    -------
    list[pathlib.Path]
        Sorted by filename, which for these names is chronological order (the
        stamp is fixed-width `YYYYmmddTHHMMSS.fffZ`). Non-empty by construction.

    Raises
    ------
    FileNotFoundError
        If `corpus_dir` does not exist, or exists but contains no
        `*.<ARCHIVE_EXTENSION>` file. Both messages name the path and say how
        to obtain the corpus.
    """
    corpus_dir = pathlib.Path(corpus_dir) if corpus_dir is not None else paths.ONC_OVERPASS_CORPUS_DIR
    paths.require_path(corpus_dir)

    suffix = "." + config.ARCHIVE_EXTENSION
    product_suffix = "." + config.PRODUCT_EXTENSION
    found = sorted(
        (entry for entry in corpus_dir.iterdir()
         if entry.is_file()
         and entry.name.endswith(suffix)
         and not entry.name.endswith(product_suffix)),
        key=lambda entry: entry.name,
    )
    if not found:
        n_product = sum(
            1 for entry in corpus_dir.iterdir()
            if entry.is_file() and entry.name.endswith(product_suffix)
        )
        raise FileNotFoundError(
            f"no {suffix!r} files in {corpus_dir} -- the B3 overpass-window corpus is "
            f"absent, which is NOT the same as an acoustically quiet corpus "
            f"(CLAUDE.md invariant 9). "
            + (f"({n_product} {product_suffix!r} file(s) ARE present: those are the "
               "hand-delivered sample's naming, not the pulled archive's.) "
               if n_product else "")
            + "How to obtain: run `python3 scripts/pull_overpass_corpus.py`."
        )
    return found
