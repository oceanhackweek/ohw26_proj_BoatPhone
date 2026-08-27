"""Time-grid algebra and ONC listing for the A1 hydrophone uptime calendar.

The module has two halves. The **pure** half (`season_bins_utc`,
`mark_available`, `assign_deployment`, `parse_file_coverage`) makes no network
call, reads no file and imports no third-party package, so the time and unit
conventions can be gated on their own before anything depends on them (decision
0002). The **impure** half (`discover_folger_locations`, `list_fft_files`,
`get_deployments`) talks to ONC through a client object that is always passed in
-- this module never constructs one and never reads a token value.

That is not by itself enough to keep a token out of a message, and saying it was
would be a claim the code did not support. The `onc` client embeds the full
request URL -- `...&token=<the real token>` -- in the text of its own exceptions,
so a 401 raised through this module would print the credential into a notebook
cell (CLAUDE.md invariant 7). **Every** call into the client is therefore wrapped
by `_client_call()`, which redacts the message and re-raises OUTSIDE the `except`
block so no token-bearing link survives on `__cause__`, `__context__`, or in a
formatted traceback. The request parameters stay in the message; the credential
does not (D8).

Conventions at every boundary:

* **Time base**: UTC only, tz-aware only. Every datetime accepted or returned
  carries a zero UTC offset; a naive datetime raises `ValueError` rather than
  being assumed to be UTC -- silently assuming is exactly how a local-time value
  becomes a seven-hour join error.
* **Intervals**: half-open `[start, end)`. Bins, coverage intervals and
  deployment intervals all use this convention, so touching an edge is *not*
  overlap (a zero-length intersection is not overlap).
* **Grid**: every bin is exactly `config.BIN_SECONDS` seconds wide and its edges
  are integer multiples of `BIN_SECONDS` seconds since the UTC epoch, regardless
  of the requested bounds. Only whole bins fully inside the requested span are
  emitted -- a partial bin is a row that is not a full bin of listening.
* **Season**: membership is `month(start_utc) in SEASON_MONTHS_UTC`, evaluated
  in UTC. No local timezone is referenced anywhere in this module (D4).
* **Density**: the calendar is dense. Every in-season bin is emitted whether or
  not anything was recorded in it; availability is a separate parallel flag.
* **Deployments** come from ONC metadata, never inferred from gaps (D6).

Units: `BIN_SECONDS` and all interval arithmetic are in seconds; timestamps are
seconds since the UTC epoch.
"""

import csv
import hashlib
import random
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from boatphone.paths import ensure_dir
from boatphone.config import (
    ARCHIVE_EXTENSION,
    BIN_SECONDS,
    DEVICE_CATEGORY_CODE,
    DEVICE_CODE,
    FFT_FILE_SECONDS,
    FOLGER_NAME_FRAGMENT,
    ONC_LISTING_MAX_PAGES,
    ONC_LISTING_MIN_SUBCHUNK_SECONDS,
    ONC_LISTING_PAGE_ROWS_OBSERVED_MAX,
    SEASON_MONTHS_UTC,
    STUDY_END_UTC,
)

__all__ = [
    "DeploymentCoverageError",
    "EmptyListingError",
    "FilenameParseError",
    "ONCListingError",
    "season_bins_utc",
    "mark_available",
    "assign_deployment",
    "parse_file_coverage",
    "discover_folger_locations",
    "list_fft_files",
    "get_deployments",
    "coverage_intervals",
    "summarise_gaps",
    "mean_availability_by_utc_hour",
    "build_uptime_calendar",
    "write_uptime_calendar_csv",
    "UPTIME_CSV_HEADER",
    "DEPLOYMENTS_CSV_HEADER",
    "ArchiveTransportError",
    "DownloadRecord",
    "download_archive_file",
    "RequestsArchiveTransport",
]


class DeploymentCoverageError(RuntimeError):
    """A bin with data listed falls inside no known deployment.

    Either the ONC deployment metadata is incomplete or the time base is wrong.
    Both are real problems, so this raises rather than attributing the bin to no
    deployment (D6).
    """


def _require_utc(value, label: str) -> datetime:
    """Return `value` if it is a tz-aware UTC datetime; otherwise raise."""
    if not isinstance(value, datetime):
        raise TypeError(f"{label} must be a datetime, got {type(value).__name__}: {value!r}")
    offset = value.utcoffset()
    if offset is None:
        raise ValueError(
            f"{label} is a naive datetime ({value.isoformat()}); A1 is UTC end to end "
            "and a naive value must never cross a boundary (D1)"
        )
    if offset != timedelta(0):
        raise ValueError(
            f"{label} has a non-zero UTC offset ({offset}); pass UTC, not local time (D1/D4)"
        )
    return value


def _month_start_epoch(moment: datetime) -> int:
    """Epoch seconds of the first instant of the UTC month *after* `moment`."""
    year, month = moment.year, moment.month
    if month == 12:
        year, month = year + 1, 1
    else:
        month += 1
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())


def season_bins_utc(start_utc, end_utc, season_months=SEASON_MONTHS_UTC):
    """Dense list of half-open `[start, end)` bins inside `[start_utc, end_utc)`.

    Each bin is exactly `BIN_SECONDS` seconds wide with epoch-aligned edges, and
    only whole bins fully contained in the requested span are returned. Bins are
    kept when `month(bin_start)` is in `season_months`, evaluated in UTC.

    `season_months=None` disables the seasonal filter. This is a **test-only
    affordance** -- it exists so out-of-season days (e.g. the March/November DST
    transition days) can be exercised. Production paths must not use it.
    """
    start_utc = _require_utc(start_utc, "start_utc")
    end_utc = _require_utc(end_utc, "end_utc")

    season = None if season_months is None else frozenset(season_months)
    width = timedelta(seconds=BIN_SECONDS)
    start_ts = int(start_utc.timestamp())
    end_ts = int(end_utc.timestamp())
    # Round the first edge UP to the grid; the grid is defined by the epoch, not
    # by when the caller happened to ask (D2).
    remainder = start_ts % BIN_SECONDS
    if remainder:
        start_ts += BIN_SECONDS - remainder

    bins = []
    ts = start_ts
    while ts + BIN_SECONDS <= end_ts:
        moment = datetime.fromtimestamp(ts, timezone.utc)
        if season is not None and moment.month not in season:
            # Jump to the next UTC month rather than stepping bin by bin; month
            # boundaries are themselves integer multiples of BIN_SECONDS.
            ts = _month_start_epoch(moment)
            continue
        bins.append((moment, moment + width))
        ts += BIN_SECONDS
    return bins


def mark_available(bins, coverage_intervals):
    """Flag each bin that overlaps any coverage interval, parallel to `bins`.

    Both bins and coverage intervals are half-open `[start, end)`. The overlap
    must have non-zero length: an interval that only touches a bin edge does not
    mark that bin.
    """
    intervals = []
    for index, interval in enumerate(coverage_intervals):
        cov_start, cov_end = interval
        cov_start = _require_utc(cov_start, f"coverage_intervals[{index}] start")
        cov_end = _require_utc(cov_end, f"coverage_intervals[{index}] end")
        if cov_end < cov_start:
            raise ValueError(
                f"coverage_intervals[{index}] ends before it starts: "
                f"{cov_start.isoformat()} -> {cov_end.isoformat()}"
            )
        if cov_end > cov_start:  # a zero-length interval covers nothing
            intervals.append((cov_start, cov_end))

    flags = []
    for bin_index, (bin_start, bin_end) in enumerate(bins):
        _require_utc(bin_start, f"bins[{bin_index}] start")
        _require_utc(bin_end, f"bins[{bin_index}] end")
        flags.append(
            any(cs < bin_end and ce > bin_start for cs, ce in intervals)
        )
    return flags


def assign_deployment(bins, deployments, available):
    """Deployment id for each bin, from ONC metadata -- never inferred from gaps.

    `deployments` is a sequence of `(id_str, start_utc, end_utc)` half-open
    intervals; a bin belongs to the deployment whose interval contains its
    **start**. A bin inside no deployment gets `""`. A bin marked available that
    is inside no deployment raises `DeploymentCoverageError`.
    """
    if len(available) != len(bins):
        raise ValueError(
            f"available has {len(available)} flags for {len(bins)} bins; they must be parallel"
        )

    windows = []
    for index, deployment in enumerate(deployments):
        dep_id, dep_start, dep_end = deployment
        if not isinstance(dep_id, str):
            raise TypeError(
                f"deployments[{index}] id is {type(dep_id).__name__}, not str; ONC's "
                "identifier is kept verbatim so the join does not lose a leading zero"
            )
        dep_start = _require_utc(dep_start, f"deployments[{index}] start")
        dep_end = _require_utc(dep_end, f"deployments[{index}] end")
        if dep_end <= dep_start:
            raise ValueError(
                f"deployments[{index}] ({dep_id!r}) is empty or reversed: "
                f"{dep_start.isoformat()} -> {dep_end.isoformat()}"
            )
        windows.append((dep_id, dep_start, dep_end))

    ids = []
    for (bin_start, _bin_end), is_available in zip(bins, available):
        _require_utc(bin_start, "bin start")
        found = ""
        for dep_id, dep_start, dep_end in windows:
            if dep_start <= bin_start < dep_end:
                found = dep_id
                break
        if not found and is_available:
            raise DeploymentCoverageError(
                f"bin starting {bin_start.isoformat()} is marked available but falls in no "
                "known deployment; a file listed outside every deployment means the ONC "
                "metadata is incomplete or the time base is wrong (D6)"
            )
        ids.append(found)
    return ids


# ---------------------------------------------------------------------------
# Impure half: ONC discovery and archive-file listing
#
# Everything below takes an already-constructed ONC client as its first
# argument. This module never builds one and never sees a token
# (boatphone.credentials owns that), so no token value can reach a message here.
# ---------------------------------------------------------------------------


class FilenameParseError(ValueError):
    """An archive filename could not be turned into a UTC interval.

    A ValueError subclass so a caller can treat it as the bad-input error it is.
    Raised rather than returning None: a name that silently parses to nothing
    turns an unreadable listing into a plausibly-short uptime calendar.
    """


class ONCListingError(RuntimeError):
    """An ONC discovery or listing request produced no usable answer (D8).

    Covers "zero Folger candidates", "zero files across the whole span" and "a
    year-chunk request failed". Every one of these otherwise reads downstream as
    'the hydrophone was off', which is a different finding entirely (invariant 9).
    """


class EmptyListingError(ONCListingError):
    """The listing requests all SUCCEEDED and returned no file for the span.

    A subclass, so every caller that treats this as `ONCListingError` keeps its
    behaviour -- `list_fft_files()` still raises for a span with nothing in it,
    which is what A1b's D8 contract requires.

    It exists as its own type because "every request came back 200 with an empty
    page" and "a request failed" are different findings (invariant 9) and only
    one of them is a bug. `build_uptime_calendar()` is the single caller allowed
    to treat this one as a MEASURED zero -- see its docstring for why, and for
    the loud print that keeps it from being silent.
    """


# One archive filename, e.g. ICLISTENHF1266_20240715T000136.000Z.fft or
# ICLISTENHF1266_20260313T000000.029Z_20260313T000500.029Z.wav. The trailing
# group swallows the extension (".fft", ".fft.gz", ".wav") and ONC's derived
# suffixes ("-spect.png"). The literal "Z" is REQUIRED: a stamp without it
# states no time base, and guessing UTC is exactly the assumption decision 0002
# forbids.
_ARCHIVE_NAME_RE = re.compile(
    r"^(?P<device>[A-Za-z0-9]+)"
    r"_(?P<t1>\d{8}T\d{6}\.\d{3}Z)"
    r"(?:_(?P<t2>\d{8}T\d{6}\.\d{3}Z))?"
    r"(?P<rest>[.\-].*)?$"
)
_STAMP_FORMAT = "%Y%m%dT%H%M%S.%fZ"

# ONC metadata timestamps ("2020-03-08T14:53:50.000Z"), with and without the
# fractional second -- ONC emits both forms.
_ONC_TIME_FORMATS = ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ")

# Client methods tried, in order, for an archive listing at a location. Source:
# the onc package's ONC class; getArchivefileByLocation is the current name and
# getListByLocation the legacy alias.
_LISTING_METHODS = ("getArchivefileByLocation", "getListByLocation")

# Two ONC 400s that are NOT failures -- they are ONC stating that no data can
# exist in the window asked about, which is a MEASURED ZERO. Both were hit
# against the live API on 2026-08-27 while listing the full study window:
#
#   "API Error 127: A device with category HYDROPHONE was deployed at location
#    FGPD but not during the provided time range"  (FGPD, 2020-02-18..2020-03-01;
#    the deployment begins 2020-03-08)
#   "API Error 25: Invalid Time Range, Start Time is in the future."
#    (FGPD, 2026-09-01..2026-10-01; config.STUDY_END_UTC runs past today)
#
# No instrument was deployed, or the window has not happened yet. Either way
# there is nothing to list and zero files is the true answer, so returning an
# empty page is a measurement rather than a swallowed error. It is PRINTED every
# time so it can never be a silent zero, and it is matched on these exact phrases
# and nothing else -- a 401, a 500 or any other 4xx still propagates, because
# those are "the method is broken" and these are "the method found nothing"
# (invariant 9). The bins concerned end up UNAVAILABLE, which is the safe
# direction for Planet quota: it withholds an order, it never invents one.
#
# Recorded in docs/decisions/0007-onc-400s-that-are-measured-zeros.md.
_NO_DATA_POSSIBLE_MARKERS = (
    "not during the provided time range",   # API Error 127: no deployment
    "Start Time is in the future",          # API Error 25: window not yet elapsed
)


def _parse_stamp(stamp: str, filename: str) -> datetime:
    """`YYYYmmddTHHMMSS.fffZ` -> tz-aware UTC datetime. Sub-second is kept."""
    try:
        naive = datetime.strptime(stamp, _STAMP_FORMAT)
    except ValueError as exc:
        raise FilenameParseError(
            f"{filename!r}: timestamp {stamp!r} is not a valid "
            f"{_STAMP_FORMAT} instant ({exc})"
        ) from exc
    return naive.replace(tzinfo=timezone.utc)


def parse_file_coverage(filename):
    """Archive filename -> the half-open `[start_utc, end_utc)` it covers.

    Pure: no network, no disk. This is the ONE place a filename becomes a time.

    Conventions:

    * **Time base**: the name's `Z` is taken at face value and the returned
      datetimes are tz-aware UTC. A stamp without `Z` RAISES -- it states no time
      base, and assuming UTC is the assumption decision 0002 exists to forbid.
    * **Sub-second**: the `.fff` fraction is milliseconds and is preserved to the
      microsecond. A 29 ms truncation is invisible in a plot and fatal in a join.
    * **End**: a two-timestamp name (`..._start_end.wav`) uses its own end. A
      single-timestamp name (`..._start.fft` / `.fft.gz`) has no end in the name,
      so its end is `start + FFT_FILE_SECONDS`.

      This is MEASURED, not assumed: 2011 unique `.fft` names listed at Folger
      Deep over 2024-07-12..18 give consecutive start deltas of exactly 300 s in
      1909 of 2010 gaps and 297-302 s in 98 more, with two genuine outage gaps
      (637 s, 1780 s). The 1-2 s jitter and the off-grid start seconds (`:36`
      here, `:04` in the local sample) mean file starts are NOT bin-aligned; the
      grid lives in `season_bins_utc`, not in the filenames.
    * **Device**: the name must carry `config.DEVICE_CODE`. Folger has carried
      more than one hydrophone, and parsing another device's file as ours yields
      uptime for an instrument with a different calibration.

    Raises `FilenameParseError` (a ValueError) on anything it cannot read;
    `TypeError` if `filename` is not a str. It never returns None.
    """
    if not isinstance(filename, str):
        raise TypeError(
            f"filename must be a str, got {type(filename).__name__}: {filename!r}"
        )
    match = _ARCHIVE_NAME_RE.match(filename)
    if match is None:
        raise FilenameParseError(
            f"{filename!r} is not an ONC archive filename: expected "
            "'<DEVICECODE>_<YYYYmmddTHHMMSS.fffZ>[_<end stamp>]<extension>'"
        )
    device = match.group("device")
    if device != DEVICE_CODE:
        raise FilenameParseError(
            f"{filename!r} carries device code {device!r}, not config.DEVICE_CODE "
            f"({DEVICE_CODE!r}); a different instrument has a different calibration"
        )

    start_utc = _parse_stamp(match.group("t1"), filename)
    end_stamp = match.group("t2")
    if end_stamp is None:
        end_utc = start_utc + timedelta(seconds=FFT_FILE_SECONDS)
    else:
        end_utc = _parse_stamp(end_stamp, filename)
    if end_utc <= start_utc:
        raise FilenameParseError(
            f"{filename!r} covers an empty or reversed interval: "
            f"{start_utc.isoformat()} -> {end_utc.isoformat()}"
        )
    return start_utc, end_utc



def _client_call(client, method_name, params, request_text):
    """Call `client.<method_name>(params)`, re-raising any failure REDACTED.

    The single door between this module and the ONC client, and the only reason
    it exists is the credential. The `onc` package puts the whole request URL,
    token included, into its own exception text, so letting one propagate writes
    the token into a notebook cell.

    The mechanism matters, not just the redaction: the redacted error is raised
    **outside** the `except` block. `raise ... from exc` keeps the original on
    `__cause__`, and `raise ... from None` still keeps it on `__context__` -- in
    both cases `traceback.format_exception` prints the token-bearing link. Only
    raising after the block leaves has a chain of length one.

    `request_text` carries the request parameters into the message so the error
    is still diagnosable (D8). It must never be built from a token.

    The caught types are deliberate and narrow: `requests.HTTPError` (what the
    ONC client raises for a 4xx/5xx) is an `OSError`, and the client raises
    `RuntimeError`/`ValueError` for its own problems. No bare `except` and no
    `except Exception` -- a swallowed 4xx becomes an empty page (D8).
    """
    failure = None
    try:
        return getattr(client, method_name)(params)
    except (OSError, RuntimeError, ValueError) as exc:
        failure = (
            f"{method_name}({request_text}) failed: "
            f"{type(exc).__name__}: {_redact(str(exc))}"
        )
    raise ONCListingError(failure)


def _location_field(row, *names):
    """Value of the first present key in `row`, or None. Rows are ONC dicts."""
    if not isinstance(row, dict):
        return None
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return None


def discover_folger_locations(client):
    """Folger location codes, DISCOVERED from ONC at runtime -- never hardcoded.

    Asks `client.getLocations()` for the locations associated with
    `config.DEVICE_CODE` and keeps the rows whose location NAME contains
    `config.FOLGER_NAME_FRAGMENT` (case-insensitively). Matching on the name, not
    on a code we already knew, is the point: pasting the code a call returned is
    the cheapest way to silently pin the analysis to one site forever, and this
    package is source-scanned to make sure nobody does.

    Returns the codes as `str`, in ONC's own order, de-duplicated. Zero
    candidates RAISES `ONCListingError` (D8): returning `[]` makes every
    downstream listing empty and the calendar blank, which reads as "the
    hydrophone was off".
    """
    rows = _client_call(
        client, "getLocations", {"deviceCode": DEVICE_CODE},
        f"deviceCode={DEVICE_CODE!r}",
    )
    if rows is None:
        raise ONCListingError(
            f"getLocations(deviceCode={DEVICE_CODE!r}) returned None, not a list of rows"
        )

    codes = []
    seen = set()
    names_seen = []
    for row in rows:
        name = _location_field(row, "locationName", "name")
        code = _location_field(row, "locationCode", "code")
        if name is None or code is None:
            continue
        names_seen.append(f"{code}={name}")
        if FOLGER_NAME_FRAGMENT in str(name).lower() and code not in seen:
            seen.add(code)
            codes.append(str(code))

    if not codes:
        raise ONCListingError(
            f"no ONC location associated with device {DEVICE_CODE!r} has a name containing "
            f"{FOLGER_NAME_FRAGMENT!r}; got {len(names_seen)} location(s): "
            f"{names_seen}. Returning an empty list here would make every downstream "
            "listing empty and the uptime calendar blank (D8)"
        )
    return codes


def _year_chunks(start_utc, end_utc):
    """Split `[start_utc, end_utc)` into one clipped chunk per calendar UTC year."""
    chunks = []
    year = start_utc.year
    while True:
        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        next_year_start = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        chunk_start = max(start_utc, year_start)
        chunk_end = min(end_utc, next_year_start)
        if chunk_start < chunk_end:
            chunks.append((year, chunk_start, chunk_end))
        if next_year_start >= end_utc:
            break
        year += 1
    return chunks


def _month_chunks(start_utc, end_utc):
    """Split `[start_utc, end_utc)` into one clipped chunk per calendar UTC month.

    Used only for REPORTING granularity (the empty-chunk count), not for issuing
    requests -- see `list_fft_files`. Returns `[(year, month, start, end), ...]`.
    """
    chunks = []
    year, month = start_utc.year, start_utc.month
    while True:
        month_start = datetime(year, month, 1, tzinfo=timezone.utc)
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        next_month_start = datetime(next_year, next_month, 1, tzinfo=timezone.utc)
        chunk_start = max(start_utc, month_start)
        chunk_end = min(end_utc, next_month_start)
        if chunk_start < chunk_end:
            chunks.append((year, month, chunk_start, chunk_end))
        if next_month_start >= end_utc:
            break
        year, month = next_year, next_month
    return chunks


def _onc_stamp(moment: datetime) -> str:
    """UTC datetime -> the ISO-8601 `...Z` string the ONC API expects."""
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _listing_call(client):
    """The NAME of the client's archive-listing method, whichever it exposes.

    Returns the name rather than the bound method so the call itself still goes
    through `_client_call()` -- there is exactly one place a client is called.
    """
    for name in _LISTING_METHODS:
        if callable(getattr(client, name, None)):
            return name
    raise ONCListingError(
        f"the ONC client ({type(client).__name__}) exposes none of {_LISTING_METHODS}; "
        "there is no archive-file listing endpoint to call"
    )


def _files_from_response(response):
    """The filename list out of an ONC listing response, in whatever shape it came."""
    if response is None:
        return None
    if isinstance(response, dict):
        files = response.get("files")
        return [] if files is None else list(files)
    if isinstance(response, (list, tuple)):
        return list(response)
    return None


def _next_page_filters(response):
    """The request kwargs that fetch the NEXT page, or None if this page is the last.

    This is the whole fix for the A1d defect, so it is worth being exact about
    what is authoritative. ONC's archivefile listing truncates a large span
    SILENTLY -- the `files` list simply stops partway through the requested
    window with no error and nothing inside the list to mark it short. The one
    signal is the response's top-level `next`. MEASURED 2026-08-27 against the
    live endpoint: a complete response carries `next: None`; a truncated one
    carries `next: {"parameters": {..., "page": "2"}, "url": ...}`.

    So "is this complete?" is answered by `next`, NEVER by comparing the row
    count against a threshold: observed page sizes for one query were 11121,
    11075, 11086 and 10745, i.e. the cap is not a fixed number of rows, and a
    row-count test would reintroduce this bug the moment ONC's page size moved.
    `config.ONC_LISTING_PAGE_ROWS_OBSERVED_MAX` exists for error messages only.

    Only the paging key is copied out. `next["parameters"]` from the real API
    also contains **the token**, and `next["url"]` is the full token-bearing
    request URL; neither is ever carried into a request we build or a message we
    raise (CLAUDE.md invariant 7).

    Shapes accepted, because the cursor is the server's to name and a client
    that only understood one of them would be fragile: `next.parameters.page`,
    `next.page`, `next.offset`, a bare integer `next`, and a top-level
    `nextOffset`. Returns e.g. `{"page": "2"}` or `{"offset": 2000}`.
    """
    if not isinstance(response, dict):
        return None
    cursor = response.get("next")
    if isinstance(cursor, bool):
        return None
    if isinstance(cursor, int):
        return {"offset": cursor}
    if isinstance(cursor, dict):
        parameters = cursor.get("parameters")
        if isinstance(parameters, dict) and parameters.get("page") is not None:
            return {"page": parameters["page"]}
        if cursor.get("page") is not None:
            return {"page": cursor["page"]}
        if cursor.get("offset") is not None:
            return {"offset": cursor["offset"]}
    next_offset = response.get("nextOffset")
    if next_offset is not None:
        return {"offset": next_offset}
    return None


def _listing_filters(code, span_start, span_end):
    """The listing request body for one location code over `[span_start, span_end)`."""
    return {
        "locationCode": code,
        "deviceCategoryCode": DEVICE_CATEGORY_CODE,
        "extension": ARCHIVE_EXTENSION,
        "dateFrom": _onc_stamp(span_start),
        "dateTo": _onc_stamp(span_end),
    }


def _fetch_page(client, method_name, code, span_start, span_end, cursor, stats,
                page_number=1, names_so_far=0):
    """ONE listing request. Returns `(names, next_cursor_or_None)`.

    `next_cursor is None` is ONC's own statement that this page is the whole
    answer for the span; anything else means the listing was truncated.

    `page_number` is 1-based and `names_so_far` is how many rows the earlier
    pages of THIS span already returned. Both exist for one reason: the
    decision-0007 absorption is only defensible on the FIRST page of a span. On
    page 2+ the time range is byte-identical to page 1's and page 1 returned
    files from it, so "no device was deployed in the provided time range" /
    "Start Time is in the future" contradicts what the same server just said.
    That is "the method is broken", not "the method found nothing" (invariant 9),
    and absorbing it would return `([], None)` -- which `_fetch_pages` reads as
    "the listing is complete" and which truncates a dense span to page 1 while
    reporting it whole (the A1d phantom outage, second route).
    """
    filters = dict(_listing_filters(code, span_start, span_end), **(cursor or {}))
    cursor_text = ", ".join(f"{k}={v!r}" for k, v in (cursor or {}).items()) or "page=1"
    # Every client call goes through _client_call: it redacts the token the ONC
    # client puts in its own exception text and re-raises outside the except
    # block, so no token-bearing link survives on the chain. The request
    # parameters stay in the message (D8) -- and the paging cursor here is only a
    # page number or an offset, never a token.
    contradiction = None
    try:
        response = _client_call(
            client, method_name, filters,
            f"locationCode={code!r}, dateFrom={filters['dateFrom']}, "
            f"dateTo={filters['dateTo']}, extension={ARCHIVE_EXTENSION!r}, {cursor_text}",
        )
    except ONCListingError as exc:
        # Narrow and typed, never bare: ONLY the two "no data can exist here"
        # answers above are turned into an empty page, and each is PRINTED so it
        # can never be a silent zero. Anything else re-raises unchanged -- the exception is
        # already redacted and its chain is already length one, so re-raising it
        # here adds no token-bearing link (invariant 7).
        message = str(exc)
        if not any(marker in message for marker in _NO_DATA_POSSIBLE_MARKERS):
            raise
        if page_number > 1:
            # Deliberately NOT absorbed: build the message here, raise it below,
            # outside the except block. `exc` is already redacted by
            # _client_call and its chain is length one, but re-raising from
            # inside would re-attach it as __context__ for no benefit.
            contradiction = (
                f"ONC answered page {page_number} of the listing for "
                f"locationCode={code!r} over {span_start.isoformat()} -> "
                f"{span_end.isoformat()} with a 'no data can exist here' message, "
                f"after page(s) 1..{page_number - 1} of the SAME time range had "
                f"already returned {names_so_far} row(s). Those two statements "
                "contradict each other, so this is not a measured zero "
                "(docs/decisions/0007-onc-400s-that-are-measured-zeros.md, which "
                "applies to the FIRST page of a span only) -- something is broken: "
                "a mishandled paging parameter, a server fault behind a 400, or "
                "ONC rewording a message. Absorbing it would stop the paging loop "
                "and report the span complete at page 1, putting an outage that "
                "never happened into the uptime calendar (invariant 9). "
                f"ONC said: {message}"
            )
        stats["requests"] += 1
        if contradiction is None:
            print(
                f"list_fft_files: ONC states NO data can exist for {code} over "
                f"{span_start.isoformat()} -> {span_end.isoformat()} (no "
                f"{DEVICE_CATEGORY_CODE} deployed in the window, or the window has not "
                f"elapsed); recorded as ZERO files -- a measured zero, not a failed "
                f"request. ONC said: {message}"
            )
            return [], None
    if contradiction is not None:
        # Raised OUTSIDE the except block, like _client_call: chaining would put
        # the token-bearing original back on __context__ (invariant 7).
        raise ONCListingError(contradiction)
    stats["requests"] += 1
    names = _files_from_response(response)
    if names is None:
        raise ONCListingError(
            f"{method_name}(locationCode={code!r}, dateFrom={filters['dateFrom']}, "
            f"dateTo={filters['dateTo']}) returned {type(response).__name__}, which "
            "carries no file list"
        )
    return names, _next_page_filters(response)


def _fetch_pages(client, method_name, code, span_start, span_end, stats):
    """Follow `next` for one span. Returns `(names, unresolved_reason_or_None)`.

    `unresolved_reason` is None when the server itself declared the listing
    complete (`next` absent). It is a short string when the server says there is
    more but following it did not make progress -- an unknown cursor shape, a
    repeated cursor, an empty page, or more pages than
    `config.ONC_LISTING_MAX_PAGES`. Those are the cases where the caller falls
    back to subdividing the span; none of them may return the short list.
    """
    names = []
    extra = None
    seen_cursors = []
    for page_index in range(ONC_LISTING_MAX_PAGES):
        # page_number/names_so_far let _fetch_page tell a first-page "no data can
        # exist" answer (a measured zero, D7) from the same message arriving on a
        # later page of a span page 1 already returned rows for, which is a
        # contradiction and must raise rather than end the loop (invariant 9).
        page_names, cursor = _fetch_page(
            client, method_name, code, span_start, span_end, extra, stats,
            page_number=page_index + 1, names_so_far=len(names),
        )
        names.extend(page_names)
        if cursor is None:
            return names, None
        if not page_names:
            return names, "the response advertised another page but returned zero rows"
        if cursor in seen_cursors:
            return names, (
                f"the response advertised the same next-page cursor twice ({cursor!r}); "
                "the paging parameter is not being honoured"
            )
        seen_cursors.append(cursor)
        extra = cursor
    return names, (
        f"more than {ONC_LISTING_MAX_PAGES} pages "
        f"(config.ONC_LISTING_MAX_PAGES) and the listing was still truncated"
    )


def _list_span_complete(client, method_name, code, span_start, span_end, stats):
    """Every name ONC has for `[span_start, span_end)` at `code`, PROVEN complete.

    Pagination first -- measured, paging a whole 2024 season span this way is 4
    requests / ~50 s for 44,027 rows, so it is the cheap route as well as the
    honest one. If
    the server says the listing is truncated but paging cannot be followed, the
    span is halved and each half re-listed -- a route that needs no cooperation
    from the paging API at all, because a small enough window fits in one page.

    It TERMINATES: each recursion halves the span, and at
    `config.ONC_LISTING_MIN_SUBCHUNK_SECONDS` (300 s -- one FFT file, the
    documented floor) it raises `ONCListingError` naming the span and the cap
    rather than subdividing further or returning what it has. A short list is
    never returned (invariant 9: a truncated listing reads downstream as an
    outage that never happened).

    Sub-spans overlap-inclusively return boundary files twice; the caller keeps a
    file only in the chunk containing its START and collects into a set, so the
    duplicates are dropped rather than double-counted.
    """
    names, unresolved = _fetch_pages(client, method_name, code, span_start, span_end, stats)
    if unresolved is None:
        return names
    span_seconds = (span_end - span_start).total_seconds()
    if span_seconds <= ONC_LISTING_MIN_SUBCHUNK_SECONDS:
        raise ONCListingError(
            f"ONC truncated its listing for locationCode={code!r} over "
            f"{span_start.isoformat()} -> {span_end.isoformat()} "
            f"({span_start.strftime('%Y-%m')}..{span_end.strftime('%Y-%m')}) at the "
            f"response page cap (~{ONC_LISTING_PAGE_ROWS_OBSERVED_MAX} rows observed; "
            f"{len(names)} row(s) here), and the truncation could not be resolved: "
            f"{unresolved}. The span is already at the "
            f"{ONC_LISTING_MIN_SUBCHUNK_SECONDS} s floor "
            f"(config.ONC_LISTING_MIN_SUBCHUNK_SECONDS) so it cannot be subdivided "
            "further. Returning the truncated list would put a phantom outage in the "
            "uptime calendar, so this raises instead (invariant 9)"
        )
    half = max(ONC_LISTING_MIN_SUBCHUNK_SECONDS, span_seconds // 2)
    midpoint = span_start + timedelta(seconds=half)
    if not span_start < midpoint < span_end:  # pragma: no cover - guarded by the floor
        raise ONCListingError(
            f"cannot subdivide {span_start.isoformat()} -> {span_end.isoformat()} "
            f"to resolve a truncated listing for locationCode={code!r}: {unresolved}"
        )
    left = _list_span_complete(client, method_name, code, span_start, midpoint, stats)
    right = _list_span_complete(client, method_name, code, midpoint, span_end, stats)
    return left + right


def list_fft_files(client, location_codes, start_utc, end_utc, *, allow_empty=False):
    """List the archive FFT files ONC reports for `[start_utc, end_utc)`.

    Returns `(filenames, empty_chunks)` and PRINTS the empty-chunk count.

    What "available" means downstream (D3): a bin is available when ONC's
    listing reports a `.fft`/`.fft.gz` file whose coverage intersects it. That is
    **ONC's belief that a file exists, not proof that it downloads.** A4 refines
    availability from the actual pull, and the pull wins.

    Conventions and choices, stated because each one moves the uptime number:

    * **Time base**: `start_utc`/`end_utc` are tz-aware UTC, half-open
      `[start, end)`. Naive input raises.
    * **Chunking**: one *query* per calendar UTC year per location code, clipped
      to the requested span, and each query is PAGINATED to completeness (see
      `_list_span_complete`). Chunking bounds the blast radius of one failed
      request.
    * **The response page cap (A1d)**: ONC truncates a large listing silently and
      marks it only in the response's `next` field. A truncated page is never
      accepted as complete -- it is paged through, or failing that the span is
      halved down to a `config.ONC_LISTING_MIN_SUBCHUNK_SECONDS` floor, or it
      raises. Measured 2026-08-27: a single request for 2024-05-01..2024-10-01
      returns 11,121 of 44,027 files and stops at 2024-06-08, which the old code
      reported as a four-month outage that never happened (invariant 9).
    * **Empty-chunk granularity**: a chunk is the calendar year, UNLESS ONC
      truncated that year -- then the year's calendar UTC months are the chunks,
      because a span ONC cannot answer in one page is not a unit it can make a
      statement about. `empty_chunks` counts chunks ONC answered COMPLETELY and
      had no file for. A cap is therefore never an empty chunk (a capped query
      is completed or raised before anything is counted), while a genuine
      no-data month next to a dense one stays visible. Those are different
      findings and only one of them is a bug (invariant 9).
    * **Ownership of a boundary file**: ONC returns files that *overlap* the
      requested window, including one that began before it. A file is kept by the
      chunk whose span contains its START, so the year-chunks partition the
      listing instead of double-counting every year boundary. Cost: a file
      starting in the last `FFT_FILE_SECONDS` before `start_utc` is dropped, so
      the very first bin of the whole span can be under-reported. That is one bin
      at one edge, and it is stated rather than hidden.
    * **Other devices**: Folger has carried more than one hydrophone. Names for a
      device other than `config.DEVICE_CODE` are dropped and the count is
      printed. A name that DOES carry our device code but cannot be parsed raises
      `FilenameParseError` -- that is a broken listing, not an absent one.
    * **Extension**: `config.ARCHIVE_EXTENSION` ("fft"), which is what the ONC
      archive index registers. `config.PRODUCT_EXTENSION` ("fft.gz") is the
      on-disk gzipped form and returns ZERO files if used as a listing filter.

    * **`allow_empty`**: decision 0008's raise-on-empty is justified BY SPAN
      SIZE -- over a year-scale span, zero files is evidence of a broken
      request, not a silent hydrophone. That premise inverts for a short span:
      B3's per-date PlanetScope overpass window is 2.5 h, where an empty
      listing is the ordinary case (one outage morning). Callers working at
      that scale pass `allow_empty=True` and get `([], empty_chunks)` back, and
      MUST record the empty window as a measured zero themselves. Default
      stays `False` so the year-scale callers keep decision 0008 unchanged.
      See `docs/decisions/0016-empty-overpass-window-is-a-measured-zero.md`.

    Errors surface (D8): zero files across the whole span raises `ONCListingError`
    -- an all-unavailable calendar looks exactly like data. A failed chunk raises
    too, naming the failing span and location code and the underlying error text
    with any token-shaped value redacted; it is never turned into an empty page.
    A single genuinely-empty month is legitimate, but it is counted and printed
    so "found nothing" stays distinguishable from "is broken" (invariant 9).
    """
    start_utc = _require_utc(start_utc, "start_utc")
    end_utc = _require_utc(end_utc, "end_utc")
    if end_utc <= start_utc:
        raise ValueError(
            f"listing span is empty or reversed: {start_utc.isoformat()} -> "
            f"{end_utc.isoformat()}"
        )
    codes = [str(c) for c in location_codes]
    if not codes:
        raise ONCListingError(
            "list_fft_files was given no location codes; discover_folger_locations() "
            "raises rather than returning an empty list, so an empty list here means a "
            "caller manufactured one (D8)"
        )

    method_name = _listing_call(client)
    chunks = _year_chunks(start_utc, end_utc)

    kept = set()
    foreign_device = 0
    empty_chunks = 0
    total_chunks = 0
    stats = {"requests": 0}
    for _year, chunk_start, chunk_end in chunks:
        # One probe request per code for the whole year. `cursor is None` is
        # ONC's own statement that the page IS the year, and that is the only
        # thing trusted here -- never a row count (A1d, _next_page_filters).
        chunk_names = []
        truncated = False
        for code in codes:
            names, cursor = _fetch_page(
                client, method_name, code, chunk_start, chunk_end, None, stats
            )
            if cursor is None:
                chunk_names.extend(names)
                continue
            truncated = True
            # The probe was cut off at the response cap, so it is discarded and
            # the span listed again to PROVEN completeness -- paged, or
            # sub-chunked, or raised, but never the short list.
            chunk_names.extend(_list_span_complete(
                client, method_name, code, chunk_start, chunk_end, stats
            ))

        if truncated:
            # ONC could not answer this year in one page, so the year is not a
            # unit it can make a statement about and the chunks become its
            # calendar UTC months. The months are cut from the COMPLETE listing
            # above -- no extra requests -- so this costs one probe per truncated
            # year and buys the empty-chunk signal a month's resolution exactly
            # where a span was dense enough to hide an outage.
            units = [(unit_start, unit_end) for _y, _m, unit_start, unit_end
                     in _month_chunks(chunk_start, chunk_end)]
        else:
            units = [(chunk_start, chunk_end)]
        total_chunks += len(units)

        # Filter to our device ONCE per year rather than once per unit, so the
        # dropped count counts names and not name-unit pairs.
        chunk_files = []
        for name in chunk_names:
            name = str(name)
            if not name.startswith(f"{DEVICE_CODE}_"):
                foreign_device += 1
                continue
            file_start, _file_end = parse_file_coverage(name)
            chunk_files.append((file_start, name))

        for unit_start, unit_end in units:
            unit_kept = 0
            for file_start, name in chunk_files:
                # Ownership by START, at the unit granularity: the units
                # partition the year chunk, so a boundary file returned by two
                # overlapping queries lands in exactly one of them. The one
                # documented cost is unchanged -- a file starting in the
                # FFT_FILE_SECONDS before start_utc is dropped.
                if unit_start <= file_start < unit_end:
                    unit_kept += 1
                    kept.add(name)
            # An empty chunk is a unit ONC answered COMPLETELY and had no file
            # for. A truncated response can never reach this line: it is either
            # completed or raised first. So "ONC reports no data" and "the
            # request came back short" can never be the same number, and a
            # genuine no-data month beside a capped one still shows (invariant 9).
            if unit_kept == 0:
                empty_chunks += 1

    filenames = sorted(kept)
    print(
        f"list_fft_files: {len(filenames)} file(s) from {total_chunks} chunk(s) "
        f"({len(chunks)} year-chunk(s), {stats['requests']} request(s) after paging) "
        f"over {len(codes)} location(s) {codes}; {empty_chunks} empty chunk(s); "
        f"{foreign_device} name(s) dropped as another device's"
    )
    if not filenames and not allow_empty:
        raise EmptyListingError(
            f"ONC listed zero {ARCHIVE_EXTENSION!r} files for {codes} over "
            f"{start_utc.isoformat()} -> {end_utc.isoformat()} "
            f"({total_chunks} chunk(s), all empty; {stats['requests']} "
            f"request(s) issued). Zero files for a whole query is a "
            "broken request, not a silent hydrophone: returning an empty list here would "
            "produce an all-unavailable uptime calendar that looks like data (D8)"
        )
    return filenames, empty_chunks


def _redact(text: str) -> str:
    """Strip anything token-shaped out of a third-party error message.

    The ONC client embeds the request URL -- including `token=...` -- in its own
    exception text. No token value may reach a log line or an exception raised by
    this package, so the query parameter is replaced wholesale.
    """
    return re.sub(r"(?i)token=[^&\s]*", "token=<redacted>", text)


def _parse_onc_time(value, label: str):
    """ONC metadata timestamp string -> tz-aware UTC datetime, or None if absent."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return _require_utc(value, label)
    if not isinstance(value, str):
        raise TypeError(f"{label} is {type(value).__name__}, not an ONC timestamp string")
    for fmt in _ONC_TIME_FORMATS:
        try:
            naive = datetime.strptime(value, fmt)
        except ValueError:
            continue
        return naive.replace(tzinfo=timezone.utc)
    raise ValueError(
        f"{label} {value!r} matches none of the ONC UTC formats {_ONC_TIME_FORMATS}; "
        "it states no time base this code will guess at (decision 0002)"
    )


def get_deployments(client):
    """ONC deployment metadata for `config.DEVICE_CODE`, as `(id, start, end)`.

    Exactly the shape `assign_deployment()` consumes, and deployments come from
    METADATA -- never inferred from gaps in a listing (D6). A gap in the listing
    means "no file", which is not the same fact as "not deployed".

    Conventions:

    * **Time base**: `begin`/`end` are ONC's UTC strings, returned tz-aware UTC.
    * **Interval**: half-open `[start, end)`.
    * **Identity**: ONC's `/deployments` response for this device carries NO
      `deploymentId` field (checked live, 2026-08-27 -- the keys are begin, end,
      locationCode, deviceCode, lat/lon/depth, citation). So the id is composed
      as `"<deviceCode>@<locationCode>:<begin>"`, which is stable across calls
      and unique per deployment. It is a constructed key, not an ONC identifier,
      and anything joining on it must join against this function -- worth a
      decision record if it outlives A1.
    * **Open-ended deployment**: a row with no `end` is still deployed. Its end
      becomes `config.STUDY_END_UTC`, the analysis bound; a row that begins at or
      after that bound raises rather than being silently emptied.

    Returned in start order. Rows for every location this device has occupied are
    included -- filtering them to Folger here would hide a bin that the listing
    attributes to a site we did not expect.
    """
    rows = _client_call(
        client, "getDeployments", {"deviceCode": DEVICE_CODE},
        f"deviceCode={DEVICE_CODE!r}",
    )
    if rows is None:
        raise ONCListingError(
            f"getDeployments(deviceCode={DEVICE_CODE!r}) returned None, not a list of rows"
        )

    deployments = []
    for index, row in enumerate(rows):
        label = f"deployment row {index} for {DEVICE_CODE}"
        begin = _parse_onc_time(_location_field(row, "begin", "dateFrom"), f"{label} begin")
        if begin is None:
            raise ONCListingError(f"{label} has no begin time: {row!r}")
        end = _parse_onc_time(_location_field(row, "end", "dateTo"), f"{label} end")
        if end is None:
            end = STUDY_END_UTC
            if end <= begin:
                raise ONCListingError(
                    f"{label} is open-ended and begins {begin.isoformat()}, at or after "
                    f"STUDY_END_UTC ({STUDY_END_UTC.isoformat()}); widen the study window "
                    "in boatphone/config.py rather than emptying the deployment here"
                )
        code = _location_field(row, "locationCode", "code")
        dep_id = f"{DEVICE_CODE}@{code}:{_onc_stamp(begin)}"
        deployments.append((dep_id, begin, end))

    if not deployments:
        raise ONCListingError(
            f"ONC reports no deployments at all for device {DEVICE_CODE!r}; with no "
            "deployment metadata every available bin would raise DeploymentCoverageError (D6)"
        )
    deployments.sort(key=lambda d: d[1])
    return deployments


def coverage_intervals(filenames):
    """Filenames -> sorted, MERGED half-open `[start_utc, end_utc)` coverage.

    The bridge from `list_fft_files()` to `mark_available()`, and it exists for a
    measured reason: `mark_available` is O(bins x intervals), and one May-September
    season at Folger Deep is ~45,000 files (8,921 measured for July 2024 alone)
    against 44,064 in-season bins -- 2e9 comparisons. Merging first collapses that
    to a handful of intervals, one per real outage.

    Two adjacent intervals are merged when the gap between them is **strictly less
    than `BIN_SECONDS`**. That tolerance is not a fudge, it is provably
    flag-preserving: merging across a gap `g` adds coverage only inside that gap,
    and a bin's flag can only change if the bin lies entirely within it, which
    needs `g >= BIN_SECONDS`. So for `g < BIN_SECONDS` the merged intervals give
    `mark_available` exactly the flags the unmerged ones would.

    The tolerance is needed because file starts jitter by 1-2 s around the 300 s
    cadence (measured: 98 of 2010 consecutive gaps in a real week are 297-302 s
    rather than 300), so consecutive files leave 1-2 s holes that are an artefact
    of ONC's clock rounding, not gaps in the recording.

    A real outage -- the 637 s and 1780 s gaps in that same week -- is larger than
    `BIN_SECONDS` and is therefore preserved, unavailable, and visible.
    """
    intervals = []
    for name in filenames:
        intervals.append(parse_file_coverage(name))
    intervals.sort()

    tolerance = timedelta(seconds=BIN_SECONDS)
    merged = []
    for start, end in intervals:
        if merged and start - merged[-1][1] < tolerance:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


# ---------------------------------------------------------------------------
# A1c: the composed uptime calendar and its CSV (outbound deliverable O1)
# ---------------------------------------------------------------------------

# Column order of hydrophone_uptime.csv. Pinned here, in the library, because
# the optical stream reads this file to decide which dates are worth Planet
# quota; a second definition elsewhere is how the two sides drift apart
# (CLAUDE.md invariant 6).
UPTIME_CSV_HEADER = ("start_utc", "end_utc", "available", "deployment_id")

# Header of the deployments sidecar CSV, in the order `get_deployments()` returns
# its tuples. Defined beside UPTIME_CSV_HEADER so the two O1 tables have ONE
# definition each, shared across workstreams (CLAUDE.md invariant 6). Times are
# tz-aware UTC isoformat strings; the interval is half-open [start, end).
DEPLOYMENTS_CSV_HEADER = ("deployment_id", "start_utc", "end_utc")


def build_uptime_calendar(client, start_utc, end_utc):
    """Uptime calendar rows for `[start_utc, end_utc)`: pure composition of A1a+A1b.

    Returns `[(bin_start_utc, bin_end_utc, available, deployment_id), ...]`, one
    row per in-season bin, in time order.

    * `bin_start_utc` / `bin_end_utc`: tz-aware UTC, half-open, exactly
      `config.BIN_SECONDS` seconds wide, epoch-aligned (`season_bins_utc`).
    * `available` (`bool`): MEASURED -- ONC listed a `.fft` file whose coverage
      overlaps this bin (`list_fft_files` -> `coverage_intervals` ->
      `mark_available`). It is never defaulted in either direction. It is ONC's
      belief that a file exists, not proof that it downloads; A4's absent-file
      log refines it and the pull wins.
    * `deployment_id` (`str`): from ONC deployment METADATA
      (`get_deployments` -> `assign_deployment`), never inferred from a gap in
      the listing (D6). An unavailable bin inside a known deployment still
      carries that deployment's id; `""` means "inside no known deployment".

    The calendar is DENSE (D5): `len(rows) == len(season_bins_utc(start, end))`
    always. This function does no network work of its own -- it takes the client
    and the span, so it can be driven with a stub -- and it deliberately does not
    know about the full 2020-2026 study window; that span belongs to a runnable
    entry point in `scripts/`, not to the library.

    Errors surface (invariant 5). A failed listing chunk, a discovery with no
    Folger candidate, or a bin listed outside every deployment all PROPAGATE. A
    short calendar must never be returned: this artefact tells the PlanetScope
    stream which dates have no hydrophone data, Planet quota is ~30 orders per
    month, and a spent order is unrecoverable, so a truncated calendar that looks
    complete wastes quota irrecoverably.

    The one deliberate exception, recorded in
    `docs/decisions/0008-empty-listing-is-a-measured-zero.md`: an
    `EmptyListingError` -- every listing request succeeded and returned nothing
    for this span -- is treated as a measured zero and produces an
    all-unavailable calendar, with a loud printed line saying how many bins that
    is. `list_fft_files` raises on it because a whole-study-window query that
    returns nothing is a broken request; but this function is also called for a
    short span, where "no file in this hour" is a real and expected finding, and
    raising there would make the calendar unable to state the one thing it
    exists to state. The distinction that keeps this honest is that
    `EmptyListingError` attests the requests SUCCEEDED; a request that failed
    raises `ONCListingError` and is not caught here.
    """
    start_utc = _require_utc(start_utc, "start_utc")
    end_utc = _require_utc(end_utc, "end_utc")
    if end_utc <= start_utc:
        raise ValueError(
            f"uptime calendar span is empty or reversed: {start_utc.isoformat()} -> "
            f"{end_utc.isoformat()}"
        )

    bins = season_bins_utc(start_utc, end_utc)
    if not bins:
        raise ValueError(
            f"{start_utc.isoformat()} -> {end_utc.isoformat()} contains no whole in-season "
            f"{BIN_SECONDS} s bin (season months {SEASON_MONTHS_UTC}, UTC). An empty calendar "
            "is indistinguishable from a scan that found nothing, so this raises rather than "
            "returning [] (invariant 9)"
        )

    codes = discover_folger_locations(client)

    empty_listing = None
    try:
        filenames, empty_chunks = list_fft_files(client, codes, start_utc, end_utc)
    except EmptyListingError as exc:
        # Caught OUTSIDE-safe: narrow subclass only. A failed request raises the
        # parent ONCListingError and is NOT caught here (see the docstring).
        empty_listing = str(exc)
        filenames, empty_chunks = [], None
    if empty_listing is not None:
        print(
            f"build_uptime_calendar: ONC listed ZERO files for {codes} over "
            f"{start_utc.isoformat()} -> {end_utc.isoformat()}; every one of the "
            f"{len(bins)} in-season bin(s) is recorded UNAVAILABLE. The requests "
            f"succeeded, so this is a measured outage, not a failure -- but check it "
            f"against A4's pull before spending Planet quota on the strength of it. "
            f"Underlying: {empty_listing}"
        )

    intervals = coverage_intervals(filenames)
    available = mark_available(bins, intervals)
    deployments = get_deployments(client)
    deployment_ids = assign_deployment(bins, deployments, available)

    rows = [
        (bin_start, bin_end, bool(flag), dep_id)
        for (bin_start, bin_end), flag, dep_id in zip(bins, available, deployment_ids)
    ]
    if len(rows) != len(bins):
        raise RuntimeError(
            f"assembled {len(rows)} rows for {len(bins)} bins; the calendar must be dense (D5)"
        )
    covered = sum(1 for row in rows if row[2])
    print(
        f"build_uptime_calendar: {len(rows)} bin(s) of {BIN_SECONDS} s over "
        f"{start_utc.isoformat()} -> {end_utc.isoformat()}; {covered} available "
        f"({100.0 * covered / len(rows):.1f}%), {len(rows) - covered} unavailable; "
        f"{len(filenames)} file(s) listed, {len(intervals)} merged coverage interval(s); "
        f"empty chunks: {empty_chunks if empty_chunks is not None else 'n/a (empty listing)'}"
    )
    return rows


def write_uptime_calendar_csv(rows, path):
    """Write calendar rows to `path` as the O1 CSV. One data row in, one out.

    Header is exactly `UPTIME_CSV_HEADER`, in that order. Serialisation
    conventions, stated because the reader is another workstream:

    * `start_utc`, `end_utc`: `datetime.isoformat()` on a tz-aware UTC value, so
      the string always carries an explicit `+00:00`. A bare naive stamp is the
      shape the next reader silently treats as local time (decision 0002), so a
      naive or non-UTC input RAISES rather than being written.
    * `available`: Python `str(bool)` -- "True"/"False".
    * `deployment_id`: ONC's composed deployment key verbatim, or empty for a
      bin inside no known deployment.

    Nothing is filtered, deduplicated or sorted here: a silent drop at the write
    step is indistinguishable from a truncated scan once the file is on disk.
    """
    path = Path(path)
    rows = list(rows)
    if not rows:
        raise ValueError(
            f"refusing to write an empty uptime calendar to {path}; a header-only CSV "
            "reads downstream as 'no data anywhere', which is a different claim from "
            "'every bin unavailable' (invariant 9)"
        )

    prepared = []
    for index, row in enumerate(rows):
        if len(row) != len(UPTIME_CSV_HEADER):
            raise ValueError(
                f"row {index} has {len(row)} field(s), expected {len(UPTIME_CSV_HEADER)} "
                f"{list(UPTIME_CSV_HEADER)}: {row!r}"
            )
        bin_start, bin_end, available, deployment_id = row
        bin_start = _require_utc(bin_start, f"row {index} start_utc")
        bin_end = _require_utc(bin_end, f"row {index} end_utc")
        if bin_end <= bin_start:
            raise ValueError(
                f"row {index} is empty or reversed: {bin_start.isoformat()} -> "
                f"{bin_end.isoformat()}"
            )
        if not isinstance(available, bool):
            raise TypeError(
                f"row {index} available is {type(available).__name__}, not bool; a truthy "
                "placeholder here would serialise as a plausible availability claim"
            )
        if not isinstance(deployment_id, str):
            raise TypeError(
                f"row {index} deployment_id is {type(deployment_id).__name__}, not str; "
                "ONC's identifier is kept verbatim so the join does not lose a leading zero"
            )
        prepared.append(
            [bin_start.isoformat(), bin_end.isoformat(), str(available), deployment_id]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(UPTIME_CSV_HEADER))
        writer.writerows(prepared)
    print(f"write_uptime_calendar_csv: {len(prepared)} data row(s) -> {path}")


# ---------------------------------------------------------------------------
# A1c: inspection -- gap runs and the UTC-hour availability profile
#
# Both take the calendar's own outputs (`season_bins_utc` bins and the parallel
# `mark_available` flags), so a notebook can render them without redefining any
# time algebra of its own.
# ---------------------------------------------------------------------------

# Hours in a day, for the UTC-hour profile. Named rather than inlined: the
# profile is indexed by the UTC hour of the bin start, and nothing here is ever
# a local-time hour (D4).
HOURS_PER_DAY = 24


def summarise_gaps(bins, available, min_seconds=0):
    """Maximal runs of contiguous UNAVAILABLE bins, as `(start_utc, end_utc, n_bins)`.

    `bins` is a sequence of half-open `(start_utc, end_utc)` bins as
    `season_bins_utc` returns; `available` is the parallel flag list from
    `mark_available`. Returns the runs in ascending time order, keeping only
    those whose duration is `>= min_seconds` (the threshold is INCLUSIVE).

    `end_utc` is the EXCLUSIVE end of the run -- the end of its last unavailable
    bin -- so a gap and the next available bin never claim the same instant.

    A run is broken by two things, and the second is load-bearing:

    * an available bin, and
    * a **discontinuity in the bin grid itself** (`bins[i][0] != bins[i-1][1]`).
      `season_bins_utc` is dense within the season and absent outside it, so the
      list jumps from 30 Sep straight to 1 May. Walking by index alone would
      report a single seven-month gap spanning a winter that was never measured,
      and that fabricated span is what Planet quota would then be steered by.
    """
    bins = list(bins)
    available = list(available)
    if len(available) != len(bins):
        raise ValueError(
            f"available has {len(available)} flags for {len(bins)} bins; they must be parallel"
        )

    gaps = []
    run_start_index = None
    for index, (bin_start, bin_end) in enumerate(bins):
        _require_utc(bin_start, f"bins[{index}] start")
        _require_utc(bin_end, f"bins[{index}] end")
        if bin_end <= bin_start:
            raise ValueError(
                f"bins[{index}] is empty or reversed: {bin_start.isoformat()} -> "
                f"{bin_end.isoformat()}"
            )
        contiguous = index > 0 and bin_start == bins[index - 1][1]
        if run_start_index is not None and not (contiguous and not available[index]):
            gaps.append((run_start_index, index))
            run_start_index = None
        if run_start_index is None and not available[index]:
            run_start_index = index
    if run_start_index is not None:
        gaps.append((run_start_index, len(bins)))

    summary = []
    for lo, hi in gaps:
        gap_start = bins[lo][0]
        gap_end = bins[hi - 1][1]  # EXCLUSIVE end of the last unavailable bin
        if (gap_end - gap_start).total_seconds() >= min_seconds:
            summary.append((gap_start, gap_end, hi - lo))
    return summary


def mean_availability_by_utc_hour(bins, available):
    """Mean availability per UTC hour: exactly 24 floats, indexed by hour of bin START.

    An hour with NO bins at all is `float('nan')`, never `0.0`. The two mean
    different things -- "not measured" versus "measured, nothing available" --
    and rendering the first as the second draws a fake outage into the plot
    (invariant 9).

    What the shape means: a continuously-recording hydrophone has no preferred
    hour of the day, so this profile should be FLAT. A dent exactly seven hours
    wide is a UTC/America-Vancouver (PDT, UTC-7) mix-up announcing itself, not a
    property of the ocean.
    """
    bins = list(bins)
    available = list(available)
    if len(available) != len(bins):
        raise ValueError(
            f"available has {len(available)} flags for {len(bins)} bins; they must be parallel"
        )

    totals = [0] * HOURS_PER_DAY
    counts = [0] * HOURS_PER_DAY
    for index, (bin_start, _bin_end) in enumerate(bins):
        _require_utc(bin_start, f"bins[{index}] start")
        hour = bin_start.hour  # UTC hour: bin_start is tz-aware UTC by the check above
        counts[hour] += 1
        if available[index]:
            totals[hour] += 1
    return [
        (totals[hour] / counts[hour]) if counts[hour] else float("nan")
        for hour in range(HOURS_PER_DAY)
    ]


# ---------------------------------------------------------------------------
# B3-B: resumable, content-hashed download primitive
#
# Extends A1's listing with the pull. `download_archive_file()` never talks to
# the real network itself -- it is handed a `transport` object exposing
# `.get(filename, range_start=0) -> response` with `.status_code`, `.headers`
# and `.iter_bytes(chunk_size)`, so the seam is testable without touching ONC
# and so a caller can swap in a `requests`-backed transport that hits
# `archivefile/download?filename=...&token=...` (the live ONC archive-file
# retrieval path -- see the B3-B retrieval-path finding below).
#
# THE HASH GUARANTEE ACTUALLY IN FORCE: ONC's `onc` client (checked live in
# `_OncArchive.downloadArchivefile`/`saveAsFile`, 2026-08-27) streams the
# response straight to disk with no server-side checksum header consulted or
# even present in a plain `requests.get` -- no ETag, no Content-MD5. So the
# guarantee here is a LOCAL one: a sha256 computed once over the completed
# file, kept in a `.sha256` sidecar beside it, and re-verified before a cache
# hit is trusted. That catches local corruption (bit rot, a hand-edited
# fixture) but cannot catch a server response that was truncated and then
# declared complete without any independent length/hash to check it against --
# stated, not silently assumed.
# ---------------------------------------------------------------------------

# Backoff schedule for HTTP 429/5xx. Exponential with jitter STRICTLY bounded
# below half the gap to the next step, so the recorded sleeps are provably
# strictly increasing regardless of the random draw (check_b3b_5 asserts
# exactly this on the recorded, not wall-clock, values).
_BACKOFF_BASE_SECONDS = 1.0
_BACKOFF_MULTIPLIER = 2.0
_BACKOFF_JITTER_FRACTION = 0.1  # jitter in [0, base * fraction); fraction < 1 keeps it monotone

# One archive file is at most a few hundred MB (a 300 s FFT/WAV chunk), so this
# many attempts is generous headroom for real 429/5xx churn while still being a
# hard stop: an implementation that can never complete (e.g. a connection that
# is interrupted on literally every attempt, check_b3b_1's fixture) must raise
# rather than retry forever.
_MAX_TRANSFER_ATTEMPTS = 8

# HTTP statuses treated as transient and worth a backed-off retry.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Local-hash streaming chunk size: large enough to amortise per-chunk overhead
# on a multi-hundred-MB file, small enough not to hold a big buffer in memory.
_HASH_CHUNK_BYTES = 65536

# Network read chunk size used when the caller does not name one. A PRODUCTION
# value: 64 KiB is what `requests`-based streaming code normally reads, and a
# real ONC archive file is a few hundred MB, so a smaller chunk buys nothing on
# the wire and costs a syscall per KiB.
#
# It is a DEFAULT, not a constant baked into the loop, because the resume checks
# need a small chunk to be meaningful: `iter_bytes()` can only be interrupted
# between chunks, so at 65536 a multi-KB fixture body arrives in ONE chunk, the
# `.part` file is already complete when the "resumed" attempt starts, and the
# resume checks pass while exercising nothing. Those checks pass `chunk_bytes`
# explicitly and small; production gets the real value.
_DEFAULT_DOWNLOAD_CHUNK_BYTES = 65536

_PART_SUFFIX = ".part"
_SHA256_SUFFIX = ".sha256"


class DownloadError(RuntimeError):
    """B3-B's own download failed for a reason that is not a measured zero.

    Distinct from `ONCListingError` because a failed pull and a failed listing
    are different findings even though both come from `_client_call`-adjacent
    code (invariant 9); this one names the file, not a location/span.
    """


class ArchiveTransportError(OSError):
    """A transport-level network failure, with any token value REDACTED.

    An `OSError` subclass on purpose: `download_archive_file`'s retry loop
    already treats `OSError`/`ConnectionError` as the transient, resumable
    class of failure, so a connect-time drop raised as this type is retried
    exactly like a mid-stream one rather than terminating the run.

    It exists because `requests`/`urllib3` embed the FULL request URL --
    `...archivefile/download?token=<the real token>&filename=...` -- verbatim
    in the text of a connection error. Raising that unmodified writes the
    credential into a notebook cell and into every traceback (CLAUDE.md
    invariant 7). `RequestsArchiveTransport.get()` therefore builds the
    message through `_redact()` (A1b's redactor, the one
    `check_a1b_network_live_401_is_redacted` already covers) and raises it
    OUTSIDE the `except` block, so no token-bearing original survives on
    `__cause__` or `__context__` either.
    """


class DownloadRecord:
    """Outcome of one `download_archive_file()` call.

    `status` is exactly one of "downloaded", "cached", "absent",
    "measured_zero" -- segment C's absent-file log consumes `status` and
    `filename` without re-parsing an exception string. `path` is the Path to
    the completed file on `status in ("downloaded", "cached")`, and `None`
    otherwise: nothing is ever left at the final path for an absent or
    measured-zero file (D7/D8, extended to the pull).

    `http_status` is the LAST HTTP status code this call observed from
    `transport.get()` -- the status that produced the terminal outcome (200 on
    a completed download, 404 on "absent", 400 on "measured_zero"). For a
    "cached" record no HTTP request was made at all, so `http_status` is
    `None` -- a fabricated 200 here would claim a network round trip that
    never happened (invariant 5). `attempts` is the number of
    `transport.get()` calls this invocation issued, counting retries after a
    429/5xx backoff and after an interrupted transfer; it is `0` for
    "cached", since again no request was issued. Both fields are the B3-C
    contract's "HTTP status, attempt count per file" -- carried in the
    manifest so a slow overnight run is diagnosable as slow-throughput versus
    slow-from-retries after the fact, without re-deriving it from logs.


    `total_size_known` records whether the transfer's expected TOTAL length was
    stated by the server (`Content-Range` on a 206, `Content-Length` on a 200)
    and could therefore be checked against what landed on disk. It is `True`
    for a verified download and for a "cached" record (whose sha256 was
    re-verified against the sidecar), and `None` for "absent"/"measured_zero",
    where no body was transferred at all. A downloaded file NEVER carries
    `False`: an unverifiable transfer is not promoted (see
    `download_archive_file`).
    """

    __slots__ = (
        "filename", "status", "path", "bytes_downloaded", "sha256", "message",
        "http_status", "attempts", "total_size_known",
    )

    def __init__(self, filename, status, path=None, bytes_downloaded=0, sha256=None,
                 message=None, http_status=None, attempts=0, total_size_known=None):
        self.filename = filename
        self.status = status
        self.path = path
        self.bytes_downloaded = bytes_downloaded
        self.sha256 = sha256
        self.message = message
        self.http_status = http_status
        self.attempts = attempts
        self.total_size_known = total_size_known

    def __repr__(self):
        return (
            f"DownloadRecord(filename={self.filename!r}, status={self.status!r}, "
            f"path={self.path!r}, bytes_downloaded={self.bytes_downloaded!r}, "
            f"http_status={self.http_status!r}, attempts={self.attempts!r}, "
            f"total_size_known={self.total_size_known!r})"
        )


def _sha256_of_file(path: Path) -> str:
    """Local sha256 of a file on disk, streamed -- never loaded whole into memory."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _backoff_sleep_seconds(attempt_index: int) -> float:
    """Sleep duration for the `attempt_index`-th (0-based) backed-off retry.

    Doubling with jitter capped at `_BACKOFF_JITTER_FRACTION` of the base keeps
    successive sleeps strictly increasing: attempt N+1's minimum (its base) is
    always greater than attempt N's maximum (its base * (1 + jitter fraction))
    because the multiplier (2.0) exceeds (1 + jitter fraction) (1.1).
    """
    base = _BACKOFF_BASE_SECONDS * (_BACKOFF_MULTIPLIER ** attempt_index)
    jitter = random.uniform(0.0, base * _BACKOFF_JITTER_FRACTION)
    return base + jitter


def download_archive_file(client, filename, *, dest_dir, transport,
                           sleep=time.sleep, expected_sha256=None,
                           chunk_bytes=_DEFAULT_DOWNLOAD_CHUNK_BYTES):
    """Pull ONE ONC archive file into `dest_dir`, resumably and content-hashed.

    `client` is accepted for interface symmetry with the listing functions
    (and so a future real transport can be built from it), but is never called
    directly here -- every byte and every status code comes from `transport`,
    the injected seam (`transport.get(filename, range_start=0)`).

    Conventions:

    * **Landing zone**: `dest_dir` is created (parents included) via
      `boatphone.paths.ensure_dir()` if absent -- the ONE place a directory
      is created as a side effect of calling this, never of importing this
      module (decision 0005).
    * **Atomicity**: bytes land in `<filename>.part` and are renamed to the
      final path only once the transfer is verified complete and hashed.
      `Path.rename` is atomic on the same filesystem, so a file observed at
      the final path is, by construction, complete (decision 0005 §1).
    * **Resume**: if a `.part` sidecar already exists, the next attempt asks
      `transport.get(filename, range_start=<bytes already on disk>)`. If the
      server answers 206 the tail is APPENDED to what is already on disk. If
      it answers a bare 200 the Range header was ignored and the full body is
      coming, so the partial is DISCARDED and the file is written from byte
      zero -- appending there would silently double the file. So resume avoids
      re-transferring bytes whenever the server honours Range, and falls back
      to a restart, loudly, when it does not.
    * **`chunk_bytes`**: bytes per `iter_bytes()` read, default
      `_DEFAULT_DOWNLOAD_CHUNK_BYTES` (65536). A caller exercising resume
      against a small fixture body must pass a small value explicitly, or the
      whole body arrives in one chunk and there is no partial state to resume
      from.
    * **Cache**: if the final path already exists, its LOCAL sha256 is
      recomputed and checked against the `.sha256` sidecar written at
      download time (or `expected_sha256` if the caller supplied one and no
      sidecar exists yet). A match is a zero-byte-transferred "cached" record.
      A mismatch RAISES `DownloadError` -- it is never silently overwritten
      (CLAUDE.md invariant 2).
    * **Backoff**: HTTP 429/5xx sleep via the injected `sleep` callable with
      strictly increasing exponential-with-jitter durations
      (`_backoff_sleep_seconds`), then retry the SAME `range_start`.
    * **Measured zeros (D7)**: a 400 whose message matches
      `_NO_DATA_POSSIBLE_MARKERS` (the identical strings A1b/A1e already key
      on for listing) returns a "measured_zero" record and does not raise --
      no instrument was deployed, or the window has not elapsed, so there is
      nothing to fetch. Nothing is written at the final path.
    * **Absent (a real 404)**: returns an "absent" record and does not raise
      -- segment C's absent-file log is the stated consumer. Nothing is
      written at the final path.
    * **Anything else** (an unexpected status code, a transfer that never
      completes within `_MAX_TRANSFER_ATTEMPTS`, a hash mismatch on a freshly
      completed transfer) raises `DownloadError`. No bare `except`.

    **What "complete" is actually verified against, stated honestly.** The only
    independent statement of length available is the server's own
    `Content-Range`/`Content-Length`, surfaced as `response.total_size`. When it
    is present, a body shorter than it is treated as an interrupted transfer and
    retried. When it is ABSENT -- which is exactly what
    `Transfer-Encoding: chunked` gives, and this transport streams -- there is
    nothing to check the body against, so a truncated response would otherwise be
    hashed, sidecar'd, renamed to the final path and reported "downloaded", and
    every later run would take a clean CACHE HIT on the truncated file: the local
    sha256 would have laundered an unverifiable file into a verified-looking one.
    That is not done. A transfer whose total size the server never stated is NOT
    promoted; it raises `DownloadError` naming the missing headers and leaves the
    bytes in the `.part` file. The guarantee in force is therefore: a file at the
    final path was as long as the server said it would be, and its sha256 matches
    its sidecar. It is NOT a guarantee that the server sent the right bytes --
    ONC serves no content checksum (checked 2026-08-27) -- and it never covers a
    response that stated no length at all.
    """
    dest_dir = ensure_dir(dest_dir)
    final_path = dest_dir / filename
    part_path = dest_dir / f"{filename}{_PART_SUFFIX}"
    sha_path = dest_dir / f"{filename}{_SHA256_SUFFIX}"

    if final_path.exists():
        on_disk_sha = _sha256_of_file(final_path)
        recorded_sha = None
        if sha_path.exists():
            recorded_sha = sha_path.read_text(encoding="utf-8").strip()
        elif expected_sha256 is not None:
            recorded_sha = expected_sha256

        if recorded_sha is not None and on_disk_sha != recorded_sha:
            raise DownloadError(
                f"{final_path} exists but its local sha256 ({on_disk_sha}) does not match "
                f"the recorded hash ({recorded_sha}); a cached file must never be silently "
                "overwritten (CLAUDE.md invariant 2) -- remove or investigate it by hand"
            )
        if expected_sha256 is not None and on_disk_sha != expected_sha256:
            raise DownloadError(
                f"{final_path} exists but its local sha256 ({on_disk_sha}) does not match "
                f"the caller-supplied expected_sha256 ({expected_sha256})"
            )
        if not sha_path.exists():
            sha_path.write_text(on_disk_sha, encoding="utf-8")
        return DownloadRecord(
            filename=filename, status="cached", path=final_path,
            bytes_downloaded=0, sha256=on_disk_sha, http_status=None, attempts=0,
            total_size_known=True,
        )

    attempt = 0
    # Separate from `attempt`: the index into the backoff schedule, advanced by
    # EVERY slept retry whatever caused it (429/5xx, a dropped connection, a
    # short body). Sharing one increasing schedule is what stops a server that
    # keeps returning short bodies from being hammered `_MAX_TRANSFER_ATTEMPTS`
    # times with no delay at all, which is what the pre-B3 code did.
    backoff_index = 0
    while True:
        attempt += 1
        if attempt > _MAX_TRANSFER_ATTEMPTS:
            raise DownloadError(
                f"{filename}: gave up after {_MAX_TRANSFER_ATTEMPTS} attempt(s) "
                f"(_MAX_TRANSFER_ATTEMPTS) without completing the transfer; the connection "
                "was interrupted on every attempt or the server never returned a usable "
                "response. A partial file, if any, remains at "
                f"{part_path} for a future resume"
            )
        range_start = part_path.stat().st_size if part_path.exists() else 0
        # INSIDE the retried region. A ConnectionError/Timeout at CONNECT time
        # is the single most common failure of a multi-thousand-file overnight
        # pull; with this call above the `try` it was never caught, never
        # retried, and killed the process. The message is redacted before it is
        # printed because `requests` puts the full token-bearing request URL in
        # its own error text (invariant 7) -- and the transport itself raises
        # `ArchiveTransportError` already redacted.
        connect_failure = None
        try:
            response = transport.get(filename, range_start=range_start)
        except (OSError, ConnectionError) as exc:
            connect_failure = f"{type(exc).__name__}: {_redact(str(exc))}"
        if connect_failure is not None:
            # Built inside the except, raised/handled outside it, like
            # `_client_call`: chaining would re-attach the original (possibly
            # token-bearing) exception on __context__.
            sleep_seconds = _backoff_sleep_seconds(backoff_index)
            backoff_index += 1
            print(
                f"download_archive_file: {filename} could not be requested at "
                f"range_start={range_start} ({connect_failure}); attempt {attempt} of "
                f"{_MAX_TRANSFER_ATTEMPTS}, sleeping {sleep_seconds:.2f}s before retrying"
            )
            sleep(sleep_seconds)
            continue
        status_code = response.status_code

        if status_code == 404:
            return DownloadRecord(
                filename=filename, status="absent", path=None,
                http_status=status_code, attempts=attempt,
            )

        if status_code == 400:
            message = getattr(response, "error_text", "") or ""
            if any(marker in message for marker in _NO_DATA_POSSIBLE_MARKERS):
                print(
                    f"download_archive_file: ONC states NO data can exist for {filename} "
                    f"(no device deployed in the window, or the window has not elapsed); "
                    f"recorded as a MEASURED ZERO, not an error. ONC said: {message}"
                )
                return DownloadRecord(
                    filename=filename, status="measured_zero", path=None,
                    http_status=status_code, attempts=attempt,
                )
            raise DownloadError(f"{filename}: request failed with 400: {message}")

        if status_code in _RETRYABLE_STATUS_CODES:
            sleep_seconds = _backoff_sleep_seconds(backoff_index)
            backoff_index += 1
            sleep(sleep_seconds)
            continue

        # 200 vs 206 on a RESUMED request (range_start > 0): whether ONC's
        # archivefile/download endpoint honours a Range header at all is
        # UNVERIFIED -- ONC's own `onc` package client never sends one. 206
        # is the server accepting the resume and returning only the tail; a
        # bare 200 here means the Range header was ignored and the FULL body
        # is about to arrive, so appending it to the existing .part would
        # silently double the file (invariant 5's "silent corruption" class).
        # The only safe move on a 200-instead-of-206 is to DISCARD the
        # partial and write from byte zero, and to say so loudly so the
        # first real run is the thing that resolves this, not a guess here.
        if status_code == 200 and range_start > 0:
            print(
                f"download_archive_file: {filename} resume requested Range: "
                f"bytes={range_start}- but the server returned 200 (not 206), meaning "
                "the Range header was ignored and the full body is being sent; ONC "
                "Range-resume support is UNVERIFIED (checked live 2026-08-27 only against "
                "source -- see boatphone/onc_client.py module docstring). Discarding the "
                f"existing {part_path.stat().st_size if part_path.exists() else 0} byte(s) "
                "of partial content and writing from byte zero rather than appending, "
                "which would silently corrupt the file"
            )
            range_start = 0
        elif status_code == 206 and range_start > 0:
            print(
                f"download_archive_file: {filename} resume Range: bytes={range_start}- "
                "was honoured (206 Partial Content) -- ONC's archivefile/download endpoint "
                "DOES support Range resume for this file"
            )
        elif status_code not in (200, 206):
            message = getattr(response, "error_text", "") or ""
            raise DownloadError(
                f"{filename}: unexpected HTTP status {status_code} at range_start="
                f"{range_start}: {message}"
            )

        mode = "ab" if range_start > 0 else "wb"
        interrupted = None
        retained = 0
        try:
            with open(part_path, mode) as handle:
                for chunk in response.iter_bytes(chunk_size=chunk_bytes):
                    handle.write(chunk)
        except (OSError, ConnectionError) as exc:
            # A dropped connection mid-stream: the bytes already written stay on
            # disk under the .part sidecar, and the loop retries with a resumed
            # range_start. Not a bare except -- narrowly the two error types a
            # broken transport actually raises.
            # `_redact`, not str(exc): a transport error carries the request URL,
            # and that URL carries the token (invariant 7).
            interrupted = f"{type(exc).__name__}: {_redact(str(exc))}"
            retained = part_path.stat().st_size if part_path.exists() else 0
        if interrupted is not None:
            sleep_seconds = _backoff_sleep_seconds(backoff_index)
            backoff_index += 1
            print(
                f"download_archive_file: {filename} interrupted mid-transfer "
                f"({interrupted}); {retained} byte(s) retained, sleeping "
                f"{sleep_seconds:.2f}s then resuming on the next attempt"
            )
            sleep(sleep_seconds)
            continue

        expected_total = getattr(response, "total_size", None)
        got_size = part_path.stat().st_size
        if expected_total is None:
            # NOT promoted. Nothing states how long this body should have been
            # (no Content-Range, no Content-Length -- what chunked transfer
            # encoding gives), so "complete" is unverifiable and hashing it here
            # would only make a possibly-truncated file look verified forever
            # after, via the sha256 sidecar and the cache hit it earns.
            raise DownloadError(
                f"{filename}: the response carried no total size (neither Content-Range "
                f"nor a usable Content-Length -- e.g. Transfer-Encoding: chunked), so the "
                f"{got_size} byte(s) received cannot be checked for truncation. Refusing to "
                f"promote an unverifiable transfer to {final_path}: the local sha256 would "
                "then present a possibly-truncated file as verified on every later run. The "
                f"bytes are left at {part_path}"
            )
        if got_size < expected_total:
            # The response ended without raising but did not deliver everything
            # it advertised -- treated the same as an interrupted transfer:
            # resume from what is on disk rather than trusting a short body.
            # It SLEEPS: a server returning short bodies would otherwise be
            # retried `_MAX_TRANSFER_ATTEMPTS` times back to back.
            sleep_seconds = _backoff_sleep_seconds(backoff_index)
            backoff_index += 1
            print(
                f"download_archive_file: {filename} received {got_size} of "
                f"{expected_total} advertised byte(s) without an error; treating it as a "
                f"short body, sleeping {sleep_seconds:.2f}s then resuming"
            )
            sleep(sleep_seconds)
            continue

        final_sha = _sha256_of_file(part_path)
        if expected_sha256 is not None and final_sha != expected_sha256:
            raise DownloadError(
                f"{filename}: freshly downloaded content's sha256 ({final_sha}) does not "
                f"match the caller-supplied expected_sha256 ({expected_sha256}); the .part "
                f"file is left at {part_path} for inspection rather than promoted"
            )
        sha_path.write_text(final_sha, encoding="utf-8")
        part_path.rename(final_path)
        return DownloadRecord(
            filename=filename, status="downloaded", path=final_path,
            bytes_downloaded=got_size, sha256=final_sha,
            http_status=status_code, attempts=attempt, total_size_known=True,
        )


# ---------------------------------------------------------------------------
# Real HTTP transport for download_archive_file() -- the seam B3-B built and
# proved only against `_B3BFakeTransport` (scripts/checks.py). Everything
# above this line is unaware this class exists; `download_archive_file` only
# ever calls `transport.get(filename, range_start=0)`.
#
# Retrieval path, established from the INSTALLED `onc` package source
# (`onc/modules/_OncArchive.py::downloadArchivefile`,
# `onc/modules/_OncService.py`), not re-derived: a plain authenticated GET on
# `{baseUrl}api/archivefile/download` with query params
# `{"token": ..., "filename": ...}`. There is no order queue and no async job
# poll for an archive file -- that machinery (`_OncDelivery`) exists only for
# derived data products, which this project does not use here.
#
# Range resume is UNVERIFIED against the live API: the `onc` package's own
# client never sends a Range header (confirmed by reading
# `downloadArchivefile` above -- `requests.get(url, filters, ...)`, no
# `headers=`), so whether `archivefile/download` honours one is genuinely
# unknown until a real resumed request is made. This transport therefore:
#
#   * sends `Range: bytes=<range_start>-` whenever `range_start > 0`;
#   * treats a `206 Partial Content` response as the server honouring the
#     resume -- exposed to `download_archive_file` as `status_code=206`,
#     which that function's own 200-vs-206 handling appends;
#   * treats a `200 OK` response to a ranged request as the server IGNORING
#     the header and sending the full body from byte zero -- exposed as
#     `status_code=200`, which `download_archive_file` handles by discarding
#     the partial `.part` content and writing from zero, never appending
#     (appending a full body onto a partial file is silent corruption);
#   * anything else propagates as whatever status code it is, and
#     `download_archive_file` raises `DownloadError` for a status it does not
#     otherwise recognise.
#
# Nothing here decides which of 200/206 ONC actually returns -- that is
# unknown until this code is run for real. It only makes sure BOTH answers are
# handled correctly rather than assumed.
# ---------------------------------------------------------------------------


class _RequestsArchiveResponse:
    """Adapts a streamed `requests.Response` to the seam `download_archive_file` expects:
    `.status_code`, `.error_text`, `.total_size`, `.iter_bytes(chunk_size)`.

    `total_size` is the size of the COMPLETE file, never of this response's
    body:

    * on a **206** it comes ONLY from `Content-Range: bytes start-end/total`.
      There is deliberately no `Content-Length` fallback here: on a 206
      `Content-Length` is the length of the TAIL being sent, and returning it
      as the whole-file size makes `download_archive_file`'s
      `got_size < expected_total` test compare a whole file against a tail
      length, which passes trivially and would wave a truncated resume
      through. A 206 without a parseable `Content-Range` yields `None`;
    * on anything else it is `Content-Length`, which for a 200 serving the
      whole file IS the total.

    `None` means "the server stated no total size". `download_archive_file`
    refuses to PROMOTE such a transfer rather than treating it as complete --
    an unverifiable body must not be laundered into a hash-verified file.
    """

    __slots__ = ("_response", "status_code", "error_text", "total_size", "headers")

    def __init__(self, response):
        self._response = response
        self.status_code = response.status_code
        self.headers = response.headers
        self.total_size = self._parse_total_size(response)
        self.error_text = "" if response.ok else self._error_text(response)

    @staticmethod
    def _parse_total_size(response):
        content_range = response.headers.get("Content-Range")
        if content_range:
            # "bytes 1024-2047/4096" -- the total is after the final "/".
            tail = content_range.rsplit("/", 1)[-1].strip()
            if tail.isdigit():
                return int(tail)
        if response.status_code == 206:
            # A partial response whose Content-Range is missing or unparseable
            # ("*/*" for an unknown total, or absent entirely). Content-Length
            # here measures the TAIL, so falling through to it would report a
            # total smaller than the file and make the truncation check vacuous
            # (see the class docstring). Unknown is the honest answer.
            return None
        content_length = response.headers.get("Content-Length")
        if content_length is not None and content_length.isdigit():
            return int(content_length)
        return None

    @staticmethod
    def _error_text(response):
        """Best-effort error text. Never raises; a malformed error body must not
        mask the status code that caused it (invariant 5)."""
        try:
            payload = response.json()
        except ValueError:
            return (response.text or "")[:2000]
        if isinstance(payload, dict) and "errors" in payload:
            try:
                return "; ".join(
                    f"API Error {e.get('errorCode')}: {e.get('errorMessage')} "
                    f"(parameter: {e.get('parameter')})"
                    for e in payload["errors"]
                )
            except (KeyError, TypeError, AttributeError):
                return str(payload)[:2000]
        return str(payload)[:2000]

    def iter_bytes(self, chunk_size=65536):
        return self._response.iter_content(chunk_size=chunk_size)


class RequestsArchiveTransport:
    """Real network transport for `download_archive_file`, built on `requests`.

    `client` is an already-constructed `onc.ONC` instance
    (`boatphone.credentials.get_onc_client()`) -- this class reads `.baseUrl`,
    `.token`, `.timeout` off it. Constructing this object makes no network
    call; only `.get()` does.

    **The token never reaches an output.** That is a property of the code, not
    a hope: the token goes in the query string, so `requests`/`urllib3` put it
    in the text of every connection error they raise --

        HTTPSConnectionPool(host='data.oceannetworks.ca', port=443): Max
        retries exceeded with url: /api/archivefile/download?token=<the real
        token>&filename=... (Caused by ...)

    -- and that message otherwise lands in a traceback and a notebook cell
    (CLAUDE.md invariant 7). So every network failure in `.get()` is caught,
    passed through `_redact()` (A1b's redactor, already covered by
    `check_a1b_network_live_401_is_redacted`), and re-raised as
    `ArchiveTransportError` from OUTSIDE the `except` block, so the
    token-bearing original is not reachable via `__cause__` or `__context__`
    either. Verified live 2026-08-27 with a deliberately bogus token against
    an unresolvable host: the redacted form is what appears.

    `.get(filename, range_start=0)` issues exactly one HTTP GET per call (no
    retry, no backoff -- that lives in `download_archive_file`) and returns a
    `_RequestsArchiveResponse`. A `Range: bytes=<range_start>-` header is
    attached whenever `range_start > 0`; see the module-level comment above
    for exactly what 200 vs 206 in the response then means.
    """

    __slots__ = ("_base_url", "_token", "_timeout")

    def __init__(self, client):
        base_url = getattr(client, "baseUrl", None)
        token = getattr(client, "token", None)
        if not base_url or not token:
            raise ValueError(
                "RequestsArchiveTransport requires an onc.ONC client exposing non-empty "
                "baseUrl and token attributes; got "
                f"baseUrl={base_url!r} token={'<redacted>' if token else token!r}"
            )
        self._base_url = base_url
        self._token = token
        self._timeout = getattr(client, "timeout", 60)

    @property
    def base_url(self):
        """The ONC deployment's base URL, for provenance. NEVER the token.

        Public because a manifest must be able to state WHICH ONC deployment
        served the bytes; the token stays private and is never exposed by any
        accessor on this class (a token leak was caught on this branch once).
        """
        return self._base_url

    def get(self, filename, range_start=0):
        import requests  # imported lazily: boatphone must import without third-party deps

        url = f"{self._base_url}api/archivefile/download"
        params = {"token": self._token, "filename": filename}
        headers = {"Range": f"bytes={range_start}-"} if range_start > 0 else None
        failure = None
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=self._timeout, stream=True,
            )
            adapted = _RequestsArchiveResponse(response)
        except (OSError, ValueError) as exc:
            # OSError covers requests' own RequestException tree (ConnectionError,
            # Timeout, and the urllib3 wrappers underneath), which is where the
            # token-bearing URL shows up. Narrow and typed, never bare.
            # `_redact` is applied to the WHOLE message, and the message is
            # raised below, outside this block.
            failure = (
                f"GET {_redact(url)} (filename={filename!r}, range_start={range_start}) "
                f"failed: {type(exc).__name__}: {_redact(str(exc))}"
            )
        if failure is not None:
            raise ArchiveTransportError(failure)
        return adapted
