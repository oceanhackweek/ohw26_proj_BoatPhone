"""Shared project constants -- one definition, for ALL workstreams (acoustics,
Planet acquisition, optical detection), not an A1-only module.

If you need the study window, the season months, or the device/location, import
them from here. A second definition in a notebook or in `scripts/config.py` makes
the optical-acoustic matchup join two different windows and produce a wrong answer
rather than an error (CLAUDE.md invariant 6).

Conventions pinned here (decision 0002, decisions D1/D2/D4 of the A1 plan):

* **Time base**: UTC end to end. Every `datetime` exported by this module is
  tz-aware with a zero UTC offset. A naive datetime is never a valid value.
* **Bin grid**: fixed-width `BIN_SECONDS` bins, half-open `[start, end)`, with
  edges at integer multiples of `BIN_SECONDS` seconds since the UTC epoch.
* **Season**: `SEASON_MONTHS_UTC` is evaluated on the **UTC** month of the bin
  start. No local-timezone conversion happens anywhere in A1.

Names carry their frame: `*_UTC` means tz-aware UTC, `*_SECONDS` means seconds.
"""

from datetime import datetime, timezone

# Bin width for the uptime calendar. Source: A1 decision D2 -- 300 s (5 min) is
# the ONC FFT product's natural file cadence, so a bin maps 1:1 onto a listing.
BIN_SECONDS = 300

# Start of the study window. Source: the validity date of the ICLISTEN HF1266
# pre-deployment calibration file shipped with the Folger Deep sample. Earlier
# Folger deployments are *different devices* with different sensitivities, so
# nothing before this date is comparable in calibrated units.
STUDY_START_UTC = datetime(2020, 2, 18, 0, 0, 0, tzinfo=timezone.utc)

# End of the study window. CHOSEN BOUND, NOT MEASURED: the planner did not pin a
# value, so this is the end of the 2026 field season (2026-10-01T00:00:00Z,
# exclusive). It is an analysis convention -- revise it here, not per notebook,
# if the deployment record extends further.
STUDY_END_UTC = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)

# Months included in the "season", evaluated on the UTC month of the bin start.
# Source: A1 decision D4 -- May through September, the recreational-vessel
# season in Barkley Sound. Deliberately UTC, not America/Vancouver: a local
# definition would make the season edge depend on DST (see decision 0002).
SEASON_MONTHS_UTC = (5, 6, 7, 8, 9)

# The hydrophone. Source: ONC metadata for the Folger Deep ICLISTEN HF1266
# instrument (device code as used by the ONC API; numeric device id alongside).
DEVICE_CODE = "ICLISTENHF1266"
DEVICE_ID = 23235

# ONC data-product extension used for uptime listing. Source: the gzipped FFT
# product present in the local sample acquisition.
PRODUCT_EXTENSION = "fft.gz"

# Extension used by ONC's *archive-file index* for the FFT product. NOT the same
# string as PRODUCT_EXTENSION: the archive index registers these files as "fft"
# (e.g. ICLISTENHF1266_20240715T000136.000Z.fft), and they arrive gzipped on disk
# as ".fft.gz" -- which is what the local sample acquisition shows. Source:
# measured 2026-08-27 against the live ONC archivefile endpoint for FGPD/
# HYDROPHONE -- a listing filtered on "fft.gz" returns ZERO files for a day that
# has 289 files under "fft".
ARCHIVE_EXTENSION = "fft"

# ONC device-category code for the hydrophone, used to scope an archive listing
# at a location. Source: ONC deployment metadata for DEVICE_CODE
# (deviceCategoryCode == "HYDROPHONE").
DEVICE_CATEGORY_CODE = "HYDROPHONE"

# Case-insensitive fragment that identifies the Folger sites in an ONC location
# NAME. Discovery matches on this, never on a location code, so the code is
# whatever ONC returns at runtime. Source: ONC location name "Folger Deep".
FOLGER_NAME_FRAGMENT = "folger"

# Nominal duration of one archive FFT file, in seconds. MEASURED, not assumed:
# 2011 unique .fft files over 2024-07-12..18 at Folger Deep give consecutive
# start deltas of 300 s for 1909 of 2010 gaps, 297-302 s for 98 more (sub-second
# start jitter, files are NOT bin-aligned), and 2 genuine outage gaps (637 s,
# 1780 s). Equal to BIN_SECONDS by measurement, kept as its own name because it
# is a property of the instrument's file cadence, not of the calendar's grid.
FFT_FILE_SECONDS = 300

# ---------------------------------------------------------------------------
# ONC archive-listing response cap (the A1d defect).
# ---------------------------------------------------------------------------
# ONC's archivefile listing endpoint returns ONE PAGE and truncates silently:
# the `files` list simply stops partway through the requested span, with no
# error and no flag inside the list itself. The only signal is the response's
# top-level `next` field.
#
# MEASURED 2026-08-27 against the live endpoint (FGPD / HYDROPHONE / "fft"):
#   2024-05-01..2024-10-01  -> 11,121 rows, stopping at 2024-06-08T15:10Z,
#                              `next` = {"parameters": {..., "page": "2"}}
#   2024-05-01..2024-07-01  -> 11,121 rows, same truncation point
#   2024-05-01..2024-12-01  -> 11,121 rows, same truncation point
#   2020-01-01..2021-01-01  ->  9,977 rows, `next` present
#   2024-07-01..2024-08-01  ->  8,922 rows, `next` = None (complete)
# Paginating the 2024 season took 4 requests / 50.6 s for 44,027 unique files,
# with page sizes 11121, 11075, 11086, 10745.
#
# So the cap is NOT a fixed row count -- page sizes differ by hundreds of rows
# between pages of one query -- which is why this constant is named "observed
# max" and is used only for DIAGNOSTIC MESSAGES, never to decide whether a
# response was truncated. That decision is made from `next`, which is
# authoritative. Believing a row-count threshold instead would reintroduce the
# defect the moment ONC's page size moved.
ONC_LISTING_PAGE_ROWS_OBSERVED_MAX = 11121

# Hard stop on the page-following loop, so a server that ignores our paging
# parameter cannot spin forever. CHOSEN BOUND, NOT MEASURED: the 2020-2026 study
# window is ~44 k files per season, i.e. single-digit pages per calendar year, so
# 200 is orders of magnitude of headroom and still terminates. Exceeding it
# raises rather than returning a short list.
ONC_LISTING_MAX_PAGES = 200

# Floor for the adaptive time-subdivision fallback used when a response says it
# is truncated but advertises no usable paging parameter. A span this short holds
# one FFT file, so it cannot be subdivided further; still-truncated at the floor
# raises with the span named. Source: FFT_FILE_SECONDS above.
ONC_LISTING_MIN_SUBCHUNK_SECONDS = FFT_FILE_SECONDS
