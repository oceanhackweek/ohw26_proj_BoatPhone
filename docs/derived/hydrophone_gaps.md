# Hydrophone uptime and no-data spans -- Folger Deep (ICLISTENHF1266)

**Deliverable O1, human-readable half. For Malachy (Planet acquisition).**
The machine-readable half is `data/derived/hydrophone_uptime.csv`, which is gitignored and
therefore does not travel; this file does.

---

## The one line that matters

**These date ranges have no hydrophone data -- do not spend Planet quota on them:**

| Span (UTC, end EXCLUSIVE) | Length | What it is |
|---|---|---|
| `2026-05-01` -> `2026-10-01` | 153 days (the entire 2026 season) | ICLISTENHF1266's deployment ENDED `2026-03-14T21:37:20Z`. ONC reports no deployment of this device at Folger during the 2026 season. The archive request for the span returns ONC's *API Error 127: A device with category HYDROPHONE was deployed at location FGPD but not during the provided time range* -- under decision 0007 that is a **measured zero**, i.e. ONC positively stating the device was not deployed, not a failed request. There is nothing to cross-calibrate against. |
| `2023-08-27T22:55:00Z` -> `2023-09-06T15:10:00Z` | 9.68 days | No listed file. The largest in-season outage in the record. |
| `2023-09-22T09:15:00Z` -> `2023-09-26T18:10:00Z` | 4.37 days | No listed file. |
| `2023-09-07T11:30:00Z` -> `2023-09-11T17:10:00Z` | 4.24 days | No listed file. |
| `2022-08-07T07:55:00Z` -> `2022-08-10T16:10:00Z` | 3.34 days | No listed file. |

Every other in-season day from 2020 through 2025 has at least partial coverage. Outside these
spans the instrument is recording ~97-100 % of the time, so **any May-September date in
2020-2025 that is not in the table above is a safe date to order imagery for.**

Ends are **exclusive**: `2022-08-10T16:10:00Z` is the first instant data resumes.

**The edges above are quoted to the minute but are only accurate to +/- 300 s.** A `.fft`
listing entry carries a start timestamp only, so coverage is modelled as
`[start, start + FFT_FILE_SECONDS)` with `FFT_FILE_SECONDS = 300` (`config`), and real file
starts are off-grid (`:36`, `:04` seconds). Read each edge as "within five minutes of", and
do not use these timestamps for any alignment finer than a bin.

---

## Span this document actually covers

* **Seasons 2020, 2021, 2022, 2023, 2024, 2025, 2026** -- the May-September window
  (`config.SEASON_MONTHS_UTC`, months 5-9) of every year in the study window
  `2020-02-18` .. `2026-10-01`. That is the full seasonal calendar, not a sample.
* **Out of season (October-April) was NOT scanned** and is not claimed either way. The uptime
  calendar is seasonal by construction; an absence of rows for 2024-12-25 means "not measured",
  not "no data".
* Threshold for listing a gap above: **>= 1 day** contiguous.

### What "available" means, and why it is an UPPER BOUND

**A bin is marked available when at least one listed file overlaps it by any non-zero amount**
(`onc_client.mark_available`). Because each `.fft` listing entry is modelled as covering
`[start, start + 300 s)` from off-grid starts, a listed file routinely overhangs into the bins on
either side of it. So a bin can be marked available on the strength of a *neighbouring* file.

The consequence is structural, not a tuning choice: **files at `t`, `t + 300 s` and `t + 600 s`
with the middle one missing still leave every affected bin marked available.** A single-file
(5-minute) dropout is therefore invisible in this document and in the CSV, and so is any dropout
shorter than about two bins (~10 minutes).

Read the percentages accordingly: **availability here is an over-estimate, an upper bound on
uptime.** Short dropouts are not merely below the reporting threshold -- the coverage model
erases them, so they are **not** in the CSV either.

The bias fails in the direction that is safe for Planet quota: it never withholds an order it
should have placed. What it can do is recommend a date whose acoustic record is holey at the
minute scale. **A4's actual download is what will surface this**, and when A4 reports lower
uptime than the numbers below, that is the expected outcome of this model, not an A4 bug.
* Time base: **UTC throughout**, bins half-open `[start, end)`, 300 s wide
  (`config.BIN_SECONDS`), epoch-aligned. No local time appears anywhere in the pipeline.

## Available fraction per season

| Season (May 1 - Oct 1, UTC) | Bins | Available | Gaps >= 1 day |
|---|---|---|---|
| 2020 | 44,064 | **98.53 %** | none |
| 2021 | 44,064 | **99.56 %** | none |
| 2022 | 44,064 | **96.90 %** | 1 |
| 2023 | 44,064 | **87.83 %** | 3 |
| 2024 | 44,064 | **99.96 %** | none |
| 2025 | 44,064 | **99.54 %** | none |
| 2026 | not deployed (no request made) | n/a | n/a |

Across the six deployed seasons 2020-2025: **97.05 %** of in-season bins have a listed file.
2023 is the outlier and it is the season to be careful with.

---

## What this is, and what it is NOT

**This is derived from ONC listings -- it is not a completed download.** Every "available" flag
here means ONC's archive index reported a `.fft` file whose coverage overlaps that 5-minute bin.
That is ONC's *belief that a file exists*, not proof that the file downloads or that its contents
are usable (decision D3).

Consequences, stated plainly:

* A gap above could turn out to be a listing artefact, and the data may actually be retrievable.
* More dangerously, a span marked *available* here could still fail to download in **A4**.
* **A4's actual pull may revise this document, and where the two disagree the pull wins.** Treat
  the table above as a planning filter that avoids obviously wasted quota, not as a settled fact
  about the instrument.

Deployment boundaries come from ONC deployment **metadata** (`get_deployments`), never inferred
from a gap in the listing (D6). "No file listed" and "not deployed" are different claims and are
kept separate above: 2026 is the second, everything else in the table is the first.

## Sanity checks run against this scan

* **Diurnal null (the time-base check).** Mean availability by **UTC** hour is flat in every
  season -- the largest peak-to-trough spread across the 24 UTC hours is at most **2.0 percentage
  points** in every one of 2020-2025 (smallest 0.2 pts in 2024, largest 2.0 pts in 2021 and
  2023). A continuously-recording hydrophone has no preferred hour of day, so a
  **~7-hour-wide dent** would be an America/Vancouver (PDT, UTC-7) offset being read as UTC. No
  such dent is present, in any season.
  **Weak discriminator, stated plainly:** on a month or season that is ~100 % available, both the
  observed profile and the shifted null are nearly flat, so the comparison is a factor between two
  near-zero spreads (observed 0.0108 vs null 0.0430 on the notebook's month) rather than the clear
  separation the synthetic 7-day check produces (0.1428). This check can rule out a gross,
  span-wide timezone offset; it cannot certify a subtler time-base error, and it is least
  informative exactly where coverage is best. See
  `contributor_folders/isaac/a1_uptime_calendar.ipynb` for the figure.
* **The method found something, and it is not broken** (invariant 9). The 2023 gaps are real
  structure in the listing, not an empty response: the 2023 season still lists 38,686 files.

## Known defect this scan had to work around -- FIXED IN THE LIBRARY 2026-08-27

**ONC's archive listing silently truncates a request that spans a whole season.** Asking for
`2024-05-01` -> `2024-10-01` in one request returns 11,120 files ending `2024-06-08`, and the
resulting calendar reads as a four-month outage that does not exist -- July 2024 alone lists
8,921 files and is 99.96 % covered. `onc_client.list_fft_files()` chunked by calendar **year**,
which is coarser than the cap, so driving `build_uptime_calendar()` over a season span reproduced
this. **This document was built with one listing request per calendar UTC month** (~8,400-8,900
files each, comfortably under the cap).

**That workaround is no longer required of the caller (A1d).** The cap is a *response page*
limit, not a fixed row count -- observed page sizes for one query were 11121 / 11075 / 11086 /
10745 -- and the only honest signal of truncation is the response's `next` field, which the real
API does expose (`next.parameters.page`). `list_fft_files()` now follows it to completeness, and
falls back to halving the span (down to a 300 s floor) or raising `ONCListingError` if it cannot;
it never returns a truncated listing as complete. Re-measured over the same span with the fix:
**44,026 files, 99.96 % of the 44,064 season bins available, 19 unavailable bins, 5 requests,
~70 s** -- the same answer this month-scan produced, from one plain call. The whole study window
(`2020-02-18` -> `2026-10-01`) now also runs in one call: **620,909 files, 7 empty months, 68
requests, 715 s**. Cross-check: an interim month-chunking implementation of the same fix returned
byte-identical totals (620,909 files, the same 7 empty months) over 87 requests, so the number
does not depend on how the span was cut.

Two further live findings from the fix, both HTTP 400s that are **not** failures. ONC answers a
window lying entirely outside every deployment with *"API Error 127: A device with category
HYDROPHONE was deployed at location FGPD but not during the provided time range"* (seen for
`2020-02-18` -> `2020-03-01`), and a window that has not elapsed with *"API Error 25: Invalid
Time Range, Start Time is in the future"* (seen for `2026-09-01` -> `2026-10-01`, since
`config.STUDY_END_UTC` runs past today). In both cases no data can exist, so zero files is the
true answer: each is recorded as a measured zero **and printed**, matched on those exact phrases
only -- a 401 or any other 4xx still raises. This is a new decision and needs a record in
`docs/decisions/`.

## Provenance

* Produced **2026-08-27** from a live ONC archive listing (`ARCHIVE_EXTENSION = "fft"`, NOT
  `"fft.gz"` -- the gzipped extension is not what the archive index registers and returns zero
  files).
* Device `ICLISTENHF1266`, location codes discovered at runtime via
  `onc_client.discover_folger_locations()`, which resolved to `['FGPD']`.
* Composition, per season year, using only `boatphone/onc_client.py`:
  `discover_folger_locations` -> `list_fft_files` (one request per UTC month, results
  concatenated) -> `coverage_intervals` -> `mark_available` over `season_bins_utc(May 1, Oct 1)`
  -> `summarise_gaps(min_seconds=86400)` and `mean_availability_by_utc_hour`. Deployment spans
  from `get_deployments`; a season with no overlapping deployment is not requested at all.
* Wall clock for the whole 2020-2026 scan: **290 s** (35 monthly listing requests). A single
  month is ~10-70 s. Budget ~5 minutes to regenerate this file.

### The entry point has now been run over the full window

The numbers above were produced by the interim per-month composition described in this section.
`scripts/build_uptime_calendar.py` has since been run over the whole study window and the O1
artifacts exist, so these numbers are reproducible from committed code rather than from an ad-hoc
composition:

* `data/derived/hydrophone_uptime.csv` -- 308,448 rows, one per 5-min in-season UTC bin
* `data/derived/deployments.csv` -- 6 deployments
* `data/derived/hydrophone_uptime.provenance.json` -- `source: "listing"`,
  `generated_utc: 2026-08-27T04:13:35Z`, `git_commit: 0a9fdfc`, `onc 2.6.0`

That run listed 620,909 files over 68 paged requests (7 empty chunks) and reports **83.2 %**
available across the **whole 2020-02-18 -> 2026-10-01 window**. That is not in tension with the
97.05 % quoted above: 97.05 % is the mean over the six *deployed* seasons (2020-2025), whereas
83.2 % is the fraction over every in-season bin in the window, including the entire 2026 season
during which the instrument was **not deployed**. Both numbers are correct; they answer different
questions, and neither should be quoted without its span.

The three artifacts are gitignored (`data/` is not for git), which is why this document exists.
