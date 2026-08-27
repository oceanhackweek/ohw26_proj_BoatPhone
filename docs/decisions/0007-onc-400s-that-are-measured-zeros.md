# 0007. Two ONC HTTP 400s are measured zeros, not failures

Status: accepted
Date: 2026-08-27

## Context

Listing the full study window (`config.STUDY_START_UTC` .. `config.STUDY_END_UTC`, 2020-02-18 ..
2026-10-01) against ONC's `archivefile` endpoint raises before it finishes. Two of the failures
are not failures at all. Both were hit against the live API on 2026-08-27, at FGPD / HYDROPHONE:

```
API Error 127: A device with category HYDROPHONE was deployed at location FGPD
               but not during the provided time range      (2020-02-18 .. 2020-03-01)
API Error 25:  Invalid Time Range, Start Time is in the future.
                                                            (2026-09-01 .. 2026-10-01)
```

Error 127 covers the span between the start of the study window and the 2020-03-08 deployment:
**no instrument existed yet**. Error 25 covers spans past today, because `STUDY_END_UTC` is the
end of the 2026 field season and deliberately runs ahead of the present: **the window has not
happened**. In both cases ONC is not reporting that it failed to answer -- it is stating that no
file can exist in the span asked about. Zero is the true answer, and treating it as an error means
the study window cannot be scanned at all.

This surfaced during the A1d pagination work, which produced a second finding worth recording
alongside it (below).

## Decision

**These two responses, and only these two, and only on the FIRST page of a span, are treated as
an empty page rather than an error** (`boatphone/onc_client.py`, `_NO_DATA_POSSIBLE_MARKERS` and
`_fetch_page(..., page_number=...)`). The bins concerned come out UNAVAILABLE, which is the safe
direction for Planet quota: it withholds an order, it never invents one.

Five things keep this honest:

- **It is printed every time.** There is no silent zero: every span absorbed this way names
  itself and the ONC message in the run output (CLAUDE.md invariant 5).
- **Matching is string-based on ONC's exact wording** -- `"not during the provided time range"`
  and `"Start Time is in the future"`. If ONC rewords either message the match stops applying and
  the request raises again. That is the safe direction to fail: a re-raise is loud and gets fixed,
  a broadened match would quietly convert real errors into zero-file spans.
- **Nothing else is absorbed.** A 401, a 500, or any other 4xx propagates untouched.
- **Only the first page of a span may be absorbed.** Either marker arriving on page 2+ of a
  paginated listing raises `ONCListingError`, naming the span, the page, and how many rows the
  earlier pages already returned. The reasoning: on page 2 the requested time range is
  byte-identical to page 1's, and page 1 demonstrably returned files from it, so "no HYDROPHONE
  was deployed in the provided time range" / "Start Time is in the future" contradicts what the
  same server just said. That is not a measured zero -- it is a mishandled paging parameter, a
  server fault behind a 400, or ONC rewording something. Absorbing it would return `([], None)`,
  which the paging loop reads as "the listing is complete", truncating a dense span at page 1 and
  reporting it whole. Measured on a July 2024 fixture, that returned 288 files of 8,928 with zero
  empty chunks and no error -- the A1d phantom outage by a second route, aimed straight at Planet
  quota.
- **It is the invariant-9 distinction, drawn explicitly**: "the method found nothing" (no
  deployment, no elapsed time) versus "the method is broken" (a bad token, a server fault). Only
  the second is a bug, and they need different follow-ups.

## The pagination finding this came from

ONC's archive listing truncates silently and **there is no fixed row cap.** Paginating the 2024
season took 4 requests for 44,027 unique files, with page sizes 11121 / 11075 / 11086 / 10745 --
page sizes differ by hundreds of rows *between pages of one query*.

Consequences, both load-bearing:

- **Truncation is detected from the response's `next` cursor, never from a row count.** A
  row-count threshold would reintroduce the defect the moment ONC's page size moved.
  `config.ONC_LISTING_PAGE_ROWS_OBSERVED_MAX` exists for diagnostic messages only and is never
  used to decide whether a response was complete.
- **`next.parameters` and `next.url` contain the API token**, so they are never copied wholesale
  into the follow-up request. Only the `page` / `offset` key is taken from them.

## Consequences

- The full 2020-2026 window scans without a special-cased date list: the pre-deployment span and
  the not-yet-elapsed span resolve themselves from ONC's own answers.
- Widening `STUDY_END_UTC` further into the future costs nothing but more printed Error-25 lines.
- A mid-pagination no-data answer is now a loud, diagnosable stop rather than a short list. The
  cost is that a span cannot be listed at all until the cause is understood; that is the intended
  direction (invariant 9: "the method is broken" never becomes silence).
- If ONC rewords either message, the scan starts raising with the span named. The fix is to read
  the new wording and add it here, deliberately -- not to widen the match to "any 400".
- The uptime calendar this produces is listing-derived (D3): ONC's belief a file exists, not proof
  it downloads. A4's pull refines it and, where they disagree, the pull wins.
