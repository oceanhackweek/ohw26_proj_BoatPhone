# 0023. A missing archive file is HTTP 400 with "API Error 96", and is an ABSENT file, not an error

Status: accepted
Date: 2026-08-27
Amends: none
Scope: `boatphone/onc_client.py` (`_FILE_DOES_NOT_EXIST_MARKERS`), `boatphone/acquire.py`,
`scripts/pull_overpass_corpus.py` (`absent_log`, decision 0020's consumer)
Source: B3 live probe against the real ONC archive, 3 in-season 2025 dates, 90 files; a direct
probe of a deliberately non-existent filename against `archivefile/download`

## Context

Requesting a filename that ONC's archive does not have (probed directly, by name, against a
non-existent file rather than encountered during the 3-date pull) returns `HTTP 400` with the body
`API Error 96: No file could be found.`

`boatphone/onc_client._NO_DATA_POSSIBLE_MARKERS` currently holds exactly two strings: "not during
the provided time range" and "Start Time is in the future" -- both listing-endpoint markers, per
decisions 0007/0008. `"API Error 96"` is not among them. As written, a 400 response for a missing
*file* (as opposed to an empty *listing*) does not match any marker, so it falls through to the
general error path and raises `DownloadError`, which the retry logic then retries.

Two consequences follow directly if this fires during a real pull:

1. It lands in the manifest as an **ERROR** row (a retried, ultimately-failed download), not an
   **ABSENT** row. Decision 0020's segment D reads `absent_log` as the pull-side ground truth for
   refining `hydrophone_uptime.csv`; a file ONC has confirmed does not exist getting misclassified
   as an error rather than a measured negative corrupts exactly the input that refinement depends
   on.
2. It burns retry budget on a condition retrying cannot fix -- ONC has already given a definitive
   answer ("no such file"), not a transient failure.

**This did not occur during the 3-date, 90-file probe** (0 errors observed in that run). It is a
prospective risk identified by directly probing a deliberately non-existent filename against the
live endpoint, not something observed to break a real pull. It has not yet been exercised inside
an actual bulk-pull run.

## Decision

**A `400` response whose body contains `"API Error 96"` is an ABSENT file -- a measured negative,
the same evidentiary status decisions 0007/0008/0016 already give a listing-side "no data
possible" response -- not a download error.** It should be caught and logged as absent, not raised
and retried.

This is narrowly scoped to that specific marker string. Per invariant 5, every *other* 400 and
every 5xx must continue to raise: a malformed request, an auth failure, a server error, or any
other 400 body is still evidence something is broken, and folding those into "absent" would
launder a real failure into a plausible-looking measured zero -- exactly the failure mode
invariant 5 exists to prevent. Only the specific, confirmed "no such file" signal gets the
measured-negative treatment.

**Implemented in the same commit as this record.** `boatphone/onc_client.py` adds a SEPARATE
constant, `_FILE_DOES_NOT_EXIST_MARKERS` (currently just `"API Error 96"`), rather than appending
to `_NO_DATA_POSSIBLE_MARKERS`. This is a deliberate choice, not an oversight: `_NO_DATA_POSSIBLE_MARKERS`
routes to `status="measured_zero"`, and `measured_zero` is reserved for decisions 0007/0008's
listing-side "no data can exist" evidence. A confirmed-missing *file* is a different evidentiary
claim -- ONC has a listing that named the file but the archive does not have it -- so
`download_archive_file` routes an `_FILE_DOES_NOT_EXIST_MARKERS` match to `status="absent"`
instead, keeping the two 400-body meanings distinct rather than collapsing them into one status
that would blur decision 0016's three-bucket scheme. `scripts/pull_overpass_corpus.py`'s
`absent_log` consumes both statuses but records which one produced each row.

## Consequences

* Without this fix, a bulk pull that happens to request a file ONC does not have (e.g. from a
  listing/download race, or a listing that is stale relative to the archive) would misrecord that
  file as an error and burn retries on it, rather than logging it as absent per decision 0016's
  three-bucket scheme (404-as-bug, positive-zero, and now this: confirmed-missing-file).
* **Operational fact from the completed 918-date bulk pull (26,689 real requests):** this path
  never fired. `errors=0` across the whole run, and every one of the 23 absences the manifest
  recorded was a decision-0016 empty overpass window (a listing with no file at all), not a
  confirmed-missing-*named*-file. So `_FILE_DOES_NOT_EXIST_MARKERS` remains production-unexercised
  -- it is evidenced only by the one deliberate probe of a non-existent filename described above,
  not by anything the real pull encountered. That the code path was written defensively and never
  needed is a fact worth recording, not a sign it is unnecessary: the listing/download race and
  stale-listing scenarios above did not happen to occur in this particular 918-date run, but
  nothing about them is precluded from occurring in a future one.
* **Unverified beyond the direct non-existent-filename probe:** whether "API Error 96" is the
  *only* 400 body ONC returns for a missing archive file, across product types, devices, or error
  locales, is not established -- only checked once, by name, against one such request.
* **Coverage caveat carried from the probe as a whole, not specific to this finding:** the 3
  probed dates were all in-season 2025 dates in `America/Vancouver`'s PDT offset. The PST branch
  of the timezone conversion (winter, standard time) was not exercised by this live probe at all
  and remains verified only synthetically, by `check_b3c_2`. Nothing in this record depends on
  DST/PST, but it is stated here because this record and 0022 together are the full set of claims
  the live probe supports, and the PST gap applies to the probe as a whole, not just this item.
