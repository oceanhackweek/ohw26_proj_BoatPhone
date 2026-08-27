# 0023. A missing archive file is HTTP 400 with "API Error 96", and is an ABSENT file, not an error

Status: accepted
Date: 2026-08-27
Amends: none
Scope: `boatphone/onc_client.py` (`_NO_DATA_POSSIBLE_MARKERS`), `boatphone/acquire.py`,
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

This record establishes the rule; it does not itself implement the fix (no code was changed to
produce this record, per this segment's scope). Implementing it means adding the marker (or an
equivalent check against the file-download error path specifically, since `_NO_DATA_POSSIBLE_MARKERS`
today is listing-scoped) and routing a match to the same `absent_log` path decision 0016 defined
for empty windows, with its own `reason`.

## Consequences

* Until implemented, a bulk pull that happens to request a file ONC does not have (e.g. from a
  listing/download race, or a listing that is stale relative to the archive) will misrecord that
  file as an error and burn retries on it, rather than logging it as absent per decision 0016's
  three-bucket scheme (404-as-bug, positive-zero, and now this: confirmed-missing-file).
* **Unverified beyond the direct non-existent-filename probe:** whether "API Error 96" is the
  *only* 400 body ONC returns for a missing archive file, across product types, devices, or error
  locales, is not established -- only checked once, by name, against one such request.
* **Coverage caveat carried from the probe as a whole, not specific to this finding:** the 3
  probed dates were all in-season 2025 dates in `America/Vancouver`'s PDT offset. The PST branch
  of the timezone conversion (winter, standard time) was not exercised by this live probe at all
  and remains verified only synthetically, by `check_b3c_2`. Nothing in this record depends on
  DST/PST, but it is stated here because this record and 0022 together are the full set of claims
  the live probe supports, and the PST gap applies to the probe as a whole, not just this item.
