# 0018. No server-side checksum; integrity is a local sha256 only

Status: accepted
Date: 2026-08-27
Scope: `boatphone/acquire.py` (`download_archive_file`)
Source: `claude/run-phase/run-phase-milestone1-b3-bulk-acquisition.handoff.md` (B3 segment B,
"Contracts already settled", open item 2)

## Context

B3's download primitive (`download_archive_file`) needs to know, on a cache hit, whether the file
already on disk is the file ONC would serve again. ONC's `archivefile/download` endpoint was
probed live and confirmed to serve **no ETag and no Content-MD5** -- there is no server-side
integrity signal to compare against, at all.

## Decision

Integrity is a local-only sha256, written to a sidecar file alongside each downloaded product at
the time of download, and re-verified against the sidecar on every subsequent cache hit before the
cached file is treated as valid.

Stated precisely so the limit is not overstated: this catches **local** corruption -- a
bit-flip, a truncated write, a partial file from a previous crashed run that was mistaken for
complete. It does **not** and cannot catch **server-side truncation** -- if ONC serves a truncated
file on the *first* download, the sha256 sidecar records the hash of the truncated bytes as
correct, and every future cache-hit check will agree.

**A related, still-open question this record does not settle:** `download_archive_file`
implements resumable download via HTTP `Range` requests, with both response branches handled --
a `206 Partial Content` response causes the partial file to be appended to, and a bare `200 OK`
response (meaning the server ignored the `Range` header) causes the partial file to be discarded
and the download restarted from zero. Which branch the live ONC endpoint actually takes is
**unverified** -- ONC's own client library never sends a `Range` header, so there is no existing
evidence either way. The code logs which branch it took on any given request.

## Consequences

* Do not read a cache-hit sha256 match as proof the file is what ONC intended to serve -- it only
  proves the file matches what was written to disk on first download.
* The first real resumed download against the live endpoint (i.e. the first case where a partial
  file already exists locally and the pull is retried) answers the 206-vs-200 question. That log
  line should be captured and recorded (e.g. folded into a future decision record or the B3
  manifest) the first time it fires, rather than left to be re-discovered.
* If ONC is later found to answer `200` on every `Range` request, resumability degrades to
  restart-from-zero on every retry -- correct but slow -- and is not a correctness bug given the
  discard-and-restart branch is implemented. If it is later found ONC serves truncated files on
  first request under some condition, that failure is invisible to this integrity scheme and would
  need a different signal (e.g. a length check against a listing-reported size) to catch.
