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

**Amendment, 2026-08-27 (B3 live probe, 3 in-season 2025 dates, 90 files): the open question below
is now CLOSED.** ONC **ignores the `Range` header**. A request sent with
`Range: bytes=357398-` against a partially-downloaded file returned **HTTP 200**, not 206 -- with
`Content-Length` equal to the *whole* file (1,429,592 bytes) and no `Content-Range` header at all.
The bare-200 branch below fired as designed: the 357,398-byte partial was discarded and the
download restarted from byte zero. The result was byte-identical to a fresh, unresumed pull of the
same file. This is a resumability cost, not a correctness bug: an interrupted large file now
always costs its full re-download rather than a resumed tail, because ONC never takes the 206
branch. The local-sha256-only integrity limitation stated above is unchanged by this finding.

~~A related, still-open question this record does not settle:~~ `download_archive_file`
implements resumable download via HTTP `Range` requests, with both response branches handled --
a `206 Partial Content` response causes the partial file to be appended to, and a bare `200 OK`
response (meaning the server ignored the `Range` header) causes the partial file to be discarded
and the download restarted from zero. Which branch the live ONC endpoint actually takes is
~~**unverified**~~ **now verified: always the bare-200 branch** -- ONC's own client library never
sends a `Range` header, so there was no prior evidence either way. The code logs which branch it
took on any given request.

## Consequences

* Do not read a cache-hit sha256 match as proof the file is what ONC intended to serve -- it only
  proves the file matches what was written to disk on first download.
* ~~The first real resumed download against the live endpoint...~~ **Done, 2026-08-27**: ONC
  answers every `Range` request with a bare 200 and the full `Content-Length`, so resumability
  degrades to restart-from-zero on every retry -- correct but slow, and not a correctness bug,
  given the discard-and-restart branch is implemented and was the branch observed. This finding is
  now the record, not a future one to capture.
* If ONC is later found to serve truncated files on first request under some condition, that
  failure is invisible to this integrity scheme and would need a different signal (e.g. a length
  check against a listing-reported size) to catch. That risk is unchanged by this amendment.
