# 0019. Refuse to promote a transfer of unknown length

Status: accepted
Date: 2026-08-27
Scope: `boatphone/acquire.py` (`download_archive_file`)
Source: `claude/run-phase/run-phase-milestone1-b3-bulk-acquisition.handoff.md` (B3 segment B,
"THE BLOCKER FOR D AND E", item 1)

## Context

`download_archive_file`'s previous behaviour, on a response that did not report a total content
length, was to accept whatever bytes arrived, hash them, write the sidecar, and promote the file
to the cache as complete. If the server had truncated the response -- for any reason, including a
dropped connection -- that behaviour wrote a wrong file to disk and then certified it as correct
via decision 0018's local sha256, permanently. A later re-run would see the cache hit, the sha256
would match (the file matches itself), and the truncation would never surface. This is exactly the
"plausible number" invariant 5 warns against: a data problem laundered into a clean-looking
result.

## Decision

`download_archive_file` now **raises** rather than promoting any transfer whose total size is
unknown at the start of the transfer (i.e. no usable `Content-Length` or equivalent). A file of
unknown length is never written to the cache as complete, sha256 sidecar or not.

## Consequences

* This is deliberately a hard failure over a silent one, per invariant 5. A run that hits this
  condition stops rather than producing a corpus with an undetectable truncated file buried in it.
* **Amendment, 2026-08-27 (B3 live probe, 3 in-season 2025 dates, 90 files): now VERIFIED SAFE
  against the live endpoint, not untested.** ONC sends a real, explicit `Content-Length` on every
  observed response (e.g. `1429592`) and **no `Transfer-Encoding` header at all** -- chunked
  transfer was not observed. `total_size_known` was `True` on all 90 rows of the probe, and the
  refusal guard never fired. The risk described below -- "if ONC serves chunked, every file
  raises and the run dies" -- did **not** materialize against the archive endpoint this project
  pulls from. The guard is kept as-is; it remains correct and cheap insurance even though the
  probe found no case where it triggers.
* ~~This is untested against the live ONC endpoint and is a known unknown, not a settled
  question.~~ Retained for the record of what was unknown before the probe: if ONC serves
  `Transfer-Encoding: chunked` for archive file downloads (reporting no `Content-Length` up
  front), every file in a pull would raise, and a six-season, 918-date run would stop on the
  first file rather than complete. That was the intended failure direction if it happened -- loud
  and immediate, not 918 dates of undetectable truncation risk. It has now been checked against
  the live endpoint's actual header behaviour for 90 files across 3 in-season 2025 dates and did
  not occur; if it is ever observed, the fix is not "catch and continue" but finding another way
  to know the transfer completed (e.g. a listing-reported size to check the received byte count
  against).
* The handoff explicitly calls this the first thing to check on the first few real files of a
  bulk pull, before letting a long unattended run proceed -- that check has now been done.
