# 0022. ONC's `.fft` archive product is plain text, not gzip -- despite the `.fft.gz` contract

Status: accepted
Date: 2026-08-27
Amends: none (corrects a contract previously stated as settled in the B3 handoff)
Scope: `boatphone/fft_io.py` (`read_fft_gz`), `scripts/pull_overpass_corpus.py`,
`docs/plans/acoustics_plan_v2.md` corpus-size estimate
Source: B3 live probe against the real ONC archive, 3 in-season 2025 dates, 90 files
(`claude/run-phase/run-phase-milestone1-b3-bulk-acquisition.handoff.md`)

## Context

The B3 handoff and earlier project notes state, as a settled contract, that "B3's corpus lands as
`*.fft` containing gzip bytes" -- i.e. the archive extension is `.fft` but the payload inside is
gzip-compressed, matching the hand-delivered sample file (`*.fft.gz`) after its container is
stripped. `boatphone/fft_io.py::read_fft_gz` was written against exactly this contract: it calls
`gzip.open` on the path it is given.

This is wrong. Checked directly across all 90 real files pulled in the probe:

* No file contains the gzip magic bytes (`1f 8b`) anywhere.
* The archive serves the product **already decompressed**, as plain ASCII text: 614,400
  whitespace-separated integers per file, reshaping cleanly to (1200 frames x 512 bins), values
  ranging 0.0-92.0.
* This is the *same payload*, in kind, that the hand-delivered `.fft.gz` sample yields once it is
  gunzipped -- only the container differs (plain text on the wire vs. gzip-in-a-file), not the
  content.
* Requesting the file under the `.fft.gz` name (as the old contract would imply) does not work:
  ONC returns `400, API Error 96: No file could be found.` There is no gzipped name in the
  archive to request.

So the `.fft` **extension** ONC serves the file under was correct all along -- the mistake was in
the assumption about what *bytes* live inside a file with that extension.

## Decision

Record the corrected contract: **ONC's archive `.fft` product is plain-text ASCII, not gzip.**
`read_fft_gz` cannot open it as written -- `gzip.open` on a plain-text file raises `BadGzipFile` --
so segment E (or anything else reading a real corpus file) is blocked until the reader is fixed to
accept both containers.

The fix is scoped for the next implementation pass, not made here: the reader should **sniff the
magic bytes** rather than trust the extension or the source, because the corpus filename (`.fft`)
and its container (plain text) now disagree with the sample filename (`.fft.gz`) and its container
(gzip) -- two different sources use the same nominal product type under incompatible containers,
and only the actual leading bytes distinguish them reliably.

## Consequences

* **`boatphone/fft_io.py::read_fft_gz` is a misnomer** now that its target format is not
  universally gzip. It should either be renamed once it accepts both containers, or split into a
  container-detecting wrapper around a shared plain-text/gzip-agnostic parser. Not renamed as part
  of this record -- this record establishes the fact, not the refactor.
* **The corpus size estimate carried in the acquisition plan was wrong by ~4.9x.** Measured: 3
  dates, 90 files, 130,083,623 bytes total -> **43.4 MB/date**. Projected over 918 dates (six
  seasons), that is **~40 GB**, not the 2.7-8 GB previously planned -- because plain text is far
  larger on disk than the gzip payload the old estimate assumed. Any storage planning or quota
  conversation for the full B3 pull must use the ~40 GB figure, not the earlier one.
* **Wall-clock timing is unaffected and does match the earlier estimate**: 11.1 s/date measured,
  projecting to ~2.8 h for 918 dates. Only the size estimate was wrong; the time estimate stands.
* Anything downstream that assumed a gzip container for the real corpus (segment E's design,
  any storage-budget arithmetic, any doc referencing "`*.fft` containing gzip bytes") needs to be
  read against this correction, not the superseded description in the B3 handoff.
* Not yet checked: whether every file in the eventual full 918-date pull is plain text, or whether
  some dates/products could still arrive gzipped (e.g. a different product version, a different
  device state). The probe covers 90 files across 3 dates; a magic-byte sniff at read time (rather
  than a container assumption baked into the reader) is the robust fix precisely because this has
  not been exhaustively checked.
