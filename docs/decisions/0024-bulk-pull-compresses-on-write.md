# 0024. The bulk overpass-corpus pull gzip-compresses each file as it is written to disk

Status: accepted
Date: 2026-08-27
Amends: [0022](0022-onc-fft-archive-product-is-plain-text-not-gzip.md) (partially -- see below)
Scope: `boatphone/acquire.py` (`resolve_corpus_files`), `boatphone/fft_io.py` (`read_fft_gz`),
`scripts/pull_overpass_corpus.py`
Source: B3 bulk pull, milestone1/b3-bulk-acquisition

## Context

Decision 0022 established, measured against the B3 live probe (90 files, 3 dates), that ONC's
`.fft` archive product is plain ASCII on the wire -- 614,400 whitespace-separated integers per
file -- and that this measures out to **43.4 MB/date**, i.e. ~40 GB projected over the full
918-date corpus (six seasons). That number replaced an earlier 2.7-8 GB plan that had assumed a
gzip container.

Plain ASCII digits compress well. The hand-delivered sample (`*.fft.gz`) gives a directly
observed ratio between its compressed size on disk and its decompressed content: roughly **4.9x**.
Applying that same ratio to the wire payload brings the projected corpus size back down to roughly
**8 GB** -- inside the size ceiling the project originally planned against, before 0022's
correction inflated it to ~40 GB. CPU cost of gzip-compressing on write is negligible measured
against 0022's 11.1 s/date of network time per file.

This record does not reopen 0022's finding about what ONC serves on the wire -- ONC still serves
plain text, not gzip, and that correction stands. What this record adds is what the bulk pull does
with those wire bytes **after** they arrive: it compresses them before they touch disk.

## Decision

`scripts/pull_overpass_corpus.py`'s bulk pull gzip-compresses each file as it is written.

**1. The sha256 sidecar hashes the WIRE bytes -- the plain payload as received -- not the
compressed container.** Two independent reasons, both worth recording:

* Decision 0018 defines the local sha256 as the record of *what was received*. If the sidecar
  hashed the compressed container instead, the check would keep passing while silently meaning
  something different -- a match against "what we wrote," not "what ONC sent."
* gzip output is not byte-stable across zlib versions or compression levels. A hash of compressed
  bytes would not even be reproducible run to run, let alone machine to machine.

Consequently, cache-hit re-verification (per 0018's re-verify-on-hit contract) **decompresses and
hashes the plain bytes**, not the bytes on disk directly.

**2. Compressed files are named `.fft.gz`**, so the name states the container honestly -- unlike
the situation 0022 described, where the archive-index name (`.fft`) and the container it actually
held (plain text) disagreed. Consequently `boatphone.acquire.resolve_corpus_files` now matches
**both** `.fft` and `.fft.gz` inside `paths.ONC_OVERPASS_CORPUS_DIR`.

This deliberately **amends** the resolver contract pinned by `check_b3c_11`, which required the
resolver match `.fft` and exclude `.fft.gz`. That exclusion existed to guard against a specific
failure: a notebook globbing `*.fft.gz` finding **zero** files in a `.fft` corpus and reporting "no
data" instead of failing loudly (the confusion invariant 9 exists to prevent). That failure mode
is guarded differently now, not left unguarded: the corpus has its own directory
(`paths.ONC_OVERPASS_CORPUS_DIR`, per decision 0020), and `resolve_corpus_files` is the **one**
glob any caller is meant to use -- it raises rather than returning an empty list (unchanged from
before this record) regardless of which container extension is present. `check_b3c_11` needs to be
updated to reflect the amended contract; that update is implementation work, not made by this
record.

**3. The corpus is permanently mixed.** The 90 files already pulled by the B3 live probe (the
basis for 0022's measurements) are plain text under the `.fft` name and will **not** be converted
or re-pulled to match the new gzip-on-write behaviour, because `data/` is immutable (decision
0001). Any consumer reading the corpus must handle both containers side by side. This is not a
transitional state that a later cleanup pass resolves -- **it is the steady state** the corpus will
remain in permanently, since decision 0001 forbids ever normalizing it.

**4. The single reader sniffs magic bytes.** Per 0022, `read_fft_gz` was already required to
detect its input's actual container rather than trust the extension, because the corpus name
(`.fft`) and the sample name (`.fft.gz`) disagreed about what was inside. That sniff-don't-trust
design is exactly what makes this record cheap: the same reader that already had to handle
"`.fft`-named but plain text" now also handles "`.fft.gz`-named and actually gzip" through the
same magic-byte check, with no second code path. `read_fft_gz` remains a misnomer under both 0022
and this record -- it reads whichever container is actually present, not specifically a `.gz` one.

## Why this is cheap, and where it interacts with 0018

**Correction, 2026-08-27 (post-review):** the paragraph originally here overstated 0018's finding
and, worse, was load-bearing in a way that could mislead a maintainer into deleting real safety
code. 0018's Range/resume finding is **n=1** -- ONE resumed request, observed live to return a
bare 200 rather than 206. It is not a general property of the endpoint verified across repeated
trials, and it does NOT mean there is "no partial-write, resume-from-offset state" in the code.

The opposite is true: `download_archive_file` DOES implement byte-offset resume, and this record's
gzip-on-write behaviour was built to compose with it, not to assume it away. Concretely, in
`boatphone/onc_client.py`: `_wire_bytes_on_disk` (around line 1829) computes a resume offset in
WIRE bytes from a compressed `.part` file (decompressing it to count, since the bytes on disk are
not the bytes ONC sent once `compress_on_write` is in effect); a `206 Partial Content` response
(around line 1947) appends a NEW gzip member to the existing `.part`, relying on concatenated gzip
members being a valid gzip stream (RFC 1952 s2.2); and `check_b3e_6` (per the module's own
check-suite) exercises exactly that append-on-206 path. This code is real, tested, and load-bearing
for the (rare, n=1-so-far) case where ONC does honour Range.

The correct, narrower claim: on the ONE resumed request the B3 live probe made, ONC returned a bare
200 and the discard-and-restart branch fired, which is what makes compress-on-write cheap **in that
observed case** -- a discarded-and-restarted download has no partial state for compression to
disturb, because it isn't resumed at all, it's redone from zero. But this is a statement about what
was *observed once*, not a guarantee about what the endpoint always does. Compressing on write is
still safe when the 206/resume path DOES fire, because the append-a-gzip-member design above was
built for exactly that case -- it does not depend on 206 never occurring. Do not read this record,
or 0018, as license to remove `_wire_bytes_on_disk` or the 206-append branch as dead code: they are
tested, they are the mechanism that keeps resume and compression correct together, and the n=1
observation that the live endpoint currently prefers bare-200 does not make them unnecessary.

## Consequences

* **Measured, 2026-08-27 (post-review, after the 918-date bulk pull completed): the corpus is 6.8
  GB on disk** (`du` over `data/raw/onc/overpass_window_corpus/`, 26,666 files), against a summed
  wire payload of ~39.3 GB across those same files (matching 0022's ~40 GB projection). This
  measured 6.8 GB is now the headline number for this record, replacing the pre-run ~8 GB
  projection below, which is kept for reference as what was predicted beforehand, not as the
  current estimate.
* **Two ratios exist and should not be conflated.** The pre-run projection (~8 GB, below) used a
  **4.9x** ratio observed on the hand-delivered sample file
  (`ICLISTENHF1266_20260313T000004.000Z.fft.gz`, decompressed/compressed). A **direct measurement
  on a real file from the completed bulk pull itself**
  (`ICLISTENHF1266_20250715T161505.000Z.fft.gz`: 1,429,592 wire bytes -> 269,487 bytes on disk,
  post-review re-check found 269,535 bytes, consistent within rounding/gzip-parameter noise) gives
  **5.30x**. The bulk-pull corpus compresses somewhat better than the hand-delivered sample
  predicted; the discrepancy is not resolved by silently picking one figure -- both are stated,
  and the 5.30x figure is the one drawn from the actual production run, not a proxy sample.
* **Original pre-run projection, kept for reference:** corpus disk usage was projected to roughly
  8 GB going forward for newly-pulled dates (the 4.9x sample ratio applied to 0022's ~40 GB
  projection, not independently re-measured against a full compressed download at the time this
  record was first written). The 90 files already on disk from the live probe remain plain text at
  0022's measured 43.4 MB/date; this record does not restate or re-derive their total. The corpus
  as a whole is not one single uniform-container total -- it is the sum of an uncompressed
  historical slice (the 90 live-probe files) and a compressed ongoing one (26,666 files, 6.8 GB,
  per the measurement above).
* `check_b3c_11` pins the pre-0024 exclusion contract and must be updated to assert the new
  both-extensions match instead. Not done as part of this record -- this is a decision record, not
  an implementation change.
* Any future code path that globs the corpus directory directly, instead of going through
  `resolve_corpus_files`, will silently miss half the corpus (whichever container it doesn't glob
  for) rather than raise. `resolve_corpus_files` remains the one sanctioned glob for exactly this
  reason.
* `read_fft_gz`'s name continues to understate what it does; renaming it is deferred, as it was
  under 0022, to a future refactor rather than bundled into this record.
* **Superseded, 2026-08-27:** this bullet originally flagged that the 4.9x ratio was unverified at
  scale. It has since been checked: the 918-date bulk pull completed, and the measured outcome
  (6.8 GB on disk, 5.30x on a spot-checked real file, both above) is now the record rather than an
  open question. The gap between 4.9x (sample) and 5.30x (production) is real but modest and does
  not change the order-of-magnitude conclusion that gzip-on-write keeps the corpus well under
  0022's uncompressed ~40 GB.
