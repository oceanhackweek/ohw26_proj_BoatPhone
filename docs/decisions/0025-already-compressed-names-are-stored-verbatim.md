# 0025. An already-compressed name is stored verbatim, never re-compressed

Status: accepted
Date: 2026-08-27
Amends: none (narrows the naming behaviour of decision 0024's compress-on-write)
Scope: `boatphone/onc_client.py` (`download_archive_file`), `boatphone/config.py`
(`GZIP_CONTAINER_SUFFIX`, `GZIP_MAGIC_BYTES`)

## Context

Decision 0024 established that the bulk pull gzip-compresses each downloaded file as it is
written, and names the stored file with `config.GZIP_CONTAINER_SUFFIX` appended (`X.fft` lands as
`X.fft.gz`). That record did not separately spell out what happens when the requested `filename`
already ends `.gz` -- which is exactly the case for the hand-delivered sample fixture the B3-B
checks (`check_b3b_1`, `check_b3b_2a`, `check_b3b_2b`, `check_b3b_3`, `check_b3b_4`) exercise.

## Decision

**A retrieved filename that already ends in `.gz` is stored verbatim and is NOT renamed or
re-compressed.** `download_archive_file` computes `compress_on_write = not
filename.endswith(GZIP_CONTAINER_SUFFIX)`; when `filename` already ends `.gz`, `local_name`
equals `filename` unchanged, and the bytes received are written through without a second gzip
pass. Only a non-`.gz` name is gzip-compressed on write and gains the suffix (decision 0024's
compress-on-write path).

Reason: re-compressing an already-gzip-named file would double-wrap it and change what "the file
ONC sent" means on disk (per 0024's own reasoning). It also keeps the B3-B checks meaningful --
their fixture is named `.fft.gz` and asserts BYTE-VERBATIM content at the final path, which only
holds if a `.gz`-named input is never touched by a second compression pass. This is consistent
with, not separate from, the new B3-E compression checks (`check_b3e_*`), which exercise the
compress-on-write branch for non-`.gz` names.

**This is a NAMING rule only. The reader never trusts it.** `boatphone.fft_io.read_fft_gz` sniffs
the container's actual magic bytes (per decision 0022) rather than inferring the container from
the filename. So a name/container disagreement in either direction -- a `.gz`-named file that
turns out to hold plain text, or a non-`.gz`-named file that turns out to already be gzip -- is
still read correctly. The naming rule in this record governs only what `download_archive_file`
writes and calls the file; it makes no claim about what any future or unusual response body
actually contains.

**Constants, single-sourced:**

* `config.GZIP_CONTAINER_SUFFIX = ".gz"` -- the suffix appended on compress-on-write, and the
  suffix tested for on the incoming `filename` to decide whether to compress at all.
* `config.GZIP_MAGIC_BYTES = b"\x1f\x8b"` -- RFC 1952 §2.3.1's gzip magic bytes, used by both
  sniff sites (write-time cache re-verification in `download_archive_file`, and read-time
  container detection in `read_fft_gz`) so the two agree on what "is gzip" means without a second
  definition drifting from the first.

## Measured compression ratio, and its scope

On the one real file measured: **1,429,592 bytes uncompressed -> 269,487 bytes compressed,
5.30x.** This is better than the ~4.9x assumed in decision 0024's ~8 GB corpus projection -- but
this is ONE file. Decision 0024's ~8 GB projection remains an estimate, not independently
re-measured at scale, and is if anything conservative in light of this single data point running
higher than assumed. This record does not revise the ~8 GB figure; it only adds one supporting
observation in the direction of "less disk, not more."

## Consequences

* `check_b3b_1`, `check_b3b_2a`, `check_b3b_2b`, `check_b3b_3`, `check_b3b_4` (byte-verbatim
  assertions against the `.fft.gz` fixture at the final path) and `check_b3e_*` (compress-on-write
  assertions against non-`.gz` names) exercise complementary branches of the same
  `compress_on_write` conditional and remain consistent with each other under this rule.
* Any future caller that hand-constructs a local filename outside `download_archive_file` must
  apply the same rule (test for the existing `.gz` suffix before deciding whether to compress) or
  risk double-wrapping a file that is already a valid gzip container.
* The 5.30x ratio is not used to update any constant; it is recorded here as a single, freshly
  measured data point that is consistent with -- and slightly better than -- 0024's assumption.

## How this would show up as wrong

If a future response ever arrived under a `.gz`-suffixed `filename` while actually carrying an
uncompressed body, the naming rule alone would store it verbatim and mislabel it -- but
`read_fft_gz`'s magic-byte sniff (0022) reads it correctly regardless, so the failure mode this
record needs to guard against (a corrupted or misread payload) does not actually occur; only a
cosmetic mismatch between name and container would exist on disk, exactly as decision 0024 already
anticipated as the corpus's permanent mixed state.

Related: `docs/decisions/0022-onc-serves-fft-as-plain-text-not-gzip.md` (the sniff-don't-trust
reader contract this record's naming rule depends on); `docs/decisions/0024-bulk-pull-compresses-on-write.md`
(the compress-on-write behaviour this record narrows).
