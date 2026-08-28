---
title: B3 close-out -- offline manifest migration and constant corrections
type: plan
status: proposed  # gate verdict landed as decisions/0027; see b5-fft-viability-gate.md
milestone: "1"
owner: Isaac Guld
related:
  - ../decisions/0018-no-server-checksum-local-sha256-only.md
  - ../decisions/0020-pull-refines-only-the-overpass-window.md
  - ../decisions/0021-manifest-run-directories-and-latest-pointer.md
  - ../decisions/0022-onc-serves-fft-as-plain-text-not-gzip.md
  - ../decisions/0024-bulk-pull-compresses-on-write.md
  - acoustics_plan_v2.md
source: ~/.claude/plans/b3-close-out-offline.md
---

# PLAN 2 of 2 — B3 close-out, offline (RUN TONIGHT)

> **Companion plan:** `b5-fft-viability-gate.md` (in this directory) — the `.fft.gz` detector viability
> gate, which runs **first**. This plan is deferred because it rewrites manifest artefacts and does
> bulk correctness work, and doing that mid-day while the corpus is being read is asking for
> trouble. **Do not start this until the gate plan has finished.**
>
> Nothing here touches the network. Nothing here needs the gate's verdict.

## Context

The B3 bulk acquisition left a red branch (`1937b6e`, "WIP B3 punch list — RED BRANCH, one real
defect open", DO NOT MERGE) with the check suite at **161 passed / 1 failed / 4 skipped / 8 not
run**. The handoff records the remaining work as a **~1h40m re-run** of
`scripts/pull_overpass_corpus.py` to regenerate the manifest under its new schema.

**That re-run is avoidable, and this plan is the replacement.** The corpus itself is complete and
untouched — 26,666 `.fft.gz` + 90 plain `.fft`, 6.8 GB, 918/918 dates, `run_complete: true`,
0 errors, `requested 26689 = present 26666 + absent 23`. The re-run would download zero bytes; its
entire cost is one ONC listing round-trip per date for 918 dates, purely to rewrite metadata that
is derivable from what is already on disk. Measured this session: **0 of 26,666 rows have a `path`
that is missing on disk**, and every row's `path` already carries the `.fft.gz` suffix.

Separately, the Planet timestamps that landed this session falsify the overpass-window constant
that both workstreams are told to import — a cross-team correctness bug that has to be fixed at the
source, not worked around.

---

## 1. Fix `check_b3c_12` against the right target

`scripts/checks.py:8021`
(`check_b3c_12_per_file_record_exposes_disk_basename_distinct_from_wire_filename`) is the one real
failure on this branch. It asserts a per-file manifest row carries `disk_basename` distinct from the
wire `filename`, and that `path` names a file that exists with basename == `disk_basename`
(`scripts/checks.py:8068-8095`).

**Before writing a fix, weigh this measurement.** The library seam is already correct:
`boatphone/onc_client.py:1774-1778` sets `local_name = f"{filename}{GZIP_CONTAINER_SUFFIX}"`, and
both return sites (`:1806` cached, `:2045` downloaded) set `path=final_path` — the `.gz` path.
`scripts/pull_overpass_corpus.py:798` copies it verbatim and derives `disk_basename` at `:813-816`.
And the real complete manifest's paths all resolve.

So `1937b6e`'s diagnosis ("`path` records the PRE-compression name") appears to describe the check's
fake transport `_B3BFakeTransport`, **not** shipping code. **Determine which is actually wrong
before editing.** If production is correct, the fix belongs in the test double, and "fixing" the
library would be a regression that also invalidates the on-disk corpus's provenance. Report which it
turned out to be — this is invariant 9 ("the method found nothing" vs "the method is broken")
applied to our own test harness.

Fixing this also un-blocks the **8 checks that currently never run** because they sit after the
failure point.

## 2. Migrate the manifest offline instead of re-running the driver

New one-off script, `scripts/migrate_manifest_schema.py`, reading the complete run at
`data/derived/runs/20260827T084833.787903Z/` and writing a **new** run dir:

- rename `bin_start_utc`/`bin_end_utc` → `file_start_utc`/`file_end_utc` on every `files[]` row.
  The rename landed in code at `1937b6e` but never in the artefact. It matters: "bin" already means
  an epoch-aligned `BIN_SECONDS` grid cell (`config.py:12-14`, `onc_client.py:1158-1161`) while
  these values are jittered file starts, and the collision would make segment D's join look like
  low coverage rather than a bug.
- add `disk_basename = os.path.basename(path)` per row (currently absent from all 26,666).
- **validate, don't assume**: assert every `path` exists, that `disk_basename` matches the file on
  disk, and that `requested == present + absent` still holds (26689 = 26666 + 23).
- **distinguish the one served-but-empty file from a present one.**
  `ICLISTENHF1266_20220825T184149.000Z.fft.gz` has `bytes_downloaded: 0`,
  `http_status: 200`, `status: "downloaded"`, and sha256 `e3b0c442...` -- the hash of the
  empty string. ONC served an empty body with a 200. It is the only one in the corpus
  (verified: exactly one zero-byte row, exactly one sub-100-byte file on disk), so
  `present = 26666` overstates usable windows by one; **26,665 are readable**. This is a
  THIRD category alongside decision 0007's measured zeros and 0023's Error-96 absences --
  listed, served, empty -- and `requested == present + absent` currently counts it on the
  wrong side. Found by the population pass, decision 0027 amendment 2.
- reconcile the **90 orphan plain `.fft` probe files** on disk that appear in no manifest row —
  either add them with an explicit `status` marking them as probe artefacts, or record why they are
  excluded. Right now the corpus and the manifest silently disagree by 90 files, and segment D
  joins on this manifest.
- write a provenance sidecar stating this is an **offline schema migration of run
  `20260827T084833.787903Z`, not a fresh pull** — source run id, the script's git SHA, and the
  fields touched. Do not let a derived artefact look like a network acquisition.
- repoint `data/derived/pull_overpass_corpus.manifest.json` **only after** validation passes. Note
  the latest pointer was hand-restored once already after an interrupted regeneration left it
  advertising a partial run; the incomplete run at
  `data/derived/runs/20260827T170451.141509Z/` (510/918, `run_complete: false`, new schema) is that
  wreckage and should be left in place or explicitly retired, not silently deleted.

This needs a short decision record (next free number, **0028**): it creates a manifest whose
provenance is "derived from another manifest", which decisions 0021 and 0018 do not contemplate.

## 3. Correct the window constants — a cross-team correctness bug

`boatphone/config.py:633-634` (`PLANET_OVERPASS_WINDOW_START_LOCAL = 09:15`,
`..._END_LOCAL = 11:45`) is measurably wrong, and it is the constant Malachy's ordering and Isaac's
pull are both told to import (CLAUDE.md invariant 6; source-of-truth "ALL WORKSTREAMS"). Measured
against `contributor_folders/malachymcc/planet_folger/gate2_survivors.csv` (30 scenes, `acquired`,
tz-aware UTC), the real overpasses fall **18:17–19:49 UTC = 11:17–12:49 local**, bimodal across the
PS2.SD / PSB.SD constellations. Coverage of the existing corpus against those scenes: **13 full, 5 partial, 12 zero.**

Do **not** silently widen the constant. Replace the assumed values with the measured ones, keep the
assumed values in a comment as the superseded prior, and cite `gate2_survivors.csv` as the evidence.
Add decision record **0029** recording that the window was assumed, was explicitly flagged as
assumed in `acoustics_plan_v2.md` §9 ("verify against the first `overpasses.csv`"), was verified, and
was wrong — including the coverage table, so nobody later reads the 6.8 GB corpus as covering the
overpasses. Update `acoustics_plan_v2.md` §5 B3 and §9, and the source of truth.

Two constant duplications of the same class, cheap to fix in the same pass:

- The hydrophone coordinate is hardcoded in Malachy's notebook (cell 6,
  `-125.278277, 48.814200`) and in `boatphone/optical.py:71` on `origin/neve`, and is **absent from
  `config.py`** — which only has `DEVICE_CODE`/`DEVICE_ID` (`config.py:62-63`). The canonical value
  currently lives only on an unmerged branch. Promote it to `config.py`; `optical.py:59` already
  says it should be there.
- Malachy's notebook uses `SUMMER = (6, 9)` (June–Aug) against
  `config.SEASON_MONTHS_UTC = (5,6,7,8,9)` (May–Sep). Two definitions of the same season. **Flag to
  Malachy; do not edit his notebook** — `.ipynb` merge conflicts are effectively unresolvable
  (invariant 1) and it is his file.

## 4. Optional if time allows — S7, the truncated `.part` crash guard

`boatphone/onc_client.py:1829` calls `_wire_bytes_on_disk` outside any `try`, so a SIGKILL-truncated
`.part` file raises `EOFError`/`BadGzipFile` and aborts the whole run. Real robustness gap, recorded
in the handoff as not done. It only bites during a pull, so it is genuinely lower priority — but the
top-up pull is the likely next acquisition, and it is exactly the run that would hit it.

Two smaller handoff items, same tier: decision 0024's 4.9× compression ratio is unverified
(`gzip -l` suggests ~5.8×), and `read_fft_gz` is a misnomer for a reader that now handles a
mixed-container corpus (decision 0022) — `read_fft_product` is the accurate name, but renaming a
shared library function has blast radius across `fft_io` consumers and should be its own change.

## Verification

1. `python3 scripts/checks.py` — must go from **161 passed / 1 failed / 4 skipped / 8 not run** to
   fully green, with the 8 previously-unreached checks **actually running**. Report the real counts,
   not "green".
2. The migration script's own validation must pass over all 26,666 rows, and the resulting manifest
   must satisfy `requested == present + absent` and the 90-orphan reconciliation.
3. Confirm the corpus is byte-identical before and after — this plan must not touch `data/raw/`
   at all (invariant 2, hook-enforced).
4. Confirm no network call was made. If the driver ran, this plan was done wrong.
5. Clear notebook outputs before committing unless the output is the point.

This is the plan that greens the branch. Nothing merges until it lands.
