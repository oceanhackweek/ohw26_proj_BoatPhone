# 0021. Manifests get a durable run directory; the fixed-basename copy is a volatile pointer

Status: accepted
Date: 2026-08-27
Amends: none
Scope: `scripts/pull_overpass_corpus.py` (`run_output_dir`, `persist`)
Source: implementation of B3's bulk pull, `scripts/pull_overpass_corpus.py`

## Context

`scripts/pull_overpass_corpus.py`'s plan explicitly contemplates more than one invocation writing
into the same `manifest_dir`: a staged pull, and segment F's WAV second pass. Decision 0020
already establishes that consumers of a B3 manifest -- notably segment D, refining
`hydrophone_uptime.csv` -- must be able to tell what a given pull covered. If a second invocation
overwrote the first invocation's manifest and `absent_log` in place, the absent-file log, which is
a B3 deliverable and not a byproduct, would be destroyed by the very next run that touched the
same directory. That would happen silently: nothing about a second `run()` call failing to preserve
the first run's evidence would raise or print.

At the same time, `check_b3c_5` (which sabotages `manifest_dir/PROVENANCE_BASENAME`) and
`check_b3c_6`/`check_b3c_9` (which read it) were written against a fixed-basename path. Dropping
the fixed path in favour of only a per-run directory would have broken those checks, which
predate this decision.

## Decision

Every invocation of `run()` writes its manifest and provenance to **two** locations:

1. **The run directory**, `<manifest_dir>/runs/<UTC stamp with microseconds>/`, created by
   `run_output_dir()`. Uniqueness is deterministic, not probabilistic: directory creation is
   `mkdir()` without `exist_ok`, and a `FileExistsError` triggers a detect-and-suffix retry
   (`-1`, `-2`, ...). A whole-second stamp collides when two runs start in the same second, which
   is exactly what a scripted staged pull does; a 20-run back-to-back stress test produced 20
   distinct directories. This copy is never overwritten by a later run and is the **authoritative,
   durable** record of that invocation.
2. **The fixed basenames at the top level**, `<manifest_dir>/MANIFEST_BASENAME` and
   `<manifest_dir>/PROVENANCE_BASENAME`. This copy is a **volatile "latest" pointer**, and is
   deliberately overwritten on every run into the same `manifest_dir`.

**Write order is run-dir provenance, then latest provenance, then run-dir manifest, then latest
manifest.** This is not incidental ordering: the existing "no manifest without provenance"
invariant (`persist`'s docstring, unchanged by this decision) has to hold at *both* locations, not
just the run directory, or a failed sidecar write partway through could leave a stale-provenance
manifest pair sitting at the path most consumers actually open.

**The latest pointer was not made optional.** `check_b3c_5` sabotages
`manifest_dir/PROVENANCE_BASENAME` specifically, and `check_b3c_6`/`check_b3c_9` read it;
removing the fixed-basename path in favour of run directories only would have broken existing
checks that predate this work.

Two consumer-visible contracts follow from this, currently stated only in the module's docstrings
and made explicit here:

1. **The run directory is authoritative and durable; the fixed-basename path is a volatile
   convenience pointer to the most recent run.** Anything that needs a manifest or absent-file log
   to survive a *later* run into the same `manifest_dir` -- most concretely, segment D's read of
   the absent-file log across a staged pull plus segment F's WAV second pass -- must read the run
   directory it cares about, not the fixed path. Reading only the fixed path after a second run has
   landed silently substitutes the second run's coverage for the first's.
2. Both copies are written on every invocation; a reader that only wants "whatever is most recent"
   may use the fixed path, but must accept the volatility that name implies.

## Consequences

* A staged pull, or B3 followed later by segment F's second pass, cannot destroy an earlier run's
  manifest or `absent_log` as long as the consumer reads `runs/<run id>/` rather than the fixed
  basenames. `check_b3c_5`/`_6`/`_9` continue to exercise the fixed path unmodified.
* **Known limitation, accepted rather than fixed:** the latest pointers are rewritten on every
  mid-run flush (`_MANIFEST_FLUSH_EVERY_N_DATES = 10`), so a concurrent reader of the fixed path
  can observe a partial manifest mid-run. The run-directory copy has the identical property --
  `persist()` is called from the same flush loop and from `finally`, so a run-dir manifest can also
  be read while a later flush for the *same* run is still in flight. Nothing reads either path
  concurrently with a running pull today; this is a real gap, not yet exercised.
* **Known limitation, accepted rather than fixed:** `run_output_dir()` creates the run directory
  lazily, on first flush. A run that fails before any flush (before
  `_MANIFEST_FLUSH_EVERY_N_DATES` dates or the `finally`, whichever comes first, actually executes
  `persist()`) leaves no on-disk trace of itself at all -- no run directory, no updated latest
  pointer. This is correct behaviour, not a bug (an instantly-failing run has produced no evidence
  worth recording), but it does mean such a run is invisible to anyone scanning `runs/` afterward,
  and its failure is only visible in the process's own stdout/stderr at the time.
* Cross-references: decision 0020 governs `ONC_OVERPASS_CORPUS_DIR`, the directory this
  `manifest_dir`/`runs/` structure lives under; decision 0028 governs the content written into
  `absent_log` that this decision's run/latest split is protecting.
