# 0005. `data/raw/<provider>/` is an append-only acquisition landing zone

Status: accepted
Date: 2026-08-27
Amends: `0001-raw-data-immutability.md`

## Context

Decision 0001 says files under `data/` are immutable and sanctions writes only under
`data/derived/` and `data/processed/`. That was written when `data/` held exactly one thing: a
hand-delivered ONC sample.

Segment A4 changes the shape of the problem. It pulls **2.7-8 GB** of ONC hydrophone files -- the
largest write in this project -- with a cache-and-resume loop, so it must have a directory it is
allowed to write into, and it must be able to re-run without re-downloading what it already has.
Those files are acquisitions, not derived products: they are exactly what the provider served, and
`data/derived/` is the wrong home for them both semantically and because provenance for a derived
product means "what produced it", while provenance for an acquisition means "what served it".

Two things were already true and inconsistent with 0001 as written:

- `.claude/hooks/raw-data-guard.sh:19` **denied** every write under `data/raw/`, so the enforced
  rule contradicted the plan the team had already approved.
- The hook only covers the `Edit`/`Write`/`NotebookEdit` tools. **Python file I/O is not
  hook-covered at all**, so from A4 onward the acquisition module would write to `data/raw/`
  unimpeded while the hook still said no. The enforced rule and actual practice would diverge
  silently, which is the failure mode a decision record exists to prevent.

Separately, the blanket `*.flac` / `*.wav` / `*.mp3` / `*.fft.gz` / `*.zip` ignore rules made a
*deliberate* small test fixture uncommittable: a new `.fft.gz` dropped into the repo is silently
invisible to `git add`, sitting next to peers that are already tracked.

## Decision

**1. `data/raw/<provider>/` is an append-only acquisition landing zone.**

- Written **only** by the acquisition module (A1/A4). No notebook writes here; no analysis stage
  writes here.
- A file is immutable **once it is COMPLETE**. A partial or in-flight resumable download may be
  overwritten or truncated by the resume logic; the instant it is finished it is an original and
  falls under decision 0001 in full -- never edited, never renamed, never deleted.
- Path is a named constant, one definition: `boatphone.paths.RAW_DIR` and
  `boatphone.paths.ONC_RAW_DIR` (CLAUDE.md invariant 6). It was previously defined in
  `boatphone/credentials.py`, which is the wrong module for a filesystem layout constant, and
  which invited A4 to re-derive the path locally and create a second source of truth for the
  biggest write in the project.
- Gitignored in full. The landing zone never enters git.

**2. Nothing creates a `data/` subtree as a side effect.** `boatphone.paths.ensure_dir()` is the
one function that creates a directory, and the acquisition module calls it at download time.
`get_onc_client()` no longer calls `mkdir` -- merely constructing a client to check a token must
not materialise `data/raw/onc/`, which contradicted `paths.py`'s own stated policy that no
directory is created eagerly and `require_path()` raises with an acquisition hint instead.

**3. `data/samples/` is the one TRACKED fixture zone inside `data/`.** Small committed fixtures
only, negated out of the `.gitignore` media rules, with the reason written in `.gitignore` itself
so the next person finds the answer where they hit the problem. Contents are governed by
`data/samples/README.md`; fixtures are still immutable once committed.

**4. `.claude/hooks/raw-data-guard.sh` is extended** to permit `data/raw/` and `data/samples/`
alongside `derived/` and `processed/`, and still fails open. The hook cannot distinguish a
complete file from an in-flight one, so the "immutable once complete" half of this decision rests
on the acquisition module and on review, not on the hook.

## Consequences

- A4 can cache and resume without fighting a guard, and re-running it is cheap.
- `data/` now has four distinct regimes -- originals (immutable), `raw/` (append-only), fixtures
  (tracked, small), derived/processed (freely rewritable). That is more to hold in your head than
  0001's single rule, which is the cost of this amendment.
- The guarantee for completed acquisitions is now weaker than 0001's: enforcement is by convention
  and review inside one module rather than by the hook. Accepted because the hook never covered
  the Python writes that actually do the damage anyway.
- Anyone adding a provider adds `data/raw/<provider>/` as a constant in `paths.py` with a
  `_HOW_TO_OBTAIN` entry, not as a string in their own module.
