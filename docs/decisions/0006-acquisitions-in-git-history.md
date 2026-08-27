# 0006. The ONC delivery zip stays, and tracked acquisitions stay in history

Status: accepted
Date: 2026-08-27

## Context

The acoustics plan's A0 housekeeping step asked for two cleanups. Both are wrong, and both are the
kind of wrong that gets re-attempted by the next person who reads the plan, so they are recorded
here rather than silently skipped.

**1. Deleting `search78910238.zip` (71.7 MB).** The plan calls it "fully redundant -- it contains
only the calibration txt and the same WAV already extracted." That is precisely backwards as a
provenance argument: the zip is the **original ONC delivery envelope**, and the loose WAV and the
calibration `.txt` in the working tree are *derived from it* by extraction. Deleting the envelope
and keeping the extractions destroys the only artefact that proves what ONC actually served.
It also directly contradicts `docs/decisions/0001-raw-data-immutability.md` (acquisitions are
never deleted) and is denied by `.claude/hooks/raw-data-guard.sh`.

**2. Un-tracking the acquisitions already in git.** **127.3 MB across 7 files under `data/` is
already committed and pushed** (measured with `git ls-files -s` + `git cat-file -s`): the 71.7 MB
zip, the 50.1 MB FLAC, the 4.8 MB MP3, two `.fft.gz`, the calibration `.txt`, and `data/data.csv`.
A `.gitignore` rule does not un-track what is already tracked, so the tempting follow-up is
`git rm --cached`.

## Decision

**The zip stays.** It is an original acquisition under decision 0001.

**The 127.3 MB of already-tracked acquisitions stay in history.** No `git rm --cached`, no history
rewrite.

Two independent reasons, either sufficient:

- `git rm --cached` does not shrink history by a single byte -- the objects are already in every
  clone. What it *does* do is **delete those files from Neve's and Malachy's working trees on their
  next pull**, silently, in the middle of a one-week project. The cost is real and the benefit is
  zero.
- Actually shrinking history means rewriting it, and this repo is pushed and shared. CLAUDE.md
  invariant 8 puts force-pushing `main` off the table; `git revert` is the shared-branch undo, and
  it cannot remove a blob.

The `.gitignore` media rules therefore mean exactly one thing: **no *further* acquisitions enter
git.** The one-off `ICLISTENHF1266_...wav` line already in `.gitignore` is left in place for the
same reason. `data/samples/` is the deliberate, negated exception
(`0005-raw-acquisition-landing-zone.md`).

The acoustics plan's A0 text has been amended in place to say this, so the stale instruction stops
resurfacing.

## Consequences

- Every clone of this repo costs ~127 MB and always will. Accepted; the alternative breaks
  teammates mid-week.
- Because history is permanent, anything added to `data/samples/` is permanent too. Hence the
  size discipline in `data/samples/README.md`.
- If the repo is ever published or archived after the week, a one-time squash-to-a-fresh-repo is
  the option that remains available -- with everyone's work already merged, not mid-project.
