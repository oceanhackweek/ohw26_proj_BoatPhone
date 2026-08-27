# 0001. Raw data under `data/` is immutable

Status: accepted
Date: 2026-08-26

## Context

Everything under `data/` is an original acquisition: hydrophone WAV/FLAC/FFT files pulled from
ONC/CIOOS (with their pre-deployment calibration files), PlanetScope scenes, and AIS extracts.
The project's entire claim -- that acoustic energy corresponds to vessels seen optically --
rests on those bytes being exactly what the provider served, at the timestamps the provider
recorded.

An edit to one of these is uniquely dangerous because it is invisible. Nothing downstream fails;
the analysis simply produces a number that cannot be reproduced from the source, and there is no
check that would catch it. Re-downloading is possible but slow, and for a scene or a deployment
window that has aged out of an API, sometimes not possible at all.

Filenames also carry meaning here (`ICLISTENHF1266_20260313T000000.029Z...`): instrument id and
UTC start time. Renaming is as destructive as editing.

## Decision

Files under `data/` are read-only. Never edit, overwrite, rename, or delete one.

Derived products go to `data/derived/` or `data/processed/`, and record what produced them --
the script or notebook, the input files, and the parameters.

This is enforced, not merely agreed: `.claude/hooks/raw-data-guard.sh` denies any Edit/Write/
NotebookEdit whose path falls under `data/` outside those two subdirectories.

## Consequences

- Cleaning, decimating, calibrating, and reformatting all produce new files. Disk is cheaper than
  an unreproducible result.
- Large acquisitions stay out of git (`.gitignore` plus the shared OHW storage); the guard
  protects whatever is present locally regardless.
- If an acquisition really is corrupt, the fix is to re-download it or to record the problem and
  exclude it in code -- not to repair it in place.
- Cost: an extra copy of anything you transform, and the discipline of naming derived outputs.
