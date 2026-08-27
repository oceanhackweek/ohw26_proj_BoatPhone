# Environment audit -- 2026-08-27

Measured, not assumed. Produced by:

```
PYTHONPATH=. python3 -m boatphone.env_audit
```

plus `importlib.metadata.version(...)` for the packages outside `REQUIRED`/`OPTIONAL`,
and `which ffmpeg flac bellhop` for the CLI binaries.

**Interpreter:** `/home/.pixi/envs/default/bin/python` -- Python **3.14.6**
(the shared Oceanhackweek JupyterHub environment; there is no project-pinned env).

## Present

| Package | Version | Used for |
|---|---|---|
| onc | 2.6.0 | ONC Oceans 3.0 client -- hydrophone acquisition (no `__version__` attribute; version comes from `importlib.metadata`) |
| numpy | 2.5.2 | arrays |
| scipy | 1.18.0 | spectrograms, filtering, resampling |
| xarray | 2025.9.0 | labelled time/frequency arrays |
| pandas | 3.0.5 | tabular joins (optical <-> acoustic) |
| pyproj | 3.7.2 | geodesy -- hydrophone/vessel ranges |
| pyarrow | 22.0.0 | parquet for `data/derived/` |
| matplotlib | 3.11.1 | figures |
| torch | 2.12.0 | YOLO / optical detection |
| scikit-learn | 1.9.0 | the count/regression model |
| pyyaml | 6.0.3 | manifest parsing in `scripts/checks.py` |
| tzdata | 2026.3 | IANA time-zone database behind `zoneinfo`; A4's DST-aware `America/Vancouver` window. Present on the hub but was missing from `environment.yml` -- added, because off-hub (notably Windows) `zoneinfo` has no system fallback |

**Present as CLI binaries** (both at `/home/.pixi/envs/default/bin/`): `ffmpeg`, `flac`.

## Absent

| Missing | Consequence |
|---|---|
| `arlpy` | gates **A8** (propagation modelling) |
| `pypam` | gates **A5** (decidecade / third-octave band levels) |
| `mbari-pbp` (imports as `pbp`, **not** `mbari_pbp`) | gates **A5** (HMD products) |
| `soundfile` | FLAC reads only -- not needed, see below |
| `python-dotenv` | none: `boatphone/credentials.py` parses `.env` with the standard library |
| `pytest` | no test runner; `scripts/checks.py` is the deterministic check suite instead |
| `bellhop` binary | `arlpy` ray tracing needs an acoustics-toolbox FORTRAN executable that is **not** a pip dependency. `pip install arlpy` would not make A8 work on its own. |

**Nothing was installed.** The hub environment is shared with teammates and none of the
absent packages is needed before A5; installing into it would change their environment
and would not persist as a reproducible dependency of the analysis anyway.

## Where the plan's A0 text is wrong

Recorded here rather than silently worked around.

1. **The deletion instruction conflicts with decision 0001.** A0 asks for
   `data/.../search78910238.zip` (71.7 MB) to be deleted. `data/` is immutable
   (`docs/decisions/0001-raw-data-immutability.md`, hook-enforced): acquisitions are
   originals and are never edited, renamed, or deleted. It stays. Recorded durably in
   `docs/decisions/0006-acquisitions-in-git-history.md`, and the plan's A0 text has been
   amended in place so the instruction stops resurfacing.
2. **`.gitignore` does not un-track what is already tracked.** The new
   `*.zip`/`*.flac`/`*.wav`/`*.mp3`/`*.fft.gz` rules prevent *future* additions only.
   **127.3 MB across 7 files under `data/` is already committed and pushed** (measured
   with `git ls-files -s` + `git cat-file -s`): the 71.7 MB zip, the 50.1 MB FLAC, the
   4.8 MB MP3, two `.fft.gz`, the calibration `.txt`, and `data/data.csv`. Removing
   them from the index would not shrink history and would break teammates' checkouts,
   so no `git rm --cached` was run. The one-off ICLISTEN `.wav` line already in
   `.gitignore` was left in place for the same reason. See
   `docs/decisions/0006-acquisitions-in-git-history.md`. `data/samples/` is negated back
   in as the one tracked fixture zone (`docs/decisions/0005-raw-acquisition-landing-zone.md`).
3. **The plan's directory list does not match disk.** `data/raw/` and `data/interim/`
   **do not exist**; `data/derived/` and `data/processed/` were omitted from the plan's
   list but are the destinations CLAUDE.md actually names. All four are defined as
   constants in `boatphone/paths.py` (`RAW_DIR`/`ONC_RAW_DIR`, `INTERIM_DIR`,
   `DERIVED_DIR`, `PROCESSED_DIR`) and ignored by `.gitignore`; none is created
   eagerly -- `require_path()` raises with an acquisition hint instead, and
   `ensure_dir()` is the single explicit place that creates one at write time
   (`docs/decisions/0005-raw-acquisition-landing-zone.md`). `ONC_RAW_DIR` previously
   lived in `boatphone/credentials.py`, which made this paragraph false; it does not
   any more.
4. **`soundfile` is unnecessary.** Its only role here is reading the FLAC, and both
   `flac` and `ffmpeg` are already on `PATH`; decoding to WAV via the CLI avoids adding
   a package to a shared environment for a one-line job.

## Conventions this segment pins

- **UTC end to end.** Nothing in A0 is time-bearing yet, but every downstream boundary
  names its base (`t_utc_s`) -- `docs/decisions/0002-time-alignment-and-units.md`.
- **Credentials:** the process environment overrides `.env`; an absent, empty, or
  placeholder `ONC_TOKEN` raises `MissingCredentialError` rather than returning `None`
  (an unauthenticated request returns an empty download that reads as "no vessels").
  No token value ever enters an exception message, a log line, or a repr.
  **That guarantee covers `boatphone`, not the `onc` client it hands back.**
  `onc.ONC` stores the token as `self.token`, so a committed notebook cell containing
  `vars(client)` (or `client.__dict__`, or `%xmode Verbose`, which prints every local
  in each traceback frame) would put the token into git in plain text. Keep the client
  out of cell output.

## B0 out-of-tree installs -- now disposable (2026-08-27)

B0-5/B0-6 needed five packages to even attempt loading the ONC SSAMBA/Vision-Mamba checkpoint:
`timm` 1.0.11, `einops` 0.8.2, `mamba_ssm` 2.2.5 (built without the CUDA extension -- no `nvcc`,
no GPU), `triton` 3.7.1, `transformers` 5.16.1.

**All five were installed out-of-tree only**, via `pip install --no-deps --target
<scratchpad>/pylibs` plus `PYTHONPATH=<scratchpad>/pylibs`. **Nothing was installed into
`/home/.pixi/envs/default`.** None of it persists for teammates, none of it is a dependency of
any notebook in `final_notebooks/` or `contributor_folders/`, and none of it is required by
`scripts/checks.py`.

Per decision `0012`, B0 returned **NO-GO** on the whole pretrained-checkpoint path (independent
of these packages loading at all -- the checkpoint has no usable Engine Noise output and scoring
the corpus once would cost ~12 CPU-days). **These five packages are therefore disposable.** They
are recorded here only so a future session does not wonder where `boatphone/onc_model_cpu.py`'s
imports were meant to come from.
