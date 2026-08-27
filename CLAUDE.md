# BoatPhone

Oceanhackweek 2026 project. **Cross-calibrating optical vessel detection against passive
acoustics in Barkley Sound.**

Satellites see every boat but only for a fraction of a second per day; the hydrophone listens
continuously but can't count. The project uses each to fix the other's blind spot: PlanetScope
imagery gives ground-truth vessel presence for short windows, which trains and validates an
acoustic model that then produces a **continuous** estimate of vessel counts -- with particular
interest in small recreational vessels, which carry no AIS and are therefore invisible to the
usual methods.

**Team:** Neve Foreman (optical detection / YOLO), Malachy McCaffrey (Planet imagery acquisition),
Isaac Guld (hydrophone data and acoustic modelling).

## Read these first

| File | What it is |
|---|---|
| `docs/plans/Project_Source_of_Truth.txt` | **Authoritative.** Goals, milestones, who owns what, next steps. Read at the start of any session; it overrides everything else. |
| `docs/agent-working-agreements.md` | How to work here: branching, the fleet, verification, orchestration. |
| `docs/decisions/0002-time-alignment-and-units.md` | The one that prevents the project's most likely wrong answer. |
| `docs/plans/` | Per-topic plans (`accoutics_plan.md`, `proposed_plan_IG.md`). |
| `references/` | Background reading and method notes. |

## Layout

```
contributor_folders/<person>/  per-person working notebooks (avoid merge conflicts)
final_notebooks/               shared final notebooks
boatphone/                     importable LIBRARY: constants, paths, credentials, loaders
scripts/                       runnable ENTRY POINTS (checks.py, acquisition CLIs)
data/                          IMMUTABLE acquisitions -- see below
data/samples/                  the one TRACKED small-fixture zone (decision 0005)
docs/plans/                    source of truth + plans
docs/decisions/                durable decision records
references/                    background research
.claude/                       agent fleet, hooks, commands
claude/run-phase/              local run state (gitignored)
```

Data present locally: an ONC/CIOOS Folger Deep hydrophone sample (ICLISTEN HF1266 -- FLAC, WAV,
MP3, gzipped FFTs, and a **pre-deployment calibration file**). PlanetScope imagery and AIS are
pending; write code that fails clearly when its input is absent rather than obscurely.

## Environment

The shared OHW JupyterHub environment (`/home/.pixi/envs/default`, Python 3.14). There is **no
pinned project environment file** -- no `pixi.toml`, no `environment.yml`, no `requirements.txt`.

Available: `numpy`, `scipy`, `xarray`, `pandas`, `matplotlib`, `geopandas`, `torch`,
`ultralytics`, `scikit-learn`, `dask`, `h5py`, `netCDF4`, `jupyter`/`nbconvert`.

**Not** available: `pytest`, `ruff`, `black`, `librosa`, `soundfile`, `obspy`, `rasterio`, `jq`.
Check before importing something not on the first list rather than assuming. If you install
something, say so -- it does not persist for teammates, and an analysis that silently depends on
a local install is not reproducible.

Consequence worth stating plainly: **there is no lint, no type check, and no test suite.** The
verification that exists is described in `docs/agent-working-agreements.md` §5, and the strongest
check available is re-executing a notebook top-to-bottom in a fresh kernel.

## Invariants

1. **Branch before the first edit.** Never work on `main`. A `.ipynb` merge conflict is
   effectively unresolvable and three people share this repo.
2. **`data/` is immutable.** Never edit, rename, or delete an acquisition
   (`docs/decisions/0001-raw-data-immutability.md`, hook-enforced). Derived products go to
   `data/derived/` with a record of what produced them.
3. **UTC end to end; state every convention.** Time base, sample rate, calibration, FFT scaling,
   coordinate frame -- read them from the data, never assume, and name them at every boundary
   (`t_utc_s`, not `t`). Prove the pipeline against a synthetic tone of known level, frequency,
   and time before anything depends on it. See `docs/decisions/0002-time-alignment-and-units.md`.
4. **Be suspicious of a result that works.** A clean correlation between vessels and acoustic
   energy is exactly what a time-alignment bug also produces. Check against a null -- shuffle the
   labels, shift the time base by an hour, use a quiet period -- and report that you did.
5. **Errors surface.** No bare `except`, no `fillna`/`interpolate` across a real recording gap,
   no truncating to fix a length mismatch. Each turns a data problem into a plausible number. If
   you drop samples, print how many.
6. **No magic numbers, and `boatphone/` is where the shared ones live.** Band edges, thresholds,
   window lengths, the hydrophone location, the study window, filesystem paths, the FAO size
   classes (0-12 m / 12-24 m / >24 m) -- named constants with their source in a comment, and
   **one** definition shared across notebooks, not one per person.
   The split, for all three workstreams: **`boatphone/` is the importable library** (constants and
   functions -- `paths.py`, `config.py`, `credentials.py`); **`scripts/` holds runnable entry
   points** (`checks.py`, acquisition CLIs) which import from `boatphone/` and define nothing
   shared themselves. This matters across teams, not just within acoustics: if optical puts the
   study window in `scripts/config.py` while acoustics has it in `boatphone/config.py`, the
   optical-acoustic matchup joins two different definitions of the same window and the join
   quietly produces a wrong answer rather than an error.
7. **Clear notebook outputs before committing** unless the output is the point; keep large files
   out of git; never commit a Planet or ONC API token.
8. **Integrate before you push.** Others are pushing to this repo. Pull when the remote is ahead
   (hook-enforced), and never force-push `main` -- `git revert` is the shared-branch undo. Never
   hand-edit conflict markers inside a `.ipynb`; take one side's file whole and re-run it.
9. **Distinguish "the method found nothing" from "the method is broken."** Only the second is a
   bug, and they need different follow-ups. Say which one you're looking at.

## The agent fleet

`.claude/agents/` holds a standardized fleet (planner, three coder tiers, test-author,
test-runner, and two reviewers) plus a report-only audit suite. `.claude/agents/README.md` is the
roster; `.claude/agents/_shared-standards.md` is what every agent reads first.

Use it for the parts of this project where being wrong is cheap to cause and expensive to
discover -- the acoustic-to-optical time join, the detection threshold, the range estimate, the
count estimator. For a single plot or a notebook tidy-up, just do the work directly; the
orchestration would cost more than it saves in a one-week project.

| Command | What it does |
|---|---|
| `/run-phase` | Orchestrate a milestone through the fleet. Hook-enforced: dependency ordering, a reviewer gate behind deterministic validation, an anti-stall Stop guard, a context budget with a validated handoff. |
| `/phase-review` | Full quality gate on the current branch. |
| `/promote-plan` | Move an approved plan from scratch into `docs/plans/` and reflect it in the source of truth. |
| `/architecture-audit` | Periodic drift audit -- duplicated method, diverging constants, unreproducible outputs, claims the code no longer supports. |

Hooks in `.claude/hooks/` enforce the invariants above (branch guard, `data/` immutability,
commit advisory, push guard) and the orchestration protocol. All of them **fail open** and are no-ops outside
an active `/run-phase` run, so they never wedge ordinary work.

## Working here

- Read `docs/plans/Project_Source_of_Truth.txt` first; it changes daily.
- Prefer adding to an existing notebook over creating a parallel one. When the same helper appears
  a third time, promote it to `scripts/`.
- Reproducibility is the deliverable: every figure in `final_notebooks/` should be regenerable
  from what is in the repo plus the shared storage.
- Update the source of truth when something lands. In a one-week project, a stale source of truth
  is the documentation failure that actually costs the team.
