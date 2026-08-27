# Agent working agreements

How Claude agents work in this repository. `CLAUDE.md` carries the project context; this file
carries the process. `.claude/agents/_shared-standards.md` is the lean pointer every agent reads
before acting.

**Precedence:** `docs/plans/Project_Source_of_Truth.txt` > this file > `CLAUDE.md` >
`docs/decisions/`.

---

## 1. What this project is, and what that changes

A one-week Oceanhackweek project, three people, whose output is notebooks, figures, and a
presentation -- not a shipped application. Adapted from a production software fleet, so a few
things are deliberately different from what those agent definitions would normally assume:

- **There is no build.** Nothing type-checks, nothing lints, and there is no test suite. The
  closest thing to CI is "does the notebook run top-to-bottom in a fresh kernel."
- **The expensive errors are silent, not loud.** A production bug throws. A time-base error here
  produces a plausible correlation and a confident wrong answer. Verification is aimed there.
- **Three people work in parallel folders that nobody reviews.** Duplication and divergence are
  the normal failure, not architectural decay.
- **The deadline is real and close.** Process that doesn't pay for itself inside a week is not
  worth running. Use the fleet for the parts where being wrong is expensive; do the rest directly.

## 2. Branching

**Branch before the first edit.** Never do analysis work on `main`
(`.claude/hooks/branch-guard.sh` asks before letting you).

`milestoneN/short-name` for milestone work, `explore/short-name` for exploration that may be
thrown away, `docs/short-name` for documentation.

This matters more here than in a normal repo: a merge conflict inside a `.ipynb` is effectively
unresolvable by hand, and three people are pushing to one repository. Related, and cheap: clear
notebook outputs before committing unless the output is the point (`.claude/hooks/commit-advisory.sh`
reminds you), and keep an outputs-clearing commit separate from an analysis commit so the diff
stays readable.

### Integrate before you push

`.claude/hooks/push-guard.sh` checks the remote before any `git push` and **asks** when it has
commits you don't have locally. Git would reject that push anyway as non-fast-forward -- the
point of catching it first is that you integrate deliberately rather than under pressure with a
failed push on screen.

The guard checks with `git ls-remote`, not `git fetch`, so it never mutates your remote-tracking
refs, and it fails open on a slow or unreachable network (10s cap).

**When a `.ipynb` conflicts, do not hand-edit the conflict markers** -- the notebook JSON will not
survive it. Check out one side's version of the file whole, re-run it, and commit that. If both
sides made real changes to the same notebook, that is a conversation in Slack, not a guess.

**Force-pushing to `main` is denied outright.** It permanently discards whatever a teammate
pushed since your last fetch, with no warning and no recourse for anyone who hasn't fetched. To
undo a bad commit on a shared branch, `git revert` it. Force-pushing your own branch is asked
about, not blocked; prefer `--force-with-lease`.

## 3. Data

`data/` is immutable (`docs/decisions/0001-raw-data-immutability.md`, enforced by
`.claude/hooks/raw-data-guard.sh`). Derived products go to `data/derived/` and record what
produced them: the script or notebook, the input files, and the parameters.

Large acquisitions do not go in git -- they live in the shared OHW storage. Code that needs one
must fail clearly when it is absent, not fail obscurely three cells later.

## 4. The time/units gate (the one that matters)

`docs/decisions/0002-time-alignment-and-units.md` is the most important document in this
repository. Read it before writing any code that touches a timestamp, a sample rate, a level, or
a position.

In short: UTC end to end; read the sample rate from the file; apply calibration before comparing
levels; state the convention at every boundary; prove it with a synthetic tone of known level,
frequency, and time before anything depends on it; and check a working result against a null
before believing it.

In `/run-phase` this is always its own gated segment, and it comes first.

## 5. The verification standard

Work is done when:

- the code runs clean **top-to-bottom in a fresh kernel** (restart, run all -- not "it worked
  when I ran the cells in the order I happened to run them")
- new analysis has a check that would **fail if it were wrong** -- ideally a synthetic case with
  known ground truth
- units and time bases are **stated** at every boundary
- constants are **named**, with their source in a comment
- anything written to `data/derived/` carries its **provenance**
- a result a human will read states **what data it came from**: the time range, the sample size,
  and the threshold

**A skipped check is not a pass.** Most tooling is absent from the hub environment; the honest
report is "nothing was verified," not a green tick.

## 6. The fleet

See `.claude/agents/README.md` for the roster and `.claude/agents/_shared-standards.md` for the
rules every agent starts from.

**Lanes are real.** test-authors don't write analysis code; coders don't write their own proof
checks; test-runners and reviewers don't edit. The point is that the thing checking the work is
not the thing that produced it.

**Escalate on contact with time or calibration.** Any task that turns out to touch a time base,
a sample rate, or a calibration constant becomes `coder-complex` work regardless of how small it
looked.

**Subagents cannot spawn subagents.** Multi-agent flows are driven from the main conversation via
`/run-phase` or `/architecture-audit`.

## 7. Async orchestration

The `Agent` tool is asynchronous: spawn -> **end your turn** -> resume on the completion
notification. Do not narrate waiting.

During an active `/run-phase` run this loop is enforced by hooks rather than by convention
(`docs/decisions/0003-hook-orchestration-enforcement.md`): a dependency gate, a reviewer gate that
stays closed until deterministic validation greens, a Stop guard that refuses a stall stop, and a
context budget with a validated handoff exit
(`docs/decisions/0004-context-budget-and-handoff.md`).

Every hook fails open and is a no-op outside an active run, so ordinary work is untouched.
Run state lives in `claude/run-phase/` -- gitignored, local-only, and deliberately out of `docs/`
so concurrent runs by different people never collide with the shared documentation tree.

## 8. Documentation policy

`docs/plans/Project_Source_of_Truth.txt` is what the team actually reads. Keep it current: an
approved plan goes into `docs/plans/` via `/promote-plan` and is reflected there, not left in
`~/.claude/plans/` where nobody else can see it and where it vanishes at the end of the week.

Durable decisions get a record in `docs/decisions/` (see that README for what counts). Background
reading goes in `references/`.

The most consequential documentation failure in a week-long project is the source of truth going
stale while the code moves. `/architecture-audit` checks for exactly that.

## 9. Commands

| Command | What it does |
|---|---|
| `/run-phase` | Orchestrate a milestone through the fleet, hook-enforced. |
| `/phase-review` | Full quality gate on the current branch. |
| `/promote-plan` | Move an approved plan from scratch into `docs/plans/`. |
| `/architecture-audit` | Periodic drift audit across notebooks, scripts, and docs. |
