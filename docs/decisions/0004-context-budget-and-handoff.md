# 0004. Long runs stop on a budget, via a validated handoff

Status: accepted
Date: 2026-08-26

## Context

A long `/run-phase` run degrades in a way that is invisible from inside it. By the last segment
the orchestrator is reasoning about new contracts through a context full of finished work: earlier
decisions get re-litigated, contracts get misremembered, and the quality of the last segments is
measurably worse than the first -- without anything failing.

Meanwhile, decision 0003's Stop guard refuses to end a turn with work remaining. That is right for
the stall it was built for, but it leaves no exit for a *healthy* run that should stop -- exactly
the degraded-context case. Without an exit, the guard forces the run to continue precisely when
continuing is the wrong call.

An unconditional "I'm done" button would solve that and immediately become the way every
inconvenient run ends.

## Decision

**A budget makes the degradation visible.** Each run carries a token budget (`--context-budget`,
then `$RUN_PHASE_CONTEXT_BUDGET`, then a per-model default). At 80% the spawn gate advises landing
the current segment and handing off; at 100% it denies new worker spawns while leaving everything
needed to wrap up open -- reading the ledger, validating, marking segments done, writing the
handoff. The orchestrator is never trapped mid-segment.

**The exit is gated on an artifact.** `run-phase-handoff.sh` pauses the run only once a handoff
document exists that a fresh session could resume *cold* from, covering: remaining segments by id
and what each still needs, contracts already settled, files touched, the current validation state,
and traps hit. The script validates it -- rejecting a document under 1500 bytes, or one that never
names a remaining segment id -- and only then pauses the ledger, after which every enforcement hook
steps back.

That gate is the whole design. Writing a real handoff is more work than finishing the segment
would have been in most cases, so it stays an exit for the case it was built for.

## Consequences

- Runs end deliberately and resumably rather than degrading quietly to a bad finish.
- `--resume` clears the pause and the stall counter, so a resumed run starts clean.
- The budget only engages when context usage is actually reported (`$RUN_PHASE_CONTEXT_USED`); if
  it can't be determined, nothing limits, and the Stop guard remains the backstop.
- The size and segment-naming checks are proxies for "is this a real handoff." They can be gamed
  by anyone determined to. They are meant to stop drift, not adversaries.
