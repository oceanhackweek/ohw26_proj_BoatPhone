# 0003. `/run-phase` orchestration is enforced by hooks, not convention

Status: accepted
Date: 2026-08-26

## Context

The `Agent` tool is asynchronous: a spawn returns "launched in the background," the orchestrating
turn ends, and the harness re-invokes on the completion notification. A multi-agent pipeline
therefore has no single continuous thread of control -- it is a state machine spread across many
turns.

Left to convention, that state machine fails in specific, repeatable ways:

- The orchestrator ends a turn mid-run with work remaining and nothing running. The run silently
  stops and the human doesn't notice for twenty minutes.
- It spawns a dependent step before its dependency finished, so a coder runs against checks that
  don't exist yet.
- It hand-maintains its own notion of what's in flight, which drifts from reality.
- It drifts into doing the work itself instead of delegating, defeating the point of the fleet.
- It spends expensive reviewer tokens on work that doesn't even run.

Each of these is an instruction the orchestrator can be *told* not to do, and each is one it will
eventually do anyway across a long run, because instructions decay as context fills.

## Decision

The enforcement lives in hooks, which are deterministic and don't forget:

| Hook | Event | Enforces |
|---|---|---|
| `run-phase-spawn-gate.sh` | PreToolUse/Agent | dependency ordering, the reviewer gate, the context budget |
| `run-phase-spawn-record.sh` | PostToolUse/Agent | records inflight, injects the async resume protocol |
| `run-phase-complete.sh` | SubagentStop | reconciles a finished worker out of inflight |
| `run-phase-stop-guard.sh` | Stop | blocks the stall stop; reconciles against the harness's running list |
| `run-phase-readonly-guard.sh` | PreToolUse/Edit·Write | keeps the orchestrator from doing the work |

State lives in a hook-owned JSON sidecar at `claude/run-phase/run-phase-<branch>.json`. **The
orchestrator never writes it**; a human-readable `.md` alongside carries the narrative. Two
escape hatches are sanctioned and audited (`run-phase-segment.sh`, `run-phase-reconcile.sh`);
hand-editing the JSON is not.

Three properties make this safe rather than another way to get stuck:

1. **Fail open.** Every hook treats any error, missing tool, or ambiguity as "not in an active
   run" and allows. A broken ledger can never wedge a normal turn.
2. **Scoped to an active run.** With no active ledger for the branch every hook is a no-op, so
   ordinary multi-agent exploration is untouched.
3. **Bounded blocking.** The Stop guard escalates to `NEEDS_HUMAN` after three blocked stops with
   no change in the run's shape, which disengages every hook. A guard that can trap the human is
   worse than the stall it prevents.

The JSON layer is Python (`rpl.py`), not `jq` as in the fleet this was ported from: `jq` is not
installed in the OHW hub image and `python3` always is.

## Consequences

- Ledger state is trustworthy enough to reason from, because only the hooks write it.
- The orchestrator's instructions can be shorter -- the hook says no, so the prompt doesn't have
  to keep repeating it.
- Cost: ~700 lines of shell and Python to maintain, and hooks that fire on every turn. Kept
  tolerable by fail-open behaviour and the active-run scoping.
- The read-only guard ships in `observe` mode (logs, allows) until the field distinguishing a
  worker's tool call from the orchestrator's is confirmed from real payloads. Denying writes on
  an unverified assumption would block workers, which is the worse error.
