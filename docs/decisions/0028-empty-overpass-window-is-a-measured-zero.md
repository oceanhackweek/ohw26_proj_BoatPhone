# 0016. An empty 2.5 h overpass window is a measured zero, not an aborted run

Status: accepted
Date: 2026-08-27
Amends: [0008](0008-empty-listing-is-a-measured-zero.md)
Scope: `boatphone/onc_client.list_fft_files`, `scripts/pull_overpass_corpus.py`
Source: `docs/plans/acoustics_plan_v2.md` §5 "B3 -- Bulk acquisition"

## Context

Decision 0008 settled what an empty ONC listing means, and settled it correctly **for the span it
was reasoning about**. Its argument is explicitly a span-size argument: "a whole-study-window
query returning nothing is a broken request, not an empty ocean." At year scale that is right --
zero files across 2020-2025 for a device we hold a calibration file for cannot be data, so
`list_fft_files` raises `EmptyListingError` rather than returning `[]` and letting an
all-unavailable calendar pass for a measurement.

B3 inverts the premise without changing the code. `scripts/pull_overpass_corpus.py` calls the same
function once per in-season LOCAL date, over the **2.5 h PlanetScope overpass window**
(09:15-11:45 America/Vancouver). At that span an empty listing is the *ordinary* case: one outage
morning, one maintenance window, one gap in a six-year deployment. The known gaps in
`hydrophone_gaps.md` guarantee it will happen.

Two concrete consequences, both found in integration review before any real pull:

1. `run()` did not catch `EmptyListingError`, so the **first** quiet morning would kill an
   overnight six-season pull -- trading a real corpus for a false alarm.
2. A fully-empty window produced **no `absent_log` row at all**. `absent_log` recorded only files
   that were listed and then not retrieved, so a morning ONC listed nothing for appeared in
   neither bucket. `requested == present + absent` still held arithmetically while quietly
   ceasing to be a statement about coverage.

## Decision

**1. The premise change is explicit in the API, not hidden at the call site.**
`list_fft_files(..., allow_empty=False)` gains a keyword-only parameter. Default `False` keeps
0008 exactly as it stands for every year-scale caller (`build_uptime_calendar` and friends are
untouched). A caller working at a span where zero is ordinary passes `allow_empty=True` and
receives `([], empty_chunks)`, and thereby **takes on the obligation to record the empty window as
a measured zero itself**. Wrapping the existing call in `try/except` would have worked and was
rejected: the reason it is safe here is a property of the *span*, and a caller should have to say
so in the call rather than swallow an exception the library still believes in.
`scripts/pull_overpass_corpus.py` opts in through one named wrapper, `_listing_fn_allow_empty`.

**2. An empty window is a row in `absent_log`, with its own reason.** It carries
`"filename": None` (no file was ever named), `"reason": _REASON_EMPTY_WINDOW`, the local date, and
both UTC window edges. Same bucket as a 404 and as a decision-0007 measured zero -- there is still
no third bucket -- distinguishable only by `reason`, per B3's manifest contract.

**3. `requested` counts retrieval units, not filenames.** One per listed filename, **plus one per
window ONC listed no file for**. This is what makes `requested == present + absent` remain a true
statement about coverage: the pull did ask about that morning, and got an answer. Without it the
run reads as though the date was never attempted.

**4. `run()` also catches `EmptyListingError` per date** and produces the identical row. That is
not redundancy for its own sake: a `listing_fn` that has not opted in must still not be able to
abort a multi-season run at date 40 of 918.

**5. `empty_chunks` is carried into the manifest, not discarded.** `list_fft_files` returns
`(filenames, empty_chunks)`; the driver previously threw the second element away (in fact it
iterated the tuple, which is a separate bug fixed alongside). The manifest now carries
`empty_chunks_total` and, kept deliberately distinct, `empty_chunks_unreported_dates` -- dates
whose `listing_fn` reported no chunk count at all. An unknown is never written as a measured 0
(invariant 9).

## Consequences

* A six-season pull survives every quiet morning and says how many there were, on both the run's
  stdout summary and in the manifest.
* Downstream coverage analysis can now distinguish three things it previously could not: a file
  ONC listed and could not serve (404), a window ONC positively states could hold no data
  (0007, not deployed / not yet elapsed), and a window ONC served and had nothing in (this
  decision). Only the first is a possible bug.
* `requested` in a B3 manifest is **not** the number of files listed once any window is empty.
  Anyone reading the manifest as a file count must subtract `empty_windows`.
* The risk this accepts: if the ONC archive endpoint were to start returning empty for *every*
  window due to a broken request, B3 would now record 918 measured zeros instead of raising on
  the first. That is mitigated by the summary line -- an all-empty corpus is loud in the printed
  totals and in `empty_windows == n_dates` -- and is a deliberate trade, because the alternative
  fails on the ordinary case rather than the pathological one. **A B3 run whose `empty_windows`
  equals its date count must be treated as a broken request, not a dead hydrophone.**
