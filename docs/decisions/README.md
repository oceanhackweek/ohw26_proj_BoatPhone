# Decision records

Short records of decisions that are **durable** -- ones later work builds on, where quietly
changing your mind breaks something upstream or makes two people's results incomparable.

This is not process for its own sake. In a one-week project with three people working in parallel
notebooks, the expensive failure is not a bug; it is two people using different band edges, or a
result that nobody can reproduce because the parameter that produced it lives in a deleted cell.
A record is five lines and prevents that.

## When to write one

Write a record when you decide:
- a **convention** others must match (time base, units, coordinate frame, calibration)
- a **parameter** with a defensible alternative (band edges, detection threshold, window length,
  join tolerance, size classes)
- a **data source or tool** the project now depends on (a model, an API, a dataset, a licence)
- a **scope cut** ("we are not doing AIS this week") that someone would otherwise re-open

Do *not* write one for something you can re-decide freely tomorrow at no cost.

## Format

Number sequentially, `NNNN-short-name.md`, with:

```markdown
# NNNN. <Title>

Status: accepted | superseded by NNNN
Date: YYYY-MM-DD

## Context
What made this a decision. What breaks if it is made inconsistently.

## Decision
What we do. Be specific enough to check against.

## Consequences
What this costs, what it rules out, and how to tell if it was wrong.
```

Superseding is normal -- mark the old one and link forward. Silently contradicting one is not.

## Records

| # | Title | Status |
|---|---|---|
| [0001](0001-raw-data-immutability.md) | Raw data under `data/` is immutable | accepted |
| [0002](0002-time-alignment-and-units.md) | Time base, sample rate, and units are UTC-first and stated | accepted |
| [0003](0003-hook-orchestration-enforcement.md) | `/run-phase` orchestration is enforced by hooks, not convention | accepted |
| [0004](0004-context-budget-and-handoff.md) | Long runs stop on a budget, via a validated handoff | accepted |
| [0005](0005-raw-acquisition-landing-zone.md) | `data/raw/<provider>/` is an append-only acquisition landing zone | accepted (amends 0001) |
| [0006](0006-acquisitions-in-git-history.md) | The ONC delivery zip stays, and tracked acquisitions stay in history | accepted |
| [0007](0007-onc-400s-that-are-measured-zeros.md) | Two ONC HTTP 400s are measured zeros, not failures | accepted |
| [0008](0008-empty-listing-is-a-measured-zero.md) | An empty ONC listing is a measured zero, and the calendar says so | accepted |
| [0009](0009-onc-pretrained-checkpoint-is-the-model.md) | ONC's pretrained checkpoint is the vessel-presence model | accepted |
| [0010](0010-analysis-band-and-small-craft.md) | The analysis band, and why >=250 Hz is not a limitation for small craft | accepted |
| [0011](0011-level-comparability-declarations.md) | Level comparability is declared at every pipeline boundary | accepted |
| [0012](0012-b0-model-viability-outcome.md) | B0 outcome: NO-GO on the pretrained-checkpoint path; B5 promoted to primary | accepted |
| [0013](0013-fft-axis-convention-is-an-open-assumption.md) | The FFT axis convention is an assumption, not a settled fact, worth +-125 Hz | accepted |
| [0014](0014-two-spectral-ceilings-and-the-408-exclusion.md) | Two spectral ceilings, kept apart, and nothing above bin 408 enters B5 | accepted |
| [0015](0015-censoring-aware-thresholds-for-b5.md) | B5's thresholds must be censoring-aware, and its free parameters are tuned off the overpasses | accepted |
| [0016](0016-small-craft-scope-is-fao-classes.md) | The small-craft scope is FAO classes 0-12 m and 12-24 m, applied at analysis time | accepted |
| [0017](0017-land-is-defined-by-extent-not-brightness.md) | Land is defined by extent, not by brightness, and NDWI is a diagnostic | accepted |
| [0018](0018-no-server-checksum-local-sha256-only.md) | No server checksum is available; integrity is a locally-computed sha256 of the wire bytes | accepted |
| [0019](0019-refuse-unknown-length-transfer.md) | A transfer of unknown length is refused, not accepted and hoped for | accepted |
| [0020](0020-pull-refines-only-the-overpass-window.md) | The pull refines availability only inside the overpass window it actually requested | accepted |
| [0021](0021-manifest-run-directories-and-latest-pointer.md) | Manifests live in per-run directories with a volatile "latest" pointer | accepted |
| [0022](0022-onc-serves-fft-as-plain-text-not-gzip.md) | ONC serves the `.fft` product as plain text, not gzip | accepted |
| [0023](0023-missing-file-is-http-400-api-error-96.md) | A missing archive file is an HTTP 400 with API error 96, not a 404 | accepted |
| [0024](0024-bulk-pull-compresses-on-write.md) | The bulk pull compresses on write; the corpus is permanently mixed-container | accepted |
| [0025](0025-already-compressed-names-are-stored-verbatim.md) | Already-compressed names are stored verbatim | accepted |
| [0026](0026-fft-level-ceiling-86-is-not-a-ceiling-for-this-corpus.md) | `FFT_LEVEL_CEILING = 86` is falsified for the corpus (levels reach 112) | accepted |
| [0027](0027-fft-product-is-viable-for-b5-band-level-detection.md) | The `.fft.gz` product IS viable for B5 band-level detection -- GO | accepted |
| [0028](0028-empty-overpass-window-is-a-measured-zero.md) | An empty overpass window is a measured zero, not a failure | accepted |
| [0029](0029-overpass-window-is-local-time-resolved-per-date.md) | The overpass window is local time, resolved per date via zoneinfo | accepted |
| [0030](0030-time-shift-null-does-not-collapse.md) | The time-shift null is UNRESOLVED -- confounded by unequal coverage on the two sides | accepted |

> **A numbering collision happened here, and this is how it was resolved.** The acoustics and
> optical workstreams both allocated **0016** and **0017** while working in parallel on
> 2026-08-27. The optical pair (FAO size classes; land-by-extent) reached `main` first and keeps
> those numbers. The acoustics pair was renumbered to **0028** and **0029** on the
> `milestone1/b3-bulk-acquisition` branch before it was pushed. Nothing else moved. If you find a
> pre-merge citation of "decision 0016" in acoustics code or docs, it means 0028; in optical code
> or docs, it means 0016 and is correct.
