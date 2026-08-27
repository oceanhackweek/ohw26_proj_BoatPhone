# Folger Deep — PlanetScope acquisition: session handoff

**Date:** 2026-08-27
**Notebook:** `planet_folger_search_order_download.ipynb`
**Status:** Gates 1 and 2 complete. Gate 3 (vessel review) not started. Nothing ordered.

---

## 1. Where things stand

```
552 search hits
 └─ gate 1  full AOI containment      ->  48   (172 partial, 332 slivers <50%)
     └─ gate 2  UDM2 >= 95% clear     ->  30
         └─ gate 3  vessel visible    ->  NOT STARTED
```

Survivors by year, after gate 2:

| year | scenes | `aoi_clear` range |
|---|---|---|
| 2020 | 2 | 1.000 |
| 2021 | 8 | 0.998 – 1.000 |
| 2022 | 4 | 1.000 |
| 2023 | 7 | 0.974 – 1.000 |
| 2024 | 5 | 0.975 – 1.000 |
| 2025 | 4 | 0.982 – 1.000 |

Glint angles 24.4°–45.3° (mean 35.1). `GLINT_FILTER` is `None`, so nothing was dropped —
it only orders the review queue.

### Budgets

| | spent | of | note |
|---|---|---|---|
| Scene tiles | **12,372** | 98,249 | 9,236 initial fetch + 3,136 repatch |
| Imagery | **0** | 3,000 km² | `CONFIRM_ORDER` is `False`; nothing ordered |

### Artifacts on disk (`planet_folger/`)

| file | what | cost to rebuild |
|---|---|---|
| `tile_previews/` | 48 previews, 3584 px, 3.15 m/px, ~224 MB | 9,408 tiles, ~25 min |
| `aoi_clear.json` | 48 UDM2 scores | 27 min |
| `gate2_survivors.csv` | the 30-scene review queue | seconds |
| `search_results.json` | 552 items (tracked in git) | seconds |
| `tile_ledger.json` | monthly tile spend | resets monthly anyway |
| `review_labels.json` | your gate-3 calls | **NOT regenerable — human judgement** |

Everything except `review_labels.json` is cache. That one is the only irreplaceable file,
and it is gitignored because it churns on every click — run the collect cell to write
`screened_candidates.csv`, which is the durable record and *is* tracked.

---

## 2. How to resume

**Two cells.** No auth, no search, no gate rebuild.

1. **Setup cell** — under `## 1. Setup`, starting `import asyncio, json, math, os`
2. **Reviewer cell** — under `### Record what you see`, starting `import ipywidgets as W`

Both the reviewer and the collect cell fall back to `gate2_survivors.csv` when `df` is not
in memory. When you finish labelling, run the **collect cell** to produce
`VESSEL_VISIBLE` / `UNSURE` and `screened_candidates.csv`.

### Do NOT "Run All"

- Cell 26 re-fetches **5,880 tiles** over previews that already exist (~7% of the monthly
  budget, 15–20 min, zero benefit)
- Cell 8 does a live search and rewrites `search_results.json`, which is tracked
- The collect cell would run before any labels exist, writing an empty result

Ordering (stage 9) does legitimately need the auth and AOI cells.

### Reviewer mechanics

30 scenes, opening on `20230815_182609_80_24c8` — sorted clearest-then-least-glinted, not
chronological. Overview on the left is **context only** (downsampled; a 12 m hull is ~1 px).
Detection happens in the 16 numbered panels, each a 1:1 native crop of 2.8 × 2.8 km.

Panel buttons turn blue as you open them and the count persists. **Positives are cheap** —
see a wake, click Vessel, move on. **Negatives are the expensive call**: "No vessel" claims
the whole 100 km², so it wants all 16 panels. The collect cell lists negatives called with
fewer than 16 panels opened.

Estimated 1–2 h for 480 panel views. **Unmeasured** — time two or three scenes and
extrapolate. If rendering drags, `PANEL_GRID = 3` gives 9 larger panels (fewer clicks, more
scrolling within each).

---

## 3. What was fixed this session

### Tile access — the blocker

Items on this plan carry no `_links.tiles`, and `_permissions` lists only
`assets.*:download`. The working route had to be supplied by hand:

```
https://tiles.planet.com/data/v1/PSScene/{item_id}/{z}/{x}/{y}.png
```

API key as HTTP basic user, or `?api_key=`. **No asset activation needed** — every asset
reports `status="inactive"` and tiles still render. Verified on all 48 coverage-gated
scenes, 2020–2025, PS2.SD and PSB.SD.

The earlier `200 / 820 bytes` result was a **fully transparent nodata tile**, not a working
route. The probe used `df.iloc[0]`, a scene covering 14% of the AOI that does not contain
the hydrophone, so the requested tile fell outside its footprint. The tile service answers
200 for any well-formed request and encodes "no data" in the **alpha channel** — status code
alone cannot distinguish a broken route from an uncovered target. Stage 7 now classifies
every tile `nodata` / `flat` / `imagery`, and the ledger counts only tiles actually served.

### Gate 1 — containment, not a percentage

Was `intersection.area / aoi.area >= 0.999` in degree space. Now `footprint.covers(AOI)` in
an equal-area projection. Two reasons: a 99.9% threshold still admits ~0.1 km² of nodata
(two scenes sat in that band), and a degree-space area ratio carries up to 6×10⁻⁴ of error
at 48.8°N — the same order as the 10⁻³ margin it was trying to discriminate on.

`COVERAGE_TOLERANCE_M2` (default `0.0`) is the knob if you later want to admit partials.

### Gate 3 — reviews the whole AOI

Was a 5 × 5 km centre crop = **25%** of the AOI. Scenes were being *selected* on 100 km² and
*judged* on a quarter of it, so a vessel in the outer 75% would score "no vessel". The review
box is now the AOI's own bounds: 14 × 14 = 196 tiles, 100% coverage.

### Previews at native resolution

`preview_batch` defaulted to `max_px=1400`, downsampling the 3584 px mosaic by 2.56× — from
3.15 to **8.05 m/px**. At that scale a 12 m charter falls from 3.8 px to 1.5. Now
`max_px=None`, `quality=95` (RMS compression error 2.7 DN vs 4.2 at q90). 2–9 MB/scene.

### AOI inset — 30 scenes actually fit

The nominal box measures 100.00024 km², ~240 m² **over** the slice the 3,000 km² quota
divides into 30 of. Since Planet's clip accounting is exact, 30 orders would ask 3000.007 km²
and the 30th would be refused. Stage 1 now insets 5 m per edge (`AOI_INSET_M`):

| billing basis | area | |
|---|---|---|
| AOI polygon | 99.800 km² | under — asserted |
| projected bounding box | 99.889 km² | under — asserted |
| bbox + full 3 m pixel snap | 100.009 km² | **marginally over** |

The bounding box matters because a lat/lon rectangle projects to a quadrilateral whose bbox
exceeds its polygon area by ~90,000 m², and a clipped GeoTIFF is a rectangle. The third row
is the pessimistic case; ~6 m of inset would close it.

### Reliability fixes

- **503 handling.** The tile service returns 503 under sustained load; the retry loop treated
  only 429 as transient and gave up immediately on 5xx. 16 scenes lost 2–20 tiles each,
  appearing as black squares indistinguishable from nodata. Now retries 5xx (6 attempts),
  logs exhaustion, and `fetch_mosaic` defaults to `concurrency=3` — at 3 all 16 repatched
  scenes came back clean. The fix is to stop provoking it, not to retry harder.
- **UDM2 caching + concurrency.** Was strictly sequential at ~4.9 min/scene (~4 h for the
  pool) because Planet generates each `ortho_udm2` on demand. Now concurrency 8: **27 min**.
  Scores cache to `aoi_clear.json` with atomic temp+rename writes, so the run is resumable.
- **Reviewer.** Replaces hand-typing scene IDs into a Python list. Tracks which panels were
  actually opened, so an under-inspected negative is visible rather than silent.

### Correction worth carrying forward

**"Flat" is not a filter.** `classify_tile` measures the 99th percentile of *raw* DN per
tile, while `stretch` is applied *globally* across the mosaic afterwards. A flat tile can
still show a bright wake. Scene `20200730_192942_71_1059` has 106 of 196 tiles flat and is
the scene with the clearest apparent wakes. Flat means "low overall contrast", not "nothing
detectable" — do not gate on it.

---

## 4. Open items

1. **Settle the billing basis before a full batch.** Order ONE scene and read the
   `quota_used` delta in stage 10. `100.00` confirms the polygon basis and ends the question;
   anything higher means raise `AOI_INSET_M`. One scene costs 1/30th of a month.
2. **The 30-scene batch consumes the quota exactly** (30 × 100.00 = 3,000.0) — no headroom
   for a re-order. Plan on 29 as the reliable number. Raising `AOI_INSET_M` does not buy
   headroom; the 100 km² clip minimum already dominates.
3. **Validate the stretch against ground truth.** Pick a date with a known vessel passage and
   confirm you can see it before trusting any negative.
4. **Year distribution is lopsided** — 2020 has 2 scenes against 2021's 8. Treat 2020 as a
   caveat, not a data point, in any interannual comparison. Stage 9 sorts by `aoi_clear` with
   no stratification.
5. **Confirm bundle names** with `planet orders bundles` before switching off `visual`.
6. **Tiles are screening only** — reprojected, lossy, no radiometry. Anything entering a
   figure or measurement comes from the ordered scene.

### If the working set falls below 30 after gate 3

Agreed fallback order — exhaust the cheap options first:

1. **Spot-check gate-2 rejects.** UDM2's cloud model is tuned for land and misflags glint,
   whitecaps and surf over water. A wrongly-rejected containing scene beats any partial, and
   previews already exist so it costs nothing.
2. **Relax `CLEAR_MIN`** (0.95). Note the survivors are mostly exactly 1.000 and the next
   candidates below the line are 0.905 and 0.878, then a steep drop to 0.518 — so this buys
   little.
3. **Admit partials** via `COVERAGE_TOLERANCE_M2`. Two scenes sit at ~9.5k and ~18k m²
   missing, then the next jump is 150k–580k. Skip `20250810_194613_51_253d` regardless — it
   renders almost flat. Partials cost the same 100 km² and make the detection *area* vary
   between observations, so record `aoi_missing_m2` rather than rediscovering it later.

---

## 5. Gotchas

**Never edit the notebook while it is open in JupyterLab.** A browser autosave silently
overwrote an entire session's edits on 2026-08-27; nothing was in git, so every cell had to
be rewritten by hand. Commit *before* a batch of edits, not only after. After any external
edit, close the tab without saving or use **File → Reload Notebook from Disk**, then grep for
a known string to confirm the reload actually took.

**`review_labels.json` is the one file that cannot be regenerated.** It is gitignored because
it changes on every click. Run the collect cell before ending a session so
`screened_candidates.csv` captures the judgement.

**Cell 10 (route probe) is diagnostic and skippable** — `deg2tile` lives in the setup cell so
stage 7 works without it.

**Working directory drift.** `cd` persists between shell calls; a relative path check that
reports a file "missing" may just be resolving from the wrong directory.
