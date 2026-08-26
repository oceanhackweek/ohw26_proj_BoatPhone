# Option C — two-detector screening for clean single-vessel scenes

**OHW26 · PlanetScope × Folger Passage hydrophone** · Bamfield Marine Sciences Centre
**Status:** execution document. Chosen over Options A and B in `detection_model_tradeoff.md`.

| | |
|---|---|
| **Team** | 3 people — A (acoustics) / B (imagery) / C (detection) |
| **Time remaining** | **Wed 2.5 h + Thu 5 h + Fri 5 h = 12.5 h** |
| **Detection budget** | ~3 h Thursday, plus 1 h verification |

> **This document supersedes §3 of `project_ideas_ohw26.md`** (detection) and **changes §2**
> (the analysis) from a summed-source model to a **single-dominant-source** model.
> §4 (data decisions), §5 (acoustics) and §8 (limitations) of that document still stand.

> **The one thing that must not slip today:** the Planet order goes in **before you leave
> Wednesday**. See §4. Everything else in this document is recoverable; that is not.

---

## 1. What changed, and why it makes the week easier

Three decisions since `detection_model_tradeoff.md` was written:

**① We are aiming for scenes with one dominant boat, not scenes with many boats.**
The original plan summed every vessel's contribution and fitted the total. That needs high recall
everywhere, and every missed boat corrupts the sum. If instead we keep scenes where a *single*
vessel dominates the expected received level, the acoustic model collapses to a straight line:

```
RL  =  SL  −  k · log10(r)
```

One boat, one range, one point on a scatter plot. Fit a line through ~15–25 points and read off
`SL` and `k`. This is a much smaller thing to get right in the time we have.

**② Empty scenes became data, not waste.**
Scenes with no vessels give the **ambient noise floor** — waves, wind, distant shipping, and
traffic outside our box. We need that number. Without subtracting it, distant boats will look
mysteriously loud, because at long range we are mostly measuring the ocean rather than the boat.
"No boats here" is now a row in the results table.

**③ NIR is not a separate purchase.**
`ortho_analytic_4b(_sr)` is **one file containing Blue, Green, Red *and* NIR**. The `visual`
asset is listed separately in `project_ideas_ohw26.md` only because it is Planet's colour-corrected
8-bit render, matching the Sentinel-2 L1C TCI domain the model was trained on — not because it
carries different bands. So Option C's NIR half is close to free.

### What this does to the detector's job

The detector is no longer answering *"how many boats are there?"* It is answering:

> **"Is this scene clean, and how far away is the one boat that matters?"**

The dangerous failure inverts accordingly. Missing a **distant** boat now barely matters — it
would have contributed little sound anyway. Missing a **second, closer** boat is severe: the
scene gets recorded as "quiet distant source" when the hydrophone actually heard a loud near one.
A single such error can drag the whole fit.

**Everything below follows from that one sentence.**

---

## 2. What Option C is

Keep `yolo11s_tci.pt` **frozen**. No fine-tuning, no training annotations. Make the *pipeline*
PlanetScope-specific by running two detectors whose failure modes do not overlap:

| | **Detector A** | **Detector B** |
|---|---|---|
| **Input** | 8-bit RGB | NIR band |
| **Principle** | learned vessel **shape** | **contrast** — bright object on near-black water |
| **Fails on** | sub-10 m craft at 3 m GSD, sun glint, wakes | dark hulls, whitecaps/foam, shoreline bleed |
| **Needs training data** | no (pretrained) | no (adaptive threshold) |

**Detector B's primary job is not to find more boats for the fit.** It is to *independently sweep
for anything A missed*, so we can assert that a scene is clean. That is the guarantee the
single-dominant-source analysis rests on, and Detector A alone cannot provide it — the model
physically never saw the NIR band, so it cannot be checking it.

This is also what makes the method ours rather than borrowed: we are exploiting a band the
TCI-trained model is structurally incapable of using.

---

## 3. The two results this produces

**Methods result — the one nobody has published.**
Recall of the pretrained Sentinel-2 model on 3 m PlanetScope, stratified by vessel length class:

> *"The pretrained S2 detector recovers X % of NIR-confirmed small craft, and its recall degrades
> below Y m."*

Its reported F1 of 0.863 is a Sentinel-2 number at 10 m. What it does at 3 m on sub-10 m craft is
genuinely unknown, and we are in a position to measure it.

**Science result — the project.**
`SL` and `k` fitted on clean single-source events, with `k` quotable as the empirical spreading
coefficient at Folger Passage (15 ≈ intermediate/cylindrical, 20 = spherical).

---

## 4. ⏰ Before you leave Wednesday — the order

**Owner: B. Nothing here is recoverable Thursday.**

### The recommendation

```
ortho_analytic_4b_sr  +  ortho_udm2     for ALL ~30 scenes
ortho_visual                            for 2–3 scenes only  (calibration check)
```

**Why analytic for everything:** it carries NIR, and NIR is what the cleanliness check in §8 runs
on. Every scene we use in the fit needs to be checkable. A scene without NIR is a scene we cannot
certify as clean, which under the new framing makes it nearly useless.

**Why `visual` on 2–3 scenes:** both assets come from the **identical acquisition**, so they give
a perfectly controlled comparison for the check in §5 — same water, same boats, same instant, two
renderings. Two or three is enough to settle it.

### Two five-minute checks, whoever holds the Planet account

Both were already flagged as open in `project_ideas_ohw26.md` §4 and both change this order:

1. **Is quota charged on the clip polygon, or on the delivered bounding box?**
   A 5 km-radius AOI is 79 km² as a circle but ~100 km² as the delivered square. That is the
   difference between ~38 scenes and ~30.
2. **Does ordering two asset types of one scene charge twice?**
   ***If it does not, order `visual` for everything and §5 becomes unnecessary.*** Check this
   first — it is the single highest-leverage question tonight.

### Also confirm

- **Which instrument?** If the scenes are SuperDove (`PSB.SD`), `analytic_8b_sr` adds RedEdge,
  Yellow and Coastal Blue. Do not switch without confirming quota and licence treatment — 8-band
  may be charged or licensed differently. NIR is all we actually need.
- The four quota rules from `project_ideas_ohw26.md` §4 still apply: **square AOI**, filter
  cloud loosely (<25 %) and confirm from UDM2 after delivery, **dedupe by acquisition time**,
  and order only scenes with confirmed hydrophone coverage.

---

## 5. Building the model's RGB input from the analytic asset

**Owner: C · Thursday +1:00–1:30 · skip entirely if the quota check above says `visual` is free**

The model wants what Sentinel-2 L1C TCI looks like. We need to produce that from 16-bit analytic
reflectance.

1. **Use a fixed linear scale, not a per-scene stretch.** The L1C TCI convention is a fixed
   mapping (scaled reflectance ÷ 10, clipped to 0–255). A per-scene percentile stretch would make
   each scene's radiometry depend on its own content — which is *exactly* the domain drift we
   chose a pretrained model to avoid. One scene's dark water and another's bright water would be
   rendered identically, and the model would see two different worlds.
2. **This means changing existing code.** `vessel_detect.py:98-107` currently applies a 2–98 %
   percentile stretch as a *defensive fallback* for when someone passes an analytic asset by
   mistake. Under this plan that path becomes the **primary input path**, and the percentile
   stretch must be replaced with the fixed scaling. Do not leave it as-is.
3. **Know what we are not sure about.** `_sr` is *surface* reflectance; L1C TCI is
   *top-of-atmosphere*. Surface reflectance reads darker over water because the atmospheric path
   radiance has been removed. Whether that gap matters to the model is an empirical question, not
   something to reason our way to.
4. **Acceptance test — this is the whole point of the 2–3 paired scenes.** Run Detector A twice
   on each paired scene: once on `visual`, once on derived-RGB. Compare detection count and box
   positions.
   - **Agree** → proceed on analytic-only. NIR is now available everywhere, for free.
   - **Diverge badly** → order `visual` for the full set if quota allows, and demote analytic to
     NIR-only. Note this costs us NIR-based verification on some scenes, so say so in §13.

**Do not skip the acceptance test.** A silently wrong radiometric transform looks exactly like a
model that does not work on PlanetScope, and we would draw the wrong conclusion from it.

---

## 6. Detector A — recalibrate, do not retrain

**Owner: C · Thursday, spread across the morning · all of it reuses `vessel_detect.py`**

Ordered by expected payoff. Do them in this order.

1. **Settle channel order first.** Ultralytics treats a numpy array as BGR and flips it
   internally; SAHI's wrapper does the opposite. `--sweep` already tests both
   (`vessel_detect.py:395-398`). An inverted channel order is a **silent** accuracy killer — no error,
   just bad results — so resolve it before tuning anything else, and never tune on top of it.

2. **Change what the sweep optimises. This is the most important line in this section.**
   `run_sweep()` currently ranks configurations by detection **count**
   (`vessel_detect.py:373-405`). Count is maximised by **sun glint**, not by finding boats. As it
   stands, the sweep will confidently recommend the glintiest configuration available.
   Re-rank by **recall against the eyeball truth set** from §8. Until §8 exists, the sweep has
   nothing meaningful to optimise against — so it runs *after* verification, not before.

3. **Sweep tile geometry:** `slice ∈ {256, 320, 512} × imgsz ∈ {640, 1024}`. The model's real
   constraint is not metres, it is **how many pixels a vessel spans at network input**. mayrajeo
   trained on 320→640 tiles, so a 20 m vessel at 10 m GSD arrived as ~4 px. At 3 m a 10 m boat is
   ~3.3 px and a 320→640 slice presents it at ~6.6 px. We are landing apparent object scale in the
   model's comfortable range while retaining every native pixel. **Never resize the whole scene.**

4. **Drop `conf` well below the 0.25 default — try 0.05–0.10.** For a domain-shifted model, recall
   at low confidence is the thing that matters. Precision gets recovered downstream by the water
   mask, the size prior and the NIR cross-check. Do not use the confidence threshold as the
   precision tool; it throws away exactly the marginal small-craft detections we care about.

5. **Apply the water mask to A as well as B.** The S2 model never learned to reject 3 m shoreline —
   at 10 m that texture did not exist. Reuse the NDWI mask from §7, which is now available on every
   scene. Expect this to remove a large fraction of A's false positives at low `conf`.

6. **Size prior and glint.** Tighten `--min-length` so single-pixel noise cannot pass the filter
   (`--max-length 20` already implements the small-vessel cut). Characterise glint on the paired
   scenes and, if it clusters in one part of the image, consider a positional rejection.

---

## 7. Detector B — NIR screening

**Owner: C · Thursday +1:30–2:30 · ~1 hour of numpy, no training-distribution problem at all**

1. **Read** bands B/G/R/NIR from `analytic_4b_sr`.
2. **Build the water mask.** NDWI = (Green − NIR) / (Green + NIR); threshold for water.
   **Erode by ~3 px** — the Broken Group is 100+ islands and bright shoreline is the single
   dominant false positive here. Combine with UDM2 clear (`read_udm2_clear()` already exists,
   `vessel_detect.py:124`) and the scene validity mask.
3. **Threshold adaptively, inside water only.** Candidate = NIR above `local water median + N·σ`.
   Use MAD rather than standard deviation for σ — boats are outliers and would inflate a plain σ.
   Sweep `N ∈ {3, 4, 5, 6}`. Because the threshold is scene-relative, it is inherently robust to
   scene-to-scene radiometric offsets, which is why atmospheric correction is not needed here.
4. **Connected components → blobs.** Filter on area (map to roughly 6–20 m) and on
   compactness/eccentricity — glint streaks and wake lines are elongated, hulls are compact.
5. **Georeference.** Centroid → lon/lat → range, reusing `boxes_to_records()` and
   `haversine_km()` (`vessel_detect.py:260-316`). No new geometry code.
6. **Emit into the same schema** as Detector A, with `detector` and `nir_snr` columns added.

**Run B permissively (N = 3).** Its false positives are cheap — a human clears them in §8 — but a
false *negative* breaks the cleanliness guarantee, which is the one thing B exists to provide.

---

## 8. Eyeball verification — every scene, not a sample

**Owner: B (or whoever is free) · Thursday +2:30–3:30 · ~2 min/scene**

`detection_model_tradeoff.md` budgeted ~1 h for a *sampled* validation set. With 1–2 boats per
scene, a **full census** costs about the same and is strictly better.

- **Verify on the NIR band, not RGB.** On NIR the water is nearly black and vessels glow. A human
  scanning that finds leftover dots in about a minute; scanning a 3300×3300 px colour image for
  2-pixel specks is slow and unreliable. This is the second reason NIR earns its place.
- For each scene: **confirm every A and B detection** (real / glint / wake / shoreline), then
  **sweep for anything both missed**. The sweep is the part that matters — it is what turns
  "the detectors agree" into "the scene is clean."
- **Record to `truth.csv`:** `scene_id, lon, lat, length_class_m, source (A/B/eye), verdict`.
- **Two people spot-check the same 3 scenes** at the start to calibrate what counts as a boat, then
  split the rest. Below ~6 m the answer is genuinely ambiguous and we need a shared convention.

**This is the gate on §6 step 2.** Without truth data the sweep has nothing to optimise against
and will optimise for glint. Verification is not the optional polish step — it is what makes the
rest of the detection work mean anything.

---

## 9. Scene classification — the dominance rule

**Owner: C · Thursday +4:00–5:00**

1. Weight each detection by its expected intensity share: `w_i = r_i^(−k/10)`. Screen with a
   nominal **k = 15**.
2. `dominance = max(w_i) / Σ(w_i)`.
3. Classify each scene:

| Class | Rule | Use |
|---|---|---|
| **EMPTY** | no verified detections | ambient noise floor |
| **SINGLE-DOMINANT** | dominance ≥ 0.8 | **the fit** |
| **CROWDED** | dominance < 0.8 | held back; used in degrade rung 3 |

4. **Name the circularity out loud and then resolve it.** We screen with `k` and we also fit `k`.
   Fix: after fitting, re-run the classification with the fitted `k` and confirm the scene set is
   stable. It is cheap, it is honest, and it is a legitimate sensitivity check to report rather
   than a flaw to hide.
5. **Out-of-AOI traffic.** Boats beyond the 5 km box are invisible to us. At k = 15 a boat at 6 km
   contributes ~7 % of what a boat at 1 km does, so *one* is negligible — but several are not.
   State that unseen traffic is absorbed into the ambient term, and note that the EMPTY scenes
   measure an ambient which **already includes** typical out-of-AOI traffic. That is a further
   reason they are worth keeping.

---

## 10. Interfaces — agree these before anyone writes code

The single-dominant reframe **moves the join from detection level to scene level**. That means a
third interface file, and the acoustics owner (A) must sign off on it today.

```
detections.csv : scene_id, acq_time_utc, lon, lat, bbox_w_m, bbox_h_m, length_class_m,
                 confidence, aspect_ratio, wake_flag, range_km,
                 detector, nir_snr                          ← NEW columns

scenes.csv     : scene_id, acq_time_utc, n_detections, dominance, class,          ← NEW FILE
                 r_dominant_km, wake_flag_dominant, verified                       ← the join key

levels.csv     : window_start_utc, window_end_utc, band_100_2k_db, band_2k_10k_db,
                 band_20_200_db, wind_ms                    ← unchanged
```

`detections.csv` keeps the existing schema (`vessel_detect.py:344`) so nothing already built
breaks; `scenes.csv` is what the acoustics side actually joins on.

**Ambient handling for Friday:** measure the noise floor from the EMPTY scenes, subtract it in the
**intensity** domain (not dB), then fit the line. This is cleaner and much easier to defend than
fitting three free parameters at once on ~20 points.

---

## 11. Thursday / Friday schedule

### Thursday (5 h)

| Time | Owner | Task |
|---|---|---|
| **+0:00–1:00** | C | **v1 `detections.csv` from Detector A on delivered scenes, handed straight to A.** Whatever the config, ship it. **Detection must never block the acoustic join.** |
| +0:00–1:00 | A | Batch-fetch the ~30 acoustic windows, compute band levels → `levels.csv` |
| +1:00–1:30 | C | Calibration check (§5) on the paired scenes |
| +1:30–2:30 | C | Build and run Detector B (§7) |
| +2:30–3:30 | B | Eyeball verification, all scenes (§8) → `truth.csv` |
| +3:30–4:00 | C | Re-rank the sweep on recall (§6.2); **freeze the config** |
| +4:00–5:00 | C+B | Scene classification (§9) → `scenes.csv`; recall-vs-length numbers |

**Last 20 min — together. Go/no-go (§12).**

### Friday (5 h)

- **0:00–2:30** — the `SL`/`k` fit on SINGLE-DOMINANT scenes, ambient subtracted from EMPTY scenes.
  Then the wake/no-wake comparison, which is now very clean: one boat per scene, does it have a
  wake or not? Only a moving boat radiates strongly.
- **2:30 — hard scope freeze.** Anything not working becomes "future work."
- **2:30–3:30** — figures: the `RL` vs `log10(r)` fit, and **recall vs. length class for A, B and
  A∪B** — the "where the pretrained model breaks" plot.
- **3:30–5:00** — notebook cleanup, README, rehearse. The "what we'd do next" slide is fine-tuning
  on 200–300 local annotations, now with a measured baseline to beat.

---

## 12. Go/no-go and the degrade ladder

> **Decision point: Thursday close. Fewer than ~10 SINGLE-DOMINANT scenes → degrade now, not
> Friday.** Friday has no slack for a pivot.

| Rung | Trigger | Action |
|---|---|---|
| **1** | 5–10 clean scenes | Relax dominance 0.8 → 0.6. Report both thresholds and show the fit is not an artefact of the cutoff. |
| **2** | Still too few | **Presence/absence.** Band level on EMPTY vs. any-boat scenes, t-test. Still a result, still a figure. |
| **3** | Water is simply busy | Revert to the **summed-source** model over all scenes — the original `project_ideas_ohw26.md` §2 plan. Nothing is wasted; the detections table is the same. |
| **4** | Detector A finds nothing real on PlanetScope | **Detector B becomes primary rather than supporting.** Option C is the only strategy where this costs nothing, because B was built anyway. |

Rung 4 is worth dwelling on: under Option A, "the pretrained model doesn't transfer" would have
been a dead end on Thursday afternoon. Under Option C it is a *finding*, and we still have a
working detector.

---

## 13. Limitations to state out loud

Several of these are results in their own right. Naming them is worth more than hiding them.

- **Recall is measured, not improved.** We did not retrain. We are reporting where a borrowed
  model breaks, which is a different and more honest contribution than claiming we fixed it.
- **Human truth at 3 m is itself uncertain below ~6 m.** The degradation curve has a floor set by
  the labeller, not by the detector. Report the length class where labeller confidence drops.
- **Detector B is independent, not correct.** It has its own biases — dark hulls, whitecaps,
  shoreline bleed. Two detectors agreeing is evidence, not proof.
- **Dominance screening depends on assumed `k`.** Resolved by the re-run in §9.4; report the
  stability of the scene set.
- **Unmodelled vessels outside the AOI**, absorbed into the ambient term.
- **Equal-source-level assumption.** A 200 hp planing boat and a 9.9 hp kicker are not the same
  source. `SL` is an effective fleet average.
- **Mid-morning sampling only.** Sun-synchronous overpass; no diel structure from imagery.
- **Length is a size class, not a measurement.** At 3 m GSD a 6 m boat is ~2 px.
- **Wind and flow noise** contaminate the small-craft band; handled by covariate or exclusion.
- **Small n.** Put the number of clean scenes in the figure caption, not in a footnote.
- **No AIS ground truth** exists for this fleet in Canada — which is the project's premise, not a
  gap in it.

---

## 14. References

- Model weights: https://huggingface.co/mayrajeo/marine-vessel-yolo (`yolo11s_tci.pt`, AGPL-3.0)
- Method code: https://github.com/mayrajeo/ship-detection
- Paper — mapping recreational marine traffic from Sentinel-2 with YOLO (2025, *RSE*):
  https://www.sciencedirect.com/science/article/pii/S0034425725001956
- PSScene asset & bundle spec: https://developers.planet.com/docs/data/psscene/
- UDM2 usable-data mask: https://developers.planet.com/docs/data/udm-2/
- Orders API `clip` tool: https://developers.planet.com/apis/orders/tools/#clip
- SAHI sliced inference: https://github.com/obss/sahi · Ultralytics: https://docs.ultralytics.com

**Companion documents:** `detection_model_tradeoff.md` (why Option C) ·
`project_ideas_ohw26.md` (the locked plan; §4, §5, §8 still apply) ·
`vessel_detect.py` (Detector A, working today)
