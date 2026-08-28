# 0018. Detector B's threshold is a pixel BUDGET, not a sigma multiple

Status: accepted
Date: 2026-08-28

Supersedes nothing. Amends the `NIR_MAD_N` block in `boatphone/optical.py` and plan section 8's
N-sweep, both of which asked a question with no answer.

## Context

Detector B thresholds NIR at `water_median + N * sigma_MAD`, with
`sigma_MAD = 1.4826 * MAD`. The `1.4826` is the **Gaussian** consistency constant: it converts a
MAD into a standard deviation only if the data are Gaussian. Ocean NIR is not, and Neve measured
how badly on `20200730` (10.83 M water px, median 0.01740, sigma_MAD 0.003855):

| N | threshold | px above | observed frac | Gaussian frac | ratio |
|---|---|---|---|---|---|
| 6 | 0.04053 | 14,128 | 1.30e-03 | 9.87e-10 | 1.3e6 x |
| 8 | 0.04824 | 9,530 | 8.80e-04 | 6.22e-16 | 1.4e12 x |
| 10 | 0.05595 | 7,450 | 6.88e-04 | 7.62e-24 | 9.0e19 x |

At the N=6 the batch ran, a Gaussian predicts **0.01 pixels in the whole scene**; there are
14,128. The tail barely decays -- 6 -> 10 sigma only halves the count -- so **N is very nearly a
dead knob**, and the section 8 N-sweep could not have converged, because there was no value to
find. `BLOB_MIN_PIXELS = 2` then turns that tail into "vessels": 46% of the batch's 32,877
detections are exactly 2 px, 72% are <= 4 px.

Two consequences are separable and only one is fixed here.

**Comparability (fixed here).** `sigma_MAD` measures the water's *core* texture while the
candidate count is set by its *tail*, and the two are not coupled. On a scene whose core is quiet,
sigma shrinks, the cut slides toward the median, and the tail floods through. Detections swung
**258x** (32 to 8,257) over the same 100 km2 with every scene >= 98% clear. Sensitivity was being
set by sea-surface texture, not by anything about boats -- and a count that varies with sea state
is exactly the wrong input to an optical-acoustic regression, because sea state also drives
ambient noise. That is a route to a clean correlation with no vessels in it, which is invariant 4's
failure mode.

**Resolution (NOT fixed here, and not fixable by any threshold).** A 6 m boat is 2 px at
PlanetScope's 3 m GSD and a 12 m boat is ~4 px, so the FAO 0-12 m class -- the class goal 2 cares
most about -- sits at or below the resolution limit. `min_px` cannot be raised to kill the noise
without deleting the target class, and no shape test is available at 2-4 px either. This is a
claim about the sensor, not about our code, and it needs its own record.

## Decision

**Add a distribution-free threshold and make it the recommended mode: spend a fixed budget of
`NIR_BUDGET_PX_PER_KM2` candidate pixels per km2 of water and read the cut off the empirical
distribution.** `optical.quantile_threshold()`; `detect_nir_blobs(threshold_mode="quantile")`;
`detect_vessels_nir.py --threshold-mode quantile`. Only the ordering of pixels by brightness is
used, so the shape of the tail is irrelevant by construction.

### The value: 40 px/km2

From `scripts/calibrate_nir_threshold.py` over all 26 delivered SR scenes, 8 budgets x 6 sigma
multiples, 364 rows -> `data/derived/nir_threshold_calibration.csv`. The run reproduces the
shipped batch exactly at `n_mad=6` (32,877 detections; 14,128 candidate px on `20200730`), so the
two columns below are like for like. The metric is the p90/p10 spread of detections per km2 of
water **across the 26 scenes** -- how differently one threshold behaves on the same ocean:

| sigma multiple | spread | median det/scene | | budget px/km2 | spread | median det/scene |
|---|---|---|---|---|---|---|
| N=3 | 7.5x | 11,339 | | 2.5 | 5.0x | 5 |
| N=4 | 26.7x | 3,128 | | 5 | 3.3x | 10 |
| N=5 | 19.4x | 1,290 | | 20 | 3.1x | 31 |
| **N=6 (as shipped)** | **14.9x** | **730** | | **40** | **2.4x** | **50** |
| N=8 | 11.1x | 378 | | 80 | 2.5x | 92 |
| N=10 | 9.2x | 240 | | 160 | 4.0x | 227 |

40 px/km2 sits at the minimum of the spread curve: **14.9x -> 2.4x**, a ~6x improvement in
between-scene comparability at a median 50 detections per scene (9 transient) against 730 (200
transient). For scale, Detector A -- an independent detector with different physics -- returned
~48 per scene on these same 26 scenes.

**The default stays `NIR_THRESHOLD_MODE = "mad"`.** The delivered batch, the shipped
`detections.csv`, and everything Isaac has already joined against were produced at `n_mad=6`;
flipping a library default would retroactively change what those files mean. The mode is switched
deliberately at the entry point. Flipping the default is Malachy's call once the section 8 census
has run against the quantile mode.

## What this does NOT claim

* **It is not a false-alarm rate.** Real vessels are inside the budget -- they are among the
  brightest water pixels, which is Detector B's entire premise -- so the budget is a false-alarm
  budget only to the extent that vessels are a small fraction of it. **Set it below the true
  vessel pixel count and it deletes vessels.**
* **It is not calibrated to truth, because there is no truth set.** 40 px/km2 lands between 15 and
  52 sigmas depending on the scene (p10..p90), far above the N=6 the batch ran; any real vessel
  dimmer than that cut is deleted silently. No threshold sweep can discover that. The section 8
  eyeball census is still the gate and section 6.2 still forbids ranking on counts.
* **It does not make counts correct, only comparable.** That is the property the acoustic matchup
  needs and the one the MAD cut did not have; it is not the same as being right.
* **Persistence was not usable as a calibration anchor.** The fraction of detections in multi-year
  ~50 m cells rises monotonically as the budget tightens (55% at 320 px/km2 -> 76% at 5), and the
  count of persistent cells falls monotonically (2,259 -> 34), with no knee in either. Both are
  density artefacts: at high detection density, cells co-occur across years by chance. Recorded as
  a negative result so nobody re-runs it.

## Consequences

* `robust_threshold()` is kept, with its limitation stated in its own docstring, as the reference
  the batch was measured with. Disagreement between the two modes on one scene is itself a
  diagnostic.
* `make_synthetic_scene(heavy_tail_df=...)` draws water noise from a Student-t **rescaled to the
  same sigma_MAD**: same robust scale, different tail. The self-test's section 4b now shows the
  real failure on a fixture -- 0 candidate px on Gaussian water vs 1,485 on heavy-tailed water at
  the same N=6 and the same sigma -- and shows the budget admitting exactly its 161 px on both.
  The general lesson, which is bigger than this decision: **a synthetic fixture validates the
  implementation, never the distributional assumption underneath it.** The old sweep passed for
  exactly as long as the fixture agreed with the assumption.
* `PERSISTENCE_CELL_DLAT/DLON` moved from `detect_vessels_nir.py` into `optical.py`; two entry
  points now grid on them and two definitions would silently disagree about "the same place"
  (invariant 6).
* `export_detections.load_scene()` also returns `green` as float reflectance, so the calibration
  script reuses it rather than adding a fourth raster reader.
* Plan section 8's N-sweep should be re-framed as a budget sweep. The census workload it was
  sizing drops from ~200 transient detections per scene to ~9.
