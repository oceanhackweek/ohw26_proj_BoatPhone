#!/usr/bin/env python3
"""Prove the Option C optical pipeline on a scene whose answers are known.

WHY THIS EXISTS. Invariant 3 requires the acoustic pipeline to be proven against
a synthetic tone of known level, frequency and time before anything depends on
it. This is the optical counterpart: vessels of known length at known lon/lat, so
that every stage between "pixels" and "scenes.csv" is checked against an answer
that does not depend on any imagery being delivered, any model downloading, or
rasterio being installed.

It also gives invariant 9 something to stand on. On real imagery, "Detector B
returned nothing" and "Detector B is broken" look identical. Here they do not:
if the planted vessels are not found, the code is wrong, full stop.

There is no pytest on the hub (CLAUDE.md), so this is a plain script: it prints a
line per check and exits non-zero if any check fails.

    python scripts/optical_selftest.py

Needs numpy, cv2 and pyproj only. It does NOT need rasterio, ultralytics, a
network connection, or a single delivered scene.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from boatphone import optical as opt  # noqa: E402


RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def nearest_px(blobs, row, col):
    """(distance_px, blob) for the blob closest to a truth pixel position."""
    if not blobs:
        return float("inf"), None
    d = [np.hypot(b.row - row, b.col - col) for b in blobs]
    i = int(np.argmin(d))
    return d[i], blobs[i]


# ---------------------------------------------------------------------------
# The fixture. Four vessels at known offsets from the hydrophone, spanning the
# size range the project cares about, plus one 40 m ship that the small-craft cut
# must reject and one wake-like streak that the shape filter must reject.
# ---------------------------------------------------------------------------
RES_M = 3.0
SCENE_ID, ACQ = "SYNTH_20250807", "2025-08-07T15:07:31Z"
VESSELS = [                       # (east_m, north_m, length_m)
    (300.0, 200.0, 9.0),          # small recreational craft, the project's subject
    (-450.0, 600.0, 15.0),
    (700.0, -400.0, 6.0),         # at the labeller-confidence floor
    (-200.0, -750.0, 12.0),
]

print(__doc__.split("\n")[0])
print(f"\nfixture: {len(VESSELS)} vessels, {RES_M:g} m GSD, "
      f"hydrophone {opt.HYDROPHONE_LONLAT} ({opt.HYDROPHONE_SOURCE.split(';')[0]})\n")

scene = opt.make_synthetic_scene(n_px=700, res_m=RES_M, vessels=VESSELS)

# ---------------------------------------------------------------------------
print("1. GEOMETRY -- pixel -> lon/lat -> range")
# ---------------------------------------------------------------------------
# The single highest-consequence piece of code here. Every point in the SL/k fit
# is (received level, range); a systematic range error bends the fitted spreading
# coefficient, and nothing about the resulting plot would look wrong.
errs_m = []
for t in scene.truth:
    r = opt.range_km(t["lon"], t["lat"])
    errs_m.append(abs(r - t["range_km"]) * 1000.0)
check("range from lon/lat matches the planted range",
      max(errs_m) < 1.0, f"max error {max(errs_m):.3f} m over {len(errs_m)} vessels")

# A vessel due east must bear 090; due north 000. Catches a lon/lat transposition,
# which a symmetric fixture would otherwise hide.
east_lon, east_lat = scene.to_lonlat(*[[v] for v in [(1000.0 / RES_M) + 349.5, 349.5]])
brg = opt.bearing_deg(float(east_lon[0]), float(east_lat[0]))
check("a point 1 km due east bears 090 deg", abs(brg - 90.0) < 0.5, f"bearing {brg:.2f} deg")

# ---------------------------------------------------------------------------
print("\n2. WATER MASK")
# ---------------------------------------------------------------------------
water, report = opt.water_mask(scene.green, scene.nir, scene.valid, res_m=RES_M)
print(f"     {report}")
check("land strip is excluded", not water[:60].any(),
      f"{int(water[:60].sum())} land px in mask")
check("the land strip is found as ONE large component", report.land_components == 1,
      f"{report.land_components} components >= {opt.MIN_LAND_AREA_M2/1e4:g} ha")
check("the shore buffer pulled the mask back", report.buffered_px > 0,
      f"{report.buffered_px:,} px")

# The vessels are bright in NIR, so they are bright COMPONENTS -- they must be
# counted as too-small-to-be-land, never as land.
check("vessels are counted as bright-but-too-small, not as land",
      report.small_bright_components == len(scene.truth),
      f"{report.small_bright_components} small bright components for "
      f"{len(scene.truth)} vessels")
check("every vessel centre is inside the water mask",
      all(water[int(round(t["row"])), int(round(t["col"]))] for t in scene.truth))

# THE REGRESSION. Measured on the first delivered Planet scene: a per-pixel
# brightness cut classified the two BRIGHTEST objects in the scene as land and
# removed them from the mask, so Detector B could not see them at any threshold.
# Disable the extent rule and confirm that failure really does occur.
water_bright, rep_bright = opt.water_mask(scene.green, scene.nir, scene.valid,
                                          res_m=RES_M, min_land_area_m2=0.0)
in_mask = sum(1 for t in scene.truth
              if water_bright[int(round(t["row"])), int(round(t["col"]))])
check("a per-pixel brightness cut DELETES the vessels (the bug it fixes)",
      in_mask == 0, f"{in_mask}/{len(scene.truth)} vessel centres survive it")

# A brighter vessel must not be more likely to be called land. This is the
# property the extent rule guarantees and brightness thresholding cannot.
bright_scene = opt.make_synthetic_scene(n_px=700, res_m=RES_M, vessels=VESSELS, seed=3)
for t in bright_scene.truth:                      # 3x brighter hulls
    r0, c0 = int(round(t["row"])), int(round(t["col"]))
    bright_scene.nir[r0-1:r0+2, c0-3:c0+4] = 0.60
w2, rep2 = opt.water_mask(bright_scene.green, bright_scene.nir, bright_scene.valid,
                          res_m=RES_M)
check("tripling vessel brightness does not turn them into land",
      all(w2[int(round(t["row"])), int(round(t["col"]))] for t in bright_scene.truth),
      f"land still {rep2.land_components} component(s)")

# NDWI is reported, not used. On an all-water scene it has no land mode and would
# call ~half the water land; the number exists so that is visible, not hidden.
print(f"     NDWI would call {100*report.ndwi_land_fraction:.0f}% of this scene land; "
      f"the extent rule found {100*report.land_px/report.valid_px:.0f}%")

# ---------------------------------------------------------------------------
print("\n3. DETECTOR B -- recall on known vessels")
# ---------------------------------------------------------------------------
blobs, diag = opt.detect_nir_blobs(scene.nir, water, RES_M)
print(f"     threshold {diag['threshold_reflectance']:.4f} "
      f"(water median {diag['water_median']:.4f}, sigma {diag['sigma_mad']:.5f}, "
      f"N={diag['n_mad']:g}) | raw {diag['raw']} -> kept {diag['kept']}")

def recall(blobs, truth, tol_px=1.5):
    """(n_found, n_truth, worst_error_px) -- how many planted vessels were hit."""
    d = [nearest_px(blobs, t["row"], t["col"])[0] for t in truth]
    return sum(1 for x in d if x <= tol_px), len(truth), (max(d) if d else 0.0)


found, n, worst = recall(blobs, scene.truth)
check("every planted vessel is found", found == n,
      f"{found}/{n}, worst centroid error {worst:.2f} px")

len_err = []
for t in scene.truth:
    _, b = nearest_px(blobs, t["row"], t["col"])
    len_err.append(abs(b.length_m - t["length_m"]))
# Exact, not "within a pixel". A tolerance of one pixel here would have hidden
# the systematic minAreaRect centre-to-centre undercount (see detect_nir_blobs):
# every length was short by exactly 1 px, which is 3 m at PlanetScope GSD and
# enough to push vessels across the 12 m FAO edge.
check("measured length matches truth exactly, with no 1 px bias",
      max(len_err) < 0.5, f"max error {max(len_err):.2f} m")
check("length bias is not merely small but zero-mean",
      abs(float(np.mean([b.length_m - t["length_m"]
                         for t in scene.truth
                         for b in [nearest_px(blobs, t["row"], t["col"])[1]]]))) < 0.5,
      "a one-sided error is a bug; a two-sided one is quantisation")

# Everything B returned that is not a planted vessel. At N = 3 this is NOT zero
# and is not meant to be: the plan runs B permissively because a false negative
# breaks the cleanliness guarantee while a false positive only costs a human a
# glance. The number matters anyway -- see the null check below.
print(f"     {len(blobs) - found} of {len(blobs)} kept blobs are not planted vessels")

# ---------------------------------------------------------------------------
print("\n4. THE NULL CHECK (invariant 4) -- and what N costs")
# ---------------------------------------------------------------------------
# Same water, same noise statistics, no vessels at all. Whatever B reports here
# is noise, and it sets the human-verification workload for every real scene.
empty = opt.make_synthetic_scene(n_px=700, res_m=RES_M, vessels=(), seed=99)
water_e, _ = opt.water_mask(empty.green, empty.nir, empty.valid, res_m=RES_M)

# The real AOI is 10 x 10 km at 3 m GSD. Scaling the measured false-positive
# density to it turns "a few dots" into the number of things a person has to
# look at per scene, which is what the plan's 2 min/scene budget rests on.
AOI_WATER_PX = (10_000 / RES_M) ** 2

print(f"     {'N':>4} {'recall':>8} {'FP/scene':>9} {'FP per real AOI':>17}")
sweep = {}
for n_mad in opt.SWEEP_MAD_N:
    b_v, _ = opt.detect_nir_blobs(scene.nir, water, RES_M, n_mad=n_mad)
    b_e, d_e = opt.detect_nir_blobs(empty.nir, water_e, RES_M, n_mad=n_mad)
    r_found, r_n, _ = recall(b_v, scene.truth)
    scaled = len(b_e) * AOI_WATER_PX / max(d_e["water_px"], 1)
    sweep[n_mad] = (r_found, r_n, len(b_e), scaled)
    print(f"     {n_mad:>4g} {r_found:>5}/{r_n:<2} {len(b_e):>9} {scaled:>17,.0f}")

check("recall is 100% at the permissive default N=3",
      sweep[opt.NIR_MAD_N][0] == sweep[opt.NIR_MAD_N][1])
check("a stricter N still finds every vessel", sweep[6.0][0] == sweep[6.0][1],
      "so N can be raised to buy precision without losing these vessels")
check("a stricter N removes the noise detections", sweep[6.0][2] < sweep[3.0][2],
      f"{sweep[3.0][2]} FP at N=3 -> {sweep[6.0][2]} at N=6")
check("the null scene's false positives are all sub-vessel specks",
      all(b.n_px <= 4 for b in opt.detect_nir_blobs(empty.nir, water_e, RES_M)[0]),
      "they are 2-4 px noise pairs, not hull-sized")

# ---------------------------------------------------------------------------
print("\n4b. THE THRESHOLD THAT DOES NOT ASSUME A DISTRIBUTION")
# ---------------------------------------------------------------------------
# Section 4 sweeps N over GAUSSIAN water and concludes N=6 is clean. On real
# scenes the same N=6 returned ~1,265 detections per scene. Both are true: the
# sweep proved the code, and the code's premise is false over water. Ocean NIR
# has a heavy tail, and sigma_MAD -- which measures the CORE -- cannot see it.
#
# So the fixture has to be able to express the failure before the fix can be
# tested. `heavy_tail_df` draws the water noise from a Student-t rescaled to the
# SAME sigma_MAD: identical robust scale, different tail. See decision 0018.
heavy = opt.make_synthetic_scene(n_px=700, res_m=RES_M, vessels=VESSELS,
                                 heavy_tail_df=3.0, seed=99)
water_h, _ = opt.water_mask(heavy.green, heavy.nir, heavy.valid, res_m=RES_M)

_, med_g, sig_g = opt.robust_threshold(empty.nir[water_e])
_, med_h, sig_h = opt.robust_threshold(heavy.nir[water_h])
check("the heavy-tailed fixture has the SAME robust scale as the Gaussian one",
      abs(sig_h - sig_g) / sig_g < 0.05,
      f"sigma_MAD {sig_g:.5f} vs {sig_h:.5f} -- so any difference below is the "
      f"TAIL, not the scale")

mad_g, dg = opt.detect_nir_blobs(empty.nir, water_e, RES_M, n_mad=6.0)
mad_h, dh = opt.detect_nir_blobs(heavy.nir, water_h, RES_M, n_mad=6.0)
print(f"     N=6 candidate px: gaussian {dg['candidate_px']:,} | "
      f"heavy-tailed {dh['candidate_px']:,}")
check("a MAD threshold that is clean on Gaussian water floods on a heavy tail",
      dh["candidate_px"] > 100 * max(dg["candidate_px"], 1),
      f"{dg['candidate_px']} -> {dh['candidate_px']:,} candidate px at the SAME "
      f"N and the same sigma: this is the real-scene failure, reproduced")

# N is a DEAD KNOB on the heavy tail: the thing the batch measured (6 -> 10 sigma
# only halves the count, where a Gaussian drops it fourteen orders of magnitude).
_, dh10 = opt.detect_nir_blobs(heavy.nir, water_h, RES_M, n_mad=10.0)
check("raising N does not rescue it -- the knob is nearly disconnected",
      dh10["candidate_px"] > dh["candidate_px"] / 10,
      f"N 6 -> 10 leaves {dh10['candidate_px']:,} of {dh['candidate_px']:,} "
      f"({100 * dh10['candidate_px'] / dh['candidate_px']:.0f}%), not ~0%")

# The fix: spend a fixed budget and read the cut off the empirical distribution.
BUDGET = opt.NIR_BUDGET_PX_PER_KM2
q_g, qdg = opt.detect_nir_blobs(empty.nir, water_e, RES_M,
                                threshold_mode="quantile", budget_px_per_km2=BUDGET)
q_h, qdh = opt.detect_nir_blobs(heavy.nir, water_h, RES_M,
                                threshold_mode="quantile", budget_px_per_km2=BUDGET)
print(f"     budget {BUDGET:g} px/km2 -> candidate px: gaussian "
      f"{qdg['candidate_px']:,} | heavy-tailed {qdh['candidate_px']:,} "
      f"(N_eff {qdg['n_mad_effective']:.1f} vs {qdh['n_mad_effective']:.1f})")
check("the budget is honoured on BOTH distributions, to the pixel",
      qdg["candidate_px"] == qdg["budget_px"] == qdh["candidate_px"] == qdh["budget_px"],
      f"{qdg['budget_px']} px asked, {qdg['candidate_px']}/{qdh['candidate_px']} admitted")
check("the SAME budget lands at a very different N -- which is the point",
      abs(qdh["n_mad_effective"] - qdg["n_mad_effective"]) > 1.0,
      f"N_eff {qdg['n_mad_effective']:.1f} vs {qdh['n_mad_effective']:.1f}: one "
      f"fixed N cannot serve both, one fixed budget can")

# A budget that bounds workload and deletes the vessels would be worse than what
# it replaces. This is the check that keeps it honest.
found_q, n_q, worst_q = recall(q_h, heavy.truth)
check("the budget still finds every planted vessel on heavy-tailed water",
      found_q == n_q, f"{found_q}/{n_q}, worst centroid error {worst_q:.2f} px")
# What the reduction actually costs and buys, counted as noise blobs rather than
# as a ratio of totals: the vessels are in both numbers and must not be credited
# to either threshold.
found_mad, _, _ = recall(mad_h, heavy.truth)
noise_mad, noise_q = len(mad_h) - found_mad, len(q_h) - found_q
check("the budget removes the tail's noise blobs, not the vessels",
      noise_q <= noise_mad / 4,
      f"non-vessel blobs {noise_mad} at N=6 -> {noise_q} at a {BUDGET:g} px/km2 "
      f"budget, with {found_q}/{n_q} vessels kept either way")

# The budget is a RATE, so it must scale with the water it is spent over.
half = water_h.copy()
half[:, half.shape[1] // 2:] = False
_, qdhalf = opt.detect_nir_blobs(heavy.nir, half, RES_M,
                                 threshold_mode="quantile", budget_px_per_km2=BUDGET)
check("the budget is a rate: halve the water, halve the pixels spent",
      abs(qdhalf["budget_px"] / qdh["budget_px"] - 0.5) < 0.05,
      f"{qdh['budget_px']} px over {qdh['water_km2']:.2f} km2 -> "
      f"{qdhalf['budget_px']} px over {qdhalf['water_km2']:.2f} km2")

# Quantised reflectance makes the budget-th largest value repeat. The overrun is
# REPORTED rather than assumed away (invariant 5).
tied = np.repeat(np.array([0.01, 0.02, 0.03], dtype=np.float64), 400)
thr_t, _, _, sel_t, _ = opt.quantile_threshold(tied, 100)
check("ties are reported as an overrun, not silently swallowed",
      sel_t == 400 and thr_t == 0.03,
      f"asked 100 px, the 100th largest value repeats 400 times -> reported {sel_t}")

for bad, why, needle in [
        (dict(threshold_mode="quantile", budget_px_per_km2=1e9),
         "a budget larger than the water mask", "entire"),
        (dict(threshold_mode="quantile", budget_px_per_km2=0.0),
         "a non-positive budget", "must be positive"),
        (dict(threshold_mode="empirical"), "an unknown threshold mode", "must be")]:
    try:
        opt.detect_nir_blobs(heavy.nir, water_h, RES_M, **bad)
        check(f"{why} raises instead of guessing", False)
    except ValueError as exc:
        check(f"{why} raises instead of guessing", needle in str(exc), str(exc)[:70])

# ---------------------------------------------------------------------------
print("\n5. SHAPE AND SIZE FILTERS")
# ---------------------------------------------------------------------------
# Two features that are NOT small craft, for opposite reasons:
#   - a 39 m vessel: a real ship, out of scope for the fit, but it MUST be
#     detected because it is what makes a scene not-clean;
#   - an 18 x 3 m streak: inside every length bound and still not a hull, which
#     is the case only the aspect filter can catch.
scene2 = opt.make_synthetic_scene(n_px=700, res_m=RES_M, vessels=VESSELS, seed=7)
scene2.nir[500:504, 100:113] = 0.22        # 39 m x 12 m: a real ship
scene2.nir[560:561, 200:206] = 0.22        # 18 m x 3 m: glint streak / wake line
water2, _ = opt.water_mask(scene2.green, scene2.nir, scene2.valid, res_m=RES_M)
kept2, diag2 = opt.detect_nir_blobs(scene2.nir, water2, RES_M)

ships = [b for b in kept2 if b.length_m >= 24.0]
check("the 39 m vessel is DETECTED, not silently dropped", len(ships) == 1,
      f"{len(ships)} vessels >= 24 m kept")
check("it is classified >24m rather than cut by a 20 m threshold",
      bool(ships) and opt.size_class(ships[0].length_m) == ">24m",
      f"length {ships[0].length_m:.0f} m -> {opt.size_class(ships[0].length_m)}"
      if ships else "not detected")
check("the thin streak is cut by the shape filter, not by length",
      diag2["too_elongated"] >= 1,
      f"too_elongated={diag2['too_elongated']} implausible_length="
      f"{diag2['implausible_length']}")
f2, n2, _ = recall(kept2, scene2.truth)
check("the small craft survive alongside it", f2 == n2, f"{f2}/{n2} recalled")

# THE REGRESSION THIS SECTION EXISTS FOR. A 20 m cut applied at detection time
# deletes the ship, and the scene then passes as SINGLE-DOMINANT on a small craft
# the ship is acoustically swamping. Same pixels, two thresholds, opposite verdicts.
recs2 = opt.blobs_to_records(kept2, "SYNTH_SHIP", ACQ, scene2.to_lonlat)
in_scope, tally2 = opt.select_size_classes(recs2)
print(f"     size classes present: {tally2}")
row_all_sizes = opt.classify_scene(recs2, "SYNTH_SHIP", ACQ)
row_scoped = opt.classify_scene(in_scope, "SYNTH_SHIP", ACQ)
check("scoping BEFORE classification changes the verdict (the bug)",
      row_all_sizes["dominance"] != row_scoped["dominance"],
      f"all sizes -> {row_all_sizes['dominance']}, "
      f"small craft only -> {row_scoped['dominance']}")
check("the >24m vessel is counted for cleanliness",
      row_all_sizes["n_detections"] > row_scoped["n_detections"],
      f"{row_all_sizes['n_detections']} vs {row_scoped['n_detections']} detections")
check("scoping keeps only the target FAO classes",
      {opt.size_class_of(r) for r in in_scope} <= set(opt.TARGET_SIZE_CLASSES))

# ---------------------------------------------------------------------------
print("\n6. RECORDS AND FUSION")
# ---------------------------------------------------------------------------
recs_b = opt.blobs_to_records(blobs, SCENE_ID, ACQ, scene.to_lonlat)
check("Detector B rows match the detections schema",
      all(set(r) == set(opt.DETECTIONS_FIELDS) for r in recs_b),
      f"{len(recs_b)} rows x {len(opt.DETECTIONS_FIELDS)} columns")

# Stand-in for Detector A: it sees the 9 m, 15 m and 12 m vessels (boxes offset
# by a pixel, as a YOLO box drawn round hull-plus-wake would be) and misses the
# 6 m one -- which is the transfer gap the whole methods result is about. The
# real Detector A is not run here: it needs weights, a download and ~minutes of
# CPU, and what is under test is the fusion arithmetic, not the model.
A_SEES = (0, 1, 3)          # indices into VESSELS; index 2 is the 6 m craft
boxes = []
for t in [scene.truth[i] for i in A_SEES]:
    half = t["length_m"] / RES_M / 2.0
    boxes.append((((t["col"] - half + 1), (t["row"] - half + 1),
                   (t["col"] + half + 1), (t["row"] + half + 1)), 0.42))
recs_a, counts_a = opt.boxes_to_records(boxes, SCENE_ID, ACQ, RES_M, scene.to_lonlat,
                                        valid=scene.valid, water=water)
check("Detector A rows survive the water mask", len(recs_a) == 3, str(counts_a))

fused, tally = opt.fuse(recs_a, recs_b)
print(f"     {tally}")
check("A and B agree on the vessels both saw", tally["AB"] == 3, f"AB={tally['AB']}")


def rows_near(rows, t, tol_m=opt.FUSE_TOL_M):
    """Fused rows within tol_m of a planted vessel."""
    return [r for r in rows
            if opt.range_km(r["lon"], r["lat"], (t["lon"], t["lat"])) * 1000 <= tol_m]


per_truth = [len(rows_near(fused, t)) for t in scene.truth]
check("every planted vessel appears exactly once after fusion",
      per_truth == [1] * len(scene.truth), f"rows per vessel: {per_truth}")

# The 6 m craft is the one Detector A misses. B-only detections stratified by
# length ARE the methods result -- "the pretrained S2 model recovers X% of
# NIR-confirmed small craft, and degrades below Y m".
missed = rows_near(fused, scene.truth[2])
check("the 6 m craft A missed is carried as B-only",
      len(missed) == 1 and missed[0]["detector"] == "B",
      f"detector={missed[0]['detector'] if missed else 'MISSING'}")
check("agreed rows carry B's NIR SNR",
      all(r["nir_snr"] != "" for r in fused if r["detector"] == "AB"))
check("A-only and B-only rows are distinguishable in the output",
      {r["detector"] for r in fused} <= {"A", "B", "AB"},
      f"detectors present: {sorted({r['detector'] for r in fused})}")

# ---------------------------------------------------------------------------
print("\n6b. BOUNDING-BOX CORNER EXPORT")
# ---------------------------------------------------------------------------
# A box of KNOWN pixel extent must round-trip to corners whose real separation is
# the known width and height. Getting corners right but the centroid wrong (or
# vice versa) would put the CSV and the GeoJSON on different objects.
from pyproj import Geod as _Geod
_geod = _Geod(ellps="WGS84")
W_PX, H_PX = 20.0, 8.0                      # 60 m x 24 m at 3 m GSD
cx0, cy0 = 350.0, 350.0
test_box = (cx0 - W_PX / 2, cy0 - H_PX / 2, cx0 + W_PX / 2, cy0 + H_PX / 2)

corner_rows, _ = opt.boxes_to_corner_records(
    [(test_box, 0.42)], SCENE_ID, ACQ, RES_M, scene.to_lonlat,
    shape=scene.valid.shape, config={"slice_px": 320})
cr = corner_rows[0]
top_m = _geod.inv(cr["ul_lon"], cr["ul_lat"], cr["ur_lon"], cr["ur_lat"])[2]
left_m = _geod.inv(cr["ul_lon"], cr["ul_lat"], cr["ll_lon"], cr["ll_lat"])[2]
check("corner separation equals the known box width",
      abs(top_m - W_PX * RES_M) < 1.0, f"{top_m:.2f} m vs {W_PX*RES_M:.0f} m")
check("corner separation equals the known box height",
      abs(left_m - H_PX * RES_M) < 1.0, f"{left_m:.2f} m vs {H_PX*RES_M:.0f} m")

# The two export paths must agree on where the detection IS.
pt_rows, _ = opt.boxes_to_records([(test_box, 0.42)], SCENE_ID, ACQ, RES_M,
                                  scene.to_lonlat)
check("corner-export centroid matches boxes_to_records() exactly",
      (cr["centroid_lon"], cr["centroid_lat"]) == (pt_rows[0]["lon"], pt_rows[0]["lat"]),
      f"{cr['centroid_lon']},{cr['centroid_lat']} vs "
      f"{pt_rows[0]['lon']},{pt_rows[0]['lat']}")
check("corners are ordered UL/UR/LR/LL (north-up: NW/NE/SE/SW)",
      cr["ul_lat"] > cr["ll_lat"] and cr["ur_lon"] > cr["ul_lon"])
check("provenance columns are carried", cr["slice_px"] == 320)
check("range and bearing are both populated",
      cr["range_km"] != "" and cr["bearing_deg_from_hydrophone"] != "")

# THE INVARIANT THE NIR FLAG EXISTS UNDER: it annotates, it never filters. Boxes
# placed on bare water have no NIR support and must still every one produce a row.
blank = [((c, 300.0, c + 6.0, 306.0), 0.3) for c in (100.0, 200.0, 300.0, 400.0)]
with_nir, _ = opt.boxes_to_corner_records(blank, SCENE_ID, ACQ, RES_M,
                                          scene.to_lonlat, nir=scene.nir,
                                          valid=scene.valid)
without, _ = opt.boxes_to_corner_records(blank, SCENE_ID, ACQ, RES_M, scene.to_lonlat)
check("the NIR flag annotates and never filters",
      len(with_nir) == len(without) == len(blank),
      f"{len(with_nir)} with NIR, {len(without)} without, {len(blank)} boxes in")
check("empty water is flagged, not dropped",
      all(r["possible_false_positive"] == 1 for r in with_nir),
      f"{sum(r['possible_false_positive'] for r in with_nir)}/{len(with_nir)} flagged")

# A box ON a planted vessel must come back supported -- otherwise the flag would
# be condemning real detections.
t = scene.truth[1]
on_boat = ((t["col"] - 4, t["row"] - 4, t["col"] + 4, t["row"] + 4), 0.5)
sup, _ = opt.boxes_to_corner_records([on_boat], SCENE_ID, ACQ, RES_M,
                                     scene.to_lonlat, nir=scene.nir, valid=scene.valid)
check("a box on a real vessel is NIR-supported, not flagged",
      sup[0]["possible_false_positive"] == 0,
      f"support={sup[0]['nir_support']}, {sup[0]['nir_bright_px']} bright px")
check("corner rows match DETECTION_CORNER_FIELDS",
      set(cr) == set(opt.DETECTION_CORNER_FIELDS) - {"nir_peak","nir_x_water",
          "nir_bright_px","nir_fill","nir_support","possible_false_positive",
          "nir_threshold_rho"} | set())

# ---------------------------------------------------------------------------
print("\n7. DOMINANCE AND SCENE CLASSIFICATION")
# ---------------------------------------------------------------------------
k = opt.NOMINAL_SPREADING_K
w = np.array([1.0, 4.0]) ** (-k / 10.0)
check("dominance matches the hand calculation",
      abs(opt.dominance([1.0, 4.0], k) - w.max() / w.sum()) < 1e-12,
      f"one boat at 1 km vs one at 4 km -> {opt.dominance([1.0, 4.0], k):.3f}")
check("an empty scene has no dominant source (NaN, not 1.0)",
      np.isnan(opt.dominance([])))

row_empty = opt.classify_scene([], "S_EMPTY", ACQ)
check("no detections -> EMPTY (the ambient-noise rung)",
      row_empty["class"] == opt.CLASS_EMPTY)

row_single = opt.classify_scene(fused[:1], SCENE_ID, ACQ)
check("one detection -> SINGLE-DOMINANT", row_single["class"] == opt.CLASS_SINGLE,
      f"dominance {row_single['dominance']}")

row_all = opt.classify_scene(fused, SCENE_ID, ACQ)
check("four spread vessels -> CROWDED", row_all["class"] == opt.CLASS_CROWDED,
      f"dominance {row_all['dominance']}")
check("the dominant vessel is the nearest one",
      abs(row_all["r_dominant_km"] - min(r["range_km"] for r in fused)) < 1e-9)
check("scenes.csv row matches its schema", set(row_all) == set(opt.SCENES_FIELDS))

# The circularity the plan names: screening with k while also fitting k. Confirm
# the classification is at least reported as k-dependent rather than assumed stable.
doms = {kk: opt.classify_scene(fused, SCENE_ID, ACQ, k=kk)["dominance"]
        for kk in (10.0, 15.0, 20.0)}
print(f"     dominance vs assumed k: {doms}   <- re-run with the FITTED k (plan 9.4)")

# ---------------------------------------------------------------------------
print("\n8. SIZE CLASSES AND THE ACOUSTICS I3 VIEW")
# ---------------------------------------------------------------------------
check("FAO class edges are exact at 12 m and 24 m",
      [opt.size_class(x) for x in (0.0, 11.99, 12.0, 23.99, 24.0, 40.0)]
      == ["0-12m", "0-12m", "12-24m", "12-24m", ">24m", ">24m"])

i3 = opt.to_i3_records(fused, res_m=RES_M)
check("the I3 view matches the acoustics schema",
      all(set(r) == set(opt.I3_FIELDS) for r in i3), f"{len(i3)} rows")
check("I3 detection ids are unique", len({r["detection_id"] for r in i3}) == len(i3))
check("I3 length_px and length_m_est are consistent",
      all(abs(r["length_px"] * RES_M - r["length_m_est"]) < 0.05 for r in i3))

# ---------------------------------------------------------------------------
print("\n9. OUTPUT SCHEMAS")
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "detections.csv"
    n = opt.write_csv(p, fused, opt.DETECTIONS_FIELDS)
    header = p.read_text().splitlines()[0]
    check("detections.csv header is exactly the agreed schema",
          header == ",".join(opt.DETECTIONS_FIELDS), f"{n} rows")

    bad = dict(fused[0])
    bad.pop("range_km")
    try:
        opt.write_csv(Path(tmp) / "bad.csv", [bad], opt.DETECTIONS_FIELDS)
        check("a missing column raises rather than writing a short row", False)
    except ValueError as exc:
        check("a missing column raises rather than writing a short row",
              "range_km" in str(exc))

# ---------------------------------------------------------------------------
print("\n10. THINGS THAT MUST ERROR, NOT GUESS (invariant 5)")
# ---------------------------------------------------------------------------
idx = opt.ndwi(np.array([[0.0, 0.1]], np.float32), np.array([[0.0, 0.1]], np.float32))
check("NDWI is NaN where both bands are zero, not 0.0", np.isnan(idx[0, 0]),
      "a 0 would sit exactly on the water threshold")

try:
    opt.robust_threshold(np.full(500, 0.02))
    check("constant NIR raises instead of returning a threshold", False)
except ValueError as exc:
    check("constant NIR raises instead of returning a threshold",
          "constant" in str(exc))

try:
    opt.dominance([0.0, 2.0])
    check("a detection at range 0 raises", False)
except ValueError as exc:
    check("a detection at range 0 raises", "hydrophone" in str(exc))

try:
    opt.water_mask(scene.green, scene.nir, scene.valid, res_m=RES_M,
                   clear=np.ones((10, 10), bool))
    check("a mismatched UDM2 mask raises instead of being ignored", False)
except ValueError as exc:
    check("a mismatched UDM2 mask raises instead of being ignored",
          "does not match" in str(exc))

# ---------------------------------------------------------------------------
n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
print(f"\n{'-' * 72}\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
if n_fail:
    print("FAILED:")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  - {name}  {detail}")
sys.exit(1 if n_fail else 0)
