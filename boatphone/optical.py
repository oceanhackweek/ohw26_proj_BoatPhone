"""Optical vessel detection: shared constants and the Option C detector-B stack.

This is the optical half of CLAUDE.md invariant 6. Anything the optical workstream
shares with acoustics or acquisition -- the hydrophone position, the AOI geometry,
the FAO size classes, the emitted schemas -- is defined HERE, once, and imported.
It is not restated in a notebook.

Scope, deliberately: this module holds **no file I/O**. Reading a `.SAFE` tree or a
Planet GeoTIFF needs `rasterio`, which is NOT in the hub environment by default
(see MODULE NOTE below); every function here takes plain numpy arrays plus a
`to_lonlat` callable, so the whole detector stack is exercisable on a synthetic
scene with no imagery and no geospatial stack at all. `scripts/optical_selftest.py`
does exactly that.

The split, per CLAUDE.md invariant 6:
    boatphone/optical.py   constants + functions      (this file, importable)
    the notebook           loaders + plots + narrative (the entry point)

Option C in one sentence (docs/plans/option_c_detection_plan.md): keep the
pretrained Sentinel-2 YOLO frozen, and run a second, independent NIR
bright-object detector whose failure modes do not overlap with it, so that a
scene can be certified CLEAN -- one dominant vessel and nothing else -- which is
the assumption the single-dominant-source acoustic fit rests on.

MODULE NOTE -- `rasterio` is not in `/home/.pixi/envs/default` (confirmed
2026-08-27; CLAUDE.md's "not available" list is right about this one). It was
installed for this workstream with `pip install --user rasterio` into
`/home/jovyan/.local`, which persists across container restarts but is NOT shared
with teammates and is NOT in `environment.yml`. Neither is `ultralytics`,
`huggingface_hub` or `opencv`. If the optical path is to be reproducible off-hub,
those four belong in the manifest.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

# `boatphone` promises a cheap, stdlib-only package import (see __init__.py), so
# nothing here may run at import time beyond numpy. cv2 and pyproj are imported
# inside the functions that need them.


# ==========================================================================
# 1. SITE GEOMETRY
# ==========================================================================
# WHERE THIS LIVES AND WHY. As of 2026-08-27 the hydrophone position exists in
# the repo in exactly one place: `HYD_LON, HYD_LAT` in cell 6 of
# contributor_folders/malachymcc/planet_folger_search_order_download.ipynb. It is
# NOT in boatphone/config.py, which is under separate adjudication and must not be
# edited from here (source of truth, Isaac, 2026-08-27). So this module becomes
# its first importable home.
#
# TWO CONSEQUENCES, both for the team, not for this file:
#   1. Malachy's notebook should import HYDROPHONE_LONLAT from here rather than
#      restating it -- otherwise this is a second definition, which is the drift
#      invariant 6 exists to prevent, and open item 5 of the acquisition handoff
#      is already an instance of exactly that failure.
#   2. When B7 lands `boatphone/geo.py` (acoustics_plan_v2 section 5), site geometry
#      should move there and this module should import it. Do not let a second
#      copy appear in geo.py.
#
# Source: the AOI was constructed as a 10 x 10 km box centred on this coordinate
# and the notebook asserts the centre is within 50 m of it. Acoustics interface I2
# (Malachy -> Isaac, "hydrophone coordinates, AOI centre and radius") is the
# authority; if I2 delivers a different number, change it HERE and nowhere else.
HYDROPHONE_LONLAT: tuple[float, float] = (-125.278277, 48.814200)
HYDROPHONE_SOURCE: str = (
    "planet_folger_search_order_download.ipynb cell 6 (AOI centre); pending "
    "confirmation against ONC device metadata for ICLISTENHF1266 via interface I2"
)

# Half-width of the analysis box, km. The Planet clip is a 10 x 10 km square
# centred on the hydrophone (data/folger_core_aoi.geojson), so 5.0 km reproduces
# the delivered extent exactly. Detections are only meaningful inside it.
AOI_HALF_KM: float = 5.0

# The AOI polygon Malachy ordered against. Read it rather than reconstructing the
# box from HYDROPHONE_LONLAT +/- AOI_HALF_KM: the ordered polygon carries a 5 m
# inset (handoff section 3) that a reconstructed box would not.
AOI_GEOJSON_NAME: str = "folger_core_aoi.geojson"


# ==========================================================================
# 2. VESSEL SIZE
# ==========================================================================
# FAO length-overall classes, named in CLAUDE.md invariant 6 and required by
# acoustics interface I3 (`size_class`). Edges in metres; the classes are
# [0, 12), [12, 24), [24, inf).
FAO_SIZE_EDGES_M: tuple[float, float] = (12.0, 24.0)
FAO_SIZE_LABELS: tuple[str, str, str] = ("0-12m", "12-24m", ">24m")

# WHICH FAO CLASSES THE SCIENCE TARGETS. This is the small-craft scope, and it is
# expressed as CLASSES, not as a length -- so it lands on FAO edges by
# construction and cannot bisect a class.
#
# It replaces the inherited `--max-length 20` in vessel_detect.py (plan section 6.6).
# 20 m is not an FAO edge: it sits inside [12, 24), so a 22 m vessel was dropped
# and an 18 m vessel kept while both report `12-24m`. That censors the 12-24 m
# class by an unknown fraction, and every recall-by-size-class number computed
# from it is biased low with nothing in the output saying so.
#
# WHY 12-24 m IS IN. The project targets vessels without AIS, and AIS carriage is
# set by tonnage and service, not by length -- Canadian requirements bind at
# 300 GT international / 500 GT domestic plus passenger vessels, none of which
# maps to a length edge. A 15 m private motor yacht is as AIS-dark as a 6 m
# runabout. Excluding 12-24 m would cut the sample on a criterion that does not
# track the property the project cares about, and would roughly halve an already
# small n. Decision 0016.
#
# APPLY THIS AT ANALYSIS TIME, NOT AT DETECTION TIME -- see select_size_classes()
# and the warning on classify_scene(). A large vessel that is out of scope for the
# fit is still very much in scope for deciding whether a scene is CLEAN.
TARGET_SIZE_CLASSES: tuple[str, ...] = ("0-12m", "12-24m")

# ---------------------------------------------------------------------------
# The other job the old constant was doing: rejecting things that are not
# vessels at all. That is an IMPLAUSIBILITY bound, not a science cut, and it
# belongs nowhere near an FAO edge.
#
# Set generously on purpose. Barkley Sound carries real large traffic -- the
# Bamfield ferry is ~38 m and Port Alberni bulk traffic transits the approaches --
# and those vessels MUST be detected even though the fit does not use them,
# because they are exactly what makes a scene not-clean. Above this a bright
# elongated feature at 3 m GSD is a wake, a slick or shoreline bleed.
#
# It is a backstop, not the primary wake filter: BLOB_MAX_ASPECT and
# BLOB_MIN_FILL do that work, and a wake long enough to fail this bound has
# already failed those. Sanity-check it against the largest vessel that actually
# transits the AOI before quoting any rejection count.
MAX_PLAUSIBLE_VESSEL_M: float = 150.0

# Lower bound. At 3 m GSD one pixel is 3 m, so anything reported below ~1.5 px of
# extent is quantisation noise, not a hull. Raise once the noise floor is measured
# on real imagery (plan section 6.6).
MIN_PLAUSIBLE_VESSEL_M: float = 0.0

# Below this length human labellers stop agreeing at 3 m GSD (plan section 13).
# Report the degradation curve's floor rather than pretending it has none.
LABELLER_CONFIDENCE_FLOOR_M: float = 6.0


# ==========================================================================
# 3. RADIOMETRY
# ==========================================================================
# Sentinel-2 L2A, processing baseline >= 04.00, carries an additive offset that
# MUST be applied before scaling. Miss it and every reflectance is low by 0.1:
# water goes near-black, NDWI shifts, the NIR threshold silently misfires, and
# nothing raises. The development scene is N0511, so it applies.
S2_QUANT: float = 10000.0
S2_BOA_OFFSET: float = -1000.0

# PlanetScope surface reflectance: scaled by 10000, no additive offset. CONFIRM
# against the first delivered scene's metadata rather than trusting this line.
PLANET_QUANT: float = 10000.0
PLANET_SR_OFFSET: float = 0.0

# Band order of `ortho_analytic_4b_sr`, 1-based as rasterio indexes.
PLANET_4B_BANDS: dict[str, int] = {"blue": 1, "green": 2, "red": 3, "nir": 4}
# SuperDove's 8-band product puts NIR at band 8, not band 4. A loader that assumes
# 4 bands and gets 8 will read RedEdge as NIR and every NIR statistic downstream
# is then wrong without erroring -- so a loader MUST assert the band count.
PLANET_8B_NIR_INDEX: int = 8

# The Sentinel-2 L1C TCI recipe: DN8 = clip(reflectance * 10000 / 10, 0, 255).
# A FIXED linear scale, deliberately not a per-scene percentile stretch: a stretch
# makes each scene's radiometry depend on its own content, which is precisely the
# domain drift a pretrained model is chosen to avoid (plan section 5.1).
TCI_DIVISOR: float = 10.0


# ==========================================================================
# 4. DETECTOR A (frozen pretrained YOLO)
# ==========================================================================
YOLO_HF_REPO: str = "mayrajeo/marine-vessel-yolo"
YOLO_HF_FILE: str = "yolo11s_tci.pt"          # AGPL-3.0; see plan section 14

# Tile size and network input size are the two knobs that set apparent object
# scale at the network input, which is the model's real constraint -- not metres.
# mayrajeo trained on 320 -> 640 tiles, so a 320 -> 640 slice at 3 m GSD presents
# a 10 m boat at ~6.6 px. NEVER resize the whole scene.
SLICE_PX: int = 320
IMGSZ: int = 640
TILE_OVERLAP: float = 0.2
NMS_IOU: float = 0.5

# Deliberately far below ultralytics' 0.25 default. For a domain-shifted model,
# recall at low confidence is what matters; precision is recovered downstream by
# the water mask, the size cut and the NIR cross-check. Using `conf` as the
# precision tool throws away exactly the marginal small-craft detections that are
# the point of the project.
CONF: float = 0.10

# ultralytics treats a numpy array as BGR and flips it internally; SAHI's wrapper
# does the opposite. An inverted channel order is a SILENT accuracy killer.
# Resolve it with the sweep before tuning anything else (plan section 6.1).
FEED_BGR: bool = True

SWEEP_SLICES: tuple[int, ...] = (256, 320, 512)
SWEEP_IMGSZ: tuple[int, ...] = (640, 1024)


# ==========================================================================
# 5. DETECTOR B (NIR bright-object screening)
# ==========================================================================
# THE WATER MASK IS BUILT BY SUBTRACTING LAND, AND LAND IS DEFINED BY EXTENT.
#
# The obvious rule -- "bright in NIR is land" -- deletes the targets. A vessel is
# bright in NIR; that is the entire basis of Detector B. Measured on the first
# delivered Planet scene (20200730_192942_71_1059): a per-pixel `nir > 0.05` land
# cut classified the two BRIGHTEST objects in the scene as land and removed them
# from the mask, so Detector B could not see them at any threshold. Of the 335
# bright-NIR connected components in that scene, only 8 are large enough to be
# land; the other 327 are the candidate vessel population.
#
# So: threshold NIR, take connected components, and call a component land only if
# it is LARGE. Size is what separates an island from a boat, and brightness is not.
LAND_NIR_MIN: float = 0.05        # a pixel bright enough to be land OR a vessel
MIN_LAND_AREA_M2: float = 20_000.0  # 2 ha. A 24 m vessel is ~200 m2, 100x smaller.

# NDWI = (green - nir) / (green + nir), kept as a DIAGNOSTIC, not as the mask.
#
# It is not the mask because it silently fails on an all-water AOI. On the same
# Planet scene, water reads green 0.0173 / NIR 0.0157 -- NDWI ~= -0.003, sitting
# exactly on the textbook 0.0 threshold, so it split the water's own noise in half
# and admitted 49%. There was nothing for it to segment: the AOI is 98% open
# water, with 1.75 km2 of land in 100 km2. `ndwi_land_fraction()` reports whether
# a land mode exists at all, so the caller finds out rather than being fooled.
NDWI_WATER_MIN: float = 0.0

# Buffer the land mask outward by this much before subtracting it. The Broken
# Group is 100+ islands and bright shoreline is the single dominant false
# positive here (plan section 7.2). Metres, not pixels, so 10 m S2 and 3 m Planet
# pull back from the shore by the same real distance.
WATER_ERODE_M: float = 9.0

# SUPERSEDED, kept only as an explanation of a bug that no longer exists here.
#
# An earlier version built the mask from NDWI and then closed holes up to a
# vessel beam, because a bright vessel has a NEGATIVE NDWI and punched a
# boat-shaped hole in the mask meant to contain it. Defining land by EXTENT
# removes the hole at its source: a vessel is never a large component, so it is
# never land, so there is no hole to fill. Two fixes for one bug is one too many
# -- the closing is gone, and this comment is what survives of it.

# Candidate = NIR above (water median + N * sigma), sigma from the MAD. Boats are
# outliers and would inflate a plain standard deviation, which is self-defeating:
# the brighter the boats, the higher the threshold that is supposed to find them.
# Run B permissively -- its false positives are cheap (a human clears them in the
# verification pass) but a false NEGATIVE breaks the cleanliness guarantee, which
# is the one thing B exists to provide.
NIR_MAD_N: float = 3.0
SWEEP_MAD_N: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0)

# ---------------------------------------------------------------------------
# THE THRESHOLD THAT DOES NOT ASSUME A DISTRIBUTION.
#
# The block above is wrong about the ocean, and the batch measured how wrong.
# Ocean NIR has a heavy tail: over the delivered 26 scenes, 6 sigma -> 10 sigma
# only HALVED the surviving pixel count (14,128 -> 7,450 on the reference scene)
# where a Gaussian predicts a drop of fourteen orders of magnitude. So `n_mad` is
# very nearly a DEAD KNOB -- there is no clean value to sweep to, which is why
# sweeping it harder was not the fix. Detection counts swung 258x (32 to 8,257)
# over the same 100 km2 with every scene >= 98% clear, driven by sea-surface
# texture rather than by anything about boats, because sigma measures the water's
# core while the count is set by its tail.
#
# The replacement spends a fixed CANDIDATE-PIXEL BUDGET per km2 of water and reads
# the cut off the empirical distribution (quantile_threshold). Read that
# docstring before changing this: the budget bounds the review workload, not the
# error, and a budget set below the true vessel pixel count deletes vessels.
#
# SOURCE OF THE VALUE: scripts/calibrate_nir_threshold.py over all 26 delivered
# SR scenes, 8 budgets x 6 sigma-multiples -> data/derived/nir_threshold_calibration.csv.
# (The same run reproduces the shipped batch exactly at n_mad=6: 32,877 detections,
# 14,128 candidate px on 20200730 -- so the comparison below is like for like.)
#
# The measurement that decides it, p90/p10 spread of detections per km2 of water
# ACROSS the 26 scenes -- how differently the same threshold behaves on the same
# 100 km2 of ocean:
#
#     n_mad = 3    7.5x        budget   2.5 px/km2   5.0x
#     n_mad = 4   26.7x        budget     5 px/km2   3.3x
#     n_mad = 5   19.4x        budget    20 px/km2   3.1x
#     n_mad = 6   14.9x   <--  budget    40 px/km2   2.4x   <-- this default
#     n_mad = 8   11.1x        budget    80 px/km2   2.5x
#     n_mad = 10   9.2x        budget   160 px/km2   4.0x
#
# 40 px/km2 sits at the minimum of that curve and yields a median 50 detections
# per scene (9 transient), against 730 (200 transient) at the n_mad=6 the batch
# ran. For scale, Detector A -- an independent detector with different physics --
# returned ~48 per scene on the same 26 scenes. That is a sanity anchor, NOT a
# truth set; there is no truth set, and section 6.2 still forbids ranking on
# counts.
#
# WHAT THIS DEFAULT RISKS, SAID PLAINLY. 40 px/km2 lands between 15 and 52 sigmas
# depending on the scene (p10..p90), far above the n_mad=6 the batch ran. Any real
# vessel dimmer than that cut is deleted, silently, and no sweep can find that out
# without the section 8 census. The reason it is still the better default is that
# the alternative is not "no deletion" but "deletion at an UNKNOWN and
# scene-dependent rate" -- which is what a fixed n_mad does, and the 14.9x spread
# is the measurement of it.
#
# See docs/decisions/0018-nir-threshold-is-a-budget-not-a-sigma.md.
NIR_BUDGET_PX_PER_KM2: float = 40.0

# "mad" keeps the batch reproducible against how it was measured; "quantile" is
# the recommended operating mode for new runs. Changing this default changes
# every downstream count, so it is switched deliberately, at the entry point.
NIR_THRESHOLD_MODE: str = "mad"

# Consistent-estimator constant: sigma ~= 1.4826 * MAD for Gaussian noise.
MAD_TO_SIGMA: float = 1.4826

# Blob shape filters. Glint streaks and wake lines are elongated; hulls are
# compact. Length is the long side of the minimum-area rectangle.
BLOB_MAX_ASPECT: float = 4.0
# Fraction of the enclosing rectangle the blob actually fills. A wake line is a
# thin diagonal that fills its bounding rectangle poorly.
BLOB_MIN_FILL: float = 0.35
BLOB_MIN_PIXELS: int = 2

# ---------------------------------------------------------------------------
# PERSISTENCE CELL. A detection recurring at the same coordinates in DIFFERENT
# YEARS is a fixed object -- rock, islet, beacon -- not a boat. This substitutes
# for the national rock layers mayrajeo/ship-detection filters against and we do
# not have, and it is the one part of Detector B that assumes nothing about the
# brightness distribution, which is why it survived decision 0018 intact.
#
# ~50 m cells: wide enough to absorb inter-scene georeferencing jitter, narrow
# enough not to merge a moored boat with the rock it is moored beside. Here
# rather than in a script because two entry points now grid on it and two
# definitions would silently disagree about what "the same place" means
# (invariant 6).
PERSISTENCE_CELL_DLAT: float = 0.00045
PERSISTENCE_CELL_DLON: float = 0.00068

# Above this aspect ratio a detection is flagged as carrying a wake. A crude
# proxy: only a moving boat radiates strongly, so this feeds the acoustic side --
# but VALIDATE IT BY EYE before it is used as anything but a flag.
WAKE_ASPECT: float = 2.5

# ---------------------------------------------------------------------------
# NIR SUPPORT FOR A DETECTOR A BOX -- an ADVISORY flag, never a filter.
#
# The model is fed 3-band RGB and never sees NIR, so asking "is there anything
# NIR-bright inside this box?" is genuinely independent evidence about a box the
# model produced. That independence is the same property Detector B is built on,
# applied per-box instead of scene-wide.
#
# IT MUST NOT CHANGE THE MODEL'S OUTPUT. Every box the model returns gets a row;
# this only says where a human should look first. A flag that removes rows is a
# filter wearing a flag's name.
#
# ABSENCE OF NIR BRIGHTNESS IS NOT ABSENCE OF A BOAT. A dark hull on dark water
# has no NIR excess, and dark hulls are Detector B's own documented failure mode
# (plan section 2: "fails on: dark hulls, whitecaps/foam, shoreline bleed"). The
# flag means UNSUPPORTED BY NIR, which is why `nir_support` is graded beside the
# boolean rather than replaced by it.
NIR_SUPPORT_MAD_N: float = 8.0    # stricter than NIR_MAD_N: this confirms, not discovers
NIR_SUPPORT_WEAK_PX: int = 2      # 1-2 bright px is "weak", 3+ is "strong", 0 is "none"


# ==========================================================================
# 6. FUSION AND SCENE CLASSIFICATION
# ==========================================================================
# Two detections are the same vessel if their centroids are within this distance.
# 30 m is 10 px at Planet's 3 m GSD and 3 px at S2's 10 m: wide enough to absorb
# the centroid difference between a YOLO box on a hull-plus-wake and a NIR blob on
# the hull alone, tight enough that two moored boats do not merge.
FUSE_TOL_M: float = 30.0

# Screening spreading coefficient for the dominance weight w_i = r_i ** (-k/10).
# 15 is intermediate/cylindrical spreading; 20 would be spherical. THIS IS THE
# CIRCULARITY the plan names out loud (section 9.4): we screen with k and we also
# fit k. The resolution is to re-run classification with the fitted k afterwards
# and report whether the scene set is stable -- a sensitivity check, not a flaw,
# but only if it is actually run.
NOMINAL_SPREADING_K: float = 15.0

# dominance = max(w_i) / sum(w_i). At or above this, one vessel accounts for
# enough of the expected received level that the scene can be treated as a single
# source. 0.6 is degrade rung 1 (plan section 12) -- report both.
DOMINANCE_MIN: float = 0.8
DOMINANCE_MIN_DEGRADED: float = 0.6

CLASS_EMPTY: str = "EMPTY"
CLASS_SINGLE: str = "SINGLE-DOMINANT"
CLASS_CROWDED: str = "CROWDED"


# ==========================================================================
# 7. SCHEMAS
# ==========================================================================
# Option C plan section 10. `detections.csv` keeps the schema vessel_detect.py
# already emits so nothing built against it breaks; `detector` and `nir_snr` are
# the Option C additions.
DETECTIONS_FIELDS: tuple[str, ...] = (
    "scene_id", "acq_time_utc", "lon", "lat", "bbox_w_m", "bbox_h_m",
    "length_class_m", "confidence", "aspect_ratio", "wake_flag", "range_km",
    "detector", "nir_snr",
)

# `scenes.csv` is the NEW file and the actual join key for the acoustic side under
# the single-dominant-source framing: the join moves from detection level to scene
# level.
SCENES_FIELDS: tuple[str, ...] = (
    "scene_id", "acq_time_utc", "n_detections", "dominance", "class",
    "r_dominant_km", "wake_flag_dominant", "verified",
)

# Acoustics interface I3 (acoustics_plan_v2.md section 6) asks for a DIFFERENT
# artefact: data/interim/detections.parquet, keyed on `overpass_id`, with
# `length_px`, `length_m_est` and `size_class`. That is not a competing schema to
# argue about -- it is a view. `to_i3_records()` below produces it from the same
# rows, so there is one source of truth and two contracts, rather than two
# pipelines that can disagree.
I3_FIELDS: tuple[str, ...] = (
    "overpass_id", "detection_id", "lat", "lon", "length_px", "length_m_est",
    "size_class", "confidence",
)

# A per-scene QC and hand-off export: one CSV per image, carrying the BOX rather
# than only its centroid. The box is the evidence a human judges a detection on;
# the point is the summary. Deliberately SEPARATE from DETECTIONS_FIELDS, which
# plan section 10 froze for the acoustics join and which must not grow columns.
#
# Corners are UL/UR/LR/LL in image order, which for a north-up raster is
# NW/NE/SE/SW. `_x`/`_y` are in the scene's own CRS (EPSG:32610 for these
# deliveries) so the file drops straight onto the imagery with no reprojection;
# `_lon`/`_lat` are EPSG:4326 for everything else.
DETECTION_CORNER_FIELDS: tuple[str, ...] = (
    "scene_id", "acq_time_utc", "detection_id", "confidence",
    "ul_lon", "ul_lat", "ur_lon", "ur_lat", "lr_lon", "lr_lat", "ll_lon", "ll_lat",
    "ul_x", "ul_y", "ur_x", "ur_y", "lr_x", "lr_y", "ll_x", "ll_y",
    "centroid_lon", "centroid_lat", "centroid_x", "centroid_y",
    "range_m", "range_km", "bearing_deg_from_hydrophone",
    "bbox_w_m", "bbox_h_m", "length_m", "aspect_ratio", "area_m2", "size_class",
    "dist_to_scene_edge_m",
    "nir_peak", "nir_x_water", "nir_bright_px", "nir_fill", "nir_support",
    "possible_false_positive", "nir_threshold_rho",
    "slice_px", "imgsz", "conf_threshold", "channel_order", "weights_file",
    # radiometry is 'toa' or 'sr'. It belongs with the other provenance columns
    # because it moves the result more than any of them: same scene, same config,
    # 46 detections on TOA against 17 on SR. Two files without this column cannot
    # be told apart after the fact.
    "radiometry",
)

# The same file without the NIR annotation: strictly what the model produced.
DETECTION_CORNER_FIELDS_NO_NIR: tuple[str, ...] = tuple(
    f for f in DETECTION_CORNER_FIELDS
    if not f.startswith("nir_") and f != "possible_false_positive")


# ==========================================================================
# 8. SIZE AND GEOMETRY
# ==========================================================================


def size_class(length_m: float) -> str:
    """FAO length-overall class for a detection. See FAO_SIZE_EDGES_M."""
    if not np.isfinite(length_m) or length_m < 0:
        raise ValueError(f"length_m must be finite and non-negative, got {length_m!r}")
    lo, hi = FAO_SIZE_EDGES_M
    if length_m < lo:
        return FAO_SIZE_LABELS[0]
    if length_m < hi:
        return FAO_SIZE_LABELS[1]
    return FAO_SIZE_LABELS[2]


def size_class_of(record: dict) -> str:
    """FAO class of a detections-schema row, from its `length_class_m`.

    Not stored as a column: `detections.csv` is the schema the plan froze
    (section 10) and `length_class_m` already determines the class, so a stored
    `size_class` would be a second copy that can disagree with it.
    """
    return size_class(float(record["length_class_m"]))


def select_size_classes(records: Sequence[dict],
                        classes: Sequence[str] = TARGET_SIZE_CLASSES
                        ) -> tuple[list[dict], dict]:
    """Apply the small-craft SCOPE, on FAO edges, at analysis time.

    Returns (in-scope rows, count by class over ALL rows). The tally is returned
    rather than discarded because "we fitted 18 small craft" means very little
    without "and excluded 4 vessels over 24 m from the same scenes" beside it.

    Call this when selecting what goes INTO the SL/k fit -- never before
    classify_scene().
    """
    tally: dict[str, int] = {label: 0 for label in FAO_SIZE_LABELS}
    for r in records:
        tally[size_class_of(r)] += 1
    keep = [r for r in records if size_class_of(r) in classes]
    return keep, tally


def range_km(lon: float, lat: float, origin_lonlat: Sequence[float] | None = None) -> float:
    """Geodesic distance from the hydrophone, km, on the WGS84 ellipsoid.

    Uses pyproj.Geod rather than the haversine in vessel_detect.py. The two agree
    to ~0.2% at this latitude, which is far below the range uncertainty of a 3 m
    pixel -- the reason to prefer Geod is that acoustics B7 does its range and
    bearing with pyproj (acoustics_plan_v2 section 5), and two range columns computed
    two ways is the kind of quiet inconsistency that shows up as scatter in the
    SL/k fit and gets blamed on the physics.
    """
    from pyproj import Geod

    o = HYDROPHONE_LONLAT if origin_lonlat is None else origin_lonlat
    _, _, dist_m = Geod(ellps="WGS84").inv(o[0], o[1], lon, lat)
    return float(dist_m) / 1000.0


def bearing_deg(lon: float, lat: float, origin_lonlat: Sequence[float] | None = None) -> float:
    """Forward azimuth hydrophone -> detection, degrees clockwise from true north.

    Not used by the Option C fit, but it is what a directional/aspect-dependence
    check would need and it is free here.
    """
    from pyproj import Geod

    o = HYDROPHONE_LONLAT if origin_lonlat is None else origin_lonlat
    az, _, _ = Geod(ellps="WGS84").inv(o[0], o[1], lon, lat)
    return float(az) % 360.0


# ==========================================================================
# 9. WATER MASK
# ==========================================================================


@dataclass
class MaskReport:
    """What the water mask did, in numbers. Printed, not swallowed.

    Invariant 5: if pixels are dropped, say how many. A water mask that quietly
    removes 90% of a scene looks identical downstream to a scene with no boats
    in it, and only one of those is a bug.
    """
    valid_px: int
    water_px: int
    land_px: int
    land_components: int
    small_bright_components: int   # bright but too small to be land: candidate vessels
    buffered_px: int
    cloud_removed_px: int
    ndwi_land_fraction: float      # DIAGNOSTIC ONLY -- see NDWI_WATER_MIN

    @property
    def water_fraction(self) -> float:
        return self.water_px / self.valid_px if self.valid_px else float("nan")

    def __str__(self) -> str:
        return (f"water mask: {self.water_px:,} px "
                f"({100 * self.water_fraction:.1f}% of {self.valid_px:,} valid) | "
                f"land {self.land_px:,} px in {self.land_components} components "
                f"(+{self.buffered_px:,} px shore buffer) | "
                f"{self.small_bright_components} bright components too small to be "
                f"land -> candidate vessels | "
                f"cloud/haze removed {self.cloud_removed_px:,} | "
                f"NDWI would call {100*self.ndwi_land_fraction:.0f}% land")


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """(green - nir) / (green + nir), NaN where the denominator vanishes.

    NaN, not 0: a zero denominator means both bands read zero, which is nodata or
    deep shadow, and calling that "NDWI = 0" would place it exactly on the water
    threshold. Invariant 5 -- do not paper over a data hole with a plausible
    number. Callers combine with `valid` and count what they dropped.
    """
    green = np.asarray(green, dtype=np.float32)
    nir = np.asarray(nir, dtype=np.float32)
    if green.shape != nir.shape:
        raise ValueError(f"green {green.shape} and nir {nir.shape} differ in shape")
    denom = green + nir
    out = np.full(green.shape, np.nan, dtype=np.float32)
    ok = denom != 0
    np.divide(green - nir, denom, out=out, where=ok)
    return out


def land_mask(nir: np.ndarray, valid: np.ndarray, res_m: float,
              nir_min: float = LAND_NIR_MIN,
              min_area_m2: float = MIN_LAND_AREA_M2) -> tuple[np.ndarray, int, int]:
    """Land = a LARGE connected region of NIR-bright pixels. Returns (land, kept, small).

    The size test is the whole point. Brightness alone cannot separate an island
    from a boat, because a boat is bright -- that is why Detector B works at all.
    `small` counts the bright components rejected as too small to be land; those
    are the candidate vessel population, and reporting the number is how a caller
    notices when a mask change has quietly eaten them.
    """
    import cv2

    nir = np.asarray(nir, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    bright = ((nir > nir_min) & valid).astype(np.uint8)
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)

    px_m2 = res_m * res_m
    keep = [i for i in range(1, n_lab)
            if stats[i, cv2.CC_STAT_AREA] * px_m2 >= min_area_m2]
    land = np.isin(lab, keep) if keep else np.zeros(nir.shape, bool)
    return land, len(keep), max(n_lab - 1 - len(keep), 0)


def ndwi_land_fraction(green: np.ndarray, nir: np.ndarray, valid: np.ndarray,
                       ndwi_min: float = NDWI_WATER_MIN) -> float:
    """Fraction NDWI would call land. A DIAGNOSTIC, deliberately not the mask.

    Read it against the land fraction the extent rule actually found. If NDWI says
    50% and the extent rule says 2%, NDWI has no land mode to find and is splitting
    the water's own noise -- which is what it did on the first delivered Planet
    scene, and what would have been invisible without this number.
    """
    valid = np.asarray(valid, dtype=bool)
    idx = ndwi(green, nir)[valid]
    idx = idx[np.isfinite(idx)]
    return float((idx < ndwi_min).mean()) if idx.size else float("nan")


def water_mask(green: np.ndarray, nir: np.ndarray, valid: np.ndarray,
               clear: np.ndarray | None = None, res_m: float = 3.0,
               nir_min: float = LAND_NIR_MIN,
               min_land_area_m2: float = MIN_LAND_AREA_M2,
               buffer_m: float = WATER_ERODE_M) -> tuple[np.ndarray, MaskReport]:
    """water = valid AND NOT land AND NOT cloud, with land defined by EXTENT.

    Built by SUBTRACTING land rather than by selecting water. That ordering is
    what keeps vessels in: anything bright and small is left in the mask by
    construction, instead of being selected out and then patched back.

    `buffer_m` dilates land before subtracting, pulling the mask back from the
    shoreline -- the dominant false-positive source in an archipelago. In METRES,
    so 10 m Sentinel-2 and 3 m PlanetScope pull back the same real distance.

    Set `min_land_area_m2 = 0` to disable the extent rule and fall back to the
    brightness cut. That switch exists so the self-test can demonstrate the bug
    the extent rule prevents rather than asserting it in a comment.
    """
    import cv2

    valid = np.asarray(valid, dtype=bool)
    land, n_land, n_small = land_mask(nir, valid, res_m, nir_min, min_land_area_m2)
    if min_land_area_m2 <= 0:                      # the broken behaviour, on purpose
        land = (np.asarray(nir, dtype=np.float32) > nir_min) & valid
        n_land, n_small = -1, -1
    land_px = int(land.sum())

    r = max(1, int(round(buffer_m / max(res_m, 1e-6))))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    buffered = cv2.dilate(land.astype(np.uint8), disk, iterations=1).astype(bool)
    water = valid & ~buffered

    cloud_removed = 0
    if clear is not None:
        clear = np.asarray(clear, dtype=bool)
        if clear.shape != water.shape:
            raise ValueError(
                f"UDM2 clear mask {clear.shape} does not match scene {water.shape}. "
                "Silently ignoring it would leave cloud edges in the water mask, "
                "where they read as bright NIR blobs -- i.e. as boats.")
        cloud_removed = int((water & ~clear).sum())
        water &= clear

    return water, MaskReport(
        valid_px=int(valid.sum()), water_px=int(water.sum()), land_px=land_px,
        land_components=n_land, small_bright_components=n_small,
        buffered_px=int(buffered.sum()) - land_px, cloud_removed_px=cloud_removed,
        ndwi_land_fraction=ndwi_land_fraction(green, nir, valid))


# ==========================================================================
# 10. DETECTOR B -- adaptive NIR threshold
# ==========================================================================


@dataclass
class Blob:
    """One NIR bright-object candidate, in pixel space."""
    row: float                 # centroid, image row
    col: float                 # centroid, image col
    n_px: int
    length_m: float            # long side of the minimum-area rectangle
    width_m: float             # short side
    fill: float                # blob pixels / rectangle pixels
    nir_peak: float            # reflectance
    nir_snr: float             # (peak - water median) / sigma_MAD
    rejected: str = ""         # non-empty if filtered out, saying which filter

    @property
    def aspect_ratio(self) -> float:
        return self.length_m / max(self.width_m, 1e-6)


def _water_scale(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(finite values, median, sigma_MAD) -- the one definition of the water scale.

    Shared by both thresholds so a scene's reported `water_median` and `sigma_mad`
    mean the same thing whichever threshold produced its detections. The quantile
    threshold does not USE sigma to place its cut, but every downstream consumer
    reads nir_snr, which is expressed in sigmas.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite values to threshold -- is the water mask empty?")
    med = float(np.median(v))
    sigma = MAD_TO_SIGMA * float(np.median(np.abs(v - med)))
    if sigma == 0.0:
        # Quantised imagery over flat water can give an exactly-zero MAD. Falling
        # back to the s.d. is documented, not silent: with MAD == 0 there is no
        # robust scale to be had, and returning 0 would make the threshold equal
        # the median and select half the ocean.
        sigma = float(v.std())
        if sigma == 0.0:
            raise ValueError(
                "NIR is constant over the whole water mask (MAD and s.d. both 0). "
                "That is a data problem -- a stuck band or an empty mask -- not a "
                "scene with no boats in it.")
    return v, med, sigma


def robust_threshold(values: np.ndarray, n_mad: float = NIR_MAD_N) -> tuple[float, float, float]:
    """(threshold, median, sigma) with sigma from the MAD, not the s.d.

    Boats are the outliers we are looking for. A plain standard deviation is
    inflated by exactly the pixels of interest, so the threshold rises with the
    brightness of the thing it is meant to catch.

    KNOWN LIMITATION, MEASURED, NOT HYPOTHETICAL -- see quantile_threshold() and
    docs/decisions/0018. `sigma` is a Gaussian scale estimate and ocean NIR is not
    Gaussian, so `n_mad` is very nearly a dead knob: on the delivered batch,
    raising it from 6 to 10 only halves the surviving pixel count where a Gaussian
    would drop it fourteen orders of magnitude. Kept as the reference detector and
    for the sweep, not as the recommended operating mode.
    """
    _, med, sigma = _water_scale(values)
    return med + n_mad * sigma, med, sigma


def quantile_threshold(values: np.ndarray,
                       target_px: int) -> tuple[float, float, float, int, float]:
    """(threshold, median, sigma, selected_px, quantile) for a fixed pixel budget.

    THE DISTRIBUTION-FREE REPLACEMENT FOR robust_threshold(). Instead of asking
    "how many sigmas above the water is bright?" -- a question whose answer depends
    on a Gaussian assumption the ocean violates -- it asks "which cut admits
    `target_px` pixels?" and reads the answer off the empirical distribution. The
    ordering of pixels by brightness is all it uses, so the shape of the tail is
    irrelevant by construction.

    WHAT THIS FIXES. Under the MAD threshold, sigma measures the water's CORE
    texture while the candidate count is set by its TAIL, and the two are not
    coupled: on a scene whose core is quiet the threshold slides toward the median
    and the tail floods through. That is the mechanism behind the batch's 258x
    swing in detection count over the same 100 km2. A pixel budget makes the
    candidate count a controlled input rather than an emergent property of sea
    state, which is what makes counts comparable BETWEEN scenes.

    WHAT THIS DOES NOT FIX, AND MUST NOT BE READ AS FIXING. The budget bounds the
    workload, not the error. Real vessels are inside it -- they are among the
    brightest water pixels, which is the whole premise of Detector B -- so this is
    a FALSE-ALARM BUDGET only to the extent that vessels are a small fraction of
    it. Set it below the true vessel pixel count and it deletes vessels silently,
    exactly as a too-high `n_mad` would. It buys comparability, not truth.

    TIES ARE REPORTED, NOT HIDDEN. PlanetScope reflectance is quantised, so the
    budget-th largest value can repeat; `>= threshold` then admits more than
    `target_px`. The achieved count is returned so the caller can see the overrun
    instead of assuming the budget was met (invariant 5).
    """
    v, med, sigma = _water_scale(values)
    n = v.size
    target_px = int(target_px)
    if target_px < 1:
        raise ValueError(f"target_px must be at least 1, got {target_px}")
    if target_px >= n:
        raise ValueError(
            f"target_px {target_px} >= {n} water pixels: the budget is the entire "
            "water mask, so there is no threshold to place. Either the budget is "
            "wrong or the water mask is nearly empty -- check which before "
            "lowering the budget.")
    # The budget-th largest value. Everything >= it is inside the budget.
    thr = float(np.partition(v, n - target_px)[n - target_px])
    selected = int((v >= thr).sum())
    return thr, med, sigma, selected, 1.0 - target_px / n


def detect_nir_blobs(nir: np.ndarray, water: np.ndarray, res_m: float,
                     n_mad: float = NIR_MAD_N,
                     min_length_m: float = MIN_PLAUSIBLE_VESSEL_M,
                     max_length_m: float = MAX_PLAUSIBLE_VESSEL_M,
                     max_aspect: float = BLOB_MAX_ASPECT,
                     min_fill: float = BLOB_MIN_FILL,
                     min_px: int = BLOB_MIN_PIXELS,
                     keep_rejected: bool = False,
                     threshold_mode: str = NIR_THRESHOLD_MODE,
                     budget_px_per_km2: float = NIR_BUDGET_PX_PER_KM2,
                     ) -> tuple[list[Blob], dict]:
    """Detector B: bright objects on dark water, found by contrast alone.

    The model in Detector A has physically never seen the NIR band, so it cannot
    be checking it. That is the whole reason B exists: an independent sweep for
    what A missed, which is what turns "the detectors agree" into "this scene is
    clean".

    The length bounds here are IMPLAUSIBILITY bounds -- they reject things that
    are not vessels. They are deliberately not the small-craft scope: a 40 m
    vessel is detected, classified `>24m`, and excluded from the fit later by
    select_size_classes(). Dropping it here instead would delete it from the
    dominance calculation and let a swamped scene pass as clean.

    TWO WAYS TO PLACE THE THRESHOLD, and they answer different questions.
    `threshold_mode="mad"` is the original: water median + `n_mad` * sigma_MAD.
    `threshold_mode="quantile"` spends a fixed budget of `budget_px_per_km2`
    candidate pixels per km2 of water and reads the cut off the empirical
    distribution -- see quantile_threshold() for why the MAD cut behaves badly on
    real ocean NIR and what the budget does and does not buy. Both are kept: the
    MAD cut is the reference the batch was measured with, and disagreement
    between them on the same scene is itself a diagnostic.

    Returns (blobs, diagnostics). Rejected candidates are counted by reason and,
    with `keep_rejected`, returned too -- the rejection reasons are what you tune
    on, and a filter that silently eats every real boat looks exactly like water
    with no boats in it.
    """
    import cv2

    nir = np.asarray(nir, dtype=np.float32)
    water = np.asarray(water, dtype=bool)
    if nir.shape != water.shape:
        raise ValueError(f"nir {nir.shape} and water {water.shape} differ in shape")

    water_km2 = float(water.sum()) * res_m * res_m / 1e6
    budget_px: int = 0
    quantile: float | None = None
    selected_px: int | None = None
    if threshold_mode == "mad":
        thr, med, sigma = robust_threshold(nir[water], n_mad)
    elif threshold_mode == "quantile":
        if budget_px_per_km2 <= 0:
            raise ValueError(
                f"budget_px_per_km2 must be positive, got {budget_px_per_km2}")
        budget_px = int(round(budget_px_per_km2 * water_km2))
        if budget_px < 1:
            raise ValueError(
                f"a budget of {budget_px_per_km2} px/km2 over {water_km2:.3f} km2 "
                "of water rounds to zero candidate pixels. That is a scene with "
                "almost no water in it, not a scene with no boats -- check the "
                "mask before lowering the budget.")
        thr, med, sigma, selected_px, quantile = quantile_threshold(nir[water], budget_px)
    else:
        raise ValueError(
            f"threshold_mode must be 'mad' or 'quantile', got {threshold_mode!r}")
    candidate = (nir >= thr) & water

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8)

    px_area = res_m * res_m
    blobs: list[Blob] = []
    counts = {"raw": max(n_labels - 1, 0), "too_few_px": 0, "sub_pixel_length": 0,
              "implausible_length": 0, "too_elongated": 0, "too_sparse": 0, "kept": 0}

    for lab in range(1, n_labels):
        n_px = int(stats[lab, cv2.CC_STAT_AREA])
        x0 = int(stats[lab, cv2.CC_STAT_LEFT])
        y0 = int(stats[lab, cv2.CC_STAT_TOP])
        w = int(stats[lab, cv2.CC_STAT_WIDTH])
        h = int(stats[lab, cv2.CC_STAT_HEIGHT])
        sub = (labels[y0:y0 + h, x0:x0 + w] == lab)

        # Minimum-area rectangle gives length and width for a hull at any heading;
        # the axis-aligned bounding box would report a diagonal 10 m boat as 14 m.
        found = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        contours = found[-2]
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.concatenate(contours))
        # +1 px on each side. minAreaRect measures corner-to-corner between pixel
        # CENTRES, so an n-pixel-wide object measures (n-1). Left uncorrected that
        # is a systematic 1 px undercount on every detection -- 3 m at PlanetScope
        # GSD, which is 50% of a 2 px hull and enough to move vessels across the
        # 12 m FAO edge in the wrong direction. A single-pixel component measures
        # 0 and becomes 1, which is the smallest honest size for it.
        long_px = max(rw, rh) + 1.0
        short_px = min(rw, rh) + 1.0
        length_m = long_px * res_m
        width_m = short_px * res_m
        fill = n_px / max(long_px * short_px, 1e-6)

        peak = float(nir[y0:y0 + h, x0:x0 + w][sub].max())
        snr = (peak - med) / sigma

        blob = Blob(row=float(centroids[lab][1]), col=float(centroids[lab][0]),
                    n_px=n_px, length_m=length_m, width_m=width_m, fill=fill,
                    nir_peak=peak, nir_snr=float(snr))

        if n_px < min_px:
            blob.rejected = "too_few_px"
        elif length_m < min_length_m:
            blob.rejected = "sub_pixel_length"
        elif length_m > max_length_m:
            blob.rejected = "implausible_length"    # wake, slick, shoreline bleed
        elif blob.aspect_ratio > max_aspect:
            blob.rejected = "too_elongated"     # wake line or glint streak
        elif fill < min_fill:
            blob.rejected = "too_sparse"        # scattered glint, not a hull

        counts[blob.rejected or "kept"] += 1
        if not blob.rejected or keep_rejected:
            blobs.append(blob)

    diagnostics = {
        "threshold_mode": threshold_mode,
        "threshold_reflectance": thr, "water_median": med, "sigma_mad": sigma,
        # `n_mad` is what was ASKED FOR and is meaningless in quantile mode;
        # `n_mad_effective` is where the cut actually landed, in sigmas, and is
        # defined in both modes. In mad mode they are equal by construction. The
        # spread of n_mad_effective across scenes at a FIXED budget is the direct
        # measurement of how badly a fixed n_mad travels between scenes.
        "n_mad": n_mad, "n_mad_effective": (thr - med) / sigma,
        "budget_px_per_km2": budget_px_per_km2 if threshold_mode == "quantile" else None,
        "budget_px": budget_px, "budget_selected_px": selected_px,
        "quantile": quantile,
        "candidate_px": int(candidate.sum()),
        "water_px": int(water.sum()), "water_km2": water_km2,
        "px_area_m2": px_area, **counts,
    }
    return [b for b in blobs if keep_rejected or not b.rejected], diagnostics


# ==========================================================================
# 11. RECORDS, FUSION, SCENE CLASSIFICATION
# ==========================================================================

ToLonLat = Callable[[Sequence[float], Sequence[float]], tuple[np.ndarray, np.ndarray]]
"""(cols, rows) -> (lon, lat). Supplied by the loader so this module needs no CRS
handling and therefore no rasterio. Scene.to_lonlat in the notebook matches it."""


def _record(scene_id: str, acq: str, lon: float, lat: float, w_m: float, h_m: float,
            length_m: float, conf: float, detector: str, nir_snr: float | str,
            origin_lonlat: Sequence[float] | None) -> dict:
    short = max(min(w_m, h_m), 1e-6)
    aspect = length_m / short
    return {
        "scene_id": scene_id,
        "acq_time_utc": acq,
        "lon": round(float(lon), 6),
        "lat": round(float(lat), 6),
        "bbox_w_m": round(float(w_m), 1),
        "bbox_h_m": round(float(h_m), 1),
        "length_class_m": round(float(length_m), 1),
        "confidence": round(float(conf), 4),
        "aspect_ratio": round(float(aspect), 2),
        "wake_flag": int(aspect >= WAKE_ASPECT),
        "range_km": round(range_km(lon, lat, origin_lonlat), 3),
        "detector": detector,
        "nir_snr": (round(float(nir_snr), 2) if isinstance(nir_snr, (int, float))
                    else nir_snr),
    }


def boxes_to_records(boxes: Iterable[tuple[Sequence[float], float]], scene_id: str,
                     acq: str, res_m: float, to_lonlat: ToLonLat,
                     valid: np.ndarray | None = None,
                     water: np.ndarray | None = None,
                     min_length_m: float = MIN_PLAUSIBLE_VESSEL_M,
                     max_length_m: float = MAX_PLAUSIBLE_VESSEL_M,
                     origin_lonlat: Sequence[float] | None = None) -> tuple[list[dict], dict]:
    """Detector A boxes (xyxy in scene pixels, score) -> detections-schema rows.

    Applying `water` to A as well as B is plan section 6.5: the Sentinel-2 model never
    learned to reject 3 m shoreline because at 10 m that texture did not exist.

    As in detect_nir_blobs, the length bounds are implausibility bounds, not the
    small-craft scope. Scope is applied by select_size_classes() at analysis time.
    """
    kept, counts = [], {"raw": 0, "outside_valid": 0, "outside_water": 0,
                        "sub_pixel_length": 0, "implausible_length": 0, "kept": 0}
    cols, rows, staged = [], [], []
    for (x1, y1, x2, y2), score in boxes:
        counts["raw"] += 1
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ci, ri = int(round(cx)), int(round(cy))
        ref = valid if valid is not None else water
        if ref is not None and not (0 <= ri < ref.shape[0] and 0 <= ci < ref.shape[1]):
            counts["outside_valid"] += 1
            continue
        if valid is not None and not valid[ri, ci]:
            counts["outside_valid"] += 1
            continue
        if water is not None and not water[ri, ci]:
            counts["outside_water"] += 1
            continue
        w_m, h_m = (x2 - x1) * res_m, (y2 - y1) * res_m
        length = max(w_m, h_m)
        if length < min_length_m:
            counts["sub_pixel_length"] += 1
            continue
        if length > max_length_m:
            counts["implausible_length"] += 1
            continue
        cols.append(cx)
        rows.append(cy)
        staged.append((w_m, h_m, length, float(score)))

    if staged:
        lons, lats = to_lonlat(cols, rows)
        for (w_m, h_m, length, score), lon, lat in zip(staged, lons, lats):
            kept.append(_record(scene_id, acq, lon, lat, w_m, h_m, length, score,
                                "A", "", origin_lonlat))
    counts["kept"] = len(kept)
    return kept, counts


def blobs_to_records(blobs: Iterable[Blob], scene_id: str, acq: str,
                     to_lonlat: ToLonLat,
                     origin_lonlat: Sequence[float] | None = None) -> list[dict]:
    """Detector B blobs -> the same schema, so A and B rows are interchangeable.

    `confidence` for B is the NIR SNR passed through a bounded squash. It is NOT a
    probability and must not be compared numerically against A's YOLO confidence;
    the column exists so one schema serves both detectors, and `detector` is what
    tells them apart.
    """
    blobs = list(blobs)
    if not blobs:
        return []
    lons, lats = to_lonlat([b.col for b in blobs], [b.row for b in blobs])
    out = []
    for b, lon, lat in zip(blobs, lons, lats):
        pseudo_conf = float(np.tanh(b.nir_snr / 10.0))
        out.append(_record(scene_id, acq, lon, lat, b.width_m, b.length_m,
                           b.length_m, pseudo_conf, "B", b.nir_snr, origin_lonlat))
    return out


def box_corners(box: Sequence[float]) -> tuple[list[float], list[float]]:
    """(x1,y1,x2,y2) -> (cols, rows) for the four corners, UL/UR/LR/LL.

    Image order, which for a north-up raster is NW/NE/SE/SW. Returned as two
    parallel lists so they feed a `to_lonlat`-shaped callable directly.

    PIXEL CONVENTION, stated rather than assumed. These are passed to rasterio's
    `xy(..., offset="center")`, matching `boxes_to_records()`, so corners and
    centroid are consistent with each other by construction. Whether YOLO's box
    edges should instead be read as pixel *edges* (offset="ul") is an open
    half-pixel question worth 1.5 m at 3 m GSD and 5 m at 10 m -- below the box's
    own looseness, but real. Same class of open convention as decision 0013's
    edge-vs-centre FFT binning; recorded here so a later change is deliberate.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    return [x1, x2, x2, x1], [y1, y1, y2, y2]


def nir_support_for_box(box: Sequence[float], nir: np.ndarray, threshold: float,
                        water_median: float) -> dict:
    """Independent NIR evidence for one Detector A box. ADVISORY -- never a filter.

    The model was fed 3-band RGB and never saw this band, so a bright cluster
    inside its box is evidence it did not have. Returns the counts and a graded
    `nir_support`; the caller decides nothing on the strength of it except where
    to send a human first. See NIR_SUPPORT_MAD_N.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    h, w = nir.shape
    c0, c1 = max(int(np.floor(x1)), 0), min(int(np.ceil(x2)) + 1, w)
    r0, r1 = max(int(np.floor(y1)), 0), min(int(np.ceil(y2)) + 1, h)
    sub = nir[r0:r1, c0:c1]
    if sub.size == 0:
        return {"nir_peak": "", "nir_x_water": "", "nir_bright_px": 0,
                "nir_fill": 0.0, "nir_support": "none", "possible_false_positive": 1}
    bright = int((sub > threshold).sum())
    peak = float(sub.max())
    support = ("none" if bright == 0
               else "weak" if bright <= NIR_SUPPORT_WEAK_PX else "strong")
    return {
        "nir_peak": round(peak, 4),
        "nir_x_water": (round(peak / water_median, 1) if water_median > 0 else ""),
        "nir_bright_px": bright,
        "nir_fill": round(bright / sub.size, 3),
        "nir_support": support,
        # "unsupported by NIR", NOT "this is not a boat" -- a dark hull on dark
        # water produces exactly this. The human pass decides.
        "possible_false_positive": int(support == "none"),
    }


def boxes_to_corner_records(boxes: Iterable[tuple[Sequence[float], float]],
                            scene_id: str, acq: str, res_m: float,
                            to_lonlat: ToLonLat, to_xy: ToLonLat | None = None,
                            shape: tuple[int, int] | None = None,
                            nir: np.ndarray | None = None,
                            valid: np.ndarray | None = None,
                            config: dict | None = None,
                            origin_lonlat: Sequence[float] | None = None,
                            ) -> tuple[list[dict], float]:
    """Detector A boxes -> DETECTION_CORNER_FIELDS rows. Returns (rows, nir_threshold).

    EVERY box becomes a row. Nothing here filters -- not the size bounds, not the
    water mask, not the NIR evidence. This is the model's output plus annotation,
    which is what makes it usable as the input to human verification rather than
    as a result.

    `detection_id` is `boat_1`, `boat_2`, ... by descending confidence, so the
    first rows are the ones worth looking at first.
    """
    boxes = sorted(boxes, key=lambda d: -float(d[1]))
    threshold = float("nan")
    water_median = float("nan")
    if nir is not None:
        ref = nir[valid] if valid is not None else nir
        threshold, water_median, _ = robust_threshold(ref, NIR_SUPPORT_MAD_N)

    rows = []
    for i, (box, score) in enumerate(boxes, 1):
        x1, y1, x2, y2 = (float(v) for v in box)
        cols, rows_px = box_corners(box)
        lons, lats = to_lonlat(cols, rows_px)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        clon, clat = to_lonlat([cx], [cy])
        clon, clat = float(clon[0]), float(clat[0])

        w_m, h_m = (x2 - x1) * res_m, (y2 - y1) * res_m
        length = max(w_m, h_m)
        short = max(min(w_m, h_m), 1e-6)
        rng_km = range_km(clon, clat, origin_lonlat)

        row = {
            "scene_id": scene_id, "acq_time_utc": acq,
            "detection_id": f"boat_{i}", "confidence": round(float(score), 4),
            "centroid_lon": round(clon, 6), "centroid_lat": round(clat, 6),
            "range_m": round(rng_km * 1000.0, 1), "range_km": round(rng_km, 3),
            "bearing_deg_from_hydrophone": round(
                bearing_deg(clon, clat, origin_lonlat), 2),
            "bbox_w_m": round(w_m, 1), "bbox_h_m": round(h_m, 1),
            "length_m": round(length, 1), "aspect_ratio": round(length / short, 2),
            "area_m2": round(w_m * h_m, 1), "size_class": size_class(length),
        }
        for name, lo, la in zip(("ul", "ur", "lr", "ll"), lons, lats):
            row[f"{name}_lon"], row[f"{name}_lat"] = round(float(lo), 6), round(float(la), 6)

        if to_xy is not None:
            xs, ys = to_xy(cols, rows_px)
            for name, X, Y in zip(("ul", "ur", "lr", "ll"), xs, ys):
                row[f"{name}_x"], row[f"{name}_y"] = round(float(X), 2), round(float(Y), 2)
            cX, cY = to_xy([cx], [cy])
            row["centroid_x"], row["centroid_y"] = round(float(cX[0]), 2), round(float(cY[0]), 2)
        else:
            for name in ("ul", "ur", "lr", "ll"):
                row[f"{name}_x"] = row[f"{name}_y"] = ""
            row["centroid_x"] = row["centroid_y"] = ""

        # A box near the clip edge is TRUNCATED, so its length is short by an
        # unknown amount. Flagging the distance is cheaper than pretending the
        # measurement is sound.
        if shape is not None:
            H, W = shape
            row["dist_to_scene_edge_m"] = round(
                min(cx, cy, W - cx, H - cy) * res_m, 1)
        else:
            row["dist_to_scene_edge_m"] = ""

        if nir is not None:
            row.update(nir_support_for_box(box, nir, threshold, water_median))
            row["nir_threshold_rho"] = round(threshold, 5)

        cfg = config or {}
        for key in ("slice_px", "imgsz", "conf_threshold", "channel_order",
                    "weights_file", "radiometry"):
            row[key] = cfg.get(key, "")
        rows.append(row)
    return rows, threshold


def fuse(records_a: Sequence[dict], records_b: Sequence[dict],
         tol_m: float = FUSE_TOL_M) -> tuple[list[dict], dict]:
    """Merge A and B rows into one detection list, tagging agreement.

    `detector` becomes "AB" where the two independently found the same vessel,
    and stays "A" or "B" where only one did. The A-only / B-only split IS the
    methods result: B-only detections are what the pretrained Sentinel-2 model
    missed at 3 m, stratified by length.

    Greedy nearest-neighbour matching, deliberately. It is order-dependent in
    principle; with 1-2 vessels per scene and a 30 m tolerance the ambiguous case
    does not arise, and a greedy pass that can be read in ten seconds beats a
    Hungarian assignment nobody on the team will check.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    fused: list[dict] = []
    used_b: set[int] = set()

    for ra in records_a:
        best, best_d = None, math.inf
        for j, rb in enumerate(records_b):
            if j in used_b:
                continue
            _, _, d = geod.inv(ra["lon"], ra["lat"], rb["lon"], rb["lat"])
            if d < best_d:
                best, best_d = j, d
        if best is not None and best_d <= tol_m:
            used_b.add(best)
            merged = dict(ra)
            merged["detector"] = "AB"
            merged["nir_snr"] = records_b[best]["nir_snr"]
            # A's box is the better extent estimate for a vessel it actually saw
            # (it was trained to bound hulls); B's SNR is the better brightness
            # evidence. Keep one of each rather than averaging two different
            # quantities into a number that means nothing.
            fused.append(merged)
        else:
            fused.append(dict(ra))

    for j, rb in enumerate(records_b):
        if j not in used_b:
            fused.append(dict(rb))

    tally = {
        "A_total": len(records_a), "B_total": len(records_b),
        "AB": sum(1 for r in fused if r["detector"] == "AB"),
        "A_only": sum(1 for r in fused if r["detector"] == "A"),
        "B_only": sum(1 for r in fused if r["detector"] == "B"),
        "fused_total": len(fused),
    }
    return fused, tally


def dominance(ranges_km: Sequence[float], k: float = NOMINAL_SPREADING_K) -> float:
    """max(w_i) / sum(w_i) for w_i = r_i ** (-k/10).

    The share of expected received intensity contributed by the loudest vessel.
    NaN for an empty scene: there is no dominant source, and returning 1.0 would
    make every empty scene look like a perfect single-source scene.
    """
    r = np.asarray([x for x in ranges_km if x is not None and np.isfinite(x)],
                   dtype=np.float64)
    if r.size == 0:
        return float("nan")
    if np.any(r <= 0):
        raise ValueError(
            "a detection at range 0 km means it sits on the hydrophone; w = r**(-k/10) "
            "is infinite there. Check the hydrophone coordinate before going further.")
    w = r ** (-k / 10.0)
    return float(w.max() / w.sum())


def classify_scene(records: Sequence[dict], scene_id: str, acq: str,
                   k: float = NOMINAL_SPREADING_K,
                   dominance_min: float = DOMINANCE_MIN,
                   verified: bool = False) -> dict:
    """One scenes.csv row: EMPTY / SINGLE-DOMINANT / CROWDED.

    PASS EVERY DETECTION, OF EVERY SIZE CLASS. This function answers "is this
    scene clean?", and a 40 m vessel at 1 km makes a scene emphatically not clean
    even though it is out of scope for the small-craft fit. Filter to
    TARGET_SIZE_CLASSES after this, not before: a scene screened on small craft
    alone classifies as SINGLE-DOMINANT while the hydrophone is actually
    listening to a ship, which is the plan's stated worst failure -- a loud near
    source recorded as a quiet distant one, dragging the whole fit.

    EMPTY scenes are DATA, not waste -- they measure the ambient noise floor,
    which has to be subtracted in the intensity domain before the SL/k fit, or
    distant boats come out mysteriously loud because at long range the hydrophone
    is mostly measuring the ocean (plan sections 1.2 and 10).
    """
    ranges = [r["range_km"] for r in records]
    if not records:
        return {"scene_id": scene_id, "acq_time_utc": acq, "n_detections": 0,
                "dominance": "", "class": CLASS_EMPTY, "r_dominant_km": "",
                "wake_flag_dominant": "", "verified": int(verified)}

    dom = dominance(ranges, k)
    i_dom = int(np.argmin(ranges))  # smallest range == largest w for any k > 0
    return {
        "scene_id": scene_id,
        "acq_time_utc": acq,
        "n_detections": len(records),
        "dominance": round(dom, 3),
        "class": CLASS_SINGLE if dom >= dominance_min else CLASS_CROWDED,
        "r_dominant_km": records[i_dom]["range_km"],
        "wake_flag_dominant": records[i_dom]["wake_flag"],
        "verified": int(verified),
    }


# ==========================================================================
# 12. OUTPUTS
# ==========================================================================


def write_csv(path, rows: Sequence[dict], fields: Sequence[str]) -> int:
    """Write rows to `path` with exactly `fields`, erroring on any mismatch.

    A DictWriter with extrasaction='ignore' would silently drop a column that a
    downstream join needs, so the default (raise) is kept and missing keys are
    caught here with a message that names them.
    """
    for i, r in enumerate(rows):
        missing = set(fields) - set(r)
        extra = set(r) - set(fields)
        if missing or extra:
            raise ValueError(f"row {i} does not match the schema: "
                             f"missing={sorted(missing)} unexpected={sorted(extra)}")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def to_i3_records(detections: Sequence[dict],
                  overpass_id_by_scene: dict[str, str] | None = None,
                  res_m: float = 3.0) -> list[dict]:
    """detections.csv rows -> the acoustics I3 view (acoustics_plan_v2 section 6).

    Two contracts were written independently and do not match: Option C section 10
    specifies `detections.csv` keyed on `scene_id`, while acoustics I3 specifies
    `data/interim/detections.parquet` keyed on `overpass_id` with `length_px`,
    `length_m_est` and `size_class`. Rather than pick a winner, derive one from
    the other -- there is then a single detector output and two views of it, and
    they cannot drift apart.

    `overpass_id` is Malachy's key (interface I1, `overpasses.csv`). Until that
    file exists, `scene_id` is passed through unchanged, which is honest: it says
    the join key has not been agreed rather than inventing an id that will not
    match I1 later.
    """
    out = []
    for i, d in enumerate(detections):
        scene = d["scene_id"]
        length_m = float(d["length_class_m"])
        out.append({
            "overpass_id": (overpass_id_by_scene or {}).get(scene, scene),
            "detection_id": f"{scene}:{i:04d}",
            "lat": d["lat"],
            "lon": d["lon"],
            "length_px": round(length_m / max(res_m, 1e-6), 2),
            "length_m_est": length_m,
            "size_class": size_class(length_m),
            "confidence": d["confidence"],
        })
    return out


# ==========================================================================
# 13. SYNTHETIC SCENE -- the optical analogue of invariant 3's synthetic tone
# ==========================================================================


@dataclass
class SyntheticScene:
    """A scene with known answers, for proving the pipeline before trusting it.

    Invariant 3 requires the acoustic pipeline to be proven against a tone of
    known level, frequency and time before anything depends on it. This is the
    same idea for the optical side: vessels of KNOWN length at KNOWN lon/lat, so
    "the detector found nothing" and "the detector is broken" can be told apart
    (invariant 9) with no imagery at all.
    """
    green: np.ndarray
    nir: np.ndarray
    rgb8: np.ndarray
    valid: np.ndarray
    res_m: float
    truth: list[dict]                      # lon, lat, length_m, range_km
    to_lonlat: ToLonLat = field(repr=False, default=None)


def make_synthetic_scene(n_px: int = 600, res_m: float = 3.0,
                         vessels: Sequence[tuple[float, float, float]] = (),
                         centre_lonlat: Sequence[float] | None = None,
                         water_nir: float = 0.02, land_nir: float = 0.30,
                         noise: float = 0.004, land_rows: int = 60,
                         seed: int = 20260827,
                         heavy_tail_df: float | None = None) -> SyntheticScene:
    """Build a square scene: dark water, a bright land strip, and known vessels.

    `vessels` are (offset_east_m, offset_north_m, length_m) about the centre. The
    land strip is at the top of the image and exists to make the water mask and
    its erosion do real work -- a synthetic scene of pure water would pass a
    shoreline test that the real Broken Group would fail.

    `heavy_tail_df` EXISTS BECAUSE THE GAUSSIAN DEFAULT ONCE HID A REAL BUG. The
    N-sigma threshold self-test passed on Gaussian water noise while the same
    threshold over-predicted by three orders of magnitude on real scenes, for the
    one reason the fixture could not express: ocean NIR has a heavy tail and
    Gaussian noise does not. A synthetic fixture validates the IMPLEMENTATION,
    never the distributional assumption underneath it -- so the assumption has to
    become a fixture PARAMETER before it can be tested at all.

    Set it to a Student-t degrees of freedom (3 is heavy, 30 is nearly Gaussian)
    and the water noise is drawn from that instead, RESCALED TO THE SAME sigma_MAD
    as the Gaussian case. Same robust scale, different tail -- which is exactly the
    blind spot, because sigma_MAD measures the core while the false alarms come
    from the tail. The negative tail is CLIPPED at zero rather than folded with
    abs(): folding would manufacture bright pixels out of dark ones, a different
    artefact from the one being modelled.
    """
    from pyproj import Transformer

    rng = np.random.default_rng(seed)
    centre = tuple(HYDROPHONE_LONLAT if centre_lonlat is None else centre_lonlat)

    # Local azimuthal-equidistant projection about the centre: metres offsets are
    # exact by construction, so the truth ranges are exact and any error found by
    # the geometry self-test belongs to the code under test, not the fixture.
    proj = f"+proj=aeqd +lat_0={centre[1]} +lon_0={centre[0]} +datum=WGS84 +units=m"
    to_ll = Transformer.from_crs(proj, "EPSG:4326", always_xy=True)

    half = n_px * res_m / 2.0

    def to_lonlat(cols, rows):
        cols = np.asarray(cols, dtype=np.float64)
        rows = np.asarray(rows, dtype=np.float64)
        east = (cols + 0.5) * res_m - half        # +0.5: pixel centre, not corner
        north = half - (rows + 0.5) * res_m       # image rows increase southward
        lon, lat = to_ll.transform(east, north)
        return np.asarray(lon), np.asarray(lat)

    if heavy_tail_df is None:
        water_noise = np.abs(rng.normal(water_nir, noise, (n_px, n_px))) - water_nir
    else:
        t = rng.standard_t(heavy_tail_df, (n_px, n_px))
        mad_sigma = MAD_TO_SIGMA * float(np.median(np.abs(t - np.median(t))))
        if mad_sigma == 0.0:
            raise ValueError(
                f"heavy_tail_df={heavy_tail_df} produced a degenerate draw with "
                "zero MAD; there is no scale to match the Gaussian case against.")
        water_noise = t * (noise / mad_sigma)
    nir = np.clip(water_nir + water_noise, 0.0, None).astype(np.float32)
    green = np.abs(rng.normal(0.06, noise, (n_px, n_px))).astype(np.float32)
    nir[:land_rows] = land_nir + rng.normal(0, noise, (land_rows, n_px))
    green[:land_rows] = 0.12 + rng.normal(0, noise, (land_rows, n_px))

    truth = []
    for east_m, north_m, length_m in vessels:
        col = (east_m + half) / res_m - 0.5
        row = (half - north_m) / res_m - 0.5
        n_long = max(1, int(round(length_m / res_m)))
        n_short = max(1, int(round(0.35 * length_m / res_m)))
        r0, c0 = int(round(row - n_short / 2)), int(round(col - n_long / 2))
        nir[r0:r0 + n_short, c0:c0 + n_long] = 0.22      # hulls are bright in NIR
        green[r0:r0 + n_short, c0:c0 + n_long] = 0.20
        lon, lat = to_lonlat([col], [row])
        truth.append({"lon": float(lon[0]), "lat": float(lat[0]),
                      "length_m": float(n_long * res_m),
                      "range_km": math.hypot(east_m, north_m) / 1000.0,
                      "row": float(row), "col": float(col)})

    rgb8 = np.clip(np.stack([green * 0.9, green, green * 1.1], -1)
                   * S2_QUANT / TCI_DIVISOR, 0, 255).astype(np.uint8)
    valid = np.ones((n_px, n_px), dtype=bool)
    return SyntheticScene(green=green, nir=nir, rgb8=rgb8, valid=valid, res_m=res_m,
                          truth=truth, to_lonlat=to_lonlat)
