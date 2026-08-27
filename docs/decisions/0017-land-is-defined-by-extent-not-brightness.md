# 0017. Land is defined by extent, not by brightness, and NDWI is a diagnostic

Status: accepted
Date: 2026-08-27

## Context

The water mask decides what Detector B is allowed to look at. Two successive versions of it
deleted the detector's own targets, in two different ways, and neither raised an error. Both were
found only because a real scene was run and the numbers were printed.

**Failure 1 -- NDWI.** The original mask was `NDWI >= 0` per the textbook. On the first delivered
PlanetScope scene (`20200730_192942_71_1059`) water reads green 0.0173 / NIR 0.0157, so
NDWI ~= -0.003 -- sitting exactly on the threshold. It admitted 49% of the water and rejected the
rest, splitting the water's own noise in half. There was nothing for it to segment: the AOI is
**98% open water**, 1.75 km2 of land in 100 km2. NDWI assumes a land mode exists; this AOI has
almost none.

Separately, a vessel is bright in NIR, so a vessel has a *negative* NDWI and punched a
vessel-shaped hole in the mask meant to contain it. That was patched with a morphological closing
sized to a vessel beam -- a fix for a symptom.

**Failure 2 -- a brightness cut.** Replacing NDWI with `nir > 0.05` for land looked obviously
correct and was worse. On the same scene it classified the **two brightest objects in the whole
scene** as land and removed them, so Detector B could not recover them at any threshold from N=4
to N=10. It recovered only the two *faintest* Detector A hits -- exactly backwards.

The common root cause: **a bright vessel is what Detector B exists to find, so any rule of the
form "bright means not water" deletes the targets.** The brighter and more obvious the vessel, the
more certainly it is removed. Both failures are silent -- an empty result from a real scene is
indistinguishable from a scene with no boats in it, which is CLAUDE.md invariant 9.

## Decision

**1. Land is a LARGE connected region of NIR-bright pixels.** `boatphone/optical.py:land_mask()`
thresholds NIR, takes connected components, and keeps a component as land only if its area is at
least `MIN_LAND_AREA_M2 = 20_000` (2 ha). A 24 m vessel is ~200 m2, a hundred times smaller.
**Size is what separates an island from a boat; brightness is not.**

**2. The mask is built by SUBTRACTING land, not by selecting water.**

```
water = valid AND NOT dilate(land, WATER_ERODE_M) AND clear
```

Anything bright and small therefore stays in the mask *by construction*, rather than being
selected out and patched back. This supersedes the hole-fill closing: with land defined by extent
there is no vessel-shaped hole to fill. Two fixes for one bug was one too many; the closing is
removed and only its explanation survives as a comment.

**3. NDWI is a diagnostic, never the mask.** `ndwi_land_fraction()` reports what NDWI *would* have
called land, to be read against what the extent rule actually found. On this scene that is **51%
against 2%** -- a disagreement that would otherwise have been invisible.

**4. `MaskReport` names the candidate population.** It carries `land_px`, `land_components` and
`small_bright_components`, so a mask change that eats the vessels shows up in one printed line
rather than as a quiet zero downstream.

## Consequences

- On `20200730_192942_71_1059`: **8** of 335 bright-NIR components are large enough to be land.
  The other **327** are the candidate vessel population, and the brightness cut was discarding all
  of them. All four Detector A hits are now inside the mask (previously two), and Detector B
  recovers three of them at every N from 6 to 12.
- **Sun glint is the known weakness.** A large enough glint patch exceeds 2 ha and would be called
  land, removing real water from the mask. `land_components` and `land_px` are printed so this is
  visible; if land area jumps between scenes of the same AOI, suspect glint before suspecting
  geography.
- **`MIN_LAND_AREA_M2 = 20_000` is a judgement, not a measurement.** It sits between the largest
  plausible vessel (~150 m x 25 m = 3,750 m2) and the smallest islet worth calling land. Islets
  below 2 ha will appear as detection candidates and must be cleared by the human pass -- that is
  a deliberate bias toward recall, consistent with Detector B's job of certifying scenes clean.
- **How to tell this was wrong:** if the eyeball pass finds that most Detector B candidates are
  small islets and rocks rather than glint or vessels, the threshold is too high for this
  archipelago and should come down, with a shoreline vector used to separate the two instead.
- Three regressions are pinned in `scripts/optical_selftest.py`: vessels must be counted as
  bright-but-too-small rather than land; disabling the extent rule must delete every planted
  vessel; and **tripling a vessel's brightness must not turn it into land** -- the property extent
  guarantees and thresholding cannot.
