# 0016. The small-craft scope is FAO classes 0-12 m and 12-24 m, applied at analysis time

Status: accepted
Date: 2026-08-27

## Context

`vessel_detect.py` shipped with `--max-length 20`, a metres threshold that dropped any detection
longer than 20 m at detection time. `option_c_detection_plan.md` §6.6 adopted it as "the
small-vessel cut". It was inherited, not derived -- no document says where 20 came from.

Two things break because of it.

**It censors an FAO class invisibly.** CLAUDE.md invariant 6 and source-of-truth goal 2 name the
FAO length-overall classes 0-12 m / 12-24 m / >24 m as the project's size convention. 20 m is not
one of their edges: it sits *inside* [12, 24). So an 18 m vessel is kept and a 22 m vessel is
dropped while both report `12-24m`. Every count and every recall-by-size-class number for that bin
is then computed on a truncated subset, biased low by an unknown fraction, with nothing in the
output saying so. Interface I3 hands `size_class` to the acoustics side, so the bias propagates
into the detectability curve (G3) rather than staying local to the optical notebook.

**It breaks the cleanliness guarantee, which is worse.** Option C classifies a scene
SINGLE-DOMINANT when one vessel accounts for >= 80% of expected received intensity, and the whole
single-source acoustic fit rests on that. A 20 m cut applied *before* classification deletes a
39 m vessel at 1 km from the scene, and the scene then passes as clean on a small craft that
vessel is acoustically swamping. The plan names this as its worst failure mode in §1: "Missing a
second, closer boat is severe -- the scene gets recorded as 'quiet distant source' when the
hydrophone actually heard a loud near one. A single such error can drag the whole fit." The size
cut was manufacturing exactly that error.

The root cause is that one constant was doing two unrelated jobs: *which vessels the science is
about*, and *which bright blobs are not vessels at all*.

## Decision

**1. The small-craft scope is a set of FAO classes, not a length.**

```python
TARGET_SIZE_CLASSES = ("0-12m", "12-24m")     # boatphone/optical.py
```

Expressed as classes, it lands on FAO edges by construction and cannot bisect one.

**2. `12-24 m` is IN scope.** The project targets vessels without AIS, and AIS carriage is set by
tonnage and service, not by length -- Canadian requirements bind at 300 GT international / 500 GT
domestic, plus passenger vessels, none of which maps to a length edge. A 15 m private motor yacht
is as AIS-dark as a 6 m runabout. Excluding 12-24 m would therefore cut the sample on a criterion
that does not track the property the project cares about, and would roughly halve an already small
n. Where AIS carriage matters to a claim, state it as a caveat on that claim rather than encoding
it in the detector.

**3. Scope is applied at ANALYSIS time, never at detection time.** `select_size_classes()` is
called when choosing what enters the SL/k fit. `classify_scene()` receives *every* detection of
*every* size class, because it answers "is this scene clean?" and a 39 m ship makes a scene
emphatically not clean whether or not it is in scope for the fit. The function carries a warning
saying so.

**4. The old constant's other job becomes an explicit implausibility bound.**

```python
MAX_PLAUSIBLE_VESSEL_M = 150.0
MIN_PLAUSIBLE_VESSEL_M = 0.0
```

These reject features that are not vessels -- wakes, slicks, shoreline bleed. Set generously,
because Barkley Sound carries real large traffic (the Bamfield ferry is ~38 m) and those vessels
must be detected. It is a backstop behind `BLOB_MAX_ASPECT` and `BLOB_MIN_FILL`, which do the
actual wake rejection; a wake long enough to fail this bound has already failed those.

**5. Report `length_m_est` alongside `size_class`, always.** At 3 m GSD the measurement
uncertainty is +-1 px = +-3 m, so a true 12 m vessel measures 9-15 m and the class edges are
themselves noisy -- more so than the ~6 m labeller-confidence floor. I3 already asks for both
fields; the requirement here is that figures and tables do not reduce to the class alone.

**6. `vessel_detect.py --max-length 20` is superseded.** Its default is corrected in place. That
script remains useful as the Detector A CLI and for the tile/`imgsz` sweep, but anything
schema-bearing goes through `boatphone/optical.py`.

## Consequences

- **A latent wrong answer is removed, not a feature added.** The scene-cleanliness bug was silent:
  it produced a plausible SINGLE-DOMINANT scene set and a plausible fit. `scripts/optical_selftest.py`
  now pins it as a regression -- same pixels, dominance 0.338 with all sizes present against 0.372
  when scoped first, and 8 detections against 7.
- **More detections survive to `detections.csv`,** including vessels the fit will not use. That is
  intended: `scenes.csv` needs them, and the `>24m` count per scene is itself reportable.
- **The 12-24 m bin is now complete,** so recall-by-size-class is defensible for it. Previously it
  was quietly truncated at 20 m.
- **`MAX_PLAUSIBLE_VESSEL_M = 150` is a guess and is labelled as one.** Sanity-check it against the
  largest vessel that actually transits the AOI before quoting any rejection count. If real
  detections cluster against it, that is evidence the bound is wrong, not that the water is busy.
- **How to tell this was wrong:** if the `>24m` class turns out to be empty across all 30 scenes,
  the scope question was moot and the extra machinery bought nothing. If `12-24m` detections
  dominate the fit and behave differently from `0-12m` in the SL/k residuals, they are a distinct
  source population and should be split rather than pooled -- report the residuals by class.

## Related

A separate defect found while testing this: `cv2.minAreaRect` measures corner-to-corner between
pixel *centres*, so every blob length was short by exactly 1 px -- 3 m at PlanetScope GSD, 50% of a
2 px hull, and enough to move vessels across the 12 m edge in the wrong direction. Corrected in
`detect_nir_blobs`. On the self-test fixture the correction alone moved the class tally from
`{0-12m: 6, 12-24m: 1}` to `{0-12m: 5, 12-24m: 2}`, which is the bias made visible.
