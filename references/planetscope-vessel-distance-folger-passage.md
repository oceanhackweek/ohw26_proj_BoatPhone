# Vessel Detection and Kinematic Estimation from PlanetScope Imagery

**Folger Passage, Barkley Sound, British Columbia — three-day technical scope**

---

## Part I — Feasibility of a Machine Learning Vessel Detector

Short answer: **yes, but not the way the question is usually framed.** A working vessel detection pipeline is very achievable in three days. Training a deep object detector from scratch is not, and the reason isn't time — it's that two independent constraints make the standard approach structurally wrong for this problem.

### Constraint 1: the targets are 2–10 pixels

This is the thing that invalidates most off-the-shelf advice. At 3 m/pixel:

| Vessel | Length | Pixels |
|---|---|---|
| Sport fishing boat | 5–8 m | 2–3 |
| Salmon troller / gillnetter | 10–15 m | 3–5 |
| *Frances Barkley* | 62 m | ~20 |

YOLO, Faster R-CNN, and DETR variants downsample by 8–32× in the backbone before the detection head. A 3-pixel vessel disappears entirely by the first or second stride. It can be forced — upsample 4×, add a P2 head — but that costs a day of fighting the architecture for worse results than a simpler method gives in an hour.

More usefully: **for moving vessels, the wake is the detectable object, not the hull.** A wake runs 50–200 m, which is 17–67 pixels — an order of magnitude larger than the boat. The pipeline should target the wake signature for underway vessels and hull contrast for stationary ones. These are two different detection problems, and conflating them is a common failure.

### Constraint 2: GFW Sentinel-2 detections will not work as training labels

The blocker is temporal offset.

Sentinel-2 crosses at roughly 10:30 local; the PlanetScope flock spreads across a range of morning crossing times. Realistic offsets are tens of minutes. A vessel at 8 knots moves 4.1 m/s, so:

- **60 seconds offset** → 247 m → **82 PlanetScope pixels** of displacement
- **20 minutes offset** → ~5 km

A label that lands 82 pixels from a 3-pixel object is not a label. Add geolocation error on both sensors — combined roughly 15–20 m, or 5–7 pixels — and even perfectly simultaneous acquisition would need a search-and-recentre step.

The partial exception is stationary vessels. GFW's product includes speed estimates, so near-zero-speed detections could have their positions transferred. But that yields a training set composed entirely of stationary boats, which will not generalise to the wake-bearing moving vessels that dominate the imagery. Volume is fatal in any case: Folger Passage is a small area, and the archive will likely hold a handful to a few dozen S2 detections in total — not the thousands a trained detector needs.

**Use GFW Sentinel-2 as the validation reference instead.** That is where it is genuinely valuable: an independent, published, ML-derived detection set with length and speed attributes that the team did not label itself. Comparing PlanetScope detections against it on near-coincident dates is a stronger result than any model trained on it.

Labels must come from hand-annotation. Four people for two focused hours yields several hundred to a couple thousand chips — enough for the recommended approach.

### The technique that solves most of the problem for free

Before any ML: **build a temporal median background from the PlanetScope stack.** Compute the per-pixel median across N dates. Rocks, land, permanent kelp beds, and moored infrastructure appear in the median. Vessels, being transient, do not. Subtract, and most false positives vanish without a single label.

In Folger Passage this matters more than usual. The site is at the exposed mouth of Barkley Sound, and the false positive sources are severe: bull kelp canopy around Folger Island, Wizard Islet, and the Deer Group; whitecaps in any wind; drying reefs such as Swale Rock; sun glint; and large quantities of drift logs, which at 3 m are close to indistinguishable from a small boat without a wake.

Background subtraction removes the static sources. The residual ML problem is discriminating vessels from **kelp, whitecaps, and logs** — all transient, all bright. That is a tractable, well-posed classification task, and it is where the three days should go.

### Architecture recommendation

**Stage 1 — candidate generation, no ML.** Background subtraction, then anomaly thresholding on NIR and a floating-object index, connected components, size and eccentricity filters. High recall, low precision by design. Half a day.

**Stage 2 — false positive rejection. This is the ML contribution.** Extract 16×16 or 32×32 chips around every candidate, then classify. With under ~1,000 labels, **gradient-boosted trees on engineered features will likely beat a small CNN**, and train in seconds rather than minutes. Features: all eight band values and key ratios, blob shape descriptors, local water variance, distance to shore, and a wake-orientation term.

**Stretch, only if labels exceed ~1,500.** A small U-Net doing full-resolution binary segmentation, or a CenterNet-style heatmap regressor predicting Gaussian peaks at vessel centres. Both handle tiny objects far better than box-based detectors. Do not start here.

**Known but probably not worth the time:** the Airbus Ship Detection dataset (~1.5 m optical) and DOTA's ship class could pretrain a backbone if downsampled to simulate 3 m. Legitimate approach, real domain gap, roughly a full day of the three. Skip.

---

## Part II — Adding Speed, Orientation, and Range to the Hydrophone

Three requests, three very different levels of cost. One is free, one reinforces the wake-first framing above, and one is a genuine new subsystem — but that third one changes the project's shape in a way that is worth the trouble, because it rescues the acoustic module from being a weak correlation.

### Distance: free, with one decision to get right

Once a detection has a geographic centroid, range to the node is a geodesic calculation. No pipeline change.

The one thing that is not free is **which node**. Folger Passage has instrumentation at more than one site and depth — a shallow pinnacle installation and a deeper one — and they sit in different transmission-loss regimes. Pull exact platform coordinates and depth from the Oceans 3.0 metadata rather than using a nominal "Folger Passage" position, and commit to one hydrophone for the whole analysis.

### Orientation: aligns with, and confirms, the wake-first approach

Hull orientation from a 3–5 pixel blob is not recoverable. The second-moment major axis of a 3-pixel object is dominated by pixel quantisation — reporting it would be noise with false precision. For the *Frances Barkley* at ~20 pixels it is fine, but she is not the population of interest.

Wake axis is different: 17–67 pixels, strongly linear, and robustly fittable by PCA or a Hough transform on the background-subtracted residual. The Kelvin wake's V-shape also resolves the 180° heading ambiguity for free — the vessel sits at the apex, so the apex indicates heading.

**This creates a selection effect that must be stated up front:** orientation exists only for vessels under way. A drifting boat with gear in the water has no wake and no recoverable heading. The kinematic sample is therefore a biased subset of the detection sample, and the bias correlates with exactly the behaviour the project would want to distinguish.

### Speed: the real departure

Two independent methods. Run both, because they cross-validate.

**Method A — band parallax.** SuperDove's filter strips image each band at a slightly different moment — roughly 800 ms between green and blue — so a moving vessel is displaced between bands while static terrain is not. Measure per-band sub-pixel centroids, divide displacement by the known Δt, and the result is a full velocity vector: speed *and* direction, including heading for vessels with no visible wake.

The catch is magnitude. At 8 knots (4.1 m/s), 800 ms gives 3.3 m — about **1.1 pixels**. That is sub-pixel, recoverable only with intensity-weighted centroiding on a bright compact target. At 20 knots it is ~2.7 pixels and comfortable. The method therefore works well for transiting vessels and marginally at trolling speeds. Use the full eight-band sweep rather than green-to-blue alone to lengthen the time baseline.

Two practical warnings:

1. Orthorectification aligns the *ground*, so moving-object displacement should survive in the analytic product — but verify this empirically before trusting it.
2. Planet may not publish per-scene inter-band timing. If not, **calibrate Δt empirically against a vessel of known speed.** The *Frances Barkley* runs a scheduled route at a known service speed and carries AIS. This self-calibration turns a potential blocker into a half-hour task.

**Method B — transverse wake wavelength.** Independent physics, and a useful check. Note that the Kelvin half-angle is ~19.5° regardless of speed, so the wake's opening angle carries no speed information — a common trap. What does carry speed is transverse wave spacing, λ = 2πV²/g. At 5 m/s that is 16 m (~5 px, marginal); at 10 m/s it is 64 m (~21 px, clean). This method also favours fast vessels, but through entirely different error sources than band parallax, which is what makes agreement between them meaningful.

### What changes in the pipeline

**A third stage.** Part I ended at candidate generation → chip classification, which outputs presence/absence. A 16×16 chip classifier discards precisely the spatial structure kinematics requires. Add a kinematic characterisation stage operating on the full-resolution multi-band residual, downstream of classification.

**A different label protocol.** Wake presence flags are now needed, plus manually digitised wake axes on a validation subset — orientation cannot be validated without reference orientations. Budget more of Day 2 for labelling.

**Product-level care.** Per-band radiometry must be preserved and acquisition timestamps must be precise. Confirm this before ordering, not after.

### Why this strengthens the acoustic module

Without kinematics, the acoustic question is "does vessel count correlate with sound level?" — a weak, confounded result resting on perhaps a dozen usable satellite passes. With speed and range per vessel, a physical model becomes available:

> **RL ≈ SL(V) − TL(R)**

Small-craft source level scales steeply with speed, and transmission loss in shallow water goes roughly as 15–20·log₁₀(R). With multiple vessels in a scene, predict the energy sum, 10·log₁₀(Σ10^(RLᵢ/10)) — which a single omnidirectional hydrophone cannot decompose but *can* verify in aggregate.

That converts a correlation into a **prediction with an interrogable residual**, and residuals are where the interesting findings live. Vessels the model over-predicts may be quieter hull types; under-predicted scenes may contain vessels that were missed entirely, which is itself an independent estimate of detection recall.

**One design consequence to act on immediately:** a small vessel is probably only detectable above ambient out to a few kilometres in this environment. The acoustically relevant AOI is therefore a circle of roughly 5 km radius around the node — under 80 km². The 3,000 km² quota becomes non-binding, allowing many dates over a small footprint, which is exactly what both the temporal median background *and* the acoustic regression need. The requirement that looked like added scope actually resolves the quota tension.

---

## Part III — Three-Day Schedule

**Day 1, morning — go/no-go checks.** Two queries decide whether the project survives.

1. Query GFW's Sentinel-2 detections over Folger Passage and count them. Fewer than about ten means the validation plan needs rethinking now, not Thursday.
2. Run the Planet archive search for cloud-free scenes. The exposed WCVI coast is fog-prone, and enough dates are needed to build a stable temporal median. Below roughly 8–10 clear scenes, widen the AOI to Trevor Channel and Imperial Eagle Channel.

**Day 1, afternoon.** Order and clip imagery. Co-register the stack. Build the land and intertidal mask. Compute the temporal median. Confirm ONC node coordinates and depth.

**Day 2, morning — labelling, everyone, same room, same rules.** Agree edge cases first: is a kayak a vessel, is a log boom one target or many. Include wake presence flags. Have two people independently label an overlapping subset so inter-annotator agreement can be reported — that number will be one of the more honest things in the presentation.

**Day 2, afternoon.** Candidate generator, feature extraction, train the classifier. Cross-validate by *scene*, not by chip — chips from one image are not independent, and random splits will badly inflate accuracy.

**Day 3, morning.** Kinematic stage: band-parallax velocity, wake-axis orientation, range to node. Calibrate Δt against the *Frances Barkley*. Validate against GFW Sentinel-2 detections on the closest-coincident date. Expect disagreement and interpret it rather than explaining it away.

**Day 3, afternoon.** Acoustic regression, figures, error analysis on failure cases, presentation.

---

## Part IV — Scope Decisions

### What to cut

- **The CNN vs. gradient-boosting comparison.** A nice-to-have. Kinematics is better science and needs the day.
- **Absolute speed accuracy as a success criterion.** Report speed as a **class** — stationary, slow (under ~5 knots), transiting — validated against the ferry and any AIS coincidences. Three well-separated classes that can be defended beat a continuous estimate with uncharacterised error, and the acoustic model works fine on classes.

### What to promise on Friday

Not "we built a vessel detector." Promise a precision-recall curve for small-vessel detection at 3 m in a high-clutter coastal environment, with a characterised false positive taxonomy, plus a kinematically-informed acoustic prediction and its residuals. That is the defensible result — and unlike a trained detector, it cannot fail to exist by Friday afternoon.

### Fallback

If Folger Passage proves too quiet, the ONC hydrophone record in the same water supports a thin acoustic cross-check as a standalone contribution if the imagery disappoints.

---

## Summary of Key Numbers

| Quantity | Value | Implication |
|---|---|---|
| Pixel size | 3 m | Small vessels are 2–5 px |
| Wake length | 50–200 m | 17–67 px — the actual detectable object |
| Inter-band Δt (green–blue) | ~800 ms | Velocity estimation baseline |
| Displacement at 8 kn over 800 ms | 3.3 m (~1.1 px) | Sub-pixel; needs weighted centroiding |
| Displacement at 20 kn over 800 ms | 8.2 m (~2.7 px) | Comfortably resolvable |
| S2 / PlanetScope offset at 60 s, 8 kn | 247 m (82 px) | Why S2 cannot supply training labels |
| Kelvin half-angle | 19.5°, speed-invariant | Carries no speed information |
| Transverse wavelength at 10 m/s | 64 m (~21 px) | Independent speed estimator |
| Transmission loss exponent | 15–20·log₁₀(R) | Shallow-water acoustic model |
| Acoustically relevant AOI | ~5 km radius, <80 km² | Quota becomes non-binding |
