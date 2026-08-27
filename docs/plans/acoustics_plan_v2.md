# BoatPhone -- Acoustics Workstream Plan v2

**Owner:** Isaac Guld
**Supersedes:** `accoutics_plan.md` (v1)
**Parent plan:** `docs/plans/proposed_plan_IG.md` (rev. 3)
**Sibling plans:** PlanetScope acquisition (Malachy), YOLO vessel detection (Neve)
**Day 1 = Thu 2026-08-27**

> v1 stays in the repo for its evidence sections only -- the 250 Hz bin-mapping derivation and the
> `.fft.gz` anomaly characterisation, which §3 below cites. Where the two plans disagree on
> anything else, **v2 wins**.

---

## 1. What this workstream is for

`Project_Source_of_Truth.txt` gives Isaac "hydrophone data and acoustic modelling." Four project
goals depend on it. Each row states what *done* means, so progress is checkable rather than
narrative.

| # | Source-of-truth goal | Done means | Priority |
|---|---|---|---|
| **G1** | Continuous estimate of vessel presence / count | A per-bin vessel-presence score across the covered archive, validated against optical labels, with a stated false-positive rate | **Must** |
| **G2** | Distinguish small (non-AIS) from large vessels | A measured statement about which size classes are separable acoustically -- **including "they are not"**, which is a valid result | **Must** |
| **G3** | Useful acoustic detection range | P(detect) vs range vs size class, with the optical label-noise floor carried through | **Must** |
| **G4** | Speed from pitch / frequency | Any measured statement at all | **Stretch** |

**The thing to avoid:** a clean vessel-to-energy correlation is exactly what a time-alignment bug
also produces (CLAUDE.md invariant 4). Every headline number below is paired with a null check.

---

## 2. Status

### Complete

- **A0 -- environment, credentials, deterministic gate.** ONC token loading with redaction,
  `boatphone/paths.py`, `.gitignore` hardening, and `scripts/checks.py` as the check harness
  (there is no pytest in this environment). Merged at `0a9fdfc`.
- **A1 -- uptime calendar.** `boatphone/onc_client.py` (paging-complete ONC archive listing,
  deployment assignment, gap summarisation), `boatphone/config.py`,
  `scripts/build_uptime_calendar.py`, and `docs/derived/hydrophone_gaps.md` -- **deliverable O1,
  shipped to Malachy**. Decisions 0007 and 0008. Full seasonal scan 2020-2026.
- **B0 -- model viability gate. COMPLETE, verdict NO-GO.** Ran end to end (`e1c1544`), then an
  adversarial recheck upheld the verdict and corrected one finding. No CNN baseline exists; the
  released checkpoint is a single-logit binary detector with no distinct Engine Noise output; zero
  Engine Noise rows exist at `ICLISTENHF1266` in the eval set; the CPU forward pass OOMs under a
  3.90 GB cgroup cap (not a 30 GB host limit as first framed); scoring the corpus once would cost
  ~12 CPU-days independent of RAM. Full detail: `docs/decisions/0012-b0-model-viability-outcome.md`.
  §4, §5 and §7 below are rewritten accordingly; B5 is now the primary detector.
- **B0-2a -- `boatphone/fft_io.py` landed** (`47237d1`). Shared `.fft.gz` reader:
  `read_fft_gz`, `frequency_axis_hz`, `time_axis_utc_s`, `calibrated_band_hz`,
  `assert_calibratable`, `assert_tone_at`, `structural_zero_report`. Start time is read from the
  **filename** (`onc_client.parse_file_coverage`) -- the decompressed payload carries no header, no
  timestamp and no sample rate at all. Null checks (tone ladder, frame-shuffle, alternate time
  bases) run and reported in the B0-2a-impl log. Two checks remain deliberately failing as
  data-dependent findings, now **adjudicated and closed** by B1a -- see §3.1-3.3. Both contracts
  were **restated to match measured reality**, not weakened: the top of the band is three regions
  (with a *harder* exact-zero assertion over cols 425-511), and the 38 kHz "line" is a hump asserted
  by power-excess centroid. The third question -- edge- vs centre-binning, +-125 Hz on every band
  edge -- is **still open** and is now *carried* in code rather than resolved (§3.2).

### The finding that reshapes everything else

**`ICLISTENHF1266`'s deployment ended 2026-03-14T21:37:20Z. The entire 2026 season has no
hydrophone data.** Today is 2026-08-27.

Consequences, which are project-wide and not acoustics-only:

1. **All matchups must use ARCHIVE PlanetScope imagery, 2020-2025 seasons.** New tasking produces
   scenes with no acoustic counterpart at all. **Tell Malachy before any order is placed** -- this
   is the same class of unrecoverable quota waste that O1 was built to prevent.
2. Usable in-season span is 2020-05-01 -> 2025-10-01, minus the four outages already enumerated in
   `hydrophone_gaps.md` (largest: 9.68 days, Aug-Sep 2023).
3. October-April was never scanned and is claimed neither way. If archive imagery turns out to be
   plentiful in the shoulder months, extending the scan is cheap and worth doing.

Note the availability figure is an **upper bound**: a bin counts as available if any listed file
overlaps it, and each file is modelled as 300 s of coverage, so dropouts shorter than ~10 min
cannot appear. B3's pull is expected to report lower uptime; that disagreement is this model, not
a bug.

### Dropped, and why

- **VTUAD and the transfer experiment (v1 §A8).** Verified during planning: VTUAD carries **no
  vessel size or length label**, so v1's headline size-stratified transfer gap cannot be computed
  at all; its audio is **uncalibrated** where ours is calibrated, so an absolute-level gap would be
  a calibration artefact indistinguishable from domain shift; and it sits behind a **paid IEEE
  DataPort Subscriber tier with no download API**, not the "overnight scripted download" v1
  assumed. Superseded by §4, which supplies a *device-matched* reference for free. Facts recorded
  in `docs/vtuad-facts.md`.
- **"Frozen PANNs/VGGish + linear head" as the model (v1 §A8).** **No public AIS-trained acoustic
  weights exist.** Domingos releases MIT training code but no checkpoints; Decrop et al. (JSTARS
  2025) release an open dataset but no code or weights; Renaud et al. (`arXiv:2607.13840`) is
  method-only. Any such model would therefore have been ours, which makes a poor transfer result
  ambiguous between "the method fails" and "we built it badly" -- the ambiguity invariant 9 exists
  to prevent.
- **v1's framing of >=250 Hz as a general limitation.** Small planing craft radiate peak energy at
  roughly 1-10 kHz (cavitation broadband); it is *large* ships that carry the diagnostic blade-rate
  tonals at 10-100 Hz. The floor costs us large-ship detection, **not** small-boat detection --
  and small craft are the target population. v1 half-stated this in §A5 and contradicted it
  elsewhere.

### Carried forward

v1's A2 (`.fft.gz` gate), A3 (calibration), A4 (bulk acquisition), A5 (features), A6 (ambient and
detectability) and A9 (matchups and split hygiene), renumbered below. `boatphone/models.py` from
the A8 worktree survives intact: its `assert_band_matched`, `band_limit` and `assert_comparable`
are 40/40 green and mutation-tested against eight deliberate sabotages.

---

## 3. Settled facts -- do not re-derive

| Fact | Source |
|---|---|
| **250 Hz per bin** (1024-pt FFT at 256 kHz, 512 bins over 0-128 kHz) | v1 §"What changed since rev. 3", **re-confirmed B1a** on three absolute measurements on the 128 kHz sample WAV (below) |
| **Echosounder source centre = 37,650 +- 150 Hz** (not 38.0 kHz) | B1a, measured **absolutely on the 128 kHz WAV** -- no product axis involved. `FFT_ECHOSOUNDER_ABS_CENTRE_HZ` |
| ~~Anti-alias shoulder onset at bin 408 confirms the mapping~~ | **STRUCK as mapping evidence, B1a** -- see below. Retained only as a structural check |
| Calibration file covers **10 Hz - 51.2 kHz only** = bins 0-205. Bins 206-417 are uncalibratable | the file's own header |
| Decidecade bands resolved only **above ~2.2 kHz**; no hybrid-millidecade compliance | 250 Hz bins; a decidecade band at *f* is ~0.23*f* wide |
| **~42 dB unexplained shape difference** across 250 Hz -> 2 kHz survives the remapping | v1 §A2 -- still open, gated in B1 |
| `.fft.gz` is **0.29 MB per 5-min file**; WAV for the same span is ~3.5 TB | v1 §A4 |
| ONC 400s of the "not deployed during range" form are **measured zeros** | decision 0007 |
| An empty listing over a deployed span is a **measured zero**, not a failure | decision 0008 |
| Study window **2020-02-18** (calibration validity) -> **2026-03-14** (deployment end) | calibration file; `hydrophone_gaps.md` |

### 3.1 What confirms 250 Hz/bin -- and what was struck (B1a, 2026-08-27)

**Struck: the anti-alias shoulder.** v1 counted the shoulder onset near bin 408
(`408 x 250 Hz = 102 kHz = 0.4 x 256 kHz`) as one of three independent confirmations of the
250 Hz/bin mapping. It is **information-free** for that purpose. The shoulder sits at `0.4 x fs`
and the bin width is `fs / 1024`, so the shoulder bin is `0.4 x 1024 = 409.6` **for any `fs`
whatsoever**. It is exactly as consistent with 125 Hz/bin as with 250 Hz/bin and never
discriminated between them. Recorded here rather than quietly deleted, because "three independent
confirmations" is the kind of claim that gets cited later: it was two. The shoulder survives as a
**structural** check only (`FFT_ROLLOFF_ONSET_BIN = 408`) -- it still catches a mis-strided read.

**Replaced by three absolute measurements.** The sample WAV
(`ICLISTENHF1266_20260313T000000.029Z_..._000500.029Z.wav`) is **128 kHz / 24-bit mono** -- it sees
0-64 kHz and states frequencies without reference to the product's axis at all. Against it:

| Measurement on the WAV | Predicted at 250 Hz/bin | Predicted at 125 Hz/bin | Result |
|---|---|---|---|
| The bins 140-162 feature | 35.0-40.5 kHz | 17.5-20.3 kHz | **+14 dB hump measured at 35-40.5 kHz**; <= 2.4 dB at 17.5-20.3 kHz |
| The bins 399/400 spur | 99.8-100.1 kHz (above WAV Nyquist) | 49.8-50.1 kHz | **Nothing at 49.8-50.1 kHz** (<= 0.11 dB) |
| The bins 322-329 feature | 80.5-82.3 kHz (above WAV Nyquist) | 40.2-41.2 kHz | **Nothing at 40.2-41.2 kHz** (<= 0.30 dB) |

250 Hz/bin is therefore confirmed on physics, not on arithmetic about the product.

**The 38 kHz landmark is restated.** "A line at bin 152 +- 1" was **nominal-derived**
(`38000 / 250 = 152`) and no statistic reproduces it on either fixture. The feature is a
**~5 kHz-wide hump over bins 140-162**, +14 dB over the band median, whose source measures
**37,634 Hz** by power-excess centroid on the WAV (stable to 2 Hz across the file's two halves) and
peaks in per-band temporal std at 37,672 Hz with 11.0 dB of variation -- the intermittency that
makes it an echosounder and not a resonance. Consistent with an ASL AZFP 38 kHz ping (a 300 us
pulse gives a ~3.3 kHz sinc mainlobe) or an EK80 sweep on an ES38-class transducer (nominal
34-45 kHz). "38 kHz" was always a nominal label. The check now asserts a **power-excess centroid**
in bins 149.0-152.0 (measured 150.36 and 150.17), never an argmax -- the argmax of a 5 kHz hump
quantised to integer counts flips between 149 and 150 across two files five minutes apart, while
the centroid moved 0.19 bins. Its job is to reject a **2x mapping error**, where it discriminates
by ~150 bins.

### 3.2 OPEN: centre-vs-edge binning -- +-125 Hz on every band edge

See `docs/decisions/0013-fft-axis-convention-is-an-open-assumption.md` for the full evidence, the
resolution order, and the residual weakness in how the uncertainty is enforced.

**Not settled, and deliberately not settled by fiat.** The reader stays pinned to bin **centres**
(`frequency_axis_hz()[1] == 250.0` exactly) because it must be deterministic, but that is now a
**named assumption** (`FFT_AXIS_CONVENTION = "centre"`), not a settled fact. B1a put it at roughly
60/40 toward ONC meaning bin **edges** (bin *k* spanning `[k*dF, (k+1)*dF)`), which would move
every named frequency up by half a bin.

- *For edge*: a narrow, reproducible spur straddles bins 399/400 with a power centroid of 399.47 /
  399.52 on two files five minutes apart -- a tone exactly halfway between two bin **centres**,
  which under the edge convention is the round 100.000 kHz. 512 bins of 250 Hz also tile
  `[0, 128000)` exactly under edge, and there demonstrably *is* a low-frequency filter (the 42 dB
  deficit above).
- *For centre*: a 1024-pt real FFT gives 513 bins (DC..Nyquist); a 512-column product is most
  naturally that with Nyquist dropped, `k = 0..511` at centres `k*dF` -- which makes column 0 the
  DC bin and explains its near-exact zero far more economically than a high-pass.
- *Not evidence*: the 100 kHz roundness is a **prior**, not a measurement. Of six narrowband lines
  found, the other five are round under neither convention, and there is no plausible 100 kHz
  source at Folger (AZFP: 38/67/125/200/455/769 kHz; ADCPs: 300/600/1200 kHz). The WAV cross-check
  is **circular until B1 pins counts to dB**: the implied offset swings from -0.10 to +0.89 bins as
  the assumed level scale moves from 0.25 to 3.0 dB/count, i.e. clean across both hypotheses.
- *Also not evidence*: the echosounder hump. Under edge it reads 37.70 kHz and under centre
  37.58 kHz, and which is closer flips with the background model and the dB scale.

**Consequence, carried not hidden:** `FFT_AXIS_OFFSET_UNCERTAINTY_HZ = 125.0` (one-sided, toward
higher frequency). **Every band-edge consumer widens its support by it** --
`models.band_limit(..., axis_offset_uncertainty_hz=...)`, supplied automatically by
`fft_io.band_limit_product`. `check_b0_2a_axis_uncertainty_is_carried` asserts both that the
constant is positive and that `band_limit` *actually widens*, so a later edit cannot silently drop
the open question. **No check may assert a bin position tighter than +-1 bin until this settles.**

B5 is **not blocked**, provided it carries the +-125 Hz and puts no band edge inside a narrow
feature. B6's calibration interpolation inherits <= 125 Hz of error: negligible above ~2 kHz where
the calibration curve is flat, non-negligible only in bins 1-4, which the 42 dB low-frequency
anomaly already gates.

**To resolve, in order:** (a) **ask ONC for the product definition** -- one sentence settles it,
free, and it is already B1 step 4; (b) a **scale-free two-bin-split census** over the B3 corpus --
histogram the sub-bin centroid fraction of every narrow (<= 2-bin) line; a cluster at half-integers
means edge, at integers means centre, and it uses only the two-bin ratio so no level scale is
needed; (c) once B1 pins counts to dB, re-run the WAV centroid comparison.

### 3.3 The top of the band is THREE regions, and the level scale is censored

See `docs/decisions/0014-two-spectral-ceilings-and-the-408-exclusion.md` for the two-ceiling
decision and the measured three-region split, and
`docs/decisions/0015-censoring-aware-thresholds-for-b5.md` for the censoring-aware requirement on
B5's thresholds and the "tune on overpasses once" constraint.

Measured over both fixtures (2,400 frames). The single "zeros at col 0 and 419-511" statement was
wrong at **both** ends.

| Region | Constant | Measured | Assertion |
|---|---|---|---|
| cols **425-511** | `FFT_STRUCTURAL_ZERO_COLS_HIGH` | 0 of 104,400 nonzero on **each** fixture | **HARD exact zero** -- any nonzero raises (reader/format failure) |
| cols **419-424** | `FFT_ROLLOFF_TAIL_COLS` | 68 / 74 of 7,200 nonzero, max 6 | bounded (max <= 10, mean <= 0.05) **and per-bin mean over bins 405-424 non-increasing** |
| col **0** | `FFT_DC_COL` | 14 / 8 of 1,200 nonzero, max 3 | bounded (max <= 5, nonzero fraction <= 0.02) -- **not** exactly zero |

The old `(419, 511)` was **off by six columns**: 419-424 is the tail of one continuous anti-alias
skirt whose per-bin mean runs smoothly from 9.8 at bin 405 through 0.28 at 418 to ~0.001 at 424.
There is no boundary at 419. The **monotonicity** assertion is the one that earns its keep -- a
mis-strided or wrapped row moves a bin mean by order 1, which a cell-count bound would not notice.
(One correction to the B1a ledger: the tail is *not* strictly monotonic. On fixture `...000004`
bin 423 has mean 0.0000 and bin 424 has mean 0.0008 -- one count in one frame out of 1200. At the
far end of the skirt the mean is quantised to multiples of 1/1200 and its ordering is integer
noise, so the assertion carries a tolerance of exactly one count in one frame,
`FFT_ROLLOFF_MONOTONIC_TOL_LEVEL`, three orders of magnitude below any real bug.)

**Two ceilings, kept apart in `config.py`:**

- `FFT_B5_CALIBRATED_CEILING_BIN = 205` (51.2 kHz) -- above it, no absolute dB re 1 uPa exists.
- `FFT_B5_RELATIVE_CEILING_BIN = 408` (~102 kHz) -- **nothing above bin 408 may enter a B5
  statistic**, even a relative one. Bins 409-424 are instrument response, not ocean; they are
  **floor-censored** (99.94% of cells at 0), so any mean over them is a censoring artefact biased
  upward by an unboundable amount -- averaging them converts "we cannot measure this" into a
  number.

**Upper censoring is real, not hypothetical.** `levels_db` is the product's own uncalibrated
integer scale, clipped into `[0, 86]` at **both** ends. Three cells sit at the 86 ceiling in bins
140-165 of fixture `...000004`, on a **quiet ambient** window; a close vessel pass -- the event of
interest -- will clip far harder. `fft_io.censoring_report()` returns the per-window counts at
each limit, and **every B5 band level must be reported next to them**: a band level from a window
with ceiling hits is a lower bound, not a measurement, and every threshold or regression built on
these levels must be censoring-aware.


---

## 4. The model: CLOSED, NO-GO (decision 0012)

**This section previously proposed Ocean Networks Canada's pretrained
[`selfsupervision_anomalies_onc`](https://github.com/OceanNetworksCanada/selfsupervision_anomalies_onc)
checkpoint as the vessel-presence model. B0 (§2, `docs/decisions/0012-b0-model-viability-outcome.md`)
tested that proposal against the real artefacts and returned NO-GO on three of the claims this
section made, plus two blockers independent of those claims. Rewritten rather than annotated,
because leaving the old claims as prose is how the same dead path gets re-adopted later in the
week by someone reading only this plan.**

What B0 actually found, against each claim v1-of-this-section made:

- **"`Engine Noise` is an output class" -- false.** The released, finetuned checkpoint
  (`ft-cls_best_checkpoint.pth`) has a single-logit head (`mlp_head.1.weight`, shape `(1, 768)`):
  it is a binary normal-vs-anomalous detector. `Engine Noise` is one of eight anomaly types pooled
  into the single positive class, and in the eval set that positive class is measured at ~80%
  non-Engine-Noise (31/1158 rows). There is no separable vessel signal to read out.
- **"`ICLISTENHF1266` is in its training/eval device list, with per-device figures" -- true, but
  the figures do not mean what they look like they mean.** The P 0.96 / R 0.88 / AUC 0.99 (n=149)
  reproduces bit-for-bit against the exact checkpoint we hold, but it is a binary-anomalous score
  at threshold 0.5, at **epoch 1**, and `result.csv`'s per-device header-to-column mapping is only
  name-attributable at that epoch (it drifts under a shuffled loader afterwards). And at
  `ICLISTENHF1266` specifically, the eval set carries **zero** Engine Noise rows at all -- only
  `Sensitivity` (20) and `Dropout` (4), both instrument faults. No device-matched engine-noise
  validation was ever possible, even before the head-shape problem.
- **"Use the CPU CNN baseline" -- does not exist.** Live `HfApi` enumeration of every `merileo/*`
  repo found no `cnn_baseline` directory anywhere. This was a literature-sweep error. The only
  checkpoints released are SSAMBA/Vision-Mamba `.pth` files, which need `mamba_ssm` and, on CPU,
  run through the reference (non-CUDA) selective-scan kernel -- 24 layers x 2 directions of
  pure-Python scan at the model's required sequence length. Measured at **~2 min/sample on 4
  cores**; scoring B3's ~9,300-file corpus once is **~12 CPU-days**, independent of RAM and of the
  head-shape problem above. Decisive on its own.
- The forward pass also OOM-kills under this container's actual **3.90 GB cgroup v2 cap**
  (`/sys/fs/cgroup/memory.max`), not the host's 30 GB that `free` reports -- corrected from an
  earlier, wrong framing that treated this as possibly architectural. It is "broken on this host"
  (invariant 9), and it does not matter: (e) above is fatal regardless of RAM.
- The repo's real licence is BSD-3-Clause, not MIT as first recorded (both permissive; not a
  decision input).

**Consequence: §5's B2 (spectrogram adapter) and B4 (reproduce-then-run) are struck.** B5
(band-level SPL, §5 below) is promoted from "B4's independent check" to **primary detector**. No
project goal is lost -- G1, G2 and G3 all still resolve on the B5 path; what is lost is a
cross-check and a headline (0012's "Consequences").

**How to tell this was wrong:** revisit only if ONC releases a checkpoint with a genuine
multi-label head that includes Engine Noise as a distinct, separately-thresholdable output, **and**
a GPU host becomes available -- both conditions, not either. A larger RAM allocation on this
container does not on its own reopen this path: only the OOM finding was RAM-bound, and it was the
least load-bearing of the findings above (the compute-cost finding and the missing-output finding
both hold regardless of RAM).

---

## 5. Segments

In dependency order. Each names its module, its gate, and its fallback.

### B0 -- Model viability gate *(Day 1, half a day -- this segment can kill §4)* -- **DONE, NO-GO**

**Complete.** Verdict: NO-GO on the whole §4 pretrained-checkpoint path. See
`docs/decisions/0012-b0-model-viability-outcome.md` and §2/§4 above for the full finding set. The
checklist below is left as the record of what B0 actually checked.

- **Input compatibility -- the real risk.** `onc_ssamba/utilities/spectrogram_utils.py` expects an
  ONC `.mat` spectrogram `[F, T]` of shape `[854, 1000]`. Ours is `.fft.gz`, 1200 x 512. Read one
  of each; compare shape, frequency axis, time axis, units and scaling. **Quantify the gap
  precisely** -- it decides whether B2 is an afternoon or the whole segment. *(Moot -- B2 struck.)*
- Confirm `ICLISTENHF1266` really appears in `result.csv`, and what those per-device numbers mean.
  *(Confirmed verbatim, but epoch-1/header-drift caveats apply -- 0012g.)*
- Confirm the CNN baseline loads and runs on CPU under torch 2.12, and that its dependencies exist
  here -- **check `cv2` in particular** (`spectrogram_utils` calls `cv2.resize`; CLAUDE.md does not
  list it as available). Report anything installed, per CLAUDE.md. *(No CNN baseline exists --
  0012a. `cv2` 5.0.0 confirmed present; CLAUDE.md corrected.)*
- Read class ordering from `args.pkl` or the h5 `label_names`. **Never assume the YAML order** --
  the model cards are a bare `license: mit` with no documented ordering. *(Neither source carried
  an explicit label list; order recovered empirically, and the head is single-logit binary
  regardless -- 0012b.)*

**Fallback taken:** the band-level SPL route (B5), which needs no weights at all and is the method
this community actually publishes, is now the **primary** detector, decided Day 1 as planned.

### B1 -- The `.fft.gz` gate *(Day 1-2, timeboxed to one day)*

`boatphone/fft_io.py`. Unchanged in substance from v1 §A2, and still the project's named blocker --
now doubly important, because the model in §4 consumes this exact surface.

1. **Re-run the magnitude regression under the 250 Hz mapping.** Product bin *b* against WAV bin
   *2b*, bins 1-205 only, censoring-aware (drop values at 0 and 86 -- regressing over clipped
   values biases the slope toward 1 spuriously), like estimator against like estimator
   (median-of-dB vs median-of-dB, never median vs Welch mean).
2. **Resolve time alignment empirically.** Cross-correlate on the **38 kHz echosounder transient**,
   which is narrowband, intermittent and high-contrast -- worth far more than broadband ambient.
   Do not assume the nominal 3.971 s offset between WAV and FFT file starts.
3. **Run the gate on a loud window, not the supplied sample.** The sample is quiet ambient (WAV
   broadband std 0.6 dB over 5 minutes); **any r from it is uninformative in either direction.**
   Source a loud window by ranking B3's corpus on 1-10 kHz broadband level.
4. **Characterise the ~42 dB low-frequency anomaly.** Test: high-pass in product generation;
   per-bin noise-floor subtraction; a different averaging convention in the low bins. In parallel,
   **email ONC for the product definition on Day 1 regardless of progress** -- it is a
   documentation question and the answer may take days.

**Pass:** slope ~ 1 within stated tolerance over bins 1-205; intercept characterised; convention
identified (power / PSD-per-Hz / spectral level; window-energy corrected; one-sided x2). Record as
documented constants and re-run whenever the reader changes.

**Fallback, decided now and not on Day 4:** proceed on relative and shape-based features --
spectral slope, band ratios, level *changes* through closest point of approach. G1, G2 and G3 all
survive this. Only the physics baseline (B6) is genuinely lost.

**Resources.** Peak RAM: unmeasured, estimate from array size -- one frame set is 1200x512 int
values (~0.6M cells, low tens of MB per file held in memory); a loud-window batch of a handful of
files stays well under 1 GB. Disk: negligible beyond the corpus B3 already pulls. Wall-clock: not
the binding constraint -- I/O and the cross-correlation search dominate over minutes, not hours.
Runs fine in the current 3.90 GB cgroup cap; no raise needed.

### B2 -- Spectrogram adapter *(struck -- see decision 0012)*

**Struck.** Bridged `.fft.gz` to the shape `onc_ssamba` expects. Moot once §4 returned NO-GO --
there is no model to adapt input for. Its allocated time goes to B5 and B7.

### B3 -- Bulk acquisition *(Day 1 evening, overnight)*

`boatphone/onc_client.py`, extending A1's listing code, which already handles paging and retry.

**Changed from v1:** target the **2024 and 2025 seasons first**. v1 said "the two most recent
summers", which now resolves to the dead 2026 and a partial 2025. Roughly 2.7 GB, 1.5-3 h at 1-2
files/sec. Extend backwards toward 2020 if throughput allows (~8 GB for the full span).

- Window: PlanetScope crosses 09:30-11:30 local, padded to **09:15-11:45 `America/Vancouver`**,
  computed per date with `zoneinfo` -- **never a hardcoded UTC offset**, which silently loses the
  shoulder months across a DST boundary.
- Cache by content hash; resume on interrupt; exponential back-off on rate limits. A pull this size
  **will** hit a transient failure -- make resumption the default path, not the exception.
- **Log every requested-but-absent file.** That log refines O1 with measured rather than inferred
  availability. Where the listing and the pull disagree, **the pull wins**; reissue O1 to Malachy
  with a note if any orderable date changes.

What this buys beyond the corpus itself: the loud window B1 needs, on Day 1; ambient statistics
with real N rather than the ~25 windows that carry optical labels; time-of-day-matched negatives
for free; and a hedge if the optical yield disappoints.

**Sampling conditionality, to be stated wherever a statistic from this corpus appears:** it is a
~10:30-local window, so it supports diurnal claims not at all, and seasonal claims only within that
hour band.

**Required second pass, or B1b deadlocks permanently.** B1's step 1 regresses product bin *b*
against **WAV** bin *2b*, but as written above B3 pulls only `.fft.gz`. B3 must add a second pass
pulling **WAV for the top ~5 loud windows** (~230 MB each, ~1 GB total) once the corpus can be
ranked on 1-10 kHz broadband level from the `.fft.gz` pull. Without this, B1's magnitude-regression
step has no WAV to regress against and cannot resolve.

**Resources.** Disk: ~2.7 GB for the 2024-2025 seasons (the immediate target), up to ~8 GB for the
full 2020-2025 span if throughput allows, **plus ~1 GB for the required WAV second pass above**
(measured file sizes, per §3/§5's own figures: `.fft.gz` is 0.29 MB/5-min file; WAV for the same
span is far larger, ~230 MB for a 5-min loud window). Peak RAM: unmeasured, estimate from array
size -- the listing/download loop holds one file at a time plus a resumable manifest, low
hundreds of MB. Wall-clock is the binding constraint, not RAM: 1.5-3 h at 1-2 files/sec for the
primary target, run overnight for exactly that reason. Runs fine in the current 3.90 GB cgroup
cap; no raise needed.

### B4 -- Reproduce, then run the model *(struck -- see decision 0012)*

**Struck.** Would have reproduced ONC's `result.csv` numbers on `ICLISTENHF1266` rows, then scored
`Engine Noise` across the B3 corpus with null checks. Moot: the checkpoint has no distinct Engine
Noise output to reproduce or score (0012b), and scoring the corpus once would cost ~12 CPU-days
even if it did (0012e). Its allocated time goes to B5 and B7. The null-check discipline it would
have exercised is preserved -- see §7's replacement "Detector agreement" row, now non-optional.

### B5 -- Physical baseline detector *(Day 3)* -- **now the primary detector**

`boatphone/features.py`, `boatphone/ambient.py`. **Not optional, and no longer just a
cross-check** -- with B4 struck (decision 0012), this is the project's vessel-presence detector.

- Decidecade band levels -- **above 2.2 kHz only**; below that, raw-bin levels **labelled as such**,
  not as standards-compliant band levels. Via PyPAM / MBARI `pbp` / MHKiT, the last of which
  explicitly supports icListen. All three operate on PSD, so **this runs on `.fft.gz` without ever
  touching WAV**.
- Rolling percentile ambient baseline by season, time-of-day and band; excess-over-ambient
  threshold with a minimum-duration constraint.
- **Compute on the time-frequency surface, not one averaged spectrum.** 1200 frames at 0.25 s
  across a +/-15 min window gives level-versus-time through closest point of approach: the CPA
  time, the slope of rise and fall, and the width of the peak jointly constrain range and speed in
  a way a single spectrum cannot. This is the most valuable structure in the data and the basis of
  standard single-hydrophone ranging.
- **Far-field diagnostic:** windows with zero in-AOI detections but elevated broadband level are
  direct evidence of out-of-AOI traffic. It falls out of the `n_vessels == 0` negatives already
  collected, and it is the AIS-free way to bound how often the far field contaminates labels.

Agreement between independent detectors is a far stronger result than either alone. With only one
model-free detector left, "independent" now means the band-split check in §7, the far-field
diagnostic above, and the null checks -- not a second trained model. Disagreeing among these is
itself a finding worth reporting.

**Resources.** Peak RAM: unmeasured, estimate from array size -- operates on PSD from `.fft.gz`
(1200x512 per file) or the decidecade-band reduction of it, never WAV; per-file and rolling-window
aggregates are on the order of the corpus size in `.fft.gz` form, low hundreds of MB at the
~2.7-8 GB disk footprint B3 delivers. Disk: none beyond B3's corpus. Wall-clock: band-level
reduction and percentile-baseline computation over ~9,300 files is expected to be minutes, not
hours -- CPU-bound on the reduction, not I/O-bound. Runs fine in the current 3.90 GB cgroup cap; no
raise needed.

### B6 -- Calibration and physics *(Day 3-4, gated on B1)*

`boatphone/calibrate.py`, `boatphone/priors.py`, `boatphone/propagation.py`.

- Parse the sensitivity file (3 header lines, 5120 rows, 10 Hz -> 51,200 Hz at 10 Hz spacing,
  dB re Counts^2/uPa^2). Interpolate onto the 250 Hz grid; subtract; return dB re 1 uPa^2/Hz.
- **Emit a band-validity mask.** Bins 0-205 calibratable; bins 206-417 are not covered by the file
  we hold. Every downstream consumer respects the mask and violations raise.
- **Request the 256 kHz sensitivity curve from ONC** in the same message as B1's product-definition
  question.
- Sanity-check quiet-period levels against Wenz ambient curves for the sea state -- expect
  agreement within a few dB.
- SL from JOMOPANS-ECHO (+/-6 dB, validated in BC through the ECHO programme) plus the small-vessel
  extension; TL via Bellhop with GEBCO/CHS bathymetry and SSP from ONC's co-located Folger CTD.

**Two gates, and this is the first segment to cut if the week compresses.** Neither B4 nor B5
depends on absolute calibration -- the model consumes normalised spectrograms and B5 works on
excess-over-ambient. And `arlpy` is not installed, with Bellhop needing an acoustics-toolbox FORTRAN
binary that `pip install arlpy` does not supply (`docs/environment-audit.md`).

**Deliver a posterior with stated width, not a point inversion.** A single hydrophone does not
jointly determine range, size, hull class, speed, count and the local correction term. Put priors
on class, speed and count; sample over range and size; report posterior width. The error budget up
front: +/-6 dB SL uncertainty against a ~15-18 dB/decade TL slope is roughly a **factor of ~2 in
range**, before class, speed and count uncertainty.

**Resources.** Peak RAM: unmeasured, estimate from array size -- the sensitivity file is 5,120
rows, negligible; the posterior sampling over range/size/class/speed is the largest allocation and
is expected to be low hundreds of MB for a Monte Carlo grid of this scale. Disk: negligible
(GEBCO/CHS bathymetry tiles for the AOI only, order tens of MB). Wall-clock/CPU is the binding
constraint if Bellhop ray-tracing runs at all: ray tracing over a bathymetry grid is the slow step,
not the RAM footprint -- and it is moot regardless, since `arlpy`/Bellhop are not installed
(`docs/environment-audit.md`) and this is "the first segment to cut" per the text above. Runs fine
in the current 3.90 GB cgroup cap; no raise needed.

### B7 -- Matchups and split hygiene *(Day 4, gated on I3)*

`boatphone/geo.py`, `boatphone/matchup.py`, `boatphone/splits.py`.

- Range and bearing from the hydrophone via `pyproj`. Join each detection to its +/-15 min acoustic
  window.
- Emit `matchups.parquet` with **`overpass_id` on every row** and **AIS columns present but
  nullable**, so an optional AIS stage drops in later against an unchanged schema.
- **Dominance stratification:** `R2 > 2*R1` -> acoustically dominated, the MVP subset (larger than
  `n_vessels == 1` would be); genuine mixtures -> nearest-vessel range plus count;
  `n_vessels == 0` -> negatives.
- **`splits.py` is the single source of splits** -- overpass-grouped, time-blocked CV, with a hard
  assertion that no `overpass_id` appears in both train and test.

**The effective sample size is the number of usable overpasses (order 20-30), not the number of
detections (100-400).** A scene with 8 vessels gives 8 labels on *one* acoustic mixture -- eight
labels on one measurement. For anything acoustic -- sea state, tide, SSP, ambient, propagation,
hydrophone state -- the unit of analysis is the overpass. Report uncertainty clustered by overpass,
and **state both counts alongside every metric**: "n = 214 detections across 23 overpasses."

**Resources.** Peak RAM: unmeasured, estimate from array size -- `matchups.parquet` is order
100-400 detection rows across ~20-30 overpasses, trivial; the geodesy join (`pyproj`) is per-row
and does not hold the acoustic corpus in memory simultaneously. Disk: negligible, one parquet file.
Wall-clock: not the binding constraint. Runs fine in the current 3.90 GB cgroup cap; no raise
needed.

### B8 -- Results *(Day 5)*

Detectability curve (G3), size separability including a negative result (G2), continuous presence
estimate (G1), the B5-vs-band-split-agreement result (replacing model-versus-physics agreement,
which needed the struck B4), and the matchup dataset. Final notebooks, README workflow section,
presentation.

**Every model number carries its overpass count. Every population statistic carries the
sampling-conditionality caveat** -- PlanetScope is sun-synchronous at ~10:30 local and usable scenes
are cloud-free summer days, precisely the conditions that maximise recreational traffic. No
population statistic here generalises to Folger Passage at large.

**Resources.** Peak RAM: unmeasured, estimate from array size -- final notebooks re-load
`data/derived`/`data/processed` outputs (parquet, csv), not the raw corpus; expected low hundreds
of MB. Disk: negligible beyond outputs already produced. Wall-clock: notebook re-execution time,
not a new computation. Runs fine in the current 3.90 GB cgroup cap; no raise needed.

---

## 6. Interfaces

These are contracts. Each names the artefact, the owner, the schema, and what breaks without it.

### IN

| # | From | Artefact | When | Notes |
|---|---|---|---|---|
| **I1** | Malachy | `data/interim/overpasses.csv` -- `overpass_id`, `acquired_utc`, `scene_id`, `cloud_fraction`, `aoi_wkt`, `ordered` | Day 3 | `acquired_utc` **must be UTC and must be the acquisition instant**, not the order or publish time. A systematic timestamp error propagates directly and silently into every range label. Agree the convention in writing on Day 1. |
| **I2** | Malachy | Hydrophone coordinates, AOI centre and radius | Day 1 | |
| **I3** | Neve | `data/interim/detections.parquet` -- `overpass_id`, `detection_id`, `lat`, `lon`, `length_px`, `length_m_est`, `size_class` (FAO bin), `confidence` | Day 3 | Blocks B7 only |
| **I4** | Neve | Measured detector P/R on the hand-labelled set, per size class | Day 4 | The acoustic detectability curve cannot be cleaner than the optical labels that define it |

### OUT

| # | To | Artefact | When |
|---|---|---|---|
| **O1** | Malachy | `docs/derived/hydrophone_gaps.md` + `data/derived/hydrophone_uptime.csv` | **Delivered.** Reissue after B3's pull refines it |
| **O1b** | Malachy | **"2026 has no hydrophone data -- order ARCHIVE imagery, 2020-2025 seasons only"** | **Day 1, urgent** |
| **O2** | Both | Acoustic band support + calibration validity note | Day 2 |
| **O3** | Joint | `data/processed/acoustic_features.parquet` | Day 3 |
| **O4** | Joint | `data/processed/matchups.parquet` | Day 4 |
| **O5** | Write-up | Detectability curve, band-level (B5) detector results plus the band-split agreement check, physics baseline | Day 5 |

**B1-B6 are deliberately independent of I1 and I3.** Only B7 blocks, and B3's speculative pull
means it starts the moment `overpasses.csv` lands, with no download latency.

---

## 7. Verification

There is no pytest, no lint and no type check (CLAUDE.md). What exists is `scripts/checks.py` and
re-executing a notebook top-to-bottom in a fresh kernel.

| What | How | Pass criterion |
|---|---|---|
| **Synthetic control** | Tone of known level, frequency and time pushed through `.fft.gz` -> model input | Lands in the right bin and the right `t_utc_s` (invariant 3) |
| FFT reader -- structure | Row/bin counts; the three top-of-band regions (§3.3) | 1200 x 512; **exact** zero over cols 425-511; col 0 and cols 419-424 bounded; skirt mean non-increasing over bins 405-424 |
| FFT reader -- mapping | Echosounder-hump power-excess **centroid** (§3.1); WAV absolute cross-check | Centroid in bins 149.0-152.0; hump at 35-40.5 kHz on the 128 kHz WAV. Shoulder onset ~ bin 408 is a **structural** check only -- struck as mapping evidence |
| FFT reader -- axis convention | `check_b0_2a_axis_uncertainty_is_carried` (§3.2) | `FFT_AXIS_OFFSET_UNCERTAINTY_HZ > 0` **and** `band_limit` demonstrably widens by it; no check pins a bin tighter than +-1 |
| FFT reader -- magnitude | Product bin *b* vs WAV bin *2b*, bins 1-205, censoring-aware, loud window | Slope ~ 1; intercept characterised; convention documented |
| Time alignment | Cross-correlate on the 38 kHz transient | Lag resolved to < 1 frame (0.25 s) |
| Calibration coverage | Assert every absolute-level output falls in bins 1-205 | Violations raise, enforced in `calibrate.py` |
| Calibration accuracy | Quiet-period levels vs Wenz curves for the sea state | Within a few dB |
| Band claims | Assert decidecade bands only requested above 2.2 kHz | Enforced in `features.py` |
| ~~Model reproduction~~ | ~~Our `Engine Noise` metrics vs ONC's `result.csv`, HF1266 rows only~~ | **Struck -- decision 0012, no model to reproduce.** ONC's own numbers were confirmed verbatim against the checkpoint (epoch-1, header-drift caveats apply) but that reproduction was never the gate on a downstream number, since the model itself is not used. |
| **Null checks** | Shuffled labels; +1 h time shift; known-quiet period | Reported explicitly, pass or fail. **Non-optional**, not a nice-to-have -- this is now the project's main defence against the exact false-positive pattern invariant 4 warns about, with only one model-free detector in play. |
| **Detector agreement** *(replaces the "ONC model vs B5" row, struck with B4)* | Three independent checks over the same windows: (i) **band-split agreement** -- an independent detector on a disjoint band (e.g. 1-4 kHz vs 8-20 kHz) against B5's primary band; (ii) the **§5 far-field diagnostic** (elevated broadband level with zero in-AOI detections); (iii) the **null checks** above | Agreement/consistency rate reported for each; disagreements characterised, not hidden |
| Split hygiene | Assert no `overpass_id` in both train and test | Zero overlap, enforced in `splits.py` |
| Deployment stability | Re-run the gate on one file from each deployment 2020-2026 | Consistent, or per-deployment constants recorded |
| Repo hygiene | `git log --stat` | No `data/` paths (inv. 2), no notebook outputs (inv. 7), no tokens |
| Reproducibility | Fresh clone + environment manifest, run the notebooks end to end | Works from an ONC token alone, no AIS required |

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| **`.fft.gz` is not the array the model expects** | Moot -- B0 returned NO-GO on the model itself (decision 0012), independent of this gap. Historical: B0 quantified it before the NO-GO closed the question. |
| ~~ONC's reported numbers do not reproduce~~ | **Resolved, not a live risk.** They reproduced verbatim (epoch-1 caveats apply) -- the NO-GO came from the head shape and compute cost, not a reproduction failure. |
| **B1's low-frequency anomaly never resolves** | Relative and shape-based features, decided Day 1. G1-G3 survive; only B6 is lost. |
| ~~No GPU in this environment -> CPU CNN baseline; confirmed as a supported path in B0~~ | **False -- corrected by decision 0012.** No CNN baseline exists at all (0012a), and the actual SSAMBA/Mamba CPU path costs ~12 CPU-days per corpus pass regardless of GPU/RAM (0012e). The real mitigation is the same as always: B5 needs no weights and is now primary. |
| **`arlpy` / Bellhop unavailable** | B6 is the first segment to cut, and nothing else depends on it. |
| **No 256 kHz calibration curve from ONC** | Restrict absolute levels to <= 51.2 kHz (bins 1-205). Costs the 51-104 kHz band, which matters little for vessel noise. |
| **Archive imagery scarce for 2020-2025** | Flagged Day 1 via O1b so Malachy can check yield before spending quota. If scarce, B5 still delivers a continuous presence estimate with no optical labels; only G2 and G3 need matchups. |
| **Effective N too small for any ML claim** | B5 and B6 need ~0 training data and produce a result regardless. |
| **I1 timestamps in local or publish time** | Agree the convention in writing Day 1; assert UTC and sanity-check against solar time (~10:30 local overpass). |
| **B3 throughput worse than 1 file/sec** | Pull the newest season first and work backwards, so the corpus is always useful at whatever depth it reaches. |
| **The container's RAM cap is now a provisionable parameter, not a fixed constraint** | Up to 30 GB is available on request (current cgroup v2 cap measured at 3.90 GB). Each segment's Resources line in §5 states what it needs and whether it needs a raise -- as of this writing, none of B1/B3/B5/B6/B7/B8 needs one. **This does not reopen the B0 NO-GO.** Of 0012's findings, only the OOM finding (d) was RAM-bound, and it was already the least load-bearing of the three primary blockers -- the missing multi-label output (b) and the ~12-CPU-day compute cost (e) are both independent of RAM and hold regardless of any raise. See 0012h. |

---

## 9. Assumptions

- **Timeline is the OHW hackathon week**, Day 1 = Thu 2026-08-27.
- **This stream owns the full acoustic stack** -- acquisition through physics through the acoustic
  half of the modelling. Optical acquisition and detection are owned elsewhere.
- **`matchup.py` sits on the acoustics side**, since the join is against acoustic windows on the
  acoustic time base.
- **PlanetScope overpasses fall within 09:30-11:30 local** for the whole period of interest, per
  the team. B3's speculative pull rests on this; verify against the first `overpasses.csv` rather
  than trusting it.
- **AIS stays optional.** Nothing above depends on it. If access lands it adds the blind-spot
  fraction, range-label validation, and a within-site control.

---

## 10. Out of scope

Acoustic Doppler speed estimation beyond a single measured statement (G4 is a stretch goal, and the
JASA finding that planing hulls show weak speed-SL dependence makes it harder than it looks); the
Folger Pinnacle hydrophone (build for Folger Deep alone, with the hydrophone as a config parameter,
and add Pinnacle only if Day 3 lands early); a second observatory site; pre-2020 deployments; bulk
WAV/FLAC acquisition; sub-250 Hz features as a pipeline component; training any model from scratch;
blind source separation; VTUAD v1 or v2; the ONC-AIS/CPA labelling pivot.
