"""Shared project constants -- one definition, for ALL workstreams (acoustics,
Planet acquisition, optical detection), not an A1-only module.

If you need the study window, the season months, or the device/location, import
them from here. A second definition in a notebook or in `scripts/config.py` makes
the optical-acoustic matchup join two different windows and produce a wrong answer
rather than an error (CLAUDE.md invariant 6).

Conventions pinned here (decision 0002, decisions D1/D2/D4 of the A1 plan):

* **Time base**: UTC end to end. Every `datetime` exported by this module is
  tz-aware with a zero UTC offset. A naive datetime is never a valid value.
* **Bin grid**: fixed-width `BIN_SECONDS` bins, half-open `[start, end)`, with
  edges at integer multiples of `BIN_SECONDS` seconds since the UTC epoch.
* **Season**: `SEASON_MONTHS_UTC` is evaluated on the **UTC** month of the bin
  start. No local-timezone conversion happens anywhere in A1.

Names carry their frame: `*_UTC` means tz-aware UTC, `*_SECONDS` means seconds.
"""

from datetime import datetime, timezone

# Bin width for the uptime calendar. Source: A1 decision D2 -- 300 s (5 min) is
# the ONC FFT product's natural file cadence, so a bin maps 1:1 onto a listing.
BIN_SECONDS = 300

# Start of the study window. Source: the validity date of the ICLISTEN HF1266
# pre-deployment calibration file shipped with the Folger Deep sample. Earlier
# Folger deployments are *different devices* with different sensitivities, so
# nothing before this date is comparable in calibrated units.
STUDY_START_UTC = datetime(2020, 2, 18, 0, 0, 0, tzinfo=timezone.utc)

# End of the study window. CHOSEN BOUND, NOT MEASURED: the planner did not pin a
# value, so this is the end of the 2026 field season (2026-10-01T00:00:00Z,
# exclusive). It is an analysis convention -- revise it here, not per notebook,
# if the deployment record extends further.
STUDY_END_UTC = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)

# Months included in the "season", evaluated on the UTC month of the bin start.
# Source: A1 decision D4 -- May through September, the recreational-vessel
# season in Barkley Sound. Deliberately UTC, not America/Vancouver: a local
# definition would make the season edge depend on DST (see decision 0002).
SEASON_MONTHS_UTC = (5, 6, 7, 8, 9)

# The hydrophone. Source: ONC metadata for the Folger Deep ICLISTEN HF1266
# instrument (device code as used by the ONC API; numeric device id alongside).
DEVICE_CODE = "ICLISTENHF1266"
DEVICE_ID = 23235

# ONC data-product extension used for uptime listing. Source: the gzipped FFT
# product present in the local sample acquisition.
PRODUCT_EXTENSION = "fft.gz"

# Extension used by ONC's *archive-file index* for the FFT product. NOT the same
# string as PRODUCT_EXTENSION: the archive index registers these files as "fft"
# (e.g. ICLISTENHF1266_20240715T000136.000Z.fft), and they arrive gzipped on disk
# as ".fft.gz" -- which is what the local sample acquisition shows. Source:
# measured 2026-08-27 against the live ONC archivefile endpoint for FGPD/
# HYDROPHONE -- a listing filtered on "fft.gz" returns ZERO files for a day that
# has 289 files under "fft".
ARCHIVE_EXTENSION = "fft"

# ONC device-category code for the hydrophone, used to scope an archive listing
# at a location. Source: ONC deployment metadata for DEVICE_CODE
# (deviceCategoryCode == "HYDROPHONE").
DEVICE_CATEGORY_CODE = "HYDROPHONE"

# Case-insensitive fragment that identifies the Folger sites in an ONC location
# NAME. Discovery matches on this, never on a location code, so the code is
# whatever ONC returns at runtime. Source: ONC location name "Folger Deep".
FOLGER_NAME_FRAGMENT = "folger"

# Nominal duration of one archive FFT file, in seconds. MEASURED, not assumed:
# 2011 unique .fft files over 2024-07-12..18 at Folger Deep give consecutive
# start deltas of 300 s for 1909 of 2010 gaps, 297-302 s for 98 more (sub-second
# start jitter, files are NOT bin-aligned), and 2 genuine outage gaps (637 s,
# 1780 s). Equal to BIN_SECONDS by measurement, kept as its own name because it
# is a property of the instrument's file cadence, not of the calendar's grid.
FFT_FILE_SECONDS = 300

# ---------------------------------------------------------------------------
# ONC archive-listing response cap (the A1d defect).
# ---------------------------------------------------------------------------
# ONC's archivefile listing endpoint returns ONE PAGE and truncates silently:
# the `files` list simply stops partway through the requested span, with no
# error and no flag inside the list itself. The only signal is the response's
# top-level `next` field.
#
# MEASURED 2026-08-27 against the live endpoint (FGPD / HYDROPHONE / "fft"):
#   2024-05-01..2024-10-01  -> 11,121 rows, stopping at 2024-06-08T15:10Z,
#                              `next` = {"parameters": {..., "page": "2"}}
#   2024-05-01..2024-07-01  -> 11,121 rows, same truncation point
#   2024-05-01..2024-12-01  -> 11,121 rows, same truncation point
#   2020-01-01..2021-01-01  ->  9,977 rows, `next` present
#   2024-07-01..2024-08-01  ->  8,922 rows, `next` = None (complete)
# Paginating the 2024 season took 4 requests / 50.6 s for 44,027 unique files,
# with page sizes 11121, 11075, 11086, 10745.
#
# So the cap is NOT a fixed row count -- page sizes differ by hundreds of rows
# between pages of one query -- which is why this constant is named "observed
# max" and is used only for DIAGNOSTIC MESSAGES, never to decide whether a
# response was truncated. That decision is made from `next`, which is
# authoritative. Believing a row-count threshold instead would reintroduce the
# defect the moment ONC's page size moved.
ONC_LISTING_PAGE_ROWS_OBSERVED_MAX = 11121

# Hard stop on the page-following loop, so a server that ignores our paging
# parameter cannot spin forever. CHOSEN BOUND, NOT MEASURED: the 2020-2026 study
# window is ~44 k files per season, i.e. single-digit pages per calendar year, so
# 200 is orders of magnitude of headroom and still terminates. Exceeding it
# raises rather than returning a short list.
ONC_LISTING_MAX_PAGES = 200

# Floor for the adaptive time-subdivision fallback used when a response says it
# is truncated but advertises no usable paging parameter. A span this short holds
# one FFT file, so it cannot be subdivided further; still-truncated at the floor
# raises with the span named. Source: FFT_FILE_SECONDS above.
ONC_LISTING_MIN_SUBCHUNK_SECONDS = FFT_FILE_SECONDS


# ===========================================================================
# VTUAD constants -- RETAINED, but the corpus is NOT acquired.
#
# acoustics_plan_v2 replaced the VTUAD transfer experiment with ONC's own
# pretrained checkpoint (decision 0009), so nothing below is on the critical
# path. They are kept because they are sourced, dated, and drift-checked by
# scripts/checks.py (A8a) against docs/vtuad-facts.md -- re-deriving them costs
# more than carrying them, and the band figure is cited by decision 0010.
# ===========================================================================

# ---------------------------------------------------------------------------
# VTUAD -- Vessel Type Underwater Acoustic Data (Domingos, Skelton, Santos).
#
# DOI 10.21227/msg0-ag12. Derived from the ONC Fraser River Delta Lower Slope
# hydrophone (icListen AF, device ICLISTENAF2523, 147 m depth, Strait of
# Georgia), 2017-06-24 to 2017-11-03, labelled from AIS.
#
# Full provenance, retrieval dates and the UNKNOWN rows: docs/vtuad-facts.md.
# scripts/checks.py (A8a) asserts these constants against that document, so a
# change here without a change there is a hard failure, not a silent drift.
#
# CALIBRATION WARNING: VTUAD audio is UNCALIBRATED raw PCM (arbitrary counts),
# while Folger levels are calibrated dB re 1 uPa. Band-matching alone does NOT
# make the two comparable -- see the calibration section of docs/vtuad-facts.md
# and docs/decisions/0002-time-alignment-and-units.md.
# ---------------------------------------------------------------------------

# Source: RIFF `fmt ` chunk of ICLISTENAF2523_20170701T000105.178Z.wav, read via
# the ONC Oceans 3.0 archivefile API from the VTUAD source deployment,
# https://doi.org/10.34943/07de823a-41c0-48b5-9bc3-704513d55ccc (2026-08-27).
# Mono, 24-bit PCM. Nyquist = 16000 Hz. NOT stated in any VTUAD document --
# measured. Read the rate from each file anyway (decision 0002 rule 2); this
# constant is the value an unexpected rate should be rejected AGAINST, never a
# substitute for reading the header.
VTUAD_SAMPLE_RATE_HZ: int = 32000

# Source: docs/vtuad-facts.md `label_schema` row, from the IEEE DataPort landing
# page and dataset README, https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data
# (2026-08-27). Kept as one string so the A8a drift check can compare it to the
# facts document verbatim. Note what is ABSENT: no continuous range, and no
# vessel length or size class.
VTUAD_LABEL_SCHEMA: str = (
    "5-class vessel type (background, cargo, tanker, tug, passengership) "
    "x 3 range-bin scenarios (2000-4000, 3000-5000, 4000-6000 m); "
    "no vessel length or size class"
)

# Source: IEEE DataPort file listing (archive names inclusion_2000_exclusion_4000,
# inclusion_3000_exclusion_5000, inclusion_4000_exclusion_6000) and the dataset
# author's clarification in the landing-page comments (Lucas Domingos,
# 2024-07-02): inclusion_3000_exclusion_5000 means the vessel is anywhere between
# 3 km and 5 km of the hydrophone.
# https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data (2026-08-27)
#
# The abstract and README describe the scenarios as 2-3 km, 3-4 km and 4-6 km,
# which CONTRADICTS the archive filenames. The filenames plus the author's own
# comment are taken as authoritative; confirm against a downloaded metadata.csv
# in A8b before any range-dependent result depends on it.
VTUAD_ZONE_RADII_M: tuple[tuple[int, int], ...] = (
    (2000, 4000),
    (3000, 5000),
    (4000, 6000),
)

# Source: IEEE DataPort "Dataset Files" listing (2026-08-27): 3.31 GB + 5.83 GB
# + 4.4 GB of compressed ZIPs. APPROXIMATE -- the listing does not say whether
# GB means 10^9 or 2^30; this uses 10^9. At 2^30 the total is about 14.5e9
# bytes. Do not treat this as an exact byte count; it exists to size a download,
# and it refutes the ~1 TB figure that previously circulated in the plans.
VTUAD_TOTAL_SIZE_BYTES: int = 13_540_000_000

# Source: same listing. The three scenarios are INDEPENDENT archives, so the
# smallest useful download is one scenario, not the whole corpus.
VTUAD_SMALLEST_UNIT_SIZE_BYTES: int = 3_310_000_000

# Source: Welch PSD (Hann, nperseg=8192) of the first 30 s of the archived file
# named above, https://doi.org/10.34943/07de823a-41c0-48b5-9bc3-704513d55ccc
# (2026-08-27). Anti-alias rolloff reaches -3 dB at 11652 Hz relative to the
# 2-10 kHz median, so the upper edge is the -3 dB corner, not Nyquist.
VTUAD_BAND_POPULATED_HZ: tuple[int, int] = (10, 11652)


# ===========================================================================
# ONC SSAMBA pretrained checkpoint -- label space and its hard limits (B0-4).
#
# Pinned artefacts (docs/derived/b0_external_provenance.json):
#   repo       OceanNetworksCanada/selfsupervision_anomalies_onc @ e3aebfc
#   checkpoint merileo/finetune-amba-...-noexclude @ eef6c151,
#              finetune/ft-cls_best_checkpoint.pth
#              sha256 e474ddc5a407c78731b9dcfc944a83e53e1da91797bee6851ad2b852c3dc35ca
#   eval h5    merileo/onc-ssl-tutorial @ ed38793a,
#              different_locations_incl_backgroundpipelinenormals_multilabel_SMALL.h5
#
# READ THIS BEFORE USING ANY INDEX BELOW.
#
# The finetuned checkpoint is NOT an 8-class Engine Noise classifier. Measured
# from the checkpoint's own tensors on 2026-08-27:
#     mlp_head.1.weight  ->  shape (1, 768)
# i.e. the model emits ONE logit. `onc_ssamba/utilities/training_utils.py`
# create_model() confirms the rule that produced it:
#     label_dim = 1 if args.task == 'ft_cls' and args.n_class == 2
# and `onc_ssamba/dataset.py` shows the target it was trained against:
#     labels = torch.tensor(float(sample['is_anomalous']))
# where `is_anomalous` is TRUE for ANY label other than 'normal'. The single
# logit is therefore a BINARY normal-vs-anomalous score in which Engine Noise is
# pooled with Anomaly, Data Gap, Dropout, Rain, Sensitivity, Tonal and Unknown
# Feature. There is NO index into the model output at which Engine Noise can be
# read out. ENGINE_NOISE_LABEL_INDEX below indexes the DATASET label matrix, not
# the model.
# ===========================================================================

# Ordered label list of the eval h5's `labels` matrix, columns 0..7.
#
# DERIVED FROM THE DATA, NOT FROM THE YAML. The h5 has NO `label_names` dataset
# (its only datasets are index_map_original, label_strings, labels, sources,
# spectrograms, split/*). The order below was recovered on 2026-08-27 by
# cross-tabulating each column of `labels` (1158, 8) against `label_strings`:
# every row with column c set was checked to contain the c-th name below in its
# semicolon-separated label string, and no other column explains it. Column sums
# were [22, 25, 55, 31, 28, 20, 59, 27].
#
# It happens to agree with external/onc_ssamba/config/dataset_config.yaml's
# `anomaly_labels`, but that agreement is a CHECK, not the source: the YAML is
# not what built the array.
ONC_MODEL_LABEL_NAMES: tuple[str, ...] = (
    "Anomaly",
    "Data Gap",
    "Dropout",
    "Engine Noise",
    "Rain",
    "Sensitivity",
    "Tonal",
    "Unknown Feature",
)

# Column of ONC_MODEL_LABEL_NAMES that means vessel engine noise. This indexes
# the DATASET label matrix (h5 `labels[:, 3]`). It is NOT a model output index --
# see the block comment above. Any use of it to slice a logit vector is a bug.
ENGINE_NOISE_LABEL_INDEX: int = ONC_MODEL_LABEL_NAMES.index("Engine Noise")

# Number of logits the finetuned checkpoint actually emits. MEASURED from
# mlp_head.1.weight.shape == (1, 768), not read from args.pkl (whose `n_class`
# and `num_classes` are both 2 and describe the UNUSED VisionMamba v.head).
ONC_MODEL_N_LOGITS: int = 1

# Decision threshold the ONC training code applies to sigmoid(logit) when it
# reports precision/recall. Source: `calculate_binary_metrics(..., threshold=0.5)`
# in onc_ssamba/utilities/metrics/hydrophone_metrics.py. Stated here because
# every P/R figure quoted from result.csv is a figure AT THIS THRESHOLD.
ONC_MODEL_DECISION_THRESHOLD: float = 0.5

# Input the model hard-requires: (freq, time) = (512, 512), single channel.
# Source: AMBAModel.forward raises ValueError for anything else, and the
# checkpoint's v.pos_embed is (1, 2501, 768) = 50x50 patches + 1 cls token,
# which is exactly (512 - 16)//10 + 1 = 50 per axis at patch 16 / stride 10.
ONC_MODEL_INPUT_FDIM: int = 512
ONC_MODEL_INPUT_TDIM: int = 512

# Per-spectrogram normalisation constants used by the finetune run, applied as
#     (x - mean) / (2 * std)
# Source: args.pkl of the pinned run (`dataset_mean`, `dataset_std`) and
# ONCSpectrogramDataset.normalise() in onc_ssamba/dataset.py. These are in the
# units of the ONC .mat `SpectData.PSD` field. They are NOT calibrated dB re
# 1 uPa and they are NOT interchangeable with Folger calibrated levels --
# see docs/decisions/0002-time-alignment-and-units.md.
ONC_MODEL_DATASET_MEAN: float = 51.506817
ONC_MODEL_DATASET_STD: float = 13.638703


# ---------------------------------------------------------------------------
# B0-2a: the .fft.gz product's frequency and time axes.
# ---------------------------------------------------------------------------
# ONE definition of the axis facts (CLAUDE.md invariant 6). boatphone/fft_io.py
# and every notebook import them FROM HERE; they are not restated anywhere else.
#
# The product is a gzipped ASCII grid of whitespace-separated integer levels,
# row-major, frames-then-bins. VERIFIED on both local fixtures: each file holds
# exactly 614,400 values == 1200 x 512, and only the row-major (C-order)
# reshape reproduces the documented structural zeros and the documented
# anti-alias shoulder onset (the column-major reading puts 90,543 nonzero
# values inside the 419-511 "zero" block instead of 74).

# Frames per file and bins per frame. Source: acoustics_plan_v2 SS3 ("1200 x 512")
# and a direct value count of both fixtures under data/Folger Deep Hydrophone
# Data Sample/ (614400 == 1200 * 512, with no header line).
FFT_N_FRAMES: int = 1200
FFT_N_BINS: int = 512

# Hz per FFT bin. Source: acoustics_plan_v2 SS3 -- 1024-pt FFT at 256 kHz gives
# 512 one-sided bins over 0-128 kHz, i.e. 250 Hz/bin, with bin 1 at 250 Hz.
#
# 250 Hz/bin is CONFIRMED on absolute physics, not on the product's own axis.
# The sample WAV (ICLISTENHF1266_20260313T000000.029Z_...wav) is 128 kHz / 24-bit
# mono -- it sees 0-64 kHz and needs no product axis to state a frequency. Three
# measurements on it (B1a adjudication, 2026-08-27):
#   * a +14.1 dB hump over 35.0-40.5 kHz, exactly where 250 Hz/bin puts the
#     bins 140-162 feature. At 125 Hz/bin the same feature would sit at
#     17.5-20.3 kHz, where the WAV shows <= 2.4 dB.
#   * NOTHING at 49.8-50.1 kHz, where 125 Hz/bin would put the bins 399/400 spur.
#   * NOTHING at 40.2-41.2 kHz, where 125 Hz/bin would put the bins 322-329 feature.
# NOT evidence: the anti-alias shoulder near bin 408. The shoulder sits at
# 0.4*fs and the bin width is fs/1024, so the shoulder bin is 0.4*1024 = 409.6
# FOR ANY fs -- it is equally consistent with 125 and 250 Hz/bin and never
# discriminated between them. It survives only as a STRUCTURAL check
# (FFT_ROLLOFF_ONSET_BIN below).
#
# The "+14.1 dB" / "NOTHING (<=X dB)" figures above, and FFT_ECHOSOUNDER_ABS_
# CENTRE_HZ below, are stated relative to a REFERENCE BAND, not the WAV's full
# 0-64 kHz span -- an unstated convention until this correction. Only a
# 20-60 kHz reference band reproduces both headline figures (+14.06 dB /
# -6.24 dB); the full 0-64 kHz band gives +11.64 dB / -8.82 dB instead. Named
# here per CLAUDE.md invariants 3 and 6 rather than left implicit in a script.
FFT_WAV_REFERENCE_BAND_HZ: tuple[float, float] = (20_000.0, 60_000.0)
FFT_BIN_WIDTH_HZ: float = 250.0

# Which point of a bin FFT_BIN_WIDTH_HZ * k names: the bin CENTRE, so bin 1 is
# 250 Hz. THIS IS A NAMED ASSUMPTION, NOT A SETTLED FACT.
#
# The reader must be deterministic, so the axis is pinned to centres and
# frequency_axis_hz()[1] == 250.0 exactly. But the alternative -- ONC intending
# bin k to SPAN [k*dF, (k+1)*dF), i.e. an edge/filterbank convention -- is not
# excluded, and the B1a adjudication put it at roughly 60/40 in favour of EDGE:
#   FOR EDGE: a narrow, reproducible spur straddles bins 399/400 with a power
#     centroid of 399.47 / 399.52 on two files five minutes apart -- the
#     signature of a tone exactly halfway between two bin CENTRES, which under
#     the edge convention is the round 100.000 kHz. Also, 512 bins of 250 Hz
#     tile [0, 128000) exactly under EDGE, and there IS a low-frequency filter
#     (a 42 dB deficit in the lowest bins).
#   FOR CENTRE: a 1024-pt real FFT gives 513 bins (DC..Nyquist); a 512-column
#     product is most naturally that with Nyquist dropped, k = 0..511 at centres
#     k*dF -- which makes column 0 the DC bin and explains its near-exact zero
#     far more economically than a high-pass filter.
#   NOT EVIDENCE EITHER WAY: the 100 kHz roundness argument is a prior, not a
#     measurement -- of six narrowband lines found, the other five are round
#     under NEITHER convention, and there is no plausible 100 kHz source at
#     Folger (the AZFP runs 38/67/125/200/455/769 kHz, ADCPs 300/600/1200 kHz).
#     The WAV cross-check is CIRCULAR until B1 pins counts->dB: the implied
#     offset swings from -0.10 to +0.89 bins as the assumed level scale moves
#     from 0.25 to 3.0 dB/count.
# TO RESOLVE, in order: (a) ask ONC for the product definition -- one sentence
# settles it; (b) a scale-free two-bin-split census of narrow lines over the B3
# corpus (sub-bin centroids clustering at half-integers => edge, at integers =>
# centre); (c) re-run the WAV centroid comparison once B1 pins the level scale.
# UNTIL THEN: no check may assert a bin position tighter than +/- 1 bin.
FFT_AXIS_CONVENTION: str = "centre"

# The price of the open question above, in Hz, on EVERY band edge derived from
# this axis. If ONC means edges rather than centres, the true centre of bin k
# is (k + 0.5) * dF, i.e. up to half a bin ABOVE where we name it -- so the
# raw uncertainty is one-sided, toward HIGHER frequency. But the mask must be
# a SUPERSET under EITHER convention (we do not know which is right), so it is
# CARRIED SYMMETRICALLY on both edges of every band: `boatphone.models.band_limit`
# widens both the low and the high edge by this amount (verified,
# `check_b0_2a_axis_uncertainty_is_carried`), and
# `boatphone.fft_io.band_limit_product` does the same. Widening only the high
# edge would be correct if the true convention were known to be "edges"; since
# it is not, symmetric widening is the only choice that cannot silently drop a
# bin the other convention would have included.
#
# Consequences, stated where the number is: B5 is NOT blocked provided it
# carries this and puts no band edge inside a narrow feature. B6's calibration
# interpolation inherits <= 125 Hz of error -- negligible above ~2 kHz where the
# calibration curve is flat, non-negligible only in bins 1-4, which the 42 dB
# low-frequency anomaly already gates.
FFT_AXIS_OFFSET_UNCERTAINTY_HZ: float = 0.5 * FFT_BIN_WIDTH_HZ

# Duration of one frame, in seconds. Source: acoustics_plan_v2 SS3 -- 1200 frames
# span one 300 s product file, so 300 / 1200 = 0.25 s. Consistent with
# FFT_FILE_SECONDS above by construction, asserted at import below.
FFT_FRAME_SECONDS: float = 0.25

# Sample rate the product was computed at, in Hz. DERIVED from the bin grid
# (2 * n_bins * bin_width = 256000), not assumed: the .fft.gz carries no header
# and no sample-rate field, so this is the only statement of it available, and
# it is the number decision 0002 requires be named rather than assumed. Any
# absolute level compared across instruments must be traced back to this value.
FFT_PRODUCT_FS_HZ: float = 2.0 * FFT_N_BINS * FFT_BIN_WIDTH_HZ

# ---------------------------------------------------------------------------
# The top of the band, as THREE distinct regions -- not one "structural zero".
# ---------------------------------------------------------------------------
# MEASURED on both local fixtures (2,400 frames, B1a adjudication). The single
# "cols 419-511 are zero" statement this replaces was wrong at BOTH ends: it
# called six roll-off columns zero, and it called the DC column exactly zero.
#
#   cols 425-511: 0 of 104,400 nonzero on EACH fixture (208,800 cells checked).
#                 A TRUE structural zero -- any nonzero value there is a reader
#                 or format failure and must raise.
#   cols 419-424: 68 and 74 of 7,200 nonzero, max 6 and 5, per-bin mean <= 0.07.
#                 The TAIL OF THE ANTI-ALIAS SKIRT, not a zero block: the per-bin
#                 mean runs smoothly from 9.8 at bin 405 through 0.28 at 418 to
#                 ~0.001 at 424. There is no boundary at 419; it is one
#                 continuous filter response.
#   col 0:        14 and 8 of 1,200 nonzero, max 3 and 2, mean 0.016 against
#                 16.4 in column 1. NEAR-zero, not zero.
FFT_DC_COL: int = 0
FFT_STRUCTURAL_ZERO_COL0: int = FFT_DC_COL
FFT_STRUCTURAL_ZERO_COLS_HIGH: tuple[int, int] = (425, 511)  # inclusive; HARD zero
FFT_ROLLOFF_TAIL_COLS: tuple[int, int] = (419, 424)  # inclusive; skirt, not zero

# Bin where the anti-alias roll-off begins. 408 * 250 Hz = 102 kHz = 0.4 * fs.
# Kept as a STRUCTURAL landmark only -- see the note under FFT_BIN_WIDTH_HZ for
# why its position is information-free as evidence for the 250 Hz mapping.
FFT_ROLLOFF_ONSET_BIN: int = 408

# Bounds the DC column must respect. MEASURED max 3 and nonzero fraction 1.17%;
# these carry ~2x headroom. Deliberately NOT "exactly zero" -- that assertion is
# too strong for real data and was one of the two deliberately-failing checks.
FFT_DC_COL_MAX_LEVEL: int = 5
FFT_DC_COL_MAX_NONZERO_FRACTION: float = 0.02

# Bounds the roll-off tail must respect. MEASURED max 6, mean 0.015.
FFT_ROLLOFF_TAIL_MAX_LEVEL: int = 10
FFT_ROLLOFF_TAIL_MAX_MEAN_LEVEL: float = 0.05

# Slack allowed when asserting that the per-bin mean level is NON-INCREASING
# across the roll-off skirt (bins FFT_ROLLOFF_ONSET_BIN-3 .. 424). That
# monotonicity is the assertion that actually catches a mis-strided or wrapped
# row -- such a bug moves a bin mean by order 1, not by a rounding step.
#
# MEASURED, AND A CORRECTION TO THE B1a LEDGER: the ledger reports the tail as
# strictly monotonic, but on fixture ...000004 bin 423 has mean 0.0000 and bin
# 424 has mean 0.0008 -- ONE count in ONE frame out of 1200. At the far end of
# the skirt the mean is quantised to multiples of 1/FFT_N_FRAMES and its
# ordering is pure integer-quantisation noise, so strict monotonicity is not a
# property of the data. The tolerance is therefore exactly one count in one
# frame: the smallest nonzero step the product can express. Set to TWICE that
# (2 counts in one frame) rather than exactly one: the measured worst case
# (fixture ...000004, bin 423->424) is 1/FFT_N_FRAMES to the bit, so a
# tolerance of exactly that value depended on the check using strict `>` and
# floats comparing bit-identical -- honest but fragile. 2/FFT_N_FRAMES removes
# the float-equality dependency without changing what the bound is FOR: it is
# still ~3 orders of magnitude below the order-1 step a real stride bug makes.
FFT_ROLLOFF_MONOTONIC_TOL_LEVEL: float = 2.0 / FFT_N_FRAMES

# ---------------------------------------------------------------------------
# The echosounder hump near 38 kHz -- restated. It is a HUMP, not a line, and
# its source is genuinely not at 38.0 kHz.
# ---------------------------------------------------------------------------
# The previous FFT_38KHZ_LINE_BIN = 152 (+/- 1) was NOMINAL-DERIVED (38000/250)
# and could not be reproduced with any statistic on either fixture.
#
# MEASURED ABSOLUTELY ON THE 128 kHz WAV -- no product axis involved: mean
# power-excess centroid 37,634 Hz, stable to 2 Hz across the file's two halves;
# ping-excess centroid 37,576 Hz; peak per-band temporal std at 37,672 Hz with
# 11.0 dB of variation, which is what identifies it as an intermittent source
# rather than a resonance. Extent ~35.0-40.5 kHz, ~4-5 kHz wide, +14 dB over the
# band median. Consistent with an ASL AZFP 38 kHz narrowband ping (a 300 us
# pulse gives a ~3.3 kHz sinc mainlobe) or an EK80 FM sweep on an ES38-class
# transducer (nominal 34-45 kHz). "38 kHz" was always a nominal label.
#
# The hump's job is to REJECT A 2x MAPPING ERROR, where it discriminates by
# ~150 bins. It must NOT be used to adjudicate centre-vs-edge: under edge it
# reads 37.70 kHz and under centre 37.58 kHz, and which is closer flips with the
# background model and the assumed dB scale. Assert it LOOSELY.
FFT_ECHOSOUNDER_HUMP_BINS: tuple[int, int] = (140, 162)  # inclusive

# Where the power-excess centroid of the hump must fall, in (fractional) bins.
# MEASURED 150.36 and 150.17 on the two fixtures -- 0.19 bins apart. Assert on
# the CENTROID, never on the argmax: argmax on a 5 kHz hump quantised to integer
# counts is unstable at +/- 1 bin (it reads 149 on one fixture and 150 on the
# other, from the same source).
FFT_ECHOSOUNDER_CENTROID_BIN_RANGE: tuple[float, float] = (149.0, 152.0)

# The DURABLE form of this landmark: the absolute centre frequency of the source,
# measured on the 128 kHz WAV, +/- 150 Hz. It is independent of the product's
# frequency axis and survives any later change to FFT_AXIS_CONVENTION.
FFT_ECHOSOUNDER_ABS_CENTRE_HZ: float = 37650.0
FFT_ECHOSOUNDER_ABS_CENTRE_TOL_HZ: float = 150.0

# Bins used to estimate the straight-line-in-POWER background under the hump,
# and the span the excess is integrated over. Named because the centroid is only
# reproducible against a stated background model.
FFT_ECHOSOUNDER_BG_LOW_BINS: tuple[int, int] = (120, 135)   # half-open, [lo, hi)
FFT_ECHOSOUNDER_BG_HIGH_BINS: tuple[int, int] = (168, 180)  # half-open
FFT_ECHOSOUNDER_EXCESS_BINS: tuple[int, int] = (135, 170)   # half-open

# Secondary, weaker landmark: the bin of peak per-frame temporal std, which is
# what makes this an echosounder rather than a resonance. MEASURED 151 and 150.
FFT_ECHOSOUNDER_TEMPORAL_STD_SEARCH_BINS: tuple[int, int] = (130, 175)  # half-open
FFT_ECHOSOUNDER_TEMPORAL_STD_ARGMAX_BIN_RANGE: tuple[int, int] = (147, 155)

# ---------------------------------------------------------------------------
# The TWO ceilings. Kept apart on purpose -- they answer different questions.
# ---------------------------------------------------------------------------
# Bins WHOLLY INSIDE the span the pre-deployment calibration file actually
# covers: it states 10 Hz - 51.2 kHz. Bin 0 (0 Hz, the DC column -- itself
# near-zero, see FFT_DC_COL) sits BELOW the stated 10 Hz floor, and bin 205
# (51,250 Hz) sits ABOVE the stated 51,200 Hz ceiling -- both would only be
# admitted by EXTRAPOLATING the calibration curve past its documented span.
# 250 / 250 = 1 -> bin 1 (250 Hz) is the first bin wholly inside; 51000 / 250
# = 204 -> bin 204 (51,000 Hz) is the last. Bins 205-417 carry signal but
# CANNOT be turned into an absolute dB re 1 uPa level without extrapolating.
# Source: acoustics_plan_v2 SS3 / SS7 and
# ICLISTENHF1266_...-hydrophonePreDeploymentCalibration.txt in the sample.
# CHOICE (review-2 [MEDIUM 2]): narrowed from (0, 205) rather than kept with a
# documented extrapolation, because this is the function a cross-team caller
# (optical side) uses to ask "can I state a dB here?" -- the wholly-inside
# range needs no caveat at the call site.
FFT_CALIBRATED_BIN_RANGE: tuple[int, int] = (1, 204)

# B5 PRECONDITION 1 -- the CALIBRATED ceiling. Nothing above this bin can be
# expressed in dB re 1 uPa at all.
FFT_B5_CALIBRATED_CEILING_BIN: int = FFT_CALIBRATED_BIN_RANGE[1]  # 204 == 51.0 kHz

# B5 PRECONDITION 2 -- the UNCALIBRATED / RELATIVE ceiling. Even a purely
# relative statistic must stop here (~102 kHz). NOTHING ABOVE BIN 408 MAY ENTER
# A B5 STATISTIC, for three independent reasons:
#   (i)   bins 409-424 are instrument response (the anti-alias skirt), not ocean;
#   (ii)  409-418 is REAL FILTER SKIRT, not censoring -- only ~half its cells sit
#         at the floor (measured 49.35%/49.19%). 419-424 alone is ~99% at floor
#         (99.06%/98.97%), and 419-511 together is 99.94%/99.93% at floor -- that
#         figure belongs to 419-511, NOT to the whole 409-424 span. Excluding
#         409-418 rests on (i) and (iii), not on floor-censoring;
#   (iii) it is moot for calibrated work anyway, since calibration stops at bin 204.
FFT_B5_RELATIVE_CEILING_BIN: int = FFT_ROLLOFF_ONSET_BIN  # 408 == 102 kHz

# The product's own integer level scale runs [0, 86] in the local sample. The
# TWO ends are NOT equally well established -- see the correction below.
#
# LOWER END (floor, 0): REAL CENSORING, well established. 18.7% of cells sit
# exactly at 0 across the local fixtures -- far too dense a pile-up to be
# ordinary quantisation, and consistent with a hard clip at the scale's floor.
# This is missing data with a KNOWN DIRECTION, and the direction moves with
# ambient -- i.e. it is confounded with the signal a percentile baseline is
# trying to measure. FFT_LEVEL_FLOOR = 0 and the floor-censoring-aware design
# in decision 0015 rest on this alone and are UNAFFECTED by the correction below.
#
# UPPER END (ceiling, 86): AN ASSUMPTION, NOT A MEASUREMENT.
# [CORRECTION, 2026-08-27, pre-merge quality review] This comment previously
# asserted the scale "is clipped into [0, 86], not merely quantised" as fact.
# The only evidence is that 86 is the observed max on two quiet 5-minute
# fixtures, and the tail argues AGAINST a hard clip: counts at 84/85/86 are
# 8/3/3 on one fixture, and the OTHER fixture tops out at 84 with ZERO cells at
# 85 or 86 -- a smooth decay, not the pile-up a real ceiling clip produces (the
# floor shows exactly that pile-up; the ceiling does not). Ten quiet minutes
# cannot distinguish "the scale ceilings at 86" from "the loudest event in
# these two windows happened to reach 86". FFT_LEVEL_CEILING = 86 is kept as a
# CONSERVATIVE assumption (if wrong, the true ceiling is >= 86, so treating 86
# as a possible clip point is the safe direction), but it is not established
# the way the floor is. RESOLUTION PATH: a loud window from B3's corpus pull
# (e.g. a close vessel pass) would settle this immediately -- either genuine
# repeated pile-up at 86 (clip confirmed) or values above 86 appear (clip
# refuted, scale is simply wider than observed so far).
FFT_LEVEL_FLOOR: int = 0
FFT_LEVEL_CEILING: int = 86  # ASSUMED ceiling from the max observed in two quiet
# fixtures -- not confirmed as a hard clip; see the correction above.

# The two axis facts must agree with the file-cadence fact above; a silent
# disagreement here would put every frame timestamp on the wrong grid.
if FFT_N_FRAMES * FFT_FRAME_SECONDS != FFT_FILE_SECONDS:
    raise ValueError(
        f"config inconsistency: FFT_N_FRAMES ({FFT_N_FRAMES}) * FFT_FRAME_SECONDS "
        f"({FFT_FRAME_SECONDS}) = {FFT_N_FRAMES * FFT_FRAME_SECONDS} s, which is not "
        f"FFT_FILE_SECONDS ({FFT_FILE_SECONDS} s)"
    )
