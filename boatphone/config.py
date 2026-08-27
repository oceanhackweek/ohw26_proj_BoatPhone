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
# Three independent confirmations are recorded in accoutics_plan.md (v1)
# SS"What changed since rev. 3"; the anti-alias shoulder onset near bin 408
# (408 * 250 Hz = 102 kHz = 0.4 * 256 kHz) is reproduced on both fixtures here.
FFT_BIN_WIDTH_HZ: float = 250.0

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

# Columns ONC's product generation leaves structurally empty: column 0 (DC) and
# columns 419-511 (above the anti-alias shoulder, where there is no content).
# Source: acoustics_plan_v2 SS3 verification table ("zeros at col 0 and 419-511").
#
# MEASURED CAVEAT, recorded here rather than buried: on the two local fixtures
# these columns are *almost* but not exactly zero. Column 0 has 14 and 8 nonzero
# frames respectively (of 1200), and the 419-511 block has 74 and ~70 nonzero
# cells (of 111,600), all of value 1-5 and all crowded into columns 419-424 --
# i.e. the tail of the roll-off, not a scattering across the block. Treat these
# columns as carrying no usable signal; do NOT assert exact zero on real data.
FFT_STRUCTURAL_ZERO_COL0: int = 0
FFT_STRUCTURAL_ZERO_COLS_HIGH: tuple[int, int] = (419, 511)  # inclusive

# Bin where the 38 kHz echosounder line is expected: 38000 / 250 = 152.
# Source: acoustics_plan_v2 SS3 verification table ("Line at bin 152 +/- 1").
#
# MEASURED CAVEAT: on both local fixtures the line's observed centre is bin
# 150-151, not 152. Mean-over-frames argmax is 150 and 149; per-bin temporal
# std argmax is 150 and 151; the ping-excess power centroid (mean of the 61
# loudest frames minus the mean of the rest) is 150.8 and 151.0. The feature is
# a ~5 kHz-wide hump spanning bins ~140-162, not a single-bin line, so "the peak
# bin" is only defined to about +/- 1. See the B0-2a report: this needs a
# decision record before FFT_38KHZ_LINE_BIN_TOL is relied on as an axis check.
FFT_38KHZ_LINE_BIN: int = 152
FFT_38KHZ_LINE_BIN_TOL: int = 1

# Bins the pre-deployment calibration file actually covers, inclusive: it spans
# 10 Hz - 51.2 kHz, and 51200 / 250 = 204.8 -> bin 205 is the last bin the
# calibration reaches. Bins 206-417 carry signal but CANNOT be turned into an
# absolute dB re 1 uPa level. Source: acoustics_plan_v2 SS3 / SS7 and
# ICLISTENHF1266_...-hydrophonePreDeploymentCalibration.txt in the sample.
FFT_CALIBRATED_BIN_RANGE: tuple[int, int] = (0, 205)

# The two axis facts must agree with the file-cadence fact above; a silent
# disagreement here would put every frame timestamp on the wrong grid.
if FFT_N_FRAMES * FFT_FRAME_SECONDS != FFT_FILE_SECONDS:
    raise ValueError(
        f"config inconsistency: FFT_N_FRAMES ({FFT_N_FRAMES}) * FFT_FRAME_SECONDS "
        f"({FFT_FRAME_SECONDS}) = {FFT_N_FRAMES * FFT_FRAME_SECONDS} s, which is not "
        f"FFT_FILE_SECONDS ({FFT_FILE_SECONDS} s)"
    )
