# VTUAD facts (A8a gate)

**Dataset:** VTUAD: Vessel Type Underwater Acoustic Data — Domingos, Skelton, Santos.
IEEE DataPort, DOI [10.21227/msg0-ag12](https://doi.org/10.21227/msg0-ag12).

Every row below was established from a primary source that was actually fetched on the stated
date. Nothing here is carried forward from `docs/plans/accoutics_plan.md`,
`docs/plans/proposed_plan_IG.md`, or a previous agent. Where a fact could not be established
remotely, the value is `UNKNOWN` and the source URL shows where the search stopped.

Two facts were established by *measurement* rather than by reading a document, because no
document states them: the sample rate and the populated band were read from a real archived WAV
file from the exact ONC deployment VTUAD is derived from (Fraser River Delta Lower Slope
Hydrophone, deployed 2017-06-24, device `ICLISTENAF2523`), pulled through the ONC Oceans 3.0
archivefile API. The method is recorded in the notes column and reproduced below the table.

## Facts table

| key | value | source | retrieved | notes |
|-----|-------|--------|-----------|-------|
| sample_rate_hz | 32000 (mono, 24-bit PCM) | https://doi.org/10.34943/07de823a-41c0-48b5-9bc3-704513d55ccc | 2026-08-27 | Read from the `fmt ` chunk of `ICLISTENAF2523_20170701T000105.178Z.wav`, fetched from the ONC archivefile API for the VTUAD source deployment. Nyquist = 16000 Hz. Not stated in the DataPort README; measured, not assumed. |
| file_format | WAV audio (24-bit PCM mono) plus CSV metadata, distributed as three ZIP archives | https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data | 2026-08-27 | Landing page "Data Format: *.wav; *.csv"; file listing shows three `.zip` plus three `_sha256.txt`. |
| clip_duration_s | 1 (metadata rows); underlying WAV files are variable-length per AIS encounter interval | https://raw.githubusercontent.com/lucascesarfd/onc_dataset/master/src/config.py | 2026-08-27 | `METADATA_SECONDS=1` in the authors' generation pipeline; pipeline step 12 splits each encounter into 1 s metadata rows via a `sub_init` offset column rather than cutting the audio. So a "clip" is a 1 s window addressed into a longer WAV, not a 1 s file. |
| clip_count | UNKNOWN | https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data | 2026-08-27 | The per-scenario `metadata.csv` carries one row per clip, but it lives inside the subscription-gated ZIPs. Neither the landing page, the README, nor either author repository states a count. Not determinable without downloading. |
| band_populated_hz | 10 to 13000 (usable); flat to about 11700 Hz, anti-alias rolloff -3 dB near 11652 Hz and -20 dB by 13430 Hz, noise floor above about 14500 Hz | https://doi.org/10.34943/07de823a-41c0-48b5-9bc3-704513d55ccc | 2026-08-27 | Welch PSD (Hann, nperseg=8192) of the first 30 s of the same archived file. Nyquist is 16000 Hz but the top ~3 kHz is anti-alias skirt, not signal. |
| label_schema | 5-class vessel type (background, cargo, tanker, tug, passengership) x 3 range-bin scenarios (2000-4000, 3000-5000, 4000-6000 m); no vessel length or size class | https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data | 2026-08-27 | Classes from the abstract and README. Range is a coarse scenario BIN, not a continuous range label. MMSI and a NOAA `class_code` are in the metadata, but vessel length is not; size class would have to be joined in from an external vessel registry. |
| licence | CC BY 4.0 | https://creativecommons.org/licenses/by/4.0/ | 2026-08-27 | Declared in the schema.org JSON-LD `Dataset.license` embedded in the DataPort landing page. Attribution required; publishing derived features is permitted. Note the tension: the licence is open, but *access* is subscription-gated (see download_mechanism). |
| total_size_bytes | 13540000000 (approximate: 3.31 + 5.83 + 4.4 GB as listed, GB read as 10^9) | https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data | 2026-08-27 | Compressed ZIP sizes from the "Dataset Files" listing. The listing does not say whether GB means 10^9 or 2^30; at 2^30 the total is about 14.5e9 bytes. Either way this REFUTES the ~1 TB figure by roughly two orders of magnitude. |
| smallest_downloadable_unit_bytes | 3310000000 (inclusion_2000_exclusion_4000.zip, listed 3.31 GB) | https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data | 2026-08-27 | The three scenarios are independent archives, so the closest scenario alone is a 3.31 GB download. Far below the 150 GB stop-and-ask threshold. |
| download_mechanism | Per-file browser download from IEEE DataPort, gated behind a paid Subscriber login; files served as presigned AWS S3 URLs | https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data | 2026-08-27 | Landing page states "LOGIN TO ACCESS DATASET FILES" and "as a Standard Dataset, these files are only available to Subscribers" (sigadmin, 2025-09-11). No anonymous bulk, torrent or Globus endpoint. An interactive login is required to obtain the URL; an unattended overnight `wget` is NOT possible without first capturing a session. |
| precomputed_features_available | no | https://raw.githubusercontent.com/lucascesarfd/underwater_snd/blob/master/README.md | 2026-08-27 | Only WAV and CSV are published. The authors' training repo generates `mel`, `cqt` and `gammatone` folders locally via `preprocessing_generator.py`; those features are not distributed. |

## How the measured rows were obtained

Sample rate and populated band were not available from any document, so they were read from the
data. The ONC Oceans 3.0 `archivefile/device` API was queried for device `ICLISTENAF2523` over
2017-07-01, which returned the citation "Fraser River Delta Lower Slope Hydrophone Deployed
2017-06-24" — the same instrument, site and deployment start that the VTUAD README names. One
5-minute file (`ICLISTENAF2523_20170701T000105.178Z.wav`, 28,800,044 bytes) was downloaded and its
RIFF `fmt ` chunk parsed: PCM, 1 channel, 32000 Hz, 24 bit, byte rate 96000 — consistent with
300.0 s duration, which confirms the header rather than merely trusting it.

This is an inference about the *source* recordings, not a direct read of a VTUAD WAV. The
generation pipeline concatenates and slices ONC WAVs with `pydub`, which preserves sample rate,
and performs no resampling anywhere in `src/`, so 32000 Hz should carry through. **Verify against
an actual VTUAD file's header (and the metadata `sample_rate` column) as the first step of A8b.**

## Calibration: VTUAD audio is UNCALIBRATED — a real gap in the A8 plan

VTUAD ships raw 24-bit PCM WAV. There is no hydrophone sensitivity, no gain, and no `dB re 1 uPa`
reference anywhere in the distribution: the README's metadata description lists `label`,
`duration_sec`, `file_index`, `sample_rate`, `class_code`, `date`, `MMSI` and CTD variables, and
nothing else. The generation pipeline
([`src/format.py`](https://raw.githubusercontent.com/lucascesarfd/onc_dataset/master/src/format.py))
concatenates ONC WAVs with `pydub` and exports; it never applies a calibration curve. The
pipeline README further describes step 7 as splitting into "1 minute **normalized** pieces of
audio", so a per-segment amplitude normalisation may additionally have destroyed even *relative*
level information across files.

Consequence, stated plainly because the plan does not currently account for it: **Folger levels
are calibrated (dB re 1 uPa) and VTUAD levels are in arbitrary counts.** Band-limiting both sides
to a common band — which is all `models.py` is currently specified to assert — does NOT make them
comparable. Any "transfer gap" measured across an absolute-level feature would be a preprocessing
artefact of the calibration difference, exactly the failure mode
`docs/decisions/0002-time-alignment-and-units.md` exists to prevent, and it would not look wrong.

Two survivable options, both of which need a decision record:

1. **Use level-invariant features only** (per-clip normalised spectra, spectral shape, ratios
   between bands) on both sides, and state that absolute level is deliberately discarded.
2. **Recover VTUAD calibration from ONC.** The source device is `ICLISTENAF2523`; ONC may hold a
   pre-deployment calibration for it as it does for Folger's HF1266. If both sides can be put in
   dB re 1 uPa, absolute-level features become legitimate. If the "normalized" step turns out to
   be a real per-segment gain change, this option dies regardless of the calibration curve.

## Decisions that need recording

- **VTUAD is uncalibrated; the common-band assertion is necessary but not sufficient.** The
  level-invariance rule above belongs in a decision record alongside
  `0002-time-alignment-and-units.md`, not buried in a notebook cell.
- **Common analysis band with Folger is 250 Hz to about 11.7 kHz** — the intersection of Folger's
  >=250 Hz usable support with VTUAD's anti-alias -3 dB corner. This is much narrower than the
  <=51.2 kHz Folger calibrated band the plan assumes.
- **VTUAD carries no vessel size label.** The plan's headline claim, a transfer gap *stratified by
  size class*, cannot be computed from VTUAD labels as distributed. It needs either an external
  MMSI-to-length join or a restructure of the claim.
