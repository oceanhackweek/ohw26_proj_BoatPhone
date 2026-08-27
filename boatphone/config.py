"""Shared scientific constants for BoatPhone.

Standard library only -- see boatphone/__init__.py for why.

One definition of every band edge, threshold and dataset fact, shared by every
notebook and by both workstreams (CLAUDE.md invariant 6: no magic numbers, one
definition, source in a comment). Every constant below carries the URL of the
primary source it was read from, and the facts backing the VTUAD_* block are
written up with retrieval dates in docs/vtuad-facts.md.
"""

from __future__ import annotations

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
