"""Band-matching contract for VTUAD <-> Folger cross-domain comparison.

Two hydrophone corpora are compared in this project: the ONC Folger Deep
deployment (icListen HF1266, Barkley Sound) and VTUAD (derived from the ONC
Fraser River Delta Lower Slope deployment, device ``ICLISTENAF2523``). They do
not share a sample rate and they do not share a populated band, so a feature
computed on one is not comparable to a feature computed on the other until both
have been restricted to their **common support**: the intersection of the two
populated bands, further clipped to the Nyquist frequency of the *lower* of the
two sample rates. A frequency that one of the two instruments cannot even
represent cannot be "common".

This module implements that contract and, deliberately, nothing else. It holds
no VTUAD constants: both bands and both sample rates enter as **arguments**.
The real values live in :mod:`boatphone.config` (``VTUAD_SAMPLE_RATE_HZ``,
``VTUAD_BAND_POPULATED_HZ``) and are supplied by the caller, so that this
arithmetic can be proved against fabricated bands independently of whether the
VTUAD facts have landed.

CALIBRATION -- READ THIS BEFORE COMPARING ANY LEVEL
---------------------------------------------------
**Band-matching is necessary but not sufficient.** Folger levels are calibrated
dB re 1 uPa (a pre-deployment calibration file ships with the deployment).
VTUAD audio is **uncalibrated**: raw 24-bit PCM counts, with no sensitivity, no
gain and no reference anywhere in the distribution, and the authors' generation
pipeline additionally exports "normalized" 1-minute segments, so even *relative*
level between VTUAD files may not survive. See the calibration section of
``docs/vtuad-facts.md``.

The consequence, stated plainly: **until an ONC calibration for device
``ICLISTENAF2523`` is recovered (and the per-segment normalisation is shown not
to have destroyed it), level-invariant features are the ONLY safe cross-domain
comparison.** Per-clip normalised spectra, spectral shape, and ratios between
sub-bands are safe; absolute level in dB, band energy, SNR against an absolute
floor, and any threshold expressed in dB re 1 uPa are not. A "transfer gap"
measured on an absolute-level feature across this boundary would be a
calibration artefact that looks exactly like a domain shift -- the silent
corruptor ``docs/decisions/0002-time-alignment-and-units.md`` exists to prevent.
:func:`assert_comparable` refuses that comparison rather than warning about it.

Conventions (decision 0002 SS4, named at every boundary)
--------------------------------------------------------
* frequencies: ``freq_hz``, ``band_hz = (lo_hz, hi_hz)``, ``fs_hz`` -- all in Hz;
* levels: ``level_db_re_1upa`` -- one-sided spectrum levels in dB re 1 uPa;
* :func:`band_limit` operates on an **already-computed one-sided spectrum**. It
  does not window, transform, or scale anything, so it inherits whatever FFT
  scaling and window normalisation the caller stated upstream and cannot fix a
  convention error made there.
* nothing here is time-bearing; all project times are UTC.

Failures RAISE. Nothing in this module warns, clips silently, or returns a
degenerate band as if it were a number.
"""

from __future__ import annotations

import enum
import math

import numpy as np

__all__ = [
    "CalibrationState",
    "FeatureKind",
    "BandMatchError",
    "CalibrationMismatchError",
    "common_support_hz",
    "assert_band_matched",
    "assert_comparable",
    "band_limit",
]


# ---------------------------------------------------------------------------
# Constants (no magic numbers -- source/rationale in the comment)
# ---------------------------------------------------------------------------

# Tolerance for comparing two band edges that should be the *same number*.
# common_support_hz is pure min/max on its four inputs, so the only expected
# difference between a band and the common support it was derived from is
# machine epsilon on the caller's own arithmetic. This is a float-equality
# fudge, not a physical tolerance, and must never be widened to "roughly the
# same band".
BAND_EDGE_TOL_HZ: float = 1e-6

# Floor on the width of a usable common band. One hertz is the bin spacing of a
# 1-second analysis window, so an intersection narrower than this cannot hold a
# single independent spectral bin at any sample rate. This is a *degeneracy*
# floor -- the point below which the answer is arithmetic nonsense -- and NOT a
# statement that a 1 Hz common band is scientifically adequate. Callers with a
# real feature should pass a larger `min_bandwidth_hz`.
MIN_COMMON_BANDWIDTH_HZ: float = 1.0


class CalibrationState(enum.Enum):
    """Declared calibration state of one side of a comparison.

    There is no default and no "unknown-but-probably-fine": a side whose state
    is not declared is a side whose levels cannot be interpreted, and
    :func:`assert_comparable` raises rather than guessing.
    """

    #: Levels are absolute, in dB re 1 uPa, with the instrument's calibration
    #: applied (decision 0002 SS3). Folger, after the calibration gate.
    CALIBRATED_DB_RE_1UPA = "calibrated_db_re_1upa"

    #: Levels are in arbitrary units (raw PCM counts, or counts-derived dB with
    #: no reference). VTUAD as distributed. Absolute levels are meaningless and
    #: even relative levels between files may have been destroyed by the
    #: pipeline's per-segment normalisation.
    UNCALIBRATED_COUNTS = "uncalibrated_counts"


class FeatureKind(enum.Enum):
    """Whether the feature being compared depends on absolute level."""

    #: Value changes if the whole spectrum is scaled by an unknown gain --
    #: band level, SNR against an absolute floor, any dB re 1 uPa threshold.
    ABSOLUTE_LEVEL = "absolute_level"

    #: Value is unchanged by an unknown per-clip gain -- per-clip normalised
    #: spectra, spectral shape, band ratios, spectral centroid.
    LEVEL_INVARIANT = "level_invariant"


class BandMatchError(ValueError):
    """Two spectra were compared without both being limited to the common support."""


class CalibrationMismatchError(ValueError):
    """An absolute-level comparison was attempted across a calibration boundary."""


# ---------------------------------------------------------------------------
# Band arithmetic
# ---------------------------------------------------------------------------

def _validated_band_hz(band_hz, label: str) -> tuple[float, float]:
    """Coerce and validate a (lo_hz, hi_hz) band. Raises on anything malformed."""
    try:
        lo_hz, hi_hz = (float(v) for v in band_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{label} band must be a (lo_hz, hi_hz) pair in Hz, got {band_hz!r}"
        ) from exc
    if not (math.isfinite(lo_hz) and math.isfinite(hi_hz)):
        raise ValueError(f"{label} band has a non-finite edge: {(lo_hz, hi_hz)} Hz")
    if lo_hz < 0.0:
        raise ValueError(f"{label} band has a negative lower edge: {lo_hz} Hz")
    if hi_hz <= lo_hz:
        raise ValueError(
            f"{label} band is not increasing: lo_hz={lo_hz} Hz, hi_hz={hi_hz} Hz"
        )
    return lo_hz, hi_hz


def _validated_fs_hz(fs_hz, label: str) -> float:
    fs = float(fs_hz)
    if not math.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"{label} sample rate must be finite and positive, got {fs_hz!r} Hz")
    return fs


def nyquist_hz(fs_hz: float) -> float:
    """Nyquist frequency in Hz for a recorded sample rate ``fs_hz`` in Hz.

    ``fs_hz`` must be the rate READ FROM THE FILE, never an assumed one
    (decision 0002 SS2).
    """
    return _validated_fs_hz(fs_hz, "sample") / 2.0


def common_support_hz(
    vtuad_band_hz,
    vtuad_fs_hz,
    folger_band_hz,
    folger_fs_hz,
    *,
    min_bandwidth_hz: float = MIN_COMMON_BANDWIDTH_HZ,
):
    """Common analysis band, in Hz, for a VTUAD-vs-Folger comparison.

    The intersection of the two populated bands, clipped to the Nyquist of the
    lower of the two sample rates.

    Parameters
    ----------
    vtuad_band_hz, folger_band_hz : (lo_hz, hi_hz)
        Populated/usable support of each corpus, in Hz.
    vtuad_fs_hz, folger_fs_hz : float
        Sample rate of each corpus, in Hz, read from the data.
    min_bandwidth_hz : float
        Reject an intersection narrower than this (see
        :data:`MIN_COMMON_BANDWIDTH_HZ`).

    Returns
    -------
    (lo_hz, hi_hz) : tuple of float

    Raises
    ------
    ValueError
        If the intersection is empty or degenerately narrow. The message names
        BOTH input bands, both Nyquist limits, and the resulting intersection,
        so the diagnosing human is not left guessing which side was wrong.

    Notes
    -----
    Nothing about VTUAD is hardcoded here. With the real constants from
    ``boatphone.config`` (``VTUAD_SAMPLE_RATE_HZ`` = 32000 Hz,
    ``VTUAD_BAND_POPULATED_HZ`` = (10, 11652) Hz) against Folger's calibrated
    support, this returns roughly 250 Hz -- 11.65 kHz. PROVENANCE CAVEAT: those
    VTUAD values were measured from an archived WAV of the *source* ONC
    deployment (device ICLISTENAF2523), not from a VTUAD distribution file --
    the ZIPs are login-gated. They are INFERRED to carry through the authors'
    pipeline (which does no resampling), not measured from the distribution.
    Verify against a real VTUAD header and the metadata ``sample_rate`` column
    before any published number rests on them.
    """
    vtuad_lo_hz, vtuad_hi_hz = _validated_band_hz(vtuad_band_hz, "VTUAD")
    folger_lo_hz, folger_hi_hz = _validated_band_hz(folger_band_hz, "Folger")
    vtuad_nyquist_hz = nyquist_hz(_validated_fs_hz(vtuad_fs_hz, "VTUAD"))
    folger_nyquist_hz = nyquist_hz(_validated_fs_hz(folger_fs_hz, "Folger"))

    lower_nyquist_hz = min(vtuad_nyquist_hz, folger_nyquist_hz)
    lo_hz = max(vtuad_lo_hz, folger_lo_hz)
    hi_hz = min(vtuad_hi_hz, folger_hi_hz, lower_nyquist_hz)

    width_hz = hi_hz - lo_hz
    if width_hz < min_bandwidth_hz:
        raise ValueError(
            "no usable common support between VTUAD and Folger: "
            f"VTUAD band {tuple(vtuad_band_hz)} Hz at {float(vtuad_fs_hz)} Hz "
            f"(Nyquist {vtuad_nyquist_hz} Hz) and "
            f"Folger band {tuple(folger_band_hz)} Hz at {float(folger_fs_hz)} Hz "
            f"(Nyquist {folger_nyquist_hz} Hz) intersect, after clipping to the "
            f"lower Nyquist ({lower_nyquist_hz} Hz), to ({lo_hz}, {hi_hz}) Hz -- "
            f"a width of {width_hz} Hz, below the {min_bandwidth_hz} Hz minimum. "
            "There is nothing these two corpora can be compared over; this is an "
            "empty/degenerate intersection, not a small one."
        )
    return (lo_hz, hi_hz)


def assert_band_matched(band_hz, common_band_hz, *, label: str = "") -> None:
    """Raise unless ``band_hz`` IS the common support ``common_band_hz``.

    The guard against comparing a feature that was never restricted to the
    intersection -- e.g. a Folger spectrum still carrying its full calibrated
    support up to 51.2 kHz being compared against a VTUAD spectrum that stops at
    Nyquist/anti-alias. Nothing is clipped here: a mismatch is a caller bug, and
    silently fixing it would hide the fact that the two features were computed
    over different amounts of the spectrum.

    NECESSARY BUT NOT SUFFICIENT for a cross-domain comparison -- it says
    nothing about calibration. Use :func:`assert_comparable` for the full guard.

    Raises
    ------
    BandMatchError
        naming both bands and the difference.
    """
    lo_hz, hi_hz = _validated_band_hz(band_hz, label or "input")
    common_lo_hz, common_hi_hz = _validated_band_hz(common_band_hz, "common support")
    dlo_hz = abs(lo_hz - common_lo_hz)
    dhi_hz = abs(hi_hz - common_hi_hz)
    if dlo_hz > BAND_EDGE_TOL_HZ or dhi_hz > BAND_EDGE_TOL_HZ:
        who = f"{label} " if label else ""
        raise BandMatchError(
            f"{who}band {(lo_hz, hi_hz)} Hz is not the common support "
            f"{(common_lo_hz, common_hi_hz)} Hz "
            f"(lower edge differs by {dlo_hz} Hz, upper edge by {dhi_hz} Hz; "
            f"tolerance {BAND_EDGE_TOL_HZ} Hz). Band-limit this side to the "
            "common support with band_limit() before comparing it; do not compare "
            "features computed over different amounts of the spectrum."
        )


def _validated_calibration_state(state, label: str) -> CalibrationState:
    """Coerce a declared calibration state. Raises if it is absent or unrecognised."""
    if state is None:
        raise CalibrationMismatchError(
            f"{label} calibration state is not declared (got None). Every side of a "
            "cross-domain comparison must declare a CalibrationState: Folger is "
            f"{CalibrationState.CALIBRATED_DB_RE_1UPA.value!r} once the calibration "
            f"gate has run, VTUAD as distributed is "
            f"{CalibrationState.UNCALIBRATED_COUNTS.value!r}. There is no default "
            "(decision 0002 SS3)."
        )
    if isinstance(state, CalibrationState):
        return state
    try:
        return CalibrationState(state)
    except ValueError as exc:
        valid = [member.value for member in CalibrationState]
        raise CalibrationMismatchError(
            f"{label} calibration state {state!r} is not a recognised "
            f"CalibrationState; expected one of {valid}"
        ) from exc


def assert_comparable(
    band_a_hz,
    band_b_hz,
    common_band_hz,
    *,
    calibration_a,
    calibration_b,
    feature_kind,
    label_a: str = "side A",
    label_b: str = "side B",
) -> None:
    """Full guard on a cross-domain comparison: band match AND calibration compatibility.

    Both sides must already be limited to ``common_band_hz`` (band-matching), and
    the feature being compared must be interpretable given the two sides'
    declared calibration states.

    The calibration rule, and why it is this strict: an
    :attr:`FeatureKind.ABSOLUTE_LEVEL` feature is only comparable when BOTH
    sides are :attr:`CalibrationState.CALIBRATED_DB_RE_1UPA`. Comparing a
    calibrated dB re 1 uPa level against uncalibrated counts produces a
    difference set by the unknown instrument sensitivity and gain, which is
    indistinguishable from a real domain shift and will not look wrong. Two
    *uncalibrated* sides are refused for the same reason: without a reference,
    their zero points are unrelated, and VTUAD's per-segment normalisation may
    have moved the gain per file as well.
    :attr:`FeatureKind.LEVEL_INVARIANT` features pass regardless -- that is the
    escape hatch, and the only currently safe VTUAD<->Folger comparison.

    Raises
    ------
    BandMatchError
        if either side is not limited to the common support.
    CalibrationMismatchError
        if a calibration state is undeclared/unrecognised, or if an
        absolute-level comparison crosses a calibration boundary.
    """
    assert_band_matched(band_a_hz, common_band_hz, label=label_a)
    assert_band_matched(band_b_hz, common_band_hz, label=label_b)

    state_a = _validated_calibration_state(calibration_a, label_a)
    state_b = _validated_calibration_state(calibration_b, label_b)

    if feature_kind is None:
        raise ValueError(
            "feature_kind is not declared (got None); state whether the feature is "
            f"{FeatureKind.ABSOLUTE_LEVEL.value!r} or "
            f"{FeatureKind.LEVEL_INVARIANT.value!r}"
        )
    kind = feature_kind if isinstance(feature_kind, FeatureKind) else FeatureKind(feature_kind)

    if kind is FeatureKind.LEVEL_INVARIANT:
        return

    uncalibrated = [
        f"{lbl} ({st.value})"
        for lbl, st in ((label_a, state_a), (label_b, state_b))
        if st is not CalibrationState.CALIBRATED_DB_RE_1UPA
    ]
    if uncalibrated:
        raise CalibrationMismatchError(
            f"refusing an {kind.value} comparison over {tuple(common_band_hz)} Hz: "
            f"{label_a} is {state_a.value} and {label_b} is {state_b.value}; "
            f"not calibrated to dB re 1 uPa: {', '.join(uncalibrated)}. "
            "Band-matching alone does NOT make these comparable -- the difference "
            "would be set by unknown sensitivity/gain (and, for VTUAD, a possible "
            "per-segment amplitude normalisation) and would be indistinguishable "
            "from a real domain shift. Use a FeatureKind.LEVEL_INVARIANT feature "
            "(per-clip normalised spectrum, spectral shape, band ratio), or recover "
            "an ONC calibration for device ICLISTENAF2523 first. See "
            "docs/vtuad-facts.md and docs/decisions/0002-time-alignment-and-units.md."
        )


# ---------------------------------------------------------------------------
# Spectrum band-limiting
# ---------------------------------------------------------------------------

def band_limit(freq_hz, level_db_re_1upa, fs_hz, band_hz):
    """Mask an ALREADY-COMPUTED one-sided spectrum to ``band_hz``.

    Parameters
    ----------
    freq_hz : array of float
        One-sided frequency axis in Hz, ascending, as returned by e.g.
        ``numpy.fft.rfftfreq(n, d=1/fs_hz)``.
    level_db_re_1upa : array of float
        Level at each bin, same length as ``freq_hz``. The name states the unit
        (decision 0002 SS4); this function does not apply calibration and cannot
        tell whether it was applied -- see :func:`assert_comparable`.
    fs_hz : float
        Sample rate of the source recording, in Hz, READ FROM THE FILE.
    band_hz : (lo_hz, hi_hz)
        Band to keep, inclusive of both edges (within
        :data:`BAND_EDGE_TOL_HZ`).

    Returns
    -------
    (freq_hz, level_db_re_1upa) : the masked arrays.

    Raises
    ------
    ValueError
        if the two arrays disagree in length, if ``freq_hz`` is not ascending,
        if ``band_hz``'s upper edge exceeds this source's Nyquist -- rather than
        silently clipping to it, which would misrepresent the analysis band
        (decision 0002 SS2) -- or if the mask selects no bins at all.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    level_db_re_1upa = np.asarray(level_db_re_1upa, dtype=float)
    if freq_hz.ndim != 1:
        raise ValueError(f"freq_hz must be 1-D, got shape {freq_hz.shape}")
    if freq_hz.shape != level_db_re_1upa.shape:
        raise ValueError(
            f"freq_hz {freq_hz.shape} and level_db_re_1upa {level_db_re_1upa.shape} "
            "must have the same shape -- never truncate one to match the other"
        )
    if freq_hz.size == 0:
        raise ValueError("freq_hz is empty; there is no spectrum to band-limit")
    if np.any(np.diff(freq_hz) <= 0.0):
        raise ValueError("freq_hz must be strictly ascending (a one-sided spectrum axis)")

    lo_hz, hi_hz = _validated_band_hz(band_hz, "requested")
    source_nyquist_hz = nyquist_hz(fs_hz)
    if hi_hz > source_nyquist_hz + BAND_EDGE_TOL_HZ:
        raise ValueError(
            f"requested band {(lo_hz, hi_hz)} Hz reaches above this source's Nyquist "
            f"({source_nyquist_hz} Hz, from fs_hz={float(fs_hz)} Hz). Refusing to "
            "silently clip: a band the recording cannot represent is a caller error, "
            "and clipping it would report a narrower analysis band than the number "
            "was actually computed over (decision 0002 SS2)."
        )

    mask = (freq_hz >= lo_hz - BAND_EDGE_TOL_HZ) & (freq_hz <= hi_hz + BAND_EDGE_TOL_HZ)
    kept = int(mask.sum())
    if kept == 0:
        raise ValueError(
            f"band {(lo_hz, hi_hz)} Hz selects no bins of a spectrum spanning "
            f"{freq_hz[0]} to {freq_hz[-1]} Hz at {freq_hz[1] - freq_hz[0]} Hz "
            "resolution; there is nothing to compare"
        )
    return freq_hz[mask], level_db_re_1upa[mask]
