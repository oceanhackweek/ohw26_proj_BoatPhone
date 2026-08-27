"""B5 band levels and excess-over-ambient, computed on the `.fft.gz` surface.

The library half of segment B5 (`docs/plans/acoustics_plan_v2.md` SS5), which
decision 0012 promoted from cross-check to the project's PRIMARY vessel-presence
detector. Runnable entry points live in `scripts/`; this module defines the
method and nothing operational (CLAUDE.md invariant 6).

EVERY LEVEL HERE IS RELATIVE AND UNCALIBRATED, and that is now a settled fact
rather than a limitation to be worked around later. ONC's product owner states
(`references/ONC_communication.txt`) that the `.fft.gz` product is a proprietary
Ocean Sonics format, "essentially uncalibrated unless you have all the metadata
from Ocean Sonics, which I don't believe ONC currently publishes", with
filtering applied that they cannot document. The 256 kHz sensitivity curve they
can supply is explicitly "not applicable to those files". So:

* no function here returns, or can be made to return, dB re 1 uPa;
* the unit is the product's own dB-like integer scale, named
  ``level_product_db`` at every boundary, never ``level_db`` and never
  ``level_db_re_1upa`` (decision 0002 SS4);
* two levels are comparable only if computed by the same function over the same
  band -- carried explicitly on :class:`BandLevelSeries` so it is visible rather
than assumed.

WHAT IS DELIBERATELY NOT HERE: any absolute-level claim, any decidecade band
below ``config.FFT_DECIDECADE_MIN_CENTRE_HZ``, and any hybrid-millidecade
claim at all (decision 0010 SS3).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .config import (
    FFT_B5_RELATIVE_CEILING_BIN,
    FFT_BAND_LEVEL_STATISTIC,
    FFT_BIN_WIDTH_HZ,
    FFT_DECIDECADE_MIN_CENTRE_HZ,
    FFT_LEVEL_FLOOR,
    FFT_LOWEST_REPRESENTABLE_HZ,
)
from .fft_io import band_limit_product, censoring_report

__all__ = [
    "BandLevelSeries",
    "BandExcess",
    "UnrepresentableBandError",
    "relative_ceiling_hz",
    "assert_band_representable",
    "band_level_series",
    "is_decidecade_resolvable",
    "ambient_baseline_product_db",
    "band_excess",
]


class UnrepresentableBandError(ValueError):
    """A requested band reaches outside what the product can represent.

    A ValueError rather than a silent clip: a caller asking for a band this
    product does not carry has a wrong model of the instrument, and returning
    the nearest available band instead would answer a question they did not ask
    and hide the mistake (CLAUDE.md invariant 5).
    """


def relative_ceiling_hz(bin_width_hz: float = FFT_BIN_WIDTH_HZ) -> float:
    """Highest frequency admissible in ANY B5 statistic, even a relative one.

    Derived from :data:`config.FFT_B5_RELATIVE_CEILING_BIN` (bin 408) rather
    than restated, so a change to that constant moves this too. Above it the
    product is anti-alias filter skirt and floor-censored tail -- instrument
    response, not ocean (decision 0014).
    """
    return float(FFT_B5_RELATIVE_CEILING_BIN) * float(bin_width_hz)


def assert_band_representable(band_hz, *, label: str = "requested",
                              bin_width_hz: float = FFT_BIN_WIDTH_HZ) -> None:
    """Raise unless ``band_hz`` lies wholly inside the product's usable support.

    Two independent limits, checked separately so the error message says which
    one was hit:

    * the FLOOR, :data:`config.FFT_LOWEST_REPRESENTABLE_HZ` (bin 1, 250 Hz).
      Below it there is bin 0, which is DC and structurally near-zero. This is
      the check that catches an attempt to implement ONC's ~100 Hz
      recommendation literally -- see ``config.FFT_B5_SHIP_PROXY_BAND_HZ``.
    * the CEILING, :func:`relative_ceiling_hz` (bin 408, ~102 kHz), decision
      0014's hard rule for B5.

    Note this is the RELATIVE support. It is deliberately NOT
    ``fft_io.assert_calibratable``, which is a stricter, different question
    (bins 1-204, "can I state a dB re 1 uPa here?") whose answer is now always
    effectively no for this product.
    """
    lo_hz, hi_hz = (float(band_hz[0]), float(band_hz[1]))
    if not lo_hz < hi_hz:
        raise UnrepresentableBandError(
            f"{label} band {(lo_hz, hi_hz)} Hz is not ascending"
        )
    floor_hz = float(FFT_LOWEST_REPRESENTABLE_HZ)
    if lo_hz < floor_hz:
        raise UnrepresentableBandError(
            f"{label} band starts at {lo_hz} Hz, below the lowest frequency this "
            f"product represents ({floor_hz} Hz = bin 1 at {bin_width_hz} Hz/bin). "
            "Bin 0 is DC and structurally near-zero, so there is no sub-250 Hz "
            "content to band-limit to at any width. If this came from ONC's "
            "~100 Hz suggestion, see config.FFT_B5_SHIP_PROXY_BAND_HZ for the "
            "nearest reachable proxy and why it is not the same band."
        )
    ceiling_hz = relative_ceiling_hz(bin_width_hz)
    if hi_hz > ceiling_hz:
        raise UnrepresentableBandError(
            f"{label} band ends at {hi_hz} Hz, above the B5 relative ceiling "
            f"({ceiling_hz} Hz = bin {FFT_B5_RELATIVE_CEILING_BIN}). Nothing above "
            "that bin may enter a B5 statistic even relatively: it is anti-alias "
            "filter skirt and floor-censored tail, i.e. instrument response, not "
            "ocean (decision 0014)."
        )


def is_decidecade_resolvable(centre_hz: float) -> bool:
    """Whether a decidecade band at ``centre_hz`` is resolved by 250 Hz bins.

    True only above :data:`config.FFT_DECIDECADE_MIN_CENTRE_HZ` (decision 0010
    SS3). Callers use this to decide what to CALL a level, not whether to
    compute one: below the threshold a level is still computable, it just is not
    a standards-compliant decidecade band level and must not be labelled as one.
    """
    return float(centre_hz) > float(FFT_DECIDECADE_MIN_CENTRE_HZ)


@dataclasses.dataclass(frozen=True)
class BandLevelSeries:
    """Per-frame band level through one window, with its censoring context.

    Attributes
    ----------
    t_utc_s : ndarray, shape (n_frames,)
        Frame start times, ABSOLUTE UTC epoch seconds (decision 0002).
    level_product_db : ndarray, shape (n_frames,)
        Band level per frame in the PRODUCT'S OWN uncalibrated dB-like scale.
        Not dB re 1 uPa and not convertible to it.
    band_hz : (float, float)
        The band requested. Note the support actually kept is WIDER by
        ``config.FFT_AXIS_OFFSET_UNCERTAINTY_HZ`` at each edge, because
        ``fft_io.band_limit_product`` carries the open centre-vs-edge axis
        question (decision 0013) rather than assuming it away.
    n_bins_in_band : int
        How many bins survived the band limit. A band-matching bug usually shows
        up here first, as a count that is not what the band width implies.
    statistic : str
        ``config.FFT_BAND_LEVEL_STATISTIC`` -- carried on the result so two
        series cannot be compared without the reduction being visible.
    censoring : dict
        ``fft_io.censoring_report`` over the IN-BAND cells only. Report this
        beside every level (decision 0015).
    fraction_in_band_at_floor : float
        Fraction of in-band cells sitting exactly at ``FFT_LEVEL_FLOOR``. The
        median reduction is unbiased only while this is below 0.5; above that
        the level is a floor artefact, and this field is what makes that
        checkable instead of assumed.
    decidecade_resolvable : bool
        Whether the band centre clears ``FFT_DECIDECADE_MIN_CENTRE_HZ``. When
        False the level is a RAW-BIN level and must be labelled as such
        (decision 0010 SS3).
    """

    t_utc_s: np.ndarray
    level_product_db: np.ndarray
    band_hz: tuple[float, float]
    n_bins_in_band: int
    statistic: str
    censoring: dict
    fraction_in_band_at_floor: float
    decidecade_resolvable: bool


def band_level_series(product, band_hz) -> BandLevelSeries:
    """Reduce one product window to a band level PER FRAME, not one spectrum.

    The time axis is the point (`acoustics_plan_v2` SS5 B5): 1200 frames at
    0.25 s across a window give level-versus-time through closest point of
    approach -- the CPA time, the rise and fall slope, and the peak width --
    which jointly constrain range and speed in a way a single averaged spectrum
    cannot. Averaging the window to one spectrum first destroys exactly the
    structure the segment exists to use.

    The reduction across bins is ``config.FFT_BAND_LEVEL_STATISTIC``; see that
    constant for why it is a median of the product's dB-like values and not an
    energy sum.

    Parameters
    ----------
    product : fft_io.FftProduct
        One window as read by ``fft_io.read_fft_gz``.
    band_hz : (lo_hz, hi_hz)
        Band to score. Validated against the product's representable support
        first; a band reaching outside it raises rather than being clipped.
    """
    assert_band_representable(band_hz, label="band_level_series")

    freq_hz = np.asarray(product.freq_hz, dtype=float)
    levels = np.asarray(product.levels_db, dtype=float)
    if levels.ndim != 2:
        raise ValueError(
            f"expected a (n_frames, n_bins) surface, got shape {levels.shape}"
        )
    if levels.shape[1] != freq_hz.size:
        raise ValueError(
            f"frequency axis has {freq_hz.size} bins but the surface has "
            f"{levels.shape[1]}; refusing to truncate either to match"
        )

    # Band-limit ONE representative row to learn the mask, then apply that same
    # mask to every frame. band_limit_product supplies the axis-convention
    # uncertainty (decision 0013); calling models.band_limit directly here would
    # default it to zero, which is the bug that wrapper exists to prevent.
    kept_freq_hz, _kept = band_limit_product(freq_hz, levels[0], band_hz)
    if kept_freq_hz.size == 0:
        raise UnrepresentableBandError(
            f"band {tuple(float(b) for b in band_hz)} Hz kept no bins at "
            f"{FFT_BIN_WIDTH_HZ} Hz/bin -- it is narrower than one bin"
        )
    in_band = np.isin(freq_hz, kept_freq_hz)
    n_bins_in_band = int(in_band.sum())

    band_cells = levels[:, in_band]
    level_product_db = np.median(band_cells, axis=1)

    at_floor = int(np.count_nonzero(band_cells <= FFT_LEVEL_FLOOR))
    fraction_at_floor = at_floor / band_cells.size if band_cells.size else float("nan")

    centre_hz = 0.5 * (float(band_hz[0]) + float(band_hz[1]))
    return BandLevelSeries(
        t_utc_s=np.asarray(product.t_utc_s, dtype=float),
        level_product_db=level_product_db,
        band_hz=(float(band_hz[0]), float(band_hz[1])),
        n_bins_in_band=n_bins_in_band,
        statistic=FFT_BAND_LEVEL_STATISTIC,
        censoring=censoring_report(band_cells),
        fraction_in_band_at_floor=float(fraction_at_floor),
        decidecade_resolvable=is_decidecade_resolvable(centre_hz),
    )


def ambient_baseline_product_db(level_product_db, percentile: float = 10.0) -> float:
    """A low percentile of a band-level series: the ambient reference.

    A PERCENTILE, not a mean or a minimum. The mean is pulled up by the very
    vessel passes being detected, and the minimum is a single floor-censored
    sample. The 10th percentile is low enough to sit in ambient even when a
    pass occupies a good fraction of the window, and high enough not to be the
    censoring floor itself.

    CENSORING CAVEAT, carried not hidden (decision 0015): the product's floor at
    ``FFT_LEVEL_FLOOR`` is real censoring whose density MOVES WITH AMBIENT. When
    a quiet window pushes in-band cells onto the floor, this baseline is biased
    UPWARD relative to the true ambient by an amount not boundable from the data
    alone -- which shrinks the measured excess. The bias direction is therefore
    CONSERVATIVE for detection: it makes a real vessel look weaker, never a
    quiet window look louder. Check it with
    ``BandLevelSeries.fraction_in_band_at_floor``.
    """
    level_product_db = np.asarray(level_product_db, dtype=float)
    if level_product_db.size == 0:
        raise ValueError("cannot take an ambient baseline of an empty series")
    if not 0.0 < float(percentile) < 100.0:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    return float(np.percentile(level_product_db, float(percentile)))


@dataclasses.dataclass(frozen=True)
class BandExcess:
    """Excess of a band level over its own ambient baseline, in product dB.

    ``peak_excess_product_db`` is the detection statistic; ``t_peak_utc_s`` is
    where in the window it happened, which is what makes a time-shift null
    meaningful (a real pass moves when the labels move; a processing artefact
    does not).
    """

    band_hz: tuple[float, float]
    baseline_product_db: float
    baseline_percentile: float
    excess_product_db: np.ndarray
    peak_excess_product_db: float
    t_peak_utc_s: float
    fraction_in_band_at_floor: float
    n_bins_in_band: int
    decidecade_resolvable: bool


def band_excess(series: BandLevelSeries, *, percentile: float = 10.0,
                baseline_product_db: float | None = None) -> BandExcess:
    """Excess-over-ambient for one band-level series.

    ``baseline_product_db`` may be supplied to score a window against an
    ambient estimated ELSEWHERE -- from matched-hour negative windows, say.
    That is the honest way to score a window that may be entirely occupied by a
    pass: taking its baseline from itself would subtract the signal off.
    """
    if baseline_product_db is None:
        baseline_product_db = ambient_baseline_product_db(
            series.level_product_db, percentile=percentile
        )
    excess = np.asarray(series.level_product_db, dtype=float) - float(baseline_product_db)
    peak_idx = int(np.argmax(excess))
    return BandExcess(
        band_hz=series.band_hz,
        baseline_product_db=float(baseline_product_db),
        baseline_percentile=float(percentile),
        excess_product_db=excess,
        peak_excess_product_db=float(excess[peak_idx]),
        t_peak_utc_s=float(series.t_utc_s[peak_idx]),
        fraction_in_band_at_floor=series.fraction_in_band_at_floor,
        n_bins_in_band=series.n_bins_in_band,
        decidecade_resolvable=series.decidecade_resolvable,
    )
