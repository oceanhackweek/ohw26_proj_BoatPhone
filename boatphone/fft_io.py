"""Reader and axis vocabulary for the ONC `.fft.gz` hydrophone product.

This module is the ONE place a `.fft.gz` file becomes an array with a stated
frequency axis in Hz and a stated time axis in ABSOLUTE UTC epoch seconds.
Everything downstream -- band levels, the detection threshold, the join against
PlanetScope acquisition times -- inherits its correctness from here, so the
conventions are named out loud rather than assumed
(docs/decisions/0002-time-alignment-and-units.md).

Conventions, stated once
------------------------
Time base
    UTC end to end. Frame times are ABSOLUTE epoch seconds and are named
    ``t_utc_s``. There is no relative-time mode: :func:`time_axis_utc_s` RAISES
    when handed no absolute start. A bare ``0.0, 0.25, 0.5, ...`` axis wearing
    the name ``t_utc_s`` is the single most expensive mistake this project can
    make -- it produces a clean-looking correlation with nothing behind it.

Where the start time comes from
    THE FILENAME, parsed by :func:`boatphone.onc_client.parse_file_coverage`.
    The payload -- gzipped or plain, see :func:`read_fft_gz` -- is a bare ASCII
    grid of integers and nothing else: 614,400
    whitespace-separated values and nothing else -- no header, no timestamp, no
    sample rate. So the filename stamp is not merely the convenient source, it
    is the ONLY source in the file. It is trustworthy to the extent that ONC's
    archive naming is: the stamp is written by the acquisition system, carries
    an explicit ``Z``, and is preserved to the millisecond
    (``ICLISTENHF1266_20260313T000004.000Z.fft.gz``). ``parse_file_coverage``
    refuses a stamp without ``Z`` rather than assuming UTC, and refuses a
    filename carrying another device code. This is an ASSUMPTION WITH A NAMED
    OWNER, not a verified fact: the acoustics plan's own step 2 is to confirm
    the frame-boundary phase by cross-correlating the 38 kHz transient against
    the WAV. Until that lands, treat frame times as accurate to the file start,
    not to the sub-frame phase within it.

Frequency axis
    One-sided, bin ``k`` centred at ``k * FFT_BIN_WIDTH_HZ``, so bin 0 is DC and
    bin 1 is 250 Hz. The 250 Hz bin WIDTH is confirmed on absolute physics (see
    config.FFT_BIN_WIDTH_HZ). The CENTRE convention is not: ONC may instead
    intend bin ``k`` to span ``[k*dF, (k+1)*dF)``, which would move every named
    frequency up by half a bin. That question is OPEN, and it is carried, not
    hidden: ``config.FFT_AXIS_CONVENTION`` names the assumption and
    ``config.FFT_AXIS_OFFSET_UNCERTAINTY_HZ`` (125 Hz) is its price on every
    band edge. Carried SYMMETRICALLY on both edges of the band, because the
    mask must be a superset under either convention -- the raw error is
    one-sided (toward higher frequency, if ONC means edges) but which side is
    unknown, so both edges widen. Use :func:`band_limit_product` rather than
    ``models.band_limit`` directly, so the uncertainty travels with the band.

Censoring
    ``levels_db`` runs ``[0, 86]`` in the local sample. The FLOOR (0) is
    confirmed censoring (18.7% of cells sit exactly there). The CEILING (86)
    is an ASSUMPTION, not a confirmed hard clip -- see
    ``config.FFT_LEVEL_CEILING``'s comment for why. :func:`censoring_report`
    returns the per-window counts at each limit and every band level should be
    reported next to them regardless.

Units
    ``levels_db`` is the product's own integer dB-like level. It is NOT dB re
    1 uPa: no calibration has been applied here, and this module never applies
    one. :func:`calibrated_band_hz` states which bins a calibration CAN cover;
    :func:`assert_calibratable` refuses a band outside it. Use
    ``boatphone.models.assert_comparable`` before comparing levels across
    instruments or time.

Nothing here fills, interpolates or truncates. A file whose value count is not
exactly ``FFT_N_FRAMES * FFT_N_BINS`` raises with both counts named, because a
short file is a data problem and reshaping it to fit would turn that problem
into a plausible number (CLAUDE.md invariant 5).
"""

from __future__ import annotations

import datetime as _dt
import gzip
import pathlib
from dataclasses import dataclass

import numpy as np

from . import models
from .config import (
    GZIP_MAGIC_BYTES,
    FFT_AXIS_CONVENTION,
    FFT_AXIS_OFFSET_UNCERTAINTY_HZ,
    FFT_BIN_WIDTH_HZ,
    FFT_CALIBRATED_BIN_RANGE,
    FFT_FRAME_SECONDS,
    FFT_N_BINS,
    FFT_N_FRAMES,
    FFT_PRODUCT_FS_HZ,
    FFT_DC_COL,
    FFT_ECHOSOUNDER_BG_HIGH_BINS,
    FFT_ECHOSOUNDER_BG_LOW_BINS,
    FFT_ECHOSOUNDER_EXCESS_BINS,
    FFT_ECHOSOUNDER_TEMPORAL_STD_SEARCH_BINS,
    FFT_LEVEL_CEILING,
    FFT_LEVEL_FLOOR,
    FFT_ROLLOFF_ONSET_BIN,
    FFT_ROLLOFF_TAIL_COLS,
    FFT_STRUCTURAL_ZERO_COLS_HIGH,
)
from .onc_client import parse_file_coverage

__all__ = [
    "FftProduct",
    "ToneMismatchError",
    "read_fft_gz",
    "frequency_axis_hz",
    "time_axis_utc_s",
    "calibrated_band_hz",
    "calibrated_bin_range",
    "assert_calibratable",
    "assert_tone_at",
    "band_limit_product",
    "censoring_report",
    "echosounder_centroid_bin",
    "temporal_std_argmax_bin",
    "structural_zero_report",
]


class ToneMismatchError(AssertionError):
    """A tone is not where the caller claimed it was, in frequency, time or level.

    An AssertionError subclass because it is a proving failure, not a data
    problem: the caller asserted something about the pipeline and it is false.
    """


@dataclass(frozen=True)
class FftProduct:
    """One `.fft.gz` file, with its axes named in the units they are in.

    Attributes
    ----------
    levels_db : ndarray, shape (n_frames, n_bins)
        The product's own uncalibrated dB-like level. NOT dB re 1 uPa.
    freq_hz : ndarray, shape (n_bins,)
        One-sided bin centre frequencies, Hz.
    t_utc_s : ndarray, shape (n_frames,)
        Frame start times as ABSOLUTE UTC epoch seconds.
    fs_hz : float
        Sample rate the product was computed at, Hz. Derived from the bin grid
        (see config.FFT_PRODUCT_FS_HZ) -- the file states no sample rate.
    start_utc : datetime.datetime
        Timezone-aware UTC start, from the filename.
    path : pathlib.Path
        The file this came from.
    """

    levels_db: np.ndarray
    freq_hz: np.ndarray
    t_utc_s: np.ndarray
    fs_hz: float
    start_utc: _dt.datetime
    path: pathlib.Path


def frequency_axis_hz(n_bins: int = FFT_N_BINS,
                      bin_width_hz: float = FFT_BIN_WIDTH_HZ) -> np.ndarray:
    """One-sided bin-centre frequency axis, Hz: ``k * bin_width_hz`` for k in 0..n_bins-1.

    DERIVED from its arguments, never a baked-in array: passing a different
    ``bin_width_hz`` must move the axis, otherwise a mapping bug would be
    invisible to every check that uses this function.
    """
    n_bins = int(n_bins)
    bin_width_hz = float(bin_width_hz)
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if not np.isfinite(bin_width_hz) or bin_width_hz <= 0.0:
        raise ValueError(f"bin_width_hz must be finite and positive, got {bin_width_hz}")
    return np.arange(n_bins, dtype=float) * bin_width_hz


def _as_utc_epoch_seconds(frame_start_utc_s, label: str) -> float:
    """Coerce an absolute UTC start to epoch seconds. Raises rather than guessing."""
    if frame_start_utc_s is None:
        raise ValueError(
            f"{label} is None: refusing to build a time axis with no absolute UTC "
            "anchor. A relative 0.0, 0.25, 0.5, ... axis returned under the name "
            "t_utc_s is exactly what docs/decisions/0002-time-alignment-and-units.md "
            "forbids -- it makes an hour-shifted join look like a clean result. "
            "Pass the file's absolute UTC start (read_fft_gz gets it from the "
            "ONC archive filename)."
        )
    if isinstance(frame_start_utc_s, _dt.datetime):
        if frame_start_utc_s.tzinfo is None:
            raise ValueError(
                f"{label} is a NAIVE datetime ({frame_start_utc_s!r}); it states no "
                "time base. Attach timezone.utc explicitly -- assuming UTC is the "
                "assumption decision 0002 exists to forbid."
            )
        return frame_start_utc_s.timestamp()
    if isinstance(frame_start_utc_s, bool):
        raise TypeError(f"{label} must be epoch seconds or an aware datetime, got a bool")
    value = float(frame_start_utc_s)
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite, got {value}")
    return value


def time_axis_utc_s(frame_start_utc_s,
                    n_frames: int = FFT_N_FRAMES,
                    frame_seconds: float = FFT_FRAME_SECONDS) -> np.ndarray:
    """Frame START times as ABSOLUTE UTC epoch seconds.

    ``t_utc_s[i] = frame_start_utc_s + i * frame_seconds``.

    Parameters
    ----------
    frame_start_utc_s : float or aware datetime
        Absolute UTC instant of frame 0. REQUIRED. ``None``, or a naive
        datetime, RAISES -- see :func:`_as_utc_epoch_seconds` for why.
    """
    start_s = _as_utc_epoch_seconds(frame_start_utc_s, "frame_start_utc_s")
    n_frames = int(n_frames)
    frame_seconds = float(frame_seconds)
    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    if not np.isfinite(frame_seconds) or frame_seconds <= 0.0:
        raise ValueError(f"frame_seconds must be finite and positive, got {frame_seconds}")
    return start_s + np.arange(n_frames, dtype=float) * frame_seconds


def read_fft_gz(path, *,
                n_frames: int = FFT_N_FRAMES,
                n_bins: int = FFT_N_BINS,
                bin_width_hz: float = FFT_BIN_WIDTH_HZ,
                frame_seconds: float = FFT_FRAME_SECONDS,
                fs_hz: float = FFT_PRODUCT_FS_HZ) -> FftProduct:
    """Read one ONC `.fft.gz` into an :class:`FftProduct` with named axes.

    CONTAINER IS SNIFFED, NEVER INFERRED FROM THE NAME. The first two bytes are
    compared against ``config.GZIP_MAGIC_BYTES`` (``1f 8b``): a match is read
    through ``gzip.open``, anything else as plain ASCII. The extension is not
    consulted, because it is not reliable here and the corpus is permanently
    MIXED (decision 0024): ONC's archive serves the product as plain ASCII under
    a ``.fft`` name (decision 0022), the bulk pull writes gzip under a ``.fft.gz``
    name, the hand-delivered sample is gzip under ``.fft.gz``, and decision 0001
    forbids ever normalising the 90 already-pulled plain files. Both
    disagreements (``.fft.gz`` holding plain text, ``.fft`` holding gzip) decode
    identically here. This is ONE reader with ONE code path -- a second,
    format-specific reader is what invariant 6 forbids. The function name is a
    misnomer under 0022/0024 and renaming it is deferred there, not here.

    The payload either way is ASCII: whitespace-separated integer levels,
    ROW-MAJOR (frames vary slowest, bins fastest). The row-major reading is the verified
    one -- the column-major reading scatters ~90,000 nonzero values through the
    documented all-but-zero 419-511 block, the row-major reading leaves 74.

    The start time comes from the FILENAME (see the module docstring for why
    that is the only source and how far it can be trusted).

    Raises
    ------
    FileNotFoundError
        if ``path`` does not exist.
    ValueError
        if the file does not hold exactly ``n_frames * n_bins`` values. Nothing
        is padded, dropped or reshaped to fit: a short or long file is a data
        problem, and both counts are named in the message.
    boatphone.onc_client.FilenameParseError
        if the filename does not carry a parseable UTC stamp for our device.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such .fft.gz product: {path}")

    start_utc, _end_utc = parse_file_coverage(path.name)

    with open(path, "rb") as probe:
        first_bytes = probe.read(len(GZIP_MAGIC_BYTES))
    is_gzip = first_bytes == GZIP_MAGIC_BYTES
    opener = gzip.open if is_gzip else open
    with opener(path, "rt", encoding="ascii") as handle:
        tokens = handle.read().split()
    expected = int(n_frames) * int(n_bins)
    if len(tokens) != expected:
        raise ValueError(
            f"{path.name}: holds {len(tokens)} values, expected exactly {expected} "
            f"({n_frames} frames x {n_bins} bins). Refusing to truncate or pad: a "
            "product of the wrong length is a data problem, and reshaping it to fit "
            "would hide it (CLAUDE.md invariant 5)."
        )
    levels_db = np.asarray(tokens, dtype=float).reshape(int(n_frames), int(n_bins))
    if not np.all(np.isfinite(levels_db)):
        raise ValueError(
            f"{path.name}: contains {int((~np.isfinite(levels_db)).sum())} non-finite "
            "value(s); refusing to fill or interpolate them"
        )

    return FftProduct(
        levels_db=levels_db,
        freq_hz=frequency_axis_hz(n_bins=n_bins, bin_width_hz=bin_width_hz),
        t_utc_s=time_axis_utc_s(start_utc, n_frames=n_frames, frame_seconds=frame_seconds),
        fs_hz=float(fs_hz),
        start_utc=start_utc,
        path=path,
    )


def calibrated_bin_range() -> tuple[int, int]:
    """The inclusive bin range the pre-deployment calibration covers (config)."""
    return FFT_CALIBRATED_BIN_RANGE


def calibrated_band_hz(bin_width_hz: float = FFT_BIN_WIDTH_HZ) -> tuple[float, float]:
    """The frequency band the calibration covers, Hz, as ``(lo_hz, hi_hz)`` inclusive.

    Derived from :data:`boatphone.config.FFT_CALIBRATED_BIN_RANGE` and the bin
    width -- bins 1-204, i.e. 250 Hz to 51,000 Hz, WHOLLY INSIDE the
    calibration file's documented 10 Hz - 51,200 Hz span (no extrapolation).
    Bin 0 (below the file's 10 Hz floor) and bin 205 (above its 51,200 Hz
    ceiling) are deliberately excluded rather than admitted by extrapolating
    the sensitivity curve -- see decision 0014 and review-2 [MEDIUM 2]. Above
    bin 204 the product still carries signal but no absolute level can be
    stated for it without extrapolating past the calibration's stated range.
    """
    lo_bin, hi_bin = FFT_CALIBRATED_BIN_RANGE
    axis_hz = frequency_axis_hz(n_bins=hi_bin + 1, bin_width_hz=bin_width_hz)
    return (float(axis_hz[lo_bin]), float(axis_hz[hi_bin]))


def assert_calibratable(band_hz, *, label: str = "requested", bin_width_hz: float = FFT_BIN_WIDTH_HZ) -> None:
    """Raise unless ``band_hz`` lies wholly inside the calibratable support.

    Delegates the comparison to ``boatphone.models.assert_band_matched`` rather
    than inventing a second band comparator (CLAUDE.md invariant 6): the
    requested band is clipped to the calibratable support and then required to
    be UNCHANGED by that clip. A sub-band of the support survives the clip and
    is accepted; a band reaching past bin 205 does not, and raises
    ``models.BandMatchError`` naming both bands.

    Nothing is clipped for the caller -- the clipped band is only the yardstick.
    Requesting an absolute level above 51.25 kHz is a caller error, and quietly
    narrowing the band would report a number computed over a different span
    than the one asked for.
    """
    support_lo_hz, support_hi_hz = calibrated_band_hz(bin_width_hz=bin_width_hz)
    lo_hz, hi_hz = float(band_hz[0]), float(band_hz[1])
    clipped_hz = (max(lo_hz, support_lo_hz), min(hi_hz, support_hi_hz))
    if clipped_hz[1] <= clipped_hz[0]:
        raise models.BandMatchError(
            f"{label} band {(lo_hz, hi_hz)} Hz does not overlap the calibratable "
            f"support {(support_lo_hz, support_hi_hz)} Hz (bins "
            f"{FFT_CALIBRATED_BIN_RANGE[0]}-{FFT_CALIBRATED_BIN_RANGE[1]}) at all"
        )
    models.assert_band_matched(band_hz, clipped_hz, label=label)


def assert_tone_at(levels_db, freq_hz, t_utc_s, *,
                   expected_freq_hz: float,
                   expected_t_utc_s: float,
                   freq_tol_hz: float = FFT_BIN_WIDTH_HZ,
                   time_tol_s: float = FFT_FRAME_SECONDS,
                   expected_level_db: float | None = None,
                   level_tol_db: float | None = None) -> None:
    """Assert the loudest cell of ``levels_db`` is where/when/how loud it is claimed.

    The proving primitive of decision 0002: push a tone of KNOWN frequency,
    time and level through the real axis mapping and require it to come back out
    at the right frequency, the right time and the right level. Returns None on
    success; raises :class:`ToneMismatchError` otherwise, naming what was
    claimed, what was found, and the tolerance.

    Parameters
    ----------
    levels_db : ndarray, shape (n_frames, n_bins)
        Levels on the ``t_utc_s`` x ``freq_hz`` grid.
    freq_hz, t_utc_s : ndarray
        The axes, in Hz and ABSOLUTE UTC epoch seconds.
    expected_freq_hz, expected_t_utc_s : float
        Where the caller claims the tone is.
    freq_tol_hz, time_tol_s : float
        Tolerances. Defaults are one bin and one frame -- the finest the grid
        can resolve. They are tolerances on the GRID, not noise margins.
    expected_level_db, level_tol_db : float or None
        Optional level check. Both must be given together; giving one without
        the other raises, because a level tolerance with no expected level (or
        vice versa) silently checks nothing.
    """
    levels_db = np.asarray(levels_db, dtype=float)
    freq_hz = np.asarray(freq_hz, dtype=float)
    t_utc_s = np.asarray(t_utc_s, dtype=float)
    if levels_db.ndim != 2:
        raise ValueError(f"levels_db must be 2-D (frames x bins), got shape {levels_db.shape}")
    if levels_db.shape != (t_utc_s.size, freq_hz.size):
        raise ValueError(
            f"levels_db {levels_db.shape} does not match its axes "
            f"(t_utc_s {t_utc_s.shape} x freq_hz {freq_hz.shape}); never trim one "
            "to match the other"
        )
    if (expected_level_db is None) != (level_tol_db is None):
        raise ValueError(
            "expected_level_db and level_tol_db must be given together; one without "
            "the other checks nothing while looking like it does"
        )
    if np.any(t_utc_s < 1_000_000_000.0):
        raise ValueError(
            f"t_utc_s starts at {t_utc_s.min()}, which is not an absolute UTC epoch "
            "second -- a relative axis cannot prove a time alignment (decision 0002)"
        )

    frame_idx, bin_idx = np.unravel_index(int(np.argmax(levels_db)), levels_db.shape)
    found_freq_hz = float(freq_hz[bin_idx])
    found_t_utc_s = float(t_utc_s[frame_idx])
    found_level_db = float(levels_db[frame_idx, bin_idx])

    d_freq_hz = abs(found_freq_hz - float(expected_freq_hz))
    d_time_s = abs(found_t_utc_s - float(expected_t_utc_s))
    problems = []
    if d_freq_hz > float(freq_tol_hz):
        problems.append(
            f"FREQUENCY: peak at {found_freq_hz} Hz (bin {bin_idx}), claimed "
            f"{float(expected_freq_hz)} Hz -- off by {d_freq_hz} Hz, tolerance "
            f"{float(freq_tol_hz)} Hz"
        )
    if d_time_s > float(time_tol_s):
        problems.append(
            f"TIME: peak at t_utc_s={found_t_utc_s} (frame {frame_idx}, "
            f"{_dt.datetime.fromtimestamp(found_t_utc_s, _dt.timezone.utc).isoformat()}), "
            f"claimed {float(expected_t_utc_s)} "
            f"({_dt.datetime.fromtimestamp(float(expected_t_utc_s), _dt.timezone.utc).isoformat()})"
            f" -- off by {d_time_s} s, tolerance {float(time_tol_s)} s"
        )
    if expected_level_db is not None:
        d_level_db = abs(found_level_db - float(expected_level_db))
        if d_level_db > float(level_tol_db):
            problems.append(
                f"LEVEL: peak is {found_level_db} dB, claimed {float(expected_level_db)} dB "
                f"-- off by {d_level_db} dB, tolerance {float(level_tol_db)} dB "
                "(a ~3 dB miss is a one-sided/two-sided slip, ~6 dB a "
                "power-vs-amplitude one)"
            )
    if problems:
        raise ToneMismatchError(
            "tone is not where it was claimed to be: " + "; ".join(problems)
        )


def structural_zero_report(levels_db) -> dict:
    """Report the top of the band as the THREE regions it actually is.

    Replaces a single "cols 419-511 are zero" summary that was wrong at both
    ends. Reporting only; this function never modifies anything.

    * ``hard_zero`` -- cols 425-511. A TRUE structural zero: 0 of 104,400 cells
      nonzero on each local fixture. Any nonzero value here is a reader or
      format failure, not a quiet ocean.
    * ``rolloff_tail`` -- cols 419-424. The tail of the anti-alias skirt, NOT a
      zero block: a handful of 1-6 counts, per-bin means decaying smoothly from
      bin 405. There is no boundary at 419.
    * ``dc_col`` -- col 0. NEAR-zero, not zero: 8-14 nonzero frames of 1200.
    * ``rolloff_profile_bin_means`` -- the per-bin mean from just before
      FFT_ROLLOFF_ONSET_BIN through 424. Its NON-INCREASING shape is the part
      that actually catches a mis-strided or wrapped row; the counts above are
      descriptive.
    """
    levels_db = np.asarray(levels_db, dtype=float)
    zero_lo, zero_hi = FFT_STRUCTURAL_ZERO_COLS_HIGH
    tail_lo, tail_hi = FFT_ROLLOFF_TAIL_COLS
    col0 = levels_db[:, FFT_DC_COL]
    hard = levels_db[:, zero_lo:zero_hi + 1]
    tail = levels_db[:, tail_lo:tail_hi + 1]
    profile_lo = FFT_ROLLOFF_ONSET_BIN - 3
    profile = levels_db[:, profile_lo:tail_hi + 1].mean(axis=0)
    return {
        "dc_col": FFT_DC_COL,
        "dc_nonzero": int(np.count_nonzero(col0)),
        "dc_total": int(col0.size),
        "dc_nonzero_fraction": float(np.count_nonzero(col0) / col0.size) if col0.size else float("nan"),
        "dc_max": float(col0.max()) if col0.size else float("nan"),
        "hard_zero_cols": (zero_lo, zero_hi),
        "hard_zero_nonzero": int(np.count_nonzero(hard)),
        "hard_zero_total": int(hard.size),
        "hard_zero_max": float(hard.max()) if hard.size else float("nan"),
        "rolloff_tail_cols": (tail_lo, tail_hi),
        "rolloff_tail_nonzero": int(np.count_nonzero(tail)),
        "rolloff_tail_total": int(tail.size),
        "rolloff_tail_max": float(tail.max()) if tail.size else float("nan"),
        "rolloff_tail_mean": float(tail.mean()) if tail.size else float("nan"),
        "rolloff_profile_first_bin": profile_lo,
        "rolloff_profile_bin_means": profile,
    }


def band_limit_product(freq_hz, levels_db, band_hz, *, fs_hz=FFT_PRODUCT_FS_HZ):
    """Band-limit a `.fft.gz` spectrum, CARRYING the axis-convention uncertainty.

    A thin, deliberate wrapper over :func:`boatphone.models.band_limit` -- it
    reimplements no band arithmetic (CLAUDE.md invariant 6). Its whole job is to
    supply ``axis_offset_uncertainty_hz=FFT_AXIS_OFFSET_UNCERTAINTY_HZ``, so the
    kept support is widened by 125 Hz at each edge and a band edge cannot
    silently exclude a bin that the edge convention would have included.

    Calling ``models.band_limit`` directly on a product spectrum is the bug this
    exists to prevent: it would default to zero uncertainty, which is true of a
    WAV axis and false of this one. Its correctness rests on the assumption
    named in ``config.FFT_AXIS_CONVENTION`` ({convention!r}) being wrong by at
    most half a bin.
    """
    return models.band_limit(
        freq_hz, levels_db, fs_hz, band_hz,
        axis_offset_uncertainty_hz=FFT_AXIS_OFFSET_UNCERTAINTY_HZ,
    )


band_limit_product.__doc__ = band_limit_product.__doc__.format(
    convention=FFT_AXIS_CONVENTION
)


def censoring_report(levels_db) -> dict:
    """Per-window counts of cells sitting AT the 0 floor and AT the 86 ceiling.

    A B5 PRECONDITION, not a diagnostic nicety. The product's integer scale's
    FLOOR (0) is confirmed censoring (18.7% of cells sit exactly there); its
    CEILING (86) is currently an ASSUMPTION, not a confirmed hard clip (see
    ``config.FFT_LEVEL_CEILING``). Either way, a mean or a percentile computed
    over a window touching either limit is biased toward it by an amount that
    is not boundable from the data alone -- and, for the floor, the amount
    MOVES WITH AMBIENT, i.e. is confounded with the very signal a threshold is
    trying to detect.

    3 cells reach 86 on a QUIET window of the local sample; report the counts
    regardless of whether the ceiling proves to be a hard clip, since a window
    touching it is at minimum informative about how close to the observed max
    it got.

    Reporting only -- nothing here modifies, clips or fills anything. Report
    these counts ALONGSIDE every band level; a band level whose window has
    ceiling hits is a lower bound, not a measurement.
    """
    levels_db = np.asarray(levels_db, dtype=float)
    n = int(levels_db.size)
    at_floor = int(np.count_nonzero(levels_db <= FFT_LEVEL_FLOOR))
    at_ceiling = int(np.count_nonzero(levels_db >= FFT_LEVEL_CEILING))
    return {
        "n_cells": n,
        "level_floor": FFT_LEVEL_FLOOR,
        "level_ceiling": FFT_LEVEL_CEILING,
        "n_at_floor": at_floor,
        "n_at_ceiling": at_ceiling,
        "fraction_at_floor": (at_floor / n) if n else float("nan"),
        "fraction_at_ceiling": (at_ceiling / n) if n else float("nan"),
    }


def echosounder_centroid_bin(levels_db) -> float:
    """Power-weighted centroid, in fractional bins, of the ~38 kHz echosounder hump.

    The statistic the axis check asserts on. Stated once, here, so the check and
    any notebook compute the SAME number:

    1. ``P = mean(10 ** (levels_db / 10), axis=0)`` -- the mean in POWER, not
       in the product's dB-like counts. Averaging counts would weight the
       quiet frames like the pings.
    2. A background that is a STRAIGHT LINE IN POWER between
       ``median(P[120:135])`` and ``median(P[168:180])``, each placed at the
       mean bin index of its own window.
    3. ``excess = clip(P[135:170] - background, 0, None)``.
    4. The power-weighted centroid of that excess.

    CENTROID, NEVER ARGMAX. The feature is a ~5 kHz-wide hump quantised to
    integer counts; its argmax is unstable at +/- 1 bin between two fixtures
    five minutes apart (149 vs 150), while this centroid moved 0.19 bins
    (150.36 vs 150.17). An argmax assertion would be a coin flip wearing the
    costume of a measurement.

    Raises
    ------
    ValueError
        if the excess is everywhere zero -- there is no hump to locate, and
        returning a background-weighted centroid would look like an answer.
    """
    levels_db = np.asarray(levels_db, dtype=float)
    if levels_db.ndim != 2:
        raise ValueError(f"levels_db must be 2-D (frames x bins), got {levels_db.shape}")
    power = np.mean(10.0 ** (levels_db / 10.0), axis=0)

    lo_a, lo_b = FFT_ECHOSOUNDER_BG_LOW_BINS
    hi_a, hi_b = FFT_ECHOSOUNDER_BG_HIGH_BINS
    ex_a, ex_b = FFT_ECHOSOUNDER_EXCESS_BINS
    bg_lo = float(np.median(power[lo_a:lo_b]))
    bg_hi = float(np.median(power[hi_a:hi_b]))
    x_lo = float(np.mean(np.arange(lo_a, lo_b)))
    x_hi = float(np.mean(np.arange(hi_a, hi_b)))
    bins = np.arange(ex_a, ex_b, dtype=float)
    background = bg_lo + (bg_hi - bg_lo) * (bins - x_lo) / (x_hi - x_lo)
    excess = np.clip(power[ex_a:ex_b] - background, 0.0, None)

    total = float(excess.sum())
    if total <= 0.0:
        raise ValueError(
            f"no power excess over the straight-line background in bins {ex_a}-{ex_b}: "
            "there is no echosounder hump in this window to take a centroid of"
        )
    return float((excess * bins).sum() / total)


def temporal_std_argmax_bin(levels_db) -> int:
    """Bin of peak per-bin temporal std within the echosounder search window.

    The SECONDARY, weaker landmark: intermittency is what makes this source an
    echosounder rather than a resonance. Asserted loosely (+/- several bins) --
    it is a confirmation of character, not a position measurement.
    """
    levels_db = np.asarray(levels_db, dtype=float)
    lo, hi = FFT_ECHOSOUNDER_TEMPORAL_STD_SEARCH_BINS
    return int(lo + np.argmax(levels_db.std(axis=0)[lo:hi]))
