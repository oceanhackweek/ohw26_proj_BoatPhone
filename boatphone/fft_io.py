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
    The `.fft.gz` payload is a bare gzipped ASCII grid of integers: 614,400
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
    bin 1 is 250 Hz (config.FFT_BIN_WIDTH_HZ, sourced there). See the module
    report for measured evidence that ONC may instead intend bin ``k`` to span
    ``[k*dF, (k+1)*dF)`` -- a half-bin (125 Hz) question that is open, not
    settled.

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
    FFT_38KHZ_LINE_BIN,
    FFT_38KHZ_LINE_BIN_TOL,
    FFT_BIN_WIDTH_HZ,
    FFT_CALIBRATED_BIN_RANGE,
    FFT_FRAME_SECONDS,
    FFT_N_BINS,
    FFT_N_FRAMES,
    FFT_PRODUCT_FS_HZ,
    FFT_STRUCTURAL_ZERO_COL0,
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

    The payload is gzipped ASCII: whitespace-separated integer levels, ROW-MAJOR
    (frames vary slowest, bins fastest). The row-major reading is the verified
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

    with gzip.open(path, "rt") as handle:
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
    width -- bins 0-205, i.e. 0 Hz to 51.25 kHz. Above this the product still
    carries signal but no absolute level can be stated for it.
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
    """Count the values that are NOT zero in the columns documented as structural zeros.

    Surfaces, rather than hides, the measured fact recorded in config: column 0
    and columns 419-511 are almost-but-not-exactly zero in the real product.
    Reporting only; this function never modifies anything.
    """
    levels_db = np.asarray(levels_db, dtype=float)
    lo, hi = FFT_STRUCTURAL_ZERO_COLS_HIGH
    col0 = levels_db[:, FFT_STRUCTURAL_ZERO_COL0]
    block = levels_db[:, lo:hi + 1]
    return {
        "col0_nonzero": int(np.count_nonzero(col0)),
        "col0_total": int(col0.size),
        "col0_max": float(col0.max()) if col0.size else float("nan"),
        "high_block_cols": (lo, hi),
        "high_block_nonzero": int(np.count_nonzero(block)),
        "high_block_total": int(block.size),
        "high_block_max": float(block.max()) if block.size else float("nan"),
    }
