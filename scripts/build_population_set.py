#!/usr/bin/env python3
"""One streaming pass over the WHOLE corpus: exact histograms, ambient, LTSA, band levels.

Runnable entry point. DEFINES NOTHING SHARED (CLAUDE.md invariant 6): bands, bin
geometry and the level unit come from `boatphone.config`; the reader from
`boatphone.fft_io`; the deduplicated index from `boatphone.overpasses`.

ALL 26,666 WINDOWS, NOT A SAMPLE. `references/hydrophone-methods-brief.md` and
its companion notebook establish these methods on a 180-window sample and say so.
A sample is the right way to argue for a method and the wrong way to publish a
population statistic: 180 of 26,666 windows is 0.7%, drawn 30 per season, which
cannot resolve within-season structure at all and gives every seasonal figure a
sampling error of the same order as the effects being discussed (the brief's own
8.5-vs-4.0 count discrepancy on the seasonal spread is exactly this). This pass
replaces the estimates with the population.

WHAT IT ACCUMULATES, in ONE decode per file, because decoding is the whole cost:

* **An exact per-bin integer histogram**, per season. The levels are small
  integers, so this is not a binned approximation of the distribution -- it IS
  the distribution, at 1-count resolution. Every percentile, every SPD panel and
  every ambient curve downstream is then computed from the histogram for free
  and exactly, with no quantile interpolation anywhere.
* **A per-window median spectrum**, which is the LTSA.
* **Per-window levels in all four bands** (small craft, ship proxy, rain,
  control), which is what every cross-season statistic reads.
* **Per-window quality fields** -- floor fractions, max level, in-band floor --
  so data-quality claims are population claims too.

Nothing is padded, filled or truncated. A file that fails to decode is COUNTED
and NAMED, never skipped silently (CLAUDE.md invariant 5).

Usage:  python3 scripts/build_population_set.py [--out-dir DIR] [--limit N]
                                                [--workers N]
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import multiprocessing as mp
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from boatphone import config, fft_io
from boatphone import overpasses as ov
from boatphone.paths import DERIVED_DIR, ensure_dir

# The histogram's level axis. Levels are small non-negative integers; decision
# 0026 measured 0-112 on the corpus, and this ceiling is set above that with
# headroom. A level at or above it is CLIPPED INTO THE TOP BIN AND COUNTED, and
# the count is reported -- silently dropping out-of-range values would make the
# distribution wrong in exactly the tail that matters.
LEVEL_AXIS_MAX = 160

BANDS_HZ = {
    "small_craft": config.FFT_B5_SMALL_CRAFT_BAND_HZ,
    "ship_proxy": config.FFT_B5_SHIP_PROXY_BAND_HZ,
    "rain": config.FFT_RAIN_BAND_HZ,
    "control": config.FFT_CONTROL_BAND_HZ,
}


def _band_bins(freq_hz, band_hz):
    """Bins inside a band, WIDENED by the open axis uncertainty (decision 0013)."""
    lo = band_hz[0] - config.FFT_AXIS_OFFSET_UNCERTAINTY_HZ
    hi = band_hz[1] + config.FFT_AXIS_OFFSET_UNCERTAINTY_HZ
    return np.where((freq_hz >= lo) & (freq_hz <= hi))[0]


_FREQ_HZ = fft_io.frequency_axis_hz()
_BAND_IDX = {name: _band_bins(_FREQ_HZ, hz) for name, hz in BANDS_HZ.items()}
_CEIL = config.FFT_B5_RELATIVE_CEILING_BIN


def _process_chunk(path_strs):
    """Decode a CHUNK of windows in one worker, returning ONE accumulated result.

    Chunked deliberately. Returning a per-file histogram would ship a
    (512 x 160) int64 array -- 655 KB -- back through pickle for every one of
    26,666 files, about 17 GB of inter-process traffic, which dominated the pass
    and made it slower than a single-core decode loop. Each worker instead
    accumulates its own chunk and returns once, so the IPC cost is one array per
    worker rather than one per file.
    """
    hist_by_year = {}
    med_spectra, meta, errors = [], [], []
    n_over = 0
    for path_str in path_strs:
        r = _process_one(path_str)
        if "error" in r:
            errors.append(r)
            continue
        year = r["year"]
        if year not in hist_by_year:
            hist_by_year[year] = np.zeros(
                (config.FFT_N_BINS, LEVEL_AXIS_MAX), dtype=np.int64)
        hist_by_year[year] += r.pop("hist")
        med_spectra.append(r.pop("median_spectrum"))
        n_over += r["n_at_or_over_axis_max"]
        meta.append(r)
    return {"hist_by_year": hist_by_year,
            "median_spectra": (np.asarray(med_spectra, dtype=np.float32)
                               if med_spectra else np.zeros((0, config.FFT_N_BINS),
                                                            dtype=np.float32)),
            "meta": meta, "errors": errors, "n_over": n_over}


def _process_one(path_str):
    """Decode one window and return its reductions. Runs in a worker process."""
    path = pathlib.Path(path_str)
    try:
        product = fft_io.read_fft_gz(path)
    except Exception as exc:  # named and returned, never swallowed
        return {"path": path.name, "error": f"{type(exc).__name__}: {exc}"}

    levels = product.levels_db
    clipped = np.clip(levels, 0, LEVEL_AXIS_MAX - 1).astype(np.int64)
    n_over = int((levels >= LEVEL_AXIS_MAX).sum())

    # Exact per-bin histogram for this window: (n_bins, LEVEL_AXIS_MAX).
    # ONE bincount over a flattened (bin, level) index, not 512 separate ones.
    # The per-bin Python loop dominated the whole pass -- several times the cost
    # of the decode it was attached to, which is the wrong place for a
    # full-corpus job to spend its time. Equivalence to the loop is asserted by
    # check_b5_11.
    bin_index = np.repeat(np.arange(config.FFT_N_BINS, dtype=np.int64),
                          clipped.shape[0])
    flat = bin_index * LEVEL_AXIS_MAX + clipped.T.reshape(-1)
    hist = np.bincount(
        flat, minlength=config.FFT_N_BINS * LEVEL_AXIS_MAX
    ).reshape(config.FFT_N_BINS, LEVEL_AXIS_MAX)

    band_median = {n: float(np.median(levels[:, idx])) for n, idx in _BAND_IDX.items()}
    band_p10 = {n: float(np.percentile(np.median(levels[:, idx], axis=1), 10))
                for n, idx in _BAND_IDX.items()}
    band_max = {n: float(np.median(levels[:, idx], axis=1).max())
                for n, idx in _BAND_IDX.items()}

    return {
        "path": path.name,
        "start_utc": product.start_utc.isoformat(),
        "year": product.start_utc.year,
        "month": product.start_utc.month,
        "hist": hist,
        "median_spectrum": np.median(levels, axis=0).astype(np.float32),
        "band_median": band_median,
        "band_p10": band_p10,
        "band_max": band_max,
        "max_level": float(levels.max()),
        "n_at_or_over_axis_max": n_over,
        "frac_floor_all": float((levels <= config.FFT_LEVEL_FLOOR).mean()),
        "frac_floor_inband": float(
            (levels[:, 1:_CEIL + 1] <= config.FFT_LEVEL_FLOOR).mean()),
        "frac_floor_deadband": float(
            (levels[:, 425:] <= config.FFT_LEVEL_FLOOR).mean()),
        "bin0_max": float(levels[:, 0].max()),
    }


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True,
                              cwd=pathlib.Path(__file__).resolve().parent.parent
                              ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N windows (smoke runs)")
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args(argv)

    index = ov.corpus_file_index()
    dropped = ov.corpus_index_duplicates()
    paths = [str(p) for _s, _e, p in index]
    if args.limit:
        paths = paths[: args.limit]

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (
        DERIVED_DIR / "population" / run_id)
    ensure_dir(out_dir)

    print(f"POPULATION PASS -- {len(paths):,} windows "
          f"({len(dropped)} duplicate windows dropped), {args.workers} workers")
    print(f"out -> {out_dir}\n")

    hist_by_year = collections.defaultdict(
        lambda: np.zeros((config.FFT_N_BINS, LEVEL_AXIS_MAX), dtype=np.int64))
    med_spectra, meta, errors = [], [], []
    n_over_axis = 0

    # Chunk so each worker returns one accumulated histogram, not one per file.
    n_chunks = args.workers * 8
    chunk_size = max(1, (len(paths) + n_chunks - 1) // n_chunks)
    chunks = [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]

    with mp.Pool(args.workers) as pool:
        for done, part in enumerate(pool.imap_unordered(_process_chunk, chunks), 1):
            for year, h in part["hist_by_year"].items():
                hist_by_year[year] += h
            if part["median_spectra"].size:
                med_spectra.append(part["median_spectra"])
            meta.extend(part["meta"])
            errors.extend(part["errors"])
            n_over_axis += part["n_over"]
            print(f"  chunk {done}/{len(chunks)}  ({len(meta):,} windows)", flush=True)

    print(f"\ndecoded {len(meta):,} windows, {len(errors)} failed")
    if errors:
        print("  FAILURES (named, not skipped silently):")
        for e in errors[:10]:
            print(f"    {e['path']}: {e['error']}")
    if n_over_axis:
        print(f"  NOTE {n_over_axis:,} cells at/above the level axis max "
              f"({LEVEL_AXIS_MAX}) were clipped into the top bin and counted")

    years = sorted(hist_by_year)
    np.savez_compressed(
        out_dir / "population.npz",
        freq_hz=_FREQ_HZ,
        median_spectra=(np.concatenate(med_spectra) if med_spectra
                        else np.zeros((0, config.FFT_N_BINS), dtype=np.float32)),
        years=np.asarray(years),
        **{f"hist_{y}": hist_by_year[y] for y in years},
    )
    (out_dir / "windows.json").write_text(json.dumps(meta), encoding="utf-8")

    provenance = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "produced_by": "scripts/build_population_set.py",
        "n_windows_indexed": len(index),
        "n_windows_processed": len(meta),
        "n_duplicate_windows_dropped": len(dropped),
        "n_failed": len(errors),
        "failed": [e["path"] for e in errors],
        "n_cells_clipped_at_level_axis_max": n_over_axis,
        "level_axis_max": LEVEL_AXIS_MAX,
        "bands_hz": {k: list(v) for k, v in BANDS_HZ.items()},
        "level_unit": config.FFT_LEVEL_UNIT,
        "axis_convention": config.FFT_AXIS_CONVENTION,
        "axis_offset_uncertainty_hz": config.FFT_AXIS_OFFSET_UNCERTAINTY_HZ,
        "relative_ceiling_bin": _CEIL,
        "sampling_conditionality": config.PLANET_SAMPLING_CONDITIONALITY_STATEMENT,
        "is_population_not_sample": (
            f"All {len(meta):,} unique corpus windows, not a sample. Histograms are "
            "EXACT at 1-count resolution because the levels are integers."
        ),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=1), encoding="utf-8")
    print(f"\nwrote population.npz, windows.json, provenance.json -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
