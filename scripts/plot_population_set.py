#!/usr/bin/env python3
"""Population figures: LTSA, SPD and seasonal spectra over ALL 26,666 windows.

Runnable entry point. Reads the artefacts written by
`scripts/build_population_set.py` and renders the figures the methods brief and
its companion notebook established on a 180-window sample.

WHY RE-RENDER WHAT THE NOTEBOOK ALREADY SHOWS. The notebook's versions are built
on 30 windows per season -- 0.7% of the corpus -- and it says so. That is the
right sample for arguing a method is worth having and the wrong one for
publishing a population statistic: it cannot resolve within-season structure at
all, and its sampling error is the same order as the effects being discussed.
The brief's own seasonal spread moved from 8.5 counts at 8-30 files/season to
4.0 at 40, which is that error, visible. These figures use every window.

THE HISTOGRAM IS EXACT, and that is a genuine property of this data rather than
a claim about the estimator. The levels are small integers, so counting them at
1-count resolution IS the distribution: no bin-width choice, no kernel, and no
quantile interpolation anywhere downstream. Every percentile below is read off
the cumulative histogram directly.

Usage:  python3 scripts/plot_population_set.py [--population-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from boatphone import config
from boatphone.paths import DERIVED_DIR

LO_BIN = 1
CEIL_BIN = config.FFT_B5_RELATIVE_CEILING_BIN


def exact_percentile_from_hist(hist, q):
    """Percentile per bin, read off the cumulative integer histogram.

    THE ESTIMATOR IS ``inverted_cdf`` (numpy's name for it): the smallest level
    whose cumulative count reaches q% of the total. Named explicitly because the
    choice is not free -- it disagrees with numpy's ``lower`` by one rank at
    q=95 and with the default ``linear`` almost everywhere, and a percentile
    quoted without its estimator is ambiguous at exactly the resolution these
    figures are read at. Pinned by
    ``check_b5_13_exact_percentile_from_histogram_matches_direct_percentile``.

    EXACT, not interpolated. `hist[b, l]` counts frames at level `l` in bin `b`,
    and the levels are integers, so there is no fractional level to interpolate
    toward; inventing one would fabricate resolution the data does not have.
    """
    total = hist.sum(axis=1, keepdims=True)
    if not np.all(total > 0):
        raise ValueError("a frequency bin has zero counts; cannot take a percentile")
    cum = np.cumsum(hist, axis=1)
    target = (q / 100.0) * total
    return np.argmax(cum >= target, axis=1).astype(float)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--population-dir", default=None)
    args = parser.parse_args(argv)

    if args.population_dir:
        pop_dir = pathlib.Path(args.population_dir)
    else:
        runs = sorted((DERIVED_DIR / "population").glob("*/population.npz"))
        if not runs:
            raise SystemExit(
                "no population run found. Produce one with "
                "`python3 scripts/build_population_set.py` first."
            )
        pop_dir = runs[-1].parent

    data = np.load(pop_dir / "population.npz")
    prov = json.loads((pop_dir / "provenance.json").read_text())
    meta = json.loads((pop_dir / "windows.json").read_text())
    freq_hz = data["freq_hz"]
    years = [int(y) for y in data["years"]]
    med_spectra = data["median_spectra"]
    n = prov["n_windows_processed"]

    f_khz = freq_hz[LO_BIN:CEIL_BIN + 1] / 1000.0
    print(f"population: {n:,} windows, seasons {years[0]}-{years[-1]}")

    # ---- Figure P1: LTSA over the whole corpus -----------------------------
    order = np.argsort([m["start_utc"] for m in meta])
    L = med_spectra[order][:, LO_BIN:CEIL_BIN + 1].T
    yr = np.array([meta[i]["year"] for i in order])

    fig, ax = plt.subplots(figsize=(14, 5))
    m = ax.pcolormesh(np.arange(L.shape[1]), f_khz, L, shading="nearest",
                      cmap="viridis")
    fig.colorbar(m, ax=ax, pad=0.01).set_label(f"median level ({config.FFT_LEVEL_UNIT})")
    for y in years[1:]:
        ax.axvline(np.argmax(yr == y), color="w", lw=1.0)
    for y in years:
        idx = np.flatnonzero(yr == y)
        ax.text(idx.mean(), f_khz[-1] * 0.83, str(y), color="w", ha="center",
                fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("frequency (kHz)")
    ax.set_xlabel(f"window, chronological -- ALL {n:,} windows "
                  "(x is NOT continuous time: a 2.5 h/day strip only)")
    ax.set_title(f"P1 - LTSA, all {n:,} windows, {years[0]}-{years[-1]}", fontsize=11)
    fig.savefig(pop_dir / "P1_ltsa_population.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure P2: SPD, exact ---------------------------------------------
    H = sum(data[f"hist_{y}"] for y in years)[LO_BIN:CEIL_BIN + 1]
    dens = H / H.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(13, 5))
    m = ax.pcolormesh(f_khz, np.arange(H.shape[1]), dens.T, shading="nearest",
                      cmap="viridis", vmax=float(np.percentile(dens, 99.9)))
    fig.colorbar(m, ax=ax, pad=0.01).set_label("probability density (EXACT, 1-count)")
    for q, style, lab in ((5, ":", "L95 (exceeded 95% of the time)"),
                          (50, "-", "L50 (median)"),
                          (95, "--", "L05 (exceeded 5% of the time)")):
        ax.plot(f_khz, exact_percentile_from_hist(H, q), color="w", ls=style,
                lw=1.4, label=lab)
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel(f"level ({config.FFT_LEVEL_UNIT})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_title(
        f"P2 - Spectral probability density, all {n:,} windows. Exact: levels are "
        "integers, so this IS the distribution, not an estimate of it.", fontsize=11)
    fig.savefig(pop_dir / "P2_spd_population.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure P3: seasonal median spectra + the ambient the detector can use
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(13, 9),
                                     gridspec_kw={"hspace": 0.3})
    colours = plt.cm.viridis(np.linspace(0.08, 0.92, len(years)))
    lo_c, hi_c = config.FFT_CONTROL_BAND_HZ
    for ax in (ax_a, ax_b):
        ax.axvspan(lo_c / 1000, hi_c / 1000, color="0.85", zorder=0)
        for band, colour in ((config.FFT_B5_SMALL_CRAFT_BAND_HZ, "tab:blue"),
                             (config.FFT_RAIN_BAND_HZ, "tab:orange")):
            for edge in band:
                ax.axvline(edge / 1000, color=colour, ls="--", lw=0.7, alpha=0.5)
        ax.set_xscale("log")
        ax.grid(alpha=0.25)

    ambient = {}
    for c, y in zip(colours, years):
        Hy = data[f"hist_{y}"][LO_BIN:CEIL_BIN + 1]
        ax_a.plot(f_khz, exact_percentile_from_hist(Hy, 50), color=c, lw=1.2,
                  label=str(y))
        amb = exact_percentile_from_hist(Hy, 5)
        ambient[y] = amb
        ax_b.plot(f_khz, amb, color=c, lw=1.2, label=str(y))
    ax_a.set_title("P3a - median spectrum by season (grey = 51-102 kHz control band; "
                   "blue dashes = small-craft band, orange = rain band)", fontsize=10)
    ax_b.set_title("P3b - L95 AMBIENT by season: the level exceeded 95% of the time. "
                   "This is the per-bin quiet-time floor a detector can subtract "
                   "WITHOUT a window subtracting its own vessel away.", fontsize=10)
    for ax in (ax_a, ax_b):
        ax.set_xlabel("frequency (kHz)")
        ax.set_ylabel(f"level ({config.FFT_LEVEL_UNIT})")
        ax.legend(fontsize=8, ncol=len(years))
    fig.savefig(pop_dir / "P3_seasonal_spectra_population.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    # The seasonal ambient, saved for the detector to consume.
    np.savez_compressed(
        pop_dir / "seasonal_ambient.npz",
        freq_hz=freq_hz, lo_bin=LO_BIN, ceil_bin=CEIL_BIN,
        years=np.asarray(years),
        **{f"l95_{y}": ambient[y] for y in years},
    )

    # ---- Control-band stability, the instrument reference -------------------
    print(f"\ncontrol band ({lo_c/1000:.0f}-{hi_c/1000:.0f} kHz) median level by season "
          f"-- the instrument reference (config.FFT_CONTROL_BAND_EXPECTED_COUNTS = "
          f"{config.FFT_CONTROL_BAND_EXPECTED_COUNTS:g}):")
    for y in years:
        ctrl = [m["band_median"]["control"] for m in meta if m["year"] == y]
        craft = [m["band_median"]["small_craft"] for m in meta if m["year"] == y]
        print(f"  {y}  n={len(ctrl):>5,}   control {np.median(ctrl):5.1f}   "
              f"small_craft {np.median(craft):5.1f}")

    print(f"\nfigures + seasonal_ambient.npz -> {pop_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
