#!/usr/bin/env python3
"""Build a human review set: acoustic detections laid out for comparison against imagery.

Runnable entry point. DEFINES NOTHING SHARED (CLAUDE.md invariant 6): bands and
thresholds come from `boatphone.config` and `scripts.run_b5_gate`, the reduction
from `boatphone.features`, the scene/corpus join from `boatphone.overpasses`.

WHAT THIS IS FOR. One folder per overpass, holding the acoustic evidence for that
window and an empty slot for the PlanetScope scene, so a human can put the two
side by side and say whether the acoustics and the image agree.

WHY THE SILENT WINDOWS ARE INCLUDED, and why they are not optional. Comparing
detections against imagery fills a 2x2: detector fired / stayed silent, crossed
with vessel present / absent in the scene. A folder containing only detections
can fill the top row alone, which yields "how often was I right when I fired" --
not a false-positive rate and not a detection rate, and neither of the numbers
goal G1 asks for. Every fully-covered window is rendered, including the ones
where the detector found nothing, and the 12 scenes with NO acoustic coverage
appear in the manifest as explicit absences (decisions 0008, 0016) rather than
being silently dropped.

WHAT IS NOT HERE: audio. The `.fft.gz` product is averaged magnitude with no
phase and no waveform, so it cannot be turned back into a recording. Anything
that played would be a synthesis of this code, not a hydrophone recording, and
reviewing it would be reviewing the algorithm. Audio requires a separate WAV
pull from ONC (~115 MB per 5-minute file) for chosen windows.

UNITS, stated at every boundary (decision 0002 SS4): levels are the product's own
UNCALIBRATED dB-like scale, `level_product_db`, never dB re 1 uPa -- ONC states
the product is uncalibrated and undocumented-filtered
(references/ONC_communication.txt, decision 0027). Times are absolute UTC.

Usage:  python3 scripts/build_review_set.py [--out-dir DIR] [--max-windows N]
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # no display on the hub; figures are files, not windows
import matplotlib.pyplot as plt

from boatphone import config, features, fft_io
from boatphone import overpasses as ov
from boatphone.paths import DERIVED_DIR, ensure_dir

import run_b5_gate

# Spectrograms are drawn only up to the B5 relative ceiling (decision 0014).
# Above bin 408 the product is anti-alias skirt and floor-censored tail, i.e.
# instrument response rather than ocean -- plotting it would show a reviewer a
# large dark band and invite them to interpret instrument behaviour as sea.
SPECTROGRAM_CEILING_BIN = config.FFT_B5_RELATIVE_CEILING_BIN


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, cwd=pathlib.Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_window_surface(paths):
    """Concatenate a window's files into one time-ordered surface.

    Nothing is resampled, padded or gap-filled: a window with an interior gap
    simply has fewer frames, and the frame times carry that gap honestly
    (CLAUDE.md invariant 5).
    """
    products = [fft_io.read_fft_gz(p) for p in sorted(paths)]
    levels = np.vstack([p.levels_db for p in products])
    t_utc_s = np.concatenate([p.t_utc_s for p in products])
    order = np.argsort(t_utc_s)
    return levels[order], t_utc_s[order], products[0].freq_hz


def _plot_window(scene_dir, cov, levels, t_utc_s, freq_hz, band_results):
    """Two stacked panels: the spectrogram, and the band traces that drive detection."""
    t0 = cov.overpass.acquired_utc.timestamp()
    t_min = (t_utc_s - t0) / 60.0

    fig, (ax_spec, ax_band) = plt.subplots(
        2, 1, figsize=(13, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.4], "hspace": 0.12})

    # Bin 0 is DC and structurally near-zero (decision 0014); plotting from it
    # spends a large share of the panel on a dead band and squeezes the 1-10 kHz
    # region the detector actually uses. Start at bin 1, the product's lowest
    # representable frequency (config.FFT_LOWEST_REPRESENTABLE_HZ).
    lo_bin = 1
    top_bin = min(SPECTROGRAM_CEILING_BIN, levels.shape[1] - 1)
    mesh = ax_spec.pcolormesh(
        t_min, freq_hz[lo_bin: top_bin + 1] / 1000.0,
        levels[:, lo_bin: top_bin + 1].T, shading="nearest", cmap="magma")
    cbar = fig.colorbar(mesh, ax=ax_spec, pad=0.01)
    cbar.set_label("level (product dB, UNCALIBRATED)")
    ax_spec.set_ylabel("frequency (kHz)")
    ax_spec.set_yscale("log")
    ax_spec.set_ylim(freq_hz[lo_bin] / 1000.0, freq_hz[top_bin] / 1000.0)
    # Mark the two analysis bands on the spectrogram so the reviewer can see
    # which part of the picture the traces below are actually reading.
    for band_hz, colour in ((config.FFT_B5_SMALL_CRAFT_BAND_HZ, "tab:blue"),
                            (config.FFT_B5_SHIP_PROXY_BAND_HZ, "tab:orange")):
        for edge_hz in band_hz:
            ax_spec.axhline(edge_hz / 1000.0, color=colour, lw=0.8, ls="--", alpha=0.55)
    ax_spec.set_title(
        f"{cov.overpass.scene_id}   scene acquired "
        f"{cov.overpass.acquired_utc:%Y-%m-%d %H:%M:%S} UTC   "
        f"({cov.overpass.instrument or 'instrument unknown'}, "
        f"clear {cov.overpass.clear_percent}%)", fontsize=11)

    for band_name, res in band_results.items():
        line, = ax_band.plot(res["t_min"], res["level"], lw=0.8,
                             label=f"{band_name} ({res['band_hz'][0]:.0f}-"
                                   f"{res['band_hz'][1]:.0f} Hz)")
        ax_band.axhline(res["baseline"], color=line.get_color(), ls=":", lw=1.0)
        ax_band.axhline(res["baseline"] + res["threshold"], color=line.get_color(),
                        ls="--", lw=1.0)
        for ev in res["events"]:
            ax_band.axvspan((ev["t_start_utc_s"] - t0) / 60.0,
                            (ev["t_start_utc_s"] + ev["duration_s"] - t0) / 60.0,
                            color=line.get_color(), alpha=0.13, lw=0)

    for ax in (ax_spec, ax_band):
        ax.axvline(0.0, color="cyan", lw=1.6)
    ax_band.set_xlabel("minutes relative to scene acquisition   "
                       "(cyan line = the instant the satellite imaged)")
    ax_band.set_ylabel("band level\n(product dB, UNCALIBRATED)")
    ax_band.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax_band.grid(alpha=0.25)

    n_ev = sum(len(r["events"]) for r in band_results.values())
    verdict = f"{n_ev} candidate event(s)" if n_ev else "DETECTOR SILENT -- no event"
    fig.suptitle(
        f"{verdict}   |   dotted = ambient baseline, dashed = +"
        f"{run_b5_gate.EVENT_EXCESS_THRESHOLD_DB:.0f} dB threshold, shaded = event "
        f"(>= {run_b5_gate.EVENT_MIN_DURATION_S:.0f} s)   |   levels are RELATIVE, "
        "not dB re 1 uPa",
        y=0.985, fontsize=9.5)
    fig.savefig(scene_dir / "acoustics.png", dpi=115, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args(argv)

    run_id = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else DERIVED_DIR / "review" / run_id
    ensure_dir(out_dir)
    windows_dir = ensure_dir(out_dir / "windows")

    overpass_list = ov.load_gate2_overpasses()
    index = ov.corpus_file_index()
    coverages = [ov.window_coverage(o, index) for o in overpass_list]
    summary = ov.coverage_summary(coverages)

    renderable = [c for c in coverages if c.n_files]
    if args.max_windows:
        renderable = renderable[: args.max_windows]

    bands = {
        "small_craft": config.FFT_B5_SMALL_CRAFT_BAND_HZ,
        "ship_proxy": config.FFT_B5_SHIP_PROXY_BAND_HZ,
    }

    rows = []
    print(f"review set -> {out_dir}")
    for cov in renderable:
        scene = cov.overpass
        tag = "full" if cov.is_full else "partial"
        scene_dir = ensure_dir(windows_dir / f"{scene.scene_id}__{tag}")
        ensure_dir(scene_dir / "planet_scene")
        (scene_dir / "planet_scene" / "README.txt").write_text(
            f"Drop the PlanetScope scene for {scene.scene_id} here.\n\n"
            f"Acquired {scene.acquired_utc.isoformat()} (UTC, from the 'acquired' column\n"
            f"of {config.PLANET_GATE2_SURVIVORS_RELPATH}).\n\n"
            "NOTHING HAS BEEN ORDERED YET -- planet_folger_HANDOFF.md records\n"
            "CONFIRM_ORDER = False and there are no pixels in the repo. This slot exists\n"
            "so imagery drops in beside the acoustics without the folder being rebuilt.\n",
            encoding="utf-8")

        levels, t_utc_s, freq_hz = _load_window_surface(cov.paths)
        t0 = scene.acquired_utc.timestamp()

        band_results = {}
        for band_name, band_hz in bands.items():
            series = features.band_level_series(
                fft_io.FftProduct(levels_db=levels, freq_hz=freq_hz, t_utc_s=t_utc_s,
                                  fs_hz=config.FFT_PRODUCT_FS_HZ,
                                  start_utc=cov.window_start_utc, path=cov.paths[0]),
                band_hz)
            found = run_b5_gate.find_events(series.t_utc_s, series.level_product_db)
            band_results[band_name] = {
                "band_hz": band_hz,
                "t_min": (series.t_utc_s - t0) / 60.0,
                "level": series.level_product_db,
                "baseline": found["baseline_product_db"],
                "threshold": found["threshold_db"],
                "events": found["events"],
                "censoring": series.censoring,
                "fraction_at_floor": series.fraction_in_band_at_floor,
                "n_bins": series.n_bins_in_band,
                "decidecade_resolvable": series.decidecade_resolvable,
            }
            for ev in found["events"]:
                rows.append({
                    "scene_id": scene.scene_id,
                    "overpass_id": scene.scene_id,   # B7 schema join key
                    "acquired_utc": scene.acquired_utc.isoformat(),
                    "coverage": tag,
                    "band": band_name,
                    "band_lo_hz": band_hz[0], "band_hi_hz": band_hz[1],
                    "event_peak_utc": _dt.datetime.fromtimestamp(
                        ev["t_peak_utc_s"], _dt.timezone.utc).isoformat(),
                    "minutes_from_acquisition": round(
                        (ev["t_peak_utc_s"] - t0) / 60.0, 3),
                    "duration_s": round(ev["duration_s"], 2),
                    "peak_excess_product_db": round(ev["peak_excess_product_db"], 2),
                    "baseline_product_db": round(found["baseline_product_db"], 2),
                    "fraction_in_band_at_floor": round(
                        series.fraction_in_band_at_floor, 5),
                    "vessel_in_scene": "",      # <- filled by the human reviewer
                    "reviewer_notes": "",
                })

        _plot_window(scene_dir, cov, levels, t_utc_s, freq_hz, band_results)

        n_ev = sum(len(r["events"]) for r in band_results.values())
        if n_ev == 0:
            rows.append({
                "scene_id": scene.scene_id, "overpass_id": scene.scene_id,
                "acquired_utc": scene.acquired_utc.isoformat(), "coverage": tag,
                "band": "both", "band_lo_hz": "", "band_hi_hz": "",
                "event_peak_utc": "", "minutes_from_acquisition": "",
                "duration_s": "", "peak_excess_product_db": "",
                "baseline_product_db": "", "fraction_in_band_at_floor": "",
                "vessel_in_scene": "", "reviewer_notes": "DETECTOR SILENT",
            })

        (scene_dir / "summary.json").write_text(json.dumps({
            "scene_id": scene.scene_id,
            "acquired_utc": scene.acquired_utc.isoformat(),
            "instrument": scene.instrument,
            "clear_percent": scene.clear_percent,
            "coverage": tag,
            "covered_fraction": round(cov.covered_fraction, 4),
            "n_corpus_files": cov.n_files,
            "corpus_files": [p.name for p in sorted(cov.paths)],
            "window_utc": [cov.window_start_utc.isoformat(),
                           cov.window_end_utc.isoformat()],
            "bands": {k: {"band_hz": list(v["band_hz"]),
                          "baseline_product_db": v["baseline"],
                          "threshold_db": v["threshold"],
                          "n_events": len(v["events"]),
                          "n_bins_in_band": v["n_bins"],
                          "fraction_in_band_at_floor": v["fraction_at_floor"],
                          "decidecade_resolvable": v["decidecade_resolvable"],
                          "censoring": v["censoring"]}
                      for k, v in band_results.items()},
            "level_units": "product dB, UNCALIBRATED -- not dB re 1 uPa (decision 0027)",
        }, indent=1, default=str), encoding="utf-8")
        print(f"  {scene.scene_id}  {tag:<7} {cov.n_files} files  {n_ev} event(s)")

    # The scenes with NO acoustic coverage. Explicit absences, not omissions.
    for cov in [c for c in coverages if c.n_files == 0]:
        rows.append({
            "scene_id": cov.overpass.scene_id, "overpass_id": cov.overpass.scene_id,
            "acquired_utc": cov.overpass.acquired_utc.isoformat(),
            "coverage": "NONE", "band": "", "band_lo_hz": "", "band_hi_hz": "",
            "event_peak_utc": "", "minutes_from_acquisition": "", "duration_s": "",
            "peak_excess_product_db": "", "baseline_product_db": "",
            "fraction_in_band_at_floor": "", "vessel_in_scene": "",
            "reviewer_notes": "NO ACOUSTIC DATA -- outside the pulled overpass window",
        })

    manifest = out_dir / "detections.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "produced_by": "scripts/build_review_set.py",
        "corpus_dir": str(ov.ONC_OVERPASS_CORPUS_DIR),
        "n_corpus_files_indexed": len(index),
        "scene_list": config.PLANET_GATE2_SURVIVORS_RELPATH,
        "coverage": {k: v for k, v in summary.items() if not k.endswith("_ids")},
        "bands_hz": {k: list(v) for k, v in bands.items()},
        "event_threshold_db": run_b5_gate.EVENT_EXCESS_THRESHOLD_DB,
        "event_min_duration_s": run_b5_gate.EVENT_MIN_DURATION_S,
        "band_level_statistic": config.FFT_BAND_LEVEL_STATISTIC,
        "axis_convention": config.FFT_AXIS_CONVENTION,
        "axis_offset_uncertainty_hz": config.FFT_AXIS_OFFSET_UNCERTAINTY_HZ,
        "spectrogram_ceiling_bin": SPECTROGRAM_CEILING_BIN,
        "level_units": "product dB, UNCALIBRATED -- not dB re 1 uPa",
        "sampling_conditionality": config.PLANET_SAMPLING_CONDITIONALITY_STATEMENT,
        "no_labels_caveat": (
            "No PlanetScope imagery has been ordered and no optical detections exist. "
            "Nothing in this folder is validated against a vessel label."
        ),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=1), encoding="utf-8")

    (out_dir / "README.md").write_text(f"""# Acoustic review set -- {run_id}

One folder per PlanetScope overpass under `windows/`, holding the acoustic evidence
for that window and an empty `planet_scene/` slot for the image.

## How to read `acoustics.png`

* **Top panel** -- the spectrogram, 0 to
  {SPECTROGRAM_CEILING_BIN * config.FFT_BIN_WIDTH_HZ / 1000:.0f} kHz (bin
  {SPECTROGRAM_CEILING_BIN}, the B5 relative ceiling, decision 0014). Nothing above
  it is plotted: that region is anti-alias filter skirt and floor-censored tail --
  instrument response, not ocean.
* **Bottom panel** -- band level against time, which is what the detector actually
  sees. Dotted = ambient baseline (10th percentile), dashed = the
  +{run_b5_gate.EVENT_EXCESS_THRESHOLD_DB:.0f} dB threshold, shading = a detected
  event (a run of at least {run_b5_gate.EVENT_MIN_DURATION_S:.0f} s above it).
* **Cyan vertical line** -- the instant the satellite imaged. A vessel visible in the
  scene should sit near this line; acoustic energy well away from it is a different
  vessel, or the same one before or after it entered frame.
* A vessel passing the hydrophone makes a **rise and fall over tens of seconds to
  minutes**, peaking at closest approach. A single narrow spike is not a pass.

## Reading the levels

**Uncalibrated.** The y-axis is the product's own dB-like integer scale, NOT dB re
1 uPa, and no conversion exists -- ONC states the `.fft.gz` product is uncalibrated
with undocumented filtering (`references/ONC_communication.txt`,
`docs/decisions/0027-*.md`). Levels are comparable BETWEEN these windows and to
nothing else.

## Filling in `detections.csv`

Two columns are blank for you: **`vessel_in_scene`** and **`reviewer_notes`**.

The windows where the detector stayed **silent** are included deliberately, and they
matter as much as the ones where it fired. Comparing acoustics to imagery fills a
2x2 -- fired/silent against vessel present/absent -- and a set containing only
detections can fill one row of it, which gives neither a false-positive rate nor a
detection rate. Those rows are marked `DETECTOR SILENT`.

## What this set does and does not cover

* {summary['n_full']} scenes with full acoustic coverage, {summary['n_partial']}
  partial, and **{summary['n_zero']} with no acoustic data at all** -- the corpus was
  pulled for the wrong hour (see `docs/plans/b3-close-out-offline.md`). The
  {summary['n_zero']} uncovered scenes are rows in `detections.csv` with
  `coverage = NONE`; they are absences, not omissions.
* **No audio.** The `.fft.gz` product is averaged magnitude with no phase, so it
  cannot be turned back into a recording. Audio needs a separate WAV pull from ONC
  (~115 MB per 5-minute file) for chosen windows.
* **No imagery, and therefore no labels.** Nothing has been ordered
  (`CONFIRM_ORDER = False`). Every "event" here is a *candidate* vessel pass.

Provenance, constants and thresholds: `provenance.json`.
""", encoding="utf-8")

    n_ev = len([r for r in rows if r["duration_s"] != ""])
    print(f"\n{len(renderable)} window folders, {n_ev} events, "
          f"{summary['n_zero']} absences recorded")
    print(f"manifest   -> {manifest}")
    print(f"provenance -> {out_dir / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
