"""Report which BoatPhone dependencies are present, and at what version.

Standard library only. Run:

    python3 -m boatphone.env_audit           # report, always exits 0
    python3 -m boatphone.env_audit --strict  # non-zero if a REQUIRED package is absent

Versions resolve via `importlib.metadata`, not `pkg.__version__`: `onc` -- the one
package this segment exists for -- has no `__version__` attribute at all.

A missing OPTIONAL package is not an error here. It gates a later segment, and the
audit says which one, so "pypam is absent" reads as "A5 is blocked", not as a crash
three days from now inside a notebook.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import sys

# Import names (not distribution names) of packages every BoatPhone segment needs.
# Source: approved A0 plan, contract A0.4; cross-checked against environment.yml by
# scripts/checks.py so the two cannot drift apart.
REQUIRED: tuple[str, ...] = (
    "onc",         # ONC Oceans 3.0 client -- hydrophone data acquisition
    "numpy",
    "scipy",       # spectrograms, filtering
    "xarray",
    "pandas",
    "pyproj",      # geodesy: hydrophone <-> vessel ranges
    "pyarrow",     # parquet for data/derived/
    "matplotlib",
)

# Optional packages, each annotated with the segment it gates. Absent is fine now.
# Source: approved A0 plan. Deliberately NOT installed -- the OHW hub environment is
# shared with teammates and none of these is needed before A5.
OPTIONAL: dict[str, str] = {
    "arlpy": "A8 -- propagation modelling (also needs a Bellhop Fortran binary, absent here)",
    "pypam": "A5 -- decidecade (third-octave) band levels",
    "pbp": "A5 -- MBARI pbp HMD products (distribution name: mbari-pbp; imports as 'pbp')",
    "soundfile": "FLAC reads only -- the `flac`/`ffmpeg` CLIs already cover this",
}

# Distribution name to query when it differs from the import name.
DISTRIBUTION_NAMES: dict[str, str] = {"pbp": "mbari-pbp"}

VERSION_UNKNOWN = "version-unknown"

EXIT_OK = 0
EXIT_MISSING_REQUIRED = 1


def package_version(name: str) -> str:
    """Return the installed version of import-name `name` via importlib.metadata.

    Returns VERSION_UNKNOWN when the package imports but carries no distribution
    metadata (e.g. a source checkout on sys.path). Never inspects `__version__`.
    """
    for candidate in (DISTRIBUTION_NAMES.get(name, name), name.replace("_", "-")):
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return VERSION_UNKNOWN


def is_installed(name: str) -> bool:
    """True if `name` is importable, without importing it (find_spec only)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # ValueError: a package present in sys.modules but with __spec__ None.
        return False


def audit(required=REQUIRED, strict: bool = False) -> int:
    """Print the audit table and return a process exit code.

    `required`: iterable of import names that must be present.
    `strict`:   when True, a missing required package makes the return code non-zero.
    Returns EXIT_OK or EXIT_MISSING_REQUIRED. Never raises on a missing package.
    """
    print(f"BoatPhone environment audit -- {sys.executable}")
    print(f"Python {sys.version.split()[0]}\n")

    missing_required = []
    print("REQUIRED:")
    for name in sorted(set(required)):
        if is_installed(name):
            print(f"  OK      {name:<12} {package_version(name)}")
        else:
            missing_required.append(name)
            print(f"  MISSING {name:<12} --")

    print("\nOPTIONAL (absence gates a later segment, it is not an error):")
    for name in sorted(OPTIONAL):
        if is_installed(name):
            print(f"  OK      {name:<12} {package_version(name)}")
        else:
            print(f"  ABSENT  {name:<12} gates: {OPTIONAL[name]}")

    if missing_required:
        print(f"\n{len(missing_required)} required package(s) missing: {', '.join(missing_required)}")
        if strict:
            print("--strict: exiting non-zero.")
            return EXIT_MISSING_REQUIRED
        print("(not --strict: exiting 0)")
    else:
        print("\nAll required packages present.")
    return EXIT_OK


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict", action="store_true",
        help="exit non-zero when a REQUIRED package is missing",
    )
    args = parser.parse_args(argv)
    return audit(required=REQUIRED, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
