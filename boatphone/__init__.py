"""BoatPhone: cross-calibrating optical vessel detection against passive acoustics.

Importing this package must stay cheap and standard-library only: `boatphone.paths`
is the one thing every notebook needs before it knows whether its heavy dependencies
are even installed, so nothing here may import numpy/xarray/onc/... at import time.
(Contract A0.1, enforced by scripts/checks.py.)

All times in this project are UTC (docs/decisions/0002-time-alignment-and-units.md).
"""

# Source: repo-local versioning only; this package is not published to PyPI.
__version__ = "0.1.0"

__all__ = ["paths"]
