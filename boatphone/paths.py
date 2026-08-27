"""Canonical filesystem locations for BoatPhone.

Standard library only -- see boatphone/__init__.py for why.

One definition of every directory, shared by every notebook (CLAUDE.md invariant 6:
no magic strings, one definition, source in a comment). `data/` is immutable
(docs/decisions/0001-raw-data-immutability.md); derived products go to DERIVED_DIR.
"""

from __future__ import annotations

import pathlib

# This file lives at <repo>/boatphone/paths.py, so the repo root is two parents up.
REPO_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent

# Source: CLAUDE.md "Layout" -- data/ holds IMMUTABLE acquisitions.
DATA_DIR: pathlib.Path = REPO_ROOT / "data"

# Source: `ls data/` on the OHW hub. The on-disk acquisition directory name contains
# literal SPACES; it must not be silently normalised to underscores.
SAMPLE_DIR_NAME: str = "Folger Deep Hydrophone Data Sample"
SAMPLE_DIR: pathlib.Path = DATA_DIR / SAMPLE_DIR_NAME

# Source: docs/decisions/0005-raw-acquisition-landing-zone.md -- data/raw/<provider>/
# is the append-only landing zone the acquisition module writes bulk downloads into.
# A4 lands 2.7-8 GB of ONC hydrophone files here with cache+resume; this is the ONE
# definition of that path (CLAUDE.md invariant 6). Gitignored.
RAW_DIR: pathlib.Path = DATA_DIR / "raw"
ONC_RAW_DIR: pathlib.Path = RAW_DIR / "onc"

# Source: docs/decisions/0005 -- the ONE tracked-fixture exception inside data/.
# Small committed test fixtures only (see data/samples/README.md); negated out of
# the .gitignore media rules so a deliberate fixture is actually committable.
SAMPLES_DIR: pathlib.Path = DATA_DIR / "samples"

# Source: CLAUDE.md "Layout" -- derived products go to data/derived/ with provenance.
DERIVED_DIR: pathlib.Path = DATA_DIR / "derived"

# Source: approved A0 plan. Neither of these exists on disk yet; they are the agreed
# destinations, and require_path() is how code says so out loud instead of guessing.
# INTERIM_DIR: scratch between stages (re-creatable). PROCESSED_DIR: analysis-ready.
INTERIM_DIR: pathlib.Path = DATA_DIR / "interim"
PROCESSED_DIR: pathlib.Path = DATA_DIR / "processed"

# Source: CLAUDE.md "Layout".
DOCS_DIR: pathlib.Path = REPO_ROOT / "docs"
SCRIPTS_DIR: pathlib.Path = REPO_ROOT / "scripts"

# How to obtain each input, keyed by the directory it belongs to. Used to turn a
# bare FileNotFoundError into an actionable one (CLAUDE.md invariant: errors surface,
# and "write code that fails clearly when its input is absent").
_HOW_TO_OBTAIN: dict[pathlib.Path, str] = {
    SAMPLE_DIR: (
        "download the ONC/CIOOS Folger Deep ICLISTEN HF1266 sample into "
        f"'{SAMPLE_DIR}' (ONC Oceans 3.0, https://data.oceannetworks.ca)"
    ),
    ONC_RAW_DIR: (
        "run the ONC acquisition stage (A1/A4) to download it into "
        f"'{ONC_RAW_DIR}' (ONC Oceans 3.0, https://data.oceannetworks.ca); "
        "bulk acquisitions are not in git"
    ),
    SAMPLES_DIR: (
        f"small tracked test fixtures live in '{SAMPLES_DIR}' -- see its README; "
        "if one is missing, it has not been committed yet"
    ),
    DERIVED_DIR: "produce it by running the notebook/script that writes this derived product",
    INTERIM_DIR: "produce it by running the upstream stage that writes this interim product",
    PROCESSED_DIR: "produce it by running the upstream stage that writes this processed product",
}

_DEFAULT_HOW_TO_OBTAIN = (
    "acquisitions are not in git -- fetch it from ONC/Planet or re-run the stage that writes it"
)

__all__ = [
    "REPO_ROOT", "DATA_DIR", "SAMPLE_DIR", "SAMPLE_DIR_NAME", "RAW_DIR", "ONC_RAW_DIR",
    "SAMPLES_DIR", "DERIVED_DIR", "INTERIM_DIR", "PROCESSED_DIR", "DOCS_DIR",
    "SCRIPTS_DIR", "require_path", "ensure_dir",
]


def _how_to_obtain(path: pathlib.Path) -> str:
    """The one-line acquisition hint for whichever known directory `path` sits under."""
    for root, hint in _HOW_TO_OBTAIN.items():
        if path == root or root in path.parents:
            return hint
    return _DEFAULT_HOW_TO_OBTAIN


def require_path(p: str | pathlib.Path) -> pathlib.Path:
    """Return `p` as a Path, raising FileNotFoundError if it does not exist.

    Never returns None and never creates the path: an absent input is a stop, not a
    thing to paper over. The message names the path and one line on how to obtain it.
    """
    path = pathlib.Path(p)
    if not path.exists():
        raise FileNotFoundError(f"required path does not exist: {path}\n  how to obtain this: {_how_to_obtain(path)}")
    return path


def ensure_dir(p: str | pathlib.Path) -> pathlib.Path:
    """Create directory `p` (with parents) and return it. The ONLY place data/ grows.

    Deliberately explicit and deliberately rare: nothing in this package creates a
    directory as a side effect of being imported or of constructing a client. A caller
    that is about to WRITE says so by calling this; a caller that is about to READ uses
    require_path(), which raises with an acquisition hint instead.
    """
    path = pathlib.Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path
