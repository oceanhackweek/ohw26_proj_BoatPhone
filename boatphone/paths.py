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

# Source: docs/decisions/0020 -- B3's bulk pull covers ONLY the 2.5 h PlanetScope
# overpass window (09:15-11:45 America/Vancouver, decision 0029) per in-season date,
# roughly 10% of the calendar. That sampling conditionality must be visible ON DISK,
# not only inside a manifest a downstream globber will never open: a consumer that
# globs ONC_RAW_DIR and finds this subdirectory can tell an overpass-window corpus
# from a whole-day pull by its path alone. Resolve it through
# `boatphone.acquire.resolve_corpus_files`, which is the ONE glob (invariant 6).
#
# CONTAINER (corrected -- the earlier note here said the pulled files carry only
# config.ARCHIVE_EXTENSION, "fft", and NOT config.PRODUCT_EXTENSION, "fft.gz";
# decision 0024 falsified that): the bulk pull COMPRESSES ON WRITE, so a file
# ONC serves as `X.fft` lands here as `X.fft.gz`. Both spellings are present --
# 90 plain `.fft` files from the pre-0024 live probe, and the rest `.fft.gz` --
# and `resolve_corpus_files` matches BOTH, which is why it must be the only
# glob. The manifest keeps the two names in separate fields: `filename` is the
# ONC wire name, `disk_basename` is what is actually here.
ONC_OVERPASS_CORPUS_DIR: pathlib.Path = ONC_RAW_DIR / "overpass_window_corpus"

# Windows pulled for a SPECIFIC labelled scene, OUTSIDE the corpus's 09:15-11:45
# local strip. A SEPARATE directory on purpose, and this is not tidiness.
#
# ONC_OVERPASS_CORPUS_DIR means one exact thing -- every in-season date, that
# local time strip, nothing else -- and `PLANET_SAMPLING_CONDITIONALITY_STATEMENT`
# is written against that definition. Dropping off-strip files into it would
# silently falsify every population statistic computed over the directory and
# every caption that cites the conditionality statement, with no error anywhere.
# The scenes needing this pull are precisely the ones the wrong overpass-window
# constant missed (18:17-19:49 UTC measured, 16:15-18:45 pulled), so they are
# off-strip BY CONSTRUCTION and will keep arriving.
ONC_LABELLED_WINDOW_DIR: pathlib.Path = ONC_RAW_DIR / "labelled_window_topup"

# Source: docs/decisions/0005 -- the ONE tracked-fixture exception inside data/.
# Small committed test fixtures only (see data/samples/README.md); negated out of
# the .gitignore media rules so a deliberate fixture is actually committable.
SAMPLES_DIR: pathlib.Path = DATA_DIR / "samples"

# Source: CLAUDE.md "Layout" -- derived products go to data/derived/ with provenance.
DERIVED_DIR: pathlib.Path = DATA_DIR / "derived"

# Basenames of the three A1 deliverable-O1 artifacts. Source: the A1 plan (O1).
# They live HERE, not in scripts/, because the optical workstream opens them by
# name: one definition, shared across teams (CLAUDE.md invariant 6). Strings, not
# Paths, because the output directory is chosen at run time (--out-dir).
UPTIME_CSV_NAME: str = "hydrophone_uptime.csv"
DEPLOYMENTS_CSV_NAME: str = "deployments.csv"
UPTIME_PROVENANCE_JSON_NAME: str = "hydrophone_uptime.provenance.json"

# Source: approved A0 plan. Neither of these exists on disk yet; they are the agreed
# destinations, and require_path() is how code says so out loud instead of guessing.
# INTERIM_DIR: scratch between stages (re-creatable). PROCESSED_DIR: analysis-ready.
INTERIM_DIR: pathlib.Path = DATA_DIR / "interim"
PROCESSED_DIR: pathlib.Path = DATA_DIR / "processed"

# Source: CLAUDE.md "Layout".
DOCS_DIR: pathlib.Path = REPO_ROOT / "docs"
SCRIPTS_DIR: pathlib.Path = REPO_ROOT / "scripts"

# Source: docs/decisions/0009-onc-pretrained-checkpoint-is-the-model.md and B0-1 brief.
# Third-party model code/checkpoints -- immutable to us but not OURS, so invariant 2
# (data/ immutability) does not apply; they live OUTSIDE data/ entirely, gitignored
# (large binaries, same rule as bulk ONC downloads), with provenance tracked
# separately at docs/derived/b0_external_provenance.json.
EXTERNAL_DIR: pathlib.Path = REPO_ROOT / "external"
# Clone of OceanNetworksCanada/selfsupervision_anomalies_onc (git commit SHA pinned
# in the provenance record).
ONC_MODEL_DIR: pathlib.Path = EXTERNAL_DIR / "onc_ssamba"
# Hugging Face Hub artefacts pulled from merileo/* (revision pinned in the
# provenance record): the finetuned SSAMBA/Vision-Mamba classification checkpoint,
# args.pkl, and a labelled eval .h5. NOT a "CPU CNN baseline" -- decision 0012a found
# no such artefact exists under any merileo/* repo; that was a literature-sweep error.
CHECKPOINT_DIR: pathlib.Path = EXTERNAL_DIR / "checkpoints"

# How to obtain each input, keyed by the directory it belongs to. Used to turn a
# bare FileNotFoundError into an actionable one (CLAUDE.md invariant: errors surface,
# and "write code that fails clearly when its input is absent").
_HOW_TO_OBTAIN: dict[pathlib.Path, str] = {
    SAMPLE_DIR: (
        "download the ONC/CIOOS Folger Deep ICLISTEN HF1266 sample into "
        f"'{SAMPLE_DIR}' (ONC Oceans 3.0, https://data.oceannetworks.ca)"
    ),
    ONC_LABELLED_WINDOW_DIR: (
        "produce it by running scripts/pull_labelled_windows.py for the scene(s) "
        "whose optical label you hold; these are windows OUTSIDE the corpus strip"
    ),
    ONC_OVERPASS_CORPUS_DIR: (
        "run `python3 scripts/pull_overpass_corpus.py` to pull the overpass-window "
        f"corpus into '{ONC_OVERPASS_CORPUS_DIR}' (2.5 h/day, in-season, "
        "2020-2025); it is a deliberate, human-triggered operational step and is "
        "not in git"
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
    "ONC_OVERPASS_CORPUS_DIR",
    "ONC_LABELLED_WINDOW_DIR",
    "SAMPLES_DIR", "DERIVED_DIR", "INTERIM_DIR", "PROCESSED_DIR", "DOCS_DIR",
    "SCRIPTS_DIR", "EXTERNAL_DIR", "ONC_MODEL_DIR", "CHECKPOINT_DIR",
    "UPTIME_CSV_NAME", "DEPLOYMENTS_CSV_NAME",
    "UPTIME_PROVENANCE_JSON_NAME", "require_path", "ensure_dir",
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
