#!/usr/bin/env python3
"""B0-1: acquire and pin the external ONC model artefacts.

Runnable entry point. It DEFINES NOTHING SHARED (CLAUDE.md invariant 6): the
destination directories come from `boatphone.paths` (EXTERNAL_DIR,
ONC_MODEL_DIR, CHECKPOINT_DIR); this script only does the fetching and writes
the provenance record.

Two artefact families, both OUTSIDE data/ (they are third-party model
code/checkpoints, not our acquisitions -- invariant 2 does not apply to them,
see boatphone/paths.py):

* `external/onc_ssamba/`  -- a `git clone` of
  https://github.com/OceanNetworksCanada/selfsupervision_anomalies_onc,
  pinned by its commit SHA.
* `external/checkpoints/` -- Hugging Face Hub artefacts under `merileo/*`.

**Finding, recorded rather than hidden (see docs/derived/b0_external_provenance.json
and the run report this script prints):** decision 0009 and the acoustics plan
describe the target checkpoint as `cnn_baseline/cnn_best.pt`, a "CPU CNN
baseline". As of this run, NO file matching that name/description exists under
any `merileo/*` Hugging Face repo. `merileo/*` holds only SSAMBA (Vision-Mamba)
pretrain/finetune checkpoints (`.pth`), not a plain CNN. This script pulls the
closest real, verifiable artefact -- the finetuned classification checkpoint
`merileo/finetune-amba-base-f16-t16-b16-lr1e-4-m300-custom-tr0.8-full_dataset_hydrophones-noexclude`
(`args.pkl` + `ft-cls_best_checkpoint.pth`) -- and records it plainly under
its REAL name; it does NOT rename it to `cnn_best.pt` or otherwise pretend the
planned artefact exists. Whether this checkpoint is viable on CPU without
`mamba_ssm`/`causal_conv1d` is a separate, unresolved question for B0's model-
viability gate proper, not for this acquisition step.

For the labelled eval set, `merileo/onc-ssl-tutorial` carries two files:
`different_locations_incl_backgroundpipelinenormals_multilabel.h5` (~13.6 GB)
and a `_SMALL.h5` variant (~955 MB). This script pulls the SMALL one.

Idempotent: skips any artefact whose destination file already exists AND whose
sha256 matches the value this script itself would fetch/record; re-downloads
and re-verifies otherwise. Fails loudly (raises, naming the URL/repo and the
HTTP/library error) rather than writing a partial provenance record -- no
bare except.

Usage:

    python3 scripts/fetch_onc_model.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from boatphone import paths

# Source: docs/decisions/0009-onc-pretrained-checkpoint.md.
ONC_REPO_URL = "https://github.com/OceanNetworksCanada/selfsupervision_anomalies_onc"

# Source: enumerated live against the Hugging Face Hub by this script's author
# (huggingface_hub.HfApi.list_repo_tree), NOT copied from the plan -- the plan's
# `cnn_baseline/cnn_best.pt` name does not exist under merileo/* (see module
# docstring "Finding"). These are the real repo ids/filenames that do exist.
HF_FINETUNE_REPO = (
    "merileo/finetune-amba-base-f16-t16-b16-lr1e-4-m300-custom-tr0.8-"
    "full_dataset_hydrophones-noexclude"
)
HF_FINETUNE_FILES = ["args.pkl", "ft-cls_best_checkpoint.pth"]

HF_EVAL_REPO = "merileo/onc-ssl-tutorial"
# Despite carrying labelled-eval .h5 data, this is a Hugging Face MODEL repo
# (verified live: api.model_info succeeds, api.dataset_info 401s -- it is not
# registered under the dataset namespace at all).
HF_EVAL_REPO_TYPE = "model"
# The SMALL variant (~955 MB), not the ~13.6 GB full file -- smallest labelled
# eval that exists under merileo/* (both variants carry the same ICLISTENHF1266
# rows per the ONC repo's DATA_DOWNLOAD_AND_PREPARATION.md convention; verified
# below after download by checking the h5 for that string).
HF_EVAL_FILE = "different_locations_incl_backgroundpipelinenormals_multilabel_SMALL.h5"

PROVENANCE_PATH = paths.DOCS_DIR / "derived" / "b0_external_provenance.json"
LICENCE_NOTE = (
    "MIT was assumed by docs/decisions/0009; the repo's own LICENSE file says "
    "BSD 3-Clause (Copyright 2022 Yuan Gong -- this is a fork of SSAMBA). "
    "Recorded as found, not as assumed."
)


def _run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed: {cmd!r}\nstdout: {proc.stdout}\nstderr: {proc.stderr}")
    return proc.stdout.strip()


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clone_or_reuse_onc_repo() -> dict:
    """Clone ONC_REPO_URL into paths.ONC_MODEL_DIR if absent; return its provenance dict."""
    dest = paths.ONC_MODEL_DIR
    if dest.is_dir() and (dest / ".git").is_dir():
        sha = _run(["git", "rev-parse", "HEAD"], cwd=str(dest))
        print(f"[onc_ssamba] reusing existing clone at {dest} @ {sha}")
    else:
        paths.ensure_dir(paths.EXTERNAL_DIR)
        print(f"[onc_ssamba] cloning {ONC_REPO_URL} -> {dest}")
        _run(["git", "clone", ONC_REPO_URL, str(dest)])
        sha = _run(["git", "rev-parse", "HEAD"], cwd=str(dest))
        print(f"[onc_ssamba] cloned @ {sha}")

    license_path = dest / "LICENSE"
    if not license_path.is_file():
        raise RuntimeError(f"clone at {dest} has no LICENSE file -- cannot confirm licence")
    license_text = license_path.read_text(encoding="utf-8", errors="replace")
    if "BSD 3-Clause" in license_text or "BSD-3-Clause" in license_text:
        licence = "BSD-3-Clause (per repo LICENSE file; decision 0009 assumed MIT -- wrong, see LICENCE_NOTE)"
    else:
        licence = "UNKNOWN -- see external/onc_ssamba/LICENSE, do not assume"

    # Keys are paths relative to EXTERNAL_DIR (checks.py's hash verifier resolves
    # recorded paths against external/, not against the clone dir itself).
    file_hashes = {}
    for relname in ("LICENSE", "README.md"):
        fp = dest / relname
        if fp.is_file():
            file_hashes[f"onc_ssamba/{relname}"] = _sha256_file(fp)

    return {
        "name": "onc_ssamba_repo_clone",
        "url": ONC_REPO_URL,
        "git_commit_sha": sha,
        "sha256": file_hashes,
        "size_bytes": sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()),
        "downloaded_utc": _utc_now_iso(),
        "licence": licence,
        "note": (
            "full working-tree clone; sha256 recorded per representative tracked file "
            "(LICENSE, README.md), not per file in the tree -- see B0-1 provenance "
            "resolution (c)"
        ),
    }


def fetch_hf_file(repo_id: str, filename: str, dest_dir: pathlib.Path, repo_type: str = "model") -> dict:
    """Download filename from repo_id via huggingface_hub; return its provenance dict.

    Raises with the repo id, filename, and underlying error on any failure --
    no bare except, no partial/half-written provenance.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    try:
        if repo_type == "dataset":
            refs = api.dataset_info(repo_id)
        else:
            refs = api.model_info(repo_id)
    except HfHubHTTPError as exc:
        raise RuntimeError(f"could not reach Hugging Face repo {repo_id!r}: {exc}") from exc
    revision = refs.sha

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    expected_sha256 = None
    for sib in refs.siblings:
        if sib.rfilename == filename:
            expected_sha256 = sib.lfs.get("sha256") if sib.lfs else None
            expected_size = sib.size
            break
    else:
        raise RuntimeError(f"{filename!r} not found in {repo_id!r} (repo_type={repo_type})")

    if dest_path.is_file() and expected_sha256 and _sha256_file(dest_path) == expected_sha256:
        print(f"[{repo_id}] {filename} already present and hash-verified, skipping download")
    else:
        print(f"[{repo_id}] downloading {filename} ({expected_size} bytes) @ {revision}")
        try:
            local_path = hf_hub_download(
                repo_id=repo_id, filename=filename, revision=revision,
                repo_type=repo_type, local_dir=str(dest_dir),
            )
        except HfHubHTTPError as exc:
            raise RuntimeError(f"download failed for {repo_id}/{filename}: {exc}") from exc
        local_path = pathlib.Path(local_path)
        if local_path != dest_path:
            dest_path = local_path

    got_sha256 = _sha256_file(dest_path)
    if expected_sha256 and got_sha256 != expected_sha256:
        raise RuntimeError(
            f"sha256 mismatch for {repo_id}/{filename}: expected {expected_sha256}, got {got_sha256}"
        )

    return {
        "path": str(dest_path.relative_to(paths.REPO_ROOT)),
        "url": f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}/blob/{revision}/{filename}",
        "hf_revision": revision,
        "sha256": got_sha256,
        "size_bytes": dest_path.stat().st_size,
        "downloaded_utc": _utc_now_iso(),
        "licence": "see repo card -- merileo/* is MIT-tagged per decision 0009 (not independently re-verified here)",
    }


def main() -> int:
    onc_record = clone_or_reuse_onc_repo()

    checkpoint_dir = paths.CHECKPOINT_DIR
    finetune_records = [
        fetch_hf_file(HF_FINETUNE_REPO, f, checkpoint_dir / "finetune", repo_type="model")
        for f in HF_FINETUNE_FILES
    ]
    eval_record = fetch_hf_file(HF_EVAL_REPO, HF_EVAL_FILE, checkpoint_dir / "eval", repo_type=HF_EVAL_REPO_TYPE)

    eval_path = paths.REPO_ROOT / eval_record["path"]
    import h5py
    with h5py.File(eval_path, "r") as f:
        found_iclisten = _h5_contains_iclisten(f)
    if not found_iclisten:
        raise RuntimeError(
            f"{eval_path} does not appear to contain any ICLISTENHF1266-labelled rows "
            "(checked group/dataset names and, where present, a 'device'/'hydrophone' "
            "column) -- B0 needs a labelled eval subset covering that device"
        )
    eval_record["note"] = "verified to contain ICLISTENHF1266 rows after download"
    eval_record["name"] = "merileo_labelled_eval_h5"

    checkpoint_bundle = {
        "name": "merileo_checkpoint_bundle",
        "url": f"https://huggingface.co/{HF_FINETUNE_REPO}",
        "hf_revision": finetune_records[0]["hf_revision"],
        "sha256": {r["path"]: r["sha256"] for r in (finetune_records + [eval_record])},
        "size_bytes": sum(r["size_bytes"] for r in (finetune_records + [eval_record])),
        "downloaded_utc": _utc_now_iso(),
        "licence": finetune_records[0]["licence"],
        "note": (
            "decision 0009 / acoustics_plan_v2 describe a 'CPU CNN baseline "
            "checkpoint' (cnn_baseline/cnn_best.pt) that does NOT exist under any "
            "merileo/* HF repo as of this run (enumerated live via HfApi). This "
            "bundle instead records the real artefacts found: the SSAMBA/Vision-"
            "Mamba finetuned classification checkpoint "
            f"({HF_FINETUNE_REPO}) and a labelled eval .h5 "
            f"({HF_EVAL_REPO}/{HF_EVAL_FILE}). CPU viability of this checkpoint "
            "is unresolved and is a finding for B0's model-viability gate, not "
            "this acquisition step."
        ),
        "components": finetune_records + [eval_record],
    }

    provenance = {
        "onc_ssamba_repo_clone": onc_record,
        "merileo_checkpoint_bundle": checkpoint_bundle,
    }
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {PROVENANCE_PATH}")
    return 0


def _h5_contains_iclisten(f) -> bool:
    """True if any group/dataset name, attr, or string-dtype dataset VALUE mentions
    ICLISTENHF1266 (Folger Deep's device -- see boatphone/config.py). Verified against
    this file's real layout: the device name lives per-row inside the `sources`
    dataset (e.g. b"ICLISTENHF1266_20240126T..." filenames), not in the h5 tree
    structure or attrs -- so a names-only check misses it entirely.
    """
    hit = False

    def _visit(name, obj):
        nonlocal hit
        if "ICLISTENHF1266" in name:
            hit = True
        if isinstance(obj, __import__("h5py").Dataset) and obj.dtype.kind in ("S", "O"):
            try:
                values = obj[:]
            except (OSError, ValueError):
                return
            for v in values:
                s = v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
                if "ICLISTENHF1266" in s:
                    hit = True
                    return

    f.visititems(_visit)
    if hit:
        return True
    for key in f.attrs:
        if "ICLISTENHF1266" in str(f.attrs[key]):
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
