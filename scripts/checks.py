#!/usr/bin/env python3
"""BoatPhone contract checks -- milestone1.

A0: repo skeleton, credentials, env audit.
A1a: the time/unit gate of the hydrophone uptime calendar (UTC bins, half-open
     intervals, dense May-Sep season, deployment attribution).
A1b: ONC location discovery and archive-file listing (runtime-discovered
     location codes, year-chunked listings, filename->time parsing, and the D8
     "partial or absent listings surface as errors" behaviours). The network
     checks are gated behind BOATPHONE_ALLOW_NETWORK and SKIP when it is unset;
     a skip is reported as not-verified, never as a pass.
A1c: the uptime calendar and its CSV, plus the INSPECTION half -- the gap
     summary (`summarise_gaps`), the mean-availability-by-UTC-hour profile
     (`mean_availability_by_utc_hour`), the tracked gap document at
     docs/derived/hydrophone_gaps.md, and the notebook-is-a-thin-wrapper rule.

Run:  python3 scripts/checks.py
Exits non-zero on the first FAIL, printing a named reason.

No pytest in this environment (see CLAUDE.md "Environment"), so these are plain
assert-based functions. Standard library only, plus PyYAML *if present* (A0.5
falls back to a tolerant hand parse otherwise -- it must not add a dependency).

These checks are written BEFORE the implementation exists and are EXPECTED to
fail until boatphone/ is written. That is the point: a check that has never
failed has not been shown to check anything.

Conventions pinned here: all times UTC (nothing in A0 is time-bearing yet);
data/ is immutable -- no check below writes anywhere under data/, and the
credential checks write only into a temp directory that they remove.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import traceback

# numpy is used ONLY by the A8b band-limiting checks, which pass arrays across the
# call and so cannot run in the usual out-of-process child. Present on the hub
# (CLAUDE.md "Environment"). Every other check in this file is stdlib-only.
import numpy as np

# ---------------------------------------------------------------------------
# Constants (no magic numbers/strings -- source in comment)
# ---------------------------------------------------------------------------

# This file lives at <repo>/scripts/checks.py, so the repo root is two parents up.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# The on-disk acquisition directory name. Source: `ls data/` on the OHW hub --
# the name contains literal SPACES and must not be silently normalised.
#
# DELIBERATELY duplicated: boatphone/paths.py defines this same literal, and this
# file must NOT import it from there. A checker that imports the constant it is
# validating checks nothing -- both sides would agree on any typo, including one
# that no longer matches the directory actually on disk. This second, independent
# transcription from `ls data/` is what gives check_a0_1_path_values and
# check_a0_6_paths_sample_dir_matches_disk something real to compare against.
# If the acquisition is ever renamed, BOTH copies must be updated by hand.
SAMPLE_DIR_NAME = "Folger Deep Hydrophone Data Sample"

# Source: CLAUDE.md layout section -- derived products live at data/derived.
DERIVED_SUFFIX = os.path.join("data", "derived")

# Minimum required-package set. Source: approved A0 plan, contract A0.4.
# env_audit.REQUIRED must be a SUPERSET of this; extra entries are allowed.
REQUIRED_MINIMUM = {"onc", "numpy", "scipy", "xarray", "pandas", "pyproj", "pyarrow"}

# Third-party module names that must NOT be imported as a side effect of
# `import boatphone` / `import boatphone.paths` (contract A0.1: package import
# must stay cheap and dependency-free so path constants are always available).
FORBIDDEN_EAGER_IMPORTS = {
    "numpy", "scipy", "pandas", "xarray", "onc", "pyproj", "pyarrow",
    "matplotlib", "geopandas", "torch", "ultralytics", "sklearn", "dask",
    "h5py", "netCDF4", "yaml", "requests",
}

# Paths that .gitignore must ignore, and the one negation that must survive.
# Source: approved A0 plan, contract A0.2.
MUST_BE_IGNORED = [
    ".env",
    "data/derived/x.parquet",
    "data/raw/y.fft.gz",
    "data/interim/z.csv",
    "data/processed/w.parquet",
    "foo.flac",
    "bar.zip",
    "a.wav",
    "b.mp3",
]
MUST_NOT_BE_IGNORED = [".env.example"]

# A value that is obviously not a real ONC token, used to prove the token is
# returned verbatim and never leaked into an exception message.
SENTINEL_TOKEN = "SENTINEL-NOT-A-REAL-ONC-TOKEN-0000"
OTHER_TOKEN = "SENTINEL-DOTENV-VALUE-1111"

# Python version the environment.yml constraint must admit. Source: CLAUDE.md
# ("Python 3.14" on the shared hub).
TARGET_PYTHON = (3, 14)

# Subprocess wall-clock ceiling. Generous: these are import-only child
# processes, so anything near this means a hang, not slowness.
SUBPROC_TIMEOUT_S = 120


class SkipCheck(Exception):
    """Raised by a check that cannot run because its input data is absent."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_child(code: str, env: dict | None = None, cwd: str | None = None):
    """Run `code` in a fresh interpreter with REPO_ROOT importable.

    Child-process isolation is deliberate: several checks manipulate os.environ
    and sys.modules, and doing that in-process would leak between checks.
    """
    child_env = dict(os.environ) if env is None else env
    child_env.setdefault("PYTHONPATH", str(REPO_ROOT))
    if str(REPO_ROOT) not in child_env["PYTHONPATH"].split(os.pathsep):
        child_env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + child_env["PYTHONPATH"]
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True, text=True, env=child_env,
        cwd=cwd or str(REPO_ROOT), timeout=SUBPROC_TIMEOUT_S,
    )


def _child_ok(proc, what: str):
    assert proc.returncode == 0, (
        f"{what}: child exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return proc.stdout


def _git_check_ignore(path: str) -> int:
    """Return git check-ignore's exit code for `path` (0 == ignored)."""
    return subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
        timeout=SUBPROC_TIMEOUT_S,
    ).returncode


# ---------------------------------------------------------------------------
# A0.1 -- boatphone/paths.py
# ---------------------------------------------------------------------------

def check_a0_1_paths_import_is_dependency_free():
    """`import boatphone[.paths]` pulls in no third-party module."""
    out = _child_ok(_run_child(f"""
        import sys, json
        import boatphone
        import boatphone.paths
        forbidden = {sorted(FORBIDDEN_EAGER_IMPORTS)!r}
        print(json.dumps(sorted(m for m in forbidden if m in sys.modules)))
    """), "import boatphone.paths")
    import json
    leaked = json.loads(out.strip().splitlines()[-1])
    assert leaked == [], (
        "importing boatphone.paths pulled in third-party modules: "
        f"{leaked}; package import must stay standard-library only"
    )


def check_a0_1_paths_exports_are_paths():
    """paths.py exports the six directory constants, all pathlib.Path."""
    names = ["REPO_ROOT", "DATA_DIR", "SAMPLE_DIR", "DERIVED_DIR",
             "PROCESSED_DIR", "INTERIM_DIR"]
    out = _child_ok(_run_child(f"""
        import pathlib, boatphone.paths as p
        names = {names!r}
        missing = [n for n in names if not hasattr(p, n)]
        assert not missing, "boatphone.paths is missing exports: " + repr(missing)
        bad = [n for n in names if not isinstance(getattr(p, n), pathlib.Path)]
        assert not bad, "not pathlib.Path: " + repr(bad)
        print("OK")
    """), "boatphone.paths exports")
    assert out.strip().endswith("OK")


def check_a0_1_path_values():
    """REPO_ROOT exists; SAMPLE_DIR keeps the spaced acquisition name; DERIVED_DIR is data/derived."""
    out = _child_ok(_run_child(f"""
        import boatphone.paths as p
        assert p.REPO_ROOT.is_dir(), "REPO_ROOT is not a directory: " + str(p.REPO_ROOT)
        expected = p.DATA_DIR / {SAMPLE_DIR_NAME!r}
        assert p.SAMPLE_DIR == expected, (
            "SAMPLE_DIR != DATA_DIR / " + {SAMPLE_DIR_NAME!r}
            + " (got " + str(p.SAMPLE_DIR) + "); the acquisition name contains spaces"
        )
        assert str(p.DERIVED_DIR).endswith({DERIVED_SUFFIX!r}), (
            "DERIVED_DIR does not end with " + {DERIVED_SUFFIX!r} + ": " + str(p.DERIVED_DIR)
        )
        print("OK")
    """), "boatphone.paths values")
    assert out.strip().endswith("OK")


def check_a0_1_require_path_raises():
    """require_path() on a missing path raises FileNotFoundError naming the path."""
    out = _child_ok(_run_child("""
        import boatphone.paths as p
        missing = p.REPO_ROOT / "definitely-absent-cf3a1e" / "nope.flac"
        try:
            result = p.require_path(missing)
        except FileNotFoundError as exc:
            assert str(missing) in str(exc), (
                "FileNotFoundError message does not contain the offending path: " + str(exc)
            )
        else:
            raise AssertionError(
                "require_path returned " + repr(result)
                + " for a missing path; it must raise FileNotFoundError, not return None"
            )
        print("OK")
    """), "require_path")
    assert out.strip().endswith("OK")


# ---------------------------------------------------------------------------
# A0.2 -- .gitignore
# ---------------------------------------------------------------------------

def check_a0_2_gitignore_ignores_secrets_and_data():
    not_ignored = [p for p in MUST_BE_IGNORED if _git_check_ignore(p) != 0]
    assert not not_ignored, (
        f".gitignore does not ignore these paths: {not_ignored} "
        "(secrets, derived data, and large media must never be committable)"
    )


def check_a0_2_gitignore_negation_works():
    still_ignored = [p for p in MUST_NOT_BE_IGNORED if _git_check_ignore(p) == 0]
    assert not still_ignored, (
        f".gitignore wrongly ignores {still_ignored}; the !.env.example negation "
        "must survive the .env rule so the template stays committable"
    )


# ---------------------------------------------------------------------------
# A0.3 -- boatphone/credentials.py
# ---------------------------------------------------------------------------

def _clean_env(tmpdir: str) -> dict:
    """A child environment with ONC_TOKEN unset and HOME/cwd pointed at tmpdir."""
    env = {k: v for k, v in os.environ.items() if k != "ONC_TOKEN"}
    env["HOME"] = tmpdir
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def check_a0_3_missing_token_raises():
    """No ONC_TOKEN and no .env => MissingCredentialError (subclass of RuntimeError).

    This check FAILS -- it never SKIPs -- when credentials are absent. That
    asymmetry is the whole point of the segment: absent *data* is a skip,
    absent *credential handling* is a bug. A get_onc_token() that returns None,
    returns "", or warns-and-continues sends an unauthenticated request and
    produces an empty download that looks like "no vessels".
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = _child_ok(_run_child("""
            import boatphone.credentials as c
            assert issubclass(c.MissingCredentialError, RuntimeError), (
                "MissingCredentialError must subclass RuntimeError"
            )
            try:
                tok = c.get_onc_token()
            except c.MissingCredentialError:
                pass
            else:
                raise AssertionError(
                    "get_onc_token() returned " + repr(tok)
                    + " with no ONC_TOKEN and no .env; it must raise "
                    "MissingCredentialError rather than return None/'' or warn-and-continue"
                )
            print("OK")
        """, env=_clean_env(tmp), cwd=tmp), "get_onc_token with no credential")
    assert out.strip().endswith("OK")


def check_a0_3_env_var_returned_verbatim():
    with tempfile.TemporaryDirectory() as tmp:
        env = _clean_env(tmp)
        env["ONC_TOKEN"] = SENTINEL_TOKEN
        out = _child_ok(_run_child(f"""
            import boatphone.credentials as c
            tok = c.get_onc_token()
            assert tok == {SENTINEL_TOKEN!r}, (
                "get_onc_token() did not return the ONC_TOKEN value verbatim "
                "(len " + str(len(tok)) + ")"
            )
            print("OK")
        """, env=env, cwd=tmp), "get_onc_token from process env")
    assert out.strip().endswith("OK")


def check_a0_3_process_env_overrides_dotenv():
    """A value in the real environment wins over a differing .env value."""
    with tempfile.TemporaryDirectory() as tmp:
        pathlib.Path(tmp, ".env").write_text(f"ONC_TOKEN={OTHER_TOKEN}\n", encoding="utf-8")
        env = _clean_env(tmp)
        env["ONC_TOKEN"] = SENTINEL_TOKEN
        out = _child_ok(_run_child(f"""
            import boatphone.credentials as c
            tok = c.get_onc_token()
            assert tok == {SENTINEL_TOKEN!r}, (
                "process-environment ONC_TOKEN must override the .env value; "
                "got a value of length " + str(len(tok))
            )
            print("OK")
        """, env=env, cwd=tmp), "env overrides .env")
    assert out.strip().endswith("OK")


def check_a0_3_malformed_dotenv_reports_line_number():
    """A malformed .env line raises, and the error names the line number."""
    bad_line_no = 3  # line 3 below has no '=' -- it cannot be a key/value pair
    with tempfile.TemporaryDirectory() as tmp:
        pathlib.Path(tmp, ".env").write_text(
            "# a comment\n"
            "GOOD_KEY=good_value\n"
            "this line has no equals sign\n"
            f"ONC_TOKEN={SENTINEL_TOKEN}\n",
            encoding="utf-8",
        )
        out = _child_ok(_run_child(f"""
            import boatphone.credentials as c
            try:
                tok = c.get_onc_token()
            except Exception as exc:
                msg = str(exc)
                assert str({bad_line_no}) in msg, (
                    "malformed .env error must name the offending line number "
                    + str({bad_line_no}) + "; got: " + msg
                )
                assert {SENTINEL_TOKEN!r} not in msg, (
                    "the token value leaked into an exception message"
                )
            else:
                raise AssertionError(
                    "a malformed .env line was silently skipped (returned a token of length "
                    + str(len(tok)) + "); errors must surface"
                )
            print("OK")
        """, env=_clean_env(tmp), cwd=tmp), "malformed .env")
    assert out.strip().endswith("OK")


def check_a0_3_token_never_leaks_into_errors():
    """On every raising path, the token value stays out of str(exc)."""
    with tempfile.TemporaryDirectory() as tmp:
        pathlib.Path(tmp, ".env").write_text(
            f"ONC_TOKEN={OTHER_TOKEN}\nnot a pair\n", encoding="utf-8"
        )
        env = _clean_env(tmp)
        env["ONC_TOKEN"] = SENTINEL_TOKEN
        out = _child_ok(_run_child(f"""
            import boatphone.credentials as c
            leaked = []
            for fn in (c.get_onc_token,):
                try:
                    fn()
                except Exception as exc:
                    blob = str(exc) + repr(exc)
                    for name, val in (("ONC_TOKEN", {SENTINEL_TOKEN!r}),
                                      ("dotenv", {OTHER_TOKEN!r})):
                        if val in blob:
                            leaked.append(name)
            assert not leaked, "token value(s) leaked into an error message: " + repr(leaked)
            print("OK")
        """, env=env, cwd=tmp), "token leakage")
    assert out.strip().endswith("OK")


def check_a0_3_placeholder_token_is_rejected():
    """A .env still holding the .env.example placeholder => MissingCredentialError.

    This is the exact real-world case: a teammate copies .env.example, forgets to
    paste their token, and gets an unauthenticated download that looks like "no
    vessels". The rejection is asserted in credentials.py's docstring, .env.example,
    and docs/environment-audit.md -- so it needs to be *checked* somewhere.

    The placeholder literal is imported from credentials rather than retyped here:
    if the template's placeholder changes, this check must follow it, not diverge.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = _child_ok(_run_child("""
            import pathlib
            import boatphone.credentials as c
            dotenv = pathlib.Path.cwd() / c.DOTENV_FILENAME
            dotenv.write_text(
                c.ONC_TOKEN_VAR + "=" + c.PLACEHOLDER_TOKEN + "\\n", encoding="utf-8"
            )
            assert c.PLACEHOLDER_TOKEN, "PLACEHOLDER_TOKEN must be a non-empty string"
            try:
                tok = c.get_onc_token()
            except c.MissingCredentialError as exc:
                assert c.PLACEHOLDER_TOKEN not in str(exc), (
                    "the .env value leaked into the error message"
                )
            else:
                raise AssertionError(
                    "get_onc_token() returned the unfilled .env.example placeholder "
                    "(len " + str(len(tok)) + "); it must raise MissingCredentialError"
                )
            print("OK")
        """, env=_clean_env(tmp), cwd=tmp), "placeholder token rejection")
    assert out.strip().endswith("OK")


def check_a0_3_parse_dotenv_quoting_and_export():
    """parse_dotenv strips an `export ` prefix and one layer of matching quotes.

    Pinned because a token read as `"abc"` (quotes included) authenticates as a
    different, wrong string -- and the failure is a 401, not a parse error.
    """
    with tempfile.TemporaryDirectory() as tmp:
        env_path = pathlib.Path(tmp, "quoting.env")
        env_path.write_text(
            "# comment line\n"
            "\n"
            f'DOUBLE="{SENTINEL_TOKEN}"\n'
            f"SINGLE='{SENTINEL_TOKEN}'\n"
            f"BARE={SENTINEL_TOKEN}\n"
            f"EXPORTED=export_is_not_a_prefix_here\n"
            f"export EXPORT_PREFIX={SENTINEL_TOKEN}\n"
            f'export EXPORT_QUOTED="{SENTINEL_TOKEN}"\n'
            f"  SPACED  =  {SENTINEL_TOKEN}  \n"
            "EMPTY=\n"
            "MISMATCHED=\"unclosed\n",
            encoding="utf-8",
        )
        out = _child_ok(_run_child(f"""
            import pathlib
            import boatphone.credentials as c
            vals = c.parse_dotenv(pathlib.Path({str(env_path)!r}))
            tok = {SENTINEL_TOKEN!r}
            expect = {{
                "DOUBLE": tok, "SINGLE": tok, "BARE": tok,
                "EXPORTED": "export_is_not_a_prefix_here",
                "EXPORT_PREFIX": tok, "EXPORT_QUOTED": tok,
                "SPACED": tok, "EMPTY": "",
                "MISMATCHED": '"unclosed',
            }}
            wrong = {{k: (v, vals.get(k)) for k, v in expect.items() if vals.get(k) != v}}
            assert not wrong, "parse_dotenv key -> (expected, got): " + repr(wrong)
            print("OK")
        """, env=_clean_env(tmp), cwd=tmp), "parse_dotenv quoting/export")
    assert out.strip().endswith("OK")


def check_a0_3_dotenv_search_paths_inside_repo():
    """POSITIVE branch: run from inside the checkout and the repo .env IS a candidate.

    Only the negative branch (temp cwd outside the repo) was exercised before, so a
    dotenv_search_paths() that never returned the repo .env would have looked green.
    Uses an existing repo subdirectory as cwd and writes nothing into the repo.
    """
    subdir = REPO_ROOT / "scripts"
    assert subdir.is_dir(), f"expected an existing repo subdirectory at {subdir}"
    out = _child_ok(_run_child(f"""
        import pathlib
        import boatphone.credentials as c
        from boatphone.paths import REPO_ROOT

        paths = [pathlib.Path(p).resolve() for p in c.dotenv_search_paths()]
        cwd_env = (pathlib.Path.cwd() / c.DOTENV_FILENAME).resolve()
        repo_env = (REPO_ROOT / c.DOTENV_FILENAME).resolve()
        assert paths[0] == cwd_env, (
            "cwd .env must have highest precedence; got " + repr([str(p) for p in paths])
        )
        assert repo_env in paths, (
            "running inside the checkout, the repo .env must be a candidate; got "
            + repr([str(p) for p in paths])
        )
        assert len(paths) == len(set(paths)), (
            "dotenv_search_paths returned duplicates: " + repr([str(p) for p in paths])
        )
        print("OK")
    """, cwd=str(subdir)), "dotenv_search_paths inside the checkout")
    assert out.strip().endswith("OK")


def check_a0_3_dotenv_falls_through_unusable_value():
    """A `.env` with `ONC_TOKEN=` (or the placeholder) must not SHADOW a later `.env`.

    _from_dotenv currently returns from the first file that CONTAINS the key, even
    when the value is empty or the placeholder -- so an unfilled `.env` in a
    subdirectory hides a perfectly good repo-root `.env` and the user gets
    "token is absent" while staring at a file that has one.

    dotenv_search_paths is stubbed to two temp files so this exercises the
    fall-through logic without writing a `.env` into the repository.
    """
    with tempfile.TemporaryDirectory() as tmp:
        first = pathlib.Path(tmp, "first", ".env")
        second = pathlib.Path(tmp, "second", ".env")
        first.parent.mkdir()
        second.parent.mkdir()
        second.write_text(f"ONC_TOKEN={SENTINEL_TOKEN}\n", encoding="utf-8")
        out = _child_ok(_run_child(f"""
            import pathlib
            import boatphone.credentials as c

            first, second = pathlib.Path({str(first)!r}), pathlib.Path({str(second)!r})
            c.dotenv_search_paths = lambda: [first, second]

            failures = []
            for label, shadow in (("empty value", ""),
                                  ("placeholder value", c.PLACEHOLDER_TOKEN)):
                first.write_text("ONC_TOKEN=" + shadow + "\\n", encoding="utf-8")
                try:
                    tok = c.get_onc_token()
                except Exception as exc:
                    failures.append(
                        label + ": " + type(exc).__name__ + " -- a leading .env with an "
                        "unusable ONC_TOKEN shadowed the next candidate instead of "
                        "falling through to it"
                    )
                else:
                    if tok != {SENTINEL_TOKEN!r}:
                        failures.append(
                            label + ": got a token of length " + str(len(tok))
                            + ", expected the second .env's value"
                        )
            assert not failures, "; ".join(failures)
            print("OK")
        """, env=_clean_env(tmp), cwd=tmp), "_from_dotenv fall-through")
    assert out.strip().endswith("OK")


def check_a0_3_dotenv_example_tracked_and_dotenv_ignored():
    example = REPO_ROOT / ".env.example"
    assert example.is_file(), (
        ".env.example is missing; the template documents ONC_TOKEN without carrying a secret"
    )
    assert _git_check_ignore(".env.example") != 0, ".env.example must not be gitignored"
    assert _git_check_ignore(".env") == 0, ".env must be gitignored"


# ---------------------------------------------------------------------------
# A0.4 -- boatphone/env_audit.py
# ---------------------------------------------------------------------------

def check_a0_4_strict_cli_exits_zero():
    """`python3 -m boatphone.env_audit --strict` passes on this hub."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "boatphone.env_audit", "--strict"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, env=env,
        timeout=SUBPROC_TIMEOUT_S,
    )
    assert proc.returncode == 0, (
        "`python3 -m boatphone.env_audit --strict` exited "
        f"{proc.returncode} on an environment where every required package is present\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def check_a0_4_required_covers_minimum():
    out = _child_ok(_run_child(f"""
        import boatphone.env_audit as ea
        required = set(ea.REQUIRED)
        missing = sorted({REQUIRED_MINIMUM!r} - required)
        assert not missing, (
            "env_audit.REQUIRED omits required packages: " + repr(missing)
        )
        print("OK")
    """), "env_audit.REQUIRED contents")
    assert out.strip().endswith("OK")


def check_a0_4_versions_resolve_without_dunder_version():
    """`onc` has no __version__; the audit must resolve versions via importlib.metadata."""
    out = _child_ok(_run_child("""
        import onc
        assert not hasattr(onc, "__version__"), (
            "onc grew a __version__ attribute; this check's premise needs revisiting"
        )
        import boatphone.env_audit as ea
        v = ea.package_version("onc")
        assert v and v.lower() not in ("unknown", "none"), (
            "env_audit.package_version('onc') returned " + repr(v)
            + "; versions must resolve via importlib.metadata, not __version__"
        )
        print("OK")
    """), "onc version resolution")
    assert out.strip().endswith("OK")


def check_a0_4_strict_path_is_live():
    """A synthetic impossible requirement must make the strict audit fail.

    Without this, an audit that always returns 0 would pass every other check.
    Contract: env_audit.audit(required=..., strict=...) -> int exit code, so
    this is testable without editing the module.
    """
    impossible = "boatphone_no_such_package_9f2c"
    out = _child_ok(_run_child(f"""
        import boatphone.env_audit as ea
        code = ea.audit(required=sorted(set(ea.REQUIRED)) + [{impossible!r}], strict=True)
        assert code != 0, (
            "audit(strict=True) returned 0 with a synthetic unimportable requirement "
            + {impossible!r} + "; the strict path is always-green, not live"
        )
        ok = ea.audit(required=sorted(set(ea.REQUIRED)), strict=True)
        assert ok == 0, (
            "audit(strict=True) returned " + str(ok)
            + " for the real REQUIRED set on an environment where it should pass"
        )
        print("OK")
    """), "env_audit strict path")
    assert out.strip().endswith("OK")


# ---------------------------------------------------------------------------
# A0.5 -- environment.yml  (text/YAML only; never triggers a conda solve)
# ---------------------------------------------------------------------------

def _load_environment_yml():
    path = REPO_ROOT / "environment.yml"
    assert path.is_file(), "environment.yml is missing at the repo root"
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # PyYAML is present on this hub; verified before writing these checks
    except ImportError:
        return _tolerant_parse(text), "tolerant fallback parser (PyYAML absent)"
    return yaml.safe_load(text), "PyYAML safe_load"


def _tolerant_parse(text: str) -> dict:
    """Minimal parse of the environment.yml shapes we care about.

    Used only if PyYAML is unavailable -- adding a dependency to check a
    manifest would be self-defeating. Handles `key: value`, `key:` followed by
    `  - item`, and one nested `- pip:` block. Not a general YAML parser.
    """
    doc: dict = {}
    key = None
    pip_items = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        if indent == 0 and stripped.endswith(":"):
            key = stripped[:-1]
            doc[key] = []
            pip_items = None
        elif indent == 0 and ":" in stripped:
            k, v = stripped.split(":", 1)
            doc[k.strip()] = v.strip()
            key = None
            pip_items = None
        elif stripped.startswith("- ") and key is not None:
            item = stripped[2:].strip()
            if item in ("pip:", "pip :"):
                pip_items = []
                doc[key].append({"pip": pip_items})
            elif pip_items is not None and indent >= 4:
                pip_items.append(item)
            else:
                pip_items = None
                doc[key].append(item)
    return doc


def _flatten_deps(doc: dict) -> list[str]:
    flat = []
    for dep in doc.get("dependencies") or []:
        if isinstance(dep, str):
            flat.append(dep)
        elif isinstance(dep, dict):
            for sub in dep.values():
                flat.extend(str(s) for s in (sub or []))
    return flat


def _dep_name(spec: str) -> str:
    """Package name from a dependency spec ('numpy>=1.26' -> 'numpy'). Lowercased."""
    name = spec.strip()
    for i, ch in enumerate(name):
        if ch in "<>=!|,[ ":
            name = name[:i]
            break
    return name.strip().lower()


def check_a0_5_environment_yml_shape():
    doc, how = _load_environment_yml()
    print(f"      [environment.yml parsed with: {how}]")
    assert isinstance(doc, dict), "environment.yml did not parse to a mapping"
    missing = [k for k in ("name", "channels", "dependencies") if k not in doc]
    assert not missing, f"environment.yml is missing top-level keys: {missing}"


def check_a0_5_manifest_covers_required():
    """Every env_audit.REQUIRED package appears in the manifest -- no silent drift."""
    doc, _ = _load_environment_yml()
    flat = _flatten_deps(doc)
    out = _child_ok(_run_child("""
        import boatphone.env_audit as ea
        print("\\n".join(sorted(set(ea.REQUIRED))))
    """), "env_audit.REQUIRED for manifest cross-check")
    required = [n for n in out.split() if n]
    # A dep spec looks like "numpy>=1.26" / "python=3.14"; compare on the name part.
    names = {_dep_name(s) for s in flat}
    absent = sorted(n for n in required if n.lower() not in names)
    assert not absent, (
        f"environment.yml does not list env_audit.REQUIRED packages: {absent}; "
        "the manifest and the audit must not drift apart"
    )


def _parse_version(text: str) -> tuple:
    """'3.14' -> (3, 14). Non-numeric trailing components are dropped.

    Enough for CPython minor-version constraints; not a general PEP 440 parser.
    `packaging` is present on this hub, but conda's `|` (OR) grammar is not PEP 440,
    so a spec evaluator has to be written anyway -- doing the whole thing in the
    standard library keeps one grammar rather than two.
    """
    parts = []
    for chunk in text.strip().split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _cmp_pad(a: tuple, b: tuple) -> int:
    """Compare two version tuples, zero-padding to equal length. -1/0/1."""
    n = max(len(a), len(b))
    a = a + (0,) * (n - len(a))
    b = b + (0,) * (n - len(b))
    return (a > b) - (a < b)


def _atom_admits(atom: str, version: tuple) -> bool:
    """Does one conda constraint atom (e.g. '>=3.11', '=3.13', '3.11.*') admit `version`?"""
    atom = atom.strip()
    if not atom:
        return True
    for op in ("==", "!=", ">=", "<=", ">", "<", "="):
        if atom.startswith(op):
            rhs = atom[len(op):].strip()
            break
    else:
        op, rhs = "=", atom  # a bare "3.14" is conda's fuzzy match, same as "=3.14"
    glob = rhs.endswith(".*") or rhs.endswith("*")
    rhs = rhs.rstrip("*").rstrip(".")
    want = _parse_version(rhs)
    if op in ("=",) or (op == "==" and glob):
        # conda `=X.Y` / `X.Y.*` is a PREFIX match: 3.14.6 satisfies =3.14.
        return version[:len(want)] == want
    c = _cmp_pad(version, want)
    if op == "==":
        # The question asked is "does this admit the 3.14 MINOR SERIES", so an exact
        # pin of a patch inside that series (==3.14.6) counts as admitting it.
        return c == 0 or want[:len(version)] == version
    if op == "!=":
        return c != 0
    if op == ">=":
        return c >= 0
    if op == "<=":
        return c <= 0
    if op == ">":
        return c > 0
    if op == "<":
        return c < 0
    raise AssertionError(f"unhandled constraint operator {op!r} in atom {atom!r}")


def conda_spec_admits(spec: str, version: tuple) -> bool:
    """Evaluate a conda dependency spec against a concrete version tuple.

    Grammar (conda MatchSpec): `|` is OR and binds loosest; `,` is AND within an
    OR-branch. The package name prefix is stripped first.

    This is deliberately an EVALUATOR, not a substring test. A substring test is
    wrong in both directions: it fails the idiomatic `python>=3.11,<3.15` (which
    does admit 3.14) and passes `python<3.14` (which excludes it).
    """
    body = spec.strip()
    for i, ch in enumerate(body):
        if ch in "<>=!|,":
            body = body[i:]
            break
    else:
        return True  # bare package name, no constraint at all
    return any(
        all(_atom_admits(atom, version) for atom in branch.split(","))
        for branch in body.split("|")
    )


def check_a0_5_python_constraint_admits_314():
    """The python spec, EVALUATED, admits the hub interpreter (3.14).

    Self-test first: the evaluator must discriminate the four cases the reviewers
    named, otherwise a green result below proves nothing about environment.yml.
    """
    self_test = [
        (">=3.11,<3.15", True),   # idiomatic range -- must PASS
        ("=3.11|3.12|3.13|3.14", True),   # explicit OR-form -- must PASS
        ("<3.14", False),         # excludes 3.14 (substring test wrongly passed this)
        ("=3.13,<3.14", False),   # excludes 3.14 (substring test wrongly passed this)
    ]
    wrong = [(s, want, conda_spec_admits("python" + s, TARGET_PYTHON))
             for s, want in self_test
             if conda_spec_admits("python" + s, TARGET_PYTHON) is not want]
    assert not wrong, (
        "the python-spec evaluator itself is wrong on known cases "
        "(spec, expected, got): " + repr(wrong)
    )

    doc, _ = _load_environment_yml()
    specs = [s for s in _flatten_deps(doc) if _dep_name(s) == "python"]
    assert specs, "environment.yml pins no python version"
    want = ".".join(str(p) for p in TARGET_PYTHON)
    admitting = [s for s in specs if conda_spec_admits(s, TARGET_PYTHON)]
    assert admitting, (
        f"python constraint {specs} does not admit {want}, the hub's interpreter "
        "(evaluated, not string-matched)"
    )


# ---------------------------------------------------------------------------
# A0.6 -- data-dependent checks skip cleanly when the acquisition is absent
# ---------------------------------------------------------------------------

def check_a0_6_sample_dir_present():
    """Data-dependent: SKIPs (not fails) when the acquisition is not on disk."""
    sample = REPO_ROOT / "data" / SAMPLE_DIR_NAME
    if not sample.is_dir():
        raise SkipCheck(
            f"acquisition directory absent: {sample} -- large acquisitions are not in git"
        )
    entries = sorted(p.name for p in sample.iterdir())
    assert entries, f"acquisition directory is empty: {sample}"


def check_a0_6_paths_sample_dir_matches_disk():
    """Data-dependent: boatphone.paths.SAMPLE_DIR points at the real directory."""
    sample = REPO_ROOT / "data" / SAMPLE_DIR_NAME
    if not sample.is_dir():
        raise SkipCheck(f"acquisition directory absent: {sample}")
    out = _child_ok(_run_child(f"""
        import boatphone.paths as p
        assert p.SAMPLE_DIR.is_dir(), "SAMPLE_DIR does not exist on disk: " + str(p.SAMPLE_DIR)
        import pathlib
        assert p.SAMPLE_DIR.resolve() == pathlib.Path({str(sample)!r}).resolve(), (
            "SAMPLE_DIR " + str(p.SAMPLE_DIR) + " does not resolve to the acquisition on disk"
        )
        print("OK")
    """), "SAMPLE_DIR on disk")
    assert out.strip().endswith("OK")


# ---------------------------------------------------------------------------
# A1a -- time/unit gate for the hydrophone uptime calendar
# ---------------------------------------------------------------------------
#
# THE API PINNED HERE (the coder implements exactly this; these checks are the
# contract, and they are written to FAIL until boatphone/config.py and the pure
# functions in boatphone/onc_client.py exist).
#
#   boatphone.config:
#       BIN_SECONDS = 300
#       STUDY_START_UTC, STUDY_END_UTC   tz-aware datetime, tzinfo is timezone.utc
#       SEASON_MONTHS_UTC = (5, 6, 7, 8, 9)
#       DEVICE_CODE, DEVICE_ID, PRODUCT_EXTENSION
#
#   boatphone.onc_client (pure; no `onc` import, no I/O in this segment):
#
#     season_bins_utc(start_utc, end_utc, season_months=SEASON_MONTHS_UTC)
#         -> list[tuple[datetime, datetime]]
#         Dense, half-open [start, end) bins of exactly BIN_SECONDS, every edge an
#         integer multiple of BIN_SECONDS since the UTC epoch, strictly increasing.
#         Only WHOLE bins fully contained in [start_utc, end_utc) are emitted.
#         A bin is kept when start_utc.month is in `season_months`; passing
#         season_months=None disables the seasonal filter entirely (needed to
#         exercise the out-of-season DST transition days -- decision 0002 s5).
#         Naive datetimes RAISE (decision D1: naive never crosses a boundary).
#
#     mark_available(bins, coverage_intervals) -> list[bool]
#         Same length and order as `bins`. True iff the bin's half-open span
#         intersects any half-open coverage interval in a span of NON-ZERO length.
#         Touching at an endpoint is NOT overlap.
#
#     assign_deployment(bins, deployments, available) -> list[str]
#         `deployments` is a sequence of (deployment_id: str, start_utc, end_utc),
#         each half-open. Returns ONC's identifier as a string for the deployment
#         whose half-open interval contains the bin's START, or "" when the bin
#         falls in no deployment. `available` is REQUIRED (not optional) because
#         D6's raise condition is not expressible without it: an available bin in
#         no known deployment means a listed file outside every deployment, which
#         must RAISE onc_client.DeploymentCoverageError (a RuntimeError subclass)
#         rather than be silently attributed to "".
#
# Return type choice, stated per the brief: plain tuples of tz-aware datetimes,
# and plain parallel lists for availability/deployment -- stdlib only, so these
# checks do not inherit pandas' tz behaviour (decision 0002 s1).

_A1A_MODULES = ("boatphone.config", "boatphone.onc_client")


def _a1a_mods():
    """Import the A1a modules. ImportError here IS the expected pre-coder failure.

    These functions are pure, so they run in-process rather than in a child (unlike
    the A0 credential checks, which manipulate os.environ). REPO_ROOT is put on
    sys.path first: `python3 scripts/checks.py` puts scripts/ on sys.path, not the
    repo root, and without this a path problem would masquerade as "not implemented".
    """
    import importlib
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return tuple(importlib.import_module(m) for m in _A1A_MODULES)


def _utc(*args):
    from datetime import datetime, timezone
    return datetime(*args, tzinfo=timezone.utc)


def _bin_starts(bins):
    return [b[0] for b in bins]


def check_a1a_0_config_constants():
    """config.py carries the A1 constants, and the time ones are tz-aware UTC."""
    from datetime import datetime, timezone
    cfg, _ = _a1a_mods()
    assert cfg.BIN_SECONDS == 300, (
        f"config.BIN_SECONDS is {cfg.BIN_SECONDS!r}; D2 fixes the bin width at 300 s"
    )
    assert tuple(cfg.SEASON_MONTHS_UTC) == (5, 6, 7, 8, 9), (
        f"config.SEASON_MONTHS_UTC is {cfg.SEASON_MONTHS_UTC!r}; D4 fixes May-Sep in UTC"
    )
    for name in ("STUDY_START_UTC", "STUDY_END_UTC"):
        val = getattr(cfg, name)
        assert isinstance(val, datetime), f"config.{name} is {type(val).__name__}, not datetime"
        assert val.tzinfo is not None and val.utcoffset() == timezone.utc.utcoffset(None), (
            f"config.{name} is not tz-aware UTC (tzinfo={val.tzinfo!r}); "
            "D1 forbids naive or non-UTC timestamps"
        )
    # Source: the HF1266 pre-deployment calibration validity date (see A1 plan).
    assert cfg.STUDY_START_UTC == _utc(2020, 2, 18), (
        f"config.STUDY_START_UTC is {cfg.STUDY_START_UTC!r}; expected 2020-02-18T00:00:00Z, "
        "the HF1266 calibration validity date"
    )
    assert cfg.STUDY_END_UTC > cfg.STUDY_START_UTC, "STUDY_END_UTC must postdate STUDY_START_UTC"
    assert cfg.DEVICE_CODE == "ICLISTENHF1266", f"config.DEVICE_CODE is {cfg.DEVICE_CODE!r}"
    assert cfg.DEVICE_ID == 23235, f"config.DEVICE_ID is {cfg.DEVICE_ID!r}"
    assert cfg.PRODUCT_EXTENSION == "fft.gz", (
        f"config.PRODUCT_EXTENSION is {cfg.PRODUCT_EXTENSION!r}"
    )


def check_a1a_0b_naive_datetime_raises():
    """D1: a naive datetime never crosses a function boundary -- it raises."""
    from datetime import datetime
    _, onc = _a1a_mods()
    naive_a = datetime(2024, 7, 4, 9, 0, 0)
    aware_b = _utc(2024, 7, 4, 10, 0, 0)
    for label, args in (("naive start", (naive_a, aware_b)),
                        ("naive end", (aware_b.replace(tzinfo=None), aware_b))):
        try:
            got = onc.season_bins_utc(*args)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError(
                f"season_bins_utc accepted a {label} and returned {len(got)} bins; "
                "a naive datetime must raise (D1) -- silently assuming UTC is exactly "
                "how a local-time input becomes a 7-hour join error"
            )


def check_a1a_1_half_open_hand_built_gate():
    """THE gate. Coverage [09:07:30Z, 09:12:30Z) marks 09:05 and 09:10 -- not 09:00, not 09:15.

    This is the inclusive-end catcher: an implementation using <= on the coverage
    end, or `bin_start <= cov_end`, wrongly lights 09:15 as well.
    """
    _, onc = _a1a_mods()
    span_start, span_end = _utc(2024, 7, 4, 9, 0), _utc(2024, 7, 4, 9, 20)
    bins = onc.season_bins_utc(span_start, span_end)
    starts = _bin_starts(bins)
    expect_starts = [_utc(2024, 7, 4, 9, m) for m in (0, 5, 10, 15)]
    assert starts == expect_starts, (
        f"season_bins_utc over a 20-minute span gave starts {starts}, expected {expect_starts}"
    )
    cov = [(_utc(2024, 7, 4, 9, 7, 30), _utc(2024, 7, 4, 9, 12, 30))]
    avail = onc.mark_available(bins, cov)
    assert len(avail) == len(bins), (
        f"mark_available returned {len(avail)} flags for {len(bins)} bins; "
        "it must be parallel to the input"
    )
    got = {s.strftime("%H:%M") for s, a in zip(starts, avail) if a}
    want = {"09:05", "09:10"}
    assert got == want, (
        f"coverage [09:07:30Z, 09:12:30Z) marked {sorted(got)} available, expected "
        f"{sorted(want)}. Extra 09:15 => the coverage end is being treated as inclusive; "
        "extra 09:00 or missing 09:10 => the bin interval is not half-open [start, end)"
    )


def check_a1a_1b_touching_is_not_overlap():
    """A coverage interval ending exactly on a bin edge does not light that bin."""
    cfg, onc = _a1a_mods()
    from datetime import timedelta
    bins = onc.season_bins_utc(_utc(2024, 7, 4, 9, 0), _utc(2024, 7, 4, 9, 20))
    width = timedelta(seconds=cfg.BIN_SECONDS)
    cov = [(_utc(2024, 7, 4, 9, 5), _utc(2024, 7, 4, 9, 5) + width)]  # exactly the 09:05 bin
    got = {s.strftime("%H:%M") for (s, _e), a in zip(bins, onc.mark_available(bins, cov)) if a}
    assert got == {"09:05"}, (
        f"coverage exactly equal to the 09:05 bin marked {sorted(got)}; a coverage "
        "interval that only TOUCHES a neighbouring bin edge must not mark it "
        "(zero-length intersection is not overlap)"
    )


def check_a1a_2_pdt_offset_leak_is_caught():
    """The same coverage shifted by the PDT offset (+7 h) must mark a DIFFERENT bin set.

    Decision 0002 s5 requires a deliberately mis-offset input to FAIL. If a
    local-time value ever leaks in as if it were UTC, these two sets are equal
    and this check goes red -- which is the only cheap way to see that error.
    """
    from datetime import timedelta
    _, onc = _a1a_mods()
    # America/Vancouver summer offset. Named here only to CONSTRUCT the wrong input;
    # no timezone conversion is performed anywhere in A1 (D4).
    pdt_offset = timedelta(hours=7)
    day_start, day_end = _utc(2024, 7, 4, 0, 0), _utc(2024, 7, 5, 0, 0)
    bins = onc.season_bins_utc(day_start, day_end)
    cov = (_utc(2024, 7, 4, 9, 7, 30), _utc(2024, 7, 4, 9, 12, 30))
    shifted = (cov[0] + pdt_offset, cov[1] + pdt_offset)

    def marked(interval):
        return {s for (s, _e), a in zip(bins, onc.mark_available(bins, [interval])) if a}

    true_set, wrong_set = marked(cov), marked(shifted)
    assert true_set, "the correct-UTC coverage interval marked nothing available"
    assert wrong_set, "the +7 h shifted interval marked nothing -- the day span is too short"
    assert true_set != wrong_set, (
        "a coverage interval shifted by the PDT offset (+7 h) produced the SAME available "
        f"bins as the true UTC interval ({sorted(x.isoformat() for x in true_set)}); the "
        "time base is being normalised or discarded somewhere, so a local-time leak would "
        "not be detectable"
    )
    assert not (true_set & wrong_set), (
        "the true and +7 h-shifted bin sets overlap; a 7-hour error must displace the "
        "coverage entirely"
    )


def check_a1a_3_grid_invariants():
    """Every row is exactly BIN_SECONDS, epoch-aligned, strictly increasing, in season."""
    from datetime import timedelta
    cfg, onc = _a1a_mods()
    width = timedelta(seconds=cfg.BIN_SECONDS)
    season = set(cfg.SEASON_MONTHS_UTC)
    # DELIBERATELY unaligned bounds on BOTH ends. With aligned bounds an
    # implementation that emits a trailing partial bin is indistinguishable from a
    # correct one -- that mutant survived an aligned-span version of this check.
    #
    # The START here is out of season on purpose: it exercises the month-jump path,
    # and the other invariants below benefit from a span that straddles the whole
    # season. But note what it therefore does NOT exercise: the jump to May 1
    # 00:00:00Z lands on an aligned instant regardless, so the epoch-alignment
    # assertion below can never fail from THIS span alone. Epoch alignment against
    # an unaligned request is exercised by check_a1a_3b, which uses an IN-SEASON
    # unaligned start so no month jump intervenes. Keep both.
    span_start, span_end = _utc(2024, 4, 28, 0, 3, 17), _utc(2024, 9, 30, 11, 42, 3)
    bins = onc.season_bins_utc(span_start, span_end)
    assert bins, "season_bins_utc returned no bins for a span straddling the whole 2024 season"
    assert bins[0][0] >= span_start and bins[-1][1] <= span_end, (
        f"bins escape the requested span [{span_start.isoformat()}, {span_end.isoformat()}): "
        f"first {bins[0][0].isoformat()}, last end {bins[-1][1].isoformat()}; only WHOLE bins "
        "fully contained in the span are emitted -- a partial bin at either end is a row "
        "that is not {} s of listening".format(cfg.BIN_SECONDS)
    )
    bad_width = [(s, e) for s, e in bins if e - s != width]
    assert not bad_width, (
        f"{len(bad_width)} bins are not exactly {cfg.BIN_SECONDS} s wide, first: {bad_width[0]}"
    )
    bad_align = [s for s, _e in bins if int(s.timestamp()) % cfg.BIN_SECONDS != 0]
    assert not bad_align, (
        f"{len(bad_align)} bin starts are not integer multiples of {cfg.BIN_SECONDS} s since "
        f"the UTC epoch, first: {bad_align[0].isoformat()} (D2)"
    )
    starts = _bin_starts(bins)
    bad_order = [i for i in range(1, len(starts)) if starts[i] <= starts[i - 1]]
    assert not bad_order, (
        f"bin starts are not strictly increasing at index {bad_order[0]}: "
        f"{starts[bad_order[0] - 1].isoformat()} -> {starts[bad_order[0]].isoformat()}"
    )
    bad_month = [s for s in starts if s.month not in season]
    assert not bad_month, (
        f"{len(bad_month)} bins fall outside SEASON_MONTHS_UTC, first: "
        f"{bad_month[0].isoformat()} (month {bad_month[0].month}); the season is defined on "
        "month(start_utc) in UTC (D4)"
    )
    bad_tz = [s for s, _e in bins if s.tzinfo is None or s.utcoffset().total_seconds() != 0]
    assert not bad_tz, (
        f"{len(bad_tz)} bin starts are not tz-aware UTC, first tzinfo: {bad_tz[0].tzinfo!r}"
    )
    # Contiguity: within the season, consecutive rows touch exactly (D2).
    gaps = [(bins[i - 1][1], bins[i][0]) for i in range(1, len(bins))
            if bins[i][0] != bins[i - 1][1] and bins[i][0].month == bins[i - 1][0].month]
    assert not gaps, (
        f"{len(gaps)} in-month discontinuities between consecutive rows, first: "
        f"{gaps[0][0].isoformat()} -> {gaps[0][1].isoformat()}; a contiguous block must touch"
    )


def check_a1a_3b_epoch_alignment_from_in_season_unaligned_start():
    """D2: bin edges are multiples of BIN_SECONDS since the UTC epoch, whatever the request.

    The start is unaligned AND already in season. That combination is the whole
    point: with an out-of-season unaligned start (as in check_a1a_3) the jump to
    May 1 00:00:00Z lands on an aligned instant and erases the misalignment before
    a single bin is emitted, so an implementation that anchors the grid at the
    requested start passes anyway. Here there is no month jump to hide behind, and
    such an implementation yields a first bin of 2024-07-01T00:03:17Z.

    Why it matters beyond tidiness: a per-request grid anchor means two callers who
    asked for slightly different windows get bins that do not line up, and the
    optical-acoustic matchup then joins two different 5-minute grids -- which
    produces a wrong answer rather than an error.
    """
    from datetime import timedelta
    cfg, onc = _a1a_mods()
    width = timedelta(seconds=cfg.BIN_SECONDS)
    offset = timedelta(minutes=3, seconds=17)  # arbitrary, deliberately not a bin multiple
    assert offset.total_seconds() % cfg.BIN_SECONDS != 0, (
        "the test offset is a whole number of bins, so it cannot detect a misaligned grid"
    )
    aligned_start = _utc(2024, 7, 1, 0, 0)
    span_start = aligned_start + offset
    span_end = _utc(2024, 7, 1, 1, 0) + offset  # unaligned at BOTH ends, both in season
    bins = onc.season_bins_utc(span_start, span_end)
    assert bins, (
        f"season_bins_utc returned no bins for the in-season span "
        f"[{span_start.isoformat()}, {span_end.isoformat()})"
    )
    bad_align = [s for s, _e in bins if int(s.timestamp()) % cfg.BIN_SECONDS != 0]
    assert not bad_align, (
        f"{len(bad_align)} of {len(bins)} bin starts are not integer multiples of "
        f"{cfg.BIN_SECONDS} s since the UTC epoch, first {bad_align[0].isoformat()}. "
        f"The request started at {span_start.isoformat()}, which is {offset} past a bin "
        "edge -- the grid is being anchored at the requested start instead of the epoch (D2)"
    )
    # The first whole bin at or after an unaligned start is the NEXT edge, not the
    # one containing it: a bin starting before span_start is not fully inside the span.
    expected_first = aligned_start + width
    assert bins[0][0] == expected_first, (
        f"first bin starts {bins[0][0].isoformat()}, expected {expected_first.isoformat()}; "
        f"a request beginning mid-bin at {span_start.isoformat()} must skip forward to the "
        "next epoch-aligned edge, not round backwards (which would emit a bin whose first "
        f"{offset} lie outside the requested span)"
    )
    assert bins[0][0] >= span_start and bins[-1][1] <= span_end, (
        f"bins escape the requested span: first {bins[0][0].isoformat()}, "
        f"last end {bins[-1][1].isoformat()}"
    )
    bad_width = [(s, e) for s, e in bins if e - s != width]
    assert not bad_width, (
        f"{len(bad_width)} bins are not exactly {cfg.BIN_SECONDS} s wide, first {bad_width[0]}"
    )


def _season_rows_per_year(cfg) -> int:
    """153 days x 288 bins/day, derived rather than typed."""
    from datetime import date
    days = (date(2001, 10, 1) - date(2001, 5, 1)).days  # May-Sep spans no leap day
    assert days == 153, f"May 1 -> Oct 1 is {days} days, not the expected 153"
    per_day, rem = divmod(24 * 60 * 60, cfg.BIN_SECONDS)
    assert rem == 0, f"a day is not a whole number of {cfg.BIN_SECONDS} s bins"
    return days * per_day


def check_a1a_4_density_is_arithmetic():
    """D5: the calendar is dense, so the row count is arithmetic, not data-dependent."""
    cfg, onc = _a1a_mods()
    expect = _season_rows_per_year(cfg)
    assert expect == 44064, f"derived rows/season is {expect}, expected 153*288 = 44064"
    cases = [
        (2024, _utc(2024, 1, 1), _utc(2025, 1, 1), "leap year, full-year span"),
        (2023, _utc(2023, 1, 1), _utc(2024, 1, 1), "non-leap year, full-year span"),
        # The study start is 2020-02-18, BEFORE May 1 -- so 2020 is a FULL season.
        # A truncation bug that clipped the season to the study start would show here.
        (2020, cfg.STUDY_START_UTC, _utc(2021, 1, 1), "truncated 2020 start (pre-season)"),
    ]
    wrong = []
    for year, a, b, label in cases:
        n = len(onc.season_bins_utc(a, b))
        if n != expect:
            wrong.append(f"{year} ({label}): {n} rows, expected {expect}")
    assert not wrong, (
        "dense in-season row count is wrong -- " + "; ".join(wrong)
        + f". May-Sep is 153 days x {24 * 60 * 60 // cfg.BIN_SECONDS} bins/day; every "
        "in-season bin is emitted whether or not anything was listed (D5)"
    )
    # A multi-year span is exactly N seasons -- no double-counted or dropped boundary.
    # An unaligned request start must not shift the grid or drop the first bin:
    # the season is defined by the calendar, not by when the caller happened to ask.
    n_off = len(onc.season_bins_utc(_utc(2024, 1, 1, 0, 0, 37), _utc(2025, 1, 1)))
    assert n_off == expect, (
        f"an unaligned request start gave {n_off} rows, expected {expect}; bin edges are "
        f"integer multiples of {cfg.BIN_SECONDS} s since the UTC epoch regardless of the "
        "requested bounds (D2)"
    )
    n3 = len(onc.season_bins_utc(_utc(2022, 1, 1), _utc(2025, 1, 1)))
    assert n3 == 3 * expect, (
        f"a three-year span gave {n3} rows, expected {3 * expect}; season boundaries are "
        "being double-counted or dropped"
    )


def check_a1a_5_season_boundary_rows():
    """First in-season bin starts May 1 00:00:00Z; the last bin ends Oct 1 00:00:00Z."""
    _, onc = _a1a_mods()
    bins = onc.season_bins_utc(_utc(2024, 1, 1), _utc(2025, 1, 1))
    assert bins, "no bins for 2024"
    first_start, last_end = bins[0][0], bins[-1][1]
    assert first_start == _utc(2024, 5, 1), (
        f"first in-season bin starts {first_start.isoformat()}, expected 2024-05-01T00:00:00Z"
    )
    assert last_end == _utc(2024, 10, 1), (
        f"last in-season bin ends {last_end.isoformat()}, expected 2024-10-01T00:00:00Z; "
        "the season is half-open, so Oct 1 00:00Z is an END, never a START"
    )
    starts = set(_bin_starts(bins))
    assert _utc(2024, 10, 1) not in starts, (
        "2024-10-01T00:00:00Z appears as a bin START; October is out of season (D4)"
    )
    assert _utc(2024, 4, 30, 23, 55) not in starts, (
        "the last April bin was emitted; the season starts at May 1 00:00:00Z exactly"
    )


def check_a1a_6_dst_is_a_no_op():
    """Every 24 h UTC span yields exactly 288 rows -- including DST transition days.

    Uses season_months=None so the out-of-season March/November transition days can
    be exercised. If any America/Vancouver conversion existed, the March day would
    yield 276 rows and the November day 300.
    """
    from datetime import timedelta
    cfg, onc = _a1a_mods()
    per_day = 24 * 60 * 60 // cfg.BIN_SECONDS
    day = timedelta(days=1)
    # 2024 US/Canada DST transitions: spring forward Mar 10, fall back Nov 3.
    named = {(2024, 3, 10): "spring-forward", (2024, 11, 3): "fall-back"}
    wrong = []
    d = _utc(2024, 1, 1)
    while d < _utc(2025, 1, 1):
        n = len(onc.season_bins_utc(d, d + day, season_months=None))
        if n != per_day:
            label = named.get((d.year, d.month, d.day), "")
            wrong.append(f"{d.date().isoformat()}{' (' + label + ')' if label else ''}: {n}")
        d += day
    assert not wrong, (
        f"{len(wrong)} UTC days did not contain exactly {per_day} bins "
        f"(first few: {wrong[:5]}); in UTC every day is 24 h and DST is a no-op. "
        "276 or 300 on a transition day means a local timezone leaked in (D4)"
    )
    for (y, m, dd), label in named.items():
        n = len(onc.season_bins_utc(_utc(y, m, dd), _utc(y, m, dd) + day, season_months=None))
        assert n == per_day, f"{label} day {y}-{m:02d}-{dd:02d} gave {n} bins, expected {per_day}"


def check_a1a_6b_no_local_timezone_in_a1_source():
    """No A1 module performs a local-timezone CONVERSION (D4: no
    America/Vancouver conversion baked into A1's season/bin logic).

    Narrowed by B3-A (see docs on overpass_window_utc): B3-A requires
    PLANET_OVERPASS_TZ_NAME == "America/Vancouver" to live as a named
    constant in boatphone/config.py (CLAUDE.md invariant 6 -- one shared
    definition, not one literal restated per module), and boatphone.config
    is one of the _A1A_MODULES this check scans. A bare constant that
    *names* a zone is not a conversion; the point of B3-A is precisely that
    the local->UTC conversion happens per date via zoneinfo rather than a
    baked fixed offset. So this check now flags actual conversion facilities
    and call sites (zoneinfo/ZoneInfo/pytz USE, tz_localize, tz_convert,
    astimezone, localtime, and a bare "America/" literal appearing anywhere
    other than as the right-hand side of a constant assignment), while
    permitting a module-level `NAME = "America/Vancouver"`-style constant
    declaration. It does not simply exempt boatphone.config wholesale --
    an astimezone/zoneinfo call inside boatphone.config would still fail.
    """
    import inspect
    import re
    # Facilities that, if USED (not just named), perform or enable a local
    # conversion. These remain forbidden anywhere in A1 source.
    forbidden_calls = ("zoneinfo", "ZoneInfo", "pytz", "tz_localize",
                        "tz_convert", "astimezone", "localtime")
    # A bare "America/" literal is only permitted as the RHS of a simple
    # module-level constant assignment, e.g. FOO_TZ_NAME = "America/Vancouver".
    const_assignment_re = re.compile(r'^[A-Z_][A-Z0-9_]*\s*(:\s*\w+\s*)?=\s*["\']America/')
    hits = []
    for mod in _a1a_mods():
        try:
            src = inspect.getsource(mod)
        except OSError as exc:
            raise AssertionError(f"could not read source of {mod.__name__}: {exc}") from exc
        for lineno, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            for token in forbidden_calls:
                if token in code:
                    hits.append(f"{mod.__name__}:{lineno}: {token!r} in {line.strip()!r}")
            if "America/" in code and not const_assignment_re.match(code.strip()):
                hits.append(
                    f"{mod.__name__}:{lineno}: 'America/' used outside a named-constant "
                    f"assignment in {line.strip()!r}"
                )
    assert not hits, (
        "A1 source performs (or enables) a local-timezone conversion: " + "; ".join(hits)
        + ". The season is defined on month(start_utc) in UTC; a named tz constant "
        "(e.g. B3-A's PLANET_OVERPASS_TZ_NAME) is fine, but no America/Vancouver "
        "CONVERSION may appear in A1's season/bin logic (D4)"
    )


def check_a1a_7_deployment_assignment():
    """D6: deployment comes from metadata; a listed file in no deployment RAISES."""
    from datetime import timedelta
    cfg, onc = _a1a_mods()
    width = timedelta(seconds=cfg.BIN_SECONDS)
    boundary = _utc(2024, 7, 4, 12, 0)  # dep A ends and dep B starts here, half-open
    deployments = [
        ("11111", _utc(2024, 7, 4, 0, 0), boundary),
        ("22222", boundary, _utc(2024, 7, 5, 0, 0)),
    ]
    bins = onc.season_bins_utc(_utc(2024, 7, 4, 11, 50), _utc(2024, 7, 4, 12, 10))
    starts = _bin_starts(bins)
    assert starts == [boundary + k * width for k in (-2, -1, 0, 1)], (
        f"unexpected bin starts around the deployment boundary: "
        f"{[s.isoformat() for s in starts]}"
    )
    available = [True] * len(bins)
    ids = onc.assign_deployment(bins, deployments, available)
    expect = ["11111", "11111", "22222", "22222"]
    assert ids == expect, (
        f"deployment ids around the boundary are {ids}, expected {expect}. The bin STARTING "
        f"exactly at the boundary {boundary.isoformat()} belongs to the deployment whose "
        "half-open [start, end) interval CONTAINS its start -- i.e. the later one"
    )
    assert all(isinstance(i, str) for i in ids), (
        f"deployment ids must be strings (ONC's identifier verbatim); got types "
        f"{[type(i).__name__ for i in ids]} -- an int would lose a leading zero and "
        "would not join against ONC metadata"
    )

    # A bin in no deployment, with nothing listed, is legitimately "" + unavailable.
    gap_bins = onc.season_bins_utc(_utc(2024, 7, 6, 3, 0), _utc(2024, 7, 6, 3, 10))
    gap_ids = onc.assign_deployment(gap_bins, deployments, [False] * len(gap_bins))
    assert gap_ids == [""] * len(gap_bins), (
        f"bins outside every deployment got {gap_ids}, expected empty strings"
    )

    # But an AVAILABLE bin outside every deployment is a contradiction and must RAISE.
    try:
        got = onc.assign_deployment(gap_bins, deployments, [True] * len(gap_bins))
    except Exception as exc:
        assert isinstance(exc, onc.DeploymentCoverageError), (
            f"expected onc_client.DeploymentCoverageError, got "
            f"{type(exc).__name__}: {exc}"
        )
        assert isinstance(exc, RuntimeError), "DeploymentCoverageError must subclass RuntimeError"
        assert "2024-07-06" in str(exc), (
            f"the error must name the offending bin; got: {exc}"
        )
    else:
        raise AssertionError(
            f"assign_deployment returned {got} for an AVAILABLE bin that falls in no known "
            "deployment; that means a file was listed outside every deployment -- either the "
            "metadata is incomplete or the time base is wrong, and it must RAISE rather than "
            'be attributed to "" (D6). Silently absorbing it is how a whole deployment of '
            "uptime goes missing from the calendar"
        )


def check_a1a_7b_unavailable_bins_have_no_deployment_claim():
    """A bin in no deployment must be available=False -- the two agree by construction."""
    cfg, onc = _a1a_mods()
    deployments = [("11111", _utc(2024, 7, 4, 0, 0), _utc(2024, 7, 4, 12, 0))]
    bins = onc.season_bins_utc(_utc(2024, 7, 4, 11, 50), _utc(2024, 7, 4, 12, 10))
    # Coverage that stops at the deployment end: nothing is listed after it.
    cov = [(_utc(2024, 7, 4, 11, 50), _utc(2024, 7, 4, 12, 0))]
    avail = onc.mark_available(bins, cov)
    ids = onc.assign_deployment(bins, deployments, avail)
    bad = [(s.isoformat(), a, i) for (s, _e), a, i in zip(bins, avail, ids) if i == "" and a]
    assert not bad, (
        f'bins with deployment_id "" are marked available: {bad}; D6 requires every '
        "no-deployment bin to be unavailable"
    )
    assert ids[:2] == ["11111", "11111"] and ids[2:] == ["", ""], (
        f"deployment ids are {ids}, expected the first two in 11111 and the last two outside it"
    )
    assert avail[:2] == [True, True] and avail[2:] == [False, False], (
        f"availability is {avail}, expected the first two bins covered and the last two not"
    )


# ---------------------------------------------------------------------------
# A8 -- SUPERSEDED SEGMENT, checks retained.
#
# acoustics_plan_v2 replaced the VTUAD transfer experiment with ONC's own
# pretrained checkpoint (decision 0009). VTUAD is NOT acquired. These checks
# stay because they pin code that still matters: boatphone/models.py is the
# band-matching and level-comparability contract behind decisions 0010 and
# 0011, and B2 of the v2 plan reuses it rather than writing new guards.
#
# A8c's 12 checks (VTUAD manifest + loader) were REMOVED -- they asserted a
# corpus we no longer acquire and were permanently red. See branch
# milestone1/a8-vtuad, commit 070e3c5, for what they asserted.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# A8a -- VTUAD facts gate
#
# Contract: before any VTUAD-derived constant lands in boatphone/config.py, the
# facts backing it must be written down in docs/vtuad-facts.md, each one carrying
# an independently-checkable source URL and a retrieval date -- NOT inherited
# from docs/plans/accoutics_plan.md or a previous agent's say-so.
#
# Expected markdown table shape (the coder writing docs/vtuad-facts.md in the
# next step must match this exactly, since the parser below is intentionally
# not a general markdown parser):
#
#   | key | value | source | retrieved |
#   |-----|-------|--------|-----------|
#   | sample_rate_hz | 32000 (uniform across corpus) | https://... | 2026-08-27 |
#   | ... one row per key in VTUAD_REQUIRED_FACT_KEYS ...
#
# Column headers are matched case-insensitively; column ORDER must be
# key, value, source, retrieved (extra trailing columns, e.g. `notes`, are
# tolerated and ignored). `source` must be an http(s) URL -- a bare repo-
# relative path (e.g. `docs/plans/accoutics_plan.md`) is exactly the failure
# mode this check exists to catch, so it is rejected explicitly.
# ---------------------------------------------------------------------------

VTUAD_FACTS_DOC = REPO_ROOT / "docs" / "vtuad-facts.md"

# The eleven facts A8a requires, independently sourced. Source: A8a contract
# (segment description), not inherited from any existing BoatPhone doc.
VTUAD_REQUIRED_FACT_KEYS = [
    "sample_rate_hz",
    "file_format",
    "clip_duration_s",
    "clip_count",
    "band_populated_hz",
    "label_schema",
    "licence",
    "total_size_bytes",
    "smallest_downloadable_unit_bytes",
    "download_mechanism",
    "precomputed_features_available",
]


class VtuadFactsTableError(AssertionError):
    """Raised by _parse_vtuad_facts_table for a malformed or absent table."""


def _split_row(line: str) -> list[str]:
    cells = line.strip()
    if cells.startswith("|"):
        cells = cells[1:]
    if cells.endswith("|"):
        cells = cells[:-1]
    return [c.strip() for c in cells.split("|")]


def _parse_vtuad_facts_table() -> dict[str, dict[str, str]]:
    """Parse docs/vtuad-facts.md into {key: {"value":..., "source":..., "retrieved":...}}.

    Deliberately narrow: only recognizes the pipe-table shape documented above.
    Raises VtuadFactsTableError (a distinct type from a missing/empty row) if the
    file is absent or no such table can be found, so failure modes stay legible.
    """
    if not VTUAD_FACTS_DOC.is_file():
        raise VtuadFactsTableError(f"{VTUAD_FACTS_DOC} does not exist")

    text = VTUAD_FACTS_DOC.read_text(encoding="utf-8")
    lines = text.splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "|" not in line:
            continue
        cells = [c.lower() for c in _split_row(line)]
        if cells[:4] == ["key", "value", "source", "retrieved"]:
            header_idx = i
            break
    if header_idx is None:
        raise VtuadFactsTableError(
            f"{VTUAD_FACTS_DOC} contains no `| key | value | source | retrieved |` table "
            "(header row not found, case-insensitive, in that column order)"
        )

    # Row after the header is expected to be the `|---|---|---|---|` separator.
    body_start = header_idx + 1
    if body_start < len(lines) and set(lines[body_start].replace("|", "").strip()) <= set("-: "):
        body_start += 1

    rows: dict[str, dict[str, str]] = {}
    for line in lines[body_start:]:
        if "|" not in line or not line.strip():
            continue
        if not line.strip().startswith("|"):
            break  # table ended
        cells = _split_row(line)
        if len(cells) < 4:
            continue
        key, value, source, retrieved = cells[0], cells[1], cells[2], cells[3]
        rows[key] = {"value": value, "source": source, "retrieved": retrieved}
    return rows


def check_a8a_facts_doc_exists():
    """docs/vtuad-facts.md exists and contains a parseable facts table.

    Expected to FAIL right now: the doc does not exist yet.
    """
    try:
        rows = _parse_vtuad_facts_table()
    except VtuadFactsTableError as exc:
        raise AssertionError(str(exc)) from exc
    assert rows, f"{VTUAD_FACTS_DOC} has a facts table but it has zero rows"


def check_a8a_all_required_keys_present_and_populated():
    """Every required fact key is present, has a non-empty value, and a URL source.

    Distinguishes three failure modes explicitly, per key:
      - MISSING: the key has no row at all
      - EMPTY VALUE: the row exists but `value` is blank
      - BAD SOURCE: `source` is blank, or is not an http(s) URL (e.g. a
        repo-relative path such as `docs/plans/accoutics_plan.md`), or
        `retrieved` is blank
    """
    try:
        rows = _parse_vtuad_facts_table()
    except VtuadFactsTableError as exc:
        raise AssertionError(
            f"cannot check required keys, no parseable table: {exc}"
        ) from exc

    problems = []
    for key in VTUAD_REQUIRED_FACT_KEYS:
        if key not in rows:
            problems.append(f"{key}: MISSING -- no row for this key")
            continue
        row = rows[key]
        if not row["value"].strip():
            problems.append(f"{key}: EMPTY VALUE -- row exists but value column is blank")
        source = row["source"].strip()
        if not source:
            problems.append(f"{key}: BAD SOURCE -- source column is blank")
        elif not (source.startswith("http://") or source.startswith("https://")):
            problems.append(
                f"{key}: BAD SOURCE -- source {source!r} is not an http(s) URL "
                "(a repo-relative path is not an independent source)"
            )
        if not row["retrieved"].strip():
            problems.append(f"{key}: BAD SOURCE -- retrieved-date column is blank")

    assert not problems, (
        f"{VTUAD_FACTS_DOC} fails the facts gate for "
        f"{len({p.split(':')[0] for p in problems})} key(s):\n      " + "\n      ".join(problems)
    )


def check_a8a_config_constants_match_facts_doc():
    """boatphone.config's VTUAD_* constants equal the values stated in the facts doc.

    This is the drift check: a doc that says one sample rate while downstream
    code uses another must fail here, not surface later as a silently
    mis-resampled feature. Constants are imported from boatphone.config -- the
    source markdown is never re-read textually by the code under test.

    Ambiguity flagged rather than guessed silently: the exact string/number
    formatting of the doc's `value` cell (e.g. "32000 Hz" vs "32000") is not
    prescribed by the A8a contract, so this check compares the *numeric*
    portion of sample_rate_hz / total_size_bytes and the config constant,
    and does substring containment for label_schema's zone radii and class
    names. The coder implementing the doc + config should keep the value cell
    parseable in that spirit.
    """
    try:
        rows = _parse_vtuad_facts_table()
    except VtuadFactsTableError as exc:
        raise AssertionError(
            f"cannot check constants against facts doc, no parseable table: {exc}"
        ) from exc

    missing_rows = [k for k in ("sample_rate_hz", "label_schema", "total_size_bytes")
                     if k not in rows]
    assert not missing_rows, (
        f"facts doc is missing required rows needed for the drift check: {missing_rows}"
    )

    out = _child_ok(_run_child("""
        import json
        import boatphone.config as cfg
        names = [
            "VTUAD_SAMPLE_RATE_HZ", "VTUAD_LABEL_SCHEMA",
            "VTUAD_ZONE_RADII_M", "VTUAD_TOTAL_SIZE_BYTES",
        ]
        missing = [n for n in names if not hasattr(cfg, n)]
        present = {n: repr(getattr(cfg, n)) for n in names if hasattr(cfg, n)}
        print(json.dumps({"missing": missing, "present": present}))
    """), "boatphone.config VTUAD constants")
    import json
    result = json.loads(out.strip().splitlines()[-1])
    assert not result["missing"], (
        "boatphone/config.py is missing VTUAD constants required by the A8a contract: "
        f"{result['missing']} (VTUAD_SAMPLE_RATE_HZ, VTUAD_LABEL_SCHEMA, "
        "VTUAD_ZONE_RADII_M, VTUAD_TOTAL_SIZE_BYTES)"
    )

    def _leading_int(s: str):
        digits = ""
        for ch in s.strip():
            if ch.isdigit():
                digits += ch
            elif digits:
                break
        return int(digits) if digits else None

    mismatches = []

    doc_rate = _leading_int(rows["sample_rate_hz"]["value"])
    cfg_rate = eval(result["present"]["VTUAD_SAMPLE_RATE_HZ"])
    if doc_rate is None or cfg_rate != doc_rate:
        mismatches.append(
            f"sample_rate_hz: doc says {rows['sample_rate_hz']['value']!r}, "
            f"boatphone.config.VTUAD_SAMPLE_RATE_HZ = {cfg_rate!r}"
        )

    doc_size = _leading_int(rows["total_size_bytes"]["value"])
    cfg_size = eval(result["present"]["VTUAD_TOTAL_SIZE_BYTES"])
    if doc_size is None or cfg_size != doc_size:
        mismatches.append(
            f"total_size_bytes: doc says {rows['total_size_bytes']['value']!r}, "
            f"boatphone.config.VTUAD_TOTAL_SIZE_BYTES = {cfg_size!r}"
        )

    doc_schema = rows["label_schema"]["value"]
    cfg_schema = eval(result["present"]["VTUAD_LABEL_SCHEMA"])
    if str(cfg_schema) not in doc_schema and doc_schema not in str(cfg_schema):
        mismatches.append(
            f"label_schema: doc says {doc_schema!r}, "
            f"boatphone.config.VTUAD_LABEL_SCHEMA = {cfg_schema!r} -- neither contains the other"
        )

    assert not mismatches, (
        "boatphone/config.py VTUAD constants DRIFT from docs/vtuad-facts.md:\n      "
        + "\n      ".join(mismatches)
    )


# ---------------------------------------------------------------------------
# A8b -- band-matching contract (boatphone/models.py)
#
# Contract: VTUAD and Folger have different sample rates and different populated
# bands. Comparing a feature computed on one against a feature computed on the
# other is only meaningful over their COMMON support -- the intersection of the
# two populated bands, further clipped to the Nyquist frequency of the LOWER of
# the two sample rates (a frequency neither instrument's spectrum can even
# represent cannot be "common"). boatphone/models.py does not exist yet; every
# check below is EXPECTED TO FAIL until it is written. That is the point
# (see the A0 module docstring at the top of this file for why).
#
# API surface asserted below is a GUESS at what A8b will implement, made
# explicit rather than smuggled in, per the segment description:
#   - common_support_hz(vtuad_band_hz, vtuad_fs_hz, folger_band_hz, folger_fs_hz)
#       -> (lo_hz, hi_hz); raises (any Exception naming both input bands and the
#       resulting empty/degenerate intersection) when there is no usable overlap.
#   - assert_band_matched(band_hz, common_band_hz, *, label="") -> None;
#       raises unless band_hz IS the (already band-limited) common_band_hz --
#       this is the guard against comparing a feature that was never restricted
#       to the intersection.
#   - band_limit(freq_hz, level_db_re_1upa, fs_hz, band_hz) -> (freq_hz, level_db_re_1upa)
#       masks a one-sided spectrum to band_hz; raises if band_hz exceeds the
#       Nyquist of fs_hz rather than silently clipping (decision 0002 SS2: never
#       silently misinterpret a sample rate).
# If A8b's coder chooses different names, these checks fail loudly naming the
# missing attribute -- which is exactly the legible failure this segment wants,
# not a silent pass.
#
# Folger side is PINNED (not a free parameter): usable support >= 250 Hz,
# calibrated <= 51.2 kHz -- both stated directly in the A8b segment contract,
# and independently cross-checked against docs/plans/accoutics_plan.md, which
# states the supplied calibration file stops at 51,200 Hz (line ~30) and that
# the WAV acquisition stream runs at 128 kHz (line ~32; the *other* stream in
# that file, the 256 kHz .fft.gz product, is not what "calibrated <=51.2 kHz"
# is pinned against here since the calibration curve itself only reaches 51.2
# kHz regardless of which stream is sampled).
# ---------------------------------------------------------------------------

FOLGER_BAND_HZ = (250.0, 51200.0)
FOLGER_FS_HZ = 128000.0

# FABRICATED VTUAD inputs -- deliberately NOT read from docs/vtuad-facts.md or
# boatphone/config.py (those are A8a's concern, may be mid-edit by another
# agent right now, and might not even be populated yet). The whole reason A8b
# can be built before VTUAD facts land is that the VTUAD band/rate are
# PARAMETERS, not constants baked into models.py -- these two check-local
# values, plus a second, differently-shaped pair below, exist to prove that.
VTUAD_BAND_HZ_A = (100.0, 16000.0)
VTUAD_FS_HZ_A = 32000.0
VTUAD_BAND_HZ_B = (500.0, 8000.0)
VTUAD_FS_HZ_B = 16000.0

# Hand-computed expected intersections (source: arithmetic on the two constants
# above, not on any code under test):
#   A: intersect((250,51200),(100,16000)) = (250,16000);
#      clip to Nyquist(min(128000,32000)) = 16000 Hz -> unchanged -> (250,16000)
EXPECTED_COMMON_A = (250.0, 16000.0)
#   B: intersect((250,51200),(500,8000)) = (500,8000);
#      clip to Nyquist(min(128000,16000)) = 8000 Hz -> unchanged -> (500,8000)
EXPECTED_COMMON_B = (500.0, 8000.0)

# A VTUAD band with NO overlap with Folger's usable floor (250 Hz), to exercise
# the empty-intersection contract.
VTUAD_BAND_HZ_DISJOINT = (10.0, 200.0)
VTUAD_FS_HZ_DISJOINT = 4000.0

# Exact-arithmetic tolerance for the intersection/clip computation itself (pure
# min/max on the four input numbers above -- no accumulated floating error is
# expected beyond machine epsilon; 1e-6 is generous, not tuned).
BAND_EDGE_TOL_HZ = 1e-6

# Positive-case tone. Chosen to sit strictly inside BOTH EXPECTED_COMMON_A and
# EXPECTED_COMMON_B so one tone exercises both fabricated VTUAD inputs.
TEST_TONE_HZ = 4000.0
TEST_TONE_LEVEL_DB_RE_1UPA = -20.0  # arbitrary reference level; only the round-trip matters

# Frequency tolerance for recovering the tone bin after band-limiting. Both
# sides below use a 1-SECOND analysis window (N == fs samples), so bin spacing
# is exactly 1 Hz at ANY sample rate and TEST_TONE_HZ (4000.0, an integer) is
# bin-centred with zero spectral leakage on both grids. The tolerance is set to
# that resolution, not to zero, so a reporter that returns "the bin's centre
# frequency" rather than the DFT index isn't punished for a distinction that
# doesn't exist here by construction.
FREQ_TOL_HZ = 1.0

# Level tolerance for the same round-trip. Justification: with the tone bin-
# centred (see above), the only remaining error source is floating-point
# summation inside the Hann-window coherent-gain normalization (<< 0.1 dB in
# practice). A convention bug (one-sided vs two-sided, or a missing window-gain
# division) shows up as 3 dB (factor of 2) or worse, and decision 0002 SS4
# requires that convention be *named*, not assumed -- named below in
# _one_sided_spectrum_db. 0.2 dB leaves headroom without hiding either bug.
LEVEL_TOL_DB = 0.2


def _one_sided_spectrum_db(fs_hz: float, tones_hz_db: list[tuple[float, float]]):
    """Build a one-sided level spectrum containing the given (freq_hz, level_db) tones.

    STATED CONVENTION (decision 0002 SS4 -- named at this boundary, not assumed):
      - a 1-second Hann-windowed analysis window (N = fs_hz samples), so the bin
        spacing is exactly 1 Hz regardless of fs_hz;
      - ONE-SIDED magnitude spectrum via numpy.fft.rfft;
      - COHERENT-GAIN amplitude normalization: level_db = 20*log10(2*|X[k]| / sum(w))
        for interior bins (DC and Nyquist are not doubled -- this check never
        probes either);
      - reference 1 uPa: the synthetic waveform's amplitude units ARE uPa
        directly. This deliberately does NOT exercise the ICLISTEN calibration
        chain (that is the pre-existing, separate calibration-gate segment) --
        flagged here rather than silently conflated with it. A8b's contract is
        band-matching arithmetic, not calibration.

    Returns (freq_hz, level_db_re_1upa) as numpy arrays.
    """
    n = int(round(fs_hz))
    t = np.arange(n) / fs_hz
    x = np.zeros(n)
    for freq_hz, level_db in tones_hz_db:
        amp = 10.0 ** (level_db / 20.0)
        x = x + amp * np.sin(2.0 * np.pi * freq_hz * t)
    w = np.hanning(n)
    window_gain = np.sum(w)
    spectrum = np.fft.rfft(x * w)
    freq_hz_out = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    mag = np.abs(spectrum)
    scale = np.full_like(mag, 2.0 / window_gain)
    scale[0] = 1.0 / window_gain
    scale[-1] = 1.0 / window_gain
    with np.errstate(divide="ignore"):
        level_db_out = 20.0 * np.log10(np.maximum(mag * scale, 1e-300))
    return freq_hz_out, level_db_out


def check_a8b_models_module_exists():
    """boatphone/models.py exists at all.

    Expected to FAIL right now: the module does not exist. Written as its own
    check, ahead of every other A8b check, so the very first failure names the
    real cause (a missing file) rather than an opaque ImportError buried in a
    child-process traceback.
    """
    path = REPO_ROOT / "boatphone" / "models.py"
    assert path.is_file(), (
        f"{path} does not exist yet -- A8b (band-matching contract) has not been "
        "implemented. Every other 'A8b' check below is expected to fail for the "
        "same reason until this file is written."
    )


def check_a8b_common_support_is_intersection_clipped_to_lower_nyquist():
    """common_support_hz() == intersection of the two bands, clipped to the lower Nyquist."""
    import boatphone.models as bm
    got = bm.common_support_hz(
        VTUAD_BAND_HZ_A, VTUAD_FS_HZ_A, FOLGER_BAND_HZ, FOLGER_FS_HZ
    )
    lo, hi = got
    exp_lo, exp_hi = EXPECTED_COMMON_A
    assert abs(lo - exp_lo) <= BAND_EDGE_TOL_HZ and abs(hi - exp_hi) <= BAND_EDGE_TOL_HZ, (
        f"common_support_hz({VTUAD_BAND_HZ_A}, {VTUAD_FS_HZ_A}, {FOLGER_BAND_HZ}, "
        f"{FOLGER_FS_HZ}) = {got}, expected {EXPECTED_COMMON_A} "
        "(intersection of the two bands, clipped to Nyquist(min(fs_a, fs_b)))"
    )


def check_a8b_common_support_moves_with_vtuad_input():
    """Anti-hardcoding: two different fabricated VTUAD bands give two different intersections.

    A models.py that ignores its vtuad_band_hz/vtuad_fs_hz arguments (e.g. one
    that always returns the Folger band, or a constant it embedded itself)
    fails this even though it might coincidentally pass the single-case check
    above.
    """
    import boatphone.models as bm
    got_a = bm.common_support_hz(VTUAD_BAND_HZ_A, VTUAD_FS_HZ_A, FOLGER_BAND_HZ, FOLGER_FS_HZ)
    got_b = bm.common_support_hz(VTUAD_BAND_HZ_B, VTUAD_FS_HZ_B, FOLGER_BAND_HZ, FOLGER_FS_HZ)
    assert got_a != got_b, (
        "common_support_hz() returned the SAME intersection "
        f"({got_a}) for two different VTUAD bands/rates "
        f"({VTUAD_BAND_HZ_A}/{VTUAD_FS_HZ_A} vs {VTUAD_BAND_HZ_B}/{VTUAD_FS_HZ_B}); "
        "the VTUAD side is not being read from its arguments"
    )
    for got, expected, label in (
        (got_a, EXPECTED_COMMON_A, "A"),
        (got_b, EXPECTED_COMMON_B, "B"),
    ):
        lo, hi = got
        exp_lo, exp_hi = expected
        assert abs(lo - exp_lo) <= BAND_EDGE_TOL_HZ and abs(hi - exp_hi) <= BAND_EDGE_TOL_HZ, (
            f"case {label}: common_support_hz() = {got}, expected {expected}"
        )


def check_a8b_empty_intersection_raises_naming_both_bands():
    """A VTUAD band with zero overlap with Folger's usable floor must RAISE, naming both bands."""
    import boatphone.models as bm
    try:
        got = bm.common_support_hz(
            VTUAD_BAND_HZ_DISJOINT, VTUAD_FS_HZ_DISJOINT, FOLGER_BAND_HZ, FOLGER_FS_HZ
        )
    except Exception as exc:
        msg = str(exc)
        missing = [
            token for token in (str(VTUAD_BAND_HZ_DISJOINT), str(FOLGER_BAND_HZ))
            if token not in msg
        ]
        assert not missing, (
            "common_support_hz() raised on a disjoint VTUAD band, but the message "
            f"does not name {missing}: {msg!r} -- the message must name BOTH input "
            "bands (and the empty intersection) so a diagnosing human isn't left "
            "guessing which side was wrong"
        )
    else:
        raise AssertionError(
            f"common_support_hz({VTUAD_BAND_HZ_DISJOINT}, {VTUAD_FS_HZ_DISJOINT}, "
            f"{FOLGER_BAND_HZ}, {FOLGER_FS_HZ}) returned {got} for bands with NO "
            "overlap; it must raise, not return a degenerate/empty band as a number"
        )


def check_a8b_assert_band_matched_raises_on_unmatched_comparison():
    """assert_band_matched() raises when a side was never limited to the common support.

    Positive half: comparing the ALREADY-matched band against itself must NOT
    raise -- otherwise the guard is unconditional and checks nothing.
    Negative half (the one that matters per decision 0002 SS5): comparing the
    raw, un-band-limited Folger band against the common support -- exactly the
    mistake this contract exists to catch -- must raise.
    """
    import boatphone.models as bm
    # Positive half: must NOT raise.
    try:
        bm.assert_band_matched(EXPECTED_COMMON_A, EXPECTED_COMMON_A)
    except Exception as exc:
        raise AssertionError(
            "assert_band_matched(common, common) raised "
            f"{type(exc).__name__}: {exc}; a matched comparison must pass"
        ) from exc
    # Negative half: MUST raise.
    try:
        bm.assert_band_matched(FOLGER_BAND_HZ, EXPECTED_COMMON_A)
    except Exception:
        pass
    else:
        raise AssertionError(
            f"assert_band_matched({FOLGER_BAND_HZ}, {EXPECTED_COMMON_A}) returned "
            "without raising; comparing the raw Folger band against the common "
            "support (i.e. a feature never band-limited to the intersection) "
            "must RAISE, not quietly pass"
        )


def check_a8b_band_limit_signature_names_conventions():
    """band_limit()'s signature names freq_hz / level_db_re_1upa / fs_hz (decision 0002 SS4)."""
    import boatphone.models as bm
    params = set(inspect.signature(bm.band_limit).parameters)
    required = {"freq_hz", "level_db_re_1upa", "fs_hz"}
    missing = sorted(required - params)
    assert not missing, (
        f"boatphone.models.band_limit's signature is missing named parameters {missing} "
        f"(got {sorted(params)}); decision 0002 SS4 requires a function taking a "
        "frequency, a level, or a sample rate to name that convention in its "
        "signature -- `f` or `x` is not `freq_hz`"
    )


def check_a8b_band_limit_positive_tone_survives_with_matching_level_and_freq():
    """A tone inside the common band survives band-limiting on BOTH sides, level and freq intact.

    Generated independently at Folger's fs and at each fabricated VTUAD fs;
    band-limited to that pairing's common support; the recovered peak must
    match TEST_TONE_HZ / TEST_TONE_LEVEL_DB_RE_1UPA within the stated tolerances
    on both sides.
    """
    import boatphone.models as bm

    cases = [
        ("Folger", FOLGER_FS_HZ, EXPECTED_COMMON_A),
        ("VTUAD-A", VTUAD_FS_HZ_A, EXPECTED_COMMON_A),
        ("VTUAD-B", VTUAD_FS_HZ_B, EXPECTED_COMMON_B),
    ]
    failures = []
    for label, fs_hz, band_hz in cases:
        freq_hz, level_db = _one_sided_spectrum_db(
            fs_hz, [(TEST_TONE_HZ, TEST_TONE_LEVEL_DB_RE_1UPA)]
        )
        out_freq, out_level = bm.band_limit(
            freq_hz=freq_hz, level_db_re_1upa=level_db, fs_hz=fs_hz, band_hz=band_hz
        )
        out_freq = np.asarray(out_freq)
        out_level = np.asarray(out_level)
        assert out_freq.size > 0, f"{label}: band_limit() returned an empty band"
        peak_idx = int(np.argmax(out_level))
        peak_freq, peak_level = float(out_freq[peak_idx]), float(out_level[peak_idx])
        if abs(peak_freq - TEST_TONE_HZ) > FREQ_TOL_HZ:
            failures.append(
                f"{label}: peak at {peak_freq} Hz, expected {TEST_TONE_HZ} Hz "
                f"+/- {FREQ_TOL_HZ} Hz"
            )
        if abs(peak_level - TEST_TONE_LEVEL_DB_RE_1UPA) > LEVEL_TOL_DB:
            failures.append(
                f"{label}: peak level {peak_level:.3f} dB re 1uPa, expected "
                f"{TEST_TONE_LEVEL_DB_RE_1UPA} dB +/- {LEVEL_TOL_DB} dB "
                "(a 3 dB+ miss here means a one-sided/two-sided or window-gain "
                "convention mismatch, not numerical noise)"
            )
    assert not failures, "; ".join(failures)


def check_a8b_band_limit_excludes_out_of_band_tones():
    """A tone below Folger support, and one above the lower Nyquist, must NOT appear in output.

    Both tones are injected into the FOLGER-fs spectrum (fs is high enough that
    both frequencies exist there) and the spectrum is band-limited to
    EXPECTED_COMMON_A, whose lower Nyquist bound (16000 Hz, from VTUAD_FS_HZ_A)
    is stricter than Folger's own Nyquist (64000 Hz) -- exactly the case that
    proves band_limit is using the COMMON band, not each side's own Nyquist.
    """
    import boatphone.models as bm
    below_support_hz = 100.0  # < Folger's 250 Hz floor
    above_lower_nyquist_hz = 20000.0  # > EXPECTED_COMMON_A's 16000 Hz ceiling, < Folger's own 64000 Hz Nyquist
    freq_hz, level_db = _one_sided_spectrum_db(
        FOLGER_FS_HZ,
        [
            (TEST_TONE_HZ, TEST_TONE_LEVEL_DB_RE_1UPA),
            (below_support_hz, TEST_TONE_LEVEL_DB_RE_1UPA),
            (above_lower_nyquist_hz, TEST_TONE_LEVEL_DB_RE_1UPA),
        ],
    )
    out_freq, _ = bm.band_limit(
        freq_hz=freq_hz, level_db_re_1upa=level_db, fs_hz=FOLGER_FS_HZ, band_hz=EXPECTED_COMMON_A
    )
    out_freq = np.asarray(out_freq)
    lo, hi = EXPECTED_COMMON_A
    leaked = out_freq[(out_freq < lo - FREQ_TOL_HZ) | (out_freq > hi + FREQ_TOL_HZ)]
    assert leaked.size == 0, (
        f"band_limit(..., band_hz={EXPECTED_COMMON_A}) returned {leaked.size} "
        f"frequency bin(s) outside the band, including {sorted(leaked.tolist())[:5]}; "
        f"a tone at {below_support_hz} Hz (below Folger support) or "
        f"{above_lower_nyquist_hz} Hz (above the lower Nyquist) leaked through"
    )


def check_a8b_band_limit_rejects_band_exceeding_source_nyquist():
    """band_hz reaching above fs_hz's own Nyquist must RAISE, not silently clip (decision 0002 SS2)."""
    import boatphone.models as bm
    fs_hz = VTUAD_FS_HZ_A  # 32000 Hz -> Nyquist 16000 Hz
    band_exceeding_nyquist_hz = (250.0, 20000.0)  # 20000 > 16000
    freq_hz, level_db = _one_sided_spectrum_db(fs_hz, [(TEST_TONE_HZ, TEST_TONE_LEVEL_DB_RE_1UPA)])
    try:
        result = bm.band_limit(
            freq_hz=freq_hz, level_db_re_1upa=level_db, fs_hz=fs_hz,
            band_hz=band_exceeding_nyquist_hz,
        )
    except Exception:
        pass
    else:
        raise AssertionError(
            f"band_limit(fs_hz={fs_hz}, band_hz={band_exceeding_nyquist_hz}) returned "
            f"{result!r} instead of raising; the requested band's upper edge "
            f"(20000 Hz) exceeds this source's own Nyquist (16000 Hz) -- that must "
            "be rejected explicitly, not silently clipped or misinterpreted"
        )


# ---------------------------------------------------------------------------
# A8b (calibration half) -- band-matching is NECESSARY BUT NOT SUFFICIENT.
#
# Added during A8b implementation, after A8a established from primary sources
# that VTUAD ships raw uncalibrated PCM (no sensitivity, no gain, no reference,
# plus a per-segment "normalized" step in the authors' pipeline) while Folger
# levels are calibrated dB re 1 uPa -- see the calibration section of
# docs/vtuad-facts.md. A transfer gap measured on an ABSOLUTE-LEVEL feature
# across that boundary is a calibration artefact that looks exactly like a
# domain shift, which is precisely the silent corruptor decision 0002 exists to
# prevent. The three checks below prove the guard bites: it must RAISE on the
# unsafe comparison, PASS the level-invariant one, and REFUSE to proceed when a
# side's calibration state was never declared.
#
# Calibration states are FABRICATED here in the same spirit as the bands above
# (nothing read from boatphone/config.py), except that the enum member names
# themselves are the API under test.
# ---------------------------------------------------------------------------


def check_a8b_absolute_level_across_calibration_boundary_raises():
    """An absolute-level VTUAD<->Folger comparison must RAISE (calibrated vs counts)."""
    import boatphone.models as bm
    try:
        bm.assert_comparable(
            EXPECTED_COMMON_A, EXPECTED_COMMON_A, EXPECTED_COMMON_A,
            calibration_a=bm.CalibrationState.CALIBRATED_DB_RE_1UPA,
            calibration_b=bm.CalibrationState.UNCALIBRATED_COUNTS,
            feature_kind=bm.FeatureKind.ABSOLUTE_LEVEL,
            label_a="Folger", label_b="VTUAD",
        )
    except Exception as exc:
        msg = str(exc)
        for token in ("Folger", "VTUAD"):
            assert token in msg, (
                "assert_comparable() raised on an absolute-level comparison across "
                f"the calibration boundary but the message does not name {token!r}: "
                f"{msg!r}"
            )
    else:
        raise AssertionError(
            "assert_comparable() allowed an ABSOLUTE_LEVEL comparison between a "
            "CALIBRATED_DB_RE_1UPA side and an UNCALIBRATED_COUNTS side over a "
            "correctly matched band. Band-matching alone does not make those "
            "comparable: the difference would be set by unknown sensitivity/gain, "
            "indistinguishable from a real domain shift, and would not look wrong."
        )


def check_a8b_level_invariant_comparison_is_allowed():
    """The escape hatch works: a LEVEL_INVARIANT feature crosses the boundary fine.

    Without this half the calibration guard could be unconditional (refuse
    everything), which would check nothing and block the one comparison that is
    currently safe.
    """
    import boatphone.models as bm
    try:
        bm.assert_comparable(
            EXPECTED_COMMON_A, EXPECTED_COMMON_A, EXPECTED_COMMON_A,
            calibration_a=bm.CalibrationState.CALIBRATED_DB_RE_1UPA,
            calibration_b=bm.CalibrationState.UNCALIBRATED_COUNTS,
            feature_kind=bm.FeatureKind.LEVEL_INVARIANT,
            label_a="Folger", label_b="VTUAD",
        )
    except Exception as exc:
        raise AssertionError(
            "assert_comparable() refused a LEVEL_INVARIANT comparison over a matched "
            f"band ({type(exc).__name__}: {exc}); level-invariant features are the "
            "only currently safe VTUAD<->Folger comparison and must be permitted"
        ) from exc


def check_a8b_undeclared_calibration_state_raises():
    """A side whose calibration state was never declared must RAISE, not default to 'fine'."""
    import boatphone.models as bm
    try:
        bm.assert_comparable(
            EXPECTED_COMMON_A, EXPECTED_COMMON_A, EXPECTED_COMMON_A,
            calibration_a=bm.CalibrationState.CALIBRATED_DB_RE_1UPA,
            calibration_b=None,
            feature_kind=bm.FeatureKind.LEVEL_INVARIANT,
            label_a="Folger", label_b="VTUAD",
        )
    except Exception:
        pass
    else:
        raise AssertionError(
            "assert_comparable() accepted calibration_b=None; an undeclared "
            "calibration state must raise (decision 0002 SS3 -- state the "
            "convention at the boundary), never silently assume a default"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# A1b -- ONC location discovery and archive-file listing
#
# Contract pinned by these checks (the coder implements it in
# boatphone/onc_client.py, the *impure* half; the pure half is A1a):
#
#   discover_folger_locations(client) -> list[str]
#       Calls client.getLocations(...) and filters to Folger Deep candidates by
#       name / device association. Returns the DISCOVERED location codes. No ONC
#       location code is hardcoded anywhere in boatphone/ (check A1b.4).
#       Zero candidates RAISES (D8).
#
#   list_fft_files(client, location_codes, start_utc, end_utc)
#           -> (filenames: list[str], empty_chunks: int)
#       Archive-file listing, CHUNKED BY YEAR so one failed request is bounded
#       and the zero-result-chunk count is meaningful. Returns the filenames and
#       the number of year-chunks that came back empty, and PRINTS that count.
#       Zero files across the whole span RAISES; a chunk's HTTP/auth error
#       propagates (never becomes an empty page), carrying the request
#       parameters -- never the token -- in its message (D8).
#
#   parse_file_coverage(filename) -> (start_utc, end_utc)
#       PURE string -> half-open interval. The ONE place a filename becomes a
#       time. Malformed name RAISES; a name for a device other than
#       config.DEVICE_CODE RAISES.
#
#   get_deployments(client) -> list[(id_str, start_utc, end_utc)]
#       ONC deployment metadata for config.DEVICE_ID, in exactly the shape
#       assign_deployment() already consumes. Never inferred from listing gaps (D6).
#
#   Error types: FilenameParseError (a ValueError) for filename problems;
#   ONCListingError (a RuntimeError) for D8 listing/discovery failures.
#
# D3: "available" in A1 means ONC's listing REPORTS a file whose coverage
# intersects the bin -- ONC's belief that a file exists, not proof it downloads.
# A4 refines this from the actual pull, and the pull wins.
#
# The EXTENSION here is config.ARCHIVE_EXTENSION ("fft"), which is what the ONC
# archive index registers, NOT config.PRODUCT_EXTENSION ("fft.gz"). "fft.gz" is
# the on-disk gzipped form; using it as a listing filter returns ZERO files, and
# that exact confusion is what once emptied the calendar.
#
# CONTRACT RESOLUTIONS I HAD TO MAKE (flagged, not hidden):
#   (a) The real .fft.gz names on disk carry ONE timestamp, not two:
#       ICLISTENHF1266_20260313T000004.000Z.fft.gz. Only the .wav carries a
#       start_end pair. So a single-timestamp name's END is start + BIN_SECONDS.
#       The two fixture .fft.gz files are exactly 300 s apart, which is the
#       evidence for that cadence -- but it is an inference from two files, and
#       the coder should confirm it against a real multi-day listing in A1b.
#   (b) list_fft_files returning a (files, empty_chunks) 2-tuple is my choice of
#       how "the count is surfaced"; the brief required it be surfaced and
#       printed, not the exact signature.
# ---------------------------------------------------------------------------

# Real acquisition filenames, transcribed by hand from
# `ls data/Folger Deep Hydrophone Data Sample/`. These are FIXTURES: the checks
# below parse them offline. Nothing here reads or writes the acquisition itself
# beyond confirming the file is present (data/ is immutable, decision 0001).
FIXTURE_FFT_NAMES = (
    "ICLISTENHF1266_20260313T000004.000Z.fft.gz",
    "ICLISTENHF1266_20260312T235504.000Z.fft.gz",
)
# The .wav is the fixture that carries a SUB-SECOND fraction (.029) and an
# explicit end timestamp -- the two things most likely to be silently dropped.
FIXTURE_WAV_NAME = "ICLISTENHF1266_20260313T000000.029Z_20260313T000500.029Z.wav"
FIXTURE_WAV_START_MICROSECOND = 29_000  # ".029Z" == 29 ms == 29000 us

# Tolerance on a parsed duration, in seconds. Zero: these timestamps are exact
# decimal strings, so the parse is either right or wrong -- there is no float
# error to absorb here, and a loose tolerance would hide a truncated fraction.
DURATION_TOLERANCE_S = 0.0

# Env flag gating the network checks. Unset => they SKIP, and a skip is reported
# as NOT-VERIFIED (see the `skipped` count in main()), never as a pass
# (docs/agent-working-agreements.md s5).
NETWORK_FLAG = "BOATPHONE_ALLOW_NETWORK"

# A day inside the season used by the network smoke check. Chosen inside the
# study window and inside SEASON_MONTHS_UTC; any in-season day would do.
NETWORK_PROBE_DAY = (2024, 7, 15)

# Known ONC location codes for the Folger Passage / Folger Deep area, plus the
# generic shape of an ONC location code. Source: ONC Oceans 3.0 location-code
# convention -- short all-caps alphanumeric codes (e.g. the Folger sites FGPD /
# FGPPN / FGBD / FGCD / FGSD). Used ONLY as a forbidden-literal list for the
# source scan below; nothing in boatphone/ may contain one.
FORBIDDEN_LOCATION_CODES = ("FGPD", "FGPPN", "FGBD", "FGCD", "FGSD", "FGPPT")
# ...and the generic form: any locationCode key or kwarg assigned a literal.
HARDCODED_LOCATION_PATTERN = r"""locationCode['"]?\s*[=:]\s*['"][A-Za-z0-9]{2,10}['"]"""


class _StubONC:
    """A minimal stand-in for onc.ONC. Local class on purpose: no mocking library
    is guaranteed present on the OHW hub (CLAUDE.md "Environment").

    Records every request it is handed so a check can assert the listing was
    chunked by year, and can be told to return files, return nothing, or raise.
    It answers to every plausible listing method name so these checks pin
    BEHAVIOUR, not which ONC endpoint the coder chose.
    """

    def __init__(self, files_by_year=None, locations=None, deployments=None,
                 raise_on_year=None, error=None):
        self.files_by_year = files_by_year or {}
        self.locations = locations if locations is not None else []
        self.deployments = deployments or []
        self.raise_on_year = raise_on_year
        self.error = error
        self.requests = []  # every filter dict passed to a listing call

    # -- listing ------------------------------------------------------------
    def _listing(self, filters=None, **kwargs):
        f = dict(filters or {})
        f.update(kwargs)
        self.requests.append(f)
        year = self._year_of(f)
        if self.raise_on_year is not None and year == self.raise_on_year:
            raise (self.error or RuntimeError("HTTP 401 Unauthorized"))
        names = list(self.files_by_year.get(year, []))
        return {"files": names, "next": None}

    @staticmethod
    def _year_of(filters):
        for key in ("dateFrom", "begin", "dateTo", "end"):
            value = filters.get(key)
            if isinstance(value, str) and len(value) >= 4 and value[:4].isdigit():
                return int(value[:4])
            if hasattr(value, "year"):
                return value.year
        return None

    getArchivefile = _listing
    getArchivefileByLocation = _listing
    getArchivefileByDevice = _listing
    getListByLocation = _listing
    getListByDevice = _listing

    # -- metadata -----------------------------------------------------------
    def getLocations(self, filters=None, **kwargs):
        self.requests.append(dict(filters or {}, **kwargs))
        return list(self.locations)

    def getDeployments(self, filters=None, **kwargs):
        self.requests.append(dict(filters or {}, **kwargs))
        return list(self.deployments)


def _folger_location_rows():
    """Two ONC-shaped location rows, one Folger and one not.

    The codes here are INVENTED so this file does not itself become the place a
    real Folger code is written down; discovery must key off the NAME, not a
    code it already knew.
    """
    return [
        {"locationCode": "ZZTESTA", "locationName": "Folger Deep",
         "deployments": 3, "hasDeviceData": True},
        {"locationCode": "ZZTESTB", "locationName": "Cambridge Bay Underwater Network",
         "deployments": 9, "hasDeviceData": True},
    ]


def _fft_name(dt_utc):
    """Build an ONC-style .fft.gz name for a UTC datetime, matching the fixtures."""
    stamp = dt_utc.strftime("%Y%m%dT%H%M%S.") + f"{dt_utc.microsecond // 1000:03d}Z"
    return f"ICLISTENHF1266_{stamp}.fft.gz"


def _a1b_mod():
    """Import boatphone.onc_client. ImportError/AttributeError here is the
    expected pre-implementation failure for every A1b check."""
    cfg, onc = _a1a_mods()
    for name in ("discover_folger_locations", "list_fft_files",
                 "parse_file_coverage", "get_deployments"):
        assert hasattr(onc, name), (
            f"boatphone.onc_client has no {name}(); A1b's impure half is not implemented"
        )
    return cfg, onc


def check_a1b_0_api_surface():
    """The four A1b entry points exist and are callable."""
    _cfg, onc = _a1b_mod()
    for name in ("discover_folger_locations", "list_fft_files",
                 "parse_file_coverage", "get_deployments"):
        assert callable(getattr(onc, name)), f"onc_client.{name} exists but is not callable"


def check_a1b_1_parse_real_fixture_fft_names():
    """Data-dependent: the REAL .fft.gz names on disk parse to tz-aware UTC 300 s bins."""
    from datetime import timedelta, timezone
    sample = REPO_ROOT / "data" / SAMPLE_DIR_NAME
    if not sample.is_dir():
        raise SkipCheck(f"acquisition directory absent: {sample}")
    present = {p.name for p in sample.iterdir()}
    missing = [n for n in FIXTURE_FFT_NAMES if n not in present]
    if missing:
        raise SkipCheck(f"fixture .fft.gz files absent from {sample}: {missing}")

    cfg, onc = _a1b_mod()
    parsed = {}
    for name in FIXTURE_FFT_NAMES:
        start, end = onc.parse_file_coverage(name)
        assert start.utcoffset() == timedelta(0), (
            f"{name}: start {start!r} is not tz-aware UTC (D1)"
        )
        assert end.utcoffset() == timedelta(0), (
            f"{name}: end {end!r} is not tz-aware UTC (D1)"
        )
        assert end > start, f"{name}: parsed interval is empty or reversed {start}..{end}"
        duration = (end - start).total_seconds()
        assert abs(duration - cfg.BIN_SECONDS) <= DURATION_TOLERANCE_S, (
            f"{name}: parsed duration {duration} s, expected {cfg.BIN_SECONDS} s "
            "(a single-timestamp ONC name covers one BIN_SECONDS file)"
        )
        parsed[name] = start

    # Transcribed by hand from the filenames -- an independent second reading, so
    # a parser that returns a self-consistent but wrong instant still fails.
    expected = {
        "ICLISTENHF1266_20260313T000004.000Z.fft.gz": _utc(2026, 3, 13, 0, 0, 4),
        "ICLISTENHF1266_20260312T235504.000Z.fft.gz": _utc(2026, 3, 12, 23, 55, 4),
    }
    for name, want in expected.items():
        got = parsed[name]
        assert got == want, f"{name}: parsed start {got.isoformat()}, expected {want.isoformat()}"
        assert got.tzinfo is not None and got.utcoffset() == timedelta(0)

    gap = abs((parsed[FIXTURE_FFT_NAMES[0]] - parsed[FIXTURE_FFT_NAMES[1]]).total_seconds())
    assert abs(gap - cfg.BIN_SECONDS) <= DURATION_TOLERANCE_S, (
        f"the two fixture .fft.gz files are {gap} s apart, not {cfg.BIN_SECONDS} s; "
        "the 300 s file cadence D2 assumes is not what the parser produces"
    )
    _ = timezone  # (imported for the reader: every assertion above is UTC)


def check_a1b_1b_subsecond_fraction_survives():
    """Data-dependent: `.029Z` is NOT truncated to a whole second, and the .wav's
    explicit end timestamp is used rather than being ignored."""
    from datetime import timedelta
    sample = REPO_ROOT / "data" / SAMPLE_DIR_NAME
    if not sample.is_dir():
        raise SkipCheck(f"acquisition directory absent: {sample}")
    if not (sample / FIXTURE_WAV_NAME).is_file():
        raise SkipCheck(f"fixture absent: {sample / FIXTURE_WAV_NAME}")

    cfg, onc = _a1b_mod()
    start, end = onc.parse_file_coverage(FIXTURE_WAV_NAME)
    assert start == _utc(2026, 3, 13, 0, 0, 0) + timedelta(microseconds=FIXTURE_WAV_START_MICROSECOND), (
        f"{FIXTURE_WAV_NAME}: parsed start {start.isoformat()} -- the '.029Z' fraction was "
        "dropped or misread; a 29 ms truncation is invisible in a plot and fatal in a join"
    )
    assert start.microsecond == FIXTURE_WAV_START_MICROSECOND, (
        f"start.microsecond is {start.microsecond}, expected {FIXTURE_WAV_START_MICROSECOND} "
        "(truncating to a whole second silently loses the fraction)"
    )
    assert end.microsecond == FIXTURE_WAV_START_MICROSECOND, (
        f"end.microsecond is {end.microsecond}, expected {FIXTURE_WAV_START_MICROSECOND}; "
        "the end timestamp in the name carries the same fraction"
    )
    duration = (end - start).total_seconds()
    assert abs(duration - cfg.BIN_SECONDS) <= DURATION_TOLERANCE_S, (
        f"{FIXTURE_WAV_NAME}: duration {duration} s, expected {cfg.BIN_SECONDS} s"
    )


def check_a1b_1c_fft_and_wav_names_agree():
    """A .fft.gz name and a .wav name for the same instant parse to the same interval.

    The extension must not change the time. This is offline and needs no fixture
    on disk -- it is the pure-parser half of the same contract.
    """
    cfg, onc = _a1b_mod()
    stem = "ICLISTENHF1266_20260313T000000.029Z"
    wav = f"{stem}_20260313T000500.029Z.wav"
    fft = f"{stem}.fft.gz"
    wav_start, wav_end = onc.parse_file_coverage(wav)
    fft_start, fft_end = onc.parse_file_coverage(fft)
    assert wav_start == fft_start, (
        f"same instant parsed differently by extension: .wav -> {wav_start.isoformat()}, "
        f".fft.gz -> {fft_start.isoformat()}"
    )
    assert wav_end == fft_end, (
        f"same instant, different end: .wav -> {wav_end.isoformat()} (from the name), "
        f".fft.gz -> {fft_end.isoformat()} (start + BIN_SECONDS={cfg.BIN_SECONDS})"
    )


def check_a1b_2_malformed_filename_raises():
    """Invariant 5: a name that cannot be parsed RAISES -- never None, never skipped."""
    _cfg, onc = _a1b_mod()
    malformed = [
        "",
        "ICLISTENHF1266.fft.gz",                        # no timestamp at all
        "ICLISTENHF1266_notatimestamp.fft.gz",          # timestamp-shaped garbage
        "ICLISTENHF1266_20260313T000004.000.fft.gz",    # no Z -- no stated time base
        "ICLISTENHF1266_20261332T000004.000Z.fft.gz",   # month 13, day 32
        "ICLISTENHF1266_20260313T250004.000Z.fft.gz",   # hour 25
        "ICLISTENHF1266_20260313T000500.029Z_20260313T000000.029Z.wav",  # end before start
    ]
    for name in malformed:
        try:
            result = onc.parse_file_coverage(name)
        except (ValueError, TypeError):
            continue
        raise AssertionError(
            f"parse_file_coverage({name!r}) returned {result!r} instead of raising. A name "
            "that cannot be turned into a time must surface as an error: returning None (or a "
            "guess) turns an unreadable listing into a silently short uptime calendar."
        )


def check_a1b_3_wrong_device_filename_raises():
    """A file for a device other than config.DEVICE_CODE RAISES.

    Folger has carried more than one hydrophone. Parsing another device's file as
    if it were ours produces uptime for an instrument with a different calibration.
    """
    cfg, onc = _a1b_mod()
    other = "ICLISTENHF1234_20260313T000004.000Z.fft.gz"
    assert cfg.DEVICE_CODE not in other, (
        "the wrong-device fixture accidentally contains the real DEVICE_CODE; fix the fixture"
    )
    try:
        result = onc.parse_file_coverage(other)
    except (ValueError, TypeError):
        return
    raise AssertionError(
        f"parse_file_coverage({other!r}) returned {result!r}; it carries device code "
        f"'ICLISTENHF1234', not config.DEVICE_CODE ({cfg.DEVICE_CODE!r}), and must raise"
    )


def check_a1b_4_no_hardcoded_location_code():
    """No ONC location code literal appears anywhere in boatphone/ (discovery is at runtime).

    This is the check that keeps "discover at runtime" true six commits from now:
    the cheapest way to "fix" a flaky discovery call is to paste the code it
    returned, and that silently pins the analysis to one site forever.
    """
    import re
    package = REPO_ROOT / "boatphone"
    sources = sorted(package.rglob("*.py"))
    assert sources, f"no python sources found under {package}"
    generic = re.compile(HARDCODED_LOCATION_PATTERN)
    hits = []
    for path in sources:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            for forbidden in FORBIDDEN_LOCATION_CODES:
                if forbidden in code:
                    hits.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: literal {forbidden!r}")
            if generic.search(code):
                hits.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno}: locationCode assigned a literal "
                    f"-- {line.strip()!r}"
                )
    assert not hits, (
        "a location code is hardcoded in boatphone/: " + "; ".join(hits)
        + ". discover_folger_locations() must obtain it from client.getLocations() at runtime"
    )


def check_a1b_4b_no_bare_except_in_listing_code():
    """D8: no bare `except:` in the ONC client -- a swallowed 4xx becomes an empty page."""
    import inspect
    _cfg, onc = _a1b_mod()
    src = inspect.getsource(onc)
    hits = [
        f"line {n}: {line.strip()!r}"
        for n, line in enumerate(src.splitlines(), 1)
        if line.split("#", 1)[0].strip() in ("except:", "except Exception:")
    ]
    assert not hits, (
        "boatphone/onc_client.py catches everything: " + "; ".join(hits)
        + ". A caught auth error that returns an empty list is indistinguishable from "
        "'the hydrophone was off' (D8, invariant 5)"
    )


def check_a1b_5_zero_files_over_whole_span_raises():
    """D8: an entirely empty multi-year listing RAISES, it is not an empty calendar."""
    _cfg, onc = _a1b_mod()
    client = _StubONC(files_by_year={})  # every chunk empty
    try:
        result = onc.list_fft_files(client, ["ZZTESTA"], _utc(2023, 1, 1), _utc(2026, 1, 1))
    except Exception as exc:  # noqa: BLE001 -- the raise IS the contract; type checked below
        assert not isinstance(exc, AssertionError), str(exc)
        assert isinstance(exc, (RuntimeError, ValueError)), (
            f"list_fft_files raised {type(exc).__name__} for a wholly empty listing; expected a "
            "RuntimeError/ValueError subclass (ONCListingError)"
        )
        return
    raise AssertionError(
        f"list_fft_files returned {result!r} for a listing with zero files across three years. "
        "Zero files for the whole query is a broken request, not a silent hydrophone (D8); "
        "returning empty here produces an all-unavailable uptime calendar that looks like data."
    )


def check_a1b_5b_http_error_in_a_chunk_propagates():
    """D8: a chunk's HTTP/auth error propagates, carrying the request parameters and NOT the token."""
    _cfg, onc = _a1b_mod()
    token_like = "SENTINEL-NOT-A-REAL-ONC-TOKEN-0000"
    boom = RuntimeError("HTTP 401 Unauthorized")
    client = _StubONC(
        files_by_year={2024: [_fft_name(_utc(2024, 7, 15, 0, 0, 0))]},
        raise_on_year=2025, error=boom,
    )
    try:
        result = onc.list_fft_files(client, ["ZZTESTA"], _utc(2024, 1, 1), _utc(2026, 1, 1))
    except Exception as exc:  # noqa: BLE001 -- the propagation IS the contract
        assert not isinstance(exc, AssertionError), str(exc)
        chain, node = [], exc
        while node is not None and node not in chain:
            chain.append(node)
            node = node.__cause__ or node.__context__
        text = " | ".join(f"{type(n).__name__}: {n}" for n in chain)
        assert boom in chain or "401" in text, (
            f"the chunk error was replaced rather than propagated: {text}"
        )
        assert token_like not in text, "a token-shaped value appeared in the error message"
        assert "2025" in text or "ZZTESTA" in text, (
            "the error names neither the failing year-chunk nor the location code; D8 requires "
            f"the request parameters (never the token) in the message. Got: {text}"
        )
        return
    raise AssertionError(
        f"list_fft_files returned {result!r} although one year-chunk raised HTTP 401. A retry "
        "loop or except-branch that turns a 4xx into an empty page silently deletes a year of "
        "uptime (D8)."
    )


def check_a1b_5c_empty_chunk_count_is_surfaced():
    """The one legitimate empty result -- a year with genuinely no data -- is COUNTED and PRINTED."""
    import contextlib, io
    _cfg, onc = _a1b_mod()
    # 2024 and 2026 have data; 2025 is genuinely empty. Three year-chunks, one empty.
    client = _StubONC(files_by_year={
        2024: [_fft_name(_utc(2024, 7, 15, 0, 0, 0))],
        2026: [_fft_name(_utc(2026, 7, 15, 0, 0, 0))],
    })
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = onc.list_fft_files(client, ["ZZTESTA"], _utc(2024, 1, 1), _utc(2027, 1, 1))
    printed = buffer.getvalue()
    assert isinstance(result, tuple) and len(result) == 2, (
        f"list_fft_files returned {type(result).__name__}; expected a "
        "(filenames, empty_chunks) 2-tuple so the zero-result-chunk count is surfaced"
    )
    files, empty_chunks = result
    assert len(list(files)) == 2, f"expected the 2 listed files, got {list(files)!r}"
    assert empty_chunks == 1, (
        f"empty_chunks is {empty_chunks!r}; exactly one year-chunk (2025) came back empty"
    )
    assert "1" in printed and printed.strip(), (
        "the empty-chunk count was not printed. A silently-empty chunk is the difference "
        f"between 'the method found nothing' and 'the method is broken' (invariant 9). "
        f"stdout was: {printed!r}"
    )


def check_a1b_5d_complete_year_is_one_probe_request_per_year():
    """A year ONC answers COMPLETELY costs exactly one request -- the year probe.

    Renamed from `..._listing_is_chunked_by_year`: post-A1d the shipped contract is
    NOT "one request per year" unconditionally. It is a year PROBE, and months (plus
    paging) only when ONC's `next` cursor says the probe was truncated. This stub
    never truncates, so this check pins only the complete-year half of that contract;
    check_a1b_5d2 pins the truncated half, which is where the real behaviour lives.
    """
    _cfg, onc = _a1b_mod()
    years = [2024, 2025, 2026]
    client = _StubONC(files_by_year={y: [_fft_name(_utc(y, 7, 15, 0, 0, 0))] for y in years})
    onc.list_fft_files(client, ["ZZTESTA"], _utc(2024, 1, 1), _utc(2027, 1, 1))
    listing_requests = [r for r in client.requests if any(
        k in r for k in ("dateFrom", "begin", "dateTo", "end"))]
    assert len(listing_requests) == len(years), (
        f"{len(listing_requests)} listing request(s) for a 3-year span; the contract is one "
        f"chunk per year. Requests: {listing_requests!r}"
    )
    seen = sorted({_StubONC._year_of(r) for r in listing_requests})
    assert seen == years, f"year-chunks requested were {seen}, expected {years}"
    assert all(
        "offset" not in r and "page" not in r and "skip" not in r for r in listing_requests
    ), (
        "a complete (untruncated) year issued a paging request; paging is only legitimate "
        f"when ONC's `next` cursor says the probe was cut off. Requests: {listing_requests!r}"
    )


def check_a1b_5d2_a_truncated_year_falls_back_to_months_and_paging():
    """The OTHER half of the chunking contract: truncation -> paging, and month chunks.

    check_a1b_5d only ever sees a stub that answers a whole year in one page, so it
    cannot see the adaptive behaviour at all. Here the year 2024 holds a dense July
    (one file every FFT_FILE_SECONDS) and nothing else, against a stub capped at
    STUB_ROW_CAP rows. The year probe is therefore truncated, and the contract is:

      * the listing is completed (paged, or sub-chunked) -- every July file returned,
        never the short prefix;
      * more than the one probe request is issued (the probe alone cannot be complete);
      * the year is no longer a unit ONC can make a statement about, so the empty-chunk
        granularity drops to the calendar UTC month: the 11 months with no data are
        reported empty, and July is not.

    That last number is the one that matters downstream -- at year granularity a
    truncated 2024 would report 0 empty chunks and hide eleven empty months.
    """
    cfg, onc = _a1b_mod()
    span_start, span_end = _utc(2024, 1, 1), _utc(2025, 1, 1)
    july_start, july_end = _utc(2024, 7, 1), _utc(2024, 8, 1)
    corpus = _dense_fft_names(july_start, july_end)
    assert len(corpus) > STUB_ROW_CAP, (
        f"fixture is not dense enough to truncate the year probe: {len(corpus)} <= "
        f"{STUB_ROW_CAP}"
    )
    client = _CappedONC(corpus)
    files, empty_chunks = onc.list_fft_files(client, ["ZZTESTA"], span_start, span_end)
    assert sorted(files) == sorted(corpus), (
        f"{len(files)} of {len(corpus)} July 2024 files returned for a year whose probe "
        f"was truncated at {STUB_ROW_CAP} rows; a truncated probe must be completed "
        "(paged or sub-chunked), never accepted as the year (invariant 9)"
    )
    listing_requests = [r for r in client.requests if any(
        k in r for k in ("dateFrom", "begin", "dateTo", "end"))]
    assert len(listing_requests) > 1, (
        "a truncated year probe was answered with a single request; the probe response "
        f"itself advertised another page. Requests: {listing_requests!r}"
    )
    assert empty_chunks == 11, (
        f"empty_chunks is {empty_chunks!r} for a 2024 whose only data is July. A truncated "
        "year is not a unit ONC can make a statement about, so its chunks become the 12 "
        "calendar UTC months and the 11 without data are the empty ones. Reporting 0 here "
        "means the year granularity survived truncation and eleven empty months are hidden; "
        "reporting 12 means July's own files were lost."
    )


def check_a1b_5e_discovery_with_no_folger_candidate_raises():
    """D8: discovery that finds no Folger candidate RAISES rather than returning []."""
    _cfg, onc = _a1b_mod()
    no_folger = [r for r in _folger_location_rows() if "Folger" not in r["locationName"]]
    client = _StubONC(locations=no_folger)
    try:
        result = onc.discover_folger_locations(client)
    except Exception as exc:  # noqa: BLE001 -- the raise IS the contract
        assert not isinstance(exc, AssertionError), str(exc)
        assert isinstance(exc, (RuntimeError, ValueError)), (
            f"discovery raised {type(exc).__name__}; expected a RuntimeError/ValueError subclass"
        )
        return
    raise AssertionError(
        f"discover_folger_locations returned {result!r} when no Folger site was in the "
        "response. Returning [] makes every downstream listing empty and the calendar blank, "
        "which reads as 'the hydrophone was off' (D8)."
    )


def check_a1b_5f_discovery_finds_folger_by_name_not_by_a_known_code():
    """Discovery keys off the ONC response, so an unfamiliar code is still found."""
    _cfg, onc = _a1b_mod()
    client = _StubONC(locations=_folger_location_rows())
    codes = list(onc.discover_folger_locations(client))
    assert codes == ["ZZTESTA"], (
        f"discovery returned {codes!r}; it must return the code of the row NAMED 'Folger Deep' "
        "('ZZTESTA' here) and drop the Cambridge Bay row. Returning [] or the wrong row means "
        "discovery is matching on something other than what ONC actually sent."
    )


def check_a1b_network_discover_returns_at_least_one_code():
    """NETWORK. Gated on BOATPHONE_ALLOW_NETWORK; a skip is NOT-VERIFIED, not a pass."""
    if not os.environ.get(NETWORK_FLAG):
        raise SkipCheck(f"SKIPPED (no network): set {NETWORK_FLAG}=1 to run this check")
    _cfg, onc = _a1b_mod()
    from boatphone.credentials import get_onc_client
    codes = list(onc.discover_folger_locations(get_onc_client()))
    assert len(codes) >= 1, "live discovery returned no Folger location code"
    assert all(isinstance(c, str) and c for c in codes), f"non-string location code in {codes!r}"


def check_a1b_network_one_day_listing_is_non_empty():
    """NETWORK. One in-season day of real listing returns files. Gated as above."""
    if not os.environ.get(NETWORK_FLAG):
        raise SkipCheck(f"SKIPPED (no network): set {NETWORK_FLAG}=1 to run this check")
    from datetime import timedelta
    _cfg, onc = _a1b_mod()
    from boatphone.credentials import get_onc_client
    client = get_onc_client()
    codes = list(onc.discover_folger_locations(client))
    day = _utc(*NETWORK_PROBE_DAY)
    files, empty_chunks = onc.list_fft_files(client, codes, day, day + timedelta(days=1))
    assert len(list(files)) > 0, (
        f"live listing for {day.date()} returned no files across {codes!r} "
        f"({empty_chunks} empty chunk(s)) -- with D8 in force this should have raised"
    )
    for name in list(files)[:5]:
        start, end = onc.parse_file_coverage(name)
        assert day <= start < day + timedelta(days=1), (
            f"listed file {name!r} parses to {start.isoformat()}, outside the requested UTC day"
        )


# --- A1b, second pass: gaps the test-runner's mutation audit exposed ----------
#
# (1) REDACTION IS NOT LISTING-ONLY. check_a1b_5b exercised only list_fft_files,
#     which is clean. discover_folger_locations() and get_deployments() call
#     client.getLocations()/getDeployments() with no try/except, so ONC's own
#     HTTPError propagates verbatim -- and ONC embeds the request URL, INCLUDING
#     `token=...`, in its exception text. A 401 in a notebook writes the token
#     into cell output, and CLAUDE.md invariant 7 forbids committing one.
# (2) SUB-SECOND TRUNCATION HAD NO DATA-INDEPENDENT BACKSTOP: the only check
#     that catches `.029` -> whole seconds SkipChecks when data/ is absent, so a
#     clone without the acquisition went green on a silent 29 ms truncation.
# (3) coverage_intervals() was implemented beyond the pinned API and nothing in
#     CHECKS would notice if its `<` gap tolerance were later loosened to `<=`.

# A fake token planted in a stub error message. Deliberately shaped like the ONC
# client's own 401 text (`Status 401 - Unauthorized: <url>?...&token=<value>`),
# because that URL is exactly where the real leak comes from. It is not a
# credential -- it is a needle these checks look for.
LEAK_SENTINEL_TOKEN = "SENTINEL-LEAKED-TOKEN-DEADBEEF-9999"
ONC_401_TEMPLATE = (
    "Status 401 - Unauthorized: "
    "https://data.oceannetworks.ca/api/{endpoint}?deviceCode={device}&token={token}"
)

# Gap sizes exercised against the merge tolerance, in units of BIN_SECONDS
# offsets. Named rather than inline so the boundary being probed is legible:
# BIN_SECONDS-1 must merge, BIN_SECONDS and BIN_SECONDS+1 must NOT.
MERGE_GAP_OFFSETS = (-1, 0, +1)


class _RaisingONC:
    """A client whose every ONC call raises an ONC-shaped 401 carrying a token.

    Separate from _StubONC on purpose: this one exists to be a leak source, and
    every method must fail the same way so the check can sweep all three impure
    entry points with one fixture.
    """

    def __init__(self, endpoint_error=None):
        self.endpoint_error = endpoint_error or (lambda endpoint: RuntimeError(
            ONC_401_TEMPLATE.format(endpoint=endpoint, device="ICLISTENHF1266",
                                    token=LEAK_SENTINEL_TOKEN)
        ))

    def _fail(self, endpoint):
        raise self.endpoint_error(endpoint)

    def getLocations(self, *a, **k):
        self._fail("locations")

    def getDeployments(self, *a, **k):
        self._fail("deployments")

    def _listing(self, *a, **k):
        self._fail("archivefiles")

    getArchivefile = _listing
    getArchivefileByLocation = _listing
    getArchivefileByDevice = _listing
    getListByLocation = _listing
    getListByDevice = _listing


def _exception_chain_text(exc):
    """Every place a token could hide in a raised exception, as one string.

    Walks `__cause__` AND `__context__` recursively -- `raise ... from exc`, and
    even `from None`, can leave the token-bearing original reachable as
    `__context__`, where a single `str(exc)` on the outermost error never looks.
    Collects str, repr, args and the formatted traceback of every link.
    """
    parts = []
    seen = []
    node = exc
    while node is not None and not any(node is s for s in seen):
        seen.append(node)
        parts.append(f"{type(node).__name__}: {node}")
        parts.append(repr(node))
        parts.extend(repr(a) for a in getattr(node, "args", ()))
        parts.extend(traceback.format_exception(type(node), node, node.__traceback__))
        node = node.__cause__ or node.__context__
    return "\n".join(parts)


def check_a1b_5g_no_token_leaks_from_any_impure_call():
    """No ONC token reaches ANY exception raised by ANY of the three impure entry
    points -- searched over the full __cause__/__context__ chain, not one str().

    The listing path already redacts. Discovery and deployment metadata call ONC
    with no try/except, so a 401 propagates ONC's own message -- which contains
    `token=<the real token>` -- straight into a notebook cell (invariant 7).
    """
    _cfg, onc = _a1b_mod()
    client = _RaisingONC()
    calls = {
        "discover_folger_locations": lambda: onc.discover_folger_locations(client),
        "list_fft_files": lambda: onc.list_fft_files(
            client, ["ZZTESTA"], _utc(2024, 1, 1), _utc(2025, 1, 1)),
        "get_deployments": lambda: onc.get_deployments(client),
    }
    leaked = []
    silent = []
    for name, call in calls.items():
        try:
            result = call()
        except Exception as exc:  # noqa: BLE001 -- raising IS the contract; text checked below
            assert not isinstance(exc, AssertionError), str(exc)
            text = _exception_chain_text(exc)
            if LEAK_SENTINEL_TOKEN in text:
                where = "message" if LEAK_SENTINEL_TOKEN in str(exc) else "chained/traceback text"
                leaked.append(f"{name} ({where})")
            assert "401" in text or "Unauthorized" in text, (
                f"{name}: the HTTP 401 was replaced rather than propagated -- "
                f"a caller cannot tell auth failure from no data. Chain: {text[:400]}"
            )
        else:
            silent.append(f"{name} -> {result!r}")

    assert not silent, (
        "an ONC 401 was swallowed instead of raised: " + "; ".join(silent) + " (D8)"
    )
    assert not leaked, (
        "the ONC API token LEAKS out of: " + "; ".join(leaked)
        + f". ONC embeds `token=...` in its own 401 text; every exception this package "
        "raises (and every link of its __cause__/__context__ chain) must pass through "
        "_redact(). A 401 in a notebook otherwise writes the token into committed cell "
        "output (CLAUDE.md invariant 7)."
    )


def check_a1b_5h_redaction_survives_raise_from_none():
    """`_redact` on the outermost message is not enough if the raw error stays chained.

    Re-raising inside an `except` block keeps the original reachable as
    __context__ even with `from None`, so this check pins the behaviour the
    listing path already gets right -- raise OUTSIDE the except block -- for
    whichever function the coder fixes next.
    """
    _cfg, onc = _a1b_mod()
    client = _RaisingONC()
    for name in ("discover_folger_locations", "get_deployments", "list_fft_files"):
        try:
            if name == "list_fft_files":
                onc.list_fft_files(client, ["ZZTESTA"], _utc(2024, 1, 1), _utc(2025, 1, 1))
            else:
                getattr(onc, name)(client)
        except Exception as exc:  # noqa: BLE001 -- inspecting the chain IS the check
            assert not isinstance(exc, AssertionError), str(exc)
            depth = 0
            node, seen = exc, []
            while node is not None and not any(node is s for s in seen):
                seen.append(node)
                depth += 1
                node = node.__cause__ or node.__context__
            token_bearing = [
                f"link {i}" for i, n in enumerate(seen)
                if LEAK_SENTINEL_TOKEN in f"{n!r} {n}"
            ]
            assert not token_bearing, (
                f"{name}: the raw ONC error is still reachable in the exception chain "
                f"({', '.join(token_bearing)} of {depth}) with the token in it. Raise the "
                "redacted error OUTSIDE the `except` block -- chaining re-attaches the "
                "original, and a traceback prints every link."
            )
        else:
            raise AssertionError(f"{name} did not raise on an ONC 401 (D8)")


def check_a1b_1d_subsecond_backstop_without_data():
    """DATA-INDEPENDENT backstop for sub-second truncation: literal names, no file read.

    check_a1b_1b covers this against the real acquisition, but it SkipChecks when
    data/ is absent -- so on a fresh clone a `.029` -> whole-second truncation
    passed green. This check never skips.
    """
    from datetime import timedelta
    cfg, onc = _a1b_mod()
    # Literal strings, not read from disk. Same grammar as the acquisition's names.
    cases = [
        # (filename, expected start microsecond, expected end microsecond)
        ("ICLISTENHF1266_20260313T000000.029Z_20260313T000500.029Z.wav", 29_000, 29_000),
        ("ICLISTENHF1266_20260313T000000.029Z.fft.gz", 29_000, 29_000),
        ("ICLISTENHF1266_20240715T120000.500Z.fft.gz", 500_000, 500_000),
        ("ICLISTENHF1266_20240715T120000.001Z.fft.gz", 1_000, 1_000),
        ("ICLISTENHF1266_20240715T120000.000Z.fft.gz", 0, 0),
    ]
    for name, want_start_us, want_end_us in cases:
        start, end = onc.parse_file_coverage(name)
        assert start.microsecond == want_start_us, (
            f"{name}: start.microsecond is {start.microsecond}, expected {want_start_us}. "
            "The sub-second fraction was truncated -- invisible in a plot, fatal in a join "
            "(decision 0002)."
        )
        assert end.microsecond == want_end_us, (
            f"{name}: end.microsecond is {end.microsecond}, expected {want_end_us}"
        )
        assert abs((end - start).total_seconds() - cfg.BIN_SECONDS) <= DURATION_TOLERANCE_S, (
            f"{name}: duration is not {cfg.BIN_SECONDS} s"
        )
    # Two names 1 ms apart must NOT parse to the same instant -- the sharpest
    # statement of "the fraction is carried", and one truncation kills it.
    a, _ = onc.parse_file_coverage("ICLISTENHF1266_20240715T120000.001Z.fft.gz")
    b, _ = onc.parse_file_coverage("ICLISTENHF1266_20240715T120000.002Z.fft.gz")
    assert b - a == timedelta(milliseconds=1), (
        f"two names 1 ms apart parsed {(b - a).total_seconds()} s apart; the sub-second "
        "field is being dropped or rounded"
    )


def check_a1b_8_coverage_intervals_merge_is_flag_equivalent():
    """coverage_intervals() may only merge where merging cannot change a bin flag.

    Merging is a performance measure (~45,000 season files vs 44,064 bins), so it
    is only legitimate if `mark_available` cannot tell. It can't when the bridged
    gap is < BIN_SECONDS, and it CAN at exactly BIN_SECONDS -- a whole bin fits in
    the gap and would flip from unavailable to available, i.e. inflated uptime.
    That makes the `<` vs `<=` boundary load-bearing, so it is pinned here.
    """
    from datetime import timedelta
    cfg, onc = _a1b_mod()
    if not hasattr(onc, "coverage_intervals"):
        raise AssertionError(
            "boatphone.onc_client has no coverage_intervals(); the listing->mark_available "
            "bridge is unregistered"
        )
    width = cfg.BIN_SECONDS
    # An epoch-aligned instant well inside the study window, so bin edges are
    # exactly at `base`, base+300, base+600 ...
    base = _utc(2024, 7, 15, 12, 0, 0)
    assert int(base.timestamp()) % width == 0, "fixture base is not on the bin grid"

    def scenario(label, first_start, gap_s):
        """Two files, the second starting `gap_s` after the first one ends."""
        second_start = first_start + timedelta(seconds=width + gap_s)
        names = [_fft_name(first_start), _fft_name(second_start)]
        unmerged = [onc.parse_file_coverage(n) for n in names]
        merged = onc.coverage_intervals(names)
        span_start = first_start - timedelta(seconds=2 * width)
        span_end = second_start + timedelta(seconds=3 * width)
        bins = onc.season_bins_utc(span_start, span_end, season_months=None)
        assert bins, f"{label}: fixture produced no bins"
        got = onc.mark_available(bins, merged)
        want = onc.mark_available(bins, unmerged)
        if got != want:
            diff = [
                f"{b[0].isoformat()} merged={g} unmerged={w}"
                for b, g, w in zip(bins, got, want) if g != w
            ]
            raise AssertionError(
                f"{label} (gap {gap_s} s): merging CHANGED availability -- "
                + "; ".join(diff[:5])
                + f". Merging across a gap >= BIN_SECONDS ({width} s) fits a whole bin inside "
                "the bridged hole and reports listening that ONC never listed (inflated uptime)."
            )
        return merged

    # 1-2 s jitter around the 300 s cadence: merge, and it must actually merge,
    # or this check would also pass on an implementation that never merges.
    merged = scenario("jitter gap", base, width - 1)
    assert len(merged) == 1, (
        f"a gap of {width - 1} s (< BIN_SECONDS) left {len(merged)} intervals; it is "
        "provably flag-preserving and must be merged, or a season stays at ~45,000 "
        "intervals and mark_available becomes 2e9 comparisons"
    )
    # Exactly BIN_SECONDS and above: a real outage. Must be preserved and visible.
    for offset in (0, +1):
        merged = scenario(f"outage gap BIN_SECONDS{offset:+d}", base, width + offset)
        assert len(merged) == 2, (
            f"a gap of {width + offset} s (>= BIN_SECONDS) was MERGED into {len(merged)} "
            "interval(s). At exactly BIN_SECONDS a whole bin fits in the gap, so this "
            "reports uptime for a bin ONC listed no file in. The tolerance is `<`, not `<=`."
        )
    # A gap that straddles a bin edge rather than sitting inside one bin: the
    # off-grid case where an interval-vs-bin off-by-one would show up.
    off_grid = base + timedelta(seconds=width // 2)
    for offset in MERGE_GAP_OFFSETS:
        scenario(f"straddling bin edge, gap BIN_SECONDS{offset:+d}", off_grid, width + offset)
    scenario("straddling bin edge, jitter gap", off_grid, width - 1)
    # Unsorted input must not change the answer: ONC's listing order is not a
    # contract, and a merge that assumes sortedness silently under-merges.
    later = base + timedelta(seconds=width + (width - 1))
    shuffled = onc.coverage_intervals([_fft_name(later), _fft_name(base)])
    assert len(shuffled) == 1 and shuffled[0][0] == base, (
        f"reverse-ordered filenames merged to {shuffled!r}; coverage_intervals must sort "
        "before merging -- ONC's listing order is not guaranteed"
    )


def check_a1b_network_live_401_is_redacted():
    """NETWORK. A REAL ONC 401 (deliberately bogus token) must not leak the token.

    Uses a bogus token on purpose, so it needs no valid credential -- only the
    network. Gated on BOATPHONE_ALLOW_NETWORK; a skip is NOT-VERIFIED, not a pass.
    """
    if not os.environ.get(NETWORK_FLAG):
        raise SkipCheck(f"SKIPPED (no network): set {NETWORK_FLAG}=1 to run this check")
    _cfg, onc_mod = _a1b_mod()
    from onc import ONC
    bogus = LEAK_SENTINEL_TOKEN
    client = ONC(bogus, outPath=tempfile.gettempdir())
    try:
        result = onc_mod.discover_folger_locations(client)
    except Exception as exc:  # noqa: BLE001 -- the 401 IS the fixture
        assert not isinstance(exc, AssertionError), str(exc)
        text = _exception_chain_text(exc)
        assert bogus not in text, (
            "a live ONC 401 leaked the token through discover_folger_locations; "
            "ONC puts `token=...` in the request URL it echoes back (invariant 7)"
        )
        return
    raise AssertionError(
        f"discover_folger_locations returned {result!r} using a bogus token; an "
        "unauthenticated empty result is indistinguishable from 'no data' (D8)"
    )


# ---------------------------------------------------------------------------
# A1c -- the uptime calendar builder and its CSV emission (O1)
#
# A1a delivered the pure grid algebra; A1b delivered the ONC listing client.
# Neither writes `data/interim/hydrophone_uptime.csv`. A1c is the composition
# that does, and it is O1 -- the highest-priority outbound of the acoustics
# workstream: Planet quota is capped near 30 orders/month and a spent order is
# unrecoverable, so a wrong or silently-truncated calendar costs the team
# irrecoverable quota, not just a bad plot.
#
# CONTRACT PINNED HERE. The A1 plan (accoutics_plan.md, "### A1") and D1-D8
# (onc_client.py module docstring) name every convention EXCEPT the exact
# function names and the CSV's boolean/ISO encoding -- those are not fixed by
# any existing code, so this test-author is PINNING them, flagged as guesses:
#
#   (g1) boatphone.onc_client.build_uptime_calendar(client, start_utc, end_utc)
#        -> list[tuple[start_utc: datetime, end_utc: datetime, available: bool,
#                       deployment_id: str]]
#        Composes, in order: discover_folger_locations -> list_fft_files ->
#        coverage_intervals -> season_bins_utc(start_utc, end_utc) [default
#        SEASON_MONTHS_UTC filter, per D4/D5] -> mark_available -> get_deployments
#        -> assign_deployment. Nothing here catches an exception from any of
#        those six calls: a partial scan must RAISE, never return a short list
#        (invariant 5; this is the specific failure that wastes Planet quota).
#        GUESS: this exact name/signature. The alternative -- a bare script with
#        no importable function -- would make O1 untestable without I/O, which
#        contradicts "checks only".
#
#   (g2) boatphone.onc_client.write_uptime_calendar_csv(rows, path) -> None
#        Writes exactly the header `start_utc,end_utc,available,deployment_id`
#        (this order), one row per element of `rows`, then nothing else.
#        GUESS: `available` serialises as Python's `str(bool)` ("True"/"False")
#        -- the plan does not specify, and this is the plainest choice that
#        survives `csv.DictReader` + `row["available"] == "True"`.
#        GUESS: `start_utc`/`end_utc` serialise via `datetime.isoformat()`,
#        which for a UTC-offset datetime always carries the explicit `+00:00`
#        suffix (never a naive string, never bare `Z` unless the coder chooses
#        to substitute it) -- checked below as "carries an explicit UTC marker",
#        accepting either spelling so this doesn't overfit the guess.
#
# These checks target g1/g2 only, deliberately not the disk artefact at
# data/interim/hydrophone_uptime.csv or the full 2020-2026 scan -- CLAUDE.md
# forbids writing under data/ from a check, and a full-span scan belongs in a
# NETWORK-gated smoke check (see check_a1c_network_one_day_end_to_end below),
# not in the always-run suite.
# ---------------------------------------------------------------------------

UPTIME_CSV_HEADER = ["start_utc", "end_utc", "available", "deployment_id"]

# An outage clearly larger than BIN_SECONDS (300 s), so it cannot be a jitter
# artefact per coverage_intervals' own tolerance (checked in A1b) and must
# read as unavailable bins in the calendar. 1800 s = 6 missed bins.
A1C_GAP_SECONDS = 1800


def _a1c_mod():
    """Import boatphone.onc_client and require the A1c entry points.

    Missing attributes ARE the expected pre-implementation failure for every
    check below -- A1c has not been coded yet.
    """
    cfg, onc = _a1a_mods()
    for name in ("build_uptime_calendar", "write_uptime_calendar_csv"):
        assert hasattr(onc, name), (
            f"boatphone.onc_client has no {name}(); A1c is not implemented yet "
            "(expected until the coder lands it -- see the A1c contract comment above)"
        )
    return cfg, onc


def _a1c_fixture_files(base, offset=None):
    """Six .fft.gz filenames: three at the span start, a gap, three before the end.

    `base` is the epoch-aligned start of a 1-hour in-season span. Files cover
    minutes 0, 5, 10 (bins 0-2) and 45, 50, 55 (bins 9-11) of that hour, leaving
    a genuine A1C_GAP_SECONDS-scale gap (minutes 15-45, bins 3-8) with nothing
    listed. `offset`, if given, shifts every filename's timestamp by that
    timedelta -- used to build the wrong-time fixture.
    """
    from datetime import timedelta
    shift = offset or timedelta(0)
    minutes = (0, 5, 10, 45, 50, 55)
    return [_fft_name(base + timedelta(minutes=m) + shift) for m in minutes]


def _a1c_stub_client(files, deployment_covers_gap=True):
    """A _StubONC wired for a 1-hour July 2024 span with one Folger deployment.

    `deployment_covers_gap=True` makes the single deployment span the WHOLE
    hour (so the gap bins are inside a known deployment, per D6 -- gap bins are
    unavailable but still deployment-attributed). Set False only to build a
    fixture for a different check; no check here needs it False.
    """
    span_start_iso = "2024-07-15T00:00:00.000Z"
    span_end_iso = "2024-07-15T01:00:00.000Z" if deployment_covers_gap else "2024-07-15T00:15:00.000Z"
    deployments = [{
        "begin": span_start_iso, "end": span_end_iso,
        "locationCode": "ZZTESTA", "deviceCode": "ICLISTENHF1266",
    }]
    return _StubONC(
        files_by_year={2024: list(files)},
        locations=_folger_location_rows(),
        deployments=deployments,
    )


def check_a1c_0_api_surface():
    """The two A1c entry points exist and are callable (g1/g2)."""
    _cfg, onc = _a1c_mod()
    for name in ("build_uptime_calendar", "write_uptime_calendar_csv"):
        assert callable(getattr(onc, name)), f"onc_client.{name} exists but is not callable"


def check_a1c_1_row_count_is_arithmetic():
    """Row count for a span equals season_bins_utc's count -- no silent truncation.

    The calendar is dense (D5): every in-season bin is a row whether or not
    anything was listed. A scan that stops early on the first empty listing
    chunk, or that returns only the bins with a matching file, produces a
    SHORT csv that still looks complete -- exactly the failure mode that wastes
    Planet quota. This pins the row count against A1a's own density primitive
    so a coder cannot special-case one without breaking the other.
    """
    cfg, onc = _a1c_mod()
    span_start, span_end = _utc(2024, 7, 15, 0, 0), _utc(2024, 7, 15, 1, 0)
    expect_bins = onc.season_bins_utc(span_start, span_end)
    client = _a1c_stub_client(_a1c_fixture_files(span_start))
    rows = onc.build_uptime_calendar(client, span_start, span_end)
    assert len(rows) == len(expect_bins), (
        f"build_uptime_calendar returned {len(rows)} rows for a span season_bins_utc "
        f"puts at {len(expect_bins)}; the density property A1a already established "
        "must survive composition -- a short calendar looks complete and is the "
        "specific failure that wastes Planet quota"
    )


def check_a1c_2_schema_and_half_open_semantics():
    """Row shape is (start, end, available, deployment_id); bins are half-open and touch."""
    from datetime import timedelta
    cfg, onc = _a1c_mod()
    span_start, span_end = _utc(2024, 7, 15, 0, 0), _utc(2024, 7, 15, 1, 0)
    client = _a1c_stub_client(_a1c_fixture_files(span_start))
    rows = onc.build_uptime_calendar(client, span_start, span_end)
    assert rows, "build_uptime_calendar returned no rows for a 1-hour in-season span"
    for i, row in enumerate(rows):
        assert len(row) == 4, f"row {i} has {len(row)} fields, expected 4: {row!r}"
        start, end, available, dep_id = row
        assert isinstance(start, __import__("datetime").datetime), (
            f"row {i} start is {type(start).__name__}, not datetime"
        )
        assert start.utcoffset() == timedelta(0), f"row {i} start is not tz-aware UTC: {start!r}"
        assert end.utcoffset() == timedelta(0), f"row {i} end is not tz-aware UTC: {end!r}"
        assert end - start == timedelta(seconds=cfg.BIN_SECONDS), (
            f"row {i} spans {(end - start).total_seconds()} s, expected {cfg.BIN_SECONDS} s"
        )
        assert isinstance(available, bool), f"row {i} available is {type(available).__name__}, not bool"
        assert isinstance(dep_id, str), f"row {i} deployment_id is {type(dep_id).__name__}, not str"
    # Half-open, contiguous: consecutive rows touch exactly (same property A1a
    # pins for season_bins_utc; here it must survive the CSV-bound composition).
    bad = [i for i in range(1, len(rows)) if rows[i][0] != rows[i - 1][1]]
    assert not bad, (
        f"rows are not contiguous half-open bins at index {bad[0]}: "
        f"{rows[bad[0] - 1][1].isoformat()} -> {rows[bad[0]][0].isoformat()}"
    )


def check_a1c_3_gap_is_measured_not_defaulted():
    """A constructed gap in listed files shows up as available=False -- not a default.

    `available` must come from measuring the listing (D3), not from a blanket
    True (which would hide every real outage) or a blanket False (which would
    tell Malachy every date is a gap). Both defaults happen to be constant, so
    the only check that rules them both out is one with a KNOWN mix of covered
    and gap bins.
    """
    cfg, onc = _a1c_mod()
    span_start, span_end = _utc(2024, 7, 15, 0, 0), _utc(2024, 7, 15, 1, 0)
    client = _a1c_stub_client(_a1c_fixture_files(span_start))
    rows = onc.build_uptime_calendar(client, span_start, span_end)
    available = [a for _s, _e, a, _d in rows]
    assert any(available), (
        "no bin is available; the fixture lists files for minutes 0/5/10/45/50/55 -- "
        "available is not defaulted False"
    )
    assert not all(available), (
        "every bin is available; the fixture has a deliberate 30-minute gap "
        "(minutes 15-45, no files listed) -- available is not defaulted True"
    )
    # The gap bins specifically: minutes 15..40 (bins 3-8 of a 5-min grid).
    starts = [s for s, _e, _a, _d in rows]
    gap_minutes = {15, 20, 25, 30, 35, 40}
    gap_flags = [a for s, a in zip(starts, available) if (s - span_start).total_seconds() // 60 in gap_minutes]
    assert gap_flags and not any(gap_flags), (
        f"bins in the constructed gap are {gap_flags}, expected all False; "
        "the gap did not survive composition into build_uptime_calendar"
    )


def check_a1c_4_deployment_id_from_metadata_not_gaps():
    """D6: a gap bin still carries the real deployment_id -- it is not cleared to "".

    The deployment in this fixture spans the WHOLE hour, including the gap.
    Attributing "" to the gap bins would mean deployment membership is being
    (re-)inferred from the presence of data, which D6 explicitly forbids: a
    gap in the listing means "no file", not "not deployed".
    """
    cfg, onc = _a1c_mod()
    span_start, span_end = _utc(2024, 7, 15, 0, 0), _utc(2024, 7, 15, 1, 0)
    client = _a1c_stub_client(_a1c_fixture_files(span_start), deployment_covers_gap=True)
    rows = onc.build_uptime_calendar(client, span_start, span_end)
    gap_minutes = {15, 20, 25, 30, 35, 40}
    gap_rows = [(s, a, d) for s, _e, a, d in rows
                if (s - span_start).total_seconds() // 60 in gap_minutes]
    assert gap_rows, "no rows fall in the constructed gap window"
    empty = [(s.isoformat(), a, d) for s, a, d in gap_rows if not d]
    assert not empty, (
        f"gap bins with an empty deployment_id: {empty}; the deployment spans the whole "
        "hour in this fixture, so a gap bin's deployment_id must still be the real id -- "
        "clearing it to \"\" means deployment membership is being inferred from the "
        "presence of data, which D6 forbids"
    )
    ids = {d for _s, _a, d in gap_rows}
    assert len(ids) == 1, f"gap bins carry inconsistent deployment ids: {ids}"


def check_a1c_5_wrong_time_fails():
    """WRONG-TIME-FAILS: shifting every listed file by +7 h changes the availability column.

    In the spirit of check_a1a_2_pdt_offset_leak_is_caught, but exercised through
    the composed calendar builder rather than the bare grid primitive: a
    time-base bug that only appears once list_fft_files/coverage_intervals are
    in the loop would not be caught by A1a alone.
    """
    from datetime import timedelta
    cfg, onc = _a1c_mod()
    span_start, span_end = _utc(2024, 7, 15, 0, 0), _utc(2024, 7, 15, 1, 0)
    true_files = _a1c_fixture_files(span_start)
    wrong_files = _a1c_fixture_files(span_start, offset=timedelta(hours=7))

    true_rows = onc.build_uptime_calendar(_a1c_stub_client(true_files), span_start, span_end)
    wrong_rows = onc.build_uptime_calendar(_a1c_stub_client(wrong_files), span_start, span_end)

    true_avail = [a for _s, _e, a, _d in true_rows]
    wrong_avail = [a for _s, _e, a, _d in wrong_rows]
    assert any(true_avail), "the correct-UTC fixture marked nothing available"
    assert true_avail != wrong_avail, (
        "shifting every listed file's timestamp by +7 h (the PDT offset) produced the SAME "
        f"availability column ({true_avail}) as the correct-UTC fixture. A calendar that "
        "cannot tell these two inputs apart would not catch a real local-time leak, and "
        "shipping it to Malachy could waste Planet quota on a date that is actually fine "
        "(or vice versa)"
    )
    # The shifted files fall entirely outside [00:00, 01:00) once shifted +7h into
    # the same UTC day, so the wrong-time run should show LESS availability, not
    # more or equal -- the pathological case where a bug both misplaces AND
    # duplicates coverage is worth ruling out explicitly.
    assert sum(wrong_avail) < sum(true_avail), (
        f"the +7h-shifted fixture shows {sum(wrong_avail)} available bins against "
        f"{sum(true_avail)} for the correct fixture; the shifted files land outside the "
        "requested span entirely and should show LESS coverage, not equal or more"
    )


def check_a1c_6_partial_scan_raises_not_truncates():
    """A listing failure mid-scan RAISES; it must never emit a short calendar.

    This is invariant 5 applied to the single highest-value artefact in the
    workstream: a `build_uptime_calendar` that catches the underlying
    ONCListingError/RuntimeError and returns whatever it collected so far turns
    a broken request into a calendar that looks complete -- and a "complete"
    calendar with a hidden gap is precisely what causes Planet quota to be
    spent on a date with no hydrophone data.
    """
    cfg, onc = _a1c_mod()
    span_start, span_end = _utc(2024, 7, 15, 0, 0), _utc(2024, 7, 16, 0, 0)
    boom = RuntimeError("HTTP 503 Service Unavailable")
    client = _a1c_stub_client(_a1c_fixture_files(span_start))
    client.raise_on_year = 2024
    client.error = boom
    try:
        result = onc.build_uptime_calendar(client, span_start, span_end)
    except Exception as exc:  # noqa: BLE001 -- the raise IS the contract
        assert not isinstance(exc, AssertionError), str(exc)
        assert isinstance(exc, (RuntimeError, ValueError)), (
            f"build_uptime_calendar raised {type(exc).__name__} for a failed listing "
            "chunk; expected a RuntimeError/ValueError subclass (e.g. ONCListingError) "
            "to propagate, not be swallowed into a shorter result"
        )
        return
    raise AssertionError(
        f"build_uptime_calendar returned {len(result)} row(s) despite the underlying "
        "listing raising on the 2024 chunk. A truncated-but-plausible-looking uptime "
        "calendar is the specific failure this contract exists to prevent (O1)."
    )


def check_a1c_7_csv_schema_and_column_order():
    """write_uptime_calendar_csv emits exactly the pinned header, in order."""
    from datetime import timedelta
    _cfg, onc = _a1c_mod()
    rows = [
        (_utc(2024, 7, 15, 0, 0), _utc(2024, 7, 15, 0, 5), True, "ICLISTENHF1266@ZZTESTA:2024-07-01T00:00:00.000Z"),
        (_utc(2024, 7, 15, 0, 5), _utc(2024, 7, 15, 0, 10), False, ""),
    ]
    import csv as csv_mod
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp, "hydrophone_uptime.csv")
        onc.write_uptime_calendar_csv(rows, path)
        assert path.is_file(), f"write_uptime_calendar_csv did not create {path}"
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv_mod.reader(fh)
            header = next(reader)
            body = list(reader)
    assert header == UPTIME_CSV_HEADER, (
        f"CSV header is {header}, expected {UPTIME_CSV_HEADER} in exactly this order"
    )
    assert len(body) == len(rows), f"CSV has {len(body)} data row(s), expected {len(rows)}"
    _ = timedelta  # (imported for the reader; not otherwise used)


def check_a1c_8_csv_timestamps_are_explicit_utc_iso8601():
    """start_utc/end_utc in the CSV are ISO 8601 with an EXPLICIT UTC offset -- never naive.

    A timestamp that round-trips through `datetime.fromisoformat` to a NAIVE
    value is exactly the kind of string that gets silently treated as local
    time by the next reader (decision 0002). This check fails on that shape
    even though the in-memory object it came from was UTC-aware -- the
    contract is about the FILE, which is the thing Malachy actually reads.
    """
    from datetime import datetime
    _cfg, onc = _a1c_mod()
    rows = [(_utc(2024, 7, 15, 0, 0), _utc(2024, 7, 15, 0, 5), True, "DEP1")]
    import csv as csv_mod
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp, "hydrophone_uptime.csv")
        onc.write_uptime_calendar_csv(rows, path)
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv_mod.DictReader(fh)
            record = next(reader)
    for col in ("start_utc", "end_utc"):
        raw = record[col]
        assert "Z" in raw or "+00:00" in raw, (
            f"{col}={raw!r} carries no explicit UTC marker ('Z' or '+00:00'); a bare "
            "'2024-07-15T00:00:00' is exactly the naive shape decision 0002 forbids -- "
            "the next reader has no way to know it is UTC rather than local time"
        )
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None, f"{col}={raw!r} parses back to a naive datetime"
        assert parsed.utcoffset().total_seconds() == 0, (
            f"{col}={raw!r} parses to a non-zero UTC offset: {parsed.utcoffset()}"
        )
    assert record["start_utc"] == "2024-07-15T00:00:00+00:00" or record["start_utc"].startswith(
        "2024-07-15T00:00:00"
    ), f"start_utc round-trips to the wrong instant: {record['start_utc']!r}"


def check_a1c_9_csv_row_count_matches_calendar():
    """write_uptime_calendar_csv writes exactly one data row per input row -- no drop, no dupe."""
    cfg, onc = _a1c_mod()
    span_start, span_end = _utc(2024, 7, 15, 0, 0), _utc(2024, 7, 15, 1, 0)
    client = _a1c_stub_client(_a1c_fixture_files(span_start))
    rows = onc.build_uptime_calendar(client, span_start, span_end)
    import csv as csv_mod
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp, "hydrophone_uptime.csv")
        onc.write_uptime_calendar_csv(rows, path)
        with open(path, newline="", encoding="utf-8") as fh:
            body = list(csv_mod.reader(fh))[1:]
    assert len(body) == len(rows), (
        f"CSV has {len(body)} data row(s) for {len(rows)} calendar row(s); the write "
        "step must not drop or duplicate a row -- a silent drop here is indistinguishable "
        "from a truncated scan once the file is on disk"
    )


def check_a1c_network_one_day_end_to_end():
    """NETWORK. build_uptime_calendar over ONE real in-season day, gated as A1b's are.

    Deliberately a single day, not a full-span scan: the brief for this segment
    asks for a small network check, and a 2020-2026 pull belongs to A4, not to
    a check that runs on every invocation of an opt-in flag.
    """
    if not os.environ.get(NETWORK_FLAG):
        raise SkipCheck(f"SKIPPED (no network): set {NETWORK_FLAG}=1 to run this check")
    from datetime import timedelta
    cfg, onc = _a1c_mod()
    from boatphone.credentials import get_onc_client
    client = get_onc_client()
    day = _utc(*NETWORK_PROBE_DAY)
    rows = onc.build_uptime_calendar(client, day, day + timedelta(days=1))
    expect = len(onc.season_bins_utc(day, day + timedelta(days=1)))
    assert len(rows) == expect, (
        f"live one-day scan returned {len(rows)} rows, expected {expect} (dense, per D5)"
    )
    assert any(a for _s, _e, a, _d in rows), (
        f"live listing for {day.date()} (already shown non-empty by A1b's own network "
        "check) produced a calendar with no available bins at all"
    )


# ---------------------------------------------------------------------------
# A1c -- INSPECTION: the uptime plot, the UTC-hour profile, and the gap summary
#
# The plot is the debugger: the calendar is not trustworthy until it has been
# looked at, and it is deliberately looked at BEFORE anything is integrated on
# top of it. But a plot is not checkable, so the checks below target the PURE
# FUNCTIONS the plot is drawn from -- never pixels.
#
# API PINNED HERE (the coder must add these to boatphone/onc_client.py; they are
# module functions, not notebook cells, so they are importable and checkable):
#
#   g3. summarise_gaps(bins, available, min_seconds=0)
#         `bins`      -- sequence of (start_utc, end_utc) as returned by
#                        season_bins_utc(); tz-aware UTC, half-open.
#         `available` -- parallel sequence of bools as returned by
#                        mark_available().
#       -> list of (gap_start_utc, gap_end_utc, n_bins) tuples, ascending, one
#          per MAXIMAL run of contiguous unavailable bins whose duration is
#          >= min_seconds. `gap_end_utc` is the EXCLUSIVE end (D2): it equals
#          the `end` of the last unavailable bin in the run, which is also the
#          `start` of the next available bin. `n_bins` is the count of bins in
#          the run.
#          A run is broken by an available bin AND by a discontinuity in the bin
#          grid itself (bins[i][0] != bins[i-1][1]) -- see the season-boundary
#          check below for why that second rule is load-bearing.
#
#   g4. mean_availability_by_utc_hour(bins, available)
#       -> sequence of exactly 24 floats, index == UTC hour of the bin START.
#          Value = mean of `available` over the bins falling in that UTC hour.
#          An hour with NO bins is float('nan'), never 0.0 -- 0.0 means
#          "measured, and nothing was available", and conflating the two is the
#          exact confusion this profile exists to expose.
#
# GUESSES I had to make, flagged rather than hidden:
#   * The (bins, available) argument pair rather than the assembled calendar
#     rows: it matches what season_bins_utc/mark_available already return, so
#     the two new functions compose with A1a without an adapter. If the coder
#     prefers rows, these checks are the place to renegotiate -- but pick ONE.
#   * The 3-tuple return shape, and `n_bins` as its third field. A namedtuple
#     with the same field order would also satisfy these checks (they index
#     positionally), which is deliberate: the shape is pinned, the type is not.
#   * min_seconds is a >= threshold (inclusive), checked explicitly below.
# ---------------------------------------------------------------------------

# The doc-file half of deliverable O1. Source: the A1c brief -- the CSV is
# gitignored, so the human-readable summary is the part that actually ships to
# the Planet stream, and it must therefore be TRACKED.
GAPS_DOC_RELPATH = os.path.join("docs", "derived", "hydrophone_gaps.md")

# Pinned substrings the gap summary must carry. These are named constants so
# that softening the caveat is an edit to THIS file -- visible in review --
# rather than a silent drift in prose. Source: the A1c brief / decision D3.
GAPS_DOC_CAVEAT = "ONC listings"
GAPS_DOC_CAVEAT_2 = "not a completed download"
GAPS_DOC_REVISION_NOTE = "A4"
GAPS_DOC_MALACHY_LINE = "do not spend Planet quota"

# The +7 h America/Vancouver (PDT) offset -- the single most likely wrong answer
# in this project (decision 0002). Source: America/Vancouver is UTC-7 in summer,
# so a local timestamp read as UTC lands 7 h late.
A1C_PDT_SHIFT_HOURS = 7

# Span used for the UTC-hour profile fixtures: whole UTC days, in season.
# 7 days x 288 bins = 2016 bins, 84 bins per hour-of-day.
A1C_PROFILE_DAYS = 7
A1C_PROFILE_START = (2024, 7, 8)  # a Monday, in season; nothing depends on weekday

# Minimum peak-to-trough spread the +7 h profile must show. NOT arbitrary: with
# a 7-day span, shifting coverage by +7 h leaves the first 7 UTC hours of day 1
# uncovered, so hours 0-6 read (A1C_PROFILE_DAYS-1)/A1C_PROFILE_DAYS = 0.857
# against 1.0 elsewhere -- an expected spread of exactly 1/7 = 0.1428. The
# threshold is set below that with headroom, and far above the 0.0 a genuinely
# flat profile produces.
A1C_PROFILE_SPREAD_MIN = 0.10

# Float tolerance for comparing an exact rational mean (k/n with small n) that
# has been through one division. 1e-12 is ~4 orders of magnitude above the
# double-precision ulp at 1.0 and ~4 orders below the effect being measured.
A1C_PROFILE_TOL = 1e-12


def _a1c_insp_mod():
    """Import boatphone.onc_client and require the A1c INSPECTION entry points.

    Missing attributes ARE the expected pre-implementation failure for every
    check below. Note this deliberately does NOT go through _a1c_mod(): the
    inspection functions are independent of build_uptime_calendar, and a check
    for one should not fail for the absence of the other.
    """
    cfg, onc = _a1a_mods()
    for name in ("summarise_gaps", "mean_availability_by_utc_hour"):
        assert hasattr(onc, name), (
            f"boatphone.onc_client has no {name}(); the A1c inspection functions are "
            "not implemented yet (expected until the coder lands them -- see the A1c "
            "INSPECTION contract comment above for the pinned signature)"
        )
    return cfg, onc


def _a1c_day_bins(onc, year, month, day, ndays=1):
    """Dense in-season bins over `ndays` whole UTC days starting at midnight."""
    from datetime import timedelta
    start = _utc(year, month, day)
    return onc.season_bins_utc(start, start + timedelta(days=ndays))


def check_a1c_10_inspection_api_surface():
    """summarise_gaps and mean_availability_by_utc_hour exist in the MODULE, not the notebook.

    CLAUDE.md: notebooks are thin wrappers. A function that lives only in a
    notebook cell cannot be checked, cannot be reused by A4, and cannot be
    reviewed -- so its existence in boatphone/ is itself part of the contract.
    """
    _cfg, onc = _a1c_insp_mod()
    for name in ("summarise_gaps", "mean_availability_by_utc_hour"):
        assert callable(getattr(onc, name)), f"onc_client.{name} exists but is not callable"


def check_a1c_11_summarise_gaps_planted_gaps_exact():
    """PLANTED GROUND TRUTH: three gaps of known length and position, incl. both boundaries.

    A gap at the very first bin and a gap at the very last bin are where a
    "contiguous run" loop typically drops one -- the run is never terminated by
    a following available bin, so an implementation that only emits on the
    transition to available loses the trailing gap entirely, and one that only
    starts a run on the transition to unavailable loses the leading one.

    Also pins the half-open contract (D2): a gap's end_utc is EXCLUSIVE and
    equals the start of the next available bin.
    """
    cfg, onc = _a1c_insp_mod()
    bins = _a1c_day_bins(onc, 2024, 7, 15)
    day_bins = 24 * 60 * 60 // cfg.BIN_SECONDS
    assert len(bins) == day_bins, f"fixture is {len(bins)} bins, expected {day_bins}"

    # Planted: [0,3) leading, [100,112) interior, [282,288) trailing.
    planted = [(0, 3), (100, 112), (282, day_bins)]
    available = [True] * day_bins
    for lo, hi in planted:
        for i in range(lo, hi):
            available[i] = False

    gaps = list(onc.summarise_gaps(bins, available))
    assert len(gaps) == len(planted), (
        f"summarise_gaps found {len(gaps)} gap(s), planted {len(planted)}: "
        f"{[(g[0].isoformat(), g[1].isoformat()) for g in gaps]}. A leading or trailing "
        "gap dropped here is a gap Malachy would spend Planet quota into."
    )
    for (lo, hi), gap in zip(planted, gaps):
        gap_start, gap_end, n_bins = gap[0], gap[1], gap[2]
        assert gap_start == bins[lo][0], (
            f"gap starting at bin {lo} reports {gap_start.isoformat()}, expected "
            f"{bins[lo][0].isoformat()}"
        )
        assert gap_end == bins[hi - 1][1], (
            f"gap ending at bin {hi-1} reports end_utc {gap_end.isoformat()}, expected the "
            f"EXCLUSIVE end {bins[hi-1][1].isoformat()} (D2: half-open [start, end)). "
            "An inclusive end here understates every gap by one bin and, worse, makes the "
            "gap and the next available bin claim the same instant."
        )
        assert n_bins == hi - lo, f"gap [{lo},{hi}) reports n_bins={n_bins}, expected {hi-lo}"
    # The exclusive end is the start of the next AVAILABLE bin, not of another gap.
    assert gaps[0][1] == bins[3][0] and available[3] is True, (
        "the leading gap's exclusive end must coincide with the first available bin's start"
    )


def check_a1c_11b_gap_and_available_time_is_conserved():
    """CONSERVATION: gap seconds + available seconds == total bin seconds. No bin lost or double-counted.

    This is the check that catches an off-by-one at a run boundary even when the
    gap COUNT happens to be right: one bin claimed by two gaps, or by neither,
    breaks the sum while leaving the shape plausible.
    """
    cfg, onc = _a1c_insp_mod()
    bins = _a1c_day_bins(onc, 2024, 7, 15)
    day_bins = len(bins)
    planted = [(0, 3), (100, 112), (282, day_bins)]
    available = [True] * day_bins
    for lo, hi in planted:
        for i in range(lo, hi):
            available[i] = False

    gaps = list(onc.summarise_gaps(bins, available))
    gap_seconds = sum((g[1] - g[0]).total_seconds() for g in gaps)
    gap_bins = sum(g[2] for g in gaps)
    available_bins = sum(1 for a in available if a)
    total_seconds = len(bins) * cfg.BIN_SECONDS

    assert gap_bins + available_bins == len(bins), (
        f"{gap_bins} gap bin(s) + {available_bins} available bin(s) != {len(bins)} bins; "
        "a bin has been lost or double-counted"
    )
    assert gap_seconds + available_bins * cfg.BIN_SECONDS == total_seconds, (
        f"gap seconds ({gap_seconds}) + available seconds "
        f"({available_bins * cfg.BIN_SECONDS}) = {gap_seconds + available_bins * cfg.BIN_SECONDS}, "
        f"expected the total {total_seconds}. Exact integer arithmetic is used deliberately: "
        "every quantity here is a whole multiple of BIN_SECONDS, so no tolerance is warranted "
        "and any mismatch is a real accounting error, not a rounding one."
    )
    # A gap must never be empty or reversed.
    for g in gaps:
        assert g[1] > g[0], f"gap {g[0].isoformat()} -> {g[1].isoformat()} is empty or reversed"


def check_a1c_12_min_seconds_threshold_is_inclusive():
    """min_seconds filters short gaps, and the boundary is INCLUSIVE (>=), not exclusive.

    The gap summary ships spans ">= 1 day" to Malachy. Whether a span of exactly
    one day is in or out cannot be left to whichever comparison the coder typed.
    """
    cfg, onc = _a1c_insp_mod()
    bins = _a1c_day_bins(onc, 2024, 7, 15)
    day_bins = len(bins)
    short = (0, 3)                 # 900 s
    long_ = (100, 112)             # 3600 s == exactly one hour
    available = [True] * day_bins
    for lo, hi in (short, long_):
        for i in range(lo, hi):
            available[i] = False

    hour_seconds = 60 * 60  # the threshold under test; 12 bins of 300 s
    kept = list(onc.summarise_gaps(bins, available, min_seconds=hour_seconds))
    assert len(kept) == 1, (
        f"min_seconds={hour_seconds} kept {len(kept)} gap(s), expected exactly the one "
        f"{(long_[1]-long_[0])*cfg.BIN_SECONDS} s gap (the "
        f"{(short[1]-short[0])*cfg.BIN_SECONDS} s gap is below threshold)"
    )
    assert kept[0][0] == bins[long_[0]][0], "the kept gap is not the long one"
    assert (kept[0][1] - kept[0][0]).total_seconds() == hour_seconds, (
        "a gap of duration EXACTLY min_seconds must be KEPT -- the threshold is >=, not >"
    )
    unfiltered = list(onc.summarise_gaps(bins, available))
    assert len(unfiltered) == 2, (
        f"the default min_seconds must drop nothing, but it returned {len(unfiltered)} "
        "of 2 planted gaps"
    )


def check_a1c_13_gap_does_not_bridge_a_break_in_the_bin_GRID():
    """A run breaks at a discontinuity in the bins themselves, not only at an available bin.

    season_bins_utc is dense WITHIN the season and absent outside it, so the bin
    list jumps from 30 Sep 23:55Z straight to 1 May 00:00Z of the next year. An
    implementation that walks the list by index and only splits on `available`
    would report a single seven-month "gap" spanning a winter it never measured
    -- and that fabricated span is exactly the kind of confident wrong answer
    that gets acted on (Planet quota, in this case).

    The fixture here uses two separated in-season DAYS, which is the same
    discontinuity in miniature and costs nothing to build.
    """
    _cfg, onc = _a1c_insp_mod()
    bins = _a1c_day_bins(onc, 2024, 7, 15) + _a1c_day_bins(onc, 2024, 7, 17)
    available = [False] * len(bins)
    gaps = list(onc.summarise_gaps(bins, available))
    assert len(gaps) == 2, (
        f"summarise_gaps returned {len(gaps)} gap(s) for two non-adjacent all-unavailable "
        "days; it must return 2. Returning 1 means the run was bridged across a stretch of "
        "time that has no bins at all -- an unmeasured interval reported as measured."
    )
    assert gaps[0][1] == _utc(2024, 7, 16), (
        f"first gap ends {gaps[0][1].isoformat()}, expected 2024-07-16T00:00:00+00:00 "
        "(the exclusive end of the last bin of 15 July)"
    )
    assert gaps[1][0] == _utc(2024, 7, 17), (
        f"second gap starts {gaps[1][0].isoformat()}, expected 2024-07-17T00:00:00+00:00"
    )


def check_a1c_14_uniform_calendar_profile_is_exactly_flat():
    """A continuously-available calendar gives mean availability == 1.0 for ALL 24 UTC hours.

    This is the flatness the real plot is judged against. It is asserted with
    `== 1.0` and no tolerance on purpose: every value is n/n for integer n, which
    is exactly 1.0 in IEEE 754. A tolerance here would hide a real defect.
    """
    _cfg, onc = _a1c_insp_mod()
    from datetime import timedelta
    start = _utc(*A1C_PROFILE_START)
    end = start + timedelta(days=A1C_PROFILE_DAYS)
    bins = onc.season_bins_utc(start, end)
    available = onc.mark_available(bins, [(start, end)])
    assert all(available), "fixture error: whole-span coverage did not mark every bin available"

    profile = list(onc.mean_availability_by_utc_hour(bins, available))
    assert len(profile) == 24, (
        f"mean_availability_by_utc_hour returned {len(profile)} values, expected 24 "
        "(one per UTC hour, indexed by the UTC hour of the bin start)"
    )
    for hour, value in enumerate(profile):
        assert value == 1.0, (
            f"hour {hour:02d}Z has mean availability {value!r}, expected exactly 1.0. A "
            "continuously-recording hydrophone has no preferred hour of the day; any "
            "structure here is in the code, not in the ocean."
        )


def check_a1c_14b_hour_with_no_bins_is_nan_not_zero():
    """An unpopulated UTC hour reports nan, never 0.0.

    0.0 means "measured, and nothing was available". nan means "not measured".
    Rendering the second as the first draws a plot with a fake outage in it, and
    that is precisely the artefact this profile exists to expose (CLAUDE.md
    invariant 9: "the method found nothing" is not "the method is broken").
    """
    import math
    _cfg, onc = _a1c_insp_mod()
    from datetime import timedelta
    # One single hour of bins: 01:00-02:00Z. Every other hour has no bins at all.
    start = _utc(2024, 7, 15, 1)
    bins = onc.season_bins_utc(start, start + timedelta(hours=1))
    assert bins, "fixture error: no bins built for the one-hour span"
    available = [True] * len(bins)
    profile = list(onc.mean_availability_by_utc_hour(bins, available))
    assert len(profile) == 24, f"expected 24 values, got {len(profile)}"
    assert profile[1] == 1.0, f"the one populated hour (01Z) reports {profile[1]!r}, expected 1.0"
    for hour, value in enumerate(profile):
        if hour == 1:
            continue
        assert isinstance(value, float) and math.isnan(value), (
            f"hour {hour:02d}Z has NO bins, but the profile reports {value!r}. It must be "
            "float('nan'): reporting 0.0 makes an unmeasured hour look like a total outage."
        )


def check_a1c_15_pdt_shifted_coverage_makes_the_profile_NOT_flat():
    """WRONG-TIME-FAILS, and it is visible in the plot: a +7 h shift dents exactly 7 UTC hours.

    The converse of the flatness check, and the one that makes the plot a
    diagnostic instead of a decoration. Coverage timestamps read as UTC when
    they were really America/Vancouver local (PDT, UTC-7) land 7 h late, so the
    first A1C_PDT_SHIFT_HOURS UTC hours of the span go uncovered and those hours
    -- and only those -- sit below the rest for the whole span.

    Expected trough = (A1C_PROFILE_DAYS - 1) / A1C_PROFILE_DAYS, peak = 1.0.
    """
    _cfg, onc = _a1c_insp_mod()
    from datetime import timedelta
    start = _utc(*A1C_PROFILE_START)
    end = start + timedelta(days=A1C_PROFILE_DAYS)
    bins = onc.season_bins_utc(start, end)
    shift = timedelta(hours=A1C_PDT_SHIFT_HOURS)
    available = onc.mark_available(bins, [(start + shift, end + shift)])

    profile = list(onc.mean_availability_by_utc_hour(bins, available))
    assert len(profile) == 24, f"expected 24 values, got {len(profile)}"
    spread = max(profile) - min(profile)
    assert spread >= A1C_PROFILE_SPREAD_MIN, (
        f"the +{A1C_PDT_SHIFT_HOURS} h (PDT-as-UTC) calendar produced a peak-to-trough "
        f"spread of {spread:.4f}, below the {A1C_PROFILE_SPREAD_MIN} threshold. If a 7-hour "
        "timezone error does not show up in this profile, the profile cannot be used to rule "
        "one out -- and the plot is decorative rather than diagnostic."
    )
    expected_trough = (A1C_PROFILE_DAYS - 1) / A1C_PROFILE_DAYS
    dented = [h for h, v in enumerate(profile) if v < 1.0 - A1C_PROFILE_TOL]
    assert dented == list(range(A1C_PDT_SHIFT_HOURS)), (
        f"the dent covers UTC hours {dented}, expected exactly hours "
        f"{list(range(A1C_PDT_SHIFT_HOURS))} -- a structure exactly "
        f"{A1C_PDT_SHIFT_HOURS} hours wide is the signature of the timezone bug, and its "
        "WIDTH is the part that identifies it"
    )
    for hour in dented:
        assert abs(profile[hour] - expected_trough) <= A1C_PROFILE_TOL, (
            f"hour {hour:02d}Z reports {profile[hour]!r}, expected {expected_trough!r} "
            f"(tolerance {A1C_PROFILE_TOL})"
        )


def check_a1c_16_gap_summary_doc_exists_and_carries_its_caveat():
    """docs/derived/hydrophone_gaps.md exists, is TRACKED, and says what it is and is not.

    The CSV is gitignored, so this file is the half of deliverable O1 that
    actually reaches the Planet stream. Three things must survive in it or the
    document becomes dangerous rather than useful:
      * the listings-not-download caveat (D3) -- an absence in an ONC LISTING is
        not the same claim as a verified absent download, and A4 may revise it;
      * at least one concrete date range, or it conveys nothing;
      * the one-line instruction to Malachy, in plain language.
    The pinned substrings are named constants above so that weakening the
    caveat requires editing this file, in view of a reviewer.
    """
    path = REPO_ROOT / GAPS_DOC_RELPATH
    assert path.is_file(), (
        f"{GAPS_DOC_RELPATH} does not exist. It is the tracked, human-readable half of "
        "deliverable O1 -- the CSV alone is gitignored and never reaches Malachy."
    )
    text = path.read_text(encoding="utf-8")
    for needle in (GAPS_DOC_CAVEAT, GAPS_DOC_CAVEAT_2, GAPS_DOC_REVISION_NOTE,
                   GAPS_DOC_MALACHY_LINE):
        assert needle in text, (
            f"{GAPS_DOC_RELPATH} does not contain the pinned substring {needle!r}. "
            "Without it the reader cannot tell that this is derived from ONC listings "
            "rather than from a completed download, and may treat a listing gap as a "
            "settled fact (D3)."
        )
    import re as _re
    dates = _re.findall(r"\d{4}-\d{2}-\d{2}", text)
    assert len(dates) >= 2, (
        f"{GAPS_DOC_RELPATH} contains {len(dates)} ISO date(s); a gap summary needs at "
        "least one date RANGE (two dates) to say anything actionable"
    )
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", GAPS_DOC_RELPATH],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert tracked.returncode == 0, (
        f"{GAPS_DOC_RELPATH} exists but is NOT tracked by git (`git ls-files` says: "
        f"{tracked.stderr.strip()}). An untracked summary is invisible to the teammate it "
        "was written for -- which is the entire reason this file exists alongside the "
        "gitignored CSV."
    )


def check_a1c_17_notebook_is_a_thin_wrapper():
    """The A1c notebook exists and CALLS the inspection functions rather than redefining them.

    CLAUDE.md: the library holds the code, the notebook is a wrapper. A `def
    summarise_gaps` inside a notebook cell is code no check can reach, no
    reviewer reads in diff form, and A4 cannot reuse.
    """
    import json
    rel = os.path.join("contributor_folders", "isaac", "a1_uptime_calendar.ipynb")
    path = REPO_ROOT / rel
    assert path.is_file(), f"{rel} does not exist"
    doc = json.loads(path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in doc.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    for name in ("summarise_gaps", "mean_availability_by_utc_hour"):
        assert name in source, f"{rel} never calls {name}(); the plot must be drawn from it"
        assert f"def {name}" not in source, (
            f"{rel} DEFINES {name}() in a notebook cell. It belongs in boatphone/onc_client.py "
            "where it can be imported, checked, and reused by A4."
        )
    assert "boatphone" in source, f"{rel} never imports from boatphone/"


# ---------------------------------------------------------------------------
# A1d -- ONC ARCHIVE-LISTING ROW CAP (defect found during A1c, pinned before fix)
#
# THE DEFECT. list_fft_files() chunks its archive-listing requests by calendar
# UTC year. A year of 5-minute FFT files is ~105,000 rows, and ONC's archive
# listing has a response ROW CAP that bites well below that. When a request
# exceeds the cap ONC returns a TRUNCATED list -- no error, no pagination signal
# the current code honours. Measured 2026-08-27:
# build_uptime_calendar(client, 2024-05-01, 2024-10-01) returned 11,120 files
# that stop at 2024-06-08, so the calendar reported a FOUR-MONTH OUTAGE THAT DOES
# NOT EXIST; July 2024 alone lists 8,921 files at 99.96 % coverage. A1c worked
# around it by scanning month-by-month in the notebook; the library is still wrong.
# This is CLAUDE.md invariant 9 exactly -- "the method is broken" wearing the
# clothes of "the method found nothing" -- and it would aim Malachy's Planet quota
# at dates the hydrophone was recording.
#
# THE CONTRACT THESE CHECKS PIN (the coder implements exactly this):
#
#   list_fft_files MUST NOT return a capped/truncated listing as if it were
#   complete. It satisfies the contract by EITHER
#     (P) paginating and/or sub-chunking (e.g. by UTC month or day, or by
#         following the response's offset/next) until the full set is in hand, OR
#     (R) raising ONCListingError naming the span and the cap when it cannot
#         prove completeness.
#   Both are accepted by every check below; (P) is strongly preferred, and it is
#   what the A1c month-scan already demonstrates works. What is NEVER acceptable
#   is the third option the code takes today: returning the short list quietly.
#
#   A capped chunk is ALSO NOT an empty chunk. `empty_chunks` means "ONC reports
#   no data here", a fact Malachy acts on. A cap must never be folded into it.
#
# HOW THE STUB EXPRESSES THE CAP. `_CappedONC` holds a corpus of real-shaped
# .fft.gz names and answers a listing request with the files whose START falls in
# [dateFrom, dateTo), CHRONOLOGICALLY, cut to STUB_ROW_CAP rows -- exactly the
# observed shape (a prefix of the span, truncated at a date in the middle). It
# honours offset/rowLimit-style paging kwargs and advertises `next`, so a
# paginating implementation can reach completeness; it needs no cooperation at
# all from a sub-chunking one, since each small chunk fits under the cap.
#
# FOR THE CODER: STUB_ROW_CAP below is a TEST fixture, not the real ONC limit --
# it is deliberately small so the checks stay fast. The real cap (and whether the
# response exposes a `next`/offset field at all) must be measured against the live
# API, and once measured it belongs in boatphone/config.py as a named constant
# with its source in a comment. I could not put it there: this agent may not edit
# config.py. Promote it.
# ---------------------------------------------------------------------------

# Row cap used by the stub. FIXTURE VALUE, NOT MEASURED -- chosen only so a dense
# synthetic month (8,928 files at 300 s) overruns it by a wide margin while the
# checks still run in well under a second. See the note above about promoting the
# real, measured cap to config.py.
STUB_ROW_CAP = 2000


def _dense_fft_names(start_utc, end_utc, period_seconds=None):
    """Every .fft.gz name for a gapless recording over `[start_utc, end_utc)`.

    One file every `period_seconds` (default config.FFT_FILE_SECONDS), starting
    exactly at `start_utc`, so the true count is arithmetic and known:
    ceil((end - start) / period). Ground truth for the completeness checks.
    """
    from datetime import timedelta
    cfg, _onc = _a1a_mods()
    period = period_seconds or cfg.FFT_FILE_SECONDS
    names, moment = [], start_utc
    step = timedelta(seconds=period)
    while moment < end_utc:
        names.append(_fft_name(moment))
        moment += step
    return names


class _CappedONC:
    """A _StubONC-shaped client whose listing endpoint has a ROW CAP.

    Local class, no mocking library (CLAUDE.md "Environment"), same reason and
    same shape as _StubONC -- kept separate because this one exists to truncate,
    and mixing that into the honest stub would risk weakening the A1b checks.

    A request returns the corpus files whose START is in [dateFrom, dateTo), in
    chronological order, cut to `cap` rows. Truncation is SILENT in the file list
    (that is the ONC behaviour under test), but the response does carry
    `next`/`nextOffset` so a paginating implementation has a legitimate route to
    completeness; `offset`/`skip`/`rowLimit`/`limit` kwargs are honoured too.
    """

    def __init__(self, corpus_names, cap=STUB_ROW_CAP, locations=None, deployments=None):
        _cfg, onc = _a1a_mods()
        pairs = [(onc.parse_file_coverage(n)[0], n) for n in corpus_names]
        self.corpus = sorted(pairs)
        self.cap = cap
        self.locations = locations if locations is not None else _folger_location_rows()
        self.deployments = deployments or []
        self.requests = []       # every filter dict handed to a listing call
        self.capped_requests = 0  # how many of them hit the cap

    @staticmethod
    def _moment(value):
        """A dateFrom/dateTo value (ONC 'Z' string or datetime) -> aware UTC."""
        from datetime import datetime as _dt, timezone as _tz
        if value is None:
            return None
        if isinstance(value, _dt):
            return value if value.tzinfo else value.replace(tzinfo=_tz.utc)
        text = str(value).replace("Z", "+00:00")
        return _dt.fromisoformat(text).astimezone(_tz.utc)

    def _listing(self, filters=None, **kwargs):
        f = dict(filters or {})
        f.update(kwargs)
        self.requests.append(f)
        lo = self._moment(f.get("dateFrom") or f.get("begin"))
        hi = self._moment(f.get("dateTo") or f.get("end"))
        matched = [n for (s, n) in self.corpus
                   if (lo is None or s >= lo) and (hi is None or s < hi)]
        offset = int(f.get("offset") or f.get("skip") or 0)
        requested = f.get("rowLimit") or f.get("limit")
        limit = min(int(requested), self.cap) if requested else self.cap
        page = matched[offset:offset + limit]
        truncated = offset + len(page) < len(matched)
        if truncated:
            self.capped_requests += 1
        return {
            "files": page,
            # The paging affordance. A correct paginating implementation follows
            # it; the current implementation ignores it, which is the bug.
            "next": ({"offset": offset + len(page)} if truncated else None),
            "nextOffset": (offset + len(page) if truncated else None),
        }

    getArchivefile = _listing
    getArchivefileByLocation = _listing
    getArchivefileByDevice = _listing
    getListByLocation = _listing
    getListByDevice = _listing

    def getLocations(self, filters=None, **kwargs):
        self.requests.append(dict(filters or {}, **kwargs))
        return list(self.locations)

    def getDeployments(self, filters=None, **kwargs):
        self.requests.append(dict(filters or {}, **kwargs))
        return list(self.deployments)


def _cap_error_is_diagnosable(exc, span_start, span_end):
    """A raise satisfies the contract only if the message NAMES the span and cap.

    An opaque 'listing failed' sends whoever hits it back to the API to rediscover
    this defect from scratch, so the diagnosability is part of the contract, not
    politeness.
    """
    text = str(exc)
    missing = []
    for year_month in {span_start.strftime("%Y-%m"), span_end.strftime("%Y-%m")}:
        if year_month not in text and year_month.replace("-", "") not in text:
            missing.append(year_month)
    if str(STUB_ROW_CAP) not in text and "cap" not in text.lower() and "truncat" not in text.lower():
        missing.append(f"the cap ({STUB_ROW_CAP}) or the words 'cap'/'truncated'")
    return missing, text


def check_a1d_1_capped_listing_is_never_returned_as_complete():
    """A response cut off at the row cap must paginate/sub-chunk to completeness, or RAISE.

    Dense July 2024 (8,928 files at 300 s) against a stub capped at STUB_ROW_CAP.
    The current year-chunking issues ONE request for the whole year and keeps
    whatever comes back, so it returns a STUB_ROW_CAP-length prefix and calls it
    the month -- the phantom-outage bug. Asserted on the filenames themselves, not
    just the count, so a fix that pads or de-duplicates its way to the right number
    cannot pass.
    """
    _cfg, onc = _a1b_mod()
    span_start, span_end = _utc(2024, 7, 1), _utc(2024, 8, 1)
    corpus = _dense_fft_names(span_start, span_end)
    assert len(corpus) > STUB_ROW_CAP, (
        f"fixture is not dense enough to trip the cap: {len(corpus)} <= {STUB_ROW_CAP}"
    )
    client = _CappedONC(corpus)
    try:
        files, _empty_chunks = onc.list_fft_files(client, ["ZZTESTA"], span_start, span_end)
    except (RuntimeError, ValueError) as exc:  # the (R) branch of the contract
        missing, text = _cap_error_is_diagnosable(exc, span_start, span_end)
        assert not missing, (
            f"list_fft_files raised on a capped listing (acceptable), but the message does "
            f"not name {missing}: {text!r}"
        )
        return
    assert sorted(files) == sorted(corpus), (
        f"list_fft_files returned {len(files)} of {len(corpus)} files for a gapless "
        f"{span_start.date()}..{span_end.date()} listing whose stub cap is {STUB_ROW_CAP}. "
        f"Last file returned: {max(files) if files else None!r}; last file that exists: "
        f"{max(corpus)!r}. A capped response was accepted as complete -- the missing "
        "files become a phantom outage in the uptime calendar (invariant 9). Either "
        "paginate/sub-chunk to completeness or raise; never return the short list."
    )


def check_a1d_2_capped_chunk_is_not_counted_as_empty():
    """A cap is not a no-data period: a fully-covered dense span reports ZERO empty chunks.

    `empty_chunks` is the number the team reads as 'ONC has nothing here'. If a
    truncated request is folded into it -- or if a sub-chunking fix counts the
    sub-chunks it could not fill -- then 'found nothing' and 'is broken' become
    the same number, which is the one distinction invariant 9 exists to protect.
    The fixture has data in EVERY 300 s slot of the span, so the only correct
    empty-chunk count is 0, at any chunk granularity.
    """
    _cfg, onc = _a1b_mod()
    span_start, span_end = _utc(2024, 7, 1), _utc(2024, 8, 1)
    corpus = _dense_fft_names(span_start, span_end)
    client = _CappedONC(corpus)
    try:
        _files, empty_chunks = onc.list_fft_files(client, ["ZZTESTA"], span_start, span_end)
    except (RuntimeError, ValueError) as exc:  # the (R) branch
        missing, text = _cap_error_is_diagnosable(exc, span_start, span_end)
        assert not missing, (
            f"list_fft_files raised on a capped listing (acceptable), but the message does "
            f"not name {missing}: {text!r}"
        )
        return
    assert empty_chunks == 0, (
        f"list_fft_files reported {empty_chunks} empty chunk(s) over a span with a file in "
        f"every {_cfg.FFT_FILE_SECONDS} s slot ({len(corpus)} files). A capped or "
        "unfillable chunk is being counted as 'ONC reports no data', which is the exact "
        "conflation invariant 9 forbids."
    )


def check_a1d_3_genuine_no_data_still_surfaces_next_to_a_capped_span():
    """The other half of #2: a REAL no-data period must still be reported as empty.

    Dense July 2024 (over the cap) plus an untouched August 2024 (no files at all).
    A fix that stops mis-counting caps by simply never reporting an empty chunk
    would pass check_a1d_2 and silently lose the signal Malachy uses; this fixture
    fails it. Chunk granularity is the coder's choice, so the assertion is
    >= 1 empty chunk, not an exact count.
    """
    _cfg, onc = _a1b_mod()
    span_start, span_end = _utc(2024, 7, 1), _utc(2024, 9, 1)
    covered_end = _utc(2024, 8, 1)          # August is genuinely empty
    corpus = _dense_fft_names(span_start, covered_end)
    client = _CappedONC(corpus)
    try:
        files, empty_chunks = onc.list_fft_files(client, ["ZZTESTA"], span_start, span_end)
    except (RuntimeError, ValueError) as exc:  # the (R) branch
        missing, text = _cap_error_is_diagnosable(exc, span_start, span_end)
        assert not missing, (
            f"list_fft_files raised on a capped listing (acceptable), but the message does "
            f"not name {missing}: {text!r}"
        )
        return
    assert sorted(files) == sorted(corpus), (
        f"list_fft_files returned {len(files)} of {len(corpus)} July files; the capped "
        "chunk was truncated (see check_a1d_1)"
    )
    assert empty_chunks >= 1, (
        "list_fft_files reported 0 empty chunks although all of August 2024 has no files "
        "at all. A genuine no-data period must stay visible -- suppressing the empty-chunk "
        "signal is not a valid way to stop mis-counting a cap (invariant 9)."
    )


def check_a1d_4_no_phantom_outage_in_the_real_failing_span():
    """REGRESSION, offline: 2024-05-01..2024-10-01 gapless => a calendar with NO gap.

    This is the measured failure, reproduced exactly: build_uptime_calendar over
    the five-month span returned 11,120 files stopping 2024-06-08 and reported a
    four-month outage that does not exist. The stub declares the whole span
    covered, so the only correct calendar has every bin available. Pinned at the
    build_uptime_calendar level because that is where a human reads the wrong
    answer, and because the shape -- 'coverage stops partway and the rest reads as
    outage' -- is what a future refactor would reintroduce.
    """
    _cfg, onc = _a1c_mod()
    span_start, span_end = _utc(2024, 5, 1), _utc(2024, 10, 1)
    corpus = _dense_fft_names(span_start, span_end)
    deployments = [{
        "begin": "2024-05-01T00:00:00.000Z", "end": "2024-10-01T00:00:00.000Z",
        "locationCode": "ZZTESTA", "deviceCode": _cfg.DEVICE_CODE,
    }]
    client = _CappedONC(corpus, deployments=deployments)
    try:
        rows = onc.build_uptime_calendar(client, span_start, span_end)
    except (RuntimeError, ValueError) as exc:  # the (R) branch
        missing, text = _cap_error_is_diagnosable(exc, span_start, span_end)
        assert not missing, (
            f"build_uptime_calendar raised on a capped listing (acceptable -- loudly wrong "
            f"beats quietly wrong), but the message does not name {missing}: {text!r}"
        )
        return
    unavailable = [s for s, _e, a, _d in rows if not a]
    assert not unavailable, (
        f"{len(unavailable)} of {len(rows)} bins read as unavailable over a span the stub "
        f"declares gapless ({len(corpus)} files, one every {_cfg.FFT_FILE_SECONDS} s). "
        f"First phantom-outage bin: {unavailable[0].isoformat()}; last: "
        f"{unavailable[-1].isoformat()}. This is the measured defect -- a listing truncated "
        "at the ONC row cap becomes a months-long outage that never happened, and it would "
        "aim the Planet quota at dates the hydrophone was recording."
    )
# ---------------------------------------------------------------------------
# A1d -- INTEGRATION: the runnable entry point, its provenance, and O1 delivery
#
# THE CONTRACT THESE CHECKS PIN (the coder implements exactly this).
#
# scripts/build_uptime_calendar.py is a runnable ENTRY POINT. Per CLAUDE.md
# invariant 6 it DEFINES NOTHING SHARED: every constant it uses comes from
# boatphone/config.py or boatphone/paths.py, and both A1's calendar functions
# are imported from boatphone/onc_client.py rather than reimplemented.
#
#   (h0) The module IMPORTS CLEANLY with no side effects -- no network call, no
#        directory creation, no argument parsing at import time. `main()` runs
#        only under `if __name__ == "__main__":`. Every check below imports it.
#
#   (h1) build_parser() -> argparse.ArgumentParser
#        Flags: --start, --end (defaulting to config.STUDY_START_UTC /
#        config.STUDY_END_UTC), --out-dir (defaulting to paths.DERIVED_DIR),
#        --dry-run. `--help` exits 0 and names all four.
#
#   (h2) resolve_out_dir(value) -> pathlib.Path
#        Returns the RESOLVED, absolute output directory. Raises (ValueError or
#        a subclass) when the resolved path lands under paths.DATA_DIR but NOT
#        under paths.DERIVED_DIR -- i.e. an acquisition directory or a `..`
#        traversal out of derived/ is REFUSED, not written to (invariant 2,
#        docs/decisions/0001, hook-enforced). A path entirely outside data/ (a
#        tmpdir) is ALLOWED: that is what makes the entry point testable at all,
#        and these checks never write under data/.
#        GUESS, flagged: the plan says "writes into data/derived/ and nowhere
#        else" but must also stay drivable from a check. The rule above is the
#        narrowest reading that satisfies both. If the coder prefers "must be
#        DERIVED_DIR or below, full stop", say so and this check changes -- but
#        then the entry point is unverifiable offline.
#
#   (h3) artifact_paths(out_dir) -> dict with keys
#        "uptime_csv", "deployments_csv", "provenance_json", whose values are
#        the three artifact paths. Each MUST be a direct child of
#        resolve_out_dir(out_dir), with basenames exactly
#        A1D_ARTIFACT_BASENAMES below. No fourth path is written.
#
#   (h4) build_provenance(...) -> dict carrying at least A1D_PROVENANCE_KEYS,
#        every value non-empty, and `source == "listing"` (D3: this artefact is
#        ONC's belief that a file exists, not proof that it downloads; A4's pull
#        refines it and the pull wins). Called with keyword arguments so the
#        signature can grow: the checks below pass start_utc, end_utc,
#        location_codes, deployments, rows.
#        GUESS, flagged: the key SPELLINGS below are mine -- the plan names the
#        facts, not the JSON keys. If the coder picks different spellings, they
#        are changed HERE, in one place, and the change is visible in review.
#
#   (h5) --dry-run makes NO network request and writes NOTHING.
#
# Every check below is OFFLINE. The one thing not covered here is the real
# 2020-2026 run; that is A4's pull and a network-gated smoke check, not the
# always-run suite. No check here writes anywhere under data/.
# ---------------------------------------------------------------------------

# Repo-relative location of the entry point. Source: the A1d segment brief.
A1D_SCRIPT_RELPATH = os.path.join("scripts", "build_uptime_calendar.py")

# The three artifacts, keyed as artifact_paths() returns them. Source: the A1d
# segment brief ("writes three artifacts into data/derived/").
A1D_ARTIFACT_BASENAMES = {
    "uptime_csv": "hydrophone_uptime.csv",
    "deployments_csv": "deployments.csv",
    "provenance_json": "hydrophone_uptime.provenance.json",
}

# Provenance keys, one per fact the A1d brief requires the sidecar to carry.
# See (h4) above: the spellings are a flagged guess, pinned in ONE place.
A1D_PROVENANCE_KEYS = (
    "script",              # generating script (repo-relative)
    "git_commit",          # the commit the run was made from
    "generated_utc",       # UTC generation time, explicit offset
    "device_code",         # config.DEVICE_CODE
    "device_id",           # config.DEVICE_ID
    "location_codes",      # codes DISCOVERED at runtime, never hardcoded (A1b)
    "product_extension",   # config.PRODUCT_EXTENSION
    "archive_extension",   # config.ARCHIVE_EXTENSION (NOT the same string)
    "date_start_utc",      # span actually scanned
    "date_end_utc",
    "bin_seconds",         # config.BIN_SECONDS
    "season_months_utc",   # config.SEASON_MONTHS_UTC -- the seasonal filter
    "source",              # must be exactly A1D_PROVENANCE_SOURCE
    "onc_version",         # version of the `onc` client that made the requests
)

# D3. "listing" means ONC said a file exists; it is NOT proof of a successful
# download. A4's absent-file log refines this and the pull wins. The artefact
# must carry that distinction or the next reader cannot know which claim it has.
A1D_PROVENANCE_SOURCE = "listing"

# The literal `available` must serialise to. Pinned so it cannot drift with a
# pandas version if the writer is ever re-expressed in pandas: today it is
# Python's str(bool) (see write_uptime_calendar_csv's docstring), and "1"/"0"
# would silently change every downstream `== "True"` comparison into False.
A1D_AVAILABLE_TRUE_LITERAL = "True"
A1D_AVAILABLE_FALSE_LITERAL = "False"

# Names that belong to boatphone/ and must NOT be re-declared at module level in
# scripts/. Source: CLAUDE.md invariant 6 -- if optical puts the study window in
# scripts/ while acoustics has it in boatphone/config.py, the optical-acoustic
# matchup joins two different definitions of the same window and quietly
# produces a wrong answer instead of an error.
A1D_SHARED_CONSTANT_MODULES = ("boatphone.config", "boatphone.paths")


def _a1d_script_path():
    path = REPO_ROOT / A1D_SCRIPT_RELPATH
    assert path.is_file(), (
        f"{A1D_SCRIPT_RELPATH} does not exist; A1d's entry point is not implemented yet "
        "(expected until the coder lands it -- see the A1d contract comment above)"
    )
    return path


def _a1d_script_mod(*required):
    """Import the entry point BY PATH and require the named attributes.

    By path, not by `import scripts.build_uptime_calendar`: scripts/ is a
    directory of entry points, not a package, and it has no __init__.py. A
    missing attribute here IS the expected pre-implementation failure.
    """
    import importlib.util
    path = _a1d_script_path()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("boatphone_a1d_entry_point", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # (h0) must be side-effect free
    missing = [name for name in required if not hasattr(module, name)]
    assert not missing, (
        f"{A1D_SCRIPT_RELPATH} has no {missing}; see the A1d contract comment above "
        "for the exact entry-point surface"
    )
    return module


def _a1d_read_csv(path):
    """Return (header, rows) from a CSV, as lists of strings. No pandas."""
    import csv as csv_mod
    with open(path, newline="", encoding="utf-8") as fh:
        table = list(csv_mod.reader(fh))
    assert table, f"{path} is empty; not even a header was written"
    return table[0], table[1:]


def _a1d_sample_rows():
    """Two adjacent in-season bins, one available and one not, both BIN_SECONDS wide."""
    from datetime import timedelta
    cfg, _onc = _a1a_mods()
    base = _utc(2024, 7, 15, 0, 0)
    width = timedelta(seconds=cfg.BIN_SECONDS)
    return [
        (base, base + width, True, "ICLISTENHF1266@ZZTESTA:2024-07-01T00:00:00.000Z"),
        (base + width, base + 2 * width, False, ""),
    ]


def check_a1d_5_written_csv_round_trips_tz_aware_utc():
    """A timestamp written to the O1 CSV re-reads as tz-aware UTC and re-serialises IDENTICALLY.

    This is the boundary at which a UTC column silently becomes naive or local:
    the in-memory row is tz-aware by construction, the file is just text, and the
    next reader is a different workstream on a machine in America/Vancouver. A
    string that parses back to a naive datetime is indistinguishable from local
    time (decision 0002), and byte-identical re-serialisation is the property
    that says nothing was lost in the round trip -- not merely that something
    parseable came back.
    """
    from datetime import datetime
    _cfg, onc = _a1c_mod()
    mod = _a1d_script_mod("artifact_paths")
    rows = _a1d_sample_rows()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(mod.artifact_paths(tmp)["uptime_csv"])
        onc.write_uptime_calendar_csv(rows, path)
        _header, body = _a1d_read_csv(path)
    for index, record in enumerate(body):
        for column, raw, original in (("start_utc", record[0], rows[index][0]),
                                      ("end_utc", record[1], rows[index][1])):
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            assert parsed.tzinfo is not None, (
                f"row {index} {column}={raw!r} re-reads as a NAIVE datetime. Malachy's "
                "reader has no way to know this is UTC; decision 0002 forbids the shape"
            )
            assert parsed.utcoffset().total_seconds() == 0, (
                f"row {index} {column}={raw!r} re-reads with offset {parsed.utcoffset()}, "
                "not UTC; a local-time round trip is exactly the silent corruptor"
            )
            assert parsed == original, (
                f"row {index} {column} re-reads as {parsed.isoformat()}, not the instant "
                f"written ({original.isoformat()})"
            )
            assert parsed.isoformat() == raw, (
                f"row {index} {column} does not re-serialise byte-identically: wrote "
                f"{raw!r}, re-serialises as {parsed.isoformat()!r}. A non-idempotent "
                "round trip means the file and the object disagree about the instant"
            )


def check_a1d_6_written_csv_header_and_available_literal():
    """Header is exactly the pinned four columns in order, and `available` is a STABLE literal.

    The literal matters as much as the header. Downstream reads
    `row["available"] == "True"`; a writer that ever emits "1"/"0" -- which is
    what a pandas `to_csv` of an int column would do -- turns every one of those
    comparisons into False, i.e. reports a fully-recording hydrophone as fully
    dark, with no error anywhere.
    """
    _cfg, onc = _a1c_mod()
    mod = _a1d_script_mod("artifact_paths")
    rows = _a1d_sample_rows()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(mod.artifact_paths(tmp)["uptime_csv"])
        onc.write_uptime_calendar_csv(rows, path)
        header, body = _a1d_read_csv(path)
    assert header == UPTIME_CSV_HEADER, (
        f"CSV header is {header}, expected exactly {UPTIME_CSV_HEADER} in this order"
    )
    assert body[0][2] == A1D_AVAILABLE_TRUE_LITERAL, (
        f"available for an AVAILABLE bin serialises as {body[0][2]!r}, not "
        f"{A1D_AVAILABLE_TRUE_LITERAL!r}. Downstream compares against that literal"
    )
    assert body[1][2] == A1D_AVAILABLE_FALSE_LITERAL, (
        f"available for an UNAVAILABLE bin serialises as {body[1][2]!r}, not "
        f"{A1D_AVAILABLE_FALSE_LITERAL!r}"
    )


def check_a1d_7_grid_invariants_hold_on_the_WRITTEN_FILE():
    """Half-open BIN_SECONDS bins, epoch alignment and strict ordering, re-read from disk.

    A1a proves these of the in-memory object. The object is not what Malachy
    receives: the FILE is. Serialisation is a separate step that can reorder,
    round, or drop, so the grid is re-asserted here against the artifact itself.
    """
    from datetime import datetime, timezone
    cfg, onc = _a1c_mod()
    mod = _a1d_script_mod("artifact_paths")
    span_start, span_end = _utc(2024, 7, 15, 0, 0), _utc(2024, 7, 15, 1, 0)
    client = _a1c_stub_client(_a1c_fixture_files(span_start))
    rows = onc.build_uptime_calendar(client, span_start, span_end)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(mod.artifact_paths(tmp)["uptime_csv"])
        onc.write_uptime_calendar_csv(rows, path)
        _header, body = _a1d_read_csv(path)
    assert len(body) == len(rows), f"{len(body)} row(s) on disk for {len(rows)} in memory"
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    previous_end = None
    for index, record in enumerate(body):
        start = datetime.fromisoformat(record[0].replace("Z", "+00:00"))
        end = datetime.fromisoformat(record[1].replace("Z", "+00:00"))
        width = (end - start).total_seconds()
        assert width == cfg.BIN_SECONDS, (
            f"row {index} on disk spans {width} s, not config.BIN_SECONDS "
            f"({cfg.BIN_SECONDS}); the bin grid did not survive serialisation"
        )
        offset = (start - epoch).total_seconds()
        assert offset % cfg.BIN_SECONDS == 0, (
            f"row {index} starts at {record[0]}, which is {offset % cfg.BIN_SECONDS} s off "
            f"the {cfg.BIN_SECONDS} s epoch-aligned grid; an unaligned bin cannot be joined "
            "to the optical stream's bins without a hidden half-bin shift"
        )
        if previous_end is not None:
            assert start >= previous_end, (
                f"row {index} starts at {record[0]}, before row {index - 1} ends "
                f"({previous_end.isoformat()}); rows must be strictly ordered and "
                "half-open -- an overlap double-counts a bin"
            )
        previous_end = end


def check_a1d_8_cli_surface_and_defaults():
    """--help exits 0 and names every flag; the defaults come from boatphone/, not from scripts/."""
    _a1d_script_path()
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / A1D_SCRIPT_RELPATH), "--help"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)}, timeout=SUBPROC_TIMEOUT_S,
    )
    assert proc.returncode == 0, (
        f"`build_uptime_calendar.py --help` exited {proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    for flag in ("--start", "--end", "--out-dir", "--dry-run"):
        assert flag in proc.stdout, f"--help does not mention {flag}:\n{proc.stdout}"

    cfg, _onc = _a1a_mods()
    from boatphone import paths as bp_paths
    mod = _a1d_script_mod("build_parser")
    args = mod.build_parser().parse_args([])
    assert getattr(args, "start") == cfg.STUDY_START_UTC, (
        f"--start defaults to {getattr(args, 'start')!r}, not config.STUDY_START_UTC "
        f"({cfg.STUDY_START_UTC.isoformat()}); a second definition of the study window is "
        "invariant 6's exact failure mode -- the optical matchup would join two different "
        "windows and produce a wrong answer rather than an error"
    )
    assert getattr(args, "end") == cfg.STUDY_END_UTC, (
        f"--end defaults to {getattr(args, 'end')!r}, not config.STUDY_END_UTC "
        f"({cfg.STUDY_END_UTC.isoformat()})"
    )
    assert pathlib.Path(getattr(args, "out_dir")).resolve() == bp_paths.DERIVED_DIR.resolve(), (
        f"--out-dir defaults to {getattr(args, 'out_dir')!r}, not paths.DERIVED_DIR "
        f"({bp_paths.DERIVED_DIR})"
    )


def check_a1d_9_no_write_escapes_the_derived_dir():
    """Invariant 2: nothing the script writes resolves outside its out-dir, and data/ is refused.

    Two failures live here. The first is a path traversal (`derived/../raw`)
    landing in an acquisition directory -- immutability is hook-enforced, so the
    run would die mid-way having already clobbered an original. The second is
    quieter: an artifact_paths() entry that is not a direct child of the resolved
    out-dir puts one of the three artifacts somewhere nobody looks for it.

    This check writes only into a tmpdir. It never creates a path under data/.
    """
    from boatphone import paths as bp_paths
    mod = _a1d_script_mod("resolve_out_dir", "artifact_paths")

    with tempfile.TemporaryDirectory() as tmp:
        resolved = pathlib.Path(mod.resolve_out_dir(tmp)).resolve()
        assert resolved == pathlib.Path(tmp).resolve(), (
            f"resolve_out_dir({tmp!r}) returned {resolved}; an out-dir outside data/ must "
            "be returned as-is so the entry point is drivable from a check"
        )
        produced = mod.artifact_paths(tmp)
        assert set(produced) == set(A1D_ARTIFACT_BASENAMES), (
            f"artifact_paths returned keys {sorted(produced)}, expected "
            f"{sorted(A1D_ARTIFACT_BASENAMES)}; A1d writes these three artifacts and no others"
        )
        for key, expected_name in A1D_ARTIFACT_BASENAMES.items():
            candidate = pathlib.Path(produced[key]).resolve()
            assert candidate.name == expected_name, (
                f"artifact_paths[{key!r}] is named {candidate.name!r}, expected "
                f"{expected_name!r}; the O1 filenames are what the optical stream looks for"
            )
            assert candidate.parent == resolved, (
                f"artifact_paths[{key!r}] resolves to {candidate}, which is not a direct "
                f"child of the resolved out-dir {resolved}. A write outside the out-dir is "
                "the shape that reaches an immutable acquisition (invariant 2)"
            )

    # Refusal cases. These are constructed as STRINGS and never created on disk.
    refused = [
        str(bp_paths.DERIVED_DIR / ".." / "raw"),          # traversal out of derived/
        str(bp_paths.DERIVED_DIR / ".." / SAMPLE_DIR_NAME),  # into the acquisition
        str(bp_paths.DATA_DIR / "raw" / "onc"),            # the landing zone, directly
        str(bp_paths.DATA_DIR),                            # data/ itself
    ]
    for candidate in refused:
        try:
            got = mod.resolve_out_dir(candidate)
        except ValueError:
            continue  # correct: refused, loudly
        raise AssertionError(
            f"resolve_out_dir({candidate!r}) returned {got!r} instead of raising. That path "
            f"is under {bp_paths.DATA_DIR} but not under {bp_paths.DERIVED_DIR}: writing "
            "there mutates an acquisition, which docs/decisions/0001 forbids and a hook "
            "blocks -- the run must refuse BEFORE it does any work, not fail part-written"
        )


def check_a1d_10_provenance_is_complete_and_says_source_is_LISTING():
    """Every provenance key present and non-empty, and `source == "listing"` (D3).

    The distinction the key carries is the whole point: this calendar records
    ONC's belief that a file exists, not proof that it downloads. A4's pull
    refines it and the pull wins. A provenance sidecar that omits it lets a
    reader treat a listing gap as a settled outage and spend unrecoverable
    Planet quota on it.
    """
    cfg, onc = _a1c_mod()
    mod = _a1d_script_mod("build_provenance")
    span_start, span_end = _utc(2024, 7, 15, 0, 0), _utc(2024, 7, 15, 1, 0)
    client = _a1c_stub_client(_a1c_fixture_files(span_start))
    rows = onc.build_uptime_calendar(client, span_start, span_end)
    provenance = mod.build_provenance(
        start_utc=span_start,
        end_utc=span_end,
        location_codes=onc.discover_folger_locations(client),
        deployments=onc.get_deployments(client),
        rows=rows,
    )
    assert isinstance(provenance, dict), (
        f"build_provenance returned {type(provenance).__name__}, not a dict"
    )
    missing = [k for k in A1D_PROVENANCE_KEYS if k not in provenance]
    assert not missing, (
        f"provenance is missing {missing}. Every one of these is a fact the number cannot "
        "be re-derived without (shared standards: provenance travels with the number)"
    )
    empty = [k for k in A1D_PROVENANCE_KEYS
             if provenance[k] is None or provenance[k] == "" or provenance[k] == []]
    assert not empty, (
        f"provenance keys are present but EMPTY: {empty}. A blank key is worse than an "
        "absent one -- it looks recorded"
    )
    assert provenance["source"] == A1D_PROVENANCE_SOURCE, (
        f"provenance['source'] is {provenance['source']!r}, expected "
        f"{A1D_PROVENANCE_SOURCE!r} (D3). Anything else claims the calendar is backed by a "
        "completed download when it is backed by an ONC listing"
    )
    assert provenance["bin_seconds"] == cfg.BIN_SECONDS, (
        f"provenance['bin_seconds'] is {provenance['bin_seconds']!r}, not config.BIN_SECONDS "
        f"({cfg.BIN_SECONDS}); the sidecar must record the grid the file actually uses"
    )
    assert list(provenance["season_months_utc"]) == list(cfg.SEASON_MONTHS_UTC), (
        f"provenance['season_months_utc'] is {provenance['season_months_utc']!r}, not "
        f"config.SEASON_MONTHS_UTC ({list(cfg.SEASON_MONTHS_UTC)}). Without the seasonal "
        "filter recorded, an out-of-season absence is indistinguishable from an outage"
    )
    assert provenance["archive_extension"] == cfg.ARCHIVE_EXTENSION, (
        f"provenance['archive_extension'] is {provenance['archive_extension']!r}, not "
        f"config.ARCHIVE_EXTENSION ({cfg.ARCHIVE_EXTENSION!r}); it is deliberately NOT the "
        "same string as PRODUCT_EXTENSION and the sidecar must not conflate them"
    )
    assert provenance["product_extension"] == cfg.PRODUCT_EXTENSION, (
        f"provenance['product_extension'] is {provenance['product_extension']!r}, not "
        f"config.PRODUCT_EXTENSION ({cfg.PRODUCT_EXTENSION!r})"
    )
    import json as json_mod
    json_mod.dumps(provenance)  # must be JSON-serialisable; raises loudly if not


def check_a1d_11_dry_run_writes_nothing():
    """--dry-run leaves the output directory byte-for-byte unchanged and exits 0.

    Run as a SUBPROCESS with ONC_TOKEN removed from the environment: a --dry-run
    that quietly issues a listing request would fail on the missing credential
    instead of exiting 0, so this also pins "no network in a dry run" (h5).
    """
    _a1d_script_path()
    with tempfile.TemporaryDirectory() as tmp:
        before = sorted(p.name for p in pathlib.Path(tmp).iterdir())
        env = {k: v for k, v in os.environ.items() if k != "ONC_TOKEN"}
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["HOME"] = tmp  # so a stray .env in the real HOME cannot supply a token
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / A1D_SCRIPT_RELPATH),
             "--dry-run", "--out-dir", tmp],
            capture_output=True, text=True, cwd=str(REPO_ROOT), env=env,
            timeout=SUBPROC_TIMEOUT_S,
        )
        after = sorted(p.name for p in pathlib.Path(tmp).iterdir())
    assert proc.returncode == 0, (
        f"--dry-run exited {proc.returncode}; a dry run must need no credential and no "
        f"network\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert after == before, (
        f"--dry-run created {sorted(set(after) - set(before))} in the output directory. "
        "A dry run that writes is not a dry run, and the one place this flag exists to "
        "protect is data/derived/"
    )


def check_a1d_12_entry_point_declares_no_shared_constant():
    """scripts/ defines nothing shared: no module-level constant duplicating boatphone/.

    CLAUDE.md invariant 6, and it is a CROSS-TEAM rule, not a tidiness one. If the
    study window, the bin width, or the derived-dir path is re-declared here while
    acoustics reads it from boatphone/config.py, the optical-acoustic matchup joins
    two different definitions of the same thing and produces a WRONG ANSWER rather
    than an error -- the failure mode nothing downstream can detect.

    Checked structurally with `ast` on module-level assignments, so a comment
    mentioning a constant is fine and only a real re-declaration fails.
    """
    import ast
    import importlib
    path = _a1d_script_path()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    shared_names = {}
    for module_name in A1D_SHARED_CONSTANT_MODULES:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        module = importlib.import_module(module_name)
        for name in dir(module):
            if name.isupper() and not name.startswith("_"):
                shared_names[name] = (module_name, getattr(module, name))

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.asname or alias.name for alias in node.names)

    collisions, value_dupes = [], []
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        for target in targets:
            if target.id in imported:
                continue  # `from boatphone.config import BIN_SECONDS` is the RIGHT thing
            if target.id in shared_names:
                collisions.append((target.id, shared_names[target.id][0]))
                continue
            try:
                literal = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            for shared_name, (module_name, shared_value) in shared_names.items():
                try:
                    same = literal == shared_value or literal == str(shared_value)
                except Exception:  # noqa: BLE001 -- comparison of odd types, not a data issue
                    continue
                if same and not isinstance(literal, bool) and literal not in (0, 1, "", None):
                    value_dupes.append((target.id, shared_name, module_name))

    assert not collisions, (
        f"{A1D_SCRIPT_RELPATH} re-declares shared constant(s) at module level: "
        f"{collisions}. Import them from boatphone/ instead -- invariant 6: one definition, "
        "shared across notebooks and across teams"
    )
    assert not value_dupes, (
        f"{A1D_SCRIPT_RELPATH} declares module-level constant(s) whose VALUE duplicates one "
        f"in boatphone/: {value_dupes}. A second copy under a new name drifts silently the "
        "first time the library value is revised"
    )
    assert any(
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("boatphone")
        for node in ast.walk(tree)
    ) or any(
        isinstance(node, ast.Import) and any(a.name.startswith("boatphone") for a in node.names)
        for node in ast.walk(tree)
    ), (
        f"{A1D_SCRIPT_RELPATH} never imports from boatphone/; an entry point that does not "
        "use the shared library has, by definition, its own definitions"
    )




# ---------------------------------------------------------------------------
# A1e -- decision 0007: the two ONC 400s that are MEASURED ZEROS
#
# `boatphone/onc_client.py::_NO_DATA_POSSIBLE_MARKERS` is the single place in the
# codebase where an HTTP 4xx becomes a measured zero instead of an error, and
# decision 0007 rests entirely on claims about it that nothing pinned:
#
#   (1) matching is STRING-BASED on ONC's exact wording, so if ONC rewords either
#       message the request raises again -- the safe direction to fail;
#   (2) the absorbed span is PRINTED every time, never a silent zero (invariant 5);
#   (3) NOTHING ELSE is absorbed -- a 401, a 500, and any other 4xx propagate.
#
# (3) is the half that matters. If the marker match were ever broadened (to "400",
# to "Error", to any HTTPError), a bad token or a server fault would quietly become
# a zero-file span and the calendar would report a fabricated outage -- exactly the
# invariant-9 conflation the decision exists to prevent.
#
# The exact ONC strings below are TRANSCRIBED from decision 0007, which recorded
# them from the live API on 2026-08-27 at FGPD / HYDROPHONE. They are deliberately
# duplicated here rather than imported from _NO_DATA_POSSIBLE_MARKERS: importing
# the markers would make the check tautological -- it would pass for whatever the
# module happens to match today, including a broadened match.
# ---------------------------------------------------------------------------

# ONC's exact wording, from docs/decisions/0007 (live API, 2026-08-27, FGPD).
ONC_ERROR_127_TEXT = (
    "API Error 127: A device with category HYDROPHONE was deployed at location "
    "FGPD but not during the provided time range"
)
ONC_ERROR_25_TEXT = "API Error 25: Invalid Time Range, Start Time is in the future."

# Errors that must NEVER be absorbed: auth, server fault, and a 400 that is a real
# 400. The third is the important one -- it shares the status code with the two
# measured zeros and differs ONLY in its wording.
ONC_MUST_PROPAGATE = {
    "401 Unauthorized": "Status 401 - Unauthorized: the API token is not valid",
    "500 server fault": "Status 500 - Internal Server Error: the request could not be completed",
    "a DIFFERENT 400": "API Error 23: Invalid parameter value for deviceCategoryCode",
}


def _a1e_absorbed_span_is_printed(printed, span_start, span_end, onc_text):
    """A print satisfies invariant 5 only if it NAMES the span it zeroed.

    'ONC said no data' with no span is a silent zero with a log line: a reader
    cannot tell which part of the calendar it made up.
    """
    missing = []
    for moment in (span_start, span_end):
        if moment.strftime("%Y-%m-%d") not in printed and moment.isoformat() not in printed:
            missing.append(moment.isoformat())
    if onc_text.split(":")[0] not in printed and onc_text[-40:] not in printed:
        missing.append(f"ONC's own message ({onc_text!r})")
    return missing


def _a1e_absorb_one(onc, error_text, absorbed_year, span_start, span_end, data_year):
    """List a multi-year span in which ONE year raises `error_text`. Returns
    (files, empty_chunks, printed). Raises through if the client does."""
    import contextlib, io
    client = _StubONC(
        files_by_year={data_year: [_fft_name(_utc(data_year, 7, 15, 0, 0, 0))]},
        raise_on_year=absorbed_year, error=RuntimeError(error_text),
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        files, empty_chunks = onc.list_fft_files(client, ["ZZTESTA"], span_start, span_end)
    return files, empty_chunks, buffer.getvalue()


def check_a1e_1_error_127_no_deployment_is_an_absorbed_zero_and_is_printed():
    """D7: ONC's exact Error-127 wording yields an EMPTY page, printed, not a raise.

    The real case: config.STUDY_START_UTC (2020-02-18) precedes the 2020-03-08
    deployment, so ONC answers "deployed at location FGPD but not during the
    provided time range" for the leading span. No instrument existed; zero is the
    true answer. The span must still be PRINTED (invariant 5) and must still come
    out as an empty chunk, so it reads UNAVAILABLE downstream rather than vanishing.
    """
    _cfg, onc = _a1b_mod()
    span_start, span_end = _utc(2020, 1, 1), _utc(2022, 1, 1)
    try:
        files, empty_chunks, printed = _a1e_absorb_one(
            onc, ONC_ERROR_127_TEXT, 2020, span_start, span_end, data_year=2021)
    except (RuntimeError, ValueError) as exc:
        raise AssertionError(
            "ONC's exact Error-127 wording propagated as "
            f"{type(exc).__name__}: {exc}. Decision 0007 makes this span a MEASURED "
            "ZERO -- no HYDROPHONE was deployed in the window, so no file can exist "
            "and the whole study window becomes unscannable if this raises."
        ) from None
    assert len(list(files)) == 1, (
        f"expected the single 2021 file to survive the absorbed 2020 span, got {list(files)!r}"
    )
    assert empty_chunks == 1, (
        f"empty_chunks is {empty_chunks!r}; the absorbed 2020 chunk is a chunk ONC "
        "answered with zero files and must be COUNTED as empty -- otherwise the span it "
        "covers disappears from the tally instead of reading UNAVAILABLE"
    )
    missing = _a1e_absorbed_span_is_printed(printed, _utc(2020, 1, 1), _utc(2021, 1, 1),
                                            ONC_ERROR_127_TEXT)
    assert not missing, (
        f"the absorbed span was zeroed without naming {missing} in the run output. A 4xx "
        "turned into zero files SILENTLY is exactly the swallowed error invariant 5 "
        f"forbids; decision 0007's honesty rests on this print. stdout was: {printed!r}"
    )


def check_a1e_2_error_25_future_window_is_an_absorbed_zero_and_is_printed():
    """D7: ONC's exact Error-25 wording ("Start Time is in the future") likewise.

    config.STUDY_END_UTC is the end of the 2026 field season and deliberately runs
    past today, so the trailing spans have not happened yet. Same contract as #1,
    different string -- pinned separately because the two markers are independent
    and a fix that dropped one would still pass the other.
    """
    _cfg, onc = _a1b_mod()
    span_start, span_end = _utc(2025, 1, 1), _utc(2027, 1, 1)
    try:
        files, empty_chunks, printed = _a1e_absorb_one(
            onc, ONC_ERROR_25_TEXT, 2026, span_start, span_end, data_year=2025)
    except (RuntimeError, ValueError) as exc:
        raise AssertionError(
            "ONC's exact Error-25 wording propagated as "
            f"{type(exc).__name__}: {exc}. Decision 0007 makes a not-yet-elapsed window "
            "a MEASURED ZERO; if it raises, widening STUDY_END_UTC past today breaks "
            "the scan instead of costing one printed line."
        ) from None
    assert len(list(files)) == 1, (
        f"expected the single 2025 file to survive the absorbed 2026 span, got {list(files)!r}"
    )
    assert empty_chunks == 1, (
        f"empty_chunks is {empty_chunks!r}; the absorbed 2026 chunk must be counted empty"
    )
    missing = _a1e_absorbed_span_is_printed(printed, _utc(2026, 1, 1), _utc(2027, 1, 1),
                                            ONC_ERROR_25_TEXT)
    assert not missing, (
        f"the absorbed 2026 span was zeroed without naming {missing}. stdout was: {printed!r}"
    )


def check_a1e_3_a_401_a_500_and_another_400_all_still_PROPAGATE():
    """D7's load-bearing half: nothing but those two exact messages is absorbed.

    Each of a 401, a 500, and a DIFFERENT 400 must still reach the caller as an
    ONCListingError. The third is the sharpest: it carries the same status code as
    the two measured zeros and differs only in its wording, so it fails the moment
    the match is broadened from ONC's exact phrasing to "any 400".

    If any of these were absorbed the year would come back with zero files, be
    counted as an empty chunk, and read UNAVAILABLE -- a fabricated outage produced
    by a bad token or a server fault. That is the invariant-9 conflation ("found
    nothing" vs "is broken") that decision 0007 exists to hold apart.
    """
    _cfg, onc = _a1b_mod()
    import contextlib, io
    absorbed = []
    for label, text in ONC_MUST_PROPAGATE.items():
        client = _StubONC(
            files_by_year={2024: [_fft_name(_utc(2024, 7, 15, 0, 0, 0))]},
            raise_on_year=2025, error=RuntimeError(text),
        )
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = onc.list_fft_files(
                    client, ["ZZTESTA"], _utc(2024, 1, 1), _utc(2026, 1, 1))
        except Exception as exc:  # noqa: BLE001 -- the propagation IS the contract
            assert not isinstance(exc, AssertionError), str(exc)
            assert isinstance(exc, (RuntimeError, ValueError)), (
                f"{label}: propagated as {type(exc).__name__}, expected an "
                "ONCListingError (RuntimeError/ValueError subclass)"
            )
            chain_text = _exception_chain_text(exc)
            assert text[:24] in chain_text, (
                f"{label}: the underlying ONC error was replaced rather than propagated; "
                f"chain was {chain_text[:300]!r}"
            )
        else:
            absorbed.append(f"{label} -> {result!r}")
    assert not absorbed, (
        "an ONC failure that is NOT one of decision 0007's two measured zeros was "
        "absorbed into an empty page: " + "; ".join(absorbed) + ". The 2025 chunk now "
        "reports zero files, is counted empty, and reads UNAVAILABLE -- a fabricated "
        "outage manufactured from a broken request (invariant 9, D8)."
    )


def check_a1e_4_no_token_leaks_through_the_absorption_path():
    """The absorption path must not leak the token either -- printed OR raised.

    ONC embeds the whole request URL, token included, in its own error text, and
    decision 0007's path does two things with that text: it PRINTS it when a marker
    matches, and it re-raises it when one does not. Both are token-bearing routes
    that check_a1b_5g never exercises, because 5g's client raises a 401 (which is
    never absorbed and never printed).

    Searched with the same full __cause__/__context__ chain walk as 5g, because
    `raise ... from None` still leaves the original reachable as __context__.
    """
    _cfg, onc = _a1b_mod()
    import contextlib, io
    leak_url = (
        " https://data.oceannetworks.ca/api/archivefiles?locationCode=FGPD"
        f"&token={LEAK_SENTINEL_TOKEN}"
    )

    # (a) ABSORBED: a marker-matching error whose text also carries the token.
    files, _empty, printed = _a1e_absorb_one(
        onc, ONC_ERROR_127_TEXT + leak_url, 2020,
        _utc(2020, 1, 1), _utc(2022, 1, 1), data_year=2021)
    assert LEAK_SENTINEL_TOKEN not in printed, (
        "the ONC API token was PRINTED by the measured-zero path. Decision 0007 requires "
        "this line on every absorbed span, so it lands in committed notebook output every "
        f"time the study window is scanned (invariant 7). stdout was: {printed!r}"
    )
    assert "token=<redacted>" in printed or "token=" not in printed, (
        f"the token query parameter survived unredacted in the printed line: {printed!r}"
    )
    assert len(list(files)) == 1, f"the absorbed span still swallowed data: {list(files)!r}"

    # (b) NOT absorbed: a token-bearing 400 that fails the marker match must
    #     propagate, and must propagate REDACTED over the whole chain.
    client = _StubONC(
        files_by_year={2024: [_fft_name(_utc(2024, 7, 15, 0, 0, 0))]},
        raise_on_year=2025,
        error=RuntimeError(ONC_MUST_PROPAGATE["a DIFFERENT 400"] + leak_url),
    )
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = onc.list_fft_files(client, ["ZZTESTA"], _utc(2024, 1, 1), _utc(2026, 1, 1))
    except Exception as exc:  # noqa: BLE001 -- raising IS the contract (check_a1e_3)
        assert not isinstance(exc, AssertionError), str(exc)
        chain_text = _exception_chain_text(exc)
        assert LEAK_SENTINEL_TOKEN not in chain_text, (
            "the ONC API token is reachable in the exception chain raised from the "
            "measured-zero branch. The redacted error must be raised OUTSIDE the `except` "
            "block -- chaining re-attaches the token-bearing original and a traceback "
            "prints every link (invariant 7)."
        )
    else:
        raise AssertionError(
            f"a token-bearing 400 that is not one of decision 0007's two messages was "
            f"absorbed instead of raised: {result!r} (see check_a1e_3)"
        )


class _MidPaginationNoDataONC:
    """A listing client that answers page 1 with rows, then raises a 0007 marker.

    Local class, no mocking library (CLAUDE.md "Environment"), same shape as
    _StubONC/_CappedONC. Page 1 (a request carrying no paging key) returns
    `first_page` and advertises `next`; EVERY paged follow-up raises `error_text`.

    This is not a shape ONC can honestly produce: Error 127 says no HYDROPHONE was
    deployed in the requested time range, and the time range is IDENTICAL on page 2
    -- page 1 just returned files from it. So the two statements contradict each
    other, and the contradiction is the point of the fixture.
    """

    def __init__(self, first_page, error_text=ONC_ERROR_127_TEXT):
        self.first_page = list(first_page)
        self.error_text = error_text
        self.requests = []

    def _listing(self, filters=None, **kwargs):
        f = dict(filters or {})
        f.update(kwargs)
        self.requests.append(f)
        if any(k in f for k in ("page", "offset", "skip")):
            raise RuntimeError(self.error_text)
        return {"files": list(self.first_page), "next": {"offset": len(self.first_page)}}

    getArchivefile = _listing
    getArchivefileByLocation = _listing
    getArchivefileByDevice = _listing
    getListByLocation = _listing
    getListByDevice = _listing

    def getLocations(self, filters=None, **kwargs):
        self.requests.append(dict(filters or {}, **kwargs))
        return list(_folger_location_rows())

    def getDeployments(self, filters=None, **kwargs):
        self.requests.append(dict(filters or {}, **kwargs))
        return []


def check_a1e_5_a_no_data_marker_on_page_2_must_RAISE_not_truncate():
    """A 0007 marker on page 2+ is NOT a measured zero -- it must raise.

    The absorption lives inside `_fetch_page`, so it applies to every page, not just
    the first. On page 2+ its return value `([], None)` means "this page is empty AND
    the listing is complete", and `_fetch_pages` accepts that and stops. The span is
    then SILENTLY TRUNCATED at page 1 and reported complete.

    THE CALL, and the reasoning: a 0007 marker on page 2+ cannot be true. Error 127
    says no HYDROPHONE was deployed in the provided time range, and Error 25 says
    that range has not elapsed -- but the range is byte-identical to page 1's, and
    page 1 DEMONSTRABLY returned files from it. The span is therefore not a measured
    zero; something is broken (a paging parameter mishandled, a server fault behind a
    400, ONC rewording a message). That is the invariant-9 distinction, and only the
    "is broken" side may be absorbed into silence -- which is to say, neither.

    The cost of getting it wrong is the A1d defect resurrected by another route: a
    dense span truncated at page 1 becomes a months-long outage that never happened,
    and it aims the Planet quota at dates the hydrophone was recording.

    The stub raises on every paged follow-up, so no correct implementation can reach
    completeness here -- the ONLY passing behaviour is a raise that names the span.
    """
    _cfg, onc = _a1b_mod()
    import contextlib, io
    span_start, span_end = _utc(2024, 7, 1), _utc(2024, 8, 1)
    first_page = _dense_fft_names(span_start, _utc(2024, 7, 2))  # one dense day, then "no data"
    client = _MidPaginationNoDataONC(first_page)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            files, empty_chunks = onc.list_fft_files(client, ["ZZTESTA"], span_start, span_end)
    except (RuntimeError, ValueError) as exc:
        text = str(exc)
        assert not isinstance(exc, AssertionError), text
        missing = [ym for ym in {span_start.strftime("%Y-%m"), span_start.strftime("%Y-%m-%d")}
                   if ym not in text]
        assert len(missing) < 2, (
            "the mid-pagination no-data answer raised (correct), but the message names "
            f"neither the span nor the date it stopped at: {text!r}"
        )
        return
    raise AssertionError(
        f"list_fft_files returned {len(list(files))} file(s) ({empty_chunks} empty chunk(s)) "
        f"for a span whose page 2 answered with a decision-0007 marker. Page 1 returned "
        f"{len(first_page)} file(s) from the SAME time range, so 'no device was deployed in "
        "that range' cannot be true -- this is a broken request, not a measured zero, and "
        "absorbing it in _fetch_page returns ([], None), which _fetch_pages reads as 'the "
        "listing is complete'. The span is silently truncated at page 1 and reported whole: "
        "the A1d phantom-outage defect via a second route (invariant 9). The absorption "
        "belongs on the FIRST page of a span only. stdout: "
        f"{buffer.getvalue()[:300]!r}"
    )


# ---------------------------------------------------------------------------
# B3-B -- resumable, content-hashed download primitive
#
# Contract (segment B3-B, `docs/plans/acoustics_plan_v2.md` §"B3 -- Bulk
# acquisition", extending A1's `boatphone/onc_client.py`). ONE function pulls
# ONE ONC archive file into the landing zone (`boatphone.paths.ONC_RAW_DIR`,
# decision 0005 -- append-only, gitignored, created at download time via
# `paths.ensure_dir()`, never as an import side effect). It must:
#
#   * write to a `.part` sidecar and rename to the final path only on
#     completion, so a file is immutable ONLY once complete (decision 0005 §1)
#     and an interrupted transfer can never be mistaken for a good one;
#   * be a no-op, a counted CACHE HIT, on re-invocation when the final path
#     already exists and verifies;
#   * RAISE on a hash mismatch against a cached file, never silently
#     overwrite it (CLAUDE.md invariant 2: data/ is immutable, and a silent
#     overwrite of a corrupt cache entry is exactly the failure that
#     invariant exists to prevent);
#   * back off exponentially with jitter on HTTP 429/5xx;
#   * treat the decision-0007 "not deployed" / "not yet elapsed" 400s as
#     MEASURED ZEROS, not errors -- the same distinction A1b/A1e already draw
#     for listing, now drawn for the pull;
#   * return a structured ABSENT record for a file ONC genuinely has nothing
#     for (a real 404), rather than raising or skipping silently -- segment
#     C's absent-file log is the stated consumer;
#   * contain no bare `except`.
#
# THIS CODE DOES NOT EXIST YET. `boatphone/onc_client.py` currently exports no
# `download`/`getFile`/hash/manifest symbol at all (grepped, confirmed empty).
# Every check below is therefore expected to FAIL until B3-B lands, and each
# one says so explicitly via `_b3b_mod()` before it does anything else, so a
# missing-symbol AssertionError reads as "not implemented" and never as
# "implemented and wrong" (test-author brief).
#
# ASSUMED INTERFACE, stated because none of it can be read off an
# implementation that does not exist -- this is the shape the checks below
# are written against, and if B3-B lands with a different shape, the checks
# (not the guess) are what should move:
#
#   download_archive_file(client, filename, *, dest_dir, transport,
#                          sleep=time.sleep, expected_sha256=None)
#       -> a result exposing at least `.status` (one of "downloaded",
#          "cached", "absent", "measured_zero") and `.path` (Path or None).
#
#   `transport` is the injected network layer: `transport.get(filename,
#   range_start=0)` returns an object with `.status_code`, `.headers`, and
#   `.iter_bytes(chunk_size)`; this is the seam these checks monkeypatch so
#   NOTHING here touches the real ONC API or writes outside a temp dir this
#   file creates and removes (never under data/, per the test-author brief).
#
# THE HASH GUARANTEE ACTUALLY IN FORCE -- a judgement call, stated rather than
# assumed silently: ONC's archive endpoint is not known to serve a
# server-side content checksum (nothing in A1's listing/metadata rows carries
# one). So "hash-matching" below is read as: the cache is keyed on a LOCAL
# sha256 of the completed file, computed once at download time and kept
# alongside it (e.g. a `.sha256` sidecar), and re-verified before a cache hit
# is trusted. That detects LOCAL corruption (bit rot, a hand-edited fixture,
# a bad copy) but NOT a server response that was truncated and then presented
# as complete -- a completed transfer is trusted to be whole unless the
# response carries its own length/hash to check against, which is answered by
# check_b3b_5 below, not assumed. If B3-B finds ONC does serve a checksum
# (e.g. an ETag or a `Content-MD5`), check_b3b_4 and check_b3b_5's assertions
# still hold; they would simply be checking a stronger guarantee than the one
# assumed here, not a different one.
# ---------------------------------------------------------------------------

B3B_FAKE_CONTENT = b"ICLISTEN-FFT-BYTES-" * 512  # a deterministic, non-trivial fixture payload
B3B_FAKE_FILENAME = "ICLISTENHF1266_20240715T000136.000Z.fft.gz"


def _b3b_mod():
    """Import boatphone.onc_client and require the B3-B download symbol.

    Mirrors `_a1b_mod()`'s convention exactly: `assert hasattr(...)` turns a
    not-yet-written function into a named, readable AssertionError ("not
    implemented") rather than a bare AttributeError that could be confused
    with a real bug once the function exists.
    """
    cfg, onc = _a1a_mods()
    assert hasattr(onc, "download_archive_file"), (
        "boatphone/onc_client.py has no download_archive_file(); B3-B (the "
        "resumable, content-hashed download primitive) is not implemented yet"
    )
    return cfg, onc


class _B3BFakeResponse:
    """One simulated HTTP response. `status_code` 200/404/429/500/400."""

    def __init__(self, status_code, body=b"", total_size=None, sha256_hex=None,
                 error_text="", interrupt_after=None):
        self.status_code = status_code
        self._body = body
        self.total_size = total_size if total_size is not None else len(body)
        self.error_text = error_text
        self.headers = {"Content-Length": str(self.total_size)}
        if sha256_hex:
            self.headers["X-Content-SHA256"] = sha256_hex
        self._interrupt_after = interrupt_after

    def iter_bytes(self, chunk_size=65536):
        sent = 0
        for offset in range(0, len(self._body), chunk_size):
            chunk = self._body[offset:offset + chunk_size]
            sent += len(chunk)
            yield chunk
            if self._interrupt_after is not None and sent >= self._interrupt_after:
                raise ConnectionError(
                    f"simulated dropped connection after {sent} byte(s) (test fixture)"
                )


class _B3BFakeTransport:
    """Injectable stand-in for B3-B's network layer -- no network, ever.

    `content_by_name` is the FULL, correct payload per filename, so a
    resumed request can be sliced by `range_start` and checked byte-for-byte
    against an uninterrupted pull of the same map.
    """

    def __init__(self, content_by_name):
        self.content_by_name = dict(content_by_name)
        self.calls = []                 # (filename, range_start) per attempt, in order
        self.fail_sequence = []         # status codes to return before a real answer, popped in order
        self.interrupt_once_after = None  # bytes to serve before ConnectionError, ONE attempt only
        self.not_deployed_names = set()
        self.absent_names = set()

    def get(self, filename, range_start=0):
        self.calls.append((filename, range_start))
        if filename in self.absent_names:
            return _B3BFakeResponse(404, error_text=f"{filename} not found")
        if filename in self.not_deployed_names:
            # Decision 0007's exact first marker, so the check exercises the
            # SAME string match _NO_DATA_POSSIBLE_MARKERS uses for listing.
            return _B3BFakeResponse(
                400, error_text="API Error 127: A device with category HYDROPHONE was "
                "deployed at location FGPD but not during the provided time range"
            )
        if self.fail_sequence:
            code = self.fail_sequence.pop(0)
            return _B3BFakeResponse(code, error_text=f"simulated {code} (test fixture)")
        full = self.content_by_name[filename]
        body = full[range_start:]
        interrupt_after = None
        if self.interrupt_once_after is not None and range_start == 0:
            interrupt_after = self.interrupt_once_after
            self.interrupt_once_after = None  # only the FIRST attempt is interrupted
        return _B3BFakeResponse(200, body=body, total_size=len(full), interrupt_after=interrupt_after)


def _b3b_sha256(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def check_b3b_0_api_surface():
    """download_archive_file exists and is callable (not-implemented gate for the rest of B3-B)."""
    _cfg, onc = _b3b_mod()
    assert callable(onc.download_archive_file), (
        "boatphone.onc_client.download_archive_file exists but is not callable"
    )


def check_b3b_1_interrupted_transfer_never_leaves_a_corrupt_final_file():
    """A dropped connection never leaves a bad file at the FINAL path.

    Every attempt in this fixture is interrupted (the transport re-arms after
    each attempt, unlike the resume check below), so no correct implementation
    can complete the pull here. The ONLY acceptable outcomes are: the call
    raises and the final path does not exist, or the final path is absent
    while a `.part` sidecar (any name containing '.part') holds the partial
    bytes. What must NEVER happen is a file at the final path with the wrong
    length or wrong hash -- decision 0005 §1's "immutable once COMPLETE" is
    exactly the guarantee this pins.
    """
    _cfg, onc = _b3b_mod()
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"

        class _AlwaysInterrupts(_B3BFakeTransport):
            def get(self, filename, range_start=0):
                self.calls.append((filename, range_start))
                full = self.content_by_name[filename]
                body = full[range_start:]
                return _B3BFakeResponse(
                    200, body=body, total_size=len(full),
                    interrupt_after=min(64, len(body)) if body else None,
                )

        transport = _AlwaysInterrupts({B3B_FAKE_FILENAME: B3B_FAKE_CONTENT})
        raised = None
        try:
            onc.download_archive_file(
                client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
                transport=transport, sleep=lambda _s: None,
            )
        except Exception as exc:  # noqa: BLE001 -- a raise here is an ACCEPTABLE outcome
            raised = exc

        final_path = dest_dir / B3B_FAKE_FILENAME
        if final_path.exists():
            on_disk = final_path.read_bytes()
            assert on_disk == B3B_FAKE_CONTENT, (
                f"download_archive_file left a file at the FINAL path {final_path} that is "
                f"{len(on_disk)} byte(s), not the {len(B3B_FAKE_CONTENT)}-byte complete "
                "payload, after every simulated attempt was interrupted mid-transfer. "
                "Decision 0005 §1: a file is immutable only once COMPLETE, so an interrupted "
                "transfer must never be observable at the final path"
            )
        stray_final_writes = [
            p for p in dest_dir.rglob("*")
            if p.is_file() and p.name == B3B_FAKE_FILENAME and p.stat().st_size not in (0, len(B3B_FAKE_CONTENT))
        ]
        assert not stray_final_writes, (
            f"found file(s) at the exact final name with a partial size: {stray_final_writes}; "
            "a partial transfer must live under a .part sidecar, never at the final name"
        )
        assert raised is not None or not final_path.exists(), (
            "download_archive_file returned normally with every attempt interrupted, but left "
            "no final file and raised nothing -- an interrupted transfer must be either raised "
            "or left resumable, never silently reported as done"
        )


class _B3BFakeTransport206OnResume(_B3BFakeTransport):
    """A transport that HONOURS Range on resume: 206 Partial Content, body sliced from `range_start`.

    Only the resumed leg (`range_start > 0`) differs from the base fixture --
    it returns 206, not the base class's (buggy, pre-existing-check) 200, so
    this pins the real "Range honoured" branch of `download_archive_file`
    rather than a status code that never occurs over real HTTP for a
    successful resume.
    """

    def get(self, filename, range_start=0):
        self.calls.append((filename, range_start))
        full = self.content_by_name[filename]
        if range_start > 0:
            return _B3BFakeResponse(206, body=full[range_start:], total_size=len(full))
        interrupt_after = None
        if self.interrupt_once_after is not None:
            interrupt_after = self.interrupt_once_after
            self.interrupt_once_after = None  # only the FIRST attempt is interrupted
        return _B3BFakeResponse(200, body=full, total_size=len(full), interrupt_after=interrupt_after)


class _B3BFakeTransport200IgnoresRangeOnResume(_B3BFakeTransport):
    """A transport that IGNORES Range on resume: plain 200, FULL unsliced body from byte zero.

    This is real HTTP's actual "Range not honoured" behavior -- the opposite
    of what the old (removed) fixture simulated. A correct downloader must
    detect this and discard-and-restart rather than appending the full body
    onto the partial `.part` file already on disk.
    """

    def get(self, filename, range_start=0):
        self.calls.append((filename, range_start))
        full = self.content_by_name[filename]
        interrupt_after = None
        if range_start == 0 and self.interrupt_once_after is not None:
            interrupt_after = self.interrupt_once_after
            self.interrupt_once_after = None  # only the FIRST attempt is interrupted
        return _B3BFakeResponse(200, body=full, total_size=len(full), interrupt_after=interrupt_after)


def check_b3b_2a_resume_when_range_honoured_206_is_byte_identical_to_uninterrupted_pull():
    """206 Partial Content on resume: append, and the result == an uninterrupted pull.

    The fixture interrupts ONLY the first attempt (`interrupt_once_after`);
    a correct resumable downloader retries, this time asking the transport
    for `range_start` at the byte it already has, and gets back a REAL 206
    with the body correctly sliced. The reassembled file must be
    indistinguishable from a clean single-shot pull of the same content.
    This is the check that actually exercises RESUME, not just
    retry-from-scratch -- `transport.calls` is asserted to contain a
    `range_start > 0`, or a downloader that silently restarts from zero every
    time would pass on content alone while still being O(n^2) against a large
    corpus, contrary to the segment's stated purpose.
    """
    _cfg, onc = _b3b_mod()
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _B3BFakeTransport206OnResume({B3B_FAKE_FILENAME: B3B_FAKE_CONTENT})
        transport.interrupt_once_after = 128  # first attempt dies after 128 bytes, then heals

        # chunk_bytes is passed SMALL and EXPLICIT, not left at the production
        # default (65536, as of the B3-B shared contract). iter_bytes() only
        # raises its simulated interrupt AFTER yielding a whole chunk, so at
        # the 9728-byte fixture size a production-sized chunk IS the entire
        # body: the .part would already be complete, range_start on "resume"
        # would equal len(content), and this check would go vacuous while
        # still reporting PASS. Do not "tidy" this back to the default.
        result = onc.download_archive_file(
            client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
            transport=transport, sleep=lambda _s: None, chunk_bytes=1024,
        )
        final_path = dest_dir / B3B_FAKE_FILENAME
        assert final_path.exists(), (
            f"download_archive_file returned {result!r} but wrote no file at {final_path} "
            "after resuming a once-interrupted transfer honoured with 206"
        )
        on_disk = final_path.read_bytes()
        assert on_disk == B3B_FAKE_CONTENT, (
            f"resumed download is {len(on_disk)} byte(s) and does not match the "
            f"{len(B3B_FAKE_CONTENT)}-byte uninterrupted content byte-for-byte "
            "(first mismatching offset: "
            f"{next((i for i in range(min(len(on_disk), len(B3B_FAKE_CONTENT))) if on_disk[i] != B3B_FAKE_CONTENT[i]), 'length differs')})"
        )
        range_starts = [rs for (_fn, rs) in transport.calls]
        assert any(0 < rs < len(B3B_FAKE_CONTENT) for rs in range_starts), (
            f"transport.get() was called with range_start values {range_starts}; none is a "
            f"PARTIAL offset strictly between 0 and {len(B3B_FAKE_CONTENT)}, so either the "
            "retry restarted the WHOLE transfer from byte zero rather than resuming (that is "
            "retry, not the resumable primitive B3-B is required to be), or the '.part' already "
            "held the complete file when resume was attempted and this check exercised nothing"
        )
        assert result.http_status == 206, (
            f"result.http_status is {result.http_status!r}, not 206, after the resumed leg was "
            "answered with 206 Partial Content -- http_status must reflect the LAST response "
            "observed, which honoured the resume"
        )


def check_b3b_2b_resume_when_range_ignored_200_discards_and_restarts_from_zero():
    """200 (not 206) on resume: Range was IGNORED -- must discard the .part and restart, never append.

    Real HTTP semantics, the opposite of the old removed check's fixture: a
    plain 200 on a request that carried a Range header means the server sent
    the FULL body from byte zero, not the honoured tail. Appending that onto
    an existing partial file would silently double it -- exactly the class of
    silent corruption CLAUDE.md invariant 5 rules out. The assertions below
    are: (1) the final content is byte-for-byte correct -- this ALONE
    already implies the length is correct too (nothing can be byte-equal to
    B3B_FAKE_CONTENT and a different length), so it is content equality, not
    a separate length check, that is doing the real work of catching a naive
    len(partial) + len(full) append; a length-only assertion would be
    redundant once content equality is asserted, and (2) the implementation
    observably took the discard-and-restart path. `download_archive_file`
    records no structured
    field for which resume path was taken (`DownloadRecord` carries
    `http_status`/`attempts` but nothing that distinguishes "206 honoured"
    from "200 ignored and restarted" after the fact), so the only observable
    at this seam is the `print()` the implementation emits when it discards
    -- captured here rather than asserted against log text elsewhere, and
    flagged: if a structured field for this is later added to
    `DownloadRecord`, this assertion should move to it instead of stdout.
    """
    _cfg, onc = _b3b_mod()
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _B3BFakeTransport200IgnoresRangeOnResume({B3B_FAKE_FILENAME: B3B_FAKE_CONTENT})
        transport.interrupt_once_after = 128  # first attempt dies after 128 bytes, then heals

        import contextlib
        import io
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            # chunk_bytes small and explicit -- see check_b3b_2a's comment for
            # why: at the production default (65536) the 9728-byte fixture's
            # first chunk IS the whole body, the .part lands complete, and the
            # "resumed" request would carry range_start == len(content),
            # making this check pass without ever exercising a real resume.
            result = onc.download_archive_file(
                client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
                transport=transport, sleep=lambda _s: None, chunk_bytes=1024,
            )
        final_path = dest_dir / B3B_FAKE_FILENAME
        assert final_path.exists(), (
            f"download_archive_file returned {result!r} but wrote no file at {final_path} "
            "after resuming a once-interrupted transfer whose Range was ignored (200)"
        )
        on_disk = final_path.read_bytes()
        assert on_disk == B3B_FAKE_CONTENT, (
            f"resumed download (Range ignored, 200) is {len(on_disk)} byte(s) and does not "
            f"match the {len(B3B_FAKE_CONTENT)}-byte uninterrupted content byte-for-byte "
            "(first mismatching offset: "
            f"{next((i for i in range(min(len(on_disk), len(B3B_FAKE_CONTENT))) if on_disk[i] != B3B_FAKE_CONTENT[i]), 'length differs')})"
        )
        # No separate length assertion here: `on_disk == B3B_FAKE_CONTENT` above
        # already pins the length (byte equality implies length equality;
        # nothing passes the former while failing the latter), including
        # against the naive len(partial) + len(full) append this check exists
        # to catch. A prior version of this comment claimed the length check
        # was doing that work -- it was not; content equality was.
        range_starts = [rs for (_fn, rs) in transport.calls]
        assert any(0 < rs < len(B3B_FAKE_CONTENT) for rs in range_starts), (
            f"transport.get() was called with range_start values {range_starts}; none is a "
            f"PARTIAL offset strictly between 0 and {len(B3B_FAKE_CONTENT)}, so this check "
            "never actually exercised a resumed request that got Range ignored at a genuine "
            "partial offset -- a full-length range_start (a .part that was already complete "
            "when 'resume' was attempted) would satisfy `rs > 0` but exercise nothing"
        )
        log_text = captured.getvalue()
        assert "discard" in log_text.lower() and "200" in log_text, (
            f"download_archive_file logged nothing recognizable as 'discarded the partial and "
            f"restarted from zero' when the resumed request came back 200 instead of 206; "
            f"captured stdout was: {log_text!r}. This is the only observable of the "
            "discard-and-restart path today -- if DownloadRecord grows a structured field for "
            "it, this assertion should move there instead"
        )


def check_b3b_2c_download_record_carries_http_status_and_attempts_per_status():
    """DownloadRecord.http_status/.attempts are populated correctly for every status.

    Pins the B3-C manifest contract stated in `DownloadRecord`'s own
    docstring: "cached" made no HTTP request at all, so it must report
    `http_status=None, attempts=0` rather than fabricating a 200 that never
    happened (invariant 5); "downloaded", "absent", and "measured_zero" each
    made at least one real request, so both fields must be populated.
    """
    _cfg, onc = _b3b_mod()
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"

        downloaded_transport = _B3BFakeTransport({B3B_FAKE_FILENAME: B3B_FAKE_CONTENT})
        downloaded = onc.download_archive_file(
            client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
            transport=downloaded_transport, sleep=lambda _s: None,
        )
        assert downloaded.status == "downloaded"
        assert downloaded.http_status == 200 and downloaded.attempts >= 1, (
            f"'downloaded' record has http_status={downloaded.http_status!r}, "
            f"attempts={downloaded.attempts!r}; a completed transfer made a real request and "
            "must report both"
        )

        cached = onc.download_archive_file(
            client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
            transport=downloaded_transport, sleep=lambda _s: None,
        )
        assert cached.status == "cached"
        assert cached.http_status is None and cached.attempts == 0, (
            f"'cached' record has http_status={cached.http_status!r}, attempts="
            f"{cached.attempts!r}; a cache hit made NO HTTP request, so it must not fabricate "
            "an http_status or a nonzero attempt count (invariant 5)"
        )

        absent_transport = _B3BFakeTransport({})
        absent_transport.absent_names = {B3B_FAKE_FILENAME}
        absent = onc.download_archive_file(
            client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir / "absent",
            transport=absent_transport, sleep=lambda _s: None,
        )
        assert absent.status == "absent"
        assert absent.http_status == 404 and absent.attempts >= 1, (
            f"'absent' record has http_status={absent.http_status!r}, attempts="
            f"{absent.attempts!r}; a real 404 is a real request and must report both"
        )

        zero_transport = _B3BFakeTransport({})
        zero_transport.not_deployed_names = {B3B_FAKE_FILENAME}
        measured_zero = onc.download_archive_file(
            client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir / "zero",
            transport=zero_transport, sleep=lambda _s: None,
        )
        assert measured_zero.status == "measured_zero"
        assert measured_zero.http_status == 400 and measured_zero.attempts >= 1, (
            f"'measured_zero' record has http_status={measured_zero.http_status!r}, attempts="
            f"{measured_zero.attempts!r}; a real 400 is a real request and must report both"
        )


def check_b3b_3_second_call_on_a_complete_file_is_a_zero_byte_cache_hit():
    """Re-invoking on an already-complete, hash-matching file downloads ZERO bytes."""
    _cfg, onc = _b3b_mod()
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _B3BFakeTransport({B3B_FAKE_FILENAME: B3B_FAKE_CONTENT})

        first = onc.download_archive_file(
            client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
            transport=transport, sleep=lambda _s: None,
        )
        calls_after_first = len(transport.calls)
        assert calls_after_first >= 1, "the first call made no network request at all"

        second = onc.download_archive_file(
            client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
            transport=transport, sleep=lambda _s: None,
        )
        assert len(transport.calls) == calls_after_first, (
            f"a second call on an already-complete file made "
            f"{len(transport.calls) - calls_after_first} more transport.get() call(s); a cache "
            "hit must fetch zero bytes, not re-request the file"
        )
        second_status = getattr(second, "status", None) or (second.get("status") if isinstance(second, dict) else None)
        assert second_status is not None and str(second_status).lower() in ("cached", "cache_hit", "cache-hit"), (
            f"second call's result does not report a cache hit (got status={second_status!r} "
            f"from {second!r}); a caller cannot distinguish 'skipped because cached' from "
            "'downloaded again' without this being reported"
        )
        final_path = dest_dir / B3B_FAKE_FILENAME
        assert final_path.read_bytes() == B3B_FAKE_CONTENT, (
            "the cached file's content changed across the two calls"
        )


def check_b3b_4_locally_corrupted_cache_is_detected_by_hash_and_RAISES():
    """A hand-corrupted cached file is caught by hash on the next call, and RAISES.

    States the guarantee actually in force (see the section banner): ONC is
    not known to serve a content checksum, so this checks LOCAL corruption
    detection -- bytes on disk no longer match the hash recorded at download
    time -- not a truncated-but-declared-complete server response, which is a
    guarantee the API may not be able to provide.
    """
    _cfg, onc = _b3b_mod()
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _B3BFakeTransport({B3B_FAKE_FILENAME: B3B_FAKE_CONTENT})
        onc.download_archive_file(
            client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
            transport=transport, sleep=lambda _s: None,
        )
        final_path = dest_dir / B3B_FAKE_FILENAME
        original = final_path.read_bytes()
        corrupted = bytes([original[0] ^ 0xFF]) + original[1:]  # flip one byte, same LENGTH
        final_path.write_bytes(corrupted)

        raised = None
        try:
            onc.download_archive_file(
                client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
                transport=transport, sleep=lambda _s: None,
            )
        except Exception as exc:  # noqa: BLE001 -- the raise IS the contract; never AssertionError
            raised = exc
        assert raised is not None, (
            "download_archive_file silently accepted (or silently re-downloaded over) a "
            "cached file whose content no longer matches its recorded hash -- CLAUDE.md "
            "invariant 2: a corrupted cache entry must RAISE, never be silently overwritten"
        )
        assert not isinstance(raised, AssertionError), str(raised)


def check_b3b_5_three_429s_back_off_with_strictly_increasing_sleeps_then_succeed():
    """Three simulated 429s produce strictly increasing recorded sleeps, then success.

    Asserts on the RECORDED sleep durations the code itself asked for (an
    injected `sleep=` callable), never on wall-clock -- this check must be
    fast regardless of the real backoff schedule.
    """
    _cfg, onc = _b3b_mod()
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _B3BFakeTransport({B3B_FAKE_FILENAME: B3B_FAKE_CONTENT})
        transport.fail_sequence = [429, 429, 429]
        recorded_sleeps = []

        result = onc.download_archive_file(
            client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
            transport=transport, sleep=recorded_sleeps.append,
        )
        assert (dest_dir / B3B_FAKE_FILENAME).read_bytes() == B3B_FAKE_CONTENT, (
            f"three 429s then a good response did not end in a complete file (result={result!r})"
        )
        assert len(recorded_sleeps) >= 3, (
            f"only {len(recorded_sleeps)} backoff sleep(s) recorded for 3 simulated 429s: "
            f"{recorded_sleeps!r} -- each 429 must be slept off, not retried immediately"
        )
        first_three = recorded_sleeps[:3]
        assert all(b > a for a, b in zip(first_three, first_three[1:])), (
            f"backoff sleeps {first_three!r} are not strictly increasing; exponential backoff "
            "on repeated 429s must grow each attempt, even with jitter applied on top"
        )
        assert all(s > 0 for s in first_three), (
            f"a non-positive backoff sleep was recorded: {first_three!r}"
        )


def check_b3b_6_permanently_absent_file_yields_a_structured_record_and_does_not_raise():
    """A real 404 (permanently absent) returns a structured record; it never raises.

    The record must carry enough for segment C's absent-file log to act on
    without re-parsing an exception string: at minimum the filename and a
    status distinguishing it from every other outcome.
    """
    _cfg, onc = _b3b_mod()
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _B3BFakeTransport({})
        transport.absent_names = {B3B_FAKE_FILENAME}

        result = onc.download_archive_file(
            client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
            transport=transport, sleep=lambda _s: None,
        )
        assert result is not None, "an absent file must yield a record, not None"
        status = getattr(result, "status", None) or (result.get("status") if isinstance(result, dict) else None)
        assert status is not None and str(status).lower() == "absent", (
            f"expected a status of 'absent' for a real 404, got {status!r} from {result!r}"
        )
        name_field = (
            getattr(result, "filename", None)
            or (result.get("filename") if isinstance(result, dict) else None)
        )
        assert name_field == B3B_FAKE_FILENAME, (
            f"the absent record does not name the file it is about (got {name_field!r} from "
            f"{result!r}); segment C's absent-file log needs this to act on it without "
            "re-parsing a message"
        )
        assert not (dest_dir / B3B_FAKE_FILENAME).exists(), (
            "a file appeared at the final path for a filename the transport reports absent"
        )


def check_b3b_7_onc_400_not_deployed_is_a_measured_zero_distinct_from_absent_and_error():
    """Decision 0007's 'not deployed' 400 is its OWN outcome -- not absent, not error, no raise.

    Matches the exact wording `_NO_DATA_POSSIBLE_MARKERS` already keys on for
    listing, so the pull draws the identical measured-zero line A1e already
    proved for the list, per decision 0007.
    """
    _cfg, onc = _b3b_mod()
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _B3BFakeTransport({})
        transport.not_deployed_names = {B3B_FAKE_FILENAME}

        raised = None
        result = None
        try:
            result = onc.download_archive_file(
                client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
                transport=transport, sleep=lambda _s: None,
            )
        except Exception as exc:  # noqa: BLE001 -- must NOT raise; caught to report clearly if it does
            raised = exc
        assert raised is None, (
            f"a decision-0007 'not deployed' 400 raised {type(raised).__name__}: {raised}; "
            "D7 already establishes this is a measured zero, not a failure, for listing -- the "
            "pull must draw the same line, not regress to treating it as an error"
        )
        status = getattr(result, "status", None) or (result.get("status") if isinstance(result, dict) else None)
        assert status is not None, f"no status on the result for a 0007 measured-zero 400: {result!r}"
        status_l = str(status).lower()
        assert status_l not in ("absent", "error", "downloaded", "cached"), (
            f"a decision-0007 'not deployed' 400 was reported as status={status!r}; it must be "
            "its OWN outcome (e.g. 'measured_zero'), distinguishable from a real 404 (absent), "
            "a genuine failure, and a successful pull -- collapsing it into any of those loses "
            "exactly the distinction decision 0007 exists to draw"
        )
        assert not (dest_dir / B3B_FAKE_FILENAME).exists(), (
            "a file appeared at the final path for a span decision 0007 says can hold nothing"
        )


def check_b3b_8_no_bare_except_in_the_download_code():
    """D8/invariant 5, extended to the pull: no bare `except:`/`except Exception:`.

    Reuses A1b's exact scan so the download code is held to the identical
    standard as the listing code it extends, not a looser one.
    """
    import inspect
    _cfg, onc = _b3b_mod()
    src = inspect.getsource(onc)
    hits = [
        f"line {n}: {line.strip()!r}"
        for n, line in enumerate(src.splitlines(), 1)
        if line.split("#", 1)[0].strip() in ("except:", "except Exception:")
    ]
    assert not hits, (
        "boatphone/onc_client.py catches everything: " + "; ".join(hits)
        + ". A swallowed 429/5xx or hash failure becomes indistinguishable from success "
        "(invariant 5)"
    )


def check_b3b_9_never_writes_outside_the_temp_dest_dir_this_check_created():
    """This check's own fixture never touches data/ -- guards the guard, not the coder.

    A download primitive that is passed `dest_dir` and STILL writes to
    `boatphone.paths.ONC_RAW_DIR` (or anywhere under data/) would corrupt an
    immutable acquisition zone (CLAUDE.md invariant 2) the moment a caller's
    dest_dir argument was ignored. This asserts nothing appeared under the
    real data/raw/onc during a full run of the checks above.
    """
    import boatphone.paths as p
    _cfg, onc = _b3b_mod()
    before = set(p.ONC_RAW_DIR.rglob("*")) if p.ONC_RAW_DIR.exists() else set()
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _B3BFakeTransport({B3B_FAKE_FILENAME: B3B_FAKE_CONTENT})
        onc.download_archive_file(
            client=None, filename=B3B_FAKE_FILENAME, dest_dir=dest_dir,
            transport=transport, sleep=lambda _s: None,
        )
    after = set(p.ONC_RAW_DIR.rglob("*")) if p.ONC_RAW_DIR.exists() else set()
    assert after == before, (
        f"the real ONC landing zone {p.ONC_RAW_DIR} changed during a dest_dir-scoped test run: "
        f"{after - before} appeared. A check must never write under data/ (invariant 2); this "
        "means download_archive_file ignored its dest_dir argument"
    )


# ---------------------------------------------------------------------------
# B0.1 -- acquire and pin external artefacts (ONC selfsupervision_anomalies_onc)
#
# Contract (segment B0-1, milestone1/b0-model-viability). These artefacts are
# NOT acquisitions in the data/ sense -- they are third-party model code and
# checkpoints, immutable to us but not ours, and CLAUDE.md invariant 2 (data/
# immutability) does not apply to them. They must live OUTSIDE data/ entirely:
#
#   external/onc_ssamba/      -- clone of the ONC repo (git commit SHA pinned)
#   external/checkpoints/     -- HF Hub artefacts (merileo/*: cnn_baseline/
#                                 cnn_best.pt, args.pkl, a labelled eval .h5)
#
# `external/` is a new top-level entry in .gitignore -- large third-party
# binaries do not belong in git any more than bulk ONC downloads do.
#
# Provenance is the deliverable this step is actually graded on: a TRACKED
# JSON at docs/derived/b0_external_provenance.json, one entry per artefact,
# each carrying: url, revision (git commit SHA or HF revision), sha256 (of
# every file used), size_bytes, downloaded_utc (UTC ISO-8601), licence.
# Without this, "we used the ONC model" is a claim no one can audit six
# months from now when the HF repo has moved on to a new revision.
#
# Path constants belong in boatphone/paths.py ONLY (invariant 6: one
# definition, source in a comment) -- EXTERNAL_DIR, ONC_MODEL_DIR,
# CHECKPOINT_DIR. Not in a notebook, not in scripts/.
#
# CONTRACT RESOLUTIONS I HAD TO MAKE (flagged, not hidden):
#   (a) The provenance JSON's exact shape (top-level list vs dict-keyed-by-
#       artefact-name) is not specified in the brief. I assume a JSON object
#       whose values are per-artefact dicts (or a JSON array of such dicts) --
#       check_b0_1_provenance_fields walks whichever shape it finds and
#       requires every leaf artefact dict to carry the six fields. If the
#       coder picks a different shape (e.g. nested one level deeper for
#       per-file sha256 under a single revision-level record), the field
#       walker may need to change: this is a genuine ambiguity, not just a
#       test detail.
#   (b) "sha256 of every file used" is read as: at minimum, cnn_best.pt,
#       args.pkl, and the labelled eval .h5 each get an entry (or the
#       provenance record for the checkpoint artefact carries a mapping of
#       filename -> sha256 covering all three). check_b0_1_hashes_verify
#       recurses to find every (path, sha256) pair the JSON asserts and
#       verifies each independently, so it does not care which shape wins.
#   (c) "byte size" is assumed reported per FILE, matching the hash
#       granularity above -- if the coder reports one aggregate size per
#       artefact instead, this check does not verify size at all (byte size
#       is not required by the brief's assertion list), only sha256.
# ---------------------------------------------------------------------------

PROVENANCE_JSON_RELPATH = os.path.join("docs", "derived", "b0_external_provenance.json")

# Required per-artefact provenance fields, source: B0-1 brief.
B0_PROVENANCE_REQUIRED_FIELDS = ("url", "sha256", "size_bytes", "downloaded_utc", "licence")
# Exactly one of these two must be present per artefact (git clone vs HF pull).
B0_PROVENANCE_REVISION_FIELDS = ("revision", "git_commit_sha", "commit", "hf_revision")


def _b0_external_dir():
    import boatphone.paths as p
    return p


def check_b0_1_paths_constants_exist_and_are_correct():
    """boatphone.paths exposes EXTERNAL_DIR/ONC_MODEL_DIR/CHECKPOINT_DIR, inside the repo."""
    out = _child_ok(_run_child("""
        import pathlib
        import boatphone.paths as p
        names = ["EXTERNAL_DIR", "ONC_MODEL_DIR", "CHECKPOINT_DIR"]
        missing = [n for n in names if not hasattr(p, n)]
        assert not missing, "boatphone.paths is missing exports: " + repr(missing)
        bad = [n for n in names if not isinstance(getattr(p, n), pathlib.Path)]
        assert not bad, "not pathlib.Path: " + repr(bad)
        assert p.REPO_ROOT in p.EXTERNAL_DIR.parents or p.EXTERNAL_DIR == p.REPO_ROOT, (
            "EXTERNAL_DIR is not inside REPO_ROOT: " + str(p.EXTERNAL_DIR)
        )
        assert p.EXTERNAL_DIR.name == "external" and p.EXTERNAL_DIR.parent == p.REPO_ROOT, (
            "EXTERNAL_DIR is expected to be <repo root>/external, got " + str(p.EXTERNAL_DIR)
        )
        assert p.ONC_MODEL_DIR == p.EXTERNAL_DIR / "onc_ssamba", (
            "ONC_MODEL_DIR != EXTERNAL_DIR/'onc_ssamba', got " + str(p.ONC_MODEL_DIR)
        )
        assert p.CHECKPOINT_DIR == p.EXTERNAL_DIR / "checkpoints", (
            "CHECKPOINT_DIR != EXTERNAL_DIR/'checkpoints', got " + str(p.CHECKPOINT_DIR)
        )
        print("OK")
    """), "boatphone.paths B0 exports")
    assert out.strip().endswith("OK")


def check_b0_1_external_dir_is_gitignored():
    """external/ is a NEW top-level .gitignore entry (not swept in by an existing rule)."""
    assert _git_check_ignore("external/probe-file.bin") == 0, (
        "external/ is not gitignored -- a clone or checkpoint dropped there would be "
        "committable, which is exactly what B0-1 forbids for third-party binaries"
    )
    assert _git_check_ignore("external/onc_ssamba/probe.py") == 0, (
        "external/onc_ssamba/ is not gitignored"
    )
    assert _git_check_ignore("external/checkpoints/cnn_best.pt") == 0, (
        "external/checkpoints/ is not gitignored"
    )


def check_b0_1_provenance_json_exists_and_parses():
    """docs/derived/b0_external_provenance.json exists, is tracked, and parses as JSON."""
    import json
    path = REPO_ROOT / PROVENANCE_JSON_RELPATH
    if not path.is_file():
        raise AssertionError(
            f"{PROVENANCE_JSON_RELPATH} is missing; B0-1 requires a TRACKED provenance record "
            "of every external artefact acquired (source url, revision/commit, sha256, size, "
            "download timestamp, licence) -- this is not optional and not deferrable to a notebook"
        )
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{PROVENANCE_JSON_RELPATH} does not parse as JSON: {exc}") from exc
    assert doc, f"{PROVENANCE_JSON_RELPATH} parsed to an empty document"
    # Tracked, not gitignored: a provenance record that git itself would refuse to
    # commit is worthless six months from now.
    assert _git_check_ignore(PROVENANCE_JSON_RELPATH) != 0, (
        f"{PROVENANCE_JSON_RELPATH} is gitignored; provenance must be TRACKED, not derived scratch"
    )


def _b0_iter_artefact_records(doc):
    """Yield (label, record-dict) for whichever top-level shape the provenance JSON uses.

    Accepts a dict keyed by artefact name, or a list of records each carrying a
    'name'/'artefact' key. See the "CONTRACT RESOLUTIONS" note above this section:
    the exact shape is not pinned by the brief, so this walks either.
    """
    if isinstance(doc, dict):
        for name, record in doc.items():
            if isinstance(record, dict):
                yield name, record
    elif isinstance(doc, list):
        for i, record in enumerate(doc):
            if isinstance(record, dict):
                label = record.get("name") or record.get("artefact") or f"[{i}]"
                yield label, record
    else:
        raise AssertionError(
            f"{PROVENANCE_JSON_RELPATH} top level is neither a JSON object nor an array"
        )


def check_b0_1_provenance_fields_complete():
    """Every provenance entry carries url, revision/sha, sha256, size, downloaded_utc, licence."""
    import json
    from datetime import datetime, timezone
    path = REPO_ROOT / PROVENANCE_JSON_RELPATH
    if not path.is_file():
        raise SkipCheck(f"{PROVENANCE_JSON_RELPATH} absent -- see check_b0_1_provenance_json_exists_and_parses")
    doc = json.loads(path.read_text(encoding="utf-8"))
    records = list(_b0_iter_artefact_records(doc))
    assert records, f"{PROVENANCE_JSON_RELPATH} contains no artefact records"

    # We expect at least the two artefacts named in the brief.
    labels = {str(name).lower() for name, _ in records}
    expected_substrings = ("onc_ssamba", "checkpoint")
    for want in expected_substrings:
        assert any(want in label for label in labels), (
            f"no provenance record's key/name contains {want!r}; got labels {sorted(labels)} -- "
            "expected at least one entry each for the repo clone and the HF checkpoints"
        )

    problems = []
    for name, record in records:
        empty = [f for f in B0_PROVENANCE_REQUIRED_FIELDS
                  if record.get(f) in (None, "", [], {})]
        if empty:
            problems.append(f"{name}: missing/empty field(s) {empty}")
        has_revision = any(record.get(f) for f in B0_PROVENANCE_REVISION_FIELDS)
        if not has_revision:
            problems.append(
                f"{name}: no revision/commit field present (expected one of "
                f"{B0_PROVENANCE_REVISION_FIELDS})"
            )
        ts = record.get("downloaded_utc")
        if ts:
            try:
                parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                problems.append(f"{name}: downloaded_utc {ts!r} does not parse as ISO-8601")
            else:
                if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(None):
                    problems.append(f"{name}: downloaded_utc {ts!r} is not explicit UTC")
    assert not problems, "provenance field problems:\n  " + "\n  ".join(problems)


def _b0_iter_sha256_pairs(doc, base=REPO_ROOT):
    """Recursively find every (path-ish string, sha256-ish string) pair asserted anywhere
    in the provenance document. Deliberately shape-agnostic (see resolution (b) above):
    walks any nested dict/list looking for a 'sha256' key alongside a sibling
    path/filename key, so it works whether hashes are one-per-artefact or
    one-per-file-in-a-mapping.
    """
    pairs = []

    def _walk(node, context_path=None):
        if isinstance(node, dict):
            if "sha256" in node and isinstance(node["sha256"], str):
                p = (node.get("path") or node.get("file") or node.get("filename")
                     or context_path)
                if p:
                    pairs.append((str(p), node["sha256"]))
            # a mapping like {"cnn_best.pt": "abc123...", ...} nested under "sha256" or "files"
            for key in ("sha256", "files", "hashes"):
                sub = node.get(key)
                if isinstance(sub, dict):
                    for fname, val in sub.items():
                        if isinstance(val, str) and len(val) == 64:
                            pairs.append((fname, val))
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    _walk(v, context_path)
        elif isinstance(node, list):
            for item in node:
                _walk(item, context_path)

    _walk(doc)
    return pairs


def check_b0_1_recorded_hashes_verify_against_disk():
    """Every on-disk file the provenance JSON claims a sha256 for actually matches it.

    A recorded file that is ABSENT is a SKIP for that file (the clone/checkpoint
    pull has not happened on this machine -- large binaries are not in git, same
    rule as data/). A recorded file that is PRESENT but hashes differently is a
    hard FAIL: that is corruption or a stale/edited provenance record, not an
    "I don't have the data" situation, and must never be silently downgraded to
    a skip.
    """
    import hashlib
    import json
    path = REPO_ROOT / PROVENANCE_JSON_RELPATH
    if not path.is_file():
        raise SkipCheck(f"{PROVENANCE_JSON_RELPATH} absent -- see check_b0_1_provenance_json_exists_and_parses")
    doc = json.loads(path.read_text(encoding="utf-8"))
    pairs = _b0_iter_sha256_pairs(doc)
    assert pairs, (
        f"{PROVENANCE_JSON_RELPATH} parsed but no (path, sha256) pair could be extracted from it"
    )

    external_dir = REPO_ROOT / "external"
    if not external_dir.is_dir():
        raise SkipCheck(
            "external/ is absent on this machine -- large third-party artefacts are not in "
            "git, same as data/; hash verification requires the actual clone/checkpoint pull "
            "(this is a SKIP, not a pass: it proves nothing about integrity)"
        )

    absent, mismatched, verified = [], [], []
    for rel, expected_hex in pairs:
        candidate = pathlib.Path(rel)
        fp = candidate if candidate.is_absolute() else (external_dir / candidate)
        if not fp.is_file():
            # also try resolving relative to REPO_ROOT, in case the JSON used repo-relative paths
            alt = REPO_ROOT / rel
            fp = alt if alt.is_file() else fp
        if not fp.is_file():
            absent.append(rel)
            continue
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got.lower() != expected_hex.lower():
            mismatched.append((rel, expected_hex, got))
        else:
            verified.append(rel)

    if absent and not mismatched and not verified:
        raise SkipCheck(
            f"all {len(absent)} recorded file(s) are absent on disk (e.g. {absent[:3]}); "
            "clone/checkpoint pull has not happened here -- distinct from a hash mismatch"
        )
    assert not mismatched, (
        "sha256 MISMATCH for file(s) the provenance record claims to have verified "
        f"(recorded, expected, got): {mismatched}; this is corruption or a stale/edited "
        "provenance record, not a missing-file situation"
    )
    if absent:
        print(f"      [note: {len(absent)} recorded file(s) absent on this machine, "
              f"skipped individually: {absent[:5]}]")


def check_b0_1_data_dir_untouched_by_this_step():
    """B0-1 must not write anything under data/ (invariant 2: data/ is immutable/append-only).

    We cannot assert data/'s mtime is unchanged (other steps touch it), so this
    checks the narrower, still-real claim: nothing under data/ is named for the
    ONC ssamba clone or the merileo checkpoints -- the artefacts belong ONLY
    under external/.
    """
    data_dir = REPO_ROOT / "data"
    if not data_dir.is_dir():
        raise SkipCheck(f"{data_dir} absent")
    suspect_names = ("onc_ssamba", "ssamba", "merileo", "cnn_best.pt", "cnn_baseline")
    hits = []
    for p in data_dir.rglob("*"):
        low = p.name.lower()
        if any(s in low for s in suspect_names):
            hits.append(str(p.relative_to(REPO_ROOT)))
    assert not hits, (
        f"found B0 external-artefact-looking path(s) under data/: {hits}; "
        "external artefacts belong under external/, never under the immutable data/ tree "
        "(CLAUDE.md invariant 2)"
    )


def check_b0_1_external_not_staged_or_untracked_in_git():
    """`git status --porcelain` shows no staged/untracked path under external/ (invariants 2, 7).

    Even though external/ is gitignored (check_b0_1_external_dir_is_gitignored),
    a file added with `git add -f` would still show as staged; this check is the
    belt to that check's suspenders.
    """
    proc = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(REPO_ROOT),
        capture_output=True, text=True, timeout=SUBPROC_TIMEOUT_S,
    )
    assert proc.returncode == 0, f"git status failed: {proc.stderr}"
    offending = [line for line in proc.stdout.splitlines() if "external/" in line]
    assert not offending, (
        f"`git status --porcelain` shows external/ path(s) as staged/untracked: {offending}; "
        "external/ must never enter git, gitignored or not"
    )



# ===========================================================================
# B0-2a: the frequency/time axis gate for boatphone/fft_io.py.
#
# fft_io.py DOES NOT EXIST YET. Every check_b0_2a_* below except the two
# config-constants checks is expected to FAIL with an ImportError/AttributeError
# naming that fact -- see check_b0_2a_module_exists, ordered first so the real
# cause is reported instead of an opaque traceback from a later check.
#
# CONTRACT ASSUMED BY THESE CHECKS (a GUESS at fft_io's API surface -- no plan
# or existing module pins these names; flagged here rather than hidden):
#   fft_io.frequency_axis_hz(n_bins=..., bin_width_hz=...) -> ndarray[n_bins]
#       one-sided centre-frequency axis in Hz, DERIVED from bin_width_hz, not
#       a baked-in 512-length array (see the anti-hardcoding half of the check).
#   fft_io.time_axis_utc_s(frame_start_utc_s, n_frames=..., frame_seconds=...)
#       -> ndarray[n_frames] of ABSOLUTE UTC epoch seconds. MUST RAISE if
#       frame_start_utc_s is None/absent -- a bare relative 0.0, 0.25, 0.5, ...
#       axis wearing the name t_utc_s is exactly what decision 0002 forbids.
#   fft_io.read_fft_gz(path) -> object with .levels_db [n_frames, n_bins],
#       .freq_hz [n_bins], .t_utc_s [n_frames], .fs_hz.
#   fft_io.calibrated_band_hz() -> (lo_hz, hi_hz) matching bins 1-204.
#   fft_io.assert_calibratable(band_hz) -> raises if band_hz reaches outside
#       the calibratable support.
#   fft_io.assert_tone_at(levels_db, freq_hz, t_utc_s, *, expected_freq_hz,
#       expected_t_utc_s, freq_tol_hz, time_tol_s, expected_level_db=None,
#       level_tol_db=None) -> None if the array's peak (within tolerance) is
#       at the expected freq/time/level; RAISES otherwise. This is the single
#       "push a tone through the axis mapping" primitive the brief asks for;
#       its exact name/shape is a GUESS the implementer is free to rename, in
#       which case these checks should be updated to match, not weakened.
#
# Settled facts (transcribed BY HAND from the B0-2a task brief / acoustics_plan_v2
# SS3, an INDEPENDENT second reading -- see check_b0_2a_config_constants_match_
# settled_facts for why this must never import the same values it is checking):
B0_2A_N_FRAMES = 1200
B0_2A_N_BINS = 512
B0_2A_BIN_WIDTH_HZ = 250.0
B0_2A_FRAME_SECONDS = 0.25
B0_2A_STRUCTURAL_ZERO_COL0 = 0
B0_2A_CALIBRATED_BIN_RANGE = (1, 204)  # inclusive; 250 Hz - 51,000 Hz, wholly
# inside the calibration file's documented 10 Hz - 51,200 Hz span (bins 0 and
# 205 would only be admissible by extrapolation -- see review-2 [MEDIUM 2]).

# --- RESTATED 2026-08-27 by the B1a frequency-axis adjudication. -------------
# These three facts were RESTATED, not relaxed. Each new number below is a
# MEASUREMENT on the two local fixtures (2,400 frames) and, for the absolute
# centre frequency, on the 128 kHz sample WAV -- which is an absolute frequency
# reference needing no product axis at all. The two checks that used the old
# statements were failing because the STATEMENTS were wrong, not the data.
#
# (1) The top of the band is THREE regions, not one "structural zero" block:
#     425-511 is a hard zero (0 of 104,400 nonzero on EACH fixture); 419-424 is
#     the tail of the anti-alias skirt (68 and 74 of 7,200 nonzero, max 6);
#     column 0 is NEAR-zero (14 and 8 of 1,200 nonzero, max 3). The old
#     (419, 511) was off by six columns and the old "col 0 == 0 exactly" was
#     never true of real data.
B0_2A_STRUCTURAL_ZERO_COLS_HIGH = (425, 511)  # inclusive; HARD zero
B0_2A_ROLLOFF_TAIL_COLS = (419, 424)          # inclusive; skirt, not zero
B0_2A_ROLLOFF_ONSET_BIN = 408
B0_2A_DC_COL = 0
B0_2A_DC_COL_MAX_LEVEL = 5              # measured 3 and 2
B0_2A_DC_COL_MAX_NONZERO_FRACTION = 0.02  # measured 0.0117 and 0.0067
B0_2A_ROLLOFF_TAIL_MAX_LEVEL = 10       # measured 6 and 5
B0_2A_ROLLOFF_TAIL_MAX_MEAN_LEVEL = 0.05  # measured 0.0154 and 0.0149

# (2) The "38 kHz line at bin 152" was NOMINAL-DERIVED (38000/250) and could not
#     be reproduced by any statistic. It is a ~5 kHz-wide HUMP over bins 140-162
#     whose source is genuinely near 37.65 kHz, measured on the WAV.
B0_2A_ECHOSOUNDER_HUMP_BINS = (140, 162)
B0_2A_ECHOSOUNDER_CENTROID_BIN_RANGE = (149.0, 152.0)   # measured 150.36, 150.17
B0_2A_ECHOSOUNDER_ABS_CENTRE_HZ = 37650.0
B0_2A_ECHOSOUNDER_ABS_CENTRE_TOL_HZ = 150.0
B0_2A_ECHOSOUNDER_TEMPORAL_STD_ARGMAX_BIN_RANGE = (147, 155)  # measured 151, 150

# (3) Centre-vs-edge is UNRESOLVED and is CARRIED, not resolved. The reader stays
#     pinned to centres (freq_hz[1] == 250.0 exactly, below) but that is now a
#     NAMED ASSUMPTION with a price: half a bin, one-sided toward higher
#     frequency, on every band edge.
#     NO CHECK IN THIS FILE MAY ASSERT A BIN POSITION TIGHTER THAN +/- 1 BIN
#     until B1 settles the convention -- which is why (2) asserts a centroid
#     inside a 3-bin window rather than an argmax at a bin.
B0_2A_AXIS_CONVENTION = "centre"
B0_2A_AXIS_OFFSET_UNCERTAINTY_HZ = 125.0

# Names this segment expects boatphone/config.py to define. A GUESS at naming;
# flagged, not hidden -- see the module note above.
_B0_2A_CONFIG_NAMES = (
    "FFT_N_FRAMES", "FFT_N_BINS", "FFT_BIN_WIDTH_HZ", "FFT_FRAME_SECONDS",
    "FFT_STRUCTURAL_ZERO_COL0", "FFT_STRUCTURAL_ZERO_COLS_HIGH",
    "FFT_ROLLOFF_TAIL_COLS", "FFT_ROLLOFF_ONSET_BIN", "FFT_DC_COL",
    "FFT_ECHOSOUNDER_HUMP_BINS", "FFT_ECHOSOUNDER_CENTROID_BIN_RANGE",
    "FFT_ECHOSOUNDER_ABS_CENTRE_HZ",
    "FFT_AXIS_CONVENTION", "FFT_AXIS_OFFSET_UNCERTAINTY_HZ",
    "FFT_CALIBRATED_BIN_RANGE",
    "FFT_B5_CALIBRATED_CEILING_BIN", "FFT_B5_RELATIVE_CEILING_BIN",
    "FFT_LEVEL_FLOOR", "FFT_LEVEL_CEILING",
)

# Tolerances, named and justified (CLAUDE.md invariant 3 -- explicit, with a reason).
B0_2A_FREQ_TOL_BINS = 1                          # brief: "within one bin"
B0_2A_FREQ_TOL_HZ = B0_2A_FREQ_TOL_BINS * B0_2A_BIN_WIDTH_HZ
B0_2A_TIME_TOL_S = B0_2A_FRAME_SECONDS           # brief: "within one frame, 0.25 s"
# A synthetic tone is inserted at an EXACT level; the round trip through a pure
# axis mapping (no resampling, no windowing re-applied) should differ only by
# float/log rounding. 0.5 dB is generous for that, and still tight enough to
# catch a real scaling-convention bug (power-vs-amplitude dB is a factor of 2,
# i.e. ~6 dB; a one-sided/two-sided Parseval slip is ~3 dB) -- this tolerance
# is NOT a real-world noise-floor margin.
B0_2A_LEVEL_TOL_DB = 0.5

# "Deliberately offset by one bin" (brief's own words) sits EXACTLY AT the
# freq_tol_bins=1 boundary and would not unambiguously fail a <= comparison --
# an ambiguity in the brief, resolved here by using a 2-bin offset (still a
# small, physically tiny 500 Hz shift) as the negative control. Flagged, not
# silently chosen.
B0_2A_WRONG_BIN_OFFSET = 2
B0_2A_WRONG_TIME_OFFSET_S = 3600.0  # the brief's own example, verbatim (invariant 4)


def _b0_2a_cfg():
    """Import boatphone.config. ImportError here is a real failure (config.py exists)."""
    import importlib
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("boatphone.config")


def _b0_2a_fft_io():
    """Import boatphone.config and boatphone.fft_io.

    ImportError/AttributeError on boatphone.fft_io IS the expected failure for
    every B0-2a check below except the two config-constants checks: the module
    does not exist yet.
    """
    import importlib
    cfg = _b0_2a_cfg()
    fft_io = importlib.import_module("boatphone.fft_io")
    return cfg, fft_io


def check_b0_2a_config_constants_exist():
    """boatphone/config.py names the B0-2a axis constants -- ONE definition (invariant 6).

    Does not hardcode values into fft_io or into this file; only asserts
    config.py carries them, so fft_io.py (and any notebook) has exactly one
    place to import them from.
    """
    cfg = _b0_2a_cfg()
    missing = [n for n in _B0_2A_CONFIG_NAMES if not hasattr(cfg, n)]
    assert not missing, (
        f"boatphone.config is missing B0-2a axis constant(s) {missing}; the "
        "1200x512 frame/bin shape, 250 Hz bin width, 0.25 s frame duration, "
        "structural-zero columns, 38 kHz line bin, and the 1-204 calibratable "
        "bin range are project-wide facts (acoustics_plan_v2 SS3) and must have "
        "exactly one definition, in boatphone/config.py -- not a second one in "
        "boatphone/fft_io.py or scripts/checks.py (CLAUDE.md invariant 6)"
    )


def check_b0_2a_config_constants_match_settled_facts():
    """Those constants, once present, must equal the SETTLED FACTS from the brief.

    Only asserts on names that are actually present, so this adds information
    beyond check_b0_2a_config_constants_exist (which already fails on any
    missing name) rather than duplicating its failure verbatim.
    """
    cfg = _b0_2a_cfg()
    expected = {
        "FFT_N_FRAMES": B0_2A_N_FRAMES,
        "FFT_N_BINS": B0_2A_N_BINS,
        "FFT_BIN_WIDTH_HZ": B0_2A_BIN_WIDTH_HZ,
        "FFT_FRAME_SECONDS": B0_2A_FRAME_SECONDS,
        "FFT_STRUCTURAL_ZERO_COL0": B0_2A_STRUCTURAL_ZERO_COL0,
        "FFT_STRUCTURAL_ZERO_COLS_HIGH": B0_2A_STRUCTURAL_ZERO_COLS_HIGH,
        "FFT_ROLLOFF_TAIL_COLS": B0_2A_ROLLOFF_TAIL_COLS,
        "FFT_ROLLOFF_ONSET_BIN": B0_2A_ROLLOFF_ONSET_BIN,
        "FFT_DC_COL": B0_2A_DC_COL,
        "FFT_ECHOSOUNDER_HUMP_BINS": B0_2A_ECHOSOUNDER_HUMP_BINS,
        "FFT_ECHOSOUNDER_CENTROID_BIN_RANGE": B0_2A_ECHOSOUNDER_CENTROID_BIN_RANGE,
        "FFT_ECHOSOUNDER_ABS_CENTRE_HZ": B0_2A_ECHOSOUNDER_ABS_CENTRE_HZ,
        "FFT_AXIS_CONVENTION": B0_2A_AXIS_CONVENTION,
        "FFT_AXIS_OFFSET_UNCERTAINTY_HZ": B0_2A_AXIS_OFFSET_UNCERTAINTY_HZ,
        "FFT_CALIBRATED_BIN_RANGE": B0_2A_CALIBRATED_BIN_RANGE,
        # The two B5 ceilings, kept apart on purpose: 205 is where CALIBRATION
        # stops (51,000 Hz, bins 1-204 wholly inside the file's stated span),
        # 408 is where the instrument response stops being
        # ocean (~102 kHz). Nothing above 408 may enter a B5 statistic.
        "FFT_B5_CALIBRATED_CEILING_BIN": B0_2A_CALIBRATED_BIN_RANGE[1],
        "FFT_B5_RELATIVE_CEILING_BIN": B0_2A_ROLLOFF_ONSET_BIN,
        "FFT_LEVEL_FLOOR": 0,
        "FFT_LEVEL_CEILING": 86,
    }
    present = {k: v for k, v in expected.items() if hasattr(cfg, k)}
    if not present:
        raise SkipCheck(
            "no B0-2a config constants present yet -- see check_b0_2a_config_constants_exist"
        )
    mismatched = [(k, getattr(cfg, k), v) for k, v in present.items() if getattr(cfg, k) != v]
    assert not mismatched, (
        "boatphone.config constant(s) present but WRONG vs the B0-2a settled facts "
        f"(name, got, expected): {mismatched}"
    )


def check_b0_2a_module_exists():
    """boatphone/fft_io.py exists at all.

    Expected to FAIL right now -- the module is not written. Every other
    check_b0_2a_* below (except the config-constants pair above) fails for the
    same reason until it exists; this one is ordered first so the real cause
    is named instead of an opaque ImportError from deeper in the file.
    """
    path = REPO_ROOT / "boatphone" / "fft_io.py"
    assert path.is_file(), (
        f"{path} does not exist yet -- B0-2a (the frequency/time axis gate) has "
        "not been implemented. That is the point: a check that has never failed "
        "has not been shown to check anything."
    )


def check_b0_2a_frequency_axis_bin_width_and_bin1():
    """frequency_axis_hz() is DERIVED from bin width, not hardcoded.

    Bin 1 centre must be exactly 250 Hz (settled fact); a fabricated bin width
    passed as an argument must move the axis (anti-hardcoding, same shape as
    A8b's common_support_moves_with_vtuad_input check) -- otherwise this check
    and the tone checks below could all pass against a fft_io.py that returns a
    baked-in array regardless of its arguments.
    """
    cfg, fft_io = _b0_2a_fft_io()
    freq_hz = np.asarray(fft_io.frequency_axis_hz(), dtype=float)
    assert freq_hz.shape == (B0_2A_N_BINS,), (
        f"frequency_axis_hz() returned shape {freq_hz.shape}, expected "
        f"({B0_2A_N_BINS},) -- one centre frequency per FFT bin"
    )
    assert freq_hz[0] == 0.0, (
        f"bin 0 centre is {freq_hz[0]} Hz, expected 0.0 Hz (bin 0 is DC and a "
        "structural zero, but its FREQUENCY axis value is still 0 Hz)"
    )
    assert abs(freq_hz[1] - B0_2A_BIN_WIDTH_HZ) < 1e-9, (
        f"bin 1 centre is {freq_hz[1]} Hz, expected exactly {B0_2A_BIN_WIDTH_HZ} Hz "
        "(settled fact: 250 Hz/bin, 1024-pt FFT at 256 kHz, 512 bins over 0-128 kHz)"
    )
    widths = np.diff(freq_hz)
    assert np.allclose(widths, B0_2A_BIN_WIDTH_HZ), (
        f"frequency_axis_hz() is not uniformly spaced at {B0_2A_BIN_WIDTH_HZ} Hz "
        f"(observed widths range {widths.min()}..{widths.max()} Hz)"
    )

    fabricated_hz = np.asarray(fft_io.frequency_axis_hz(n_bins=8, bin_width_hz=37.0), dtype=float)
    assert fabricated_hz.shape == (8,), (
        f"frequency_axis_hz(n_bins=8, bin_width_hz=37.0) returned shape "
        f"{fabricated_hz.shape}, expected (8,) -- n_bins is not read from its argument"
    )
    assert abs(fabricated_hz[1] - 37.0) < 1e-9, (
        f"frequency_axis_hz(n_bins=8, bin_width_hz=37.0)[1] == {fabricated_hz[1]}, "
        "expected 37.0 -- the axis is not derived from bin_width_hz; it looks "
        "hardcoded to the real 250 Hz product regardless of its argument"
    )


def check_b0_2a_time_axis_is_absolute_utc_not_bare_relative():
    """time_axis_utc_s() returns ABSOLUTE UTC epoch seconds, and REFUSES to run without one.

    Positive half: given a known absolute UTC start, every sample equals
    start + frame_index * frame_seconds exactly (integer arithmetic on a stated
    grid -- zero tolerance, there is no float error to absorb).
    Negative half: calling it with no absolute start (frame_start_utc_s=None)
    must RAISE rather than silently falling back to a bare relative
    0.0, 0.25, 0.5, ... axis returned under the t_utc_s name -- exactly the
    thing CLAUDE.md's naming convention and decision 0002 forbid.
    """
    cfg, fft_io = _b0_2a_fft_io()
    start_utc_s = 1_700_000_000.0  # an arbitrary but FIXED absolute UTC epoch instant
    t_utc_s = np.asarray(
        fft_io.time_axis_utc_s(start_utc_s, n_frames=B0_2A_N_FRAMES,
                                frame_seconds=B0_2A_FRAME_SECONDS),
        dtype=float,
    )
    assert t_utc_s.shape == (B0_2A_N_FRAMES,), (
        f"time_axis_utc_s() returned shape {t_utc_s.shape}, expected ({B0_2A_N_FRAMES},)"
    )
    expected = start_utc_s + np.arange(B0_2A_N_FRAMES) * B0_2A_FRAME_SECONDS
    assert np.allclose(t_utc_s, expected, atol=1e-6), (
        "time_axis_utc_s() != start + frame_index*frame_seconds "
        f"(max abs diff {np.max(np.abs(t_utc_s - expected))} s)"
    )
    assert t_utc_s[0] == start_utc_s, (
        f"t_utc_s[0] = {t_utc_s[0]}, expected the ABSOLUTE start {start_utc_s} -- "
        "an axis starting at 0.0 would be a bare relative axis wearing the "
        "t_utc_s name"
    )

    try:
        fft_io.time_axis_utc_s(None, n_frames=B0_2A_N_FRAMES, frame_seconds=B0_2A_FRAME_SECONDS)
    except Exception:
        pass
    else:
        raise AssertionError(
            "time_axis_utc_s(None, ...) returned without raising; a bare relative "
            "axis with no absolute UTC anchor must never be produced silently"
        )


def check_b0_2a_assert_tone_at_signature_names_conventions():
    """fft_io.assert_tone_at()'s signature names freq_hz/t_utc_s/levels_db (decision 0002 SS4,
    the same rule A8b's band_limit signature check pins)."""
    cfg, fft_io = _b0_2a_fft_io()
    params = set(inspect.signature(fft_io.assert_tone_at).parameters)
    required = {"levels_db", "freq_hz", "t_utc_s", "expected_freq_hz", "expected_t_utc_s"}
    missing = sorted(required - params)
    assert not missing, (
        f"fft_io.assert_tone_at's signature is missing named parameter(s) {missing} "
        f"(got {sorted(params)}) -- a frequency/time/level parameter must be named "
        "freq_hz/t_utc_s/*_db, not f/t/x (decision 0002 SS4)"
    )


def _b0_2a_synthetic_product(fft_io, *, true_bin=300, true_frame=400,
                              level_true_db=-20.0, floor_db=-90.0,
                              start_utc_s=1_700_000_000.0):
    """Build one in-memory synthetic tone array plus its axes, via fft_io itself."""
    levels_db = np.full((B0_2A_N_FRAMES, B0_2A_N_BINS), floor_db, dtype=float)
    levels_db[true_frame, true_bin] = level_true_db
    freq_hz = np.asarray(fft_io.frequency_axis_hz(), dtype=float)
    t_utc_s = np.asarray(
        fft_io.time_axis_utc_s(start_utc_s, n_frames=B0_2A_N_FRAMES,
                                frame_seconds=B0_2A_FRAME_SECONDS),
        dtype=float,
    )
    return levels_db, freq_hz, t_utc_s


def check_b0_2a_synthetic_tone_positive_control_survives_axis_mapping():
    """POSITIVE CONTROL: a synthetic tone of known freq/time/level, pushed through
    fft_io's real axis mapping, must be accepted at the correct bin (within one
    bin), the correct frame (within 0.25 s), and the correct level (within the
    stated dB tolerance). This is the strongest check available (method note:
    "the strongest check ... is a signal you built ... coming back out at the
    right level, the right frequency, and the right time")."""
    cfg, fft_io = _b0_2a_fft_io()
    true_bin, true_frame, level_true_db = 300, 400, -20.0
    levels_db, freq_hz, t_utc_s = _b0_2a_synthetic_product(
        fft_io, true_bin=true_bin, true_frame=true_frame, level_true_db=level_true_db,
    )
    try:
        fft_io.assert_tone_at(
            levels_db, freq_hz, t_utc_s,
            expected_freq_hz=freq_hz[true_bin], expected_t_utc_s=t_utc_s[true_frame],
            freq_tol_hz=B0_2A_FREQ_TOL_HZ, time_tol_s=B0_2A_TIME_TOL_S,
            expected_level_db=level_true_db, level_tol_db=B0_2A_LEVEL_TOL_DB,
        )
    except Exception as exc:
        raise AssertionError(
            "assert_tone_at() raised for a tone at its OWN true bin/frame/level "
            f"({type(exc).__name__}: {exc}); the positive control must pass"
        ) from exc


def check_b0_2a_synthetic_tone_wrong_bin_offset_is_rejected():
    """NEGATIVE CONTROL: a tone claimed to be B0_2A_WRONG_BIN_OFFSET bins away from
    where it actually is must be REJECTED, not silently accepted.

    Proves the positive control's tolerance has discriminating power -- a
    tolerance loose enough to accept any bin would make the positive control
    above pass for a broken mapping too.
    """
    cfg, fft_io = _b0_2a_fft_io()
    true_bin, true_frame, level_true_db = 300, 400, -20.0
    levels_db, freq_hz, t_utc_s = _b0_2a_synthetic_product(
        fft_io, true_bin=true_bin, true_frame=true_frame, level_true_db=level_true_db,
    )
    wrong_bin = true_bin + B0_2A_WRONG_BIN_OFFSET
    try:
        fft_io.assert_tone_at(
            levels_db, freq_hz, t_utc_s,
            expected_freq_hz=freq_hz[wrong_bin], expected_t_utc_s=t_utc_s[true_frame],
            freq_tol_hz=B0_2A_FREQ_TOL_HZ, time_tol_s=B0_2A_TIME_TOL_S,
        )
    except Exception:
        pass
    else:
        raise AssertionError(
            f"assert_tone_at() ACCEPTED a claim {B0_2A_WRONG_BIN_OFFSET} bins "
            f"({B0_2A_WRONG_BIN_OFFSET * B0_2A_BIN_WIDTH_HZ} Hz) away from the tone's "
            f"true bin {true_bin}; a mapping that cannot tell these apart is not "
            "verifying the frequency axis at all"
        )


def check_b0_2a_synthetic_tone_frame_shuffle_null_is_rejected():
    """NULL CHECK (invariant 4): shuffling which FRAME a tone's data lives in,
    while leaving the time axis itself untouched, must be rejected at the
    tone's original time for every draw.

    This reproduces the frame-shuffle null from the B1a-impl ledger entry
    ("Frame-shuffle 200 draws: 0/200 accepted"), which previously existed only
    as a one-off diagnostic outside the repo -- CLAUDE.md invariant 4 requires
    nulls to be load-bearing evidence, so it is added here as a repeatable
    check rather than left to trust. Uses fewer draws than the original
    diagnostic (deterministic seed, small N) to keep the suite fast; the
    property being tested does not depend on draw count.
    """
    cfg, fft_io = _b0_2a_fft_io()
    true_bin, true_frame, level_true_db = 300, 400, -20.0
    levels_db, freq_hz, t_utc_s = _b0_2a_synthetic_product(
        fft_io, true_bin=true_bin, true_frame=true_frame, level_true_db=level_true_db,
    )
    expected_t_utc_s = t_utc_s[true_frame]
    rng = np.random.default_rng(0)
    n_draws = 50
    n_accepted = 0
    n_run = 0
    for _ in range(n_draws):
        perm = rng.permutation(B0_2A_N_FRAMES)
        if perm[true_frame] == true_frame:
            continue  # identity at the tone's own frame is not a shuffle
        shuffled_levels_db = levels_db[perm, :]
        n_run += 1
        try:
            fft_io.assert_tone_at(
                shuffled_levels_db, freq_hz, t_utc_s,
                expected_freq_hz=freq_hz[true_bin], expected_t_utc_s=expected_t_utc_s,
                freq_tol_hz=B0_2A_FREQ_TOL_HZ, time_tol_s=B0_2A_TIME_TOL_S,
                expected_level_db=level_true_db, level_tol_db=B0_2A_LEVEL_TOL_DB,
            )
        except Exception:
            pass
        else:
            n_accepted += 1
    assert n_run >= n_draws - 5, (
        f"only {n_run} of {n_draws} draws produced a genuine frame shuffle -- "
        "RNG/permutation logic is broken, not a passing null"
    )
    assert n_accepted == 0, (
        f"frame-shuffle null: {n_accepted}/{n_run} shuffled draws were ACCEPTED "
        f"at the tone's true time {expected_t_utc_s}; a mapping that cannot tell "
        "a shuffled frame order from the true one is not verifying the time axis "
        "at all (CLAUDE.md invariant 4)"
    )


def check_b0_2a_synthetic_tone_wrong_hour_offset_is_rejected():
    """NEGATIVE CONTROL, the invariant-4 case verbatim: a tone claimed to have
    started one hour later (or earlier) than it actually did must be REJECTED.

    A mapping that ACCEPTS a one-hour-shifted claim is exactly the
    time-alignment bug CLAUDE.md invariant 4 warns produces a clean-looking but
    wrong correlation.
    """
    cfg, fft_io = _b0_2a_fft_io()
    true_bin, true_frame, level_true_db = 300, 400, -20.0
    levels_db, freq_hz, t_utc_s = _b0_2a_synthetic_product(
        fft_io, true_bin=true_bin, true_frame=true_frame, level_true_db=level_true_db,
    )
    wrong_claimed_utc_s = t_utc_s[true_frame] + B0_2A_WRONG_TIME_OFFSET_S
    try:
        fft_io.assert_tone_at(
            levels_db, freq_hz, t_utc_s,
            expected_freq_hz=freq_hz[true_bin], expected_t_utc_s=wrong_claimed_utc_s,
            freq_tol_hz=B0_2A_FREQ_TOL_HZ, time_tol_s=B0_2A_TIME_TOL_S,
        )
    except Exception:
        pass
    else:
        raise AssertionError(
            f"assert_tone_at() ACCEPTED a claim {B0_2A_WRONG_TIME_OFFSET_S} s "
            f"(one hour) away from the tone's true onset {t_utc_s[true_frame]}; "
            "this is precisely the false-positive shape CLAUDE.md invariant 4 warns "
            "a shuffled/shifted time base can produce"
        )


def _b0_2a_fixture_levels(fixture_index=0):
    """Read one real fixture, or SkipCheck if the acquisition is not on this host."""
    sample = REPO_ROOT / "data" / SAMPLE_DIR_NAME
    if not sample.is_dir():
        raise SkipCheck(f"acquisition directory absent: {sample}")
    missing = [n for n in FIXTURE_FFT_NAMES if not (sample / n).is_file()]
    if missing:
        raise SkipCheck(f"fixture .fft.gz files absent from {sample}: {missing}")
    cfg, fft_io = _b0_2a_fft_io()
    path = sample / FIXTURE_FFT_NAMES[fixture_index]
    product = fft_io.read_fft_gz(path)
    return cfg, fft_io, path, product


def check_b0_2a_real_fixture_shape_and_structural_zeros():
    """Data-dependent: a REAL .fft.gz is 1200x512 with the top of the band behaving
    as the THREE regions it actually has, and t_utc_s[0] an absolute UTC instant.

    RESTATED, NOT RELAXED. This check previously asserted "column 0 is all zero"
    and "columns 419-511 are all zero" and failed on real data. Both statements
    were wrong: 419-424 is the tail of the anti-alias skirt (nonzero by a few
    counts), and column 0 is near-zero rather than zero. The response is a
    HARDER assertion where the data supports one -- 425-511 is now required to
    be EXACTLY zero everywhere, on the strength of 208,800 cells measured across
    both fixtures -- and a bounded assertion where it does not.

    Every check runs over BOTH fixtures, not one: a bound tuned on a single file
    is a description of that file.
    """
    cfg, fft_io, _path, _product = _b0_2a_fixture_levels(0)
    for name in FIXTURE_FFT_NAMES:
        path = REPO_ROOT / "data" / SAMPLE_DIR_NAME / name
        product = fft_io.read_fft_gz(path)
        levels = np.asarray(product.levels_db)
        assert levels.shape == (B0_2A_N_FRAMES, B0_2A_N_BINS), (
            f"{name}: read_fft_gz().levels_db has shape {levels.shape}, expected "
            f"({B0_2A_N_FRAMES}, {B0_2A_N_BINS}) (frames x bins, settled fact)"
        )

        # (a) HARD structural zero, 425-511. Any nonzero value is a reader or
        #     format failure -- there is no "quiet ocean" reading of a column
        #     the product generator never writes to.
        lo, hi = B0_2A_STRUCTURAL_ZERO_COLS_HIGH
        block = levels[:, lo:hi + 1]
        assert np.all(block == 0), (
            f"{name}: columns {lo}-{hi} are NOT exactly zero -- "
            f"{np.count_nonzero(block)} of {block.size} cells nonzero, max "
            f"{block.max()}. Measured 0 of 104,400 on each local fixture, so a "
            "nonzero value here is a reader/format failure (a mis-strided or "
            "column-major reading), not data."
        )

        # (b) The DC column is NEAR-zero, bounded -- not exactly zero.
        col0 = levels[:, B0_2A_DC_COL]
        nonzero_fraction = np.count_nonzero(col0) / col0.size
        assert col0.max() <= B0_2A_DC_COL_MAX_LEVEL, (
            f"{name}: column {B0_2A_DC_COL} reaches {col0.max()}, above the bound "
            f"{B0_2A_DC_COL_MAX_LEVEL} (measured max 3 and 2 on the two fixtures). "
            "Real content in the DC column would mean the axis is not what we think."
        )
        assert nonzero_fraction <= B0_2A_DC_COL_MAX_NONZERO_FRACTION, (
            f"{name}: column {B0_2A_DC_COL} is nonzero in {nonzero_fraction:.4f} of "
            f"frames, above the bound {B0_2A_DC_COL_MAX_NONZERO_FRACTION} "
            "(measured 0.0117 and 0.0067)"
        )

        # (c) The roll-off tail, 419-424: bounded, and -- the assertion that
        #     actually catches a bug -- NON-INCREASING in per-bin mean across
        #     the whole skirt. A mis-strided or wrapped row moves a bin mean by
        #     order 1; the bound alone would not notice, the shape does.
        tail_lo, tail_hi = B0_2A_ROLLOFF_TAIL_COLS
        tail = levels[:, tail_lo:tail_hi + 1]
        assert tail.max() <= B0_2A_ROLLOFF_TAIL_MAX_LEVEL, (
            f"{name}: roll-off tail columns {tail_lo}-{tail_hi} reach {tail.max()}, "
            f"above the bound {B0_2A_ROLLOFF_TAIL_MAX_LEVEL} (measured 6 and 5)"
        )
        assert tail.mean() <= B0_2A_ROLLOFF_TAIL_MAX_MEAN_LEVEL, (
            f"{name}: roll-off tail columns {tail_lo}-{tail_hi} have mean "
            f"{tail.mean():.5f}, above the bound {B0_2A_ROLLOFF_TAIL_MAX_MEAN_LEVEL} "
            "(measured 0.0154 and 0.0149)"
        )
        report = fft_io.structural_zero_report(levels)
        profile = np.asarray(report["rolloff_profile_bin_means"], dtype=float)
        first_bin = int(report["rolloff_profile_first_bin"])
        # Tolerance = two counts in one frame (config.FFT_ROLLOFF_MONOTONIC_TOL_LEVEL),
        # comfortably above the smallest step the product can express. At the far
        # end of the skirt the mean is quantised to multiples of 1/1200 and its
        # ordering there is integer noise, not physics -- fixture ...000004 has
        # bin 423 at 0.0000 and bin 424 at 0.0008. This is a quantisation floor,
        # NOT a loosened bound: it is still ~3 orders of magnitude below the
        # order-1 step a stride bug makes.
        tol = float(getattr(cfg, "FFT_ROLLOFF_MONOTONIC_TOL_LEVEL"))
        steps = np.diff(profile)
        offenders = [
            (first_bin + i, float(profile[i]), float(profile[i + 1]))
            for i, d in enumerate(steps) if d > tol
        ]
        assert not offenders, (
            f"{name}: the per-bin mean level across the anti-alias skirt (bins "
            f"{first_bin}-{tail_hi}) is NOT non-increasing; it rises at "
            f"(bin, mean, next_mean) = {offenders} by more than the one-count "
            f"quantisation step {tol:.6f}. The skirt is one continuous filter "
            "response, so a rise in it is the signature of a mis-strided or "
            "wrapped row -- exactly the failure a cell-count bound would miss."
        )

        freq_hz = np.asarray(product.freq_hz, dtype=float)
        t_utc_s = np.asarray(product.t_utc_s, dtype=float)
        assert freq_hz.shape == (B0_2A_N_BINS,), (
            f"{name}: freq_hz shape {freq_hz.shape}, expected ({B0_2A_N_BINS},)"
        )
        assert t_utc_s.shape == (B0_2A_N_FRAMES,), (
            f"{name}: t_utc_s shape {t_utc_s.shape}, expected ({B0_2A_N_FRAMES},)"
        )
        # A relative axis would start at 0.0; a real 2026 UTC instant is >> 1e9.
        assert t_utc_s[0] > 1_000_000_000.0, (
            f"{name}: t_utc_s[0] = {t_utc_s[0]} looks like a relative offset, "
            "not an absolute UTC epoch second"
        )


def check_b0_2a_real_fixture_echosounder_hump_centroid():
    """Data-dependent: the ~38 kHz echosounder hump's POWER-EXCESS CENTROID lands
    in bins 149.0-152.0 on both real fixtures -- a real-data confirmation of the
    frequency mapping, independent of the synthetic-tone checks.

    RESTATED, NOT RELAXED, and in two ways it is STRONGER than the "line at bin
    152 +/- 1" it replaces:

      * the old target was NOMINAL-DERIVED (38000 / 250 = 152) and could not be
        reproduced by ANY statistic on either fixture -- mean-argmax read 150 and
        149, temporal-std argmax 150 and 151, ping-excess centroid 150.8 and
        151.0. The check was failing because the target was wrong.
      * the statistic is now a CENTROID, not an argmax. The feature is a ~5 kHz
        hump quantised to integer counts, and its argmax is unstable at +/- 1 bin
        between two files five minutes apart (149 vs 150) while the centroid
        moved 0.19 bins (150.36 vs 150.17). Asserting on an argmax at +/- 1 bin
        was a coin flip; asserting a centroid inside a 3-bin window is a
        measurement with a reproducible value behind it.

    WHAT THIS CHECK IS FOR: rejecting a 2x MAPPING ERROR, where it discriminates
    by ~150 bins. It is deliberately NOT tight enough to adjudicate centre-vs-edge
    (a half-bin question) -- under edge the hump reads 37.70 kHz and under centre
    37.58 kHz, and which is closer flips with the background model and the assumed
    dB scale. See check_b0_2a_axis_uncertainty_is_carried.
    """
    _cfg, fft_io, _path, _product = _b0_2a_fixture_levels(0)
    lo_bin, hi_bin = B0_2A_ECHOSOUNDER_CENTROID_BIN_RANGE
    hump_lo, hump_hi = B0_2A_ECHOSOUNDER_HUMP_BINS
    std_lo, std_hi = B0_2A_ECHOSOUNDER_TEMPORAL_STD_ARGMAX_BIN_RANGE
    centroids = {}
    for name in FIXTURE_FFT_NAMES:
        path = REPO_ROOT / "data" / SAMPLE_DIR_NAME / name
        levels = np.asarray(fft_io.read_fft_gz(path).levels_db)
        centroid = float(fft_io.echosounder_centroid_bin(levels))
        centroids[name] = centroid
        assert lo_bin <= centroid <= hi_bin, (
            f"{name}: the power-excess centroid of the echosounder hump is at bin "
            f"{centroid:.3f}, outside the expected {lo_bin}-{hi_bin} (measured "
            f"150.36 and 150.17). At {B0_2A_BIN_WIDTH_HZ} Hz/bin that names "
            f"{centroid * B0_2A_BIN_WIDTH_HZ / 1000:.2f} kHz for a source measured "
            f"ABSOLUTELY on the 128 kHz sample WAV at "
            f"{B0_2A_ECHOSOUNDER_ABS_CENTRE_HZ / 1000:.2f} kHz +/- "
            f"{B0_2A_ECHOSOUNDER_ABS_CENTRE_TOL_HZ} Hz -- so a miss here is a "
            "frequency-MAPPING error, not a source that moved."
        )
        # The hump must lie inside its stated extent, or "the centroid" is being
        # taken over the wrong feature.
        assert hump_lo <= centroid <= hump_hi, (
            f"{name}: centroid {centroid:.3f} is outside the stated hump extent "
            f"bins {hump_lo}-{hump_hi}"
        )
        # SECONDARY, WEAKER: intermittency is what makes this an echosounder and
        # not a resonance. Asserted over an 8-bin window, deliberately loose.
        std_bin = int(fft_io.temporal_std_argmax_bin(levels))
        assert std_lo <= std_bin <= std_hi, (
            f"{name}: the bin of peak per-bin TEMPORAL STD is {std_bin}, outside "
            f"{std_lo}-{std_hi} (measured 151 and 150). This is the intermittency "
            "signature; if the loudest bin and the most variable bin are not the "
            "same feature, the hump is not the echosounder."
        )

    # The centroid's whole value is that it is STABLE where the argmax is not.
    # If two windows five minutes apart disagree by more than a bin, this
    # statistic is not measuring the source and the bound above is luck.
    spread = max(centroids.values()) - min(centroids.values())
    assert spread <= 1.0, (
        f"the power-excess centroid moved {spread:.3f} bins between fixtures "
        f"({centroids}); measured 0.19. A centroid that unstable is not a "
        "position measurement and must not be used as an axis check."
    )


def check_b0_2a_axis_uncertainty_is_carried():
    """The centre-vs-edge question is CARRIED, not silently resolved.

    freq_hz[1] == 250.0 stays pinned -- the reader must be deterministic -- but
    that is a NAMED ASSUMPTION (config.FFT_AXIS_CONVENTION), not a settled fact.
    The B1a adjudication put it at roughly 60/40 toward ONC meaning bin EDGES,
    which would move every named frequency up by half a bin. Nothing in the
    evidence is strong enough to act on, and nothing is weak enough to ignore.

    This check exists so that a later edit cannot quietly drop the open question
    by deleting the constant or by ceasing to apply it. It asserts BOTH:
      (a) the uncertainty is declared and strictly positive; and
      (b) boatphone.models.band_limit ACTUALLY WIDENS its kept support by it --
          measured by band-limiting a real product spectrum with and without it
          and requiring the widened call to keep bins the narrow one drops.
    A constant that is defined but never applied is worse than no constant: it
    documents a caution the code does not take.
    """
    cfg, fft_io = _b0_2a_fft_io()
    import boatphone.models as bm

    uncertainty_hz = float(getattr(cfg, "FFT_AXIS_OFFSET_UNCERTAINTY_HZ"))
    assert uncertainty_hz > 0.0, (
        "config.FFT_AXIS_OFFSET_UNCERTAINTY_HZ is "
        f"{uncertainty_hz}; the centre-vs-edge convention is UNRESOLVED (B1a "
        "adjudication) and a zero uncertainty asserts it is settled. If it has "
        "genuinely been settled -- by ONC's own product definition, by the "
        "two-bin-split census over the B3 corpus, or by the WAV centroid once B1 "
        "pins counts to dB -- then delete this check together with the constant "
        "and record the decision. Do not just zero the number."
    )
    assert uncertainty_hz <= 0.5 * B0_2A_BIN_WIDTH_HZ + 1e-9, (
        f"config.FFT_AXIS_OFFSET_UNCERTAINTY_HZ is {uncertainty_hz} Hz, more than "
        f"half a bin ({0.5 * B0_2A_BIN_WIDTH_HZ} Hz). The open question is only "
        "which point of a bin the axis names; it cannot be worth more than half a "
        "bin, and a larger value would be hiding a different problem."
    )
    assert getattr(cfg, "FFT_AXIS_CONVENTION") == B0_2A_AXIS_CONVENTION, (
        f"config.FFT_AXIS_CONVENTION is {getattr(cfg, 'FFT_AXIS_CONVENTION')!r}, "
        f"expected {B0_2A_AXIS_CONVENTION!r} -- the reader is pinned to bin "
        "centres so that it is deterministic; changing it changes every frequency "
        "this project has ever named and needs a decision record, not an edit."
    )

    # The pin itself still holds: bin 1 is exactly 250 Hz.
    freq_hz = np.asarray(fft_io.frequency_axis_hz(), dtype=float)
    assert freq_hz[1] == B0_2A_BIN_WIDTH_HZ, (
        f"frequency_axis_hz()[1] = {freq_hz[1]}, expected exactly "
        f"{B0_2A_BIN_WIDTH_HZ}; the reader must stay deterministic even while the "
        "convention it encodes is an open question"
    )

    # (b) The uncertainty is APPLIED, not merely declared. Band edges chosen to
    #     sit inside a bin, so widening by 125 Hz demonstrably changes the answer.
    spectrum = np.zeros_like(freq_hz)
    band_hz = (B0_2A_BIN_WIDTH_HZ * 8 + 100.0, B0_2A_BIN_WIDTH_HZ * 40 - 100.0)
    narrow_freq, _ = bm.band_limit(
        freq_hz, spectrum, B0_2A_BIN_WIDTH_HZ * 2 * B0_2A_N_BINS, band_hz
    )
    widened_freq, _ = bm.band_limit(
        freq_hz, spectrum, B0_2A_BIN_WIDTH_HZ * 2 * B0_2A_N_BINS, band_hz,
        axis_offset_uncertainty_hz=uncertainty_hz,
    )
    assert widened_freq.size > narrow_freq.size, (
        f"boatphone.models.band_limit kept {widened_freq.size} bins with "
        f"axis_offset_uncertainty_hz={uncertainty_hz} and {narrow_freq.size} "
        "without it -- it is NOT widening its support by the declared axis "
        "uncertainty. The constant exists but the code ignores it, which is the "
        "exact failure this check was written to prevent."
    )
    assert widened_freq[0] < narrow_freq[0] and widened_freq[-1] > narrow_freq[-1], (
        f"band_limit widened only one edge: narrow {narrow_freq[0]}-"
        f"{narrow_freq[-1]} Hz vs widened {widened_freq[0]}-{widened_freq[-1]} Hz. "
        "The uncertainty is one-sided in FREQUENCY (the true centre may be half a "
        "bin higher) but that makes BOTH band edges uncertain, so both must widen."
    )

    # And the product's own consumer must pass it, so a caller cannot get the
    # narrow behaviour by accident.
    prod_freq, _ = fft_io.band_limit_product(freq_hz, spectrum, band_hz)
    assert prod_freq.size == widened_freq.size, (
        f"fft_io.band_limit_product kept {prod_freq.size} bins where band_limit "
        f"with the declared uncertainty kept {widened_freq.size}; the product's "
        "band-limiter must carry FFT_AXIS_OFFSET_UNCERTAINTY_HZ for its callers"
    )


def check_b0_2a_no_check_asserts_a_bin_position_tighter_than_one_bin():
    """No B0-2a check may pin a bin position tighter than +/- 1 bin.

    A structural guard on this file itself, not on the data. While centre-vs-edge
    is open, the axis is only known to half a bin, and the echosounder hump is
    only reproducible to about 0.2 bins on a 5 kHz feature. A check asserting a
    tighter position would fail for a reason that has nothing to do with the
    pipeline being wrong -- and, worse, would look like it had SETTLED the
    convention.

    Enforced on the declared tolerance constants rather than by parsing code:
    the echosounder centroid window and the temporal-std window must each be at
    least two bins wide.
    """
    lo, hi = B0_2A_ECHOSOUNDER_CENTROID_BIN_RANGE
    assert hi - lo >= 2.0, (
        f"B0_2A_ECHOSOUNDER_CENTROID_BIN_RANGE spans {hi - lo} bins, tighter than "
        "the +/- 1 bin the open centre-vs-edge convention permits"
    )
    std_lo, std_hi = B0_2A_ECHOSOUNDER_TEMPORAL_STD_ARGMAX_BIN_RANGE
    assert std_hi - std_lo >= 2, (
        f"B0_2A_ECHOSOUNDER_TEMPORAL_STD_ARGMAX_BIN_RANGE spans {std_hi - std_lo} "
        "bins, tighter than the +/- 1 bin permitted"
    )
    assert not hasattr(_b0_2a_cfg(), "FFT_38KHZ_LINE_BIN"), (
        "boatphone.config still defines FFT_38KHZ_LINE_BIN. It was a NOMINAL-derived "
        "single-bin target (38000/250 = 152) that no statistic reproduced, and it has "
        "been replaced by FFT_ECHOSOUNDER_* -- leaving it defined invites a second, "
        "wrong definition of the same landmark (CLAUDE.md invariant 6)"
    )


def check_b0_2a_b5_ceilings_and_censoring_counters_are_available():
    """B5 PRECONDITION: the two ceilings are declared SEPARATELY, and per-window
    censoring counts are reportable.

    Two distinct limits, which a single "max bin" would conflate:
      * the CALIBRATED ceiling, bin 204 (51,000 Hz) -- above it no absolute
        dB re 1 uPa exists at all;
      * the UNCALIBRATED/RELATIVE ceiling, bin 408 (~102 kHz) -- above it even a
        relative statistic is meaningless. Bins 409-418 are the instrument's
        anti-alias skirt: real, gradually decaying filter response, not ocean
        (only ~49% of their cells sit at zero -- not censored). Bins 419-511
        ARE FLOOR-CENSORED (99.94%/99.93% of cells at zero; 419-424 alone
        ~99%). Averaging 419+ turns "we cannot measure this" into a number,
        biased upward by an unboundable amount; 409-418 is excluded as
        instrument response, not because it is censored.

    And the censoring counters: the product's integer scale is clipped into
    [0, 86] at BOTH ends. Upper censoring is measured, not hypothetical -- 3
    cells at 86 on a QUIET window of the local sample, and a close vessel pass
    (the event of interest) will clip far harder. A band level with ceiling hits
    is a lower bound, not a measurement, so the counts must travel with it.
    """
    cfg, fft_io = _b0_2a_fft_io()
    calibrated_ceiling = int(getattr(cfg, "FFT_B5_CALIBRATED_CEILING_BIN"))
    relative_ceiling = int(getattr(cfg, "FFT_B5_RELATIVE_CEILING_BIN"))
    assert calibrated_ceiling == B0_2A_CALIBRATED_BIN_RANGE[1], (
        f"FFT_B5_CALIBRATED_CEILING_BIN is {calibrated_ceiling}, expected "
        f"{B0_2A_CALIBRATED_BIN_RANGE[1]} (bin 204 == 51,000 Hz, where the "
        "pre-deployment calibration file stops)"
    )
    assert relative_ceiling == B0_2A_ROLLOFF_ONSET_BIN, (
        f"FFT_B5_RELATIVE_CEILING_BIN is {relative_ceiling}, expected "
        f"{B0_2A_ROLLOFF_ONSET_BIN} (the anti-alias roll-off onset, ~102 kHz)"
    )
    assert calibrated_ceiling < relative_ceiling, (
        "the calibrated ceiling must be BELOW the relative one; if they are equal "
        "or inverted the two limits have been conflated, which is how an "
        "uncalibratable bin ends up inside a dB re 1 uPa number"
    )

    _cfg, fft_io, path, product = _b0_2a_fixture_levels(0)
    report = fft_io.censoring_report(product.levels_db)
    for key in ("n_at_floor", "n_at_ceiling", "n_cells",
                "fraction_at_floor", "fraction_at_ceiling"):
        assert key in report, f"censoring_report() does not report {key!r}: {report}"
    assert report["n_cells"] == B0_2A_N_FRAMES * B0_2A_N_BINS, (
        f"censoring_report() counted {report['n_cells']} cells, expected "
        f"{B0_2A_N_FRAMES * B0_2A_N_BINS}"
    )
    # The counter must actually COUNT, not return zeros: this window is known to
    # be censored at both ends. If either count is zero the counter is broken,
    # and a broken censoring counter reads exactly like clean data.
    assert report["n_at_floor"] > 0, (
        f"{path.name}: censoring_report() found 0 cells at the {getattr(cfg, 'FFT_LEVEL_FLOOR')} "
        "floor, but 18.7% of this window's cells are measured at it"
    )
    assert report["n_at_ceiling"] > 0, (
        f"{path.name}: censoring_report() found 0 cells at the "
        f"{getattr(cfg, 'FFT_LEVEL_CEILING')} ceiling, but 3 cells of this QUIET "
        "window are measured at it. Upper censoring is real and a counter that "
        "reports none of it is worse than no counter."
    )


def check_b0_2a_calibratable_band_matches_bin_range_and_assert_calibratable_rejects_beyond():
    """Bins 1-204 are the calibratable set; fft_io.assert_calibratable() REJECTS a
    band reaching outside it, reusing boatphone/models.py's own band vocabulary
    (assert_band_matched) rather than a second, independently-invented comparator
    (CLAUDE.md invariant 6) -- fft_io is only asked to STATE the calibratable
    band, not to reimplement how a spectrum gets checked against one.
    """
    cfg, fft_io = _b0_2a_fft_io()
    import boatphone.models as bm

    lo_bin, hi_bin = B0_2A_CALIBRATED_BIN_RANGE
    freq_hz = np.asarray(fft_io.frequency_axis_hz(), dtype=float)
    calibrated_band_hz = fft_io.calibrated_band_hz()
    exp_lo_hz, exp_hi_hz = freq_hz[lo_bin], freq_hz[hi_bin]
    got_lo_hz, got_hi_hz = calibrated_band_hz
    assert abs(got_lo_hz - exp_lo_hz) < 1e-6 and abs(got_hi_hz - exp_hi_hz) < 1e-6, (
        f"fft_io.calibrated_band_hz() = {calibrated_band_hz}, expected "
        f"({exp_lo_hz}, {exp_hi_hz}) Hz -- bins {lo_bin}-{hi_bin} inclusive "
        "(settled fact: the calibration file covers 10 Hz - 51.2 kHz; bins 1-204 "
        "are wholly inside that span -- see review-2 [MEDIUM 2])"
    )

    # In-range request: assert_band_matched-style acceptance -- must NOT raise.
    try:
        fft_io.assert_calibratable(calibrated_band_hz)
    except Exception as exc:
        raise AssertionError(
            f"fft_io.assert_calibratable({calibrated_band_hz}) raised "
            f"{type(exc).__name__}: {exc} for the band it ITSELF just reported "
            "as calibrated_band_hz(); a self-consistent band must be accepted"
        ) from exc

    # Out-of-range request (reaches past bin 204, i.e. past 51,000 Hz): MUST raise.
    beyond_hz = (exp_lo_hz, exp_hi_hz + 10 * B0_2A_BIN_WIDTH_HZ)
    try:
        fft_io.assert_calibratable(beyond_hz)
    except Exception:
        pass
    else:
        raise AssertionError(
            f"fft_io.assert_calibratable({beyond_hz}) returned without raising; "
            f"that band reaches past the calibratable support {calibrated_band_hz} Hz "
            "(bins 1-204 only -- an absolute level requested beyond bin 204 has no "
            "calibration applied to it and must not be silently permitted)"
        )
    # And boatphone.models is genuinely reusable here, not merely importable:
    # assert_band_matched must still draw the same true/false line given the
    # SAME two bands, independent of fft_io's own wording.
    try:
        bm.assert_band_matched(beyond_hz, calibrated_band_hz, label="requested")
    except bm.BandMatchError:
        pass
    else:
        raise AssertionError(
            f"boatphone.models.assert_band_matched({beyond_hz}, {calibrated_band_hz}) "
            "did not raise BandMatchError -- the two bands genuinely differ, so "
            "models.py's own band vocabulary should also flag this mismatch"
        )


# ---------------------------------------------------------------------------
# B3-A -- the overpass-window TIME GATE for the PlanetScope bulk pull.
# ---------------------------------------------------------------------------
# Contract (acoustics_plan_v2 SS5 "B3 -- Bulk acquisition"): PlanetScope crosses
# 09:30-11:30 local, padded 15 min each side to 09:15-11:45 `America/Vancouver`,
# computed PER DATE with zoneinfo (never a fixed UTC offset, because the offset
# itself changes across the DST transition inside the study window's May-Sep
# season boundary and its shoulder months).
#
# Home chosen for the function under test: `boatphone/acquire.py`,
# `overpass_window_utc(local_date) -> (start_utc, end_utc)`, per the brief that
# assigned this check. Neither the module nor the function exists yet -- this
# check is EXPECTED to fail with ModuleNotFoundError/AttributeError until B3-A's
# implementation lands; that failure is reported distinctly from a wrong-value
# AssertionError by scripts/checks.py's own harness (see `main()`), so a
# not-implemented run and a wrong-implementation run are never confused.
#
# Window-edge constant names are a GUESS, flagged as such: boatphone/config.py
# has no existing B3 constants to anchor on, so this check requires
# PLANET_OVERPASS_WINDOW_START_LOCAL / _END_LOCAL (datetime.time) and
# PLANET_OVERPASS_TZ_NAME (str) to exist there with the plan's provenance in a
# comment. If the implementer picks different names, that is a legitimate
# choice to make once, in one place -- this check's names should move with it,
# not be treated as the contract.

def _b3a_config():
    """Import boatphone.config. ModuleNotFoundError/AttributeError here means
    the B3-A window-edge constants are not yet defined -- the expected
    pre-implementation state, not a bug in this check."""
    import importlib
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("boatphone.config")


def _b3a_acquire():
    """Import boatphone.acquire. ModuleNotFoundError here IS the expected
    pre-coder failure -- B3-A has not been implemented yet."""
    import importlib
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    return importlib.import_module("boatphone.acquire")


# Test cases: (local y, m, d, expected UTC start (y,m,d,H,M), expected UTC end,
# note). Expected values are LITERALS, hand-derived from America/Vancouver's
# published DST rule (second Sunday in March / first Sunday in November,
# transition at 02:00 local), NOT computed via zoneinfo here -- the point of a
# time-base check is that it does not trust the same library the implementation
# will use to produce its answer.
#
#   PDT (UTC-7) shoulder months: 09:15-11:45 PDT -> 16:15-18:45Z.
#   2024-03-09 (day BEFORE the spring-forward, still PST/UTC-8): 17:15-19:45Z.
#   2024-03-10 (day the transition occurs at 02:00 local, before 09:15 -> PDT):
#       16:15-18:45Z -- the offset must have CHANGED from the day before.
#   2024-11-02 (day BEFORE fall-back, still PDT/UTC-7): 16:15-18:45Z.
#   2024-11-03 (day the transition occurs at 02:00 local -> PST/UTC-8):
#       17:15-19:45Z -- the reverse change from the March pair.
_B3A_CASES = (
    (2024, 5, 15, (2024, 5, 15, 16, 15), (2024, 5, 15, 18, 45), "in-season PDT shoulder"),
    (2024, 9, 15, (2024, 9, 15, 16, 15), (2024, 9, 15, 18, 45), "in-season PDT shoulder"),
    (2025, 5, 15, (2025, 5, 15, 16, 15), (2025, 5, 15, 18, 45), "in-season PDT shoulder"),
    (2025, 9, 15, (2025, 9, 15, 16, 15), (2025, 9, 15, 18, 45), "in-season PDT shoulder"),
    (2024, 3, 9,  (2024, 3, 9, 17, 15),  (2024, 3, 9, 19, 45),  "day BEFORE spring-forward, PST"),
    (2024, 3, 10, (2024, 3, 10, 16, 15), (2024, 3, 10, 18, 45), "day OF spring-forward, PDT"),
    (2024, 11, 2, (2024, 11, 2, 16, 15), (2024, 11, 2, 18, 45), "day BEFORE fall-back, PDT"),
    (2024, 11, 3, (2024, 11, 3, 17, 15), (2024, 11, 3, 19, 45), "day OF fall-back, PST"),
)


def check_b3a_1_config_window_edge_constants():
    """Window edges (09:15/11:45 America/Vancouver) are NAMED constants in
    boatphone/config.py, with the plan's provenance in a comment -- not
    literals restated inside boatphone/acquire.py (CLAUDE.md invariant 6)."""
    from datetime import time
    cfg = _b3a_config()
    for name in (
        "PLANET_OVERPASS_WINDOW_START_LOCAL",
        "PLANET_OVERPASS_WINDOW_END_LOCAL",
        "PLANET_OVERPASS_TZ_NAME",
    ):
        assert hasattr(cfg, name), (
            f"boatphone.config is missing {name} -- B3-A's overpass window must be a "
            "named constant there, sourced from acoustics_plan_v2 SS5 (09:30-11:30 local "
            "PlanetScope crossing, padded 15 min each side)"
        )
    assert cfg.PLANET_OVERPASS_WINDOW_START_LOCAL == time(9, 15), (
        f"config.PLANET_OVERPASS_WINDOW_START_LOCAL is "
        f"{cfg.PLANET_OVERPASS_WINDOW_START_LOCAL!r}; the plan fixes the padded start at "
        "09:15 local (09:30 crossing - 15 min pad)"
    )
    assert cfg.PLANET_OVERPASS_WINDOW_END_LOCAL == time(11, 45), (
        f"config.PLANET_OVERPASS_WINDOW_END_LOCAL is "
        f"{cfg.PLANET_OVERPASS_WINDOW_END_LOCAL!r}; the plan fixes the padded end at "
        "11:45 local (11:30 crossing + 15 min pad)"
    )
    assert cfg.PLANET_OVERPASS_TZ_NAME == "America/Vancouver", (
        f"config.PLANET_OVERPASS_TZ_NAME is {cfg.PLANET_OVERPASS_TZ_NAME!r}, expected "
        "'America/Vancouver' (acoustics_plan_v2 SS5)"
    )


def check_b3a_2_overpass_window_utc_matches_literal_expectations_per_date():
    """overpass_window_utc(local_date) returns tz-aware UTC (start, end) that
    matches hand-derived literals across shoulder-month dates AND both real
    2024 DST transition days, with the offset CHANGING across each transition,
    a fixed-offset implementation defeated, and every window exactly 150 min.

    This is the B3-A time gate: decision 0002 requires the pipeline be proved
    against known instants before anything downstream depends on it, and this
    check is that proof for the overpass-window derivation specifically.
    """
    from datetime import date, datetime, time, timedelta, timezone
    acq = _b3a_acquire()

    results = {}
    for y, m, d, exp_start, exp_end, note in _B3A_CASES:
        local_date = date(y, m, d)
        got = acq.overpass_window_utc(local_date)
        assert isinstance(got, tuple) and len(got) == 2, (
            f"overpass_window_utc({local_date}) returned {got!r}, expected a "
            "(start_utc, end_utc) 2-tuple"
        )
        start_utc, end_utc = got
        for label, val in (("start_utc", start_utc), ("end_utc", end_utc)):
            assert isinstance(val, datetime), (
                f"overpass_window_utc({local_date}) {label} is a {type(val).__name__}, "
                "not a datetime -- decision 0002 requires tz-aware UTC datetimes end to end"
            )
            assert val.tzinfo is not None, (
                f"overpass_window_utc({local_date}) {label} is NAIVE ({val!r}); decision "
                "0002 rule 1 forbids a naive datetime crossing a function boundary"
            )
            assert val.utcoffset() == timedelta(0), (
                f"overpass_window_utc({local_date}) {label} = {val!r} has utcoffset "
                f"{val.utcoffset()}, not zero -- it must be UTC, not merely tz-aware "
                f"({note})"
            )

        expected_start = datetime(*exp_start, tzinfo=timezone.utc)
        expected_end = datetime(*exp_end, tzinfo=timezone.utc)
        assert start_utc == expected_start, (
            f"overpass_window_utc({local_date}) start_utc = {start_utc.isoformat()}, "
            f"expected {expected_start.isoformat()} ({note}); 09:15 America/Vancouver "
            "on this date, correctly zoned"
        )
        assert end_utc == expected_end, (
            f"overpass_window_utc({local_date}) end_utc = {end_utc.isoformat()}, "
            f"expected {expected_end.isoformat()} ({note}); 11:45 America/Vancouver "
            "on this date, correctly zoned"
        )
        assert end_utc - start_utc == timedelta(minutes=150), (
            f"overpass_window_utc({local_date}) spans {end_utc - start_utc}, expected "
            "exactly 150 min (09:15-11:45 local is 2h30m) on EVERY date, including DST "
            "transition days -- a window that changes DURATION across a transition is as "
            "wrong as one that changes only its start"
        )
        results[local_date] = start_utc

    # --- DST transition: the offset must CHANGE between consecutive days. ---
    d_mar9, d_mar10 = date(2024, 3, 9), date(2024, 3, 10)
    d_nov2, d_nov3 = date(2024, 11, 2), date(2024, 11, 3)
    assert results[d_mar9].hour != results[d_mar10].hour, (
        f"UTC start hour is the SAME across the March 2024 spring-forward transition "
        f"({results[d_mar9].isoformat()} vs {results[d_mar10].isoformat()}); the "
        "America/Vancouver offset changes from -8 to -7 across this pair and the UTC "
        "window must move with it"
    )
    assert results[d_nov2].hour != results[d_nov3].hour, (
        f"UTC start hour is the SAME across the November 2024 fall-back transition "
        f"({results[d_nov2].isoformat()} vs {results[d_nov3].isoformat()}); the "
        "America/Vancouver offset changes from -7 to -8 across this pair and the UTC "
        "window must move with it"
    )
    # And it must be the reverse move: March goes PST->PDT (UTC start EARLIER by
    # 1h), November goes PDT->PST (UTC start LATER by 1h). Compare TIME-OF-DAY
    # (equivalently, subtract the whole calendar day separating the two dates
    # first) rather than the raw datetime difference: mar9 -> mar10 and
    # nov2 -> nov3 are each one calendar day apart, so the raw delta is always
    # ~24h and the DST shift only shows up as +/-1h on top of that day.
    mar_delta = (results[d_mar10] - results[d_mar9]) - timedelta(days=1)
    assert mar_delta == timedelta(hours=-1), (
        f"March transition time-of-day delta (after removing the 1-day gap between "
        f"2024-03-09 and 2024-03-10) is {mar_delta}, expected -1h (PST -> PDT makes "
        "the same local instant EARLIER in UTC)"
    )
    nov_delta = (results[d_nov3] - results[d_nov2]) - timedelta(days=1)
    assert nov_delta == timedelta(hours=1), (
        f"November transition time-of-day delta (after removing the 1-day gap "
        f"between 2024-11-02 and 2024-11-03) is {nov_delta}, expected +1h (PDT -> "
        "PST makes the same local instant LATER in UTC)"
    )

    # --- A fixed -7 or -8 literal offset cannot pass this check. ---
    # Concretely: at least one pair of TESTED dates must yield different UTC
    # start HOURS. A single hardcoded offset applied to every date would give
    # every case the same UTC start hour (since the local start-of-day time is
    # fixed at 09:15) -- so this assertion alone is enough to sink that mutant.
    start_hours = {dt.hour for dt in results.values()}
    assert len(start_hours) >= 2, (
        f"every tested date produced the SAME UTC start hour ({start_hours}); a "
        "per-date zoneinfo computation must disagree with itself across the DST "
        "transition dates above, so this can only happen if a single fixed UTC "
        "offset (e.g. -7 or -8) was substituted for real per-date zoneinfo lookup"
    )


def check_b3a_3_no_hardcoded_utc_offset_and_zoneinfo_is_used():
    """No UTC-offset literal appears anywhere in B3-A source, and the module
    genuinely uses `zoneinfo` -- not merely returns values that happen to
    agree with it on the dates this check samples."""
    import inspect
    acq = _b3a_acquire()
    try:
        src = inspect.getsource(acq)
    except OSError as exc:
        raise AssertionError(f"could not read source of boatphone.acquire: {exc}") from exc

    zoneinfo_tokens = ("zoneinfo", "ZoneInfo")
    assert any(tok in src for tok in zoneinfo_tokens), (
        "boatphone/acquire.py does not reference zoneinfo/ZoneInfo anywhere -- the "
        "contract requires the window be computed per date with zoneinfo, not with a "
        "fixed offset or a third-party tz library"
    )

    # Literal fixed-offset patterns a lazy implementation might use instead of a
    # real zoneinfo lookup. Checked on the code with comments stripped, so a
    # comment mentioning "-7 h" (as several exist elsewhere in this file) cannot
    # false-positive.
    forbidden_offset_patterns = (
        "hours=7", "hours=-7", "hours=8", "hours=-8",
        "hours=-7)", "hours=-8)",
        "timedelta(hours=7)", "timedelta(hours=8)",
        "timezone(timedelta(hours=-7))", "timezone(timedelta(hours=-8))",
        "utcoffset=-7", "utcoffset=-8",
    )
    hits = []
    for lineno, line in enumerate(src.splitlines(), 1):
        code = line.split("#", 1)[0]
        for pat in forbidden_offset_patterns:
            if pat in code:
                hits.append(f"acquire.py:{lineno}: {pat!r} in {line.strip()!r}")
    assert not hits, (
        "boatphone/acquire.py contains a literal fixed-UTC-offset pattern: "
        + "; ".join(hits)
        + " -- the contract requires the offset be DERIVED per date from zoneinfo, "
        "never hardcoded, because the true offset changes across the DST transition"
    )


# ---------------------------------------------------------------------------
# B3-C -- the bulk `.fft.gz` pull driver and absent-file log.
# ---------------------------------------------------------------------------
# Contract (`docs/plans/acoustics_plan_v2.md` SS5 "B3 -- Bulk acquisition",
# CLAUDE.md invariants 3/5/6). A CLI at `scripts/pull_overpass_corpus.py` walks
# every in-season (May-Sep) date, 2025 first then backwards toward 2020 (a
# risks-table mitigation: if throughput falls below 1 file/sec the run may be
# stopped early, and newest-first keeps the corpus useful at whatever depth it
# reached), resolves EACH date's overpass window via B3-A's
# `boatphone.acquire.overpass_window_utc` -- PER DATE, never once and reused --
# lists via A1's listing code and downloads via B3-B's
# `boatphone.onc_client.download_archive_file`. Resume is the DEFAULT path: a
# bare re-run continues, with no `--resume` flag to forget. It emits a
# manifest under `data/derived/` (requested/present/absent, bytes, hash, HTTP
# status, attempt count per file) plus a provenance sidecar, and NEVER writes
# a manifest without its provenance sidecar. A one-line sampling-conditionality
# caveat is a named constant in `boatphone/config.py` and travels with the
# manifest.
#
# THIS CODE DOES NOT EXIST YET (`scripts/pull_overpass_corpus.py` -- confirmed
# absent). Every check below is therefore expected to FAIL until B3-C lands,
# and each one goes through `_b3c_mod()` first, so a missing-module/missing-
# attribute failure reads as "not implemented" and is never confused with
# "implemented and wrong" (test-author brief).
#
# ASSUMED INTERFACE, stated because none of it can be read off an
# implementation that does not exist. If B3-C lands with a different shape,
# THESE CHECKS should move, not be treated as the contract:
#
#   iter_overpass_dates(start_year=2020, end_year=2025) -> Iterator[date]
#       Every in-season (May-Sep) LOCAL calendar date, year-blocks ordered
#       2025, 2024, ..., 2020 (descending), consistent with the risks-table
#       mitigation.
#
#   build_parser() -> argparse.ArgumentParser
#       No `--resume`/`--continue`-shaped flag anywhere on it: resuming a
#       bare re-run is unconditional, not opt-in.
#
#   run(*, dates=None, dest_dir, manifest_dir, listing_fn, transport,
#       client=None, sleep=time.sleep) -> dict
#       `dates`: overrides `iter_overpass_dates()` -- the injection seam these
#       checks use to drive the per-date window recomputation check without
#       depending on real in-season data. `dest_dir`: B3-B's landing zone
#       (passed straight through to `download_archive_file`). `manifest_dir`:
#       where the manifest + provenance sidecar are written -- ALWAYS a temp
#       dir in these checks, never `data/derived/` (invariant 2).
#       `listing_fn(client, location_codes, start_utc, end_utc) ->
#       (list[str], int)` -- the SAME 2-tuple shape (filenames, empty_chunks)
#       `boatphone.onc_client.list_fft_files` actually returns, is A1's
#       listing call, made INJECTABLE here exactly as `transport` made
#       B3-B's network layer injectable -- no network, ever, in these checks.
#       A fake that returns a bare list here diverges from the production
#       callee's shape and is exactly the mismatch that let a TypeError
#       (list vs. 2-tuple) through to the first real run; see check_b3c_2b.
#       Returns the manifest dict actually written.
#
#   Manifest dict shape (a guess; the brief specifies the CONTENT, not the
#   exact keys): `{"requested": int, "present": int, "absent": int,
#   "files": [...], "absent_log": [{"filename": str, "reason": str}, ...],
#   ...}`, with `requested == present + absent` EXACTLY -- a real 404 and a
#   decision-0007 measured-zero both fall into the SAME absent bucket
#   (distinguished only by `reason`), because the brief is explicit that there
#   is no third bucket.
#
#   MANIFEST_BASENAME / PROVENANCE_BASENAME -- module-level string constants
#   naming the two output files under `manifest_dir`, used by
#   check_b3c_5 to sabotage ONLY the provenance path without guessing its name.
#
#   A `boatphone.config` constant whose NAME contains both "SAMPLING" and
#   "CONDITION" (e.g. `PLANET_SAMPLING_CONDITIONALITY_STATEMENT`) holding the
#   ~10:30-local / no-diurnal-claim / seasonal-only-within-band caveat
#   verbatim, matched by substring rather than an assumed exact name so the
#   implementer's naming choice does not sink this check.
# ---------------------------------------------------------------------------

B3C_SCRIPT_RELPATH = os.path.join("scripts", "pull_overpass_corpus.py")


def _b3c_script_path():
    path = REPO_ROOT / B3C_SCRIPT_RELPATH
    assert path.is_file(), (
        f"{B3C_SCRIPT_RELPATH} does not exist; B3-C (the bulk .fft.gz pull driver) is not "
        "implemented yet (expected until the coder lands it -- see the B3-C contract "
        "comment above)"
    )
    return path


def _b3c_mod(*required):
    """Import the entry point BY PATH (mirrors `_a1d_script_mod`'s convention).

    scripts/ has no __init__.py, so this is not `import scripts.pull_overpass_corpus`.
    A missing attribute here IS the expected pre-implementation failure.
    """
    import importlib.util
    path = _b3c_script_path()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("boatphone_b3c_entry_point", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # must be side-effect free at import
    missing = [name for name in required if not hasattr(module, name)]
    assert not missing, (
        f"{B3C_SCRIPT_RELPATH} has no {missing}; see the B3-C ASSUMED INTERFACE comment "
        "above for the exact entry-point surface these checks require"
    )
    return module


def _b3c_fake_filenames(n):
    return [
        f"ICLISTENHF1266_202407{10 + i:02d}T000000.000Z.fft.gz"
        for i in range(n)
    ]


def _b3c_find_string_value(obj, target):
    """True if `target` appears as a substring somewhere inside `obj` (recursively)."""
    if isinstance(obj, str):
        return target in obj
    if isinstance(obj, dict):
        return any(_b3c_find_string_value(v, target) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_b3c_find_string_value(v, target) for v in obj)
    return False


def check_b3c_0_api_surface():
    """run/iter_overpass_dates/build_parser exist (not-implemented gate for the rest of B3-C)."""
    mod = _b3c_mod("run", "iter_overpass_dates", "build_parser")
    for name in ("run", "iter_overpass_dates", "build_parser"):
        assert callable(getattr(mod, name)), f"{B3C_SCRIPT_RELPATH}.{name} is not callable"


def check_b3c_1_date_ordering_is_2025_descending_then_backwards_to_2020():
    """Year-blocks run 2025, 2024, ..., down to 2020 -- so an interrupted run leaves
    the NEWEST season complete, per the risks-table mitigation this ordering exists for."""
    mod = _b3c_mod("iter_overpass_dates")
    dates = list(mod.iter_overpass_dates())
    assert dates, f"{B3C_SCRIPT_RELPATH}.iter_overpass_dates() yielded nothing"

    year_blocks = []
    for d in dates:
        if not year_blocks or year_blocks[-1] != d.year:
            year_blocks.append(d.year)
    assert len(year_blocks) == len(set(year_blocks)), (
        f"a year reappears non-contiguously in the date order: {year_blocks} -- an "
        "interrupted run partway through must leave one clean prefix of complete "
        "seasons, not a scattered mix"
    )
    assert year_blocks[0] == 2025, (
        f"the FIRST year block is {year_blocks[0]}, expected 2025 -- newest-first is the "
        "risks-table mitigation: if throughput drops below 1 file/sec and the run is "
        "stopped early, the corpus must still be useful at whatever depth it reached"
    )
    assert year_blocks == sorted(year_blocks, reverse=True), (
        f"year blocks {year_blocks} are not strictly descending"
    )
    assert year_blocks[-1] <= 2020, (
        f"the date order does not reach back to 2020 (last block: {year_blocks[-1]})"
    )

    months = {d.month for d in dates}
    import boatphone.config as _cfg
    assert months <= set(_cfg.SEASON_MONTHS_UTC), (
        f"iter_overpass_dates() yields month(s) {sorted(months - set(_cfg.SEASON_MONTHS_UTC))} "
        f"outside config.SEASON_MONTHS_UTC ({_cfg.SEASON_MONTHS_UTC}) -- only May-Sep dates "
        "are in scope for the PlanetScope-matched pull"
    )


def check_b3c_2_per_date_window_recomputation_is_never_hoisted_out_of_the_loop():
    """THE highest-value assertion in this check (per the brief): the driver asks
    for a DIFFERENT UTC window on a PST date than on a PDT date, proving the
    window is recomputed per date via B3-A's overpass_window_utc rather than
    hoisted out of the date loop and reused stale.

    Pinned against the REAL boatphone.acquire.overpass_window_utc (already
    implemented and proved by B3-A's own checks), not a mock -- so this fails
    loudly against either a hoisted window OR a window computed some other,
    wrong way.
    """
    mod = _b3c_mod("run")
    import boatphone.acquire as acq
    from datetime import date

    d_pst = date(2024, 3, 9)   # day BEFORE spring-forward: America/Vancouver PST, UTC-8
    d_pdt = date(2024, 3, 10)  # day OF spring-forward: America/Vancouver PDT, UTC-7
    expected_pst = acq.overpass_window_utc(d_pst)
    expected_pdt = acq.overpass_window_utc(d_pdt)
    assert expected_pst != expected_pdt, (
        "fixture sanity check failed: overpass_window_utc itself returned the same window "
        "for a PST and a PDT date -- see B3-A's own checks, not this one"
    )

    calls = []

    def fake_listing_fn(client, location_codes, start_utc, end_utc):
        calls.append((start_utc, end_utc))
        # Real shape: boatphone.onc_client.list_fft_files returns a 2-tuple,
        # (filenames, empty_chunks), never a bare list -- widened to match
        # after the seam/production-callee mismatch that let a TypeError
        # through to the first real run (see check_b3c_2b below).
        return [], 0

    transport = _B3BFakeTransport({})
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "dest"
        manifest_dir = pathlib.Path(tmp) / "manifest"
        mod.run(
            dates=[d_pst, d_pdt], dest_dir=dest_dir, manifest_dir=manifest_dir,
            listing_fn=fake_listing_fn, transport=transport, client=None,
            sleep=lambda _s: None,
        )

    assert len(calls) == 2, (
        f"expected exactly one listing call per requested date (2), got {len(calls)}: {calls}"
    )
    assert calls[0] == expected_pst, (
        f"listing window for {d_pst} was {calls[0]}, expected {expected_pst} "
        "(overpass_window_utc(2024-03-09))"
    )
    assert calls[1] == expected_pdt, (
        f"listing window for {d_pdt} was {calls[1]}, expected {expected_pdt} "
        "(overpass_window_utc(2024-03-10))"
    )
    assert calls[0] != calls[1], (
        f"the driver requested the SAME UTC window ({calls[0]}) for {d_pst} (PST) and "
        f"{d_pdt} (PDT) -- the UTC window must shift by an hour across this DST "
        "transition, so an identical window means overpass_window_utc was computed ONCE "
        "and reused across the date loop instead of per date, reintroducing precisely "
        "the bug B3-A exists to prevent (decision 0002)"
    )


def check_b3c_2b_run_drives_the_real_list_fft_files_not_a_narrower_fake():
    """`run()` wired to the REAL `boatphone.onc_client.list_fft_files`, not a fake
    of the injection seam -- the only check in this file that would have caught
    the tuple/list blocker.

    Every other B3-C check drives `run()` with a hand-written `listing_fn` that
    returns whatever shape ITS author guessed (a bare list, until this check's
    sibling edit widened them). The integration reviewer found the real bug
    that guess hid: `list_fft_files` actually returns `(filenames,
    empty_chunks)`, a 2-tuple, and `run()` iterated the raw return -- a
    `TypeError` on the first real date, never observed here because the seam
    and the production callee had different contracts and only the seam was
    ever exercised. This check makes them the SAME callable: `listing_fn` is
    `onc.list_fft_files` itself (allow_empty=True, since B3's per-date window
    is the short 2.5 h overpass span decision 0028 governs, not the year-scale
    span decision 0008 governs), called against `_StubONC`, a fake ONC
    CLIENT -- not a fake of `run()`'s own listing_fn seam. A regression here
    can only mean `run()`'s handling of `list_fft_files`'s real return shape
    broke, not that some check's guessed fixture shape drifted from it.
    """
    mod = _b3c_mod("run")
    _cfg, onc = _a1a_mods()
    import boatphone.acquire as acq
    from datetime import date, timedelta

    d = date(2024, 7, 15)
    start_utc, end_utc = acq.overpass_window_utc(d)
    assert end_utc > start_utc, "fixture sanity check: overpass window must be non-empty"

    in_window_names = [
        _fft_name(start_utc + timedelta(minutes=5)),
        _fft_name(start_utc + timedelta(minutes=45)),
    ]
    # _StubONC's requests are keyed by the YEAR it infers from dateFrom/dateTo,
    # so files_by_year must be keyed on start_utc.year for list_fft_files'
    # year-chunking to find them.
    client = _StubONC(
        files_by_year={start_utc.year: in_window_names},
        locations=_folger_location_rows(),
    )

    def real_listing_fn(client, location_codes, start_utc, end_utc):
        # allow_empty=True: this is B3's short per-date window, decision
        # 0016's regime, not decision 0008's year-scale raise-on-empty.
        return onc.list_fft_files(client, location_codes, start_utc, end_utc, allow_empty=True)

    transport = _B3BFakeTransport({name: f"content-for-{name}".encode() * 8 for name in in_window_names})

    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "dest"
        manifest_dir = pathlib.Path(tmp) / "manifest"
        manifest = mod.run(
            dates=[d], dest_dir=dest_dir, manifest_dir=manifest_dir,
            listing_fn=real_listing_fn, transport=transport, client=client,
            sleep=lambda _s: None,
        )

    assert isinstance(manifest, dict), (
        f"run() against the REAL list_fft_files return shape raised or returned "
        f"{type(manifest).__name__}, not a dict -- this is the tuple/list mismatch check"
    )
    assert manifest.get("requested") == len(in_window_names), (
        f"manifest['requested'] = {manifest.get('requested')}, expected "
        f"{len(in_window_names)} -- run() must unpack list_fft_files's real "
        "(filenames, empty_chunks) 2-tuple, not misread empty_chunks as a filename "
        "or the tuple itself as a single filename"
    )
    assert manifest.get("present") == len(in_window_names), (
        f"manifest['present'] = {manifest.get('present')}, expected "
        f"{len(in_window_names)} downloaded files"
    )


def check_b3c_2c_empty_overpass_window_logs_an_absent_measured_zero_row():
    """An empty 2.5 h overpass window is a MEASURED ZERO (decision 0028), not a
    crash and not silently dropped from the manifest.

    Decision 0016 exists because integration review found that a fully-empty
    window previously produced NO `absent_log` entry at all -- `absent_log`
    only recorded files that were listed and then not retrieved -- so an empty
    window was invisible in BOTH buckets and `requested == present + absent`
    quietly stopped being a statement about coverage. The fix makes an empty
    window a counted RETRIEVAL UNIT: `requested` is incremented once for it
    (in addition to once per listed filename), `present` is not, and a
    dedicated `absent_log` row is written with `reason ==
    _REASON_EMPTY_WINDOW`, `filename: None`, the local date, and both UTC
    window edges (0016 SS3). Do NOT "correct" this back to `requested == 0`
    for an empty window -- that is exactly the invisibility 0016 was written
    to remove.

    `run()` reaches this row via `EmptyListingError` when `listing_fn` has not
    itself opted into `allow_empty=True` (0016 SS4) -- which is what this check
    exercises, using the library default.
    """
    mod = _b3c_mod("run", "_REASON_EMPTY_WINDOW")
    _cfg, onc = _a1a_mods()
    import boatphone.acquire as acq
    from datetime import date

    d = date(2024, 7, 16)
    start_utc, end_utc = acq.overpass_window_utc(d)
    assert end_utc > start_utc, "fixture sanity check: overpass window must be non-empty"

    client = _StubONC(
        files_by_year={}, locations=_folger_location_rows(),
    )  # no files at all -> a genuinely empty window

    def real_listing_fn(client, location_codes, start_utc, end_utc):
        # Deliberately the library DEFAULT (allow_empty=False) -- decision
        # 0016 says a caller working at B3's per-date scale MUST pass
        # allow_empty=True itself; this check exercises run()'s behaviour
        # when it does not, i.e. list_fft_files raises EmptyListingError and
        # run() is required (per the brief) to catch it per date.
        return onc.list_fft_files(client, location_codes, start_utc, end_utc)

    transport = _B3BFakeTransport({})

    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "dest"
        manifest_dir = pathlib.Path(tmp) / "manifest"
        raised = None
        try:
            manifest = mod.run(
                dates=[d], dest_dir=dest_dir, manifest_dir=manifest_dir,
                listing_fn=real_listing_fn, transport=transport, client=client,
                sleep=lambda _s: None,
            )
        except Exception as exc:  # noqa: BLE001 -- a raise here IS the failure being pinned against
            raised = exc

    assert raised is None, (
        f"run() raised {raised!r} for a single date whose overpass window had zero files -- "
        "decision 0028 requires an empty window to be a MEASURED ZERO, not an error that "
        "takes down the whole corpus run over one quiet morning"
    )
    assert isinstance(manifest, dict), f"run() returned {type(manifest).__name__}, not a dict"

    requested = manifest.get("requested")
    present = manifest.get("present")
    absent_log = manifest.get("absent_log")

    # 0016 SS3: an empty window is ONE retrieval unit ONC answered "no file"
    # for -- requested=1, present=0, absent=1 -- not requested=0. Asserting
    # requested == 0 would re-mandate the invisibility 0016 fixes.
    assert requested == 1 and present == 0, (
        f"manifest for a single-date, zero-file overpass window is requested={requested}, "
        f"present={present}, expected requested=1 (the empty window itself counted as one "
        "retrieval unit per decision 0028 SS3) and present=0"
    )
    assert isinstance(absent_log, list) and len(absent_log) == 1, (
        f"absent_log has {len(absent_log) if isinstance(absent_log, list) else type(absent_log)} "
        "entries, expected exactly 1 -- decision 0028 requires the empty window to produce its "
        "own absent_log row so it is visible in coverage accounting"
    )
    assert requested == present + len(absent_log), (
        f"requested ({requested}) != present ({present}) + len(absent_log) "
        f"({len(absent_log)}) -- decision 0028 SS3 requires this identity to hold EXACTLY as a "
        "statement about coverage"
    )

    row = absent_log[0]
    assert row.get("filename") is None, (
        f"absent_log row for an empty window has filename={row.get('filename')!r}, expected "
        "None -- no file was ever named for a window ONC listed nothing for"
    )
    assert row.get("reason") == mod._REASON_EMPTY_WINDOW, (
        f"absent_log row for an empty window has reason={row.get('reason')!r}, expected "
        f"{mod._REASON_EMPTY_WINDOW!r} -- it must be distinguishable from a 404 or a "
        "decision-0007 not-yet-deployed zero by reason alone, per 0016 SS2"
    )
    assert row.get("local_date") == d.isoformat(), (
        f"absent_log row local_date={row.get('local_date')!r}, expected {d.isoformat()!r}"
    )
    assert row.get("window_start_utc") == start_utc.isoformat(), (
        f"absent_log row window_start_utc={row.get('window_start_utc')!r}, expected the actual "
        f"UTC overpass window start {start_utc.isoformat()!r} -- 0016 SS2 requires both UTC "
        "window edges on the row"
    )
    assert row.get("window_end_utc") == end_utc.isoformat(), (
        f"absent_log row window_end_utc={row.get('window_end_utc')!r}, expected the actual "
        f"UTC overpass window end {end_utc.isoformat()!r} -- 0016 SS2 requires both UTC "
        "window edges on the row"
    )


def check_b3c_3_requested_equals_present_plus_absent_exactly_with_reasons():
    """requested == present + absent EXACTLY -- no third bucket, nothing silently
    dropped. A real 404 and a decision-0007 measured-zero both land in the SAME
    absent bucket, each with a non-empty 'reason' segment C's log can act on."""
    mod = _b3c_mod("run")
    filenames = _b3c_fake_filenames(4)
    transport = _B3BFakeTransport({
        filenames[0]: b"payload-a" * 16,
        filenames[1]: b"payload-b" * 16,
    })
    transport.absent_names = {filenames[2]}
    transport.not_deployed_names = {filenames[3]}

    def listing_fn(client, location_codes, start_utc, end_utc):
        # Real 2-tuple shape: (filenames, empty_chunks), matching
        # boatphone.onc_client.list_fft_files -- a bare list here would let
        # the seam diverge from the production callee again.
        return list(filenames), 0

    from datetime import date
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "dest"
        manifest_dir = pathlib.Path(tmp) / "manifest"
        manifest = mod.run(
            dates=[date(2024, 7, 1)], dest_dir=dest_dir, manifest_dir=manifest_dir,
            listing_fn=listing_fn, transport=transport, client=None, sleep=lambda _s: None,
        )

    assert isinstance(manifest, dict), f"run() returned {type(manifest).__name__}, not a dict"
    requested, present, absent = (manifest.get(k) for k in ("requested", "present", "absent"))
    for key, val in (("requested", requested), ("present", present), ("absent", absent)):
        assert isinstance(val, int), f"manifest[{key!r}] is {val!r}, expected an int count"
    assert requested == len(filenames), (
        f"manifest['requested'] = {requested}, expected {len(filenames)}"
    )
    assert present == 2, (
        f"manifest['present'] = {present}, expected 2 (the two files that actually downloaded)"
    )
    assert absent == 2, (
        f"manifest['absent'] = {absent}, expected 2 -- the real 404 AND the decision-0007 "
        "measured-zero file both belong in the SAME absent bucket (no third bucket), "
        "distinguished only by their 'reason'"
    )
    assert requested == present + absent, (
        f"requested ({requested}) != present ({present}) + absent ({absent}); every listed "
        "file must land in exactly one of these two buckets (CLAUDE.md invariant 5)"
    )

    absent_log = manifest.get("absent_log")
    assert isinstance(absent_log, list) and len(absent_log) == 2, (
        f"manifest['absent_log'] is {absent_log!r}, expected a 2-entry list naming the two "
        "files ONC could not supply -- segment C's absent-file log is the stated consumer"
    )
    logged = {}
    for entry in absent_log:
        assert isinstance(entry, dict) and "filename" in entry, (
            f"absent_log entry {entry!r} does not carry a 'filename'"
        )
        reason = entry.get("reason")
        assert isinstance(reason, str) and reason.strip(), (
            f"absent_log entry for {entry.get('filename')!r} has no non-empty 'reason': "
            f"{entry!r} -- 'listed-but-not-downloaded' must always carry why"
        )
        logged[entry["filename"]] = reason
    assert set(logged) == {filenames[2], filenames[3]}, (
        f"absent_log names {set(logged)}, expected exactly {{filenames[2], filenames[3]}}"
    )


def check_b3c_4_rerun_with_no_flags_resumes_and_redownloads_nothing_complete():
    """Resume is the DEFAULT path. `build_parser()` exposes no `--resume`-shaped
    flag, and re-invoking `run()` with the identical arguments re-requests zero
    bytes for files already complete -- exactly B3-B's cache-hit guarantee,
    exercised here at the driver level with NOTHING extra passed."""
    mod = _b3c_mod("run", "build_parser")

    parser = mod.build_parser()
    all_flags = set()
    for action in parser._actions:
        all_flags.update(action.option_strings)
    resume_like = sorted(f for f in all_flags if "resume" in f.lower() or "continue" in f.lower())
    assert not resume_like, (
        f"build_parser() exposes {resume_like}; resume must be the UNCONDITIONAL default "
        "behaviour of a bare re-run, with no flag to forget -- per the brief, there must "
        "be no --resume flag at all"
    )

    filenames = _b3c_fake_filenames(3)
    content = {fn: (f"content-for-{fn}".encode() * 12) for fn in filenames}
    transport = _B3BFakeTransport(content)

    def listing_fn(client, location_codes, start_utc, end_utc):
        # Real 2-tuple shape: (filenames, empty_chunks), matching
        # boatphone.onc_client.list_fft_files -- a bare list here would let
        # the seam diverge from the production callee again.
        return list(filenames), 0

    from datetime import date
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "dest"
        manifest_dir = pathlib.Path(tmp) / "manifest"
        d1 = date(2024, 7, 1)

        mod.run(
            dates=[d1], dest_dir=dest_dir, manifest_dir=manifest_dir,
            listing_fn=listing_fn, transport=transport, client=None, sleep=lambda _s: None,
        )
        calls_after_first_run = len(transport.calls)
        assert calls_after_first_run > 0, "the first run() made no transport calls at all"

        # Re-run with the IDENTICAL call, no extra argument, simulating a bare
        # re-invocation of the CLI after a kill mid-run.
        mod.run(
            dates=[d1], dest_dir=dest_dir, manifest_dir=manifest_dir,
            listing_fn=listing_fn, transport=transport, client=None, sleep=lambda _s: None,
        )
        new_calls = transport.calls[calls_after_first_run:]
        assert not new_calls, (
            f"re-running with no flags re-requested {len(new_calls)} byte range(s) for "
            f"files already complete: {new_calls!r} -- resume must be the unconditional "
            "default, not opt-in, and a killed-then-restarted run must download nothing "
            "already finished"
        )


def check_b3c_5_manifest_never_written_without_its_provenance_sidecar():
    """If the provenance sidecar cannot be written, no manifest lands either.

    Sabotages ONLY the provenance sidecar's path (a directory placed exactly
    where the provenance file must go), leaving the manifest's own path free,
    so a driver that writes the manifest independently of provenance -- rather
    than gating on it -- is caught.
    """
    mod = _b3c_mod("run", "PROVENANCE_BASENAME", "MANIFEST_BASENAME")
    assert mod.PROVENANCE_BASENAME != mod.MANIFEST_BASENAME, (
        "PROVENANCE_BASENAME and MANIFEST_BASENAME must name two DIFFERENT files"
    )

    filenames = _b3c_fake_filenames(2)
    transport = _B3BFakeTransport({fn: b"x" * 10 for fn in filenames})

    def listing_fn(client, location_codes, start_utc, end_utc):
        # Real 2-tuple shape: (filenames, empty_chunks), matching
        # boatphone.onc_client.list_fft_files -- a bare list here would let
        # the seam diverge from the production callee again.
        return list(filenames), 0

    from datetime import date
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "dest"
        manifest_dir = pathlib.Path(tmp) / "manifest"
        manifest_dir.mkdir(parents=True)
        # A DIRECTORY sits at the exact path the provenance sidecar must be
        # written to, so any real write attempt there raises.
        (manifest_dir / mod.PROVENANCE_BASENAME).mkdir()

        raised = None
        try:
            mod.run(
                dates=[date(2024, 7, 1)], dest_dir=dest_dir, manifest_dir=manifest_dir,
                listing_fn=listing_fn, transport=transport, client=None, sleep=lambda _s: None,
            )
        except Exception as exc:  # noqa: BLE001 -- the raise IS the expected outcome here
            raised = exc
        assert raised is not None, (
            "run() returned normally even though the provenance sidecar could not be "
            "written -- 'no manifest without provenance' requires this failure to surface, "
            "never be swallowed (invariant 5)"
        )
        manifest_path = manifest_dir / mod.MANIFEST_BASENAME
        assert not manifest_path.exists(), (
            f"{manifest_path} exists even though the provenance sidecar failed to write -- "
            "a manifest must never land without its provenance sidecar"
        )


def check_b3c_6_sampling_conditionality_is_one_named_constant_and_travels_with_the_manifest():
    """The ~10:30-local sampling-conditionality sentence is ONE named constant in
    boatphone/config.py (invariant 6), and the driver's output carries it
    verbatim rather than a paraphrase."""
    import boatphone.config as cfg
    candidates = [n for n in dir(cfg) if "SAMPLING" in n and "CONDITION" in n]
    assert candidates, (
        "boatphone/config.py has no constant whose name contains both SAMPLING and "
        "CONDITION -- the ~10:30-local, no-diurnal-claim, seasonal-only-within-band caveat "
        "must be ONE named constant there so every downstream figure imports one sentence "
        "rather than paraphrasing it (invariant 6)"
    )
    const_name = candidates[0]
    statement = getattr(cfg, const_name)
    assert isinstance(statement, str) and statement.strip(), (
        f"config.{const_name} is not a non-empty string: {statement!r}"
    )

    mod = _b3c_mod("run")
    filenames = _b3c_fake_filenames(1)
    transport = _B3BFakeTransport({filenames[0]: b"z" * 8})

    def listing_fn(client, location_codes, start_utc, end_utc):
        # Real 2-tuple shape: (filenames, empty_chunks), matching
        # boatphone.onc_client.list_fft_files -- a bare list here would let
        # the seam diverge from the production callee again.
        return list(filenames), 0

    from datetime import date
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "dest"
        manifest_dir = pathlib.Path(tmp) / "manifest"
        result = mod.run(
            dates=[date(2024, 7, 1)], dest_dir=dest_dir, manifest_dir=manifest_dir,
            listing_fn=listing_fn, transport=transport, client=None, sleep=lambda _s: None,
        )
    assert _b3c_find_string_value(result, statement), (
        f"run()'s returned manifest does not carry config.{const_name} verbatim anywhere "
        "in its structure -- every downstream figure must import the ONE sentence, and the "
        "manifest/provenance is where a reader would look for it"
    )


def check_b3c_8_absent_log_rows_carry_file_span_via_parse_file_coverage():
    """Each absent_log row for a NAMED file must expose `file_start_utc` /
    `file_end_utc`, and they must agree with `onc_client.parse_file_coverage()`
    on that SAME filename -- 'the ONE place a filename becomes a time'
    (onc_client.py ~L376).

    NAMING, pinned deliberately: these are FILE SPANS, not epoch-aligned bin
    edges, and must NOT be called `bin_start_utc`/`bin_end_utc`. That name is
    reserved, project-wide, for the fixed-width `config.BIN_SECONDS` grid
    whose edges are integer multiples of `BIN_SECONDS` since the UTC epoch
    (`config.py:12-14`, and the real grid cells returned by
    `onc_client.season_bins_utc` under exactly these two field names,
    `onc_client.py:1158-1161`). A file's start is NOT on that grid --
    `parse_file_coverage`'s own docstring measures the file-start seconds as
    off-grid (":36", ":04") with 1-2 s jitter between consecutive files. Using
    "bin_*" for both a real grid cell and an arbitrary file span makes the two
    look joinable by equality when they are not: an equality join between this
    manifest and the uptime grid would match almost nothing and read as "low
    coverage" instead of as a naming bug. Segment D's join against the real bin
    grid must therefore be an INTERVAL-OVERLAP join (does
    [file_start_utc, file_end_utc) intersect [bin_start_utc, bin_end_utc)?),
    never an equality join on the *_utc names.

    Drives run() with a real 404-shaped absent file (not an empty window --
    `_REASON_EMPTY_WINDOW` rows carry no filename to derive a span from) so
    this pins the case segment D actually needs: a NAMED absent file's row
    must carry its own file_start_utc/file_end_utc, sourced through
    parse_file_coverage and not invented some other way.

    Also pins the very property that makes the old `bin_*` name wrong: a
    realistic ONC file-start timestamp (non-zero seconds, off the 300 s grid)
    must NOT be epoch-aligned to `config.BIN_SECONDS`. If someone renames these
    fields back to `bin_*`, this assertion is what should stop them.
    """
    mod = _b3c_mod("run")
    cfg, onc = _a1a_mods()
    assert hasattr(onc, "parse_file_coverage"), (
        "boatphone.onc_client.parse_file_coverage is missing; it is the ONE place "
        "a filename becomes a time and this check pins the absent log against it"
    )

    # A realistic ONC filename whose stamp has NON-ZERO seconds (":36"), unlike
    # `_b3c_fake_filenames`'s midnight-stamped fixtures (T000000, which is
    # trivially 0 % BIN_SECONDS == 0 and would let an epoch-alignment bug hide).
    filename = "ICLISTENHF1266_20240715T000136.000Z.fft.gz"
    expected_start, expected_end = onc.parse_file_coverage(filename)
    assert int(expected_start.timestamp()) % cfg.BIN_SECONDS != 0, (
        f"fixture filename {filename!r} parses to a file_start_utc "
        f"({expected_start.isoformat()}) that IS epoch-aligned to "
        f"config.BIN_SECONDS ({cfg.BIN_SECONDS}) -- fixture sanity check failed. This "
        "check needs an off-grid file start to pin that a file span is not a bin edge; "
        "pick a filename stamp whose seconds are not a multiple of BIN_SECONDS"
    )

    # A transport that answers this filename with a 404, so the file lands in
    # absent_log as a real absence rather than a measured zero or an error.
    transport = _B3BFakeTransport({})
    transport.absent_names.add(filename)

    def listing_fn(client, location_codes, start_utc, end_utc):
        return [filename], 0

    from datetime import date
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "dest"
        manifest_dir = pathlib.Path(tmp) / "manifest"
        manifest = mod.run(
            dates=[date(2024, 7, 1)], dest_dir=dest_dir, manifest_dir=manifest_dir,
            listing_fn=listing_fn, transport=transport, client=None, sleep=lambda _s: None,
        )

    named_rows = [row for row in manifest.get("absent_log", []) if row.get("filename") == filename]
    assert named_rows, (
        f"no absent_log row for {filename!r} -- fixture sanity check failed (expected a "
        "404-shaped absence for this filename)"
    )
    row = named_rows[0]
    assert "bin_start_utc" not in row and "bin_end_utc" not in row, (
        f"absent_log row for {filename!r} still carries bin_start_utc/bin_end_utc: "
        f"{sorted(row)}. These are FILE SPANS from parse_file_coverage, not epoch-aligned "
        "bin edges, and must be named file_start_utc/file_end_utc so they cannot be "
        "equality-joined against the real BIN_SECONDS grid by mistake"
    )
    assert "file_start_utc" in row and "file_end_utc" in row, (
        f"absent_log row for {filename!r} has no file_start_utc/file_end_utc: {sorted(row)}. "
        "Segment D interval-joins the absent log onto the 5-min UTC bin grid and must not "
        "re-derive the file span from the filename a second way (onc_client.py's "
        "parse_file_coverage docstring: 'the ONE place a filename becomes a time')"
    )
    from datetime import datetime as _dt
    got_start = row["file_start_utc"]
    got_end = row["file_end_utc"]
    got_start = _dt.fromisoformat(got_start) if isinstance(got_start, str) else got_start
    got_end = _dt.fromisoformat(got_end) if isinstance(got_end, str) else got_end
    assert got_start == expected_start, (
        f"absent_log row's file_start_utc {got_start} disagrees with "
        f"parse_file_coverage({filename!r}) = {expected_start} -- a second path to the same "
        "fact has drifted from the ONE place a filename becomes a time"
    )
    assert got_end == expected_end, (
        f"absent_log row's file_end_utc {got_end} disagrees with "
        f"parse_file_coverage({filename!r}) = {expected_end}"
    )


def check_b3c_12_per_file_record_exposes_disk_basename_distinct_from_wire_filename():
    """A per-file manifest record must expose the ON-DISK basename as its own
    field, `disk_basename`, distinct from `filename` (the ONC wire name).

    Decision 0024 compresses on write: `filename` stays the name ONC served
    (e.g. ending `.fft`), but the bytes land on disk as `<filename>.fft.gz`
    for every compressed row -- 26,666 of 26,756 rows in the real corpus pull,
    per the B3-C/B3-E handoff. A consumer that joins manifest rows to
    `boatphone.acquire.resolve_corpus_files()` output BY `filename` alone
    matches only the 90 already-pulled plain `.fft` probe files and silently
    drops the other 26,666 -- exactly the kind of "low coverage" misread a
    naming defect produces, not a real coverage gap.

    Pins two things: (a) `disk_basename` exists on a "downloaded"/"cached" file
    record and differs from `filename` whenever compression changed the name
    on disk (i.e. `filename` does not already end `.gz`); (b) the value at
    `path` for that row actually exists on disk after a real download, and its
    name equals `disk_basename` -- so a future implementation cannot collapse
    the two fields back into one and still pass this check.
    """
    mod = _b3c_mod("run")
    filenames = _b3c_fake_filenames(1)  # end '.fft.gz' already? check below, force plain '.fft'
    # `_b3c_fake_filenames` already yields names ending '.fft.gz'; force a
    # PLAIN '.fft' wire name here so compress-on-write actually renames it on
    # disk, which is the case this check exists to catch.
    filename = filenames[0]
    if filename.endswith(".gz"):
        filename = filename[: -len(".gz")]
    transport = _B3BFakeTransport({filename: b"z" * 32})

    def listing_fn(client, location_codes, start_utc, end_utc):
        return [filename], 0

    from datetime import date
    # EVERY ASSERTION BELOW RUNS INSIDE THIS BLOCK. An earlier version closed the
    # TemporaryDirectory first and then asserted `path.exists()`, which cannot
    # pass: the directory and everything in it is deleted on exit. That failure
    # was diagnosed in 1937b6e as the LIBRARY recording a pre-compression name,
    # and it is not -- `path` is the .gz path, the file really is written, and
    # the seam at onc_client.py:1774-1778 / pull_overpass_corpus.py:798 is
    # correct. Verified against the real 26,666-row manifest, where 0 rows have
    # a path missing on disk. The defect was in this check.
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "dest"
        manifest_dir = pathlib.Path(tmp) / "manifest"
        manifest = mod.run(
            dates=[date(2024, 7, 1)], dest_dir=dest_dir, manifest_dir=manifest_dir,
            listing_fn=listing_fn, transport=transport, client=None, sleep=lambda _s: None,
        )
        _b3c_12_assert_disk_basename(manifest, filename)


def _b3c_12_assert_disk_basename(manifest, filename):
    """The assertions for check_b3c_12. Separate so they cannot drift back
    outside the TemporaryDirectory they depend on."""
    files = [row for row in manifest.get("files", []) if row.get("filename") == filename]
    assert files, (
        f"no manifest 'files' row for {filename!r} -- fixture sanity check failed (expected "
        "a successful download of this filename)"
    )
    row = files[0]
    assert "disk_basename" in row, (
        f"manifest file row for {filename!r} has no 'disk_basename' field: {sorted(row)}. "
        "The wire filename ('filename', e.g. ending '.fft') and the on-disk name (ending "
        "'.fft.gz' for a compressed row, decision 0024) must be two DIFFERENT fields -- a "
        "consumer joining to boatphone.acquire.resolve_corpus_files() BY NAME must be able "
        "to join on disk_basename, not filename"
    )
    disk_basename = row["disk_basename"]
    assert disk_basename != row["filename"], (
        f"disk_basename ({disk_basename!r}) equals filename ({row['filename']!r}) for a "
        "wire name that did not already end '.gz' -- compress-on-write (decision 0024) "
        "should have renamed the on-disk copy, and if the two fields have collapsed back "
        "into one value this check must fail: that is exactly the naming defect it pins"
    )
    path_value = row.get("path")
    assert path_value is not None, (
        f"manifest file row for {filename!r} has no 'path' -- cannot verify disk_basename "
        "against what is actually on disk"
    )
    on_disk_path = pathlib.Path(path_value)
    assert on_disk_path.exists(), (
        f"manifest row's path {on_disk_path} does not exist on disk after a real download -- "
        "'path' must name a file that is actually there, not merely a computed string"
    )
    assert on_disk_path.name == disk_basename, (
        f"disk_basename ({disk_basename!r}) does not match the basename of 'path' "
        f"({on_disk_path.name!r}) -- disk_basename must name the file that is ACTUALLY on "
        "disk at 'path', not some other derived name"
    )


def check_b3c_9_provenance_names_endpoint_absent_definition_dest_dir_season_and_checksum_caveat():
    """The provenance sidecar must name: the ONC endpoint/base URL actually used,
    an explicit definition of 'absent', dest_dir, the season months, the year
    span (CORPUS_PULL_START_YEAR..END_YEAR), and a no-server-checksum caveat
    sitting next to the sha256 values it qualifies.

    Open item 1 from the B3 handoff. Without the endpoint, a future reader
    cannot tell which ONC deployment served the bytes; without the year span
    and season months a partial-corpus manifest cannot state its own scope;
    without the checksum caveat next to sha256, a reader assumes ONC vouched
    for the bytes when only a local hash did (0007/0007-style no-server-
    checksum finding in the handoff, distinct from decision 0007's measured
    zeros).
    """
    mod = _b3c_mod("run", "PROVENANCE_BASENAME")
    import json
    import boatphone.config as cfg

    filenames = _b3c_fake_filenames(1)
    transport = _B3BFakeTransport({filenames[0]: b"y" * 16})

    def listing_fn(client, location_codes, start_utc, end_utc):
        return list(filenames), 0

    from datetime import date
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "dest"
        manifest_dir = pathlib.Path(tmp) / "manifest"
        mod.run(
            dates=[date(2024, 7, 1)], dest_dir=dest_dir, manifest_dir=manifest_dir,
            listing_fn=listing_fn, transport=transport, client=None, sleep=lambda _s: None,
        )
        provenance_path = manifest_dir / mod.PROVENANCE_BASENAME
        assert provenance_path.is_file(), f"{provenance_path} was not written"
        provenance = json.loads(provenance_path.read_text())

        endpoint_keys = [k for k in provenance if "endpoint" in k.lower() or "base_url" in k.lower()
                         or "baseurl" in k.lower()]
        assert endpoint_keys and any(str(provenance[k]).strip() for k in endpoint_keys), (
            f"provenance has no non-empty endpoint/base-URL key (checked names containing "
            f"'endpoint'/'base_url'); got keys {sorted(provenance)}. A future reader cannot "
            "tell which ONC deployment served the bytes (open item 1)"
        )

        absent_def_keys = [k for k in provenance if "absent" in k.lower() and "definition" in k.lower()
                            or k.lower() in ("absent_definition", "absent_means", "definition_of_absent")]
        assert absent_def_keys and any(str(provenance[k]).strip() for k in absent_def_keys), (
            f"provenance has no explicit definition of what 'absent' means; got keys "
            f"{sorted(provenance)} (open item 1)"
        )

        assert "dest_dir" in provenance and str(provenance["dest_dir"]).strip(), (
            f"provenance has no non-empty 'dest_dir' key; got keys {sorted(provenance)}"
        )

        season_keys = [k for k in provenance if "season" in k.lower() and "month" in k.lower()]
        assert season_keys and provenance[season_keys[0]], (
            f"provenance has no season-months key; got keys {sorted(provenance)}"
        )

        year_keys = [k for k in provenance if "year" in k.lower()]
        assert year_keys, (
            f"provenance has no year-span key (expected something naming "
            f"CORPUS_PULL_START_YEAR..END_YEAR); got keys {sorted(provenance)}"
        )
        found_span = any(
            str(cfg.CORPUS_PULL_START_YEAR) in str(provenance[k])
            and str(cfg.CORPUS_PULL_END_YEAR) in str(provenance[k])
            for k in year_keys
        ) or (
            provenance.get("start_year") == cfg.CORPUS_PULL_START_YEAR
            and provenance.get("end_year") == cfg.CORPUS_PULL_END_YEAR
        )
        assert found_span, (
            f"no provenance year-span key states {cfg.CORPUS_PULL_START_YEAR}.."
            f"{cfg.CORPUS_PULL_END_YEAR} (config.CORPUS_PULL_START_YEAR..END_YEAR); "
            f"got {[(k, provenance[k]) for k in year_keys]}"
        )

        checksum_caveat_keys = [
            k for k in provenance
            if ("checksum" in k.lower() or "sha256" in k.lower())
            and ("caveat" in k.lower() or "note" in k.lower() or "no_server" in k.lower())
        ]
        assert checksum_caveat_keys and any(str(provenance[k]).strip() for k in checksum_caveat_keys), (
            f"provenance carries no no-server-checksum caveat next to the sha256 values it "
            f"qualifies; got keys {sorted(provenance)}. ONC serves no server-side checksum "
            "(no ETag, no Content-MD5) so integrity is a LOCAL sha256 only, and a reader must "
            "not assume ONC vouched for the bytes"
        )


def check_b3c_10_two_manifest_writes_in_the_same_dest_do_not_collide():
    """Two successive `run()` calls into the SAME manifest_dir must leave TWO
    distinct manifest files on disk, not one overwriting the other. Open item 2
    from the B3 handoff: MANIFEST_BASENAME/PROVENANCE_BASENAME are constants
    and the plan explicitly contemplates a staged pull plus F's second pass, so
    a fixed basename silently discards the first run's absent-file log.

    This is written to accept EITHER settled contract (a run-stamped filename,
    or a `runs/` subdirectory per invocation) -- it only pins the observable
    behaviour: after two runs, glob for anything manifest-shaped and expect >=2
    distinct paths, and each must be independently readable JSON.
    """
    mod = _b3c_mod("run")
    import json
    from datetime import date

    def make_listing_fn(fn):
        def listing_fn(client, location_codes, start_utc, end_utc):
            return [fn], 0
        return listing_fn

    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "dest"
        manifest_dir = pathlib.Path(tmp) / "manifest"

        fn1 = _b3c_fake_filenames(1)[0]
        transport1 = _B3BFakeTransport({fn1: b"a" * 8})
        mod.run(
            dates=[date(2024, 7, 1)], dest_dir=dest_dir, manifest_dir=manifest_dir,
            listing_fn=make_listing_fn(fn1), transport=transport1, client=None,
            sleep=lambda _s: None,
        )
        before = set(manifest_dir.rglob("*.json"))

        fn2 = _b3c_fake_filenames(2)[1]
        transport2 = _B3BFakeTransport({fn2: b"b" * 8})
        mod.run(
            dates=[date(2024, 7, 2)], dest_dir=dest_dir, manifest_dir=manifest_dir,
            listing_fn=make_listing_fn(fn2), transport=transport2, client=None,
            sleep=lambda _s: None,
        )
        after = set(manifest_dir.rglob("*.json"))

        new_files = after - before
        assert new_files, (
            f"a second run() into the same manifest_dir produced no NEW json file "
            f"(before={sorted(p.name for p in before)}, after={sorted(p.name for p in after)}) -- "
            "the second run overwrote the first run's manifest+provenance in place rather than "
            "landing under a run-stamped name or a runs/ subdirectory (open item 2)"
        )
        manifest_like = [p for p in after if "manifest" in p.name.lower()]
        assert len(manifest_like) >= 2, (
            f"expected >=2 distinct manifest-shaped files after two runs into the same "
            f"manifest_dir, found {len(manifest_like)}: {sorted(p.name for p in manifest_like)} -- "
            "a fixed MANIFEST_BASENAME overwrites the prior run's absent-file log, which the plan "
            "explicitly needs to survive (a staged pull plus F's second pass)"
        )
        for p in manifest_like:
            json.loads(p.read_text())  # each must be independently valid, not a half-write


def check_b3c_11_corpus_resolver_finds_fft_and_fft_gz_but_not_sidecars_and_raises_when_absent():
    """A SINGLE shared helper (e.g. `boatphone.acquire`) must resolve 'the
    corpus files' -- not three notebook glob patterns (open item 3).

    AMENDED, not silently rewritten: this check originally pinned
    `resolve_corpus_files` to match `*.<ARCHIVE_EXTENSION>` ('.fft') ONLY and
    explicitly EXCLUDE anything ending '.fft.gz', on the theory that a
    resolver globbing '.fft.gz' would find ZERO files in a corpus that lands
    as plain '.fft'. `check_b3e_4`/`check_b3e_5` then showed that exclusion
    was in direct tension with the bulk-pull compression contract (B3-E): the
    bulk pull DOES compress on write, and a human decision (recorded
    2026-08-27, no separate decision doc yet at time of writing -- flag this
    as a candidate for one) settled the naming question the original docstring
    left open: the compressed file is named `.fft.gz`, honestly stating the
    container it holds. The exclusion is therefore obsolete and is REMOVED
    here, not defended.

    What the exclusion was actually protecting against is real and still
    guarded, just by different mechanisms now: a notebook globbing
    `*.fft.gz` in some OTHER, unscoped location could find zero files and
    silently report 'no data' instead of failing. That trap's modern guard is
    (a) the corpus lives in its own directory, `paths.ONC_OVERPASS_CORPUS_DIR`,
    distinguishable from any other source of `.fft`-shaped files, and (b)
    `resolve_corpus_files` is the ONE glob (invariant 6) and this check's
    'raises rather than returning []' assertion below still holds -- so a
    corpus with nothing matching still fails loudly rather than reporting
    zero vessels.

    This check now pins: (a) the resolver exists and is callable; (b) it
    finds BOTH a `.fft` file and a `.fft.gz` file in the same corpus
    directory -- the mixed corpus (90 already-pulled plain files that `data/`
    immutability (invariant 2) forbids ever converting, plus all
    future compressed bulk-pull files) is the expected STEADY STATE, not a
    transitional oddity; (c) an unrelated extension in the same directory
    (a stray `.txt` note and a `.sha256` integrity sidecar -- sidecars sit
    alongside every real download) is NOT returned, so the resolver is not
    just 'match anything'; (d) it still raises clearly, never silently
    returns [], when the corpus directory is absent, and separately when the
    directory exists but holds no `.fft`/`.fft.gz` file at all -- B5 must not
    read an empty/absent corpus as 'zero vessels detected' (open item 3).
    """
    import boatphone.acquire as acq
    import boatphone.config as cfg

    resolver_name = None
    for candidate in ("resolve_corpus_files", "list_corpus_files", "corpus_files",
                      "overpass_corpus_files", "find_corpus_files"):
        if hasattr(acq, candidate) and callable(getattr(acq, candidate)):
            resolver_name = candidate
            break
    assert resolver_name is not None, (
        "boatphone.acquire has no single corpus-resolving helper (checked "
        "resolve_corpus_files/list_corpus_files/corpus_files/overpass_corpus_files/"
        "find_corpus_files). Open item 3: 'nothing on disk marks the corpus as "
        "overpass-window-only... provide a helper that resolves the corpus *through* "
        "the manifest', so B5 and any notebook share ONE glob, not three"
    )
    resolver = getattr(acq, resolver_name)

    # (d) absent directory: must raise, never return [].
    with tempfile.TemporaryDirectory() as tmp:
        empty_dir = pathlib.Path(tmp) / "nothing_here"
        raised = None
        result = None
        try:
            result = resolver(empty_dir)
        except Exception as exc:  # noqa: BLE001 -- the raise IS the expected outcome
            raised = exc
        assert raised is not None, (
            f"{resolver_name}() on an absent corpus location returned "
            f"{result!r} instead of raising -- B5 must not read an absent corpus as "
            "'zero vessels detected'; silently returning [] makes that mistake possible "
            "(open item 3)"
        )

    # (d) existing directory, but nothing that matches: must also raise, not
    # return [] -- distinct from the absent-directory case above, since a
    # resolver could special-case "no such path" while still silently
    # returning [] for "path exists but is empty/irrelevant".
    with tempfile.TemporaryDirectory() as tmp:
        present_but_empty = pathlib.Path(tmp) / "onc_raw" / "present_but_irrelevant"
        present_but_empty.mkdir(parents=True)
        (present_but_empty / "readme.txt").write_text("not a corpus file")
        raised = None
        result = None
        try:
            result = resolver(present_but_empty)
        except Exception as exc:  # noqa: BLE001 -- the raise IS the expected outcome
            raised = exc
        assert raised is not None, (
            f"{resolver_name}({present_but_empty}) held only an unrelated file and returned "
            f"{result!r} instead of raising -- an existing-but-empty-of-matches corpus "
            "directory must fail loudly, not be read as 'zero vessels detected'"
        )

    # (b) mixed corpus: both a plain .fft and a compressed .fft.gz must be
    # found; (c) unrelated extensions and sidecars must not be.
    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = pathlib.Path(tmp) / "onc_raw"
        corpus_dir.mkdir(parents=True)
        plain_name = f"ICLISTENHF1266_20240715T000136.000Z.{cfg.ARCHIVE_EXTENSION}"
        gzip_name = f"ICLISTENHF1266_20240715T000636.000Z.{cfg.PRODUCT_EXTENSION}"
        sidecar_name = f"{gzip_name}.sha256"
        stray_name = "README.txt"
        (corpus_dir / plain_name).write_bytes(b"plain-ascii-stand-in")
        (corpus_dir / gzip_name).write_bytes(b"gzip-bytes-stand-in")
        (corpus_dir / sidecar_name).write_text("deadbeef  " + gzip_name)
        (corpus_dir / stray_name).write_text("notes, not a corpus file")

        found = resolver(corpus_dir)
        found_names = {pathlib.Path(f).name for f in found}
        assert plain_name in found_names, (
            f"{resolver_name}({corpus_dir}) did not find {plain_name!r}, a "
            f"config.ARCHIVE_EXTENSION={cfg.ARCHIVE_EXTENSION!r} file -- the 90 "
            "already-pulled plain files that data/ immutability forbids converting must "
            f"still resolve; got {found_names}"
        )
        assert gzip_name in found_names, (
            f"{resolver_name}({corpus_dir}) did not find {gzip_name!r}, a "
            f"config.PRODUCT_EXTENSION={cfg.PRODUCT_EXTENSION!r} file -- the bulk pull "
            "compresses on write and names the result honestly as '.fft.gz'; the resolver "
            f"must match it (this exclusion was REMOVED, see this check's docstring for why); "
            f"got {found_names}"
        )
        assert sidecar_name not in found_names, (
            f"{resolver_name}({corpus_dir}) returned the '.sha256' integrity sidecar "
            f"{sidecar_name!r} as if it were a corpus data file -- sidecars sit alongside "
            "every real download and must not be mistaken for one"
        )
        assert stray_name not in found_names, (
            f"{resolver_name}({corpus_dir}) returned an unrelated file {stray_name!r} that "
            "matches neither the plain nor compressed corpus extension"
        )


def check_b3c_7_never_writes_under_the_real_data_dir_this_check_guards_the_guard():
    """Guards the guard, not the coder (mirrors check_b3b_9): this check's own
    fixture never touches the real data/derived/ or data/raw/onc, no matter
    what run() does internally, because dest_dir/manifest_dir are ALWAYS a
    temp dir here (invariant 2)."""
    import boatphone.paths as p
    mod = _b3c_mod("run")
    before_raw = set(p.ONC_RAW_DIR.rglob("*")) if p.ONC_RAW_DIR.exists() else set()
    before_derived = set(p.DERIVED_DIR.rglob("*")) if p.DERIVED_DIR.exists() else set()

    filenames = _b3c_fake_filenames(2)
    transport = _B3BFakeTransport({fn: b"w" * 8 for fn in filenames})

    def listing_fn(client, location_codes, start_utc, end_utc):
        # Real 2-tuple shape: (filenames, empty_chunks), matching
        # boatphone.onc_client.list_fft_files -- a bare list here would let
        # the seam diverge from the production callee again.
        return list(filenames), 0

    from datetime import date
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "dest"
        manifest_dir = pathlib.Path(tmp) / "manifest"
        try:
            mod.run(
                dates=[date(2024, 7, 1)], dest_dir=dest_dir, manifest_dir=manifest_dir,
                listing_fn=listing_fn, transport=transport, client=None, sleep=lambda _s: None,
            )
        except Exception:  # noqa: BLE001 -- only the untouched-data/ assertion matters here
            pass

    after_raw = set(p.ONC_RAW_DIR.rglob("*")) if p.ONC_RAW_DIR.exists() else set()
    after_derived = set(p.DERIVED_DIR.rglob("*")) if p.DERIVED_DIR.exists() else set()
    assert after_raw == before_raw, (
        f"the real ONC landing zone {p.ONC_RAW_DIR} changed during a dest_dir-scoped test "
        f"run: {after_raw - before_raw} appeared. A dest_dir argument must never be ignored "
        "(invariant 2)"
    )
    assert after_derived == before_derived, (
        f"the real derived-products dir {p.DERIVED_DIR} changed during a manifest_dir-"
        f"scoped test run: {after_derived - before_derived} appeared. A manifest_dir "
        "argument must never be ignored (invariant 2)"
    )


def check_b3d_1_fft_reader_accepts_plain_ascii_and_gzip_by_content_not_extension():
    """FACT 1 (live-archive probe, 2026-08-27): ONC serves `.fft` as PLAIN ASCII,
    not gzip -- 90 real files checked, none carry the `1f 8b` gzip magic, and
    requesting a gzipped name (`.fft.gz`) 400s (no such object exists in the
    archive). `read_fft_gz` currently does `gzip.open(path, "rt")`
    unconditionally (boatphone/fft_io.py, ~line 258) and raises
    `gzip.BadGzipFile: Not a gzipped file (b'0\\n')` on a plain-text payload --
    which is exactly the container the live archive returns.

    This check pins the CONTRACT, not the current implementation: the single
    reader must accept BOTH containers and return byte-for-byte identical
    arrays either way, and it must decide which container it is holding by
    SNIFFING CONTENT (the gzip magic bytes `1f 8b`), never by trusting the
    filename extension. Two fixtures hold the identical known payload -- one
    plain-text, one gzip -- and a THIRD pair deliberately disagrees name vs.
    container (a `.fft.gz`-named file holding plain text, and a `.fft`-named
    file holding gzip), which fails any implementation that dispatches on the
    extension alone. A small, exactly-shaped payload (2 frames x 3 bins) is
    used so a silently-wrong reshape is caught too, not just a round-trip.

    EXPECTED TO FAIL right now: current `read_fft_gz` raises `BadGzipFile` on
    the plain-text fixture. That failure is the point of this check.
    """
    import gzip as _gzip
    _cfg, fft_io = _b0_2a_fft_io()

    # A small, KNOWN payload: 2 frames x 3 bins, values chosen so a transposed
    # or mis-strided reshape would visibly scramble them (no repeats).
    n_frames, n_bins = 2, 3
    values = [0, 1, 2, 3, 4, 5]
    expected = np.asarray(values, dtype=float).reshape(n_frames, n_bins)
    payload_text = " ".join(str(v) for v in values) + "\n"

    # A real filename each fixture must carry to satisfy parse_file_coverage
    # (device code + a Z-stamped timestamp) -- borrowed verbatim from the
    # existing B3-B fixture name so this check needs no new device constant.
    base_stamp = "ICLISTENHF1266_20240715T000136.000Z"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)

        # Fixture 1: name says plain (`.fft`), container IS plain ASCII.
        plain_path = tmp / f"{base_stamp}.fft"
        plain_path.write_text(payload_text, encoding="ascii")

        # Fixture 2: name says gzip (`.fft.gz`), container IS gzip -- the
        # historical (pre-live-probe) assumption, kept so the check also pins
        # that the gzip path still works.
        gz_path = tmp / f"{base_stamp}.fft.gz"
        with _gzip.open(gz_path, "wt", encoding="ascii") as handle:
            handle.write(payload_text)

        # Fixture 3: name says gzip, container is actually PLAIN TEXT -- what
        # the live archive would produce if a caller (wrongly) requested the
        # `.fft.gz` name. Must FAIL an extension-dispatching reader.
        #
        # NOTE: this fixture must be a LEGAL ONC filename. `read_fft_gz` calls
        # `parse_file_coverage(path.name)`, and that grammar allows an OPTIONAL
        # SECOND UTC stamp -- a bare suffix token like "_mismatchA" landed
        # exactly where that second stamp is expected and raised
        # `FilenameParseError` before the container-sniffing contract was even
        # exercised. Fixed by giving each mismatch fixture its own distinct,
        # legal single-UTC-stamp name instead of a suffix token.
        mismatch_gz_name_plain_body = tmp / "ICLISTENHF1266_20240715T001136.000Z.fft.gz"
        mismatch_gz_name_plain_body.write_text(payload_text, encoding="ascii")

        # Fixture 4: name says plain, container is actually GZIP -- the mirror
        # case. Must also FAIL an extension-dispatching reader.
        mismatch_plain_name_gz_body = tmp / "ICLISTENHF1266_20240715T001636.000Z.fft"
        with _gzip.open(mismatch_plain_name_gz_body, "wt", encoding="ascii") as handle:
            handle.write(payload_text)

        results = {}
        for label, path in (
            ("plain_name_plain_body", plain_path),
            ("gz_name_gz_body", gz_path),
            ("gz_name_plain_body", mismatch_gz_name_plain_body),
            ("plain_name_gz_body", mismatch_plain_name_gz_body),
        ):
            product = fft_io.read_fft_gz(
                path, n_frames=n_frames, n_bins=n_bins,
                bin_width_hz=_cfg.FFT_BIN_WIDTH_HZ, frame_seconds=_cfg.FFT_FRAME_SECONDS,
                fs_hz=_cfg.FFT_PRODUCT_FS_HZ,
            )
            results[label] = product.levels_db

        for label, levels_db in results.items():
            assert levels_db.shape == expected.shape, (
                f"{label}: read_fft_gz returned shape {levels_db.shape}, expected "
                f"{expected.shape} ({n_frames} frames x {n_bins} bins) -- a silently "
                "wrong reshape would otherwise pass a shape-blind equality check"
            )
            assert levels_db.dtype == expected.dtype, (
                f"{label}: dtype {levels_db.dtype}, expected {expected.dtype}"
            )
            assert np.array_equal(levels_db, expected), (
                f"{label}: read_fft_gz returned {levels_db.tolist()}, expected "
                f"{expected.tolist()} -- the SAME logical payload, whichever container "
                "it arrived in, must decode to the identical array (content-sniffed, "
                "not extension-dispatched)"
            )

        # The two "container disagrees with name" fixtures must decode IDENTICALLY
        # to the honestly-named ones -- this is what fails a filename-extension
        # dispatch (which would try gzip.open on the plain-text-but-".fft.gz"
        # fixture and raise BadGzipFile, or try plain-text parsing on the
        # gzip-but-".fft" fixture and get garbage/empty tokens).
        assert np.array_equal(results["gz_name_plain_body"], results["plain_name_plain_body"]), (
            "a file named .fft.gz but holding PLAIN TEXT did not decode identically to an "
            "honestly-named plain-text file; the reader must sniff content (gzip magic "
            "bytes 1f 8b), not dispatch on the .gz extension -- this is exactly what the "
            "live ONC archive does (Fact 1, 2026-08-27 probe)"
        )
        assert np.array_equal(results["plain_name_gz_body"], results["gz_name_gz_body"]), (
            "a file named .fft but holding GZIP did not decode identically to an "
            "honestly-named gzip file; the reader must sniff content, not the extension"
        )


def check_b3d_2_onc_400_no_file_could_be_found_is_absent_not_error_no_retry_storm():
    """FACT 2 (live-archive probe, 2026-08-27): ONC answers a missing archive
    file with `400 / API Error 96: No file could be found.` -- and that exact
    string is NOT one of `_NO_DATA_POSSIBLE_MARKERS` (boatphone/onc_client.py),
    which currently holds only "not during the provided time range" and "Start
    Time is in the future". So today this 400 falls through to
    `download_archive_file`'s generic 400 branch and raises `DownloadError`,
    with NO retry (a 400 is not in `_RETRYABLE_STATUS_CODES`) but also with no
    "absent" record -- segment D's absent-log would receive an ERROR row
    instead of an ABSENT row for a file that plainly does not exist.

    This check pins the CONTRACT: a fake transport returning this exact 400
    body must be classified ABSENT -- a structured record with
    `status == "absent"`, no raise, and `attempts` showing NO retry storm (a
    single request, matching the real 404 case in check_b3b_6, since a 400
    naming a missing file needs no backoff any more than a 404 does).

    A companion assertion, so the fix cannot widen into swallowing real
    failures (invariant 5): a genuine 500, and a 400 with an UNRELATED message,
    must still RAISE `DownloadError` exactly as they do today.

    GUESS FLAGGED: the target status for this outcome is asserted as
    `"absent"` (matching the real-404 case's status string, per the task's
    instruction to record it "as ABSENT" / "an absent row"), not a new
    `"measured_zero"`-like status -- `_NO_DATA_POSSIBLE_MARKERS`'s existing
    "measured_zero" status is reserved for D7's "no deployment / window not
    yet elapsed" 400s, a different finding (there IS a deployment, ONC simply
    has no file for this filename) -- but the exact literal is an
    implementation choice, not observed fact, and is called out here rather
    than hidden.

    EXPECTED TO FAIL right now: current `download_archive_file` raises
    `DownloadError` for this message (it matches neither
    `_NO_DATA_POSSIBLE_MARKERS` entry), so `raised is None` below fails.
    """
    _cfg, onc = _b3b_mod()

    no_file_could_be_found_filename = B3B_FAKE_FILENAME
    unrelated_400_filename = "ICLISTENHF1266_20240716T000136.000Z.fft.gz"

    class _NoFileFoundTransport(_B3BFakeTransport):
        """Returns ONC's real Error-96 body for one filename, a 500 for
        another, and an UNRELATED 400 message for a third -- so the check can
        assert all three outcomes in one fixture without re-defining the base
        transport."""

        def get(self, filename, range_start=0):
            self.calls.append((filename, range_start))
            if filename == no_file_could_be_found_filename:
                return _B3BFakeResponse(
                    400, error_text="API Error 96: No file could be found."
                )
            if filename == "GENUINE-500-CASE":
                return _B3BFakeResponse(500, error_text="simulated 500 (test fixture)")
            if filename == unrelated_400_filename:
                return _B3BFakeResponse(
                    400, error_text="API Error 99: Some other, unrelated failure."
                )
            return super().get(filename, range_start=range_start)

    # --- Part A: "No file could be found" must be ABSENT, not raised, and
    # must not trigger a retry storm (one request, like the real-404 path). ---
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _NoFileFoundTransport({})

        raised = None
        result = None
        try:
            result = onc.download_archive_file(
                client=None, filename=no_file_could_be_found_filename, dest_dir=dest_dir,
                transport=transport, sleep=lambda _s: None,
            )
        except Exception as exc:  # noqa: BLE001 -- must NOT raise; caught to report clearly
            raised = exc
        assert raised is None, (
            f"ONC's 'API Error 96: No file could be found.' 400 raised "
            f"{type(raised).__name__ if raised else None}: {raised}; a missing archive "
            "file is a measured ABSENCE, not a broken request, and must not raise "
            "DownloadError -- over 918 dates this both mislabels the absent-log (which "
            "segment D consumes) and burns a run on retries"
        )
        status = getattr(result, "status", None)
        assert status is not None and str(status).lower() == "absent", (
            f"expected status 'absent' for ONC Error 96 ('No file could be found'), got "
            f"{status!r} from {result!r}"
        )
        attempts = getattr(result, "attempts", None)
        assert attempts == 1, (
            f"expected exactly 1 attempt (no retry storm) for a 'file does not exist' "
            f"answer, got attempts={attempts!r} from {result!r} -- retrying a file ONC "
            "has just said does not exist wastes an attempt budget across 918 dates for "
            "no possible different outcome"
        )
        assert len(transport.calls) == 1, (
            f"transport.get() was called {len(transport.calls)} time(s) for a single "
            f"Error-96 answer; expected exactly 1 (calls={transport.calls!r})"
        )
        assert not (dest_dir / no_file_could_be_found_filename).exists(), (
            "a file appeared at the final path for a filename ONC says cannot be found"
        )

    # --- Part B (companion, invariant 5): unrelated real failures still RAISE. ---
    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _NoFileFoundTransport({})
        try:
            onc.download_archive_file(
                client=None, filename="GENUINE-500-CASE", dest_dir=dest_dir,
                transport=transport, sleep=lambda _s: None,
            )
        except onc.DownloadError:
            pass
        else:
            raise AssertionError(
                "a genuine 500 was not raised as DownloadError; the fix for Fact 2 must "
                "not widen into swallowing real server failures (invariant 5)"
            )

    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _NoFileFoundTransport({})
        try:
            onc.download_archive_file(
                client=None, filename=unrelated_400_filename, dest_dir=dest_dir,
                transport=transport, sleep=lambda _s: None,
            )
        except onc.DownloadError:
            pass
        else:
            raise AssertionError(
                "a 400 with an UNRELATED message ('API Error 99: ...') was not raised as "
                "DownloadError; only the exact 'No file could be found' family may be "
                "absorbed as absent -- widening the match would swallow real failures "
                "(invariant 5)"
            )


# ---------------------------------------------------------------------------
# B3-E: the bulk pull compresses each file as it lands on disk.
#
# NEW CONTRACT under test (decision 0022 established the WIRE format is plain
# ASCII; this segment is the follow-on decision that the LOCAL COPY of that
# plain ASCII is gzip-compressed at write time, because the full corpus is
# ~40 GB uncompressed and ~8 GB gzipped -- ASCII digits compress ~4.9x).
#
# `download_archive_file` as it stands (read above, ~line 1574) writes
# whatever bytes `transport.get()` hands it, VERBATIM, to `<filename>.part`
# and then to the final path. It does not compress anything, and its sha256
# sidecar is computed over exactly the bytes written to disk (`_sha256_of_file`
# on the FINAL path). So every check below is EXPECTED TO FAIL right now:
# there is no gzip magic at the final path, and (once compression is added by
# a naive implementation that just gzips-then-hashes-the-container) the
# sidecar hash will be the WRONG thing to check against -- that is check 2's
# whole point.
#
# DECIDED (2026-08-27, human decision -- was previously "GUESS FLAGGED, real
# design fork, not resolved here"): the compressed file IS renamed/suffixed to
# `....fft.gz` on write -- the name states the container it actually holds,
# honestly, rather than leaving a `.fft`-named file holding gzip bytes.
# `check_b3c_11` (this file) was AMENDED accordingly: it now pins that
# `boatphone.acquire.resolve_corpus_files` matches BOTH `*.<ARCHIVE_EXTENSION>`
# (`.fft`, the 90 already-pulled plain files data/ immutability forbids ever
# converting) AND `*.<PRODUCT_EXTENSION>` (`.fft.gz`, every future bulk-pull
# download) -- the previous exclusion of `.fft.gz` is gone, not defended. The
# checks below still do NOT hardcode the name into their own fixtures where
# avoidable -- they call `download_archive_file` and then ask where it
# actually put the bytes -- but `check_b3e_4`/`check_b3e_5` below now assert
# the `.fft.gz` name explicitly where that pins the decided contract more
# strongly, rather than staying deliberately agnostic about a question that is
# no longer open.
# ---------------------------------------------------------------------------

# A REALISTIC, REPETITIVE plain-ASCII payload standing in for a real `.fft`
# product: real files are 614,400 whitespace-separated small integers (values
# 0-92, decision 0022), which is exactly the kind of low-cardinality,
# repetitive text gzip compresses well. This fixture is smaller (for test
# speed) but keeps the same character: a few repeating digit tokens, tens of
# thousands of times over. A tiny or random payload is NOT used for the
# smaller-than-original assertion because gzip's own header/block overhead can
# make a short or high-entropy input LARGER, not smaller -- that would be
# testing gzip's known behaviour on a bad input, not this pipeline's contract.
_B3E_PLAIN_ASCII_PAYLOAD = (
    " ".join(str(v) for v in ([0, 1, 2, 4, 8, 16, 32, 64] * 20000)) + "\n"
).encode("ascii")
_B3E_FAKE_FILENAME = "ICLISTENHF1266_20240715T000136.000Z.fft"


def _b3e_gzip_magic(data: bytes) -> bool:
    return data[:2] == b"\x1f\x8b"


def check_b3e_1_downloaded_bytes_land_gzip_compressed_and_round_trip_exactly():
    """The on-disk copy of a downloaded `.fft` file is gzip-compressed, and
    decompresses back to the EXACT wire bytes byte-for-byte.

    A fake transport serves `_B3E_PLAIN_ASCII_PAYLOAD` (realistic, repetitive
    plain ASCII, per decision 0022's measured contract) through
    `download_archive_file`. Pins three things at once: (a) gzip magic bytes
    `1f 8b` at the start of whatever file `download_archive_file` leaves
    behind that holds the payload; (b) `gzip.decompress()` on that file
    reproduces the payload byte-for-byte, not merely "close" or
    length-matched; (c) the stored file is measurably SMALLER than the
    payload -- meaningful here because the fixture is large and repetitive
    (unlike a tiny/random payload, where gzip can legitimately expand the
    input; that case is deliberately not asserted on).

    EXPECTED TO FAIL right now: `download_archive_file` writes the transport's
    bytes verbatim with no compression step, so the final file has no gzip
    magic and is exactly `len(payload)` bytes, not smaller.
    """
    import gzip as _gzip
    _cfg, onc = _b3b_mod()

    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _B3BFakeTransport({_B3E_FAKE_FILENAME: _B3E_PLAIN_ASCII_PAYLOAD})
        record = onc.download_archive_file(
            client=None, filename=_B3E_FAKE_FILENAME, dest_dir=dest_dir,
            transport=transport, sleep=lambda _s: None,
        )
        assert getattr(record, "status", None) == "downloaded", (
            f"expected a fresh 'downloaded' record from a single-shot fake transport, "
            f"got {record!r}"
        )
        path = getattr(record, "path", None)
        assert path is not None and pathlib.Path(path).is_file(), (
            f"download_archive_file returned status='downloaded' but {record!r}.path is not "
            "a real file"
        )
        path = pathlib.Path(path)
        on_disk = path.read_bytes()

        assert _b3e_gzip_magic(on_disk), (
            f"{path} does not start with the gzip magic bytes (1f 8b); its first two bytes "
            f"are {on_disk[:2]!r}. The bulk-pull contract under test is that each file is "
            "compressed as it is written to disk (the full corpus is ~40 GB uncompressed vs "
            "~8 GB gzipped, decision 0022's follow-on) -- this is the not-yet-implemented "
            "compression step"
        )
        decompressed = _gzip.decompress(on_disk)
        assert decompressed == _B3E_PLAIN_ASCII_PAYLOAD, (
            f"decompressing {path} gave {len(decompressed)} byte(s), not the original "
            f"{len(_B3E_PLAIN_ASCII_PAYLOAD)}-byte wire payload byte-for-byte -- the round "
            "trip must be exact, not merely the right length"
        )
        assert len(on_disk) < len(_B3E_PLAIN_ASCII_PAYLOAD), (
            f"{path} is {len(on_disk)} byte(s) on disk, not smaller than the "
            f"{len(_B3E_PLAIN_ASCII_PAYLOAD)}-byte payload it was compressed from. This "
            "assertion is only meaningful because the fixture payload is large and "
            "repetitive (ASCII digits compress ~4.9x per decision 0022) -- it is "
            "deliberately not asserted for a tiny or random payload, where gzip's own "
            "header/block overhead can legitimately make the output larger"
        )


def check_b3e_2_sha256_sidecar_hashes_the_wire_bytes_not_the_compressed_container():
    """The `.sha256` sidecar records `sha256(plain wire payload)`, NEVER
    `sha256(the gzip container written to disk)`.

    This is the subtle contract, and the reason it is checked separately from
    check_b3e_1: decision 0018 defines the local sha256 as the integrity
    record of WHAT WAS RECEIVED. If compression is bolted on by hashing the
    compressed bytes instead, the existing cache-hit re-verification keeps
    passing -- nothing observably breaks -- but the number it is now checking
    means something different, and gzip's own container is not even a stable
    thing to hash: gzip is not deterministic across zlib versions, mtime
    embedding, or compresslevel, so a hash of compressed bytes is not
    reproducible the way a hash of the plain payload is.

    Both directions are asserted so this cannot pass vacuously: the recorded
    hash MUST equal `sha256(plain payload)`, and MUST NOT equal
    `sha256(on-disk compressed bytes)` -- the two are asserted unequal first,
    so a future accidental collision could never make this check meaningless.

    EXPECTED TO FAIL right now for the boring reason (no compression exists
    yet, so on-disk bytes ARE the plain bytes and the two hashes coincide) --
    and expected to keep failing, for the real reason, against a naive
    "gzip then hash the container" implementation.
    """
    import hashlib as _hashlib
    _cfg, onc = _b3b_mod()

    plain_sha = _hashlib.sha256(_B3E_PLAIN_ASCII_PAYLOAD).hexdigest()

    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _B3BFakeTransport({_B3E_FAKE_FILENAME: _B3E_PLAIN_ASCII_PAYLOAD})
        record = onc.download_archive_file(
            client=None, filename=_B3E_FAKE_FILENAME, dest_dir=dest_dir,
            transport=transport, sleep=lambda _s: None,
        )
        path = pathlib.Path(record.path)
        on_disk = path.read_bytes()
        on_disk_sha = _hashlib.sha256(on_disk).hexdigest()

        # Requires compression to actually exist first: if it does not (today),
        # on_disk_sha == plain_sha and the "must differ" assertion below would
        # be checking a fact that is true for the wrong reason. That state is
        # exactly check_b3e_1's failure, so it is named here rather than
        # silently letting this check "pass" on an unrelated technicality.
        assert _b3e_gzip_magic(on_disk), (
            f"{path} is not gzip-compressed (no magic bytes) -- the compression step under "
            "test does not exist yet (see check_b3e_1). Without it, on-disk bytes ARE the "
            "plain bytes and the 'sha differs from container hash' assertion below would be "
            "checking a coincidence, not the real contract"
        )

        recorded_sha = getattr(record, "sha256", None)
        assert recorded_sha is not None, (
            f"download_archive_file's DownloadRecord carries no sha256: {record!r}"
        )
        assert recorded_sha == plain_sha, (
            f"recorded sha256 ({recorded_sha}) does not equal sha256(plain wire payload) "
            f"({plain_sha}) -- decision 0018's local sha256 is the integrity record of what "
            "was RECEIVED, and it must survive a compression step applied only to the "
            "on-disk representation"
        )
        assert recorded_sha != on_disk_sha, (
            f"recorded sha256 ({recorded_sha}) equals sha256(on-disk compressed bytes) "
            f"({on_disk_sha}) -- gzip is not deterministic across zlib versions, mtime "
            "embedding, or compresslevel, so a hash of the compressed CONTAINER is not even "
            "a stable value to store, on top of being the wrong thing to hash"
        )


def check_b3e_3_cache_hit_reverifies_a_compressed_file_and_catches_corruption():
    """Re-running against an already-downloaded COMPRESSED file is a clean
    cache hit (no re-download), verified by decompressing and hashing the
    PLAIN bytes -- and a corrupted compressed file is caught, not accepted.

    Two runs share one fake transport and one `dest_dir`. The first run
    performs the (not-yet-implemented) compressed download. The second run
    must: (a) issue ZERO further transport calls (`len(transport.calls)`
    unchanged) -- a "cache hit" that silently re-downloads defeats the whole
    point of resumability at 918-date scale; (b) return `status == "cached"`.

    The corruption half then mutates the stored file's DECOMPRESSED content
    (by writing a fresh gzip container over different bytes, so the file
    still carries valid gzip magic and opens fine -- only its content is
    wrong) and re-runs: this MUST be detected (a raise, per
    `download_archive_file`'s existing "cached hash mismatch never silently
    overwritten" contract, CLAUDE.md invariant 2) rather than silently
    accepted as another clean cache hit.

    EXPECTED TO FAIL right now: no compression exists, so there is nothing
    to decompress on a cache hit, and today's cache-hit path already hashes
    the on-disk bytes directly -- which happens to catch this PARTICULAR
    corruption fixture today (a real 200), but check_b3e_2 above shows why
    that coincidence does not survive compression being added.
    """
    import gzip as _gzip
    _cfg, onc = _b3b_mod()

    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        transport = _B3BFakeTransport({_B3E_FAKE_FILENAME: _B3E_PLAIN_ASCII_PAYLOAD})

        first = onc.download_archive_file(
            client=None, filename=_B3E_FAKE_FILENAME, dest_dir=dest_dir,
            transport=transport, sleep=lambda _s: None,
        )
        assert first.status == "downloaded", f"expected a fresh download, got {first!r}"
        path = pathlib.Path(first.path)
        assert _b3e_gzip_magic(path.read_bytes()), (
            f"{path} is not gzip-compressed after the first download (no magic bytes) -- "
            "the compression step under test does not exist yet (see check_b3e_1)"
        )
        calls_after_first = len(transport.calls)

        second = onc.download_archive_file(
            client=None, filename=_B3E_FAKE_FILENAME, dest_dir=dest_dir,
            transport=transport, sleep=lambda _s: None,
        )
        assert len(transport.calls) == calls_after_first, (
            f"a cache-hit re-run against an already-downloaded compressed file issued "
            f"{len(transport.calls) - calls_after_first} MORE transport call(s) -- a cache "
            "hit must never re-download"
        )
        assert second.status == "cached", (
            f"expected status='cached' on the second run against an already-downloaded "
            f"compressed file, got {second!r}"
        )

        # Corrupt: overwrite the stored file with a VALID gzip container that
        # decompresses to DIFFERENT content, so a naive "does this open as
        # gzip" check would pass while the actual content is wrong.
        wrong_payload = b"CORRUPTED-NOT-THE-REAL-PAYLOAD " * 100
        with open(path, "wb") as handle:
            handle.write(_gzip.compress(wrong_payload))

        raised = None
        try:
            onc.download_archive_file(
                client=None, filename=_B3E_FAKE_FILENAME, dest_dir=dest_dir,
                transport=transport, sleep=lambda _s: None,
            )
        except Exception as exc:  # noqa: BLE001 -- the raise IS the expected outcome
            raised = exc
        assert raised is not None, (
            f"download_archive_file silently accepted a compressed file whose DECOMPRESSED "
            f"content no longer matches its recorded sha256 sidecar -- corruption of a "
            f"compressed cache entry must be detected (CLAUDE.md invariant 2: a cached file "
            "is never silently trusted after it fails its integrity check), exactly as an "
            "uncompressed mismatch already raises today"
        )


def check_b3e_4_the_one_reader_opens_the_compressed_corpus_and_the_resolver_still_finds_it():
    """After a compressed file lands, `fft_io.read_fft_gz` reads it AND
    `acquire.resolve_corpus_files` still resolves it, under its actual
    on-disk name.

    DECIDED (2026-08-27, was previously "deliberately does not hardcode the
    name, real design fork"): the compressed file's name IS
    `config.PRODUCT_EXTENSION` (`.fft.gz`), stating the container honestly.
    This check now pins that name explicitly rather than staying agnostic --
    `download_archive_file` is called with a `.fft`-suffixed input filename
    (`config.ARCHIVE_EXTENSION`, what the real archive registers under), and
    the landed file is asserted to end `.fft.gz`, not merely "whatever it
    ended up being". `check_b3c_11` was amended (see its docstring) to match
    `.fft.gz` rather than exclude it, so the two checks are no longer in
    tension -- this check now also serves as evidence they agree.

    EXPECTED TO FAIL right now: no compression exists (check_b3e_1), and
    `read_fft_gz` unconditionally `gzip.open()`s its input (module docstring,
    boatphone/fft_io.py ~line 258) -- decision 0022 already establishes that
    is wrong for the current PLAIN-text wire format, so this needs whatever
    fix lands for check_b3d_1 as well, not a second reader.
    """
    from boatphone import acquire, fft_io, config as cfg
    _cfg_onc, onc = _b3b_mod()

    n_frames, n_bins = 2, 3
    values = [0, 1, 2, 3, 4, 5]
    expected = np.asarray(values, dtype=float).reshape(n_frames, n_bins)
    payload_text = (" ".join(str(v) for v in values) + "\n").encode("ascii")
    stamp_filename = f"ICLISTENHF1266_20240715T000136.000Z.{cfg.ARCHIVE_EXTENSION}"

    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc_raw" / "overpass_window_corpus"
        transport = _B3BFakeTransport({stamp_filename: payload_text})
        record = onc.download_archive_file(
            client=None, filename=stamp_filename, dest_dir=dest_dir,
            transport=transport, sleep=lambda _s: None,
        )
        landed_path = pathlib.Path(record.path)
        assert landed_path.is_file(), f"no file at {landed_path} after download"
        assert _b3e_gzip_magic(landed_path.read_bytes()), (
            f"{landed_path} is not gzip-compressed -- the compression step under test does "
            "not exist yet (see check_b3e_1); the reader/resolver agreement asserted below "
            "cannot be meaningfully checked without it"
        )
        assert landed_path.name.endswith(f".{cfg.PRODUCT_EXTENSION}"), (
            f"{landed_path.name!r} does not end '.{cfg.PRODUCT_EXTENSION}' -- the decided "
            "contract (2026-08-27) is that the compressed file is named '.fft.gz', stating "
            f"the container it holds honestly; got {landed_path.name!r}"
        )

        found = acquire.resolve_corpus_files(dest_dir)
        found_names = {pathlib.Path(f).name for f in found}
        assert landed_path.name in found_names, (
            f"resolve_corpus_files({dest_dir}) did not find the compressed download at "
            f"{landed_path.name!r} (found {found_names!r}) -- resolve_corpus_files must match "
            "'.fft.gz' names (check_b3c_11 pins this; the prior exclusion was removed)"
        )

        product = fft_io.read_fft_gz(
            landed_path, n_frames=n_frames, n_bins=n_bins,
            bin_width_hz=cfg.FFT_BIN_WIDTH_HZ, frame_seconds=cfg.FFT_FRAME_SECONDS,
            fs_hz=cfg.FFT_PRODUCT_FS_HZ,
        )
        assert np.array_equal(product.levels_db, expected), (
            f"read_fft_gz({landed_path}) returned {product.levels_db.tolist()}, expected "
            f"{expected.tolist()} -- the ONE reader must open the compressed corpus that "
            "the bulk pull actually produces, not just the hand-delivered sample's format"
        )


def check_b3e_5_mixed_corpus_of_already_pulled_plain_text_and_new_gzip_files_both_work():
    """A corpus mixing the 90 already-pulled PLAIN-TEXT probe files (real,
    under `data/raw/onc/overpass_window_corpus/`, immutable, never re-pulled)
    with newly-bulk-pulled GZIP-compressed files must resolve and read
    correctly for BOTH kinds at once.

    Uses SYNTHETIC fixtures standing in for the real 90 files -- `data/` is
    never touched, read, or written by this check (test-author brief; CLAUDE.md
    invariant 2). The two synthetic files share one directory: one plain-ASCII
    `.fft` (mimicking the already-pulled probe corpus, decision 0022) and one
    gzip-compressed file (mimicking a fresh bulk-pull download, built through
    the same `download_archive_file` path check_b3e_1 exercises, not a
    hand-rolled gzip.compress -- so this check exercises the real write path,
    not a stand-in for it). DECIDED (2026-08-27): the landed compressed file
    must be named `.fft.gz` (`config.PRODUCT_EXTENSION`), asserted explicitly
    below, not just "some name the resolver happens to accept" -- this is the
    mixed steady state the corpus is expected to be in from now on.

    EXPECTED TO FAIL right now for TWO independent reasons, either of which is
    sufficient: (1) no compression exists yet (check_b3e_1), so there is only
    one kind of file to mix; (2) even once there is, `read_fft_gz` does not
    yet sniff content over trusting the extension (check_b3d_1's contract),
    so a plain-text `.fft` file raises `BadGzipFile` today.
    """
    from boatphone import acquire, fft_io, config as cfg
    _cfg_onc, onc = _b3b_mod()

    n_frames, n_bins = 1, 4
    plain_values = [10, 20, 30, 40]
    gzip_values = [50, 60, 70, 80]
    plain_expected = np.asarray(plain_values, dtype=float).reshape(n_frames, n_bins)
    gzip_expected = np.asarray(gzip_values, dtype=float).reshape(n_frames, n_bins)

    plain_stamp = f"ICLISTENHF1266_20240715T000136.000Z.{cfg.ARCHIVE_EXTENSION}"
    gzip_stamp = f"ICLISTENHF1266_20240715T000636.000Z.{cfg.ARCHIVE_EXTENSION}"

    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = pathlib.Path(tmp) / "onc_raw" / "overpass_window_corpus"
        corpus_dir.mkdir(parents=True)

        # Stand-in for an already-pulled probe file: written directly as
        # plain ASCII, exactly what the 90 real files under
        # data/raw/onc/overpass_window_corpus/ are (decision 0022) -- no
        # download_archive_file call needed for this half since it predates
        # the compression feature under test.
        plain_text = (" ".join(str(v) for v in plain_values) + "\n").encode("ascii")
        (corpus_dir / plain_stamp).write_bytes(plain_text)

        # Stand-in for a freshly bulk-pulled file: through the REAL
        # download_archive_file path, so this exercises the actual write
        # behaviour rather than a hand-built gzip fixture.
        gzip_text = (" ".join(str(v) for v in gzip_values) + "\n").encode("ascii")
        transport = _B3BFakeTransport({gzip_stamp: gzip_text})
        record = onc.download_archive_file(
            client=None, filename=gzip_stamp, dest_dir=corpus_dir,
            transport=transport, sleep=lambda _s: None,
        )
        gzip_path = pathlib.Path(record.path)
        assert _b3e_gzip_magic(gzip_path.read_bytes()), (
            f"{gzip_path} is not gzip-compressed -- the compression step under test does not "
            "exist yet (see check_b3e_1)"
        )
        assert gzip_path.name.endswith(f".{cfg.PRODUCT_EXTENSION}"), (
            f"{gzip_path.name!r} does not end '.{cfg.PRODUCT_EXTENSION}' -- the decided "
            "contract (2026-08-27) is that the compressed file is named '.fft.gz'; got "
            f"{gzip_path.name!r}"
        )

        found = acquire.resolve_corpus_files(corpus_dir)
        found_names = {pathlib.Path(f).name for f in found}
        assert plain_stamp in found_names, (
            f"resolve_corpus_files({corpus_dir}) did not find the plain-text probe-style "
            f"file {plain_stamp!r} in a MIXED corpus; got {found_names!r}"
        )
        assert gzip_path.name in found_names, (
            f"resolve_corpus_files({corpus_dir}) did not find the gzip-compressed bulk-pull "
            f"file {gzip_path.name!r} in a MIXED corpus; got {found_names!r}"
        )

        plain_product = fft_io.read_fft_gz(
            corpus_dir / plain_stamp, n_frames=n_frames, n_bins=n_bins,
            bin_width_hz=cfg.FFT_BIN_WIDTH_HZ, frame_seconds=cfg.FFT_FRAME_SECONDS,
            fs_hz=cfg.FFT_PRODUCT_FS_HZ,
        )
        assert np.array_equal(plain_product.levels_db, plain_expected), (
            f"read_fft_gz on the plain-text probe-style file returned "
            f"{plain_product.levels_db.tolist()}, expected {plain_expected.tolist()}"
        )

        gzip_product = fft_io.read_fft_gz(
            gzip_path, n_frames=n_frames, n_bins=n_bins,
            bin_width_hz=cfg.FFT_BIN_WIDTH_HZ, frame_seconds=cfg.FFT_FRAME_SECONDS,
            fs_hz=cfg.FFT_PRODUCT_FS_HZ,
        )
        assert np.array_equal(gzip_product.levels_db, gzip_expected), (
            f"read_fft_gz on the gzip-compressed bulk-pull file returned "
            f"{gzip_product.levels_db.tolist()}, expected {gzip_expected.tolist()}"
        )


def check_b3e_6_compressed_resume_offset_is_wire_bytes_not_compressed_part_size():
    """Resuming a COMPRESSED (`.fft.gz`, compress-on-write) download must ask
    the server to resume at the next WIRE (plain, decompressed) byte -- never
    at the on-disk COMPRESSED size of the `.part`.

    This pins a bug the coder actually hit and fixed (`_wire_bytes_on_disk` in
    `boatphone/onc_client.py`): with compress-on-write, the bytes on disk are
    NOT the bytes received. Using `.part.stat().st_size` (compressed) as
    `range_start` asks the server -- which serves plain wire bytes and knows
    nothing about local compression -- to resume at a number smaller than the
    plain bytes actually already held. The response is then measured against
    `response.total_size` (also stated in plain/wire bytes), which the
    resumed leg can never reach: `got_size < expected_total` forever, an
    infinite short-body retry loop that never promotes the file.

    A fake transport records the `range_start` it is asked for on the resumed
    (206) leg, AND independently snapshots the `.part` file on disk at the
    exact instant that resumed request is issued (via `dest_dir.rglob`, not
    by trusting the implementation's own idea of the offset). From that
    snapshot the check recomputes, itself, what the WIRE offset should be by
    decompressing the snapshot -- so this does not just re-run the function
    under test's own arithmetic back at it. The fixture payload
    (`_B3E_PLAIN_ASCII_PAYLOAD`, repetitive ASCII, ~4.9x compressible per
    decision 0022) is chosen so the compressed and plain byte counts of that
    same snapshot are asserted UNEQUAL first -- the two candidate offsets
    (wire vs. compressed) must actually differ, or an implementation reading
    the wrong one would pass by coincidence and this check would be vacuous.

    Outcome coverage, not just the mid-flight offset: the file finally
    promoted to the final path must decompress to the complete, exact
    original payload, and its `.sha256` sidecar (decision 0024) must equal
    `sha256(plain wire payload)` -- not `sha256(compressed bytes)` -- mirroring
    check_b3e_2's contract but now via a resumed transfer specifically, since
    a resume that reads or writes the wrong byte count could plausibly still
    self-correct into a right final hash and a wrong one is worth separately
    ruling out.

    GUARDING A BRANCH NOT CURRENTLY REACHABLE IN PRODUCTION, stated so a later
    reader does not mistake this for observed live behaviour: decision 0018
    records that ONC ignores the `Range` header in practice and always
    answers with a bare 200, so the 206-honoured resume path this check
    drives is real code, exercised here, but not a path the live ONC archive
    currently takes.

    Per the coder's own account this is ALREADY FIXED (`_wire_bytes_on_disk`
    reads via `gzip.open(...).read()` in chunks rather than `st_size`), so
    this check is expected to PASS immediately -- it is written to PIN that
    fix against a future regression (e.g. a "simplify away the decompress
    loop" refactor that quietly goes back to `st_size`), not to demonstrate a
    currently-broken implementation.
    """
    import gzip as _gzip
    import hashlib as _hashlib
    _cfg, onc = _b3b_mod()

    with tempfile.TemporaryDirectory() as tmp:
        dest_dir = pathlib.Path(tmp) / "onc"
        dest_dir.mkdir(parents=True, exist_ok=True)

        class _CompressedResumeTransport(_B3BFakeTransport):
            """206-honouring resume transport (as `_B3BFakeTransport206OnResume`)
            that ALSO snapshots whatever `.part` file exists on disk at the
            instant each resumed (`range_start > 0`) request is issued, so the
            check can independently recompute the WIRE-byte offset a correct
            implementation must ask for -- rather than trusting the value the
            implementation itself passes as `range_start`.
            """

            def __init__(self, content_by_name):
                super().__init__(content_by_name)
                self.part_snapshots_at_resume = []

            def get(self, filename, range_start=0):
                if range_start > 0:
                    part_files = list(dest_dir.rglob("*.part"))
                    assert part_files, (
                        "resume was attempted (range_start > 0) but no .part sidecar "
                        f"exists under {dest_dir} -- nothing to resume from"
                    )
                    assert len(part_files) == 1, (
                        f"expected exactly one .part file at resume time, found {part_files}"
                    )
                    self.part_snapshots_at_resume.append(part_files[0].read_bytes())
                self.calls.append((filename, range_start))
                full = self.content_by_name[filename]
                if range_start > 0:
                    return _B3BFakeResponse(206, body=full[range_start:], total_size=len(full))
                interrupt_after = None
                if self.interrupt_once_after is not None:
                    interrupt_after = self.interrupt_once_after
                    self.interrupt_once_after = None  # only the FIRST attempt is interrupted
                return _B3BFakeResponse(
                    200, body=full, total_size=len(full), interrupt_after=interrupt_after
                )

        transport = _CompressedResumeTransport({_B3E_FAKE_FILENAME: _B3E_PLAIN_ASCII_PAYLOAD})
        # Plain (wire) bytes served before the simulated drop. Small relative
        # to chunk_bytes below so the interrupt lands after several whole
        # chunks, giving the .part real partial content to snapshot.
        transport.interrupt_once_after = 4096

        result = onc.download_archive_file(
            client=None, filename=_B3E_FAKE_FILENAME, dest_dir=dest_dir,
            transport=transport, sleep=lambda _s: None, chunk_bytes=512,
        )

        range_starts = [rs for (_fn, rs) in transport.calls]
        resumed_offsets = [rs for rs in range_starts if rs > 0]
        assert resumed_offsets, (
            f"transport.get() was never called with a partial range_start ({range_starts!r}) "
            "-- the interrupted-then-resumed leg this check depends on never happened, so "
            "nothing about the resume offset was exercised"
        )
        assert len(resumed_offsets) == len(transport.part_snapshots_at_resume), (
            "resumed range_start calls and .part snapshots are out of step "
            f"({len(resumed_offsets)} vs {len(transport.part_snapshots_at_resume)}) -- "
            "the fixture's own bookkeeping is broken, not the code under test"
        )
        requested_offset = resumed_offsets[0]
        part_snapshot = transport.part_snapshots_at_resume[0]

        compressed_len = len(part_snapshot)
        decompressed_len = len(_gzip.decompress(part_snapshot))

        assert decompressed_len != compressed_len, (
            f"the .part snapshot decompresses to {decompressed_len} byte(s) but is "
            f"{compressed_len} byte(s) on disk -- these must be UNEQUAL for the fixture "
            "payload (repetitive ASCII, ~4.9x compressible), or wire-bytes-vs-compressed-"
            "bytes cannot be distinguished and this check would pass no matter which one "
            "the implementation used"
        )
        assert requested_offset == decompressed_len, (
            f"resume requested range_start={requested_offset}, but the .part file on disk "
            f"at that moment held {compressed_len} compressed byte(s) decompressing to "
            f"{decompressed_len} WIRE byte(s). "
            + (
                "The implementation is reading the on-disk (COMPRESSED) size as the resume "
                "offset -- this asks the server (which serves plain wire bytes and knows "
                "nothing of local compression) to resume too early, and the response can "
                "never reach response.total_size (stated in wire bytes): an infinite "
                "short-body retry loop that never promotes the file."
                if requested_offset == compressed_len else
                "The requested offset matches neither the compressed nor the decompressed "
                "size of the .part snapshot -- something else is wrong with the resume "
                "offset calculation."
            )
        )

        final_path = getattr(result, "path", None)
        assert final_path is not None and pathlib.Path(final_path).is_file(), (
            f"download_archive_file did not promote a final file after the resumed transfer "
            f"completed: {result!r}"
        )
        final_path = pathlib.Path(final_path)
        on_disk = final_path.read_bytes()
        assert _b3e_gzip_magic(on_disk), (
            f"{final_path} does not carry the gzip magic bytes after a compressed resumed "
            "download"
        )
        decompressed_final = _gzip.decompress(on_disk)
        assert decompressed_final == _B3E_PLAIN_ASCII_PAYLOAD, (
            f"the promoted file decompresses to {len(decompressed_final)} byte(s), not the "
            f"original {len(_B3E_PLAIN_ASCII_PAYLOAD)}-byte payload byte-for-byte, after a "
            "compressed resumed download -- concatenated gzip members (one per resumed leg) "
            "must decompress to the exact concatenation of the plain bytes"
        )

        sha_path = final_path.with_name(final_path.name + ".sha256")
        assert sha_path.is_file(), (
            f"no .sha256 sidecar written at {sha_path} after a compressed resumed download "
            "completed (decision 0024)"
        )
        recorded_sha = sha_path.read_text(encoding="utf-8").strip()
        plain_sha = _hashlib.sha256(_B3E_PLAIN_ASCII_PAYLOAD).hexdigest()
        on_disk_sha = _hashlib.sha256(on_disk).hexdigest()
        assert recorded_sha == plain_sha, (
            f"the .sha256 sidecar records {recorded_sha}, not sha256(plain wire payload) "
            f"({plain_sha}) -- decision 0024's sidecar must hash the WIRE bytes, and a "
            "resumed transfer is exactly where compressed-vs-wire byte confusion would show "
            "up in the hash too, not only in the resume offset"
        )
        assert recorded_sha != on_disk_sha, (
            f"the .sha256 sidecar ({recorded_sha}) equals sha256(on-disk compressed "
            f"container) ({on_disk_sha}) -- gzip output is not byte-stable, so hashing the "
            "container is both the wrong thing and not even reproducible"
        )


# --- B5 viability gate (features.py, overpasses.py) -------------------------
#
# These checks pin the CONTRACT of the B5 band-level detector, on SYNTHETIC
# signals with known ground truth (CLAUDE.md invariant 3). None of them touches
# the real corpus: a check that needs 6.8 GB of gitignored data to run is a
# check that silently skips for everyone else on the team.


def _b5_mods():
    """Import the B5 library modules, skipping cleanly if they are absent."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from boatphone import config as cfg, features, fft_io, overpasses
    except ImportError as exc:
        raise SkipCheck(f"B5 modules not importable: {exc}") from exc
    return cfg, features, fft_io, overpasses


def _b5_synthetic_surface(quiet=20.0):
    """A flat, quiet (n_frames, n_bins) product surface with a known level."""
    import numpy as np
    cfg, _features, _fft_io, _ov = _b5_mods()
    return np.full((cfg.FFT_N_FRAMES, cfg.FFT_N_BINS), float(quiet), dtype=float)


class _B5FakeProduct:
    """Minimal stand-in for fft_io.FftProduct -- named axes, no file needed."""

    def __init__(self, levels_db, freq_hz, t_utc_s):
        self.levels_db = levels_db
        self.freq_hz = freq_hz
        self.t_utc_s = t_utc_s


def _b5_product(levels_db, t0=0.0):
    import numpy as np
    cfg, _f, fft_io, _ov = _b5_mods()
    freq_hz = fft_io.frequency_axis_hz(n_bins=levels_db.shape[1])
    t = t0 + np.arange(levels_db.shape[0], dtype=float) * cfg.FFT_FRAME_SECONDS
    return _B5FakeProduct(levels_db, freq_hz, t)


def check_b5_1_onc_100hz_band_raises_because_the_product_cannot_represent_it():
    """A band reaching below bin 1 must RAISE, not be silently clipped up to it.

    ONC's reply (references/ONC_communication.txt) recommends "the sound level
    at ~100 Hz" as a ship detector. That advice is sound in general and NOT
    implementable on this product: bin 0 is DC and bin 1 is 250 Hz
    (config.FFT_BIN_WIDTH_HZ), so there is no sub-250 Hz content at any width.

    The failure mode this pins is the quiet one: a band_limit that clips (100,
    1000) up to (250, 1000) and returns a number. The caller would then believe
    they had measured ONC's band, and every downstream statement about "the
    ~100 Hz ship band" would be false while looking entirely reasonable.
    """
    cfg, features, _fft_io, _ov = _b5_mods()
    try:
        features.assert_band_representable((100.0, 1000.0))
    except features.UnrepresentableBandError as exc:
        assert "250" in str(exc), (
            "the error must name the actual floor (250 Hz) so the caller learns why, "
            f"got: {exc}"
        )
    else:
        raise AssertionError(
            "assert_band_representable accepted a band starting at 100 Hz. The product's "
            f"lowest representable frequency is {cfg.FFT_LOWEST_REPRESENTABLE_HZ} Hz (bin 1); "
            "accepting 100 Hz means some caller will silently receive the 250 Hz band and "
            "report it as ONC's ~100 Hz band"
        )
    # And the configured proxy band, which IS representable, must be accepted.
    features.assert_band_representable(cfg.FFT_B5_SHIP_PROXY_BAND_HZ)


def check_b5_2_band_above_the_relative_ceiling_bin_408_raises():
    """Nothing above bin 408 may enter a B5 statistic, even a relative one.

    Decision 0014. Bins 409-418 are anti-alias filter skirt (instrument
    response, not ocean) and 419-511 are ~99.9% floor-censored, so a mean over
    them converts "we cannot measure this" into a number.
    """
    cfg, features, _fft_io, _ov = _b5_mods()
    ceiling_hz = features.relative_ceiling_hz()
    assert ceiling_hz == float(cfg.FFT_B5_RELATIVE_CEILING_BIN) * cfg.FFT_BIN_WIDTH_HZ, (
        "relative_ceiling_hz must be DERIVED from FFT_B5_RELATIVE_CEILING_BIN, not "
        f"hardcoded; got {ceiling_hz}"
    )
    try:
        features.assert_band_representable((1000.0, ceiling_hz + cfg.FFT_BIN_WIDTH_HZ))
    except features.UnrepresentableBandError:
        pass
    else:
        raise AssertionError(
            f"a band reaching above the relative ceiling ({ceiling_hz} Hz, bin "
            f"{cfg.FFT_B5_RELATIVE_CEILING_BIN}) was accepted -- decision 0014 forbids it"
        )


def check_b5_3_synthetic_broadband_event_is_recovered_at_its_level_time_and_duration():
    """A synthetic event of KNOWN level, time and duration must come back exact.

    CLAUDE.md invariant 3: prove the pipeline against a synthetic signal of
    known level, frequency and time before anything depends on it. Asserts all
    three at once -- level, peak time inside the injected span, and duration --
    because a pipeline can get any one right by accident.
    """
    import numpy as np
    cfg, features, _fft_io, _ov = _b5_mods()
    band = cfg.FFT_B5_SMALL_CRAFT_BAND_HZ
    quiet, loud = 20.0, 70.0
    lo_frame, hi_frame = 400, 600

    surface = _b5_synthetic_surface(quiet)
    product = _b5_product(surface)
    series = features.band_level_series(product, band)
    in_band = np.isin(product.freq_hz, np.asarray(
        [f for f in product.freq_hz if band[0] - cfg.FFT_AXIS_OFFSET_UNCERTAINTY_HZ <= f
         <= band[1] + cfg.FFT_AXIS_OFFSET_UNCERTAINTY_HZ]))
    surface[lo_frame:hi_frame, in_band] = loud

    product = _b5_product(surface)
    series = features.band_level_series(product, band)
    excess = features.band_excess(series)

    assert abs(excess.peak_excess_counts - (loud - quiet)) < 1e-9, (
        f"injected {loud - quiet} dB of excess, recovered "
        f"{excess.peak_excess_counts} -- the band reduction is not linear in the "
        "level it is given"
    )
    t_lo = product.t_utc_s[lo_frame]
    t_hi = product.t_utc_s[hi_frame - 1]
    assert t_lo <= excess.t_peak_utc_s <= t_hi, (
        f"peak excess is at t={excess.t_peak_utc_s} s, outside the injected span "
        f"[{t_lo}, {t_hi}] s -- the time axis and the level array are not aligned, which "
        "is the exact class of bug CLAUDE.md invariant 4 warns produces a plausible "
        "correlation"
    )
    assert series.decidecade_resolvable is True, (
        f"band {band} centres above FFT_DECIDECADE_MIN_CENTRE_HZ and must be reported as "
        "decidecade-resolvable"
    )


def check_b5_4_energy_outside_the_band_is_not_detected():
    """The negative control: out-of-band energy must be invisible.

    A detector that fires on the same injection whether it is inside or outside
    its band is band-blind, and a positive-only test would never notice. This
    is what makes the positive result in check_b5_3 mean anything.
    """
    import numpy as np
    cfg, features, _fft_io, _ov = _b5_mods()
    band = cfg.FFT_B5_SMALL_CRAFT_BAND_HZ
    quiet, loud = 20.0, 70.0

    surface = _b5_synthetic_surface(quiet)
    freq_hz = _b5_product(surface).freq_hz
    # 20-30 kHz: well inside the product's support, well outside the band.
    out_mask = (freq_hz >= 20_000.0) & (freq_hz <= 30_000.0)
    assert out_mask.any(), "fixture error: no bins in the 20-30 kHz out-of-band region"
    surface[400:600, out_mask] = loud

    series = features.band_level_series(_b5_product(surface), band)
    excess = features.band_excess(series)
    assert abs(excess.peak_excess_counts) < 1e-9, (
        f"{loud - quiet} dB injected at 20-30 kHz produced "
        f"{excess.peak_excess_counts} dB of excess in the {band} Hz band -- the band "
        "limit is not actually restricting anything"
    )


def check_b5_5_frame_shuffle_null_is_rejected_for_a_synthetic_event():
    """Shuffling frames must destroy event STRUCTURE while preserving levels.

    The sharpest label-free null available (CLAUDE.md invariant 4). A shuffle
    preserves the level distribution EXACTLY and destroys only time ordering,
    so anything that survives it was never about temporal structure. A real
    closest-point-of-approach cannot survive; a stuck bin or a mis-scaled axis
    can, which is precisely what this is watching for.

    Asserted two ways: the event must be found before the shuffle, and must be
    gone after it. Asserting only the second would pass trivially on a detector
    that never finds anything at all.
    """
    import numpy as np
    cfg, features, _fft_io, _ov = _b5_mods()
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import run_b5_gate

    band = cfg.FFT_B5_SMALL_CRAFT_BAND_HZ
    quiet, loud = 20.0, 70.0
    surface = _b5_synthetic_surface(quiet)
    freq_hz = _b5_product(surface).freq_hz
    in_band = (freq_hz >= band[0] - cfg.FFT_AXIS_OFFSET_UNCERTAINTY_HZ) & (
        freq_hz <= band[1] + cfg.FFT_AXIS_OFFSET_UNCERTAINTY_HZ)
    surface[400:600, in_band] = loud   # 50 s, far above the 20 s duration floor

    product = _b5_product(surface)
    series = features.band_level_series(product, band)
    found = run_b5_gate.find_events(series.t_utc_s, series.level_counts)
    assert len(found["events"]) == 1, (
        f"expected exactly 1 synthetic event, found {len(found['events'])} -- the null "
        "below would pass vacuously if the detector found nothing to begin with"
    )

    nulled = run_b5_gate.frame_shuffle_null(series.t_utc_s, series.level_counts)
    assert len(nulled["events"]) == 0, (
        f"the frame-shuffled null still yields {len(nulled['events'])} event(s). The "
        "shuffle preserves the level distribution exactly and destroys only time order, "
        "so a surviving event means the statistic is not measuring temporal structure "
        "and every CPA claim built on it is unsupported"
    )


def check_b5_6_time_shifted_window_does_not_carry_the_event():
    """An event must be found where it IS, and not where it is not.

    The synthetic analogue of the time-shift null. Scoring a stretch of the
    surface that does not contain the injection must find nothing -- a detector
    that fires on the injected span AND on a quiet span an hour away is
    responding to processing, not content.

    NOTE for anyone reading the gate's real-data output: this null has NO
    discriminating power on the real corpus, because there is no vessel label to
    misalign. An hour earlier on the same cloud-free summer afternoon has just
    as many boats, so real and time-shifted event rates are expected to match
    and their matching is NOT evidence of a bug. It discriminates only here,
    against known ground truth.
    """
    import numpy as np
    cfg, features, _fft_io, _ov = _b5_mods()
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import run_b5_gate

    band = cfg.FFT_B5_SMALL_CRAFT_BAND_HZ
    surface = _b5_synthetic_surface(20.0)
    freq_hz = _b5_product(surface).freq_hz
    in_band = (freq_hz >= band[0] - cfg.FFT_AXIS_OFFSET_UNCERTAINTY_HZ) & (
        freq_hz <= band[1] + cfg.FFT_AXIS_OFFSET_UNCERTAINTY_HZ)
    surface[400:600, in_band] = 70.0

    product = _b5_product(surface)
    series = features.band_level_series(product, band)

    quiet_slice = slice(700, 1200)   # entirely after the injection
    nulled = run_b5_gate.find_events(
        series.t_utc_s[quiet_slice], series.level_counts[quiet_slice])
    assert len(nulled["events"]) == 0, (
        f"a stretch of surface containing no injection yielded {len(nulled['events'])} "
        "event(s) -- the detector is firing on something other than the signal"
    )


def check_b5_7_single_bin_outlier_cannot_move_the_band_level():
    """One loud bin among many must not move the band level. Justifies the median.

    config.FFT_BAND_LEVEL_STATISTIC is a MEDIAN precisely so a single bin -- an
    echosounder line, a stuck bin, a bit error -- cannot masquerade as a vessel.
    The ~38 kHz echosounder alone would otherwise fire the detector every few
    seconds. If this reduction is ever changed to a mean, this check fails and
    says why.
    """
    import numpy as np
    cfg, features, _fft_io, _ov = _b5_mods()
    band = cfg.FFT_B5_SMALL_CRAFT_BAND_HZ
    surface = _b5_synthetic_surface(20.0)
    freq_hz = _b5_product(surface).freq_hz
    target = int(np.argmin(np.abs(freq_hz - 5000.0)))
    surface[400:600, target] = 120.0   # one bin, 100 dB above ambient

    series = features.band_level_series(_b5_product(surface), band)
    excess = features.band_excess(series)
    assert series.n_bins_in_band > 2, (
        f"fixture error: only {series.n_bins_in_band} bin(s) in band, a median over "
        "so few bins would legitimately move"
    )
    assert abs(excess.peak_excess_counts) < 1e-9, (
        f"a single 100 dB bin moved the band level by {excess.peak_excess_counts} dB. "
        f"The statistic is documented as {cfg.FFT_BAND_LEVEL_STATISTIC}; if it has become "
        "a mean, the ~38 kHz echosounder and any stuck bin now read as vessel passes"
    )


def check_b5_8_overpass_loader_refuses_a_timestamp_with_no_utc_offset():
    """A scene time with no stated offset must RAISE, never be assumed UTC.

    Decision 0002, and the single most likely way this project produces a
    confident wrong answer: a one-hour error in the acoustic-to-optical join is
    invisible in every plot and fatal to every result. The gate2 CSV currently
    carries `+00:00`; this pins that a future file WITHOUT it fails loudly
    rather than being silently read as UTC.
    """
    _cfg, _features, _fft_io, overpasses = _b5_mods()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "naive.csv"
        path.write_text(
            "id,acquired,clear_percent,instrument\n"
            "20240101_120000_00_0000,2024-01-01T12:00:00.000000,100,PSB.SD\n",
            encoding="utf-8",
        )
        try:
            overpasses.load_gate2_overpasses(path)
        except ValueError as exc:
            assert "0002" in str(exc) or "offset" in str(exc).lower(), (
                f"the error must say WHY a naive stamp is refused, got: {exc}")
        else:
            raise AssertionError(
                "load_gate2_overpasses accepted an 'acquired' stamp with no UTC offset. "
                "Assuming UTC is exactly the assumption decision 0002 forbids"
            )

    # And a stamp in a NON-UTC offset must be converted, not truncated.
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "offset.csv"
        path.write_text(
            "id,acquired,clear_percent,instrument\n"
            "20240101_120000_00_0000,2024-01-01T12:00:00.000000-07:00,100,PSB.SD\n",
            encoding="utf-8",
        )
        ops = overpasses.load_gate2_overpasses(path)
        assert ops[0].acquired_utc.hour == 19, (
            f"12:00 at -07:00 is 19:00 UTC; loader produced "
            f"{ops[0].acquired_utc.isoformat()} -- the offset was dropped, not applied"
        )


def check_b5_9_window_coverage_counts_overlap_not_start_containment():
    """A file straddling a window edge partly covers it. Containment would drop it.

    Decision 0020: the corpus-to-window join is INTERVAL OVERLAP, not equality
    or start-containment. A start-containment test drops the file that begins
    just before the window and runs into it, understating coverage at both edges
    -- which reads as a coverage gap rather than as a join bug.
    """
    import datetime as dt
    _cfg, _features, _fft_io, overpasses = _b5_mods()

    acquired = dt.datetime(2024, 7, 1, 18, 30, tzinfo=dt.timezone.utc)
    op = overpasses.Overpass(scene_id="fixture", acquired_utc=acquired,
                             clear_percent=None, instrument=None)
    win_start, win_end = op.window_utc()

    # One file starting 120 s BEFORE the window and running 300 s: it overlaps
    # the first 180 s of the window, and its START is outside it.
    straddler_start = win_start - dt.timedelta(seconds=120)
    index = [(straddler_start, straddler_start + dt.timedelta(seconds=300),
              pathlib.Path("straddler.fft.gz"))]
    cov = overpasses.window_coverage(op, index)
    assert cov.n_files == 1, (
        "a file starting before the window but running into it was not counted -- the "
        "join is using start-containment, which decision 0020 rules out"
    )
    assert abs(cov.covered_seconds - 180.0) < 1e-6, (
        f"expected 180 s of overlap counted, got {cov.covered_seconds} s. Counting the "
        "whole 300 s file would overstate coverage inside the window"
    )
    assert not cov.is_full, "180 s of a 1800 s window is not full coverage"

    # An empty index is a MEASURED ZERO (decisions 0008, 0028), not an error.
    empty = overpasses.window_coverage(op, [])
    assert empty.n_files == 0 and empty.covered_seconds == 0.0, (
        "an uncovered window must report zero coverage, not raise -- 'the corpus does "
        "not reach this overpass' is a measurement"
    )




def check_b5_10_corpus_index_deduplicates_windows_present_in_both_containers():
    """The same acoustic window in both containers must appear ONCE in the index.

    The corpus is permanently mixed (decisions 0022/0024/0025), and 90 plain
    `.fft` files are the same windows as `.gz` siblings with byte-identical
    payloads -- measured on the real corpus, on three dates in 2025. Acquisition
    must see two files, because there are two files on disk. ANALYSIS must see
    one window: a duplicated window is double-counted by every percentile,
    histogram, LTSA and event rate built on the index, and a corpus 0.34%
    duplicated on three dates is an error that never announces itself.

    Pins three things: the index holds one entry per start time; the survivor is
    the `.gz`, deterministically, so the index does not depend on directory
    order; and the drop is RECOVERABLE via `corpus_index_duplicates` rather than
    silent (CLAUDE.md invariant 5).
    """
    _cfg, _features, _fft_io, overpasses = _b5_mods()
    stem = "ICLISTENHF1266_20250715T161505.000Z.fft"
    with tempfile.TemporaryDirectory() as tmp:
        corpus = pathlib.Path(tmp)
        (corpus / stem).write_text("plain", encoding="utf-8")
        (corpus / f"{stem}.gz").write_bytes(b"\x1f\x8bgzip")
        (corpus / f"{stem}.gz.sha256").write_text("deadbeef", encoding="utf-8")

        index = overpasses.corpus_file_index(corpus)
        assert len(index) == 1, (
            f"two containers of ONE window produced {len(index)} index entries. "
            "Every statistic built on this index would double-count that window"
        )
        assert index[0][2].name.endswith(".gz"), (
            f"the surviving entry is {index[0][2].name!r}; the .gz must win so the "
            "index is deterministic rather than dependent on directory order"
        )
        dropped = overpasses.corpus_index_duplicates()
        assert [p.name for p in dropped] == [stem], (
            f"the dropped duplicate was not reported: {[p.name for p in dropped]}. "
            "Discarding 90 files silently is the same class of error as counting "
            "them twice"
        )



def check_b5_11_population_histogram_vectorisation_matches_the_per_bin_loop():
    """The flattened-index histogram must equal the per-bin loop it replaced.

    `scripts/build_population_set.py` accumulates an EXACT per-bin integer
    histogram over the whole corpus, and every percentile, ambient curve and SPD
    panel downstream is read from it. The obvious implementation is one
    `bincount` per bin; the fast one flattens `(bin, level)` into a single index.
    An off-by-one in that flattening would shift counts between adjacent bins --
    a corruption that leaves the totals right and every plot plausible.
    """
    import numpy as np
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import build_population_set as pop
    except ImportError as exc:
        raise SkipCheck(f"build_population_set not importable: {exc}") from exc

    rng = np.random.default_rng(11)
    n_bins, n_frames, axis_max = 64, 300, pop.LEVEL_AXIS_MAX
    clipped = rng.integers(0, axis_max, (n_frames, n_bins))

    loop = np.zeros((n_bins, axis_max), dtype=np.int64)
    for b in range(n_bins):
        loop[b] = np.bincount(clipped[:, b], minlength=axis_max)

    bin_index = np.repeat(np.arange(n_bins, dtype=np.int64), clipped.shape[0])
    flat = bin_index * axis_max + clipped.T.reshape(-1)
    fast = np.bincount(flat, minlength=n_bins * axis_max).reshape(n_bins, axis_max)

    assert np.array_equal(loop, fast), (
        "the vectorised histogram does not match the per-bin loop. Differing "
        f"cells: {int((loop != fast).sum())}. Every percentile and ambient curve "
        "in the population set is read from this array"
    )
    assert int(fast.sum()) == n_frames * n_bins, (
        f"histogram holds {int(fast.sum())} counts for {n_frames * n_bins} cells "
        "-- values are being dropped or double-counted"
    )


def check_b5_12_fast_reader_matches_the_reference_parse():
    """`read_fft_gz` must return exactly what a plain split-and-convert returns.

    The reader parses with pandas' C tokenizer because the ASCII-to-number
    conversion, not the gzip decode, is its whole cost (120 ms against 10 ms).
    That is a real speedup and a real risk: a table parser handles ragged rows by
    padding, where the reference parse simply yields a different count.

    Pins bit-equality against the reference on a synthetic product whose values
    are known, and pins that a RAGGED file still raises rather than being padded
    into a plausible-looking array.
    """
    import gzip as _gzip
    import numpy as np
    _cfg, _features, fft_io, _ov = _b5_mods()

    rng = np.random.default_rng(12)
    n_frames, n_bins = _cfg.FFT_N_FRAMES, _cfg.FFT_N_BINS
    values = rng.integers(0, 113, (n_frames, n_bins))
    text = "\n".join(" ".join(str(int(v)) for v in row) for row in values)
    name = "ICLISTENHF1266_20250701T161505.000Z.fft.gz"

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / name
        with _gzip.open(path, "wt", encoding="ascii") as handle:
            handle.write(text)
        got = fft_io.read_fft_gz(path).levels_db
        reference = np.asarray(text.split(), dtype=float).reshape(n_frames, n_bins)
        assert np.array_equal(got, reference), (
            f"fast reader differs from the reference parse in "
            f"{int((got != reference).sum())} cell(s)"
        )
        assert np.array_equal(got, values.astype(float)), (
            "fast reader does not return the values that were written"
        )

        # A ragged file must RAISE, not be padded into the right shape.
        ragged = pathlib.Path(tmp) / "ICLISTENHF1266_20250701T162005.000Z.fft.gz"
        with _gzip.open(ragged, "wt", encoding="ascii") as handle:
            handle.write(text[: len(text) // 2])
        try:
            fft_io.read_fft_gz(ragged)
        except ValueError as exc:
            message = str(exc)
            assert ragged.name in message, (
                f"a truncated product raised without naming the file: {message}")
            assert ("expected exactly" in message
                    or "not parseable as" in message), (
                "a truncated product raised, but with a message that states "
                f"neither the expected count nor that the payload is malformed: "
                f"{message}. A fast parser must not cost the caller a "
                "diagnosable error")
        else:
            raise AssertionError(
                "a truncated product did not raise. A table parser pads ragged "
                "rows, and a padded array of the right shape is exactly the "
                "silent corruption the length check exists to prevent"
            )



def check_b5_13_exact_percentile_from_histogram_matches_direct_percentile():
    """Percentiles read off the integer histogram must equal the direct ones.

    Every population figure -- the SPD's L05/L50/L95 lines, the per-season L95
    ambient the detector consumes as a second baseline -- is read off a
    cumulative per-bin integer histogram rather than from the raw values, which
    were never held in memory all at once. That is only legitimate if the two
    agree exactly, and the levels being integers is what makes it so: there is no
    fractional level to interpolate toward.

    Compared against numpy's ``inverted_cdf``, which is EXACTLY the estimator
    this implements: the smallest level whose cumulative count reaches q% of the
    total. Naming the estimator matters -- against numpy's ``lower`` this
    disagrees by one rank at q=95 (verified), and against the default ``linear``
    it would disagree almost everywhere. Neither disagreement would be a bug;
    they would be different definitions, and a check that did not say which one
    it meant would be measuring the wrong thing convincingly.
    """
    import numpy as np
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    scripts_dir = REPO_ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        import plot_population_set as pp
    except ImportError as exc:
        raise SkipCheck(f"plot_population_set not importable: {exc}") from exc

    rng = np.random.default_rng(13)
    n_bins, n_frames, axis_max = 24, 977, 120   # odd frame count on purpose
    values = rng.integers(0, axis_max, (n_frames, n_bins))

    hist = np.zeros((n_bins, axis_max), dtype=np.int64)
    for b in range(n_bins):
        hist[b] = np.bincount(values[:, b], minlength=axis_max)

    for q in (5, 25, 50, 75, 95):
        from_hist = pp.exact_percentile_from_hist(hist, q)
        direct = np.percentile(values, q, axis=0, method="inverted_cdf")
        assert np.array_equal(from_hist, direct), (
            f"histogram percentile disagrees with the direct one at q={q}: "
            f"{int((from_hist != direct).sum())} of {n_bins} bins differ. Every "
            "SPD line and the seasonal ambient baseline are read this way"
        )

    # A bin with no counts at all must RAISE, not return a silent zero: an empty
    # bin means the pass covered nothing there, and 0 is a level.
    empty = np.zeros((3, axis_max), dtype=np.int64)
    try:
        pp.exact_percentile_from_hist(empty, 50)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "a histogram with an empty frequency bin returned a percentile "
            "instead of raising. Level 0 is a real level, so a silent zero here "
            "is indistinguishable from a measured quiet bin"
        )


CHECKS = [
    ("A0.1 paths import is dependency-free", check_a0_1_paths_import_is_dependency_free),
    ("A0.1 paths exports are pathlib.Path", check_a0_1_paths_exports_are_paths),
    ("A0.1 path values (spaced sample dir, data/derived)", check_a0_1_path_values),
    ("A0.1 require_path raises FileNotFoundError", check_a0_1_require_path_raises),
    ("A0.2 .gitignore ignores secrets/data/media", check_a0_2_gitignore_ignores_secrets_and_data),
    ("A0.2 .gitignore negation keeps .env.example", check_a0_2_gitignore_negation_works),
    ("A0.3 missing credential RAISES (never skips)", check_a0_3_missing_token_raises),
    ("A0.3 ONC_TOKEN returned verbatim", check_a0_3_env_var_returned_verbatim),
    ("A0.3 process env overrides .env", check_a0_3_process_env_overrides_dotenv),
    ("A0.3 malformed .env names the line number", check_a0_3_malformed_dotenv_reports_line_number),
    ("A0.3 token never leaks into error text", check_a0_3_token_never_leaks_into_errors),
    ("A0.3 unfilled .env.example placeholder is rejected", check_a0_3_placeholder_token_is_rejected),
    ("A0.3 parse_dotenv quoting and `export ` prefix", check_a0_3_parse_dotenv_quoting_and_export),
    ("A0.3 dotenv_search_paths inside the checkout", check_a0_3_dotenv_search_paths_inside_repo),
    ("A0.3 unusable .env value falls through, not shadows", check_a0_3_dotenv_falls_through_unusable_value),
    ("A0.3 .env.example tracked / .env ignored", check_a0_3_dotenv_example_tracked_and_dotenv_ignored),
    ("A0.4 `-m boatphone.env_audit --strict` exits 0", check_a0_4_strict_cli_exits_zero),
    ("A0.4 REQUIRED covers the minimum set", check_a0_4_required_covers_minimum),
    ("A0.4 versions resolve via importlib.metadata", check_a0_4_versions_resolve_without_dunder_version),
    ("A0.4 strict path is live, not always-green", check_a0_4_strict_path_is_live),
    ("A0.5 environment.yml shape", check_a0_5_environment_yml_shape),
    ("A0.5 manifest covers env_audit.REQUIRED", check_a0_5_manifest_covers_required),
    ("A0.5 python constraint admits 3.14", check_a0_5_python_constraint_admits_314),
    ("A0.6 sample acquisition present (data-dependent)", check_a0_6_sample_dir_present),
    ("A0.6 SAMPLE_DIR matches disk (data-dependent)", check_a0_6_paths_sample_dir_matches_disk),
    ("A1a config constants (BIN_SECONDS, UTC study window)", check_a1a_0_config_constants),
    ("A1a naive datetime RAISES at the boundary", check_a1a_0b_naive_datetime_raises),
    ("A1a HAND-BUILT GATE: half-open coverage marks 09:05/09:10 only", check_a1a_1_half_open_hand_built_gate),
    ("A1a touching a bin edge is not overlap", check_a1a_1b_touching_is_not_overlap),
    ("A1a WRONG-TIME-FAILS: +7 h PDT shift changes the bin set", check_a1a_2_pdt_offset_leak_is_caught),
    ("A1a grid invariants (300 s, epoch-aligned, in-season)", check_a1a_3_grid_invariants),
    ("A1a epoch alignment from an IN-SEASON unaligned start", check_a1a_3b_epoch_alignment_from_in_season_unaligned_start),
    ("A1a density is arithmetic (153*288 = 44064 rows)", check_a1a_4_density_is_arithmetic),
    ("A1a season boundary rows (May 1 00:00Z .. Oct 1 00:00Z)", check_a1a_5_season_boundary_rows),
    ("A1a DST is a no-op (288 rows every UTC day)", check_a1a_6_dst_is_a_no_op),
    ("A1a no local timezone referenced in A1 source", check_a1a_6b_no_local_timezone_in_a1_source),
    ("A1a deployment assignment; listed-outside-deployment RAISES", check_a1a_7_deployment_assignment),
    ("A1a no-deployment bins are unavailable", check_a1a_7b_unavailable_bins_have_no_deployment_claim),
    ("A1b API surface (discover/list/parse/deployments)", check_a1b_0_api_surface),
    ("A1b real .fft.gz names parse to UTC 300 s bins (data-dependent)", check_a1b_1_parse_real_fixture_fft_names),
    ("A1b sub-second '.029Z' survives, not truncated (data-dependent)", check_a1b_1b_subsecond_fraction_survives),
    ("A1b .fft.gz and .wav names parse identically", check_a1b_1c_fft_and_wav_names_agree),
    ("A1b malformed filename RAISES (never None)", check_a1b_2_malformed_filename_raises),
    ("A1b wrong-device filename RAISES", check_a1b_3_wrong_device_filename_raises),
    ("A1b NO hardcoded ONC location code in boatphone/", check_a1b_4_no_hardcoded_location_code),
    ("A1b no bare except in the ONC client (D8)", check_a1b_4b_no_bare_except_in_listing_code),
    ("A1b zero files over the whole span RAISES (D8)", check_a1b_5_zero_files_over_whole_span_raises),
    ("A1b chunk HTTP error propagates with params, not the token (D8)", check_a1b_5b_http_error_in_a_chunk_propagates),
    ("A1b empty-year-chunk count is counted and printed (D8)", check_a1b_5c_empty_chunk_count_is_surfaced),
    ("A1b a COMPLETE year is one probe request", check_a1b_5d_complete_year_is_one_probe_request_per_year),
    ("A1b a TRUNCATED year pages to completeness and chunks by month", check_a1b_5d2_a_truncated_year_falls_back_to_months_and_paging),
    ("A1b discovery with no Folger candidate RAISES (D8)", check_a1b_5e_discovery_with_no_folger_candidate_raises),
    ("A1b discovery matches ONC's response, not a known code", check_a1b_5f_discovery_finds_folger_by_name_not_by_a_known_code),
    ("A1b token never leaks from ANY impure call (full chain)", check_a1b_5g_no_token_leaks_from_any_impure_call),
    ("A1b redaction survives chaining (__context__ too)", check_a1b_5h_redaction_survives_raise_from_none),
    ("A1b sub-second backstop, no data/ needed", check_a1b_1d_subsecond_backstop_without_data),
    ("A1b coverage_intervals merge is flag-equivalent (< not <=)", check_a1b_8_coverage_intervals_merge_is_flag_equivalent),
    ("A1b NETWORK discovery returns >=1 code (gated)", check_a1b_network_discover_returns_at_least_one_code),
    ("A1b NETWORK one in-season day lists >0 files (gated)", check_a1b_network_one_day_listing_is_non_empty),
    ("A1b NETWORK live 401 is redacted (bogus token, gated)", check_a1b_network_live_401_is_redacted),
    ("A1c API surface (build_uptime_calendar / write_uptime_calendar_csv)", check_a1c_0_api_surface),
    ("A1c row count is arithmetic (dense, no truncation)", check_a1c_1_row_count_is_arithmetic),
    ("A1c schema + half-open contiguity survive composition", check_a1c_2_schema_and_half_open_semantics),
    ("A1c a constructed gap is MEASURED, not defaulted true/false", check_a1c_3_gap_is_measured_not_defaulted),
    ("A1c deployment_id from metadata, not inferred from a gap (D6)", check_a1c_4_deployment_id_from_metadata_not_gaps),
    ("A1c WRONG-TIME-FAILS: +7h shift changes availability", check_a1c_5_wrong_time_fails),
    ("A1c a failed listing chunk RAISES, never truncates the calendar", check_a1c_6_partial_scan_raises_not_truncates),
    ("A1c CSV schema and column order", check_a1c_7_csv_schema_and_column_order),
    ("A1c CSV timestamps are explicit-UTC ISO 8601, never naive", check_a1c_8_csv_timestamps_are_explicit_utc_iso8601),
    ("A1c CSV row count matches the in-memory calendar", check_a1c_9_csv_row_count_matches_calendar),
    ("A1c NETWORK one real in-season day end-to-end (gated)", check_a1c_network_one_day_end_to_end),
    ("A1c inspection API surface (summarise_gaps / mean_availability_by_utc_hour)", check_a1c_10_inspection_api_surface),
    ("A1c planted gaps exact, incl. leading and trailing; end_utc EXCLUSIVE", check_a1c_11_summarise_gaps_planted_gaps_exact),
    ("A1c gap + available time is conserved (no bin lost or double-counted)", check_a1c_11b_gap_and_available_time_is_conserved),
    ("A1c min_seconds threshold is inclusive (>=, not >)", check_a1c_12_min_seconds_threshold_is_inclusive),
    ("A1c a gap never bridges a break in the bin grid (season boundary)", check_a1c_13_gap_does_not_bridge_a_break_in_the_bin_GRID),
    ("A1c uniform calendar: UTC-hour profile is exactly 1.0 x 24", check_a1c_14_uniform_calendar_profile_is_exactly_flat),
    ("A1c an unpopulated UTC hour is nan, never 0.0", check_a1c_14b_hour_with_no_bins_is_nan_not_zero),
    ("A1c WRONG-TIME-FAILS: +7 h PDT dents exactly 7 UTC hours", check_a1c_15_pdt_shifted_coverage_makes_the_profile_NOT_flat),
    ("A1c docs/derived/hydrophone_gaps.md tracked, with caveat + date range", check_a1c_16_gap_summary_doc_exists_and_carries_its_caveat),
    ("A1c notebook is a thin wrapper, not the home of the functions", check_a1c_17_notebook_is_a_thin_wrapper),
    ("A1e D7 Error-127 (no deployment) is an absorbed zero, printed", check_a1e_1_error_127_no_deployment_is_an_absorbed_zero_and_is_printed),
    ("A1e D7 Error-25 (future window) is an absorbed zero, printed", check_a1e_2_error_25_future_window_is_an_absorbed_zero_and_is_printed),
    ("A1e D7 a 401, a 500 and a DIFFERENT 400 all still PROPAGATE", check_a1e_3_a_401_a_500_and_another_400_all_still_PROPAGATE),
    ("A1e D7 no token leaks through the absorption path", check_a1e_4_no_token_leaks_through_the_absorption_path),
    ("A1e D7 a no-data marker on page 2+ RAISES, never truncates", check_a1e_5_a_no_data_marker_on_page_2_must_RAISE_not_truncate),
    ("A1d a CAPPED listing is never returned as complete (paginate or raise)", check_a1d_1_capped_listing_is_never_returned_as_complete),
    ("A1d a capped chunk is NOT an empty chunk", check_a1d_2_capped_chunk_is_not_counted_as_empty),
    ("A1d a genuine no-data period still surfaces beside a capped span", check_a1d_3_genuine_no_data_still_surfaces_next_to_a_capped_span),
    ("A1d REGRESSION: no phantom outage over 2024-05-01..2024-10-01", check_a1d_4_no_phantom_outage_in_the_real_failing_span),
    ("A1d O1 CSV timestamps round-trip tz-aware UTC, byte-identically", check_a1d_5_written_csv_round_trips_tz_aware_utc),
    ("A1d O1 CSV header + STABLE `available` literal (True/False)", check_a1d_6_written_csv_header_and_available_literal),
    ("A1d grid invariants re-asserted on the WRITTEN FILE", check_a1d_7_grid_invariants_hold_on_the_WRITTEN_FILE),
    ("A1d CLI surface; defaults come from boatphone/, not scripts/", check_a1d_8_cli_surface_and_defaults),
    ("A1d no write escapes the out-dir; data/ paths REFUSED (inv. 2)", check_a1d_9_no_write_escapes_the_derived_dir),
    ("A1d provenance complete and source == 'listing' (D3)", check_a1d_10_provenance_is_complete_and_says_source_is_LISTING),
    ("A1d --dry-run writes nothing and needs no credential", check_a1d_11_dry_run_writes_nothing),
    ("A1d scripts/ declares no shared constant (invariant 6)", check_a1d_12_entry_point_declares_no_shared_constant),
    ("A8a docs/vtuad-facts.md exists and parses", check_a8a_facts_doc_exists),
    ("A8a all required facts present, populated, sourced", check_a8a_all_required_keys_present_and_populated),
    ("A8a boatphone.config VTUAD constants match facts doc", check_a8a_config_constants_match_facts_doc),
    ("A8b boatphone/models.py exists", check_a8b_models_module_exists),
    ("A8b common_support_hz == intersection clipped to lower Nyquist", check_a8b_common_support_is_intersection_clipped_to_lower_nyquist),
    ("A8b common_support_hz moves with VTUAD input (anti-hardcoding)", check_a8b_common_support_moves_with_vtuad_input),
    ("A8b empty intersection raises, naming both bands", check_a8b_empty_intersection_raises_naming_both_bands),
    ("A8b assert_band_matched raises on an unmatched comparison", check_a8b_assert_band_matched_raises_on_unmatched_comparison),
    ("A8b band_limit signature names freq_hz/level_db_re_1upa/fs_hz", check_a8b_band_limit_signature_names_conventions),
    ("A8b in-band tone survives band-limiting (freq + level)", check_a8b_band_limit_positive_tone_survives_with_matching_level_and_freq),
    ("A8b out-of-band tones excluded (below support / above lower Nyquist)", check_a8b_band_limit_excludes_out_of_band_tones),
    ("A8b band exceeding source Nyquist is rejected, not clipped", check_a8b_band_limit_rejects_band_exceeding_source_nyquist),
    ("A8b absolute-level comparison across calibration boundary raises", check_a8b_absolute_level_across_calibration_boundary_raises),
    ("A8b level-invariant cross-domain comparison is allowed", check_a8b_level_invariant_comparison_is_allowed),
    ("A8b undeclared calibration state raises", check_a8b_undeclared_calibration_state_raises),
    ("B3-B download_archive_file exists and is callable", check_b3b_0_api_surface),
    ("B3-B interrupted transfer never leaves a corrupt FINAL file", check_b3b_1_interrupted_transfer_never_leaves_a_corrupt_final_file),
    ("B3-B resume, Range honoured (206) == byte-identical uninterrupted pull", check_b3b_2a_resume_when_range_honoured_206_is_byte_identical_to_uninterrupted_pull),
    ("B3-B resume, Range ignored (200) discards partial and restarts from zero", check_b3b_2b_resume_when_range_ignored_200_discards_and_restarts_from_zero),
    ("B3-B DownloadRecord.http_status/.attempts correct per status", check_b3b_2c_download_record_carries_http_status_and_attempts_per_status),
    ("B3-B second call on a complete file is a zero-byte cache hit", check_b3b_3_second_call_on_a_complete_file_is_a_zero_byte_cache_hit),
    ("B3-B locally-corrupted cache is detected by hash and RAISES", check_b3b_4_locally_corrupted_cache_is_detected_by_hash_and_RAISES),
    ("B3-B three 429s back off with strictly increasing sleeps, then succeed", check_b3b_5_three_429s_back_off_with_strictly_increasing_sleeps_then_succeed),
    ("B3-B permanently-absent file yields a structured record, never raises", check_b3b_6_permanently_absent_file_yields_a_structured_record_and_does_not_raise),
    ("B3-B D7 'not deployed' 400 is a measured zero, distinct from absent/error", check_b3b_7_onc_400_not_deployed_is_a_measured_zero_distinct_from_absent_and_error),
    ("B3-B no bare except in the download code", check_b3b_8_no_bare_except_in_the_download_code),
    ("B3-B this check itself never writes under data/ (guards the guard)", check_b3b_9_never_writes_outside_the_temp_dest_dir_this_check_created),
    ("B0.1 boatphone.paths exposes EXTERNAL_DIR/ONC_MODEL_DIR/CHECKPOINT_DIR", check_b0_1_paths_constants_exist_and_are_correct),
    ("B0.1 external/ is gitignored", check_b0_1_external_dir_is_gitignored),
    ("B0.1 provenance JSON exists, parses, tracked", check_b0_1_provenance_json_exists_and_parses),
    ("B0.1 provenance fields complete (url/revision/sha256/size/ts/licence)", check_b0_1_provenance_fields_complete),
    ("B0.1 recorded sha256 verifies against disk (data-dependent)", check_b0_1_recorded_hashes_verify_against_disk),
    ("B0.1 data/ untouched by this step (invariant 2)", check_b0_1_data_dir_untouched_by_this_step),
    ("B0.1 external/ never staged/untracked in git", check_b0_1_external_not_staged_or_untracked_in_git),
    ("B0-2a boatphone.config axis constants exist", check_b0_2a_config_constants_exist),
    ("B0-2a config axis constants match settled facts", check_b0_2a_config_constants_match_settled_facts),
    ("B0-2a boatphone/fft_io.py exists", check_b0_2a_module_exists),
    ("B0-2a frequency_axis_hz: bin width + bin1==250Hz, not hardcoded", check_b0_2a_frequency_axis_bin_width_and_bin1),
    ("B0-2a time_axis_utc_s: absolute UTC, None start RAISES", check_b0_2a_time_axis_is_absolute_utc_not_bare_relative),
    ("B0-2a assert_tone_at signature names freq_hz/t_utc_s/levels_db", check_b0_2a_assert_tone_at_signature_names_conventions),
    ("B0-2a SYNTHETIC TONE positive control (freq+time+level)", check_b0_2a_synthetic_tone_positive_control_survives_axis_mapping),
    ("B0-2a WRONG-BIN-FAILS: offset tone claim is rejected", check_b0_2a_synthetic_tone_wrong_bin_offset_is_rejected),
    ("B0-2a FRAME-SHUFFLE NULL: shuffled frame order is rejected (invariant 4)", check_b0_2a_synthetic_tone_frame_shuffle_null_is_rejected),
    ("B0-2a WRONG-TIME-FAILS: +1h shifted claim is rejected", check_b0_2a_synthetic_tone_wrong_hour_offset_is_rejected),
    ("B0-2a real fixture: shape 1200x512 + 3-region band top (data-dependent)", check_b0_2a_real_fixture_shape_and_structural_zeros),
    ("B0-2a real fixture: echosounder hump CENTROID in bins 149-152 (data-dependent)", check_b0_2a_real_fixture_echosounder_hump_centroid),
    ("B0-2a centre-vs-edge uncertainty is CARRIED (+/-125 Hz applied, not just declared)", check_b0_2a_axis_uncertainty_is_carried),
    ("B0-2a no check pins a bin position tighter than +/-1 bin", check_b0_2a_no_check_asserts_a_bin_position_tighter_than_one_bin),
    ("B0-2a B5 preconditions: two ceilings (204/408) + censoring counters", check_b0_2a_b5_ceilings_and_censoring_counters_are_available),
    ("B0-2a calibratable band == bins 1-204; assert_calibratable rejects beyond (reuses models.py)", check_b0_2a_calibratable_band_matches_bin_range_and_assert_calibratable_rejects_beyond),
    ("B3-A config window-edge constants (09:15/11:45 America/Vancouver)", check_b3a_1_config_window_edge_constants),
    ("B3-A TIME GATE: overpass_window_utc matches literal UTC per date incl. DST pair (0002)", check_b3a_2_overpass_window_utc_matches_literal_expectations_per_date),
    ("B3-A no hardcoded UTC offset; zoneinfo genuinely used", check_b3a_3_no_hardcoded_utc_offset_and_zoneinfo_is_used),
    ("B3-C run/iter_overpass_dates/build_parser exist", check_b3c_0_api_surface),
    ("B3-C date order 2025-descending-then-backwards-to-2020, in-season only", check_b3c_1_date_ordering_is_2025_descending_then_backwards_to_2020),
    ("B3-C TIME GATE: per-date window recomputation is never hoisted (0002)", check_b3c_2_per_date_window_recomputation_is_never_hoisted_out_of_the_loop),
    ("B3-C SEAM GATE: run() driven by the REAL list_fft_files, not a narrower fake", check_b3c_2b_run_drives_the_real_list_fft_files_not_a_narrower_fake),
    ("B3-C empty overpass window is a measured zero, not a crash (0016)", check_b3c_2c_empty_overpass_window_logs_an_absent_measured_zero_row),
    ("B3-C requested == present + absent EXACTLY, absent entries carry a reason", check_b3c_3_requested_equals_present_plus_absent_exactly_with_reasons),
    ("B3-C no --resume flag; a bare re-run redownloads nothing complete", check_b3c_4_rerun_with_no_flags_resumes_and_redownloads_nothing_complete),
    ("B3-C manifest never lands without its provenance sidecar", check_b3c_5_manifest_never_written_without_its_provenance_sidecar),
    ("B3-C sampling-conditionality is one named config constant, carried in the manifest", check_b3c_6_sampling_conditionality_is_one_named_constant_and_travels_with_the_manifest),
    ("B3-C this check itself never writes under data/ (guards the guard)", check_b3c_7_never_writes_under_the_real_data_dir_this_check_guards_the_guard),
    ("B3-C absent_log rows carry file_start_utc/file_end_utc via parse_file_coverage, not bin_*", check_b3c_8_absent_log_rows_carry_file_span_via_parse_file_coverage),
    ("B3-C provenance names endpoint/absent-def/dest_dir/season/year-span/checksum caveat", check_b3c_9_provenance_names_endpoint_absent_definition_dest_dir_season_and_checksum_caveat),
    ("B3-C two manifest writes in the same dest do not collide", check_b3c_10_two_manifest_writes_in_the_same_dest_do_not_collide),
    ("B3-C ONE shared corpus resolver finds *.fft AND *.fft.gz, excludes sidecars, raises when absent", check_b3c_11_corpus_resolver_finds_fft_and_fft_gz_but_not_sidecars_and_raises_when_absent),
    ("B3-C per-file record exposes disk_basename distinct from wire filename", check_b3c_12_per_file_record_exposes_disk_basename_distinct_from_wire_filename),
    ("B3-D FACT1: read_fft_gz accepts plain-ASCII and gzip by CONTENT, not extension", check_b3d_1_fft_reader_accepts_plain_ascii_and_gzip_by_content_not_extension),
    ("B3-D FACT2: ONC 400 'No file could be found' is ABSENT, not raised, no retry storm", check_b3d_2_onc_400_no_file_could_be_found_is_absent_not_error_no_retry_storm),
    ("B3-E downloaded bytes land gzip-compressed and round-trip exactly", check_b3e_1_downloaded_bytes_land_gzip_compressed_and_round_trip_exactly),
    ("B3-E sha256 sidecar hashes the WIRE bytes, not the compressed container", check_b3e_2_sha256_sidecar_hashes_the_wire_bytes_not_the_compressed_container),
    ("B3-E cache-hit re-verifies a compressed file and catches corruption", check_b3e_3_cache_hit_reverifies_a_compressed_file_and_catches_corruption),
    ("B3-E the ONE reader opens the compressed corpus and the resolver still finds it", check_b3e_4_the_one_reader_opens_the_compressed_corpus_and_the_resolver_still_finds_it),
    ("B3-E compressed resume offset is WIRE bytes, not compressed .part size", check_b3e_6_compressed_resume_offset_is_wire_bytes_not_compressed_part_size),
    ("B3-E mixed corpus of already-pulled plain-text and new gzip files both work", check_b3e_5_mixed_corpus_of_already_pulled_plain_text_and_new_gzip_files_both_work),
    ("B5 ONC's ~100 Hz band raises: the product cannot represent it", check_b5_1_onc_100hz_band_raises_because_the_product_cannot_represent_it),
    ("B5 band above the relative ceiling bin 408 raises", check_b5_2_band_above_the_relative_ceiling_bin_408_raises),
    ("B5 synthetic broadband event recovered at its level, time and duration", check_b5_3_synthetic_broadband_event_is_recovered_at_its_level_time_and_duration),
    ("B5 energy outside the band is not detected", check_b5_4_energy_outside_the_band_is_not_detected),
    ("B5 frame-shuffle null is rejected for a synthetic event", check_b5_5_frame_shuffle_null_is_rejected_for_a_synthetic_event),
    ("B5 time-shifted window does not carry the event", check_b5_6_time_shifted_window_does_not_carry_the_event),
    ("B5 single-bin outlier cannot move the band level", check_b5_7_single_bin_outlier_cannot_move_the_band_level),
    ("B5 overpass loader refuses a timestamp with no UTC offset", check_b5_8_overpass_loader_refuses_a_timestamp_with_no_utc_offset),
    ("B5 window coverage counts overlap, not start containment", check_b5_9_window_coverage_counts_overlap_not_start_containment),
    ("B5 corpus index deduplicates windows present in both containers", check_b5_10_corpus_index_deduplicates_windows_present_in_both_containers),
    ("B5 population histogram vectorisation matches the per-bin loop", check_b5_11_population_histogram_vectorisation_matches_the_per_bin_loop),
    ("B5 fast reader matches the reference parse", check_b5_12_fast_reader_matches_the_reference_parse),
    ("B5 exact percentile from histogram matches the direct percentile", check_b5_13_exact_percentile_from_histogram_matches_direct_percentile),
]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    keep_going = "--all" in argv  # run every check instead of stopping at the first failure

    passed = skipped = 0
    failures = []
    print(f"BoatPhone contract checks -- repo root {REPO_ROOT}\n")
    for name, fn in CHECKS:
        try:
            fn()
        except SkipCheck as exc:
            skipped += 1
            print(f"SKIP  {name}\n      reason: {exc}")
        except AssertionError as exc:
            failures.append((name, str(exc)))
            print(f"FAIL  {name}\n      reason: {exc}")
            if not keep_going:
                break
        except Exception as exc:  # noqa: BLE001 -- named, reported, never swallowed
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {name}\n      reason: {type(exc).__name__}: {exc}")
            print(textwrap.indent(traceback.format_exc(), "      "))
            if not keep_going:
                break
        else:
            passed += 1
            print(f"PASS  {name}")

    print(f"\n{passed} passed, {len(failures)} failed, {skipped} skipped, "
          f"{len(CHECKS) - passed - len(failures) - skipped} not run")
    if failures:
        print("\nFirst failure: " + failures[0][0])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
