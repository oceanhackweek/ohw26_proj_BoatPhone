#!/usr/bin/env python3
"""BoatPhone contract checks -- milestone1 segment A0 (repo skeleton, credentials, env audit).

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

import numpy as np  # present on the hub (CLAUDE.md "Environment"); used only by A8b

# ---------------------------------------------------------------------------
# Constants (no magic numbers/strings -- source in comment)
# ---------------------------------------------------------------------------

# This file lives at <repo>/scripts/checks.py, so the repo root is two parents up.
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Running `python3 scripts/checks.py` puts scripts/ on sys.path, not the repo
# root, so an in-process `import boatphone` would fail with ModuleNotFoundError
# regardless of whether the module under test is correct. The A0 checks dodge
# this by importing in a child process with cwd=REPO_ROOT; the A8b checks import
# directly (they pass numpy arrays across the call, which does not survive a
# subprocess boundary), so the root goes on the path here. Appended, not
# prepended: an installed `boatphone` still wins, which is what a user of the
# package would get.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

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
# A8c -- minimal acquisition plan + loader (boatphone/vtuad.py)
#
# Contract (segment description, condensed):
#   1. A subset selector producing a MANIFEST: one row per intended file, with
#      url / byte size / scenario / split / class / a one-line justification.
#      Pinned: every label class from docs/vtuad-facts.md's label_schema
#      appears; the manifest's summed byte total equals the sum of its rows;
#      train/test groups are disjoint (mirroring A9's overpass_id hygiene, so
#      an in-domain number stays honest).
#   2. A byte-total estimator with a RUNTIME budget assertion: under a named
#      VTUAD_DOWNLOAD_BUDGET_BYTES, AND under a named fraction of
#      CURRENTLY-FREE space, re-measured at call time (never a hardcoded
#      snapshot -- the ~1.3 TB free figure is shared with teammates and with
#      A4's ongoing hydrophone pull).
#   3. A loader that fails clearly when data is absent: raises, naming both the
#      expected path and the command that would fetch it.
#   4. The loader reads sample rate from each file and REJECTS (never
#      resamples) a file whose rate differs from VTUAD_SAMPLE_RATE_HZ
#      (decision 0002 SS2).
#
# boatphone/vtuad.py does not exist yet -- every check below is EXPECTED TO
# FAIL for that reason (see the A0 module docstring for why that's the point).
#
# CORRECTION carried over from the original plan: a HEAD-only Content-Length
# pre-download cross-check against live URLs is NOT possible (A8a: the files
# sit behind a paid IEEE DataPort Subscriber login, served as presigned S3
# URLs -- no anonymous HEAD). No check below touches the network or downloads
# anything, including a "small test download". Instead: the manifest must
# record WHERE each size figure came from (source + retrieved date, same shape
# as the A8a facts table), and a post-download verification entry point must
# exist for a human to run after the files are actually on disk -- exercised
# here only up to "the function exists with a plausible signature"; anything
# that needs the actual ZIPs SKIPs cleanly, in the same idiom as A0.6.
#
# API surface below is a GUESS at what A8c will implement, made explicit
# rather than smuggled in (same policy as the A8b docstring above):
#   - build_manifest() -> list[dict], one dict per intended file, each with at
#     least the keys: "url", "size_bytes", "scenario", "class", "split",
#     "group_id", "justification", "source", "retrieved". "group_id" is the
#     unit that must not straddle train/test (mirrors A9's overpass_id).
#   - VTUAD_DOWNLOAD_BUDGET_BYTES: int module constant.
#   - VTUAD_FREE_SPACE_FRACTION_MAX: float module constant in (0, 1].
#   - free_space_bytes(path) -> int; a thin, LIVE wrapper (this project has no
#     other stdlib mechanism for free disk space than shutil.disk_usage, so
#     that call is what "live" is checked against below).
#   - assert_download_fits(total_bytes, target_dir) -> None; raises
#     VtuadBudgetError naming both total_bytes and the free-space number when
#     either the fixed budget or the live-free-space fraction is exceeded.
#   - VtuadBudgetError(Exception).
#   - VtuadDataMissingError(FileNotFoundError); load_clip(path) raises it,
#     naming the path and a fetch hint, when the path does not exist.
#   - load_clip(path) -> object exposing at least a `sample_rate_hz` (or
#     `.sample_rate_hz`) read from the file's own header.
#   - VtuadSampleRateError(Exception); raised by load_clip on a rate mismatch
#     against VTUAD_SAMPLE_RATE_HZ, naming both rates.
#   - verify_downloaded_files(manifest, data_dir) -> callable entry point for
#     post-download size/checksum verification; existence and signature only
#     are pinned here (no real ZIPs in this repo to verify against).
# If A8c's coder picks different names, these checks fail loudly naming the
# missing attribute -- the legible failure this segment wants, not a silent
# pass.
# ---------------------------------------------------------------------------

VTUAD_MODULE_PATH = REPO_ROOT / "boatphone" / "vtuad.py"

# Independent transcription of docs/vtuad-facts.md's label_schema row
# (retrieved 2026-08-27), same rationale as SAMPLE_DIR_NAME above: a checker
# that imports the very constant it is validating checks nothing.
VTUAD_CLASSES_INDEPENDENT = [
    "background", "cargo", "tanker", "tug", "passengership",
]

# Independent transcription of the three scenario archive names and their
# listed (10^9-byte) sizes, from docs/vtuad-facts.md total_size_bytes /
# smallest_downloadable_unit_bytes rows (2026-08-27). NOT imported from
# boatphone.config.VTUAD_ZONE_RADII_M -- this exists to give the manifest
# check something independently-sourced to compare against.
VTUAD_SCENARIOS_INDEPENDENT = [
    ("inclusion_2000_exclusion_4000", 3_310_000_000),
    ("inclusion_3000_exclusion_5000", 5_830_000_000),
    ("inclusion_4000_exclusion_6000", 4_400_000_000),
]

# The "stop and ask before downloading this much" threshold already invoked by
# boatphone/config.py's smallest_downloadable_unit_bytes comment (source:
# the approved A8 acquisition plan / working agreement). A
# VTUAD_DOWNLOAD_BUDGET_BYTES anywhere near this would defeat the point of a
# MINIMAL acquisition plan, so it is asserted comfortably below it, not just
# "less than".
VTUAD_STOP_AND_ASK_CEILING_BYTES = 150_000_000_000

# A tiny synthetic WAV's sample rate, deliberately DIFFERENT from
# VTUAD_SAMPLE_RATE_HZ (32000), used to prove rejection rather than silent
# resampling. 16000 Hz is itself a plausible real hydrophone rate -- chosen
# specifically so a loader that "clamps to the nearest known-good rate" cannot
# accidentally look correct.
WRONG_SAMPLE_RATE_HZ = 16000

# Sanity self-check that shutil.disk_usage is even in the standard library on
# this interpreter, so a SKIP has a distinguishable cause from a real failure.
import shutil as _shutil_probe  # noqa: E402 -- deliberately local; only used for the assert below
assert hasattr(_shutil_probe, "disk_usage"), (
    "shutil.disk_usage is unavailable in this interpreter; the live-free-space "
    "check below has no standard-library mechanism to test against"
)


def check_a8c_vtuad_module_exists():
    """boatphone/vtuad.py exists at all.

    Expected to FAIL right now: the module does not exist. Written first, same
    reasoning as check_a8b_models_module_exists -- the first failure should
    name the real cause (missing file), not an opaque ImportError.
    """
    assert VTUAD_MODULE_PATH.is_file(), (
        f"{VTUAD_MODULE_PATH} does not exist yet -- A8c (minimal acquisition "
        "plan + loader) has not been implemented. Every other 'A8c' check "
        "below is expected to fail for the same reason until this file is "
        "written."
    )


def check_a8c_manifest_covers_every_label_class():
    """build_manifest() names every VTUAD label class from docs/vtuad-facts.md."""
    import boatphone.vtuad as bv
    manifest = bv.build_manifest()
    assert manifest, "build_manifest() returned an empty manifest"
    present_classes = {row.get("class") for row in manifest}
    missing = sorted(c for c in VTUAD_CLASSES_INDEPENDENT if c not in present_classes)
    assert not missing, (
        f"build_manifest() omits label class(es) {missing} named in "
        f"docs/vtuad-facts.md's label_schema row; got classes {sorted(present_classes)}"
    )


def check_a8c_manifest_byte_total_equals_sum_of_rows():
    """The manifest's reported total equals the sum of its own per-row sizes.

    Guards against a manifest whose summary total drifts from what the rows
    actually add up to -- exactly the kind of silent inconsistency that would
    later make a budget check pass on the wrong number.
    """
    import boatphone.vtuad as bv
    manifest = bv.build_manifest()
    assert manifest, "build_manifest() returned an empty manifest"
    row_sizes = [row.get("size_bytes") for row in manifest]
    missing_sizes = [i for i, s in enumerate(row_sizes) if not isinstance(s, int) or s <= 0]
    assert not missing_sizes, (
        f"build_manifest() rows at index {missing_sizes} have a missing/non-positive "
        "size_bytes; every row must carry a real byte size"
    )
    summed = sum(row_sizes)
    reported = bv.estimate_download_bytes(manifest)
    assert reported == summed, (
        f"estimate_download_bytes(manifest) = {reported}, but summing the manifest's "
        f"own size_bytes rows gives {summed}; the estimator and the manifest must agree"
    )


def check_a8c_manifest_train_test_groups_are_disjoint():
    """No group_id (mirrors A9's overpass_id) appears in both the train and test splits.

    A group straddling both splits is exactly the leak that makes an in-domain
    number dishonest: whatever unit `group_id` denotes (vessel encounter, MMSI,
    scenario file -- build_manifest()'s choice) must not be partially in train
    and partially in test.
    """
    import boatphone.vtuad as bv
    manifest = bv.build_manifest()
    assert manifest, "build_manifest() returned an empty manifest"
    missing_fields = [
        i for i, row in enumerate(manifest)
        if "split" not in row or "group_id" not in row
    ]
    assert not missing_fields, (
        f"manifest rows at index {missing_fields} are missing 'split' and/or "
        "'group_id' -- both are required to check train/test hygiene"
    )
    splits = {row["split"] for row in manifest}
    assert {"train", "test"} <= splits, (
        f"manifest splits are {sorted(splits)}; both 'train' and 'test' must be present "
        "for a disjointness check to mean anything"
    )
    by_group: dict = {}
    for row in manifest:
        by_group.setdefault(row["group_id"], set()).add(row["split"])
    leaking = sorted(g for g, s in by_group.items() if len(s) > 1)
    assert not leaking, (
        f"group_id(s) {leaking} appear in BOTH train and test splits; "
        "train/test groups must be disjoint (mirrors A9's overpass_id hygiene rule)"
    )


def check_a8c_manifest_records_size_provenance():
    """Every row states WHERE its size figure came from, and when it was retrieved.

    Stand-in for the HEAD-request cross-check that isn't possible against a
    presigned, login-gated URL (A8a). The manifest must instead be honest about
    its size figures coming from the landing-page listing, same shape as the
    A8a facts table (source URL + retrieved date), not invented.
    """
    import boatphone.vtuad as bv
    manifest = bv.build_manifest()
    assert manifest, "build_manifest() returned an empty manifest"
    bad = []
    for i, row in enumerate(manifest):
        source = str(row.get("source", "")).strip()
        retrieved = str(row.get("retrieved", "")).strip()
        if not (source.startswith("http://") or source.startswith("https://")):
            bad.append(f"row {i}: source {source!r} is not an http(s) URL")
        if not retrieved:
            bad.append(f"row {i}: retrieved date is blank")
        if not str(row.get("justification", "")).strip():
            bad.append(f"row {i}: justification is blank")
    assert not bad, "manifest provenance problems:\n      " + "\n      ".join(bad)


def check_a8c_budget_constant_is_named_and_manifest_fits_under_it():
    """VTUAD_DOWNLOAD_BUDGET_BYTES exists, is well below the stop-and-ask ceiling,
    and the manifest's own total fits under it.
    """
    import boatphone.vtuad as bv
    assert hasattr(bv, "VTUAD_DOWNLOAD_BUDGET_BYTES"), (
        "boatphone.vtuad has no VTUAD_DOWNLOAD_BUDGET_BYTES constant"
    )
    budget = bv.VTUAD_DOWNLOAD_BUDGET_BYTES
    assert isinstance(budget, int) and budget > 0, (
        f"VTUAD_DOWNLOAD_BUDGET_BYTES = {budget!r}; must be a positive int"
    )
    assert budget < VTUAD_STOP_AND_ASK_CEILING_BYTES, (
        f"VTUAD_DOWNLOAD_BUDGET_BYTES = {budget} is not comfortably below the "
        f"{VTUAD_STOP_AND_ASK_CEILING_BYTES} stop-and-ask ceiling; a 'minimal' "
        "acquisition plan should not need a budget anywhere near that"
    )
    manifest = bv.build_manifest()
    total = bv.estimate_download_bytes(manifest)
    assert total <= budget, (
        f"the manifest's own total ({total} bytes) exceeds its own module's "
        f"VTUAD_DOWNLOAD_BUDGET_BYTES ({budget} bytes); a minimal plan must fit "
        "under its own budget"
    )


def check_a8c_free_space_fraction_is_read_live_not_hardcoded():
    """The free-space check re-measures shutil.disk_usage at call time.

    Proven two ways, both via monkeypatching shutil.disk_usage (the only
    stdlib mechanism for free space -- see the module-level assert above):
      1. a TINY mocked free-space value must make an otherwise-tiny,
         budget-compliant download REJECT, and the raised message must
         contain that mocked free-space number verbatim -- proving the number
         came from the live call, not a hardcoded ~1.3 TB snapshot;
      2. a HUGE mocked free-space value for the SAME total_bytes must then be
         accepted -- proving the function actually branches on what it reads,
         rather than always rejecting.
    """
    import boatphone.vtuad as bv
    assert hasattr(bv, "VTUAD_FREE_SPACE_FRACTION_MAX"), (
        "boatphone.vtuad has no VTUAD_FREE_SPACE_FRACTION_MAX constant"
    )
    frac = bv.VTUAD_FREE_SPACE_FRACTION_MAX
    assert isinstance(frac, (int, float)) and 0 < frac <= 1, (
        f"VTUAD_FREE_SPACE_FRACTION_MAX = {frac!r}; must be a fraction in (0, 1]"
    )

    tiny_free_bytes = 1_000_000  # 1 MB -- smaller than any real request
    huge_free_bytes = 10_000_000_000_000  # 10 TB -- larger than any real request
    total_bytes = 500_000  # well under both the budget and a "normal" free-space fraction

    class _FakeUsage:
        def __init__(self, free):
            self.total = free * 10
            self.used = 0
            self.free = free

    import shutil
    import unittest.mock as mock
    target_dir = REPO_ROOT  # any existing directory; disk_usage is being mocked anyway

    with mock.patch.object(shutil, "disk_usage", return_value=_FakeUsage(tiny_free_bytes)):
        try:
            bv.assert_download_fits(total_bytes, target_dir)
        except Exception as exc:
            msg = str(exc)
            assert str(tiny_free_bytes) in msg, (
                "assert_download_fits() rejected under a mocked tiny free-space "
                f"value ({tiny_free_bytes}) but the raised message does not contain "
                f"that number verbatim: {msg!r} -- this is the check that a hardcoded "
                "~1.3 TB snapshot would fail: the message must reflect what was "
                "actually read from the (mocked) filesystem call, not a constant"
            )
        else:
            raise AssertionError(
                f"assert_download_fits({total_bytes}, ...) did not raise under a "
                f"mocked free space of only {tiny_free_bytes} bytes; either free "
                "space is never checked, or it is read from a hardcoded constant "
                "instead of shutil.disk_usage at call time"
            )

    with mock.patch.object(shutil, "disk_usage", return_value=_FakeUsage(huge_free_bytes)):
        try:
            bv.assert_download_fits(total_bytes, target_dir)
        except Exception as exc:
            raise AssertionError(
                f"assert_download_fits({total_bytes}, ...) raised {type(exc).__name__}: "
                f"{exc} under a mocked free space of {huge_free_bytes} bytes; the same "
                "total_bytes was accepted under a tiny mocked free space and rejected "
                "under a huge one, so the function is not actually branching on the "
                "live measurement -- an always-reject implementation would look "
                "'safe' here without checking anything"
            ) from exc


def check_a8c_loader_raises_naming_path_and_fetch_command():
    """load_clip() on an absent path raises, naming the path AND how to fetch it.

    Asserted explicitly (CLAUDE.md invariant 5) so a later refactor to a bare
    `except` or a silently-empty return fails HERE, not downstream as a result
    that looks like "zero vessels" instead of "the file was never there".
    """
    import boatphone.vtuad as bv
    missing = REPO_ROOT / "data" / "definitely-absent-vtuad-clip-9e21.wav"
    assert not missing.exists(), f"fixture path unexpectedly exists: {missing}"
    try:
        result = bv.load_clip(missing)
    except FileNotFoundError as exc:
        msg = str(exc)
        assert str(missing) in msg, (
            f"load_clip() raised on a missing file but did not name the path {missing} "
            f"in its message: {msg!r}"
        )
        fetch_tokens = ("ieee-dataport", "fetch", ".zip", "boatphone.vtuad", "vtuad")
        assert any(tok in msg.lower() for tok in fetch_tokens), (
            "load_clip() raised on a missing file, naming the path, but the message "
            f"names no way to fetch it (looked for any of {fetch_tokens} case-"
            f"insensitively): {msg!r}"
        )
    except Exception as exc:
        raise AssertionError(
            f"load_clip() on a missing path raised {type(exc).__name__} rather than "
            f"FileNotFoundError (or a subclass): {exc}"
        ) from exc
    else:
        raise AssertionError(
            f"load_clip({missing}) returned {result!r} for a path that does not "
            "exist; absent data must raise, never return an empty/None result "
            "that looks like 'no vessels'"
        )


def _write_synthetic_wav(path, sample_rate_hz: int, n_samples: int = 1000):
    """A minimal, real WAV file at a given rate, written with the stdlib `wave` module.

    Deliberately NOT a VTUAD download -- a from-scratch synthetic fixture, same
    spirit as the A8b tone-injection checks. Exists so the sample-rate contract
    can be tested without touching the network or the real (gated) corpus.
    """
    import struct
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # 16-bit PCM; stdlib `wave` cannot write 24-bit directly
        w.setframerate(sample_rate_hz)
        w.writeframes(struct.pack("<%dh" % n_samples, *([0] * n_samples)))


def check_a8c_loader_rejects_sample_rate_mismatch():
    """load_clip() on a real file whose header rate != VTUAD_SAMPLE_RATE_HZ RAISES.

    Never silently resamples (decision 0002 SS2). The message must name both
    rates, so a human sees exactly what mismatched.
    """
    import boatphone.config as cfg
    import boatphone.vtuad as bv
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp, "wrong_rate.wav")
        _write_synthetic_wav(path, WRONG_SAMPLE_RATE_HZ)
        try:
            result = bv.load_clip(path)
        except Exception as exc:
            msg = str(exc)
            missing = [
                tok for tok in (str(WRONG_SAMPLE_RATE_HZ), str(cfg.VTUAD_SAMPLE_RATE_HZ))
                if tok not in msg
            ]
            assert not missing, (
                f"load_clip() raised on a {WRONG_SAMPLE_RATE_HZ} Hz file (expected "
                f"{cfg.VTUAD_SAMPLE_RATE_HZ} Hz) but the message does not name "
                f"{missing}: {msg!r}"
            )
        else:
            raise AssertionError(
                f"load_clip() returned {result!r} for a file recorded at "
                f"{WRONG_SAMPLE_RATE_HZ} Hz, which differs from "
                f"VTUAD_SAMPLE_RATE_HZ ({cfg.VTUAD_SAMPLE_RATE_HZ}); it must RAISE, "
                "never silently resample or reinterpret the rate"
            )


def check_a8c_loader_reads_rate_from_header_not_assumed():
    """A file at the CORRECT rate loads and reports that rate, read from its own header.

    Positive half of the sample-rate contract: without this, a load_clip() that
    unconditionally raises would pass the mismatch check above for the wrong
    reason (always-reject, not "reads and compares").
    """
    import boatphone.config as cfg
    import boatphone.vtuad as bv
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp, "right_rate.wav")
        _write_synthetic_wav(path, cfg.VTUAD_SAMPLE_RATE_HZ)
        try:
            result = bv.load_clip(path)
        except Exception as exc:
            raise AssertionError(
                f"load_clip() raised {type(exc).__name__}: {exc} on a file correctly "
                f"recorded at VTUAD_SAMPLE_RATE_HZ ({cfg.VTUAD_SAMPLE_RATE_HZ} Hz); "
                "a matching rate must be accepted"
            ) from exc
        rate = getattr(result, "sample_rate_hz", None)
        assert rate == cfg.VTUAD_SAMPLE_RATE_HZ, (
            f"load_clip() on a {cfg.VTUAD_SAMPLE_RATE_HZ} Hz file returned an object "
            f"whose sample_rate_hz is {rate!r}; the rate must be READ from the file's "
            "own header, not assumed from the constant it is being checked against"
        )


def check_a8c_verify_downloaded_files_entry_point_exists():
    """A post-download size/checksum verification entry point exists with a plausible signature.

    Structural only -- this is the "for the human to run after they have the
    files" step named in the corrected A8c contract. It cannot be exercised
    against real data in this repo (no ZIPs are ever committed here), so this
    check pins existence and signature, not behaviour.
    """
    import boatphone.vtuad as bv
    assert hasattr(bv, "verify_downloaded_files"), (
        "boatphone.vtuad has no verify_downloaded_files() entry point for a human "
        "to run after downloading the gated ZIPs"
    )
    params = set(inspect.signature(bv.verify_downloaded_files).parameters)
    required = {"manifest"}
    missing = sorted(required - params)
    assert not missing, (
        f"verify_downloaded_files()'s signature is missing named parameter(s) "
        f"{missing} (got {sorted(params)})"
    )


def check_a8c_verify_downloaded_files_skips_without_local_zips():
    """Data-dependent: SKIPs (not fails) because the real VTUAD ZIPs are never in this repo.

    Same idiom as A0.6's data-dependent checks: a check that needs the actual
    acquisition, which is legitimately absent, must say so visibly rather than
    silently pass or fail for the wrong reason.
    """
    import boatphone.vtuad as bv
    scenario_name, _ = VTUAD_SCENARIOS_INDEPENDENT[0]
    zip_path = REPO_ROOT / "data" / "vtuad" / f"{scenario_name}.zip"
    if not zip_path.is_file():
        raise SkipCheck(
            f"VTUAD acquisition absent: {zip_path} -- the corpus is gated behind a "
            "paid IEEE DataPort login and is never committed to this repo "
            "(docs/vtuad-facts.md download_mechanism); verify_downloaded_files() "
            "cannot be exercised against real bytes here"
        )
    manifest = bv.build_manifest()
    bv.verify_downloaded_files(manifest, zip_path.parent)


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
# Runner
# ---------------------------------------------------------------------------

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
    ("A8c boatphone/vtuad.py exists", check_a8c_vtuad_module_exists),
    ("A8c manifest covers every label class", check_a8c_manifest_covers_every_label_class),
    ("A8c manifest byte total equals sum of rows", check_a8c_manifest_byte_total_equals_sum_of_rows),
    ("A8c manifest train/test groups are disjoint", check_a8c_manifest_train_test_groups_are_disjoint),
    ("A8c manifest records size provenance (source + retrieved)", check_a8c_manifest_records_size_provenance),
    ("A8c download budget constant named, manifest fits under it", check_a8c_budget_constant_is_named_and_manifest_fits_under_it),
    ("A8c free-space fraction is read live, not hardcoded", check_a8c_free_space_fraction_is_read_live_not_hardcoded),
    ("A8c loader on absent file names path + fetch command", check_a8c_loader_raises_naming_path_and_fetch_command),
    ("A8c loader rejects a sample-rate mismatch (never resamples)", check_a8c_loader_rejects_sample_rate_mismatch),
    ("A8c loader reads rate from header, not assumed", check_a8c_loader_reads_rate_from_header_not_assumed),
    ("A8c verify_downloaded_files entry point exists", check_a8c_verify_downloaded_files_entry_point_exists),
    ("A8c verify_downloaded_files (data-dependent, skips without local ZIPs)", check_a8c_verify_downloaded_files_skips_without_local_zips),
]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    keep_going = "--all" in argv  # run every check instead of stopping at the first failure

    passed = skipped = 0
    failures = []
    print(f"BoatPhone A0 contract checks -- repo root {REPO_ROOT}\n")
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
