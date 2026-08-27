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

import os
import pathlib
import subprocess
import sys
import tempfile
import textwrap
import traceback

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
