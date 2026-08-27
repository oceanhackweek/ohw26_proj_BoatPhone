"""ONC credential handling for BoatPhone.

Resolution order, and it matters: **the process environment WINS over `.env`.**
An `ONC_TOKEN` exported in the shell or set by a scheduler overrides whatever a
checked-out `.env` happens to say, so a stale file can never quietly shadow the
credential you deliberately supplied.

`python-dotenv` is NOT installed on the OHW hub and is deliberately not added
(CLAUDE.md "Environment": an analysis that silently depends on a local install is
not reproducible). The parser below is ~25 lines of standard library.

Security: **no token value ever appears in an exception message, log line, or repr.**
Errors name the variable, the file, and the line number -- never the value.

No network calls live in this module; get_onc_client() only constructs the client.
"""

from __future__ import annotations

import os
import pathlib

from boatphone.paths import ONC_RAW_DIR, REPO_ROOT

# Environment variable carrying the ONC Oceans 3.0 API token.
# Source: ONC `onc` Python client documentation.
ONC_TOKEN_VAR = "ONC_TOKEN"

# Where an ONC token is obtained. Source: ONC Oceans 3.0 registration page.
ONC_REGISTRATION_URL = "https://data.oceannetworks.ca"

DOTENV_FILENAME = ".env"

# Placeholder shipped in .env.example. A token still equal to this means the user
# copied the template and never filled it in -- treated as absent, not as a token.
PLACEHOLDER_TOKEN = "your-onc-token-here"


class MissingCredentialError(RuntimeError):
    """Raised when a required credential is absent, empty, or still the placeholder."""


class DotEnvParseError(RuntimeError):
    """Raised when a `.env` line is not a KEY=VALUE pair. Never names the value."""


def dotenv_search_paths() -> list[pathlib.Path]:
    """Candidate `.env` locations, highest precedence first.

    The current working directory comes first so a notebook run from a subdirectory,
    or a tool run from elsewhere, uses the `.env` next to it. The repository's own
    `.env` is consulted only when the process is actually running inside the
    repository checkout -- outside it, the repo's credential is none of our business,
    which is also what keeps test processes in a temp directory from picking it up.
    """
    cwd = pathlib.Path.cwd().resolve()
    candidates = [cwd / DOTENV_FILENAME]
    if cwd == REPO_ROOT or REPO_ROOT in cwd.parents:
        candidates.append(REPO_ROOT / DOTENV_FILENAME)
    seen: set[pathlib.Path] = set()
    ordered: list[pathlib.Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def parse_dotenv(path: pathlib.Path) -> dict[str, str]:
    """Parse a `.env` file into a dict. Malformed lines RAISE, naming the line number.

    Accepts `KEY=VALUE`, `#` comments, blank lines, an optional `export ` prefix, and
    surrounding single or double quotes around the value (stripped). Anything else is
    an error: silently skipping a line you cannot parse is how a typo'd key becomes an
    unauthenticated request that returns an empty download.
    """
    values: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            raise DotEnvParseError(
                f"{path}: line {lineno} is not a KEY=VALUE pair; fix or comment it out "
                "(line content withheld -- it may contain a secret)"
            )
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def _from_dotenv(name: str) -> str | None:
    """First USABLE value of `name` across the candidate `.env` files, else None.

    "Contains the key" is not good enough. A nearer `.env` holding `ONC_TOKEN=` or
    the unfilled placeholder would otherwise SHADOW a perfectly good repo-root
    `.env`, and the user gets "token is absent" while looking at a file that has
    one. An unusable value falls through to the next candidate instead.
    """
    for path in dotenv_search_paths():
        if path.is_file():
            found = parse_dotenv(path).get(name)
            if _is_usable(found):
                return found
    return None


def _is_usable(value: str | None) -> bool:
    return bool(value) and value.strip() != "" and value != PLACEHOLDER_TOKEN


def get_onc_token() -> str:
    """Return the ONC API token as a non-empty str, or raise MissingCredentialError.

    Never returns None or "": an empty token yields an unauthenticated request whose
    empty result is indistinguishable from "no data", which is the failure mode this
    function exists to prevent. The token value is never included in any error text.
    """
    from_env = os.environ.get(ONC_TOKEN_VAR)
    if _is_usable(from_env):
        return from_env  # type: ignore[return-value]

    from_file = _from_dotenv(ONC_TOKEN_VAR)
    if _is_usable(from_file):
        return from_file  # type: ignore[return-value]

    searched = ", ".join(str(p) for p in dotenv_search_paths())
    raise MissingCredentialError(
        f"{ONC_TOKEN_VAR} is absent, empty, or still the .env.example placeholder. "
        f"Set it in the environment, or copy .env.example to .env and fill it in "
        f"(searched: {searched}). Register for a token at {ONC_REGISTRATION_URL}."
    )


def get_onc_client(out_path: pathlib.Path | None = None):
    """Construct an `onc.ONC` client with the token and an explicit output directory.

    `outPath` is set explicitly: the library default is a relative 'output' directory,
    which would scatter downloads into whatever directory the notebook happened to be
    launched from. Makes no network call.

    Creates NOTHING on disk. Merely constructing a client to check a token must not
    materialise `data/raw/onc/`; the acquisition module calls
    `boatphone.paths.ensure_dir(ONC_RAW_DIR)` at download time instead, so exactly one
    place decides when the `data/` tree grows a subdirectory
    (docs/decisions/0005-raw-acquisition-landing-zone.md).
    """
    from onc import ONC  # imported lazily: boatphone must import without third-party deps

    destination = pathlib.Path(out_path) if out_path is not None else ONC_RAW_DIR
    return ONC(get_onc_token(), outPath=str(destination))
