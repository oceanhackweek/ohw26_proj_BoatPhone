# `data/samples/` -- the tracked fixture zone

This is the **one** directory inside `data/` whose contents are committed to git
(`docs/decisions/0005-raw-acquisition-landing-zone.md`). Everything else under `data/`
is either an original acquisition or a gitignored bulk download; `.gitignore` negates
this directory out of the blanket `*.flac` / `*.wav` / `*.mp3` / `*.fft.gz` / `*.zip`
rules so a deliberate fixture is actually committable.

It exists so checks and notebooks can be exercised against a real, byte-exact slice of
hydrophone data without anyone first running a multi-gigabyte ONC pull.

## Rules

- **Small files only.** One `.fft.gz` window is about **0.29 MB**, so a handful of
  fixtures is a few MB. That is the intended scale.
- **The repo already carries 127.3 MB of acquisitions in its history** and that history
  cannot be rewritten on a pushed shared repo
  (`docs/decisions/0006-acquisitions-in-git-history.md`). Every megabyte added here is
  permanent for everyone who clones. Keep it lean.
- Fixtures are still **immutable** once committed: replace by adding a new file with a
  new name, never by editing one in place (`docs/decisions/0001-raw-data-immutability.md`).
- State the provenance of every fixture you add -- instrument, UTC start and end, and the
  ONC request it came from -- in this README, next to the filename.

## Contents

**Empty for now, deliberately.** Choosing the fixture window is segment **A2**, which
ranks the A4 corpus by broadband level so the committed window contains an actual vessel
transit rather than an arbitrary five minutes of ambient noise. A fixture picked before
that would have to be replaced, and by the rule above replaced fixtures never leave the
history.
