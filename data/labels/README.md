# Optical vessel labels

Human-confirmed vessel presence or absence for PlanetScope scenes. **This is the only
place a label lives.** The acoustic detector must never write here, and nothing here is
derived from acoustics -- that independence is the whole reason a label is worth anything.

Tracked in git, unlike everything else under `data/`: these are small, hand-authored, and
the most expensive artefact in the project to reproduce. `.gitignore` excludes
`data/raw|interim|derived|processed`, not this.

## Columns

| column | meaning |
|---|---|
| `scene_id` | PlanetScope scene id; joins to `gate2_survivors.csv` and to `overpass_id` in the matchup schema |
| `acquired_utc` | acquisition instant, tz-aware UTC, copied from the scene list |
| `label` | `no_vessels` / `vessels` / `uncertain` |
| `n_vessels` | count if known, blank if not |
| `area_km2` | **the area the reviewer actually examined** |
| `reviewer` | who looked |
| `reviewed_utc` | when |
| `notes` | free text |

## `area_km2` is an AREA, and the distinction is load-bearing

A review over 10 km² is, if circular, a radius of about **1.78 km**. The hydrophone's
detection range is not bounded by that and is very likely larger -- estimating it is goal
G3, and it is not yet measured.

**So `no_vessels` means "no vessel inside the reviewed area", NEVER "acoustically
silent".** A vessel 3 km away is outside a 10 km² review and may be plainly audible.
Treating these rows as acoustic negatives without carrying the area would manufacture
false positives out of correctly-detected boats that simply sat outside the box.

This is the same far-field contamination `acoustics_plan_v2.md` §5 B5 flags as a
diagnostic: windows with no in-area vessel but elevated broadband level are direct
evidence of out-of-area traffic, and they are a finding rather than an error.

## Provenance

Labels come from human review of imagery. Record who and when, so a disagreement between
two reviewers is visible instead of silently overwritten. Append rather than edit.
