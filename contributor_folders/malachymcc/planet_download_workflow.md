# Folger Deep PlanetScope Workflow — Optimizing Under a 30-Scene Monthly Budget

Search / order / download workflow for pairing PlanetScope imagery with the Folger Deep
hydrophone record, rewritten around the E&R Basic quota model.

**Environment:** CryoCloud JupyterHub, ~4 GB RAM, Planet Python SDK v2.

---

## 1. The quota model, confirmed

Your premise checks out, and it is worth stating precisely because two details change the
optimization.

Planet's E&R Basic documentation confirms the allocation is 3,000 km² per month with
"preferred clipping," and that preferred clipping charges **a minimum of 100 km² per
intersecting scene even if your order is just 1 km²**. The Orders API FAQ gives the same
rule from the other direction: preferred-model customers are charged a minimum of 100 km²
per asset in an order, whereas premium customers are charged actual calculated area.

Two consequences follow.

**Clipping below 100 km² is pure loss.** Your original 24 km² AOI and the current 99.7 km²
core AOI cost *exactly the same*. Shrinking the AOI to stretch the budget — which I
suggested in the previous iteration — does not work under this model. That advice was wrong
and the workflow below reverses it. The core AOI at 99.7 km² sits essentially at the
threshold, which is where you want it.

**The budget is 30 scenes per month, full stop**, and no amount of geometry tuning changes
it. So every optimization has to come from *choosing better scenes*, not from buying more.

### The detail that makes this tractable

Planet's support documentation states plainly that **downloading UDM2 files via the Data API
does not count against download quota**. Quota is charged when an Orders API order is
successfully placed, or when a Data API *asset type* is downloaded — and UDM2 is carved out.

This is the linchpin of everything below. It means you can inspect the actual cloud, haze,
shadow and clear masks for **every candidate scene, at full 3 m resolution, windowed to your
exact AOI, for free** — and only then spend one of your thirty slots.

### Policy: single-raster bundles only

The Orders FAQ wording is "minimum of 100 km² **per asset** in a given order." If that is
literal, a bundle delivering two raster assets — surface reflectance *plus* a bundled UDM2 —
is charged 200 km² per scene, halving you to 15 scenes/month.

Rather than resolve the ambiguity, this workflow sidesteps it: **every order carries exactly
one raster asset.** That guarantees 100 km² per scene under either reading, so the question
never has to be answered.

There is no cost to this. The only thing a `*_udm2` bundle adds is the UDM2 raster, and you
already have that for free from the Data API in stage 3 — where it is explicitly quota-exempt.
Paying a possible second 100 km² for a file you can fetch for nothing would be the worst
trade available to you.

**Choose from these:**

| Bundle | Rasters delivered | Use when |
|---|---|---|
| `visual` | `ortho_visual` | Default. Wake and vessel spotting. |
| `analytic_sr` | `ortho_analytic_sr` | You need 4-band surface reflectance. |
| `analytic_8b_sr` | `ortho_analytic_8b_sr` | 8-band SR, SuperDove-era scenes only. |

**Never use** any bundle whose name ends in `_udm2`, and do not set `fallback_bundle` — a
fallback can silently substitute a multi-raster bundle for older scenes and double the charge
on exactly the acquisitions you are least able to re-order.

Confirm the bare (non-`_udm2`) names exist on your plan with `planet orders bundles` before
your first analytic order; Planet revises the catalogue periodically. If a bare variant is
unavailable, stay on `visual` rather than falling back to a `_udm2` bundle.

---

## 2. Recommendations

### 2.1 Screen clouds *inside the AOI*, not at scene level

This is the single highest-value change.

`cloud_cover` and `clear_percent` are computed over the **entire scene footprint** — roughly
637 km² for a SuperDove strip. Your AOI is ~100 km², about 15% of that. The scene-level
statistic tells you almost nothing about your 15%.

The failure runs both ways. A scene at 55% clear may be perfectly cloudless over Folger
Passage, with all the cloud piled over the Vancouver Island interior — and your
`cloud_cover <= 0.10` filter throws it away. Conversely, a 96%-clear scene can have its one
small cloud sitting directly on your AOI, and you spend a slot on an unusable image.

Because UDM2 is free via the Data API, you can compute the true AOI-windowed clear fraction
for every candidate before committing. In practice this both *recovers* scenes your filter
was discarding and *rejects* scenes it was passing.

Loosen the search filter to `cloud_cover <= 0.6` and let the free UDM2 screen do the real
work. Tightening the API-side filter is now actively counterproductive.

### 2.2 Add a sun-glint filter from metadata — free, and specific to water

Over land, UDM2 does a reasonable job. Over water it is out of its training distribution and
routinely misreads glint and whitecaps. But glint is deterministic geometry, so you can
predict it from metadata Planet already gives you at no cost.

The glint angle is the angle between the sensor's view direction and the specular reflection
direction of the sun:

```
cos Θg = cos θv · cos θs − sin θv · sin θs · cos(φv − φs)
```

where θs = 90° − `sun_elevation`, θv = `view_angle`, φs = `sun_azimuth`, φv =
`satellite_azimuth`. Small Θg means the sensor is looking straight into the sun's reflection.

For near-nadir PlanetScope at 48.8°N in mid-morning summer, Θg typically lands around 35–45°,
so severe glint is uncommon — but off-nadir acquisitions pointed toward the solar azimuth are
the exception, and those are exactly the scenes where wake contrast collapses. Rank on it
rather than hard-filtering, and calibrate the threshold against your first batch.

### 2.3 Use the hydrophone as your object detector

You asked how to identify scenes with **objects present**. Optically, you cannot — not before
ordering.

Scene thumbnails are a few hundred pixels across a 637 km² footprint, on the order of 100 m
per pixel. A 7 m boat is invisible; so is its wake. There is no free optical preview at a
resolution where vessels are detectable.

But you have something better than a preview: a continuous acoustic record that already knows
when vessels were present. Invert the workflow —

1. Run your vessel-noise detector across the full hydrophone archive first.
2. Extract detections falling within ±15 minutes of each candidate overpass timestamp.
3. Order **only** scenes with a coincident acoustic detection.

Every slot then buys a scene guaranteed to contain something. Thirty scenes chosen this way
are worth several times thirty chosen by cloud statistics alone, because a cloud-free image
of empty water contributes almost nothing to a detector training set.

This also fixes a sampling problem. Scenes selected purely on clarity are biased toward calm,
clear, low-traffic conditions. Scenes selected on acoustic detections span the conditions you
actually need to characterize.

Keep a small control set — perhaps 5 of every 30 — from acoustically quiet periods, so you
can measure your false-positive rate on confirmed-empty water.

### 2.4 Pre-filter on wind, which helps you twice

Whitecaps destroy wake contrast, and wind noise dominates the ambient soundscape at the
frequencies you care about. Both improve together.

Pull hourly wind from ERA5 or the ECCC station record and drop overpass times above roughly
8–10 m/s before you even look at imagery. This costs nothing and removes scenes that would
have failed QC anyway. It does bias your sample toward calm conditions — state that
explicitly in your methods, and note that vessel-noise detection is easiest in exactly the
conditions where ambient is lowest.

### 2.5 Stratify the thirty

Ranking purely on quality will concentrate your scenes in whichever summer had the best
weather. Allocate slots proportionally across years first, then rank within each year. You
want inter-annual comparability more than you want the thirty prettiest images.

### 2.6 Keep the AOI at ~100 km², and think hard before the corridor

The 100 km² minimum is a floor, not a cap — above it you are charged actual area. The
191.4 km² corridor AOI therefore costs ~1.9 slots per scene, cutting you to about 15
scenes/month. That is a real trade: roughly twice the water coverage for half the temporal
sampling. For building a detector I would stay on the core AOI; for characterizing traffic
patterns along the Trevor Channel approach, the corridor may be worth it.

### 2.7 A possible shortcut, unverified

Planet exposes an XYZ tile service for individual scenes, which at zoom 15–16 approaches
native resolution and would in principle let you *see* vessels before ordering. I could not
confirm from public documentation whether tile access is quota-exempt the way UDM2 is.

Worth asking your E&R contact about. If tiles are free, it changes the workflow materially —
you could visually confirm objects before spending a slot. Do not assume it; verify first.

---

## 3. Revised budget

| | |
|---|---|
| Monthly allocation | 3,000 km² |
| Charge per clipped scene (core AOI) | 100 km² |
| **Scenes per month** | **30** |
| Scenes per year | 360 |
| Charge per clipped scene (corridor AOI) | 191 km² → 15/month |

Quota is per calendar month and does not roll over. Unspent quota is lost, so run a fixed
monthly cadence rather than saving up.

---

## 4. Workflow

Stages 1–5 are free and repeatable. Only stage 6 spends quota, and it sits behind a manual
gate.

### Stage 0 — Setup

```python
%pip install --quiet "planet>=2.1,<3" pandas rasterio pillow

import asyncio, json, os, math, shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask, bounds as geom_bounds
from rasterio.warp import transform_geom
from planet import Auth, Session, data_filter, order_request, reporting

# ----------------------------------------------------------------- configuration
AOI_PATH  = Path("folger_core_aoi.geojson")
WORK      = Path("planet_folger"); WORK.mkdir(exist_ok=True)
UDM_DIR   = WORK / "udm2_scratch"; UDM_DIR.mkdir(exist_ok=True)

YEARS     = range(2020, 2026)
SUMMER    = (6, 9)                  # June 1 -> Sept 1 exclusive
ITEM_TYPE = "PSScene"

# Deliberately LOOSE. The free UDM2 screen in stage 3 does the real filtering,
# and a tight scene-level filter discards scenes that are clear over the AOI.
SEARCH_CLOUD_MAX = 0.60

# Quota model: preferred clipping, 100 km^2 minimum per intersecting scene.
AOI_AREA_KM2  = 99.7
MIN_CHARGE    = 100.0
MONTHLY_QUOTA = 3000.0
CHARGE_PER_SCENE = max(AOI_AREA_KM2, MIN_CHARGE)
SCENES_PER_MONTH = int(MONTHLY_QUOTA // CHARGE_PER_SCENE)

# Single-raster bundles ONLY. A "_udm2" bundle ships a second raster asset, which
# may be charged a second 100 km^2 minimum - for a file stage 3 fetches free.
SINGLE_RASTER_BUNDLES = {"visual", "analytic_sr", "analytic_8b_sr", "analytic", "analytic_8b"}

BUNDLE = "visual"

# Hard guard: fail loudly at configuration time rather than discovering a doubled
# charge in the usage report a month later.
assert BUNDLE in SINGLE_RASTER_BUNDLES, (
    f"{BUNDLE!r} is not a known single-raster bundle. Multi-raster bundles "
    f"(anything ending in '_udm2') risk a doubled quota charge. "
    f"Allowed: {sorted(SINGLE_RASTER_BUNDLES)}"
)
assert not BUNDLE.endswith("_udm2"), "Multi-raster bundle rejected."

auth = Auth.from_file()             # written by `planet auth init` in a terminal

aoi = json.load(open(AOI_PATH))
if aoi["type"] == "FeatureCollection": aoi = aoi["features"][0]["geometry"]
elif aoi["type"] == "Feature":        aoi = aoi["geometry"]

print(f"Budget: {SCENES_PER_MONTH} scenes/month at {CHARGE_PER_SCENE:.0f} km^2 each")
```

### Stage 1 — Search (free)

```python
search_filter = data_filter.and_filter([
    data_filter.geometry_filter(aoi),
    data_filter.range_filter("cloud_cover", lte=SEARCH_CLOUD_MAX),
    data_filter.or_filter([
        data_filter.date_range_filter("acquired",
                                      gte=datetime(y, SUMMER[0], 1),
                                      lt =datetime(y, SUMMER[1], 1))
        for y in YEARS
    ]),
    data_filter.permission_filter(),
    data_filter.string_in_filter("quality_category", ["standard"]),
])

async def run_search():
    async with Session(auth=auth) as sess:
        results = sess.client("data").search(
            [ITEM_TYPE], search_filter=search_filter, limit=0)
        return [i async for i in results]

items = await run_search()
json.dump(items, open(WORK / "search_results.json", "w"))
print(f"{len(items)} candidate scenes cached")
```

### Stage 2 — Metadata triage and glint modelling (free)

```python
df = pd.DataFrame([{
    "id":            it["id"],
    "acquired":      pd.to_datetime(it["properties"]["acquired"]),
    "cloud_cover":   it["properties"]["cloud_cover"],
    "clear_percent": it["properties"].get("clear_percent"),
    "instrument":    it["properties"].get("instrument"),
    "sun_elevation": it["properties"].get("sun_elevation"),
    "sun_azimuth":   it["properties"].get("sun_azimuth"),
    "view_angle":    it["properties"].get("view_angle"),
    "sat_azimuth":   it["properties"].get("satellite_azimuth"),
    "thumbnail":     it["_links"].get("thumbnail"),
} for it in items])

df["date"] = df["acquired"].dt.date
df["year"] = df["acquired"].dt.year

# Glint angle: angle between view direction and the sun's specular direction.
# Small values mean the sensor is looking into the sun glitter pattern.
ts = np.radians(90.0 - df["sun_elevation"])          # solar zenith
tv = np.radians(df["view_angle"].abs())              # sensor zenith
dphi = np.radians(df["sat_azimuth"] - df["sun_azimuth"])
df["glint_angle"] = np.degrees(np.arccos(
    np.clip(np.cos(tv) * np.cos(ts) - np.sin(tv) * np.sin(ts) * np.cos(dphi), -1, 1)))

print(df["glint_angle"].describe().round(1))

# Drop only the clearly glint-compromised; this is a ranking term, not a hard cut.
df = df[df["glint_angle"] > 20]
print(f"{len(df)} scenes after glint screen")
```

### Stage 3 — Free AOI-windowed UDM2 screening

This is where the budget is actually won. UDM2 downloads via the Data API are quota-exempt,
so every candidate can be inspected at full resolution before you commit.

```python
UDM2_CLEAR_BAND = 1     # band 1 = clear mask; 2 snow, 3 shadow, 4 light haze,
                        # 5 heavy haze, 6 cloud, 7 confidence, 8 unusable-data mask

def aoi_clear_fraction(udm_path, aoi_geojson):
    """Fraction of AOI pixels flagged clear. Reads only the AOI window and a
    single band, so peak memory stays around 10 MB regardless of scene size."""
    with rasterio.open(udm_path) as src:
        geom = transform_geom("EPSG:4326", src.crs, aoi_geojson)
        win  = from_bounds(*geom_bounds(geom), src.transform)
        win  = win.round_offsets().round_lengths()
        clear = src.read(UDM2_CLEAR_BAND, window=win)
        if clear.size == 0:
            return np.nan
        inside = geometry_mask([geom], out_shape=clear.shape,
                               transform=src.window_transform(win), invert=True)
        return float((clear[inside] == 1).mean()) if inside.any() else np.nan


async def screen_udm2(item_ids, keep_files=False):
    """Download + score UDM2 for each candidate. Costs no quota. Files are
    deleted after scoring unless keep_files, to protect CryoCloud disk."""
    out = {}
    async with Session(auth=auth) as sess:
        cl = sess.client("data")
        for n, iid in enumerate(item_ids, 1):
            try:
                asset = await cl.get_asset(ITEM_TYPE, iid, "ortho_udm2")
                await cl.activate_asset(asset)
                asset = await cl.wait_asset(asset, max_attempts=200)
                path  = await cl.download_asset(asset, directory=UDM_DIR,
                                                overwrite=False, progress_bar=False)
                out[iid] = aoi_clear_fraction(path, aoi)
                if not keep_files:
                    Path(path).unlink(missing_ok=True)
            except Exception as e:
                print(f"  {iid}: {type(e).__name__} {e}")
                out[iid] = np.nan
            if n % 25 == 0:
                print(f"  screened {n}/{len(item_ids)}")
    return out

scores = await screen_udm2(df["id"].tolist())
df["aoi_clear"] = df["id"].map(scores)

print(f"\nAOI-clear vs scene-level clear (correlation: "
      f"{df[['aoi_clear','clear_percent']].corr().iloc[0,1]:.2f})")
print(f"Scenes >=95% clear over the AOI: {(df['aoi_clear'] >= 0.95).sum()}")
print(f"...of which the old cloud_cover<=0.10 filter would have DISCARDED: "
      f"{((df['aoi_clear'] >= 0.95) & (df['cloud_cover'] > 0.10)).sum()}")

df = df[df["aoi_clear"] >= 0.95]
```

That last count is the payoff — scenes that are genuinely clear over Folger Passage but which
a scene-level cloud filter would have thrown away.

### Stage 4 — Quicklook contact sheet (free)

Thumbnails are too coarse for vessels but perfectly adequate for catching fog banks and
glint sheets that UDM2 misreads over water.

```python
import httpx
from PIL import Image
from io import BytesIO

key = auth.value    # API key from the stored credential

def contact_sheet(rows, cols=6, thumb=200):
    imgs = []
    with httpx.Client(auth=(key, ""), timeout=30, follow_redirects=True) as c:
        for _, r in rows.iterrows():
            try:
                im = Image.open(BytesIO(c.get(r["thumbnail"]).content)).convert("RGB")
                imgs.append((im.resize((thumb, thumb)), str(r["date"])))
            except Exception:
                pass
    if not imgs: return None
    n_rows = -(-len(imgs) // cols)
    sheet = Image.new("RGB", (cols * thumb, n_rows * thumb), "black")
    for i, (im, _) in enumerate(imgs):
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    return sheet

contact_sheet(df.head(36))   # inspect, then drop obvious fog/glint by id
```

### Stage 5 — Tile preview: confirm vessels before spending

Scene tiles are metered **separately** from the 3,000 km² imagery quota, so this stage is
free against the 30-scene budget. At z15 the ground resolution is 3.15 m/px, matching
PlanetScope native — z16 costs 3.4× more tiles for pure interpolation.

| Object | Pixels at z15 | | Coverage | Tiles/scene | Previews at ~98k |
|---|---|---|---|---|---|
| 7 m skiff | 2.2 | | Inner 5 × 5 km | 56 | ~1,750 |
| 12 m charter | 3.8 | | Full AOI | 196 | ~500 |
| 30 m freighter | 9.5 | | | | |
| Wake width | 19 | | | | |
| Wake length | 127 | | | | |

This confirms wakes and mid-size hulls, not small boats sitting still — the same bias the
full imagery carries, arriving early enough to act on. Preview the inner box first and pull
the full AOI only for survivors.

Take the tile URL from the item's `_links`, never hand-build it. Cap concurrency with
retry-on-429. Composite each scene to a single JPEG and discard the tiles — caching all 98k
would be ~4.9 GB. Before trusting a negative, test on a scene with a known close passage:
tiles are rendered with a land-tuned stretch that can crush wake contrast over dark water.

Full implementation is in `planet_folger_search_order_download.ipynb`, stage 6.

### Stage 6 — Acoustic join (the second object filter)

```python
# Your detector output: one row per vessel detection, UTC.
det = pd.read_csv("hydrophone_detections.csv", parse_dates=["start_utc", "end_utc"])

WINDOW = timedelta(minutes=15)

def acoustic_hits(t):
    m = (det["start_utc"] <= t + WINDOW) & (det["end_utc"] >= t - WINDOW)
    return int(m.sum())

df["n_detections"] = df["acquired"].apply(acoustic_hits)

print(f"{(df['n_detections'] > 0).sum()} of {len(df)} clear scenes have a "
      f"coincident acoustic detection")

targets  = df[df["n_detections"] > 0].copy()     # scenes with objects
controls = df[df["n_detections"] == 0].copy()    # confirmed-quiet controls
```

### Stage 7 — Rank, stratify, and spend

```python
def score(d):
    return (d["aoi_clear"] * 100
            + d["glint_angle"].clip(upper=60) * 0.5
            - d["view_angle"].abs() * 1.0
            + d["n_detections"].clip(upper=5) * 3.0)

targets["score"]  = score(targets)
controls["score"] = score(controls)

N_CONTROL = 5                                     # of every 30
N_TARGET  = SCENES_PER_MONTH - N_CONTROL

def stratify(d, n):
    """Allocate n slots proportionally across years, then rank within year."""
    if d.empty or n <= 0: return d.head(0)
    share = (d["year"].value_counts(normalize=True) * n).round().astype(int)
    return pd.concat([d[d["year"] == y].nlargest(k, "score")
                      for y, k in share.items() if k > 0])

# One dedup pass first: multiple satellites cross the same target on one day.
targets = targets.sort_values("score", ascending=False).groupby("date").head(1)

batch = pd.concat([stratify(targets, N_TARGET),
                   stratify(controls, N_CONTROL)]).sort_values("acquired")

print(f"Batch: {len(batch)} scenes = {len(batch) * CHARGE_PER_SCENE:.0f} km^2 "
      f"of {MONTHLY_QUOTA:.0f}")
print(batch.groupby("year").size())

# ------------------------------------------------------------------ SPEND GATE
CONFIRM_ORDER = False        # set True to actually place the order
```

```python
async def place_and_download(ids, name):
    out = WORK / "downloads" / name; out.mkdir(parents=True, exist_ok=True)
    async with Session(auth=auth) as sess:
        cl = sess.client("orders")
        req = order_request.build_request(
            name=name,
            products=[order_request.product(item_ids=ids,
                                            product_bundle=BUNDLE,
                                            item_type=ITEM_TYPE)],
            tools=[order_request.clip_tool(aoi=aoi)],
        )
        with reporting.StateBar(state="creating") as bar:
            order = await cl.create_order(req)
            bar.update(state="created", order_id=order["id"])
            await cl.wait(order["id"], callback=bar.update_state, max_attempts=0)
        await cl.download_order(order["id"], directory=out,
                                overwrite=False, progress_bar=True)
    return order["id"], out

if not CONFIRM_ORDER:
    print("CONFIRM_ORDER is False — nothing ordered.")
else:
    ids  = batch["id"].tolist()
    name = f"folger_{datetime.now():%Y%m}"
    oid, out = await place_and_download(ids, name)
    batch.to_csv(WORK / f"manifest_{name}.csv", index=False)
    print(f"Order {oid} -> {out}")

    # Single-raster bundles do not ship UDM2, so re-fetch and KEEP the masks for
    # the ordered scenes. Free via the Data API, and you want them alongside the
    # imagery for per-pixel QC. ~10-20 MB each; ~0.5 GB for thirty scenes.
    keep_dir = out / "udm2"; keep_dir.mkdir(exist_ok=True)
    globals()["UDM_DIR"] = keep_dir
    await screen_udm2(ids, keep_files=True)
    print(f"UDM2 masks retained in {keep_dir}")
```

The UDM2 rasters arrive as full scenes rather than clipped, so window them to the AOI at read
time with the same `aoi_clear_fraction` machinery. That is the one asymmetry introduced by
dropping `_udm2` bundles, and it costs you a windowed read rather than 100 km² of quota.

### Stage 8 — Check both budgets

```python
import httpx
r = httpx.get("https://api.planet.com/auth/v1/experimental/public/my/subscriptions",
              auth=(key, ""), timeout=30)
for s in r.json():
    if s.get("state") == "active":
        print(f"{s.get('plan',{}).get('name')}: "
              f"{s.get('quota_used')} / {s.get('quota_sqkm')} km^2 used")
```

---

## 5. What changed from the previous version

| Previous | Now | Why |
|---|---|---|
| `cloud_cover <= 0.10` at search | `<= 0.60`, then free UDM2 screen on the AOI | Scene-level stats describe 637 km², not your 100 km² |
| Shrink AOI to stretch budget | Keep AOI at ~100 km² | Below 100 km² costs the same; shrinking gains nothing |
| Rank on `clear_percent` | Rank on AOI-clear + glint + view angle + detections | Scene-level clarity is a weak proxy over water |
| Order the clearest scenes | Order scenes with coincident acoustic detections | Clear-but-empty water teaches a detector nothing |
| Batches sized to quota | Stratified across years, 5 controls per 30 | Avoids concentrating the sample in one good summer |
| Acoustics as the only object filter | Tile preview at z15, then acoustics | Tiles are a separate quota; vessels can be confirmed before spending |
| `*_udm2` bundles, `fallback_bundle` set | Single-raster bundles only, enforced by assertion | A second raster asset risks a second 100 km² charge for a file the Data API gives free |

---

## 6. Open items

1. ~~Confirm whether the 100 km² minimum applies per scene or per asset.~~ **Moot.** Orders
   now carry exactly one raster asset, so the charge is 100 km² per scene under either
   reading. Still worth asking your E&R contact out of interest, but nothing depends on it.
2. ~~Confirm whether the scene tile service is quota-exempt.~~ **Resolved.** Tiles are
   metered separately (~98k/month), so vessel presence can now be visually confirmed before
   spending an imagery slot. This is stage 5b, and it supersedes the earlier claim in this
   document that no free optical preview existed at vessel-detectable resolution.
3. **Calibrate the glint threshold** on your first batch rather than trusting the 20° cut.
4. **Verify bundle names** with `planet orders bundles` — Planet revises these periodically.
