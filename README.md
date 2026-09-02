# Road extraction from satellite imagery

Two steps: download an image, run a model. Optional third step scores it against OSM.

## Install

```bash
uv sync
```

No activation needed - `uv run` handles the environment.

## Step 1 - download

```bash
uv run download.py 23.846073, 90.389624 airport_road
```

Paste coordinates straight from OpenStreetMap or Google; the comma is fine.
Leave the name out and one is made from the coordinates.

Zoom 18, 2048x2048 px, about 0.55 m/pixel and 1.1 km across. Fixed in `config.py`.

Get coordinates by right-clicking a spot on openstreetmap.org.

Output: `data/images/airport_road.tif` (georeferenced) and `.png` (to look at).

## Step 2 - extract roads

```bash
uv run extract.py list                    # see available models
uv run extract.py airport_road            # defaults to baseline
uv run extract.py airport_road all        # every model
```

Output:

| File | What it is |
|---|---|
| `outputs/geojson/airport_road_baseline.geojson` | the roads - open this in iD |
| `outputs/images/airport_road_baseline.png` | red overlay on the image |
| `outputs/images/airport_road_baseline_mask.png` | the raw mask |

## Step 3 (optional) - score against OSM

```bash
uv sync --extra validate
# put bangladesh-260823.osm.pbf in data/osm/
uv run validate.py airport_road all
```

Appends to `reports/validation.csv`.

**Recall** is trustworthy: of the roads OSM already has, how many did the model find?

**Precision** is not: OSM in Bangladesh is incomplete, so a real unmapped road
counts as "wrong". Low precision can mean false positives (canals, field edges,
rooftops) *or* genuinely new roads. Always open the overlay PNG before judging.

## Checking the result in iD

1. Open <https://www.openstreetmap.org/edit>
2. Press `F` (Map Data panel) -> **Custom Map Data** -> load your `.geojson`
3. The predicted roads appear as an overlay you can trace over

JOSM (`File -> Open`) makes it an editable layer. QGIS is best for checking
alignment - drag the `.tif` and `.geojson` in together.

> Never bulk-upload model output to OSM. That breaks the Automated Edits policy
> and gets reverted. A human accepts each road. Look at Meta's **RapiD** editor -
> it is iD built exactly for this workflow.

## Free accuracy improvements (no training)

Two changes that make every mask model more accurate, for zero extra data:

- **Overlapping tiles** (`core/tiling.py`) - patches overlap by
  `OVERLAP_PX` (96px) and blend with a smooth window, so a road crossing a
  patch boundary no longer breaks. Applies to every model automatically.
- **Test-time augmentation** (`dlinknet.tta = True` in `models/dlinknet.py`) -
  runs each patch flipped 4 ways and averages the result. Slower (~45s vs
  6s per image on this CPU) but more accurate; the model itself is
  untouched, still zero-shot pretrained.

## Models available now

```bash
uv run extract.py list
```

| name | what it is | status |
|---|---|---|
| `baseline` | Frangi ridge filter, no weights | works, weak - it's the floor |
| `dlinknet` | D-LinkNet34, DeepGlobe winner | **works well**, ~7s/image on CPU - use this |
| `unet` | Plain U-Net, Massachusetts Roads Dataset | works, but weak - trained at ~6m/px, misses BD's narrow roads |
| `samroad` | SAM ViT-B, outputs a connected graph | needs `sam_road` repo + GPU-ish time |
| `rngdet` | RNGDet++, transformer graph tracer | needs GPU, hours on CPU - skip for now |

`dlinknet` is the one to actually use day to day. Its weights
(`models/weights/dlinknet34_deepglobe.th`) are already installed and it
runs fine on this CPU-only laptop.

`samroad` and `rngdet` output road **graphs** directly instead of a pixel
mask - see `models/samroad.py` and `models/rngdet.py` for the one-time repo
clone + checkpoint download each needs. Both are heavy; try `dlinknet` first.

## Adding a model

Create one file in `models/`. Nothing else changes.

```python
# models/dlinknet.py
from models._base import RoadModel

class DLinkNet(RoadModel):
    name = "dlinknet"                 # goes into the output filename
    description = "DeepGlobe winner"

    def load(self):
        ...                           # load weights once
        return self

    def predict(self, patch):
        ...                           # patch: uint8 (512,512,3) RGB
        return prob                   # float32 (512,512), values 0..1
```

Then `uv run extract.py list` shows it. `core/registry.py` finds it automatically. Files starting with `_` are skipped.

`models/unet.py` is a working template to copy.

## Layout

```
download.py   extract.py   validate.py   config.py   pyproject.toml
core/     geo maths, vectorizing, model discovery, overlays  (do not edit)
models/   one file per model                                 (edit here)
data/images/    downloaded GeoTIFFs
data/osm/       the .pbf and cached road extracts
outputs/images/    overlays
outputs/geojson/   road vectors
reports/           validation.csv
```

## Notes

- **Imagery source is Esri World Imagery** - free, no key, and OSM has permission
  to trace from it. Never use Google Maps or Google Earth; the licence forbids it.
- **Why zoom 18?** Pretrained road models (DeepGlobe, SpaceNet) were trained at
  about 0.5 m/pixel. Sharper imagery is not better - at z19 a road looks twice as
  wide as the model expects and it fails.
- **`baseline`** is a Frangi ridge filter. No weights, no GPU. It exists to prove
  the pipeline works end to end. Every real model should beat it.
