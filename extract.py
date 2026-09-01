"""
STEP 2 - run a model on a downloaded image and produce roads.

    uv run extract.py airport_road
    uv run extract.py airport_road baseline
    uv run extract.py airport_road all
    uv run extract.py list

Writes:
    outputs/geojson/<name>_<model>.geojson   <- drag this into iD
    outputs/images/<name>_<model>.png        <- red overlay, judge by eye
    outputs/images/<name>_<model>_mask.png
"""
import sys
import time

import numpy as np
import rasterio

import config
from core.geo import raster_center_latlon
from core.registry import discover
from core.tiling import tiled_predict
from core.vectorize import (mask_to_paths, paths_to_geojson, paths_to_mask,
                            prob_to_mask, save_geojson)
from core.viz import save_mask, save_overlay


def run(name, model_name):
    tif = config.DATA_IMAGES / f"{name}.tif"
    if not tif.exists():
        sys.exit(f"No image at {tif}\nRun download.py first.")

    with rasterio.open(tif) as src:
        image = src.read([1, 2, 3]).transpose(1, 2, 0)
        transform, crs = src.transform, src.crs
        lat, lon = raster_center_latlon(src)

    models = discover()
    if model_name not in models:
        sys.exit(f"Unknown model '{model_name}'. Available: {', '.join(sorted(models))}")

    print(f"[extract] {name}  model={model_name}")
    t0 = time.time()
    model = models[model_name]().load()

    if getattr(model, "outputs", "mask") == "graph":
        # the model returns road lines directly - no thresholding needed
        paths = model.predict_graph(image)
        mask = paths_to_mask(paths, image.shape[:2])
    else:
        prob = tiled_predict(model, image, config.PATCH_PX, config.OVERLAP_PX)
        mask = prob_to_mask(prob, config.THRESHOLD, config.MIN_ROAD_PX)
        paths = mask_to_paths(mask)
    fc = paths_to_geojson(paths, transform, crs, lat,
                          simplify_m=config.SIMPLIFY_M,
                          props={"model": model_name, "aoi": name})

    stem = f"{name}_{model_name}"
    gj = save_geojson(fc, config.OUT_GEOJSON / f"{stem}.geojson")
    ov = save_overlay(image, mask, config.OUT_IMAGES / f"{stem}.png")
    mk = save_mask(mask, config.OUT_IMAGES / f"{stem}_mask.png")

    total_km = sum(f["properties"]["length_m"] for f in fc["features"]) / 1000.0
    print(f"[extract] {len(fc['features'])} road lines, {total_km:.2f} km total"
          f"  ({time.time() - t0:.1f}s)")
    print(f"[extract] saved {gj}")
    print(f"[extract] saved {ov}")
    print(f"[extract] saved {mk}")
    return gj


USAGE = """usage: uv run extract.py <name> [model]

examples:
  uv run extract.py airport_road            # uses the baseline model
  uv run extract.py airport_road baseline
  uv run extract.py airport_road all        # every model
  uv run extract.py list                    # show available models

<name> is the name you gave download.py.
"""


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        sys.exit(USAGE)

    models = discover()
    if argv[0] in ("list", "--list"):
        for n, cls in sorted(models.items()):
            print(f"  {n:<12} {cls.description}")
        return

    name = argv[0]
    wanted = argv[1] if len(argv) > 1 else "baseline"
    targets = sorted(models) if wanted == "all" else [wanted]
    for m in targets:
        run(name, m)


if __name__ == "__main__":
    main()
