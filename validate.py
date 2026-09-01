"""
OPTIONAL STEP 3 - score a model against the roads already in OpenStreetMap.

    uv run validate.py airport_road
    uv run validate.py airport_road all

Needs:  data/osm/bangladesh-260823.osm.pbf   and   pip install pyrosm geopandas

Appends one row per run to reports/validation.csv.

HOW TO READ THE NUMBERS
-----------------------
recall     Of the roads OSM already knows about, how many did the model find?
           TRUSTWORTHY. High is good.

precision  Of the roads the model drew, how many sit on a known OSM road?
           NOT TRUSTWORTHY on its own. OSM in Bangladesh is incomplete, so a
           real road that nobody has mapped yet counts as "wrong" here.
           Low precision may mean false positives (canals, field edges,
           rooftops) OR genuinely new roads - the thing you actually want.
           Always open the overlay PNG and look before judging.
"""
import csv
import math
import sys
from datetime import datetime

import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation

import config
from core.geo import raster_bounds_latlon, raster_center_latlon
from core.registry import discover
from core.vectorize import prob_to_mask

CSV_PATH = config.REPORTS / "validation.csv"
FIELDS = ["timestamp", "aoi", "model", "recall", "precision", "f1",
          "osm_km", "pred_km", "note"]


def osm_roads_geojson(name, src):
    """Extract driving roads from the .pbf inside this image's bbox, and cache them."""
    cache = config.OSM_DIR / f"roads_{name}.geojson"
    if cache.exists():
        import geopandas as gpd
        return gpd.read_file(cache)

    if not config.OSM_PBF.exists():
        sys.exit(f"Missing {config.OSM_PBF}\nPut bangladesh-260823.osm.pbf in data/osm/")
    try:
        from pyrosm import OSM
    except ImportError:
        sys.exit("pip install pyrosm geopandas")

    west, south, east, north = raster_bounds_latlon(src)
    print(f"[validate] reading OSM pbf for bbox {west:.4f},{south:.4f},{east:.4f},{north:.4f}")
    osm = OSM(str(config.OSM_PBF), bounding_box=[west, south, east, north])
    roads = osm.get_network(network_type="driving")
    if roads is None or roads.empty:
        sys.exit("[validate] OSM has no roads in this area at all - pick another AOI.")
    roads = roads.to_crs("EPSG:4326")
    roads.to_file(cache, driver="GeoJSON")
    print(f"[validate] cached {len(roads)} OSM roads -> {cache}")
    return roads


def rasterize_roads(roads, src, width_m):
    """Draw OSM road lines as a mask on the same pixel grid as the image."""
    g = roads.to_crs(src.crs)
    buffered = g.geometry.buffer(width_m)
    return rasterize(
        ((geom, 1) for geom in buffered if not geom.is_empty),
        out_shape=(src.height, src.width),
        transform=src.transform,
        fill=0, dtype="uint8",
    ).astype(bool)


def score(pred, truth, slack_px):
    """Buffered scoring: a prediction within slack_px of a road counts as correct."""
    k = np.ones((2 * slack_px + 1, 2 * slack_px + 1), bool)
    pred_fat = binary_dilation(pred, k)
    truth_fat = binary_dilation(truth, k)

    recall = truth[pred_fat].sum() / max(truth.sum(), 1)
    precision = pred[truth_fat].sum() / max(pred.sum(), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return float(recall), float(precision), float(f1)


def run(name, model_name):
    tif = config.DATA_IMAGES / f"{name}.tif"
    if not tif.exists():
        sys.exit(f"No image at {tif}")

    with rasterio.open(tif) as src:
        image = src.read([1, 2, 3]).transpose(1, 2, 0)
        roads = osm_roads_geojson(name, src)
        truth = rasterize_roads(roads, src, config.OSM_ROAD_WIDTH)
        lat, _ = raster_center_latlon(src)
        # Web Mercator pixels are stretched by 1/cos(lat); undo it for real metres
        m_per_px = abs(src.transform.a) * math.cos(math.radians(lat))

    models = discover()
    if model_name not in models:
        sys.exit(f"Unknown model '{model_name}'. Available: {', '.join(sorted(models))}")

    from extract import _predict_full
    model = models[model_name]().load()
    prob = _predict_full(model, image)
    pred = prob_to_mask(prob, config.THRESHOLD, config.MIN_ROAD_PX)

    recall, precision, f1 = score(pred, truth, config.MATCH_SLACK_PX)

    # rough centre-line kilometres: mask area / assumed road width
    area_per_px = m_per_px ** 2
    osm_km = truth.sum() * area_per_px / (2 * config.OSM_ROAD_WIDTH) / 1000
    pred_km = pred.sum() * area_per_px / (2 * config.OSM_ROAD_WIDTH) / 1000

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "aoi": name, "model": model_name,
        "recall": round(recall, 3), "precision": round(precision, 3),
        "f1": round(f1, 3),
        "osm_km": round(osm_km, 2), "pred_km": round(pred_km, 2),
        "note": "precision unreliable - OSM incomplete",
    }

    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)

    print(f"[validate] {name} / {model_name}")
    print(f"[validate]   recall    {recall:.3f}   <- trust this")
    print(f"[validate]   precision {precision:.3f}   <- OSM is incomplete, check the overlay")
    print(f"[validate]   f1        {f1:.3f}")
    print(f"[validate] appended to {CSV_PATH}")


USAGE = """usage: uv run validate.py <name> [model]

examples:
  uv run validate.py airport_road
  uv run validate.py airport_road all

Needs data/osm/bangladesh-260823.osm.pbf and:  uv sync --extra validate
"""


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        sys.exit(USAGE)

    name = argv[0]
    wanted = argv[1] if len(argv) > 1 else "baseline"
    targets = sorted(discover()) if wanted == "all" else [wanted]
    for m in targets:
        run(name, m)


if __name__ == "__main__":
    main()
