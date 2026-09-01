"""
All fixed settings for the project.
Change values HERE and nowhere else.
"""
from pathlib import Path

ROOT = Path(__file__).parent

# ---------- folders ----------
DATA_IMAGES = ROOT / "data" / "images"     # downloaded satellite GeoTIFFs
OSM_DIR     = ROOT / "data" / "osm"        # put bangladesh-260823.osm.pbf here
OUT_IMAGES  = ROOT / "outputs" / "images"  # overlay pictures  <name>_<model>.png
OUT_GEOJSON = ROOT / "outputs" / "geojson" # road vectors      <name>_<model>.geojson
REPORTS     = ROOT / "reports"             # validation.csv
WEIGHTS     = ROOT / "models" / "weights"  # put .pth files here

# ---------- imagery standard (hardcoded on purpose) ----------
ZOOM     = 18     # ~0.55 m/pixel in Bangladesh -> matches DeepGlobe/SpaceNet training data
SIZE_PX  = 2048   # downloaded image is 2048x2048  (~1.1 km x 1.1 km)
PATCH_PX = 512    # models see 512x512 patches (16 patches per image)
OVERLAP_PX = 96   # patches overlap by this much; blended so seams disappear

BASEMAP = ("https://services.arcgisonline.com/ArcGIS/rest/services/"
           "World_Imagery/MapServer/tile/{z}/{y}/{x}")
USER_AGENT = "road-extraction-research/0.1"

# ---------- mask -> vector ----------
THRESHOLD   = 0.5   # probability above this = road
MIN_ROAD_PX = 40    # delete blobs smaller than this (removes specks)
SIMPLIFY_M  = 2.0   # line smoothing tolerance, metres

# ---------- validation ----------
OSM_PBF        = OSM_DIR / "bangladesh-260823.osm.pbf"
OSM_ROAD_WIDTH = 4.0   # metres, half-width used to draw OSM roads as a mask
MATCH_SLACK_PX = 8     # a prediction within this many pixels of OSM counts as a match

for _d in (DATA_IMAGES, OSM_DIR, OUT_IMAGES, OUT_GEOJSON, REPORTS, WEIGHTS):
    _d.mkdir(parents=True, exist_ok=True)
