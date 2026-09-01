"""
STEP 1 - download a satellite image for a lat/lon.

    uv run download.py 23.846073, 90.389624 airport_road

Writes:
    data/images/<name>.tif   georeferenced, used by extract.py
    data/images/<name>.png   plain picture, just to look at

Zoom and size are fixed in config.py (z18, 2048x2048, about 1.1 km across).
"""
import io
import math
import re
import sys

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.transform import Affine

import config
from core.geo import ORIGIN, TILE_PX, ground_res, latlon_to_pixel

Image.MAX_IMAGE_PIXELS = None


def _fetch_tile(session, z, x, y):
    url = config.BASEMAP.format(z=z, x=x, y=y)
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def download(lat, lon, name, zoom=config.ZOOM, size=config.SIZE_PX):
    # where the image sits in the global pixel grid
    cx, cy = latlon_to_pixel(lat, lon, zoom)
    x0, y0 = int(round(cx - size / 2)), int(round(cy - size / 2))

    tx0, ty0 = x0 // TILE_PX, y0 // TILE_PX
    tx1, ty1 = (x0 + size - 1) // TILE_PX, (y0 + size - 1) // TILE_PX
    n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)

    gsd = ground_res(lat, zoom)
    print(f"[download] {name}  lat={lat} lon={lon} z={zoom}")
    print(f"[download] {size}x{size} px  ~{gsd:.2f} m/px  "
          f"~{size * gsd / 1000:.2f} km across  ({n_tiles} tiles)")

    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * TILE_PX, (ty1 - ty0 + 1) * TILE_PX))
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT

    done = 0
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            tile = _fetch_tile(session, zoom, tx, ty)
            canvas.paste(tile, ((tx - tx0) * TILE_PX, (ty - ty0) * TILE_PX))
            done += 1
            print(f"\r[download]   tile {done}/{n_tiles}", end="", flush=True)
    print()

    ox, oy = x0 - tx0 * TILE_PX, y0 - ty0 * TILE_PX
    img = canvas.crop((ox, oy, ox + size, oy + size))

    # georeference: top-left corner in Web Mercator metres
    res = 2.0 * ORIGIN / (TILE_PX * 2 ** zoom)
    west = -ORIGIN + x0 * res
    north = ORIGIN - y0 * res
    transform = Affine.translation(west, north) * Affine.scale(res, -res)

    arr = np.asarray(img)
    tif = config.DATA_IMAGES / f"{name}.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=size, width=size,
                       count=3, dtype="uint8", crs="EPSG:3857",
                       transform=transform, compress="deflate") as dst:
        dst.write(arr.transpose(2, 0, 1))

    png = config.DATA_IMAGES / f"{name}.png"
    img.save(png)

    print(f"[download] saved {tif}")
    print(f"[download] saved {png}")
    return tif


USAGE = """usage: uv run download.py <lat> <lon> [name]

examples:
  uv run download.py 23.846073, 90.389624 airport_road
  uv run download.py 23.846073 90.389624 airport_road
  uv run download.py 23.846073,90.389624

Paste coordinates straight from OpenStreetMap or Google - the comma is fine.
If you leave out the name, one is made from the coordinates.
"""


def parse_cli(argv):
    """Accept '23.8, 90.3 name', '23.8,90.3 name' or '23.8 90.3 name'."""
    tokens = [t for a in argv for t in re.split(r"[,\s]+", a) if t]
    if len(tokens) < 2:
        sys.exit(USAGE)
    try:
        lat, lon = float(tokens[0]), float(tokens[1])
    except ValueError:
        sys.exit(USAGE)
    name = "_".join(tokens[2:]) if len(tokens) > 2 else f"aoi_{lat:.5f}_{lon:.5f}"
    return lat, lon, re.sub(r"[^0-9A-Za-z._-]", "_", name)


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        sys.exit(USAGE)

    lat, lon, name = parse_cli(argv)
    if not (20.0 <= lat <= 27.0 and 88.0 <= lon <= 93.0):
        print("[warn] coordinates are outside Bangladesh - are lat and lon swapped?",
              file=sys.stderr)

    download(lat, lon, name)


if __name__ == "__main__":
    main()
