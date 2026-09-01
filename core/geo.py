"""
Coordinate maths.

Two coordinate systems are used:
  EPSG:4326 - plain lat/lon degrees   (what iD / JOSM / GeoJSON use)
  EPSG:3857 - Web Mercator metres     (what map tiles use)
"""
import math

TILE_PX = 256
ORIGIN = 20037508.342789244   # half the width of the Web Mercator world, in metres


def latlon_to_pixel(lat, lon, zoom):
    """Global pixel coordinate of a lat/lon at a given zoom level."""
    n = TILE_PX * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def pixel_to_latlon(x, y, zoom):
    """Inverse of latlon_to_pixel."""
    n = TILE_PX * (2 ** zoom)
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def mercator_res(zoom):
    """Metres per pixel in EPSG:3857 (constant, not corrected for latitude)."""
    return 2.0 * ORIGIN / (TILE_PX * (2 ** zoom))


def ground_res(lat, zoom):
    """Real metres per pixel on the ground at this latitude. This is the GSD."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def raster_center_latlon(src):
    """Centre of an open rasterio dataset, as (lat, lon)."""
    from pyproj import Transformer
    b = src.bounds
    cx, cy = (b.left + b.right) / 2.0, (b.bottom + b.top) / 2.0
    t = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(cx, cy)
    return lat, lon


def raster_bounds_latlon(src):
    """Bounding box of an open rasterio dataset as (west, south, east, north) in degrees."""
    from pyproj import Transformer
    b = src.bounds
    t = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
    west, south = t.transform(b.left, b.bottom)
    east, north = t.transform(b.right, b.top)
    return west, south, east, north
