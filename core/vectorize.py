"""
Turn a road probability map into GeoJSON lines.

    probability (float 0-1)
      -> threshold        -> binary mask
      -> remove specks
      -> skeletonize      -> 1-pixel-wide centre lines
      -> trace            -> ordered pixel paths
      -> simplify         -> fewer points
      -> reproject        -> lat/lon LineStrings
"""
import inspect
import json
import math

import numpy as np
from shapely.geometry import LineString
from skimage.morphology import remove_small_objects, skeletonize

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


_RSO_ARGS = inspect.signature(remove_small_objects).parameters


def prob_to_mask(prob, threshold, min_px):
    """Threshold the probability map and delete blobs smaller than min_px."""
    mask = prob >= threshold
    if min_px > 0:
        # scikit-image renamed min_size -> max_size in 0.26 (and changed it to
        # "smaller than or equal to"), so support both spellings
        if "max_size" in _RSO_ARGS:
            mask = remove_small_objects(mask, max_size=min_px - 1)
        else:
            mask = remove_small_objects(mask, min_size=min_px)
    return mask


def _neighbors(sk, r, c):
    h, w = sk.shape
    out = []
    for dr, dc in _NB:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and sk[rr, cc]:
            out.append((rr, cc))
    return out


def mask_to_paths(mask):
    """Skeletonize the mask and return a list of pixel paths [(row, col), ...]."""
    sk = skeletonize(mask.astype(bool))
    pts = list(zip(*np.nonzero(sk)))
    if not pts:
        return []

    degree = {p: len(_neighbors(sk, *p)) for p in pts}
    nodes = [p for p in pts if degree[p] != 2]          # endpoints and junctions

    paths, started = [], set()

    def walk(a, b):
        path, prev, cur = [a, b], a, b
        while degree.get(cur, 0) == 2:
            nxt = [n for n in _neighbors(sk, *cur) if n != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            path.append(cur)
        return path

    for n in nodes:
        for nb in _neighbors(sk, *n):
            if frozenset((n, nb)) in started:
                continue
            p = walk(n, nb)
            started.add(frozenset((p[0], p[1])))
            started.add(frozenset((p[-1], p[-2])))
            paths.append(p)

    # closed loops have no endpoint or junction, so the walk above never reaches them
    left = set(pts) - {q for p in paths for q in p}
    while left:
        start = next(iter(left))
        left.discard(start)
        path, cur = [start], start
        while True:
            nxt = [n for n in _neighbors(sk, *cur) if n in left]
            if not nxt:
                break
            cur = nxt[0]
            left.discard(cur)
            path.append(cur)
        if len(path) > 1:
            paths.append(path)

    return paths


def paths_to_geojson(paths, transform, crs, center_lat,
                     simplify_m=2.0, min_len_m=15.0, props=None):
    """Convert pixel paths to a GeoJSON FeatureCollection in EPSG:4326."""
    from pyproj import Transformer

    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    # Web Mercator exaggerates distance by 1/cos(lat); undo it for real metres
    scale = math.cos(math.radians(center_lat))

    features = []
    for path in paths:
        if len(path) < 2:
            continue
        xy = [transform * (c + 0.5, r + 0.5) for r, c in path]
        line = LineString(xy).simplify(simplify_m)
        length_m = line.length * scale
        if length_m < min_len_m:
            continue

        xs = [p[0] for p in line.coords]
        ys = [p[1] for p in line.coords]
        lons, lats = to_wgs.transform(xs, ys)
        coords = [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lons, lats)]

        p = dict(props or {})
        p["length_m"] = round(float(length_m), 1)
        features.append({"type": "Feature", "properties": p,
                         "geometry": {"type": "LineString", "coordinates": coords}})

    return {"type": "FeatureCollection", "features": features}


def save_geojson(fc, path):
    with open(path, "w") as f:
        json.dump(fc, f)
    return path
