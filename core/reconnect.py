"""
Occlusion post-processing for graph-output models, adapted from:
  Fathul'ibad et al., "Improving SAM-Road Model for Occlusion Handling in
  Road Networks Extraction from Satellite Images with Gamma Correction and
  Modified A* Algorithm", JISEBI 12(1), 2026.

Two independent steps, both applied AFTER a model has already run - no
retraining, no change to the model's weights:

  1. gamma_correct() - brightens a road-probability mask so segments dimmed
     by shadows/trees/buildings become detectable. Can only help pixels the
     model already gave *some* nonzero confidence to; it cannot invent
     detections where the model saw nothing.

  2. reconnect_graph() - bridges disconnected graph fragments by running A*
     across a small window, using the (gamma-corrected) probability mask as
     a cost surface so the path prefers pixels the model was at least
     somewhat confident about.

SIMPLIFICATIONS vs the paper (documented, not hidden)
------------------------------------------------------
- The paper's MIN_GRAPH_DISTANCE governs how close two nodes may already be
  *through the existing graph* before a new bridge is disallowed. Here it is
  approximated as "candidates must be in a different connected component" -
  correct for a graph this sparse (see caller), a real simplification for a
  denser one.
- Endpoints are only bridged in the roughly-forward direction implied by
  their single existing edge (a dead end shouldn't jump backwards). Isolated
  points (no edge at all) are bridged in any direction.
- Only the nearest valid candidate per endpoint is attempted, once, in a
  single pass - not the paper's full iterative graph search.
"""
import heapq

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

_NB8 = [(-1, -1, 2 ** 0.5), (-1, 0, 1.0), (-1, 1, 2 ** 0.5), (0, -1, 1.0),
        (0, 1, 1.0), (1, -1, 2 ** 0.5), (1, 0, 1.0), (1, 1, 2 ** 0.5)]


def gamma_correct(mask_uint8, gamma):
    """f0 = fi^(1/gamma), gamma > 1 brightens dim regions. gamma<=1 is a no-op."""
    if gamma <= 1.0:
        return mask_uint8
    norm = mask_uint8.astype(np.float32) / 255.0
    corrected = norm ** (1.0 / gamma)
    return (corrected * 255.0).astype(np.uint8)


def _astar(cost_norm, p0, p1, pad, penalty):
    """Grid A* from p0 to p1 (x, y pixel coords) over a small cropped window.
    cost_norm: float array in [0,1], higher = more road-like = cheaper to cross.
    Returns a list of (row, col) pixels, or None if p0/p1 fall outside the window."""
    h, w = cost_norm.shape
    x0, y0 = p0
    x1, y1 = p1
    r0, r1 = sorted((int(y0), int(y1)))
    c0, c1 = sorted((int(x0), int(x1)))
    r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
    r1, c1 = min(h - 1, r1 + pad), min(w - 1, c1 + pad)

    start = (int(y0) - r0, int(x0) - c0)
    goal = (int(y1) - r0, int(x1) - c0)
    H, W = r1 - r0 + 1, c1 - c0 + 1
    if not (0 <= start[0] < H and 0 <= start[1] < W and 0 <= goal[0] < H and 0 <= goal[1] < W):
        return None

    def step_cost(r, c):
        return 1.0 + penalty * (1.0 - cost_norm[r0 + r, c0 + c])

    def heuristic(r, c):
        return ((r - goal[0]) ** 2 + (c - goal[1]) ** 2) ** 0.5

    open_heap = [(heuristic(*start), 0.0, start)]
    best_g = {start: 0.0}
    came_from = {}
    visited = set()

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return [(r0 + r, c0 + c) for r, c in path]

        r, c = cur
        for dr, dc, base in _NB8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                ng = g + base * step_cost(nr, nc)
                if ng < best_g.get((nr, nc), float("inf")):
                    best_g[(nr, nc)] = ng
                    came_from[(nr, nc)] = cur
                    heapq.heappush(open_heap, (ng + heuristic(nr, nc), ng, (nr, nc)))
    return None


def reconnect_graph(points_xy, edges, cost_norm, max_straight_px,
                    astar_pad=12, cost_penalty=5.0):
    """
    points_xy   : (N, 2) array of (x, y) pixel coordinates
    edges       : list of (i, j) index pairs into points_xy - the graph as it stands
    cost_norm   : float array in [0,1], same shape as the image (gamma-corrected road mask)
    max_straight_px : only search for a bridge within this pixel radius

    Returns extra paths (each [(row, col), ...]) to append alongside the
    model's own edges - bridges across gaps the model left disconnected.
    """
    n = len(points_xy)
    if n == 0:
        return []

    if edges:
        rows = [a for a, b in edges] + [b for a, b in edges]
        cols = [b for a, b in edges] + [a for a, b in edges]
        adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    else:
        adj = coo_matrix((n, n))
    _, labels = connected_components(adj, directed=False)

    degree = np.zeros(n, dtype=int)
    neighbor = {}
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
        neighbor[a] = b
        neighbor[b] = a

    candidates = np.flatnonzero(degree <= 1)   # dead ends and isolated points
    bridges = []
    bridged_pairs = set()

    for i in candidates:
        pi = points_xy[i]
        direction = None
        if degree[i] == 1:
            d = pi - points_xy[neighbor[i]]
            norm = np.linalg.norm(d)
            if norm > 0:
                direction = d / norm

        best_j, best_dist = None, float("inf")
        for j in candidates:
            if j == i or labels[j] == labels[i]:
                continue
            pj = points_xy[j]
            dist = np.linalg.norm(pj - pi)
            if dist > max_straight_px or dist >= best_dist:
                continue
            if direction is not None:
                fwd = pj - pi
                fwd_norm = np.linalg.norm(fwd)
                if fwd_norm == 0 or np.dot(fwd / fwd_norm, direction) <= 0:
                    continue
            best_j, best_dist = j, dist

        if best_j is None:
            continue
        pair = (min(i, best_j), max(i, best_j))
        if pair in bridged_pairs:
            continue

        path = _astar(cost_norm, pi, points_xy[best_j], astar_pad, cost_penalty)
        if path is not None and len(path) > 1:
            bridges.append(path)
            bridged_pairs.add(pair)
            labels[labels == labels[best_j]] = labels[i]   # merge components

    return bridges
