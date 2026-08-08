"""
skeletonization.py
==================
Stage between segmentation and the graph pipeline: turns PathMamba's raw
`prob_mask` into a networkx graph carrying the exact node/edge schema that
`zlevel.py` / `healing.py` / `pipeline.py` already expect:

    node attrs : x, y            (world coords in prediction.transform's frame)
    edge attrs : length          (meters), bridge (bool), layer (int),
                 geometry         (world (x, y) polyline for the traced centerline),
                 synthetic (bool, always False here -- these are *observed* roads,
                            not healing-fabricated bridges)

This is the "one piece still to write" referenced in README.md / pipeline.py.
Today the graph came from synthetic_occlusion.py; once real inference exists,
`graph_from_prediction(prediction)` produces the same-shaped graph from a real
RoadPrediction with zero downstream changes.

Pipeline order matters and mirrors the deck's Stage 5 wording:
    binarize -> (optional cleanup) -> Zhang-Suen thinning -> vectorize to a
    junction/endpoint graph (degree-2 chains collapsed) -> prune skeleton spurs.

DELIBERATE DESIGN NOTES (flagged, not silently chosen):

* Coordinate frame. Node x,y are emitted in the *world* frame that
  `prediction.transform.world_to_pixel` inverts -- NOT raw pixels. healing.py
  computes gap distance as hypot(dx, dy) and gates it at <= 50 *meters*; raw
  pixel coords would make that gate meaningless. The transform has no inverse
  method, so we recover pixel->world by probing it at three points (valid for
  any affine transform, incl. SimpleAffine and a rasterio Affine adapter). If
  transform is None we fall back to a pixel frame (x=col, y=row); that mode is
  for topology tests only, since the meter-based healing gates no longer apply.

* Grade separation (bridge/layer). A road-probability raster carries NO
  grade-separation signal, so every edge here is emitted bridge=False, layer=0
  (surface). On real data, flyover/surface distinction must come from an OSM
  shared-node overlay join (zlevel.py's real input), not from the mask. This is
  the honest boundary of what skeletonization alone can recover.

* Local heading vs. collapsed edges. Collapsing degree-2 chains means a stub's
  graph-neighbor is the *far* node, so healing's angle gate sees the chord
  bearing, not the local road tangent. We attach the full world polyline as edge
  'geometry' so a future healing refinement can use the true tangent -- but we do
  NOT modify healing here (its gate logic is locked). Flag, not a fix.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import networkx as nx

try:  # scipy is already a hard dep of the graph pipeline (see requirements.txt)
    from scipy import ndimage as _ndi
    _HAVE_NDIMAGE = True
except Exception:  # pragma: no cover - scipy is expected present
    _HAVE_NDIMAGE = False


# ---------------------------------------------------------------------------
# Hysteresis thresholding
# ---------------------------------------------------------------------------

def hysteresis_threshold(prob: np.ndarray, low: float, high: float) -> np.ndarray:
    """Two-threshold binarization: keep any pixel >= high, plus any pixel >= low
    that is 8-connected to a high pixel via a chain of low pixels. Standard
    Canny-style hysteresis. Falls back to a single-threshold >= low if scipy
    isn't available.

    Why: Tversky-trained road segmentation on hard scenes (canopy, shadow)
    often peaks in 0.2-0.4 along real roads. A single high threshold drops
    those; a single low threshold lights up the background. Hysteresis keeps
    faint continuations that anchor to a confident core, without inflating
    isolated background noise.
    """
    if not _HAVE_NDIMAGE:
        return prob >= low
    high_mask = prob >= high
    low_mask = prob >= low
    if not high_mask.any():
        return np.zeros_like(prob, dtype=bool)
    # label the low-threshold components; keep any component that contains at
    # least one high-threshold pixel.
    labels, n = _ndi.label(low_mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(prob, dtype=bool)
    keep_labels = set(np.unique(labels[high_mask]).tolist()) - {0}
    if not keep_labels:
        return np.zeros_like(prob, dtype=bool)
    return np.isin(labels, list(keep_labels))


# ---------------------------------------------------------------------------
# Douglas-Peucker polyline simplification (pure numpy, no shapely required)
# ---------------------------------------------------------------------------

def _douglas_peucker(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    """Iterative DP: drops vertices whose perpendicular distance to the chord
    between kept neighbors is < epsilon. Preserves first and last vertex."""
    if len(points) < 3 or epsilon <= 0:
        return points
    pts = np.asarray(points, dtype=np.float64)
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        x0, y0 = pts[i]
        x1, y1 = pts[j]
        dx, dy = x1 - x0, y1 - y0
        seg_len2 = dx * dx + dy * dy
        if seg_len2 == 0.0:
            # degenerate chord; keep midpoint by furthest point from p0
            d = np.hypot(pts[i+1:j, 0] - x0, pts[i+1:j, 1] - y0)
        else:
            # perpendicular distance from each interior point to the chord
            num = np.abs(dy * pts[i+1:j, 0] - dx * pts[i+1:j, 1] + x1 * y0 - y1 * x0)
            d = num / math.sqrt(seg_len2)
        if d.size == 0:
            continue
        k_local = int(np.argmax(d))
        if d[k_local] > epsilon:
            k = i + 1 + k_local
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return [tuple(pts[i]) for i in range(len(pts)) if keep[i]]


def simplify_edge_geometries(G: nx.Graph, epsilon_m: float) -> nx.Graph:
    """Apply Douglas-Peucker to every edge's `geometry` polyline in place.
    epsilon_m in world units (meters when transform is present)."""
    if epsilon_m <= 0:
        return G
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        if geom and len(geom) >= 3:
            data["geometry"] = _douglas_peucker(geom, epsilon_m)
    return G


# ---------------------------------------------------------------------------
# Local tangent at a stub (for healing's angle gate)
# ---------------------------------------------------------------------------

def local_tangent_deg(G: nx.Graph, node, length_m: float = 15.0) -> Optional[float]:
    """Bearing (0-180, mod-180) of the road at `node`, measured over the first
    `length_m` of its incident edge's traced geometry -- NOT the chord to the
    far collapsed neighbor.

    Returns None if the node has no edge with usable geometry (falls back to
    chord bearing at the call site). This fixes the "chord vs. tangent"
    limitation flagged in the module docstring: for a curved road that
    collapses through degree-2 pruning, the chord to the far node can be far
    from the actual local heading at the stub.
    """
    if G.degree[node] == 0:
        return None
    x0, y0 = G.nodes[node]["x"], G.nodes[node]["y"]
    # pick the incident edge; for degree>1 (a stub-ish junction), average
    # tangents across incident edges by returning the closest to any surviving
    # heading -- but here we just return the first edge's tangent; healing.py
    # calls this per-edge in its own loop.
    for nbr in G.neighbors(node):
        data = G.edges[node, nbr]
        geom = data.get("geometry") or []
        if len(geom) < 2:
            continue
        # orient the polyline so it starts at `node`
        dx_start = math.hypot(geom[0][0] - x0, geom[0][1] - y0)
        dx_end = math.hypot(geom[-1][0] - x0, geom[-1][1] - y0)
        if dx_end < dx_start:
            geom = list(reversed(geom))
        # walk along until we've covered length_m of arc, take that vector
        acc = 0.0
        px, py = geom[0]
        for (qx, qy) in geom[1:]:
            seg = math.hypot(qx - px, qy - py)
            acc += seg
            if acc >= length_m:
                return math.degrees(math.atan2(qy - y0, qx - x0)) % 180.0
            px, py = qx, qy
        # geometry shorter than length_m: use its far end
        gx, gy = geom[-1]
        return math.degrees(math.atan2(gy - y0, gx - x0)) % 180.0
    return None


def local_tangents_deg(G: nx.Graph, node, length_m: float = 15.0) -> list[float]:
    """Per-incident-edge tangent bearings at `node`, using each edge's own
    geometry (falls back to chord to neighbor when geometry is missing)."""
    if G.degree[node] == 0:
        return []
    x0, y0 = G.nodes[node]["x"], G.nodes[node]["y"]
    out = []
    for nbr in G.neighbors(node):
        data = G.edges[node, nbr]
        geom = data.get("geometry") or []
        if len(geom) < 2:
            # fallback: chord to neighbor
            x1, y1 = G.nodes[nbr]["x"], G.nodes[nbr]["y"]
            out.append(math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0)
            continue
        dx_start = math.hypot(geom[0][0] - x0, geom[0][1] - y0)
        dx_end = math.hypot(geom[-1][0] - x0, geom[-1][1] - y0)
        if dx_end < dx_start:
            geom = list(reversed(geom))
        acc = 0.0
        px, py = geom[0]
        chosen = None
        for (qx, qy) in geom[1:]:
            acc += math.hypot(qx - px, qy - py)
            if acc >= length_m:
                chosen = (qx, qy)
                break
            px, py = qx, qy
        if chosen is None:
            chosen = geom[-1]
        gx, gy = chosen
        out.append(math.degrees(math.atan2(gy - y0, gx - x0)) % 180.0)
    return out


# ---------------------------------------------------------------------------
# Edge-mean-probability filter
# ---------------------------------------------------------------------------

def filter_edges_by_mean_prob(G: nx.Graph, prob_mask: np.ndarray,
                              transform, min_mean: float) -> nx.Graph:
    """Drop edges whose mean prob_mask value along their traced polyline falls
    below min_mean. Cheap topology cleaner: catches traced edges that skim just
    above the binarization threshold across their whole run (mostly false
    positives), without raising the global threshold and losing genuine
    high-confidence roads elsewhere.

    Uses edge['geometry'] (world coords); converts to pixel via transform.
    If transform is None, assumes world coords already equal pixel coords.
    """
    if min_mean <= 0.0:
        return G
    H, W = prob_mask.shape
    to_drop = []
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry") or []
        if len(geom) < 2:
            continue
        # sample along geometry at pixel-density
        pxvals = []
        for (x, y) in geom:
            if transform is None:
                row, col = int(round(y)), int(round(x))
            else:
                r, c = transform.world_to_pixel(x, y)
                row, col = int(round(r)), int(round(c))
            if 0 <= row < H and 0 <= col < W:
                pxvals.append(float(prob_mask[row, col]))
        if pxvals and (sum(pxvals) / len(pxvals)) < min_mean:
            to_drop.append((u, v))
    for u, v in to_drop:
        if G.has_edge(u, v):
            G.remove_edge(u, v)
    # drop any nodes now isolated
    isolated = [n for n in G.nodes() if G.degree[n] == 0]
    G.remove_nodes_from(isolated)
    return G


# ---------------------------------------------------------------------------
# Zhang-Suen thinning (vectorized, pure numpy -- no scikit-image dependency).
# The deck names Zhang-Suen explicitly; skimage.morphology.skeletonize(
# method="zhang") is the same algorithm and is cross-checked in the test harness.
# ---------------------------------------------------------------------------

def zhang_suen_skeletonize(binary: np.ndarray) -> np.ndarray:
    """Thin a boolean/0-1 foreground mask to a 1-pixel-wide skeleton.

    Standard Zhang-Suen two-sub-iteration thinning, applied simultaneously
    (not pixel-sequentially) each pass via array shifts on a zero-padded copy
    so there is no np.roll wraparound at borders.
    """
    img = (np.asarray(binary) > 0).astype(np.uint8)
    if img.sum() == 0:
        return img.astype(bool)

    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            # 8 neighbors, clockwise from North (P2..P9 in the paper's naming)
            P2 = P[0:-2, 1:-1]   # N
            P3 = P[0:-2, 2:]     # NE
            P4 = P[1:-1, 2:]     # E
            P5 = P[2:, 2:]       # SE
            P6 = P[2:, 1:-1]     # S
            P7 = P[2:, 0:-2]     # SW
            P8 = P[1:-1, 0:-2]   # W
            P9 = P[0:-2, 0:-2]   # NW

            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9  # number of on-neighbors

            seq = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
            A = np.zeros_like(B)  # 0->1 transitions around the ring
            for k in range(8):
                A += ((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8)

            cond = (img == 1) & (B >= 2) & (B <= 6) & (A == 1)
            if step == 0:
                cond &= (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                cond &= (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)

            if cond.any():
                img[cond] = 0
                changed = True

    return img.astype(bool)


def _skeletonize(binary: np.ndarray, prefer_skimage: bool = True) -> np.ndarray:
    """Zhang-Suen thinning. Prefers scikit-image's `skeletonize(method="zhang")`
    when installed -- it is the same Zhang-Suen algorithm plus post-cleanup that
    fully thins diagonal 2px staircases (strict textbook Zhang-Suen leaves a few
    residual pixels there, which would spawn phantom mini-junctions in the graph).
    Falls back to the dependency-free pure-numpy implementation above otherwise."""
    if prefer_skimage:
        try:
            from skimage.morphology import skeletonize as _sk
            return np.asarray(_sk(np.asarray(binary) > 0, method="zhang"), dtype=bool)
        except Exception:
            pass
    return zhang_suen_skeletonize(binary)


# ---------------------------------------------------------------------------
# Transform inversion (pixel -> world) by probing. Works for any affine
# world_to_pixel(x, y) -> (row, col), which is all model_io.WorldToPixel
# implementations are.
# ---------------------------------------------------------------------------

def _build_pixel_to_world(transform) -> Callable[[float, float], tuple[float, float]]:
    """Return f(row, col) -> (x, y). If transform is None, pixel frame: x=col, y=row."""
    if transform is None:
        return lambda row, col: (float(col), float(row))

    # world_to_pixel is affine: (row, col) = M @ (x, y) + b. Probe 3 points.
    r0, c0 = transform.world_to_pixel(0.0, 0.0)
    rx, cx = transform.world_to_pixel(1.0, 0.0)
    ry, cy = transform.world_to_pixel(0.0, 1.0)
    # M maps (x, y) -> (row - r0, col - c0)
    M = np.array([[rx - r0, ry - r0],
                  [cx - c0, cy - c0]], dtype=np.float64)
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-12:
        raise ValueError("transform.world_to_pixel appears non-invertible (degenerate affine).")
    Minv = np.linalg.inv(M)
    b = np.array([r0, c0], dtype=np.float64)

    def pixel_to_world(row: float, col: float) -> tuple[float, float]:
        xy = Minv @ (np.array([row, col], dtype=np.float64) - b)
        return float(xy[0]), float(xy[1])

    return pixel_to_world


# ---------------------------------------------------------------------------
# Skeleton -> graph
# ---------------------------------------------------------------------------

# 8-neighborhood offsets (row, col)
_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _neighbor_degree(skel: np.ndarray) -> np.ndarray:
    """Count of on 8-neighbors for every pixel (0 where the pixel itself is off)."""
    P = np.pad(skel.astype(np.uint8), 1)
    deg = np.zeros(skel.shape, dtype=np.uint8)
    for dr, dc in _NB8:
        deg += P[1 + dr:1 + dr + skel.shape[0], 1 + dc:1 + dc + skel.shape[1]]
    deg[~skel] = 0
    return deg


def _skel_neighbors(skel: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    h, w = skel.shape
    out = []
    for dr, dc in _NB8:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w and skel[rr, cc]:
            out.append((rr, cc))
    return out


def _trace_edges(skel: np.ndarray):
    """Walk the skeleton into edges between node pixels (endpoints deg==1,
    junctions deg>=3). Degree-2 run pixels are absorbed into the edge polyline.

    Returns (node_pixels, edges) where edges is a list of
    (node_a_px, node_b_px, [pixel polyline a..b inclusive]).
    """
    deg = _neighbor_degree(skel)
    node_mask = skel & ((deg == 1) | (deg >= 3))
    node_pixels = set(zip(*np.nonzero(node_mask)))

    edges = []
    visited_steps: set[frozenset] = set()  # undirected pixel-step dedup

    def walk(start, first):
        path = [start, first]
        visited_steps.add(frozenset((start, first)))
        prev, cur = start, first
        while cur not in node_pixels:
            nbrs = [n for n in _skel_neighbors(skel, *cur) if n != prev]
            # a clean degree-2 run pixel has exactly one forward neighbor
            if len(nbrs) != 1:
                break  # ragged spot; stop the trace here
            nxt = nbrs[0]
            visited_steps.add(frozenset((cur, nxt)))
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    for start in node_pixels:
        for nb in _skel_neighbors(skel, *start):
            if frozenset((start, nb)) in visited_steps:
                continue
            path = walk(start, nb)
            end = path[-1]
            if end in node_pixels:
                edges.append((start, end, path))
            # else: dangling ragged trace -- dropped (rare thinning artifact)

    # Pure cycles have no node pixel: handle each such skeleton component by
    # promoting one arbitrary pixel to a node and tracing a self-loop.
    if _HAVE_NDIMAGE:
        labels, n = _ndi.label(skel, structure=np.ones((3, 3)))
        comps_with_node = {labels[r, c] for (r, c) in node_pixels}
        for lab in range(1, n + 1):
            if lab in comps_with_node:
                continue
            rs, cs = np.nonzero(labels == lab)
            if len(rs) < 3:
                continue  # tiny speck, skip
            seed = (int(rs[0]), int(cs[0]))
            node_pixels.add(seed)
            nbs = _skel_neighbors(skel, *seed)
            if nbs:
                path = walk(seed, nbs[0])
                edges.append((seed, path[-1], path))

    return node_pixels, edges


def _polyline_length_world(path_px, p2w) -> tuple[float, list[tuple[float, float]]]:
    """World length (sum of segment lengths) and the world polyline for a pixel path."""
    world = [p2w(r, c) for (r, c) in path_px]
    length = 0.0
    for (x0, y0), (x1, y1) in zip(world[:-1], world[1:]):
        length += math.hypot(x1 - x0, y1 - y0)
    return length, world


def _prune_and_collapse(G: nx.Graph, min_branch_m: float) -> nx.Graph:
    """Remove short dangling spurs (thinning artifacts), then collapse any
    degree-2 node by merging its two edges. Iterated to a fixed point because
    pruning a spur can turn a junction into a pass-through node."""
    changed = True
    while changed:
        changed = False

        # 1) drop short leaf edges (an endpoint with a stubby branch)
        for n in [n for n in G.nodes() if G.degree[n] == 1]:
            if G.degree[n] != 1:
                continue
            (u, v, data) = next(iter(G.edges(n, data=True)))
            if data.get("length", 0.0) < min_branch_m:
                G.remove_edge(u, v)
                if G.degree[n] == 0:
                    G.remove_node(n)
                changed = True

        # 2) collapse degree-2 pass-through nodes into a single merged edge
        for n in [n for n in G.nodes() if G.degree[n] == 2]:
            if G.degree[n] != 2:
                continue
            (a, b_and_data) = None, None
            edges = list(G.edges(n, data=True))
            if len(edges) != 2:
                continue  # self-loop or multigraph oddity; leave it
            (n1, a, d1), (n2, b, d2) = edges
            if a == b or a == n or b == n:
                continue  # would create a self-loop; leave the node in place
            if G.has_edge(a, b):
                continue  # merging would collide with an existing edge; leave it

            geom1 = _oriented_geom(d1.get("geometry", []), G.nodes[a], G.nodes[n])
            geom2 = _oriented_geom(d2.get("geometry", []), G.nodes[n], G.nodes[b])
            merged_geom = geom1[:-1] + geom2 if geom1 and geom2 else (geom1 or geom2)
            merged_len = d1.get("length", 0.0) + d2.get("length", 0.0)
            G.remove_node(n)
            G.add_edge(a, b, length=merged_len, bridge=False, layer=0,
                       synthetic=False, geometry=merged_geom,
                       n_px=d1.get("n_px", 0) + d2.get("n_px", 0))
            changed = True

    return G


def _merge_short_junction_edges(G: nx.Graph, thresh_m: float) -> nx.Graph:
    """Zhang-Suen splits a true 4-way crossing into two adjacent 3-way junctions
    joined by a 1-2px edge (the classic X -> Y-Y artifact). Contract edges
    shorter than thresh_m whose BOTH endpoints are junctions (deg>=3) back into
    a single node. Endpoints (deg 1) are never merged -- a genuinely short real
    stub must survive."""
    if thresh_m <= 0:
        return G
    changed = True
    while changed:
        changed = False
        for u, v, data in list(G.edges(data=True)):
            if not G.has_edge(u, v):
                continue
            if data.get("length", math.inf) >= thresh_m:
                continue
            if G.degree[u] >= 3 and G.degree[v] >= 3:
                # Record midpoint BEFORE contraction so the surviving node gets
                # geometrically correct x/y (contracted_nodes keeps u's attrs
                # unchanged, leaving them at u's original position, not the
                # merged location).
                mx = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2.0
                my = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2.0
                G = nx.contracted_nodes(G, u, v, self_loops=False)
                # Remove the contraction bookkeeping dict networkx adds.
                if "contraction" in G.nodes[u]:
                    del G.nodes[u]["contraction"]
                # Update world coords to the midpoint; remove stale pixel
                # coords (row/col belonged to u's original position and are
                # now meaningless for the merged node).
                G.nodes[u]["x"] = mx
                G.nodes[u]["y"] = my
                G.nodes[u].pop("row", None)
                G.nodes[u].pop("col", None)
                changed = True
                break
    return G


def _oriented_geom(geom, node_from, node_to):
    """Return geom ordered so it starts near node_from's (x, y)."""
    if not geom:
        return geom
    fx, fy = node_from["x"], node_from["y"]
    d_start = math.hypot(geom[0][0] - fx, geom[0][1] - fy)
    d_end = math.hypot(geom[-1][0] - fx, geom[-1][1] - fy)
    return geom if d_start <= d_end else list(reversed(geom))


def skeleton_to_graph(
    prob_mask: np.ndarray,
    transform=None,
    source_gsd_m: float = 1.0,
    threshold: float = 0.5,
    threshold_low: Optional[float] = None,
    min_object_px: int = 12,
    min_branch_m: float = 8.0,
    node_merge_m: Optional[float] = None,
    closing_iterations: int = 1,
    prefer_skimage: bool = True,
    dp_epsilon_m: float = 1.0,
    min_edge_mean_prob: float = 0.0,
) -> nx.Graph:
    """Vectorize a road-probability mask into a routable networkx graph.

    Parameters
    ----------
    prob_mask : (H, W) float array in [0, 1] -- PathMamba's sigmoid road output.
    transform : object with world_to_pixel(x, y)->(row, col) (SimpleAffine or a
                rasterio Affine adapter). Node x,y are emitted in this frame.
                None => pixel frame (topology testing only; see module docstring).
    source_gsd_m : meters/pixel; used only for length scaling when transform is
                None. When a transform is present, lengths come from world coords
                directly and this is ignored.
    threshold : HIGH probability cutoff for foreground (also the single-threshold
                cutoff when threshold_low is None).
    threshold_low : LOW cutoff for hysteresis thresholding. When set (< threshold),
                pixels >= threshold_low are also kept if they are 8-connected to a
                >= threshold pixel. Recommended for Tversky-trained road models
                whose real-road probabilities dip in canopy/shadow (try 0.3/0.5).
                None => single-threshold behavior (backward compatible).
    min_object_px : connected road blobs smaller than this (pixels) are dropped
                as segmentation noise before thinning.
    min_branch_m : spur branches shorter than this (meters) are pruned as thinning
                artifacts. In pixel-frame mode this is in pixels.
    closing_iterations : binary closing before thinning to bridge 1-2px threshold
                pinholes (default 1). Small closings help real inference output
                without introducing cross-road bridges (those are wider than 1 px).
                Set to 0 to disable for synthetic-mask testing.
    dp_epsilon_m : Douglas-Peucker simplification tolerance on edge geometries
                (meters). Reduces per-edge vertex count from ~1 per pixel to a
                handful, and gives healing's local-tangent helper a cleaner
                signal. Set to 0 to disable.
    min_edge_mean_prob : if > 0, drop edges whose mean prob along their traced
                polyline is below this. A cheap topology cleaner (see
                filter_edges_by_mean_prob). Try `threshold + 0.1`.

    Returns
    -------
    networkx.Graph with node attrs {x, y} and edge attrs
    {length, bridge, layer, synthetic, geometry, n_px}. Node ids are 0..N-1.
    """
    prob = np.asarray(prob_mask)
    if prob.ndim != 2:
        raise ValueError(f"prob_mask must be 2D (H, W); got shape {prob.shape}")

    if threshold_low is not None and threshold_low < threshold:
        binary = hysteresis_threshold(prob, low=threshold_low, high=threshold)
    else:
        binary = prob >= threshold

    if closing_iterations > 0 and _HAVE_NDIMAGE:
        binary = _ndi.binary_closing(binary, iterations=closing_iterations)

    if min_object_px > 0 and _HAVE_NDIMAGE and binary.any():
        labels, n = _ndi.label(binary, structure=np.ones((3, 3)))
        if n > 0:
            sizes = _ndi.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
            keep = {i + 1 for i, s in enumerate(sizes) if s >= min_object_px}
            binary = np.isin(labels, list(keep)) if keep else np.zeros_like(binary)

    skel = _skeletonize(binary, prefer_skimage=prefer_skimage)

    p2w = _build_pixel_to_world(transform)
    node_pixels, edges = _trace_edges(skel)

    G = nx.Graph()
    px_to_id: dict[tuple[int, int], int] = {}

    def node_id(px):
        if px not in px_to_id:
            nid = len(px_to_id)
            px_to_id[px] = nid
            x, y = p2w(px[0], px[1])
            G.add_node(nid, x=x, y=y, row=int(px[0]), col=int(px[1]))
        return px_to_id[px]

    for a_px, b_px, path in edges:
        ua, ub = node_id(a_px), node_id(b_px)
        if ua == ub and len(path) < 3:
            continue  # degenerate
        length, world = _polyline_length_world(path, p2w)
        if transform is None:
            length *= source_gsd_m  # pixel frame -> meters (best effort)
        if G.has_edge(ua, ub):
            # keep the shorter of parallel traces (rare)
            if G.edges[ua, ub].get("length", math.inf) <= length:
                continue
        G.add_edge(ua, ub, length=length, bridge=False, layer=0,
                   synthetic=False, geometry=world, n_px=len(path))

    # Effective world pixel size, for artifact-merge default (world dist between
    # two horizontally adjacent pixels).
    x00, y00 = p2w(0.0, 0.0)
    x01, y01 = p2w(0.0, 1.0)
    px_size_m = math.hypot(x01 - x00, y01 - y00) or (source_gsd_m if transform is None else 1.0)
    merge_thresh = node_merge_m if node_merge_m is not None else 2.5 * px_size_m

    G = _prune_and_collapse(G, min_branch_m=min_branch_m)
    G = _merge_short_junction_edges(G, thresh_m=merge_thresh)
    G = _prune_and_collapse(G, min_branch_m=min_branch_m)  # merge may create deg-2

    # Optional cleanup: drop edges whose mean prob along their traced polyline
    # is too low (catches traced edges that skim just above the threshold).
    if min_edge_mean_prob > 0.0:
        G = filter_edges_by_mean_prob(G, prob, transform, min_mean=min_edge_mean_prob)

    # Douglas-Peucker simplify edge polylines (reduces vertex count ~10-50x,
    # improves local-tangent noise for healing, cheaper visualization).
    if dp_epsilon_m > 0.0:
        G = simplify_edge_geometries(G, epsilon_m=dp_epsilon_m)

    # Relabel to contiguous ints after pruning removed nodes.
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    return G


def graph_from_prediction(prediction, **kwargs) -> nx.Graph:
    """Convenience entry point matching pipeline.py's intended call site.

        broken_graph = graph_from_prediction(prediction)

    Pulls prob_mask, transform, and source_gsd_m off the RoadPrediction so the
    caller doesn't unpack them by hand.
    """
    return skeleton_to_graph(
        prediction.prob_mask,
        transform=prediction.transform,
        source_gsd_m=getattr(prediction, "source_gsd_m", 1.0),
        **kwargs,
    )
