"""
synthetic_occlusion.py
=======================
Everything needed to test Healing / Centrality / Simulation TODAY, with no trained
model and no live OSM pull (Overpass isn't reachable from this sandbox's network
allowlist -- this generates a stand-in "OSM-like" graph instead, with the same
edge attributes a real `osmnx`/Overpass extract would carry: length, layer,
bridge, population-per-node).

Three exports matter:
    build_synthetic_city_graph()  -- a "ground truth" road network
    inject_occlusion()            -- cuts edges to fake occlusion gaps, with recorded
                                      ground truth so healing precision/recall is scorable
    rasterize_graph_to_prediction() -- turns the broken graph into a RoadPrediction
                                      object with the exact (prob_mask, confidence_mask)
                                      shape PathMamba will eventually produce, so nothing
                                      downstream needs to change when real inference lands
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from .geo_utils import SimpleAffine
from .model_io import RoadPrediction


@dataclass
class GapRecord:
    """Ground truth for one injected occlusion gap -- used only to score healing,
    never available for real Cartosat-3 data (there we don't have ground truth)."""

    u: int
    v: int
    x0: float
    y0: float
    x1: float
    y1: float
    length_m: float
    true_bearing_deg: float


def build_synthetic_city_graph(
    rows: int = 6,
    cols: int = 6,
    cell_size_m: float = 90.0,
    diagonal_highway: bool = True,
    seed: Optional[int] = 42,
) -> nx.Graph:
    """Grid-like road network standing in for an OSM extract.

    Node attrs: x, y (meters), population (int, stand-in for a WorldPop join)
    Edge attrs: length (m), layer (int, 0 = surface), bridge (bool)

    A diagonal "highway" chord is added across the grid so centrality.py has an
    obvious high-betweenness route to sanity-check against (its nodes should
    clearly outrank the grid's interior nodes).
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()

    for r in range(rows):
        for c in range(cols):
            node_id = r * cols + c
            x, y = c * cell_size_m, r * cell_size_m
            # Denser population near the grid center, sparser at the edges --
            # gives simulation.py's population-impact metric something to vary on.
            center_r, center_c = (rows - 1) / 2, (cols - 1) / 2
            dist_from_center = math.hypot(r - center_r, c - center_c)
            population = int(max(50, 800 - dist_from_center * 120 + rng.normal(0, 40)))
            G.add_node(node_id, x=x, y=y, population=population)

    def add_edge(a: int, b: int, bridge: bool = False, layer: int = 0) -> None:
        xa, ya = G.nodes[a]["x"], G.nodes[a]["y"]
        xb, yb = G.nodes[b]["x"], G.nodes[b]["y"]
        length = math.hypot(xb - xa, yb - ya)
        G.add_edge(a, b, length=length, bridge=bridge, layer=layer)

    for r in range(rows):
        for c in range(cols):
            node_id = r * cols + c
            if c + 1 < cols:
                add_edge(node_id, node_id + 1)
            if r + 1 < rows:
                add_edge(node_id, node_id + cols)

    if diagonal_highway:
        # A grade-separated chord from one corner to the other -- tagged as a
        # bridge/elevated route so zlevel.py has something real to classify.
        diag_nodes = [r * cols + r for r in range(min(rows, cols))]
        for a, b in zip(diag_nodes[:-1], diag_nodes[1:]):
            if not G.has_edge(a, b):
                add_edge(a, b, bridge=True, layer=1)

    return G


def inject_occlusion(
    G: nx.Graph, cluster_size: int = 4, seed: Optional[int] = 7
) -> tuple[nx.Graph, list[GapRecord]]:
    """Fakes an occlusion patch (a canopy/shadow/cloud blob covering a local
    NEIGHBORHOOD, not a single road segment) by BFS-growing a random cluster of
    ~`cluster_size` nodes and removing every edge that crosses the cluster
    boundary. This is deliberately NOT "remove N random edges": a grid-like
    road network is highly redundant (most edges lie on a cycle, so a single
    random cut has an alternate route and doesn't fragment anything -- this
    was caught by actually running the demo and checking connected-component
    counts, see the surrounding conversation). Cutting an entire local boundary
    guarantees the cluster is genuinely severed from the rest of the network,
    which both matches what real canopy occlusion does (it blots out an area,
    crossing several roads at once) and gives healing.py real fragmentation to
    reconnect.

    Endpoints are kept in the graph (only boundary edges are removed) --
    exactly the "broken mask, live nodes" scenario healing.py repairs.
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    center = nodes[int(rng.integers(len(nodes)))]

    cluster = {center}
    frontier = [center]
    while len(cluster) < cluster_size and frontier:
        next_frontier = []
        for n in frontier:
            for nbr in G.neighbors(n):
                if nbr not in cluster:
                    cluster.add(nbr)
                    next_frontier.append(nbr)
                if len(cluster) >= cluster_size:
                    break
            if len(cluster) >= cluster_size:
                break
        frontier = next_frontier

    boundary_edges = [(u, v) for u, v in G.edges() if (u in cluster) != (v in cluster)]
    if not boundary_edges:
        raise RuntimeError(
            f"Cluster of size {len(cluster)} has no boundary edges -- it's either "
            "the whole graph or fully enclosed with no external edges. Try a "
            "smaller cluster_size or a bigger graph."
        )

    broken = G.copy()
    gap_records: list[GapRecord] = []
    for u, v in boundary_edges:
        xu, yu = G.nodes[u]["x"], G.nodes[u]["y"]
        xv, yv = G.nodes[v]["x"], G.nodes[v]["y"]
        length = G.edges[u, v]["length"]
        bearing = math.degrees(math.atan2(yv - yu, xv - xu)) % 180.0
        broken.remove_edge(u, v)
        gap_records.append(
            GapRecord(u=u, v=v, x0=xu, y0=yu, x1=xv, y1=yv, length_m=length, true_bearing_deg=bearing)
        )

    return broken, gap_records


def rasterize_graph_to_prediction(
    G_full: nx.Graph,
    G_broken: nx.Graph,
    gap_records: list[GapRecord],
    pixel_size_m: float = 2.0,
    road_width_px: int = 2,
    background_prob: float = 0.03,
    road_prob: float = 0.92,
    gap_confidence: float = 0.55,
    noise_std: float = 0.05,
    seed: Optional[int] = 11,
) -> RoadPrediction:
    """Fakes a PathMamba output tile for the broken graph, in the exact
    (prob_mask, confidence_mask) shape/contract the real model produces.

    Design intent, matching what a trained model *should* do:
      - Surviving edges  -> high prob_mask, low-ish confidence_mask (confidence
        head is mostly interesting where the model is uncertain, i.e. at gaps)
      - Gap locations     -> LOW prob_mask (there's genuinely no visible road
        pixel there -- that's the definition of occlusion) but an ELEVATED
        confidence_mask along the straight line between the two stubs (the
        model's belief that a road plausibly continues there)
      - Everything else   -> background noise

    This lets healing.py's occlusion-confidence gate be exercised meaningfully
    even though no real inference has run yet.
    """
    rng = np.random.default_rng(seed)

    xs = [d["x"] for _, d in G_full.nodes(data=True)]
    ys = [d["y"] for _, d in G_full.nodes(data=True)]
    pad_m = pixel_size_m * 20
    min_x, max_x = min(xs) - pad_m, max(xs) + pad_m
    min_y, max_y = min(ys) - pad_m, max(ys) + pad_m

    width_px = int(math.ceil((max_x - min_x) / pixel_size_m))
    height_px = int(math.ceil((max_y - min_y) / pixel_size_m))

    prob = np.clip(rng.normal(background_prob, noise_std, size=(height_px, width_px)), 0, 1).astype(
        np.float32
    )
    conf = np.clip(rng.normal(0.05, 0.03, size=(height_px, width_px)), 0, 1).astype(np.float32)

    transform = SimpleAffine(origin_x=min_x, origin_y=min_y, pixel_size=pixel_size_m, height_px=height_px)

    def paint_line(target: np.ndarray, x0, y0, x1, y1, value: float, width_px_local: int) -> None:
        r0, c0 = transform.world_to_pixel(x0, y0)
        r1, c1 = transform.world_to_pixel(x1, y1)
        n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
        rr = np.linspace(r0, r1, n)
        cc = np.linspace(c0, c1, n)
        for r, c in zip(rr, cc):
            ri, ci = int(round(r)), int(round(c))
            for dr in range(-width_px_local, width_px_local + 1):
                for dc in range(-width_px_local, width_px_local + 1):
                    rri, cci = ri + dr, ci + dc
                    if 0 <= rri < target.shape[0] and 0 <= cci < target.shape[1]:
                        target[rri, cci] = max(target[rri, cci], value)

    # Paint surviving edges as high road-probability.
    for u, v, data in G_broken.edges(data=True):
        xu, yu = G_full.nodes[u]["x"], G_full.nodes[u]["y"]
        xv, yv = G_full.nodes[v]["x"], G_full.nodes[v]["y"]
        paint_line(prob, xu, yu, xv, yv, road_prob, road_width_px)
        paint_line(conf, xu, yu, xv, yv, 0.15, road_width_px)  # low conf where prob is already high

    # Paint gap locations: low prob (occluded), elevated confidence.
    for gap in gap_records:
        paint_line(conf, gap.x0, gap.y0, gap.x1, gap.y1, gap_confidence, road_width_px + 1)
        # A faint, broken prob trace -- occlusion rarely erases a road to pure zero,
        # it dims it below the segmentation threshold.
        paint_line(prob, gap.x0, gap.y0, gap.x1, gap.y1, background_prob + 0.1, road_width_px)

    return RoadPrediction(
        prob_mask=prob,
        confidence_mask=conf,
        transform=transform,
        crs=None,
        source_gsd_m=pixel_size_m,
        is_synthetic=True,
    )
