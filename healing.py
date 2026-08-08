"""
healing.py
==========
Stage 6 of the pipeline (per the architecture diagram): converts a broken graph
into a single routable graph by adding synthetic bridge edges, but ONLY where
a soft healing score passes. Short gaps are accepted directly, medium gaps
are allowed when they are well aligned, and longer gaps rely on a distance/
angle/confidence score rather than a brittle hard gate:

    distance <= 15m        -> pass immediately
    distance <= 30m and alignment is good -> pass immediately
    otherwise: score >= 0.35

This module's only real-world dependency is a `RoadPrediction` (model_io.py) --
whether that came from `synthetic_occlusion.rasterize_graph_to_prediction()`
(today) or `model_io.from_pathmamba_output()` (after training), this code does
not change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from .geo_utils import sample_along_line, sample_along_corridor, sample_along_corridor_robust
from .model_io import RoadPrediction

OCCLUSION_MIN = 0.10
ANGLE_MAX_DEG = 35.0
GAP_MAX_M = 50.0
SHORT_GAP_M = 15.0
MEDIUM_GAP_M = 30.0
SOFT_SCORE_THRESHOLD = 0.40
TRAJECTORY_LOOKAHEAD_M = 40.0
TRAJECTORY_MEET_TOL_M = 15.0
CONFIDENCE_WEIGHT = 0.15
TOPOLOGY_WEIGHT = 0.85


@dataclass
class UnionFind:
    """Disjoint-Set for Kruskal-style MST reconnection."""

    parent: dict = field(default_factory=dict)
    rank: dict = field(default_factory=dict)

    def make_set(self, x) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        self.make_set(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


@dataclass
class HealingCandidate:
    u: int
    v: int
    distance_m: float
    angle_deviation_deg: float
    occlusion_confidence: float
    passed: bool
    reject_reason: Optional[str] = None


@dataclass
class HealingReport:
    healed_graph: nx.Graph
    candidates_considered: list[HealingCandidate]
    bridges_added: list[HealingCandidate]
    rejected: list[HealingCandidate]
    iterations: int = 0

    def summary(self) -> dict:
        reasons: dict[str, int] = {}
        for c in self.rejected:
            reasons[c.reject_reason] = reasons.get(c.reject_reason, 0) + 1
        return {
            "iterations": self.iterations,
            "candidates_considered": len(self.candidates_considered),
            "bridges_added": len(self.bridges_added),
            "rejected": len(self.rejected),
            "rejected_by_reason": reasons,
        }


def _local_headings_deg(G: nx.Graph, node, tangent_length_m: float = 15.0) -> list[float]:
    """Bearing (0-180, undirected) of EACH surviving edge at a node individually
    -- NOT averaged.

    Uses each edge's traced `geometry` polyline to compute a LOCAL TANGENT over
    the first `tangent_length_m` of the edge starting from `node`, instead of
    the chord bearing to the far collapsed neighbor. On curved roads, chord and
    tangent can differ by 20-40 degrees -- a real through-road can then fail
    the angle gate spuriously. When geometry is missing (edge came from
    healing itself, or DP was disabled to zero vertices), falls back to the
    chord bearing so behavior on synthetic graphs is unchanged.

    Averaging was tried first and is wrong: a junction that kept 3 of its 4
    original directions after occlusion removed one has surviving bearings
    pointing three different ways, and their circular mean cancels out into a
    near-meaningless heading. What actually matters for the angle gate is
    whether ANY surviving direction is roughly collinear with the candidate
    bridge -- so we return every heading separately for the caller's `min` reducer.
    """
    from .skeletonization import local_tangents_deg  # local to avoid import cycle
    return local_tangents_deg(G, node, length_m=tangent_length_m)


def _trajectory_point(G: nx.Graph, node, lookahead_m: float = TRAJECTORY_LOOKAHEAD_M) -> tuple[float, float]:
    """Project a stub's road geometry forward to estimate where it would continue."""
    if G.degree[node] == 0:
        return G.nodes[node]["x"], G.nodes[node]["y"]

    x0, y0 = G.nodes[node]["x"], G.nodes[node]["y"]
    best_geom = None
    best_dist = None
    for nbr in G.neighbors(node):
        data = G.edges[node, nbr]
        geom = data.get("geometry") or []
        if len(geom) < 2:
            continue
        # orient so the polyline starts near this node
        dx_start = math.hypot(geom[0][0] - x0, geom[0][1] - y0)
        dx_end = math.hypot(geom[-1][0] - x0, geom[-1][1] - y0)
        if dx_end < dx_start:
            geom = list(reversed(geom))
        # choose the longest/most relevant edge geometry for this stub
        dist = math.hypot(geom[-1][0] - geom[0][0], geom[-1][1] - geom[0][1])
        if best_dist is None or dist > best_dist:
            best_geom = geom
            best_dist = dist

    if not best_geom:
        return x0, y0

    geom = best_geom
    dx_start = math.hypot(geom[0][0] - x0, geom[0][1] - y0)
    dx_end = math.hypot(geom[-1][0] - x0, geom[-1][1] - y0)
    if dx_end < dx_start:
        geom = list(reversed(geom))

    px, py = geom[0]
    acc = 0.0
    last_point = (px, py)
    for qx, qy in geom[1:]:
        seg = math.hypot(qx - px, qy - py)
        if acc + seg >= lookahead_m:
            frac = (lookahead_m - acc) / seg if seg > 0 else 1.0
            return (
                px + (qx - px) * frac,
                py + (qy - py) * frac,
            )
        acc += seg
        px, py = qx, qy
        last_point = (px, py)

    return last_point


def _trajectory_alignment(G: nx.Graph, u, v, lookahead_m: float = TRAJECTORY_LOOKAHEAD_M) -> float:
    """Higher is better when both stubs would meet if extrapolated forward."""
    pu = _trajectory_point(G, u, lookahead_m=lookahead_m)
    pv = _trajectory_point(G, v, lookahead_m=lookahead_m)
    dist = math.hypot(pu[0] - pv[0], pu[1] - pv[1])
    if dist <= TRAJECTORY_MEET_TOL_M:
        return 1.0
    return max(0.0, 1.0 - dist / max(lookahead_m, 1e-6))


def _is_stub_node(G: nx.Graph, node) -> bool:
    """Treat degree-4 junctions and dead-end-adjacent nodes as healing candidates."""
    if G.degree[node] <= 4:
        return True
    return any(G.degree(neighbor) <= 1 for neighbor in G.neighbors(node))


def find_healing_candidates(
    G: nx.Graph,
    prediction: RoadPrediction,
    max_search_radius_m: float = GAP_MAX_M,
    occlusion_min: float = OCCLUSION_MIN,
    angle_max_deg: float = ANGLE_MAX_DEG,
    gap_max_m: float = GAP_MAX_M,
    n_samples_per_line: int = 8,
) -> list[HealingCandidate]:
    """Finds every cross-component node pair within range and scores them
    using the healing heuristic. Only considers stub-like nodes (degree <= 4,
    plus dead-end-adjacent nodes) to keep the search small while still catching
    the short, visually obvious gaps that typical occlusion breaks create.
    """
    components = list(nx.connected_components(G))
    comp_of = {}
    for i, comp in enumerate(components):
        for n in comp:
            comp_of[n] = i

    stub_nodes = [n for n in G.nodes() if _is_stub_node(G, n)]
    if len(stub_nodes) < 2:
        return []

    coords = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in stub_nodes])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=max_search_radius_m)

    candidates: list[HealingCandidate] = []
    seen = set()
    for i, j in pairs:
        u, v = stub_nodes[i], stub_nodes[j]
        if comp_of[u] == comp_of[v]:
            continue  # already connected -- not a gap, a redundant loop
        key = tuple(sorted((u, v)))
        if key in seen:
            continue
        seen.add(key)

        xu, yu = G.nodes[u]["x"], G.nodes[u]["y"]
        xv, yv = G.nodes[v]["x"], G.nodes[v]["y"]
        distance = math.hypot(xv - xu, yv - yu)

        trajectory_alignment = _trajectory_alignment(G, u, v)
        angle_deviation = 180.0 * (1.0 - trajectory_alignment)

        # Robust corridor sampling: MEDIAN across the perpendicular strip
        # (kills single-pixel outliers like bright building corners), then
        # FRACTION-ABOVE-THRESHOLD along the centerline (requires sustained
        # evidence, not a single lucky spike). This makes occlusion_confidence
        # mean "what fraction of the bridge path has real support" instead of
        # "how much bright-pixel noise did we happen to catch".
        #
        # Range is still [0, 1] so the heal_occlusion_min slider still makes
        # sense; a default around 0.3-0.5 works well under the new metric
        # (a genuine occluded road tends to score 0.6-1.0, an airborne bridge
        # 0.0-0.2, a partial candidate ~0.3-0.4).
        _CENTERLINE_SUPPORT_THRESHOLD = 0.3   # per-point support cutoff
        conf_samples = sample_along_corridor_robust(
            prediction.confidence_mask, prediction.transform,
            xu, yu, xv, yv,
            n_samples=n_samples_per_line,
            corridor_half_width_m=2.0,
            n_cross=3,
        )
        occlusion_confidence = float(
            (conf_samples >= _CENTERLINE_SUPPORT_THRESHOLD).mean()
        )

        reason = None
        if distance > gap_max_m:
            reason = "gap_distance"
        elif distance <= SHORT_GAP_M:
            reason = None
        elif distance <= MEDIUM_GAP_M and trajectory_alignment >= 0.5:
            reason = None
        else:
            distance_score = max(0.0, 1.0 - distance / gap_max_m)
            angle_score = trajectory_alignment
            occlusion_score = occlusion_confidence
            bridge_score = (TOPOLOGY_WEIGHT * (0.5 * distance_score + 0.5 * angle_score) +
                            CONFIDENCE_WEIGHT * occlusion_score)
            if bridge_score >= SOFT_SCORE_THRESHOLD:
                reason = None
            else:
                reason = "soft_score"

        candidates.append(
            HealingCandidate(
                u=u,
                v=v,
                distance_m=distance,
                angle_deviation_deg=angle_deviation,
                occlusion_confidence=occlusion_confidence,
                passed=(reason is None),
                reject_reason=reason,
            )
        )

    return candidates


def heal(
    G: nx.Graph,
    prediction: RoadPrediction,
    max_search_radius_m: float = GAP_MAX_M,
    occlusion_min: float = OCCLUSION_MIN,
    angle_max_deg: float = ANGLE_MAX_DEG,
    gap_max_m: float = GAP_MAX_M,
    max_iterations: int = 2,
) -> HealingReport:
    """Runs the 3-gate check, then Kruskal's algorithm (via Union-Find) over the
    surviving candidates for up to `max_iterations` passes. Each iteration re-
    evaluates the current graph so new gap bridges can emerge after earlier
    connections have been added.
    """
    healed = G.copy()
    all_candidates: list[HealingCandidate] = []
    all_bridges_added: list[HealingCandidate] = []
    all_rejected: list[HealingCandidate] = []
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        candidates = find_healing_candidates(
            healed, prediction, max_search_radius_m, occlusion_min, angle_max_deg, gap_max_m
        )
        passing = sorted([c for c in candidates if c.passed], key=lambda c: c.distance_m)
        rejected = [c for c in candidates if not c.passed]

        all_candidates.extend(candidates)
        all_rejected.extend(rejected)

        uf = UnionFind()
        for n in healed.nodes():
            uf.make_set(n)
        for comp in nx.connected_components(healed):
            rep = next(iter(comp))
            for n in comp:
                uf.union(rep, n)

        iteration_bridges: list[HealingCandidate] = []
        for c in passing:
            if uf.union(c.u, c.v):
                healed.add_edge(
                    c.u,
                    c.v,
                    length=c.distance_m,
                    bridge=False,
                    layer=0,
                    synthetic=True,
                    healing_confidence=c.occlusion_confidence,
                )
                iteration_bridges.append(c)
            else:
                all_rejected.append(
                    HealingCandidate(
                        c.u, c.v, c.distance_m, c.angle_deviation_deg, c.occlusion_confidence,
                        passed=False, reject_reason="redundant_after_earlier_bridge",
                    )
                )

        all_bridges_added.extend(iteration_bridges)
        if not iteration_bridges:
            break

    return HealingReport(
        healed_graph=healed,
        candidates_considered=all_candidates,
        bridges_added=all_bridges_added,
        rejected=all_rejected,
        iterations=iterations,
    )


def score_against_ground_truth(report: HealingReport, gap_records: list) -> dict:
    """Synthetic-mode-only sanity check: since we know exactly which edges we
    cut (gap_records from synthetic_occlusion.py), compare against what healing
    actually reconnected. Never available on real Cartosat-3 data -- this is a
    tool for validating the healing LOGIC now, not a production metric.
    """
    true_pairs = {tuple(sorted((g.u, g.v))) for g in gap_records}
    healed_pairs = {tuple(sorted((c.u, c.v))) for c in report.bridges_added}
    true_positives = true_pairs & healed_pairs
    false_positives = healed_pairs - true_pairs
    false_negatives = true_pairs - healed_pairs
    precision = len(true_positives) / len(healed_pairs) if healed_pairs else float("nan")
    recall = len(true_positives) / len(true_pairs) if true_pairs else float("nan")
    return {
        "true_gaps": len(true_pairs),
        "bridges_added": len(healed_pairs),
        "true_positives": len(true_positives),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "precision": precision,
        "recall": recall,
    }
