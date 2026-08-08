"""
evaluation.py
=============
Graph-level segmentation metrics the deck names but which weren't in the code:

  * APLS (Average Path Length Similarity) -- the SpaceNet-standard metric for
    road-extraction evaluation. Range [0, 1], higher is better.
  * breaks / km -- number of topological breaks per km of true road length.
    Directly measures how often the predicted graph fragments the true road.
  * connectivity ratio -- fraction of true node-pairs that remain reachable in
    the predicted graph.

Only IoU (a pixel metric) existed before -- pixel IoU can be high on a graph
that is topologically shattered (every road drawn correctly but disconnected
into hundreds of pieces). APLS and its friends catch that.

INPUTS
------
Both graphs are networkx.Graph in the same node-attr schema the pipeline uses:
  nodes : x, y  (world coordinates, meters)
  edges : length (meters)
Node ids do NOT need to match across graphs. We do proximity matching:
snap each true node to the nearest predicted node within a tolerance, and use
those matches as endpoints for shortest-path comparisons.

APLS -- WHAT IT MEASURES
------------------------
For every pair of matched nodes (u_true, v_true) with proximity matches
(u_pred, v_pred): compare the true shortest-path length between u_true, v_true
to the predicted shortest-path length between u_pred, v_pred. If they differ
by more than a factor, penalize proportionally. Averaged over all pairs.

If a true node has no proximity match in the prediction, that pair contributes
the maximum penalty. This is the standard SpaceNet-3 formulation
(Van Etten et al., "SpaceNet: A Remote Sensing Dataset and Challenge Series"),
in the symmetric form (avg of APLS(gt, pred) and APLS(pred, gt)).

REFERENCE
---------
Van Etten, Lindenbaum, Bacastow 2019: "SpaceNet: A Remote Sensing Dataset and
Challenge Series" arXiv:1807.01232 -- APLS formula in Section 6.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _node_positions(G: nx.Graph) -> tuple[list, np.ndarray]:
    """Return (node_ids_list, positions_array (N, 2)) in matching order."""
    ids = list(G.nodes())
    if not ids:
        return ids, np.zeros((0, 2), dtype=np.float64)
    pos = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in ids], dtype=np.float64)
    return ids, pos


def _snap_nodes(G_from: nx.Graph, G_to: nx.Graph, tol_m: float) -> dict:
    """For each node in G_from, find the nearest node in G_to within tol_m.
    Returns {node_in_from: node_in_to} for matches only (unmatched nodes are
    absent from the dict)."""
    ids_from, pos_from = _node_positions(G_from)
    ids_to, pos_to = _node_positions(G_to)
    if not ids_from or not ids_to:
        return {}
    tree = cKDTree(pos_to)
    dists, idxs = tree.query(pos_from, k=1)
    return {ids_from[i]: ids_to[int(idxs[i])] for i in range(len(ids_from)) if dists[i] <= tol_m}


def _shortest_length(G: nx.Graph, u, v) -> Optional[float]:
    """Length of the shortest weighted path (weight='length'). None if no path
    exists or an endpoint is missing."""
    if u == v:
        return 0.0
    if u not in G or v not in G:
        return None
    try:
        return nx.shortest_path_length(G, u, v, weight="length")
    except nx.NetworkXNoPath:
        return None


# ---------------------------------------------------------------------------
# APLS
# ---------------------------------------------------------------------------

@dataclass
class APLSResult:
    apls: float          # symmetric, in [0, 1]
    apls_gt_to_pred: float
    apls_pred_to_gt: float
    n_pairs_gt: int
    n_pairs_pred: int
    n_matched_gt_nodes: int
    n_matched_pred_nodes: int


def _apls_directional(G_src: nx.Graph, G_tgt: nx.Graph, node_map: dict,
                      max_pairs: int, rng: np.random.Generator) -> tuple[float, int]:
    """One direction of APLS: how well does G_tgt reproduce G_src's shortest-
    path lengths? Averaged over up to max_pairs sampled node pairs from G_src."""
    src_nodes = [n for n in G_src.nodes() if G_src.degree[n] > 0]
    if len(src_nodes) < 2:
        return 1.0, 0  # degenerate: nothing to compare

    # Sample pairs. For small graphs, use all pairs.
    pairs = []
    max_all = len(src_nodes) * (len(src_nodes) - 1) // 2
    if max_all <= max_pairs:
        for i in range(len(src_nodes)):
            for j in range(i + 1, len(src_nodes)):
                pairs.append((src_nodes[i], src_nodes[j]))
    else:
        seen = set()
        while len(pairs) < max_pairs:
            i, j = rng.integers(0, len(src_nodes), size=2)
            if i == j:
                continue
            key = (int(min(i, j)), int(max(i, j)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((src_nodes[i], src_nodes[j]))

    if not pairs:
        return 1.0, 0

    total_diff = 0.0
    scored = 0
    for u, v in pairs:
        L_src = _shortest_length(G_src, u, v)
        if L_src is None or L_src == 0:
            continue  # source has no path -> pair excluded from denominator too
        u_m = node_map.get(u)
        v_m = node_map.get(v)
        scored += 1
        if u_m is None or v_m is None:
            total_diff += 1.0  # unmatched endpoint -> maximum penalty
            continue
        L_tgt = _shortest_length(G_tgt, u_m, v_m)
        if L_tgt is None:
            total_diff += 1.0  # unreachable in target -> max penalty
            continue
        # standard APLS per-pair penalty: min(1, |L_src - L_tgt| / L_src)
        total_diff += min(1.0, abs(L_src - L_tgt) / L_src)

    if scored == 0:
        return 1.0, 0
    apls = 1.0 - total_diff / scored
    return float(apls), scored


def apls(G_true: nx.Graph, G_pred: nx.Graph,
         snap_tolerance_m: float = 15.0,
         max_pairs: int = 500,
         seed: int = 0) -> APLSResult:
    """Symmetric APLS between a ground-truth graph and a predicted graph.

    Parameters
    ----------
    snap_tolerance_m : how close a true node must be to a predicted node
                       to count as "the same intersection". SpaceNet default ~4m
                       for 30cm imagery; on 0.28m Cartosat-3 use ~15m to allow
                       for skeletonization drift at junctions.
    max_pairs : cap on number of random node pairs sampled per direction.
                For dense city graphs full O(N^2) is slow; sampling to a few
                hundred is standard.
    seed : RNG seed for reproducible sampling.
    """
    rng = np.random.default_rng(seed)
    gt_to_pred = _snap_nodes(G_true, G_pred, snap_tolerance_m)
    pred_to_gt = _snap_nodes(G_pred, G_true, snap_tolerance_m)

    a1, n1 = _apls_directional(G_true, G_pred, gt_to_pred, max_pairs, rng)
    a2, n2 = _apls_directional(G_pred, G_true, pred_to_gt, max_pairs, rng)

    # Symmetric mean, but skip a direction that had ZERO valid pairs -- that's
    # a degenerate side (e.g. pred is empty so pred->gt has no pairs), and
    # counting it as a free 1.0 would falsely halve the penalty on the other
    # side. This matches SpaceNet's own APLS behavior.
    scores, weights = [], []
    if n1 > 0:
        scores.append(a1); weights.append(n1)
    if n2 > 0:
        scores.append(a2); weights.append(n2)
    if scores:
        apls_score = float(np.average(scores, weights=weights))
    else:
        apls_score = 1.0  # both graphs trivially empty

    return APLSResult(
        apls=apls_score,
        apls_gt_to_pred=a1,
        apls_pred_to_gt=a2,
        n_pairs_gt=n1,
        n_pairs_pred=n2,
        n_matched_gt_nodes=len(gt_to_pred),
        n_matched_pred_nodes=len(pred_to_gt),
    )


# ---------------------------------------------------------------------------
# breaks / km
# ---------------------------------------------------------------------------

@dataclass
class BreaksResult:
    breaks: int
    true_length_km: float
    breaks_per_km: float


def breaks_per_km(G_true: nx.Graph, G_pred: nx.Graph,
                  snap_tolerance_m: float = 15.0) -> BreaksResult:
    """Count topological breaks: for every edge (u, v) in the TRUE graph,
    check whether the corresponding matched predicted endpoints are in the
    SAME connected component of G_pred. If not, that's one break. Divide by
    total true road length in km.

    A break is a place where the true graph would let you cross an edge but the
    predicted graph does not connect the two sides at all (any path). This is
    stricter than "did the exact edge get predicted" -- it forgives topology
    that reroutes around, and only penalizes actual disconnection."""
    gt_to_pred = _snap_nodes(G_true, G_pred, snap_tolerance_m)
    # component id per predicted node
    pred_comp = {}
    for i, comp in enumerate(nx.connected_components(G_pred)):
        for n in comp:
            pred_comp[n] = i

    breaks = 0
    total_len = 0.0
    for u, v, data in G_true.edges(data=True):
        L = float(data.get("length", 0.0))
        total_len += L
        um = gt_to_pred.get(u)
        vm = gt_to_pred.get(v)
        if um is None or vm is None:
            breaks += 1
            continue
        if pred_comp.get(um) != pred_comp.get(vm):
            breaks += 1

    total_len_km = total_len / 1000.0
    bpk = breaks / total_len_km if total_len_km > 0 else 0.0
    return BreaksResult(breaks=breaks, true_length_km=total_len_km, breaks_per_km=bpk)


# ---------------------------------------------------------------------------
# connectivity ratio
# ---------------------------------------------------------------------------

@dataclass
class ConnectivityResult:
    ratio: float
    n_pairs_evaluated: int
    n_reachable_in_pred: int


def connectivity_ratio(G_true: nx.Graph, G_pred: nx.Graph,
                       snap_tolerance_m: float = 15.0,
                       max_pairs: int = 1000,
                       seed: int = 0) -> ConnectivityResult:
    """Fraction of true node-pairs that are reachable in the predicted graph.
    Complements APLS: APLS penalizes length distortion, this measures whether
    connectivity exists at all."""
    rng = np.random.default_rng(seed)
    gt_to_pred = _snap_nodes(G_true, G_pred, snap_tolerance_m)
    # component id per predicted node
    pred_comp = {}
    for i, comp in enumerate(nx.connected_components(G_pred)):
        for n in comp:
            pred_comp[n] = i

    nodes = [n for n in G_true.nodes() if G_true.degree[n] > 0]
    if len(nodes) < 2:
        return ConnectivityResult(ratio=1.0, n_pairs_evaluated=0, n_reachable_in_pred=0)

    pairs = []
    max_all = len(nodes) * (len(nodes) - 1) // 2
    if max_all <= max_pairs:
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pairs.append((nodes[i], nodes[j]))
    else:
        seen = set()
        while len(pairs) < max_pairs:
            i, j = rng.integers(0, len(nodes), size=2)
            if i == j:
                continue
            key = (int(min(i, j)), int(max(i, j)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((nodes[i], nodes[j]))

    reachable = 0
    for u, v in pairs:
        # true-graph reachability filter: only count pairs that were reachable in truth
        if not nx.has_path(G_true, u, v):
            continue
        um = gt_to_pred.get(u)
        vm = gt_to_pred.get(v)
        if um is None or vm is None:
            continue
        if pred_comp.get(um) == pred_comp.get(vm) and pred_comp.get(um) is not None:
            reachable += 1

    return ConnectivityResult(
        ratio=reachable / len(pairs) if pairs else 1.0,
        n_pairs_evaluated=len(pairs),
        n_reachable_in_pred=reachable,
    )


# ---------------------------------------------------------------------------
# one-shot report
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Completeness / Correctness / Quality (Heipke 1997) -- the standard road-
# extraction triple. Computed length-based via node proximity matching:
#   * A true edge is "matched" if BOTH its endpoints have a proximity match in
#     the predicted graph AND the matched endpoints share a predicted component
#     (i.e. some path exists in the prediction between them).
#   * A predicted edge is "matched" symmetrically.
#   * completeness = matched_true_length / total_true_length
#   * correctness  = matched_pred_length / total_pred_length
#   * quality      = matched_true_length /
#                    (total_true_length + total_pred_length - matched_true_length)
#
# These complement APLS: APLS penalizes length distortion between matched
# pairs, CCQ measures how much road was found vs. hallucinated.
# ---------------------------------------------------------------------------

@dataclass
class CCQResult:
    completeness: float   # aka recall over road length
    correctness: float    # aka precision over road length
    quality: float        # aka IoU-like combined score
    matched_true_length_m: float
    matched_pred_length_m: float
    total_true_length_m: float
    total_pred_length_m: float


def _matched_length(G_from: nx.Graph, G_to: nx.Graph, snap_tolerance_m: float) -> tuple[float, float]:
    """Sum of edge lengths in G_from whose endpoints both have proximity matches
    in G_to AND those matched endpoints are reachable in G_to. Also returns
    total edge length in G_from."""
    node_map = _snap_nodes(G_from, G_to, snap_tolerance_m)
    to_comp = {}
    for i, comp in enumerate(nx.connected_components(G_to)):
        for n in comp:
            to_comp[n] = i
    matched = 0.0
    total = 0.0
    for u, v, data in G_from.edges(data=True):
        L = float(data.get("length", 0.0))
        total += L
        mu, mv = node_map.get(u), node_map.get(v)
        if mu is None or mv is None:
            continue
        if to_comp.get(mu) == to_comp.get(mv):
            matched += L
    return matched, total


def ccq(G_true: nx.Graph, G_pred: nx.Graph, snap_tolerance_m: float = 15.0) -> CCQResult:
    """Completeness / Correctness / Quality by matched road length."""
    matched_true, total_true = _matched_length(G_true, G_pred, snap_tolerance_m)
    matched_pred, total_pred = _matched_length(G_pred, G_true, snap_tolerance_m)
    completeness = matched_true / total_true if total_true > 0 else 0.0
    correctness = matched_pred / total_pred if total_pred > 0 else 0.0
    denom = total_true + total_pred - matched_true
    quality = matched_true / denom if denom > 0 else 0.0
    return CCQResult(
        completeness=completeness, correctness=correctness, quality=quality,
        matched_true_length_m=matched_true, matched_pred_length_m=matched_pred,
        total_true_length_m=total_true, total_pred_length_m=total_pred,
    )


@dataclass
class EvaluationReport:
    apls: APLSResult
    breaks: BreaksResult
    connectivity: ConnectivityResult
    ccq: Optional["CCQResult"] = None

    def summary(self) -> dict:
        d = {
            "APLS": round(self.apls.apls, 4),
            "APLS_gt_to_pred": round(self.apls.apls_gt_to_pred, 4),
            "APLS_pred_to_gt": round(self.apls.apls_pred_to_gt, 4),
            "breaks": self.breaks.breaks,
            "true_length_km": round(self.breaks.true_length_km, 3),
            "breaks_per_km": round(self.breaks.breaks_per_km, 4),
            "connectivity_ratio": round(self.connectivity.ratio, 4),
            "connectivity_pairs": self.connectivity.n_pairs_evaluated,
        }
        if self.ccq is not None:
            d.update({
                "completeness": round(self.ccq.completeness, 4),
                "correctness": round(self.ccq.correctness, 4),
                "quality": round(self.ccq.quality, 4),
            })
        return d


def evaluate(G_true: nx.Graph, G_pred: nx.Graph,
             snap_tolerance_m: float = 15.0,
             max_pairs: int = 500,
             seed: int = 0) -> EvaluationReport:
    return EvaluationReport(
        apls=apls(G_true, G_pred, snap_tolerance_m, max_pairs, seed),
        breaks=breaks_per_km(G_true, G_pred, snap_tolerance_m),
        connectivity=connectivity_ratio(G_true, G_pred, snap_tolerance_m, max_pairs * 2, seed),
        ccq=ccq(G_true, G_pred, snap_tolerance_m),
    )


# ---------------------------------------------------------------------------
# Pixel-level metrics: IoU, Dice, precision, recall -- OVERALL and (when an
# occlusion mask is supplied) RESTRICTED to the occluded region.
#
# "Occlusion-recall" is the deck-promised metric that isolates the model's
# actual value proposition: anyone can predict roads in clear areas; the
# question is how much road it recovers under canopy/shadow. Restricting
# every count to pixels flagged as occluded gives you exactly that answer.
#
# Inputs are pure numpy so this doesn't care whether the masks came from a
# real PathMamba prediction, synthetic rasterization, or a hand-annotated
# reference. Works in the pixel frame; no CRS awareness needed.
# ---------------------------------------------------------------------------

@dataclass
class PixelMetricsResult:
    threshold: float
    overall_iou: float
    overall_dice: float
    overall_precision: float
    overall_recall: float
    # occluded_* are None when no occlusion mask is supplied
    occluded_iou: Optional[float]
    occluded_dice: Optional[float]
    occluded_precision: Optional[float]
    occluded_recall: Optional[float]  # <-- the headline "occlusion-recall" number
    # bookkeeping
    n_occluded_pixels: int
    n_gt_road_pixels: int
    n_pred_road_pixels: int
    occluded_fraction_of_frame: float
    occluded_fraction_of_gt_roads: float

    def summary(self) -> dict:
        d = {
            "threshold": round(self.threshold, 3),
            "overall_iou": round(self.overall_iou, 4),
            "overall_dice": round(self.overall_dice, 4),
            "overall_precision": round(self.overall_precision, 4),
            "overall_recall": round(self.overall_recall, 4),
            "n_gt_road_pixels": int(self.n_gt_road_pixels),
            "n_pred_road_pixels": int(self.n_pred_road_pixels),
        }
        if self.occluded_recall is not None:
            d.update({
                "occlusion_recall": round(self.occluded_recall, 4),
                "occlusion_iou": round(self.occluded_iou, 4),
                "occlusion_dice": round(self.occluded_dice, 4),
                "occlusion_precision": round(self.occluded_precision, 4),
                "n_occluded_pixels": int(self.n_occluded_pixels),
                "occluded_fraction_of_frame": round(self.occluded_fraction_of_frame, 4),
                "occluded_fraction_of_gt_roads": round(self.occluded_fraction_of_gt_roads, 4),
            })
        return d


def _confusion_counts(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray = None):
    """Return (TP, FP, FN) counts. If mask is given, restrict counting to
    pixels where mask is True."""
    if mask is not None:
        pred = pred & mask
        gt = gt & mask
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    return tp, fp, fn


def _iou_dice_prec_rec(tp: int, fp: int, fn: int):
    """Compute the four standard binary-classification derived metrics from
    confusion counts. Returns 0.0 for degenerate cases (no positives at all)
    rather than NaN so downstream JSON serialization stays clean."""
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return iou, dice, precision, recall


def pixel_metrics(
    pred_prob: np.ndarray,
    gt_mask: np.ndarray,
    occlusion_mask: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    occlusion_threshold: float = 0.5,
) -> PixelMetricsResult:
    """Compute pixel-level road-segmentation metrics.

    Parameters
    ----------
    pred_prob : (H, W) float array of predicted road probabilities in [0, 1].
                Typically PathMamba's sigmoid road head output.
    gt_mask : (H, W) bool or 0/1 array of ground-truth road pixels.
    occlusion_mask : (H, W) optional. Bool or float. When supplied, an
                    additional block of "occluded-*" metrics is computed by
                    restricting all counts to pixels where this mask is true.
                    In synthetic mode this is typically the RoadPrediction's
                    confidence_mask; in real inference it would be the GLCM
                    occlusion signal.
    threshold : cutoff for turning pred_prob into a hard prediction. 0.5 is
                the standard for sigmoid outputs.
    occlusion_threshold : cutoff for turning a float occlusion_mask into a
                          bool. Ignored when occlusion_mask is already bool.

    Returns
    -------
    PixelMetricsResult with `.occluded_recall` populated when an occlusion
    mask was given. That single number is the deck's "occlusion-recall": the
    fraction of ground-truth road pixels UNDER OCCLUSION that the model
    correctly predicted. High values mean the model actually recovers roads
    in hard regions; low values mean it's basically only working in the clear.
    """
    pred = pred_prob >= threshold
    gt = gt_mask.astype(bool) if gt_mask.dtype != bool else gt_mask

    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt shape mismatch: {pred.shape} vs {gt.shape}")

    tp, fp, fn = _confusion_counts(pred, gt)
    o_iou, o_dice, o_prec, o_rec = _iou_dice_prec_rec(tp, fp, fn)

    occ_iou = occ_dice = occ_prec = occ_rec = None
    n_occluded = 0
    occ_frac_frame = 0.0
    occ_frac_gt = 0.0

    if occlusion_mask is not None:
        if occlusion_mask.shape != pred.shape:
            raise ValueError(
                f"occlusion_mask shape {occlusion_mask.shape} != pred shape {pred.shape}")
        if occlusion_mask.dtype == bool:
            occ = occlusion_mask
        else:
            occ = occlusion_mask >= occlusion_threshold
        n_occluded = int(occ.sum())
        if n_occluded > 0:
            tpo, fpo, fno = _confusion_counts(pred, gt, mask=occ)
            occ_iou, occ_dice, occ_prec, occ_rec = _iou_dice_prec_rec(tpo, fpo, fno)
            occ_frac_frame = n_occluded / occ.size
            occluded_gt = int(np.logical_and(gt, occ).sum())
            total_gt = int(gt.sum())
            occ_frac_gt = occluded_gt / total_gt if total_gt > 0 else 0.0
        else:
            # empty occlusion mask -- keep None sentinels; caller sees this
            # via n_occluded_pixels==0 in the summary
            pass

    return PixelMetricsResult(
        threshold=threshold,
        overall_iou=o_iou, overall_dice=o_dice,
        overall_precision=o_prec, overall_recall=o_rec,
        occluded_iou=occ_iou, occluded_dice=occ_dice,
        occluded_precision=occ_prec, occluded_recall=occ_rec,
        n_occluded_pixels=n_occluded,
        n_gt_road_pixels=int(gt.sum()),
        n_pred_road_pixels=int(pred.sum()),
        occluded_fraction_of_frame=occ_frac_frame,
        occluded_fraction_of_gt_roads=occ_frac_gt,
    )
