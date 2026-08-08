"""
app.py -- RouteResilience UI
Run:  streamlit run app.py --server.port 6006 --server.address 0.0.0.0

Includes the live re-tuning panel: skeletonization + healing rerun on the
SAVED (prob_mask, confidence_mask) arrays, not the model, so parameters can
be adjusted and reapplied in seconds on CPU without a GPU reload.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import pickle
import random
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.set_page_config(page_title="RouteResilience", layout="wide")

# ---------------------------------------------------------------------------
# Theme: Cream (#FFFDF2) base, Beige (#DDD0C8) accents, Black text
# ---------------------------------------------------------------------------
CSS = """
<style>
* { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif !important; }
.stApp { background-color: #FFFDF2; }

h1, h2, h3 { color: #000000 !important; font-weight: 700; }
p, span, label, .stCaption, .stMarkdown { color: #000000; }

[data-testid="stMetricLabel"] { color: #6B5B50 !important; font-size: 0.95rem; }
[data-testid="stMetricValue"] { color: #000000; font-weight: 700; font-size: 1.5rem; }
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DDD0C8;
    border-radius: 10px;
    padding: 10px 14px;
}

/* Buttons: black background -- text MUST be forced white, since Streamlit
   wraps the visible label in an inner <p>/<div>/<span>, which the general
   text-color rule above would otherwise directly match and override
   (color only inherits when nothing more specific targets the element
   itself -- this is exactly that case). Every inner tag type covered,
   all !important so nothing else can re-break this again. */
div.stButton > button {
    background-color: #000000 !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    font-size: 1.02rem;
}
div.stButton > button,
div.stButton > button p,
div.stButton > button span,
div.stButton > button div {
    color: #FFFDF2 !important;
}
div.stButton > button:hover {
    background-color: #333333 !important;
}

.choice-card {
    background-color: #FFFFFF;
    border: 1px solid #DDD0C8;
    border-radius: 14px;
    padding: 1.6rem 1.8rem 1.2rem 1.8rem;
    text-align: center;
    margin-bottom: 0.6rem;
}
.choice-card h3 { margin-top: 0; }
.choice-card p { color: #6B5B50; margin-bottom: 0; }

.hint-box {
    background-color: #F5EDE4;
    border-left: 4px solid #DDD0C8;
    border-radius: 6px;
    padding: 0.7rem 1rem;
    font-size: 0.95rem;
    color: #6B5B50;
    margin-bottom: 1rem;
}

/* Team name deliberately LARGER than the page title -- it's the banner
   identity, the title is secondary to it on the front page. */
.team-name {
    color: #5C3A21;
    font-weight: 800;
    font-size: 2.6rem;
    letter-spacing: 0.03em;
    margin-bottom: 0.2rem;
    line-height: 1.1;
}
.route-title {
    color: #000000;
    font-weight: 600;
    font-size: 1.7rem;
    margin-bottom: 1rem;
}

table, th, td { color: #000000 !important; font-size: 1rem; }
thead tr th { background-color: #F5EDE4 !important; color: #000000 !important; font-weight: 600; }
tbody tr { background-color: #FFFFFF !important; }
tbody tr:nth-child(even) { background-color: #FDFAF5 !important; }

/* Half-width fit: two side-by-side panels each need a shorter max-height
   than a single full-width one did, so both fit on screen together. */
.fit-image-half img {
    max-height: 48vh;
    width: auto;
    max-width: 100%;
    object-fit: contain;
    display: block;
    margin: 0 auto;
    border-radius: 10px;
    border: 1px solid #DDD0C8;
}

.signal-strong { color: #1E7A34; font-weight: 700; }
.signal-moderate { color: #B8860B; font-weight: 700; }
.signal-weak { color: #C0392B; font-weight: 700; }
.signal-zero { color: #7A1E1E; font-weight: 700; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading -- now also loads confidence_mask.npy + tracks report_dir,
# both needed for the re-tuning panel.
# ---------------------------------------------------------------------------

def _find_report_dir(run_dir: str) -> str | None:
    direct = os.path.join(run_dir, "resilience_report.json")
    if os.path.exists(direct):
        return run_dir
    if os.path.isdir(run_dir):
        for sub in os.listdir(run_dir):
            nested = os.path.join(run_dir, sub, "resilience_report.json")
            if os.path.exists(nested):
                return os.path.join(run_dir, sub)
    return None


def load_cached_result(run_dir: str) -> dict:
    report_dir = _find_report_dir(run_dir)
    if report_dir is None:
        raise FileNotFoundError(f"No result found under {run_dir}")
    with open(os.path.join(report_dir, "resilience_report.json")) as f:
        report = json.load(f)
    with open(os.path.join(report_dir, "reconstructed_graph.gpickle"), "rb") as f:
        healed_graph = pickle.load(f)
    prob_mask_path = os.path.join(report_dir, "prob_mask.npy")
    conf_mask_path = os.path.join(report_dir, "confidence_mask.npy")
    prob_mask = np.load(prob_mask_path) if os.path.exists(prob_mask_path) else None
    confidence_mask = np.load(conf_mask_path) if os.path.exists(conf_mask_path) else None
    viz_path = os.path.join(report_dir, "visualization.png")
    viz = viz_path if os.path.exists(viz_path) else None
    input_image_path = os.path.join(report_dir, "input_image.png")
    input_image = input_image_path if os.path.exists(input_image_path) else None
    return {
        "report": report, "healed_graph": healed_graph,
        "prob_mask": prob_mask, "confidence_mask": confidence_mask,
        "viz": viz, "input_image": input_image, "report_dir": report_dir,
    }


# ---------------------------------------------------------------------------
# Live re-tuning: rerun skeletonization + healing on SAVED masks, not the
# model. Seconds on CPU. Verified against real skeleton_to_graph()/heal()/
# run_one_tile() signatures -- every parameter name here matches exactly.
# ---------------------------------------------------------------------------

def retune_graph_from_saved_masks(
    report_dir: str, pixel_size_m: float,
    threshold: float, threshold_low: float | None,
    min_object_px: int, min_branch_m: float, closing_iterations: int,
    dp_epsilon_m: float, min_edge_mean_prob: float,
    heal_occlusion_min: float, heal_angle_max_deg: float,
    heal_gap_max_m: float, heal_max_search_radius_m: float,
) -> dict:
    from route_resilience.geo_utils import SimpleAffine
    from route_resilience.model_io import RoadPrediction
    from route_resilience.skeletonization import graph_from_prediction
    from route_resilience.healing import heal
    from route_resilience.zlevel import assign_z_levels
    from route_resilience.centrality import composite_criticality
    from route_resilience.scenarios import run_default_scenarios
    from route_resilience.visualize import render_pipeline_result
    from route_resilience.pipeline_diagnostics import diagnose_prediction

    prob_mask = np.load(os.path.join(report_dir, "prob_mask.npy"))
    conf_mask = np.load(os.path.join(report_dir, "confidence_mask.npy"))
    H, _ = prob_mask.shape

    transform = SimpleAffine(origin_x=0.0, origin_y=0.0,
                             pixel_size=pixel_size_m, height_px=H)
    prediction = RoadPrediction(
        prob_mask=prob_mask, confidence_mask=conf_mask,
        transform=transform, source_gsd_m=pixel_size_m, is_synthetic=False,
    )

    diag = diagnose_prediction(prob_mask, conf_mask)

    reconstructed = graph_from_prediction(
        prediction,
        threshold=threshold, threshold_low=threshold_low,
        min_object_px=min_object_px, min_branch_m=min_branch_m,
        closing_iterations=closing_iterations,
        dp_epsilon_m=dp_epsilon_m, min_edge_mean_prob=min_edge_mean_prob,
    )
    z_graph = assign_z_levels(reconstructed)
    healing_report = heal(
        z_graph, prediction,
        occlusion_min=heal_occlusion_min,
        angle_max_deg=heal_angle_max_deg,
        gap_max_m=heal_gap_max_m,
        max_search_radius_m=heal_max_search_radius_m,
    )
    healed_graph = healing_report.healed_graph

    criticality = composite_criticality(healed_graph)
    scenarios_out = run_default_scenarios(healed_graph, criticality, seed=0)

    composite = criticality.get("composite", {})
    top10 = sorted(composite.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top10_dump = [
        {"node": int(n), "score": round(float(s), 4),
         "x": float(healed_graph.nodes[n]["x"]),
         "y": float(healed_graph.nodes[n]["y"])}
        for n, s in top10 if n in healed_graph.nodes
    ]
    headline_ri = scenarios_out.get("single_top_node", {}).get("resilience_index")
    headline_top = (scenarios_out.get("single_top_node", {})
                    .get("disabled_nodes", [None]) or [None])[0]

    def tally(rej):
        return dict(Counter(
            getattr(c, "reject_reason", "unknown") or "unknown" for c in rej))

    summary = {
        "graph_stats": {
            "reconstructed_nodes": healed_graph.number_of_nodes(),
            "reconstructed_edges": healed_graph.number_of_edges(),
            "reconstructed_components": nx.number_connected_components(healed_graph),
        },
        "healing": {
            "bridges_added": len(healing_report.bridges_added),
            "candidates_rejected": len(healing_report.rejected),
            "rejection_reasons": tally(healing_report.rejected),
        },
        "criticality_top10": top10_dump,
        "scenarios": scenarios_out,
        "prediction_diagnostics": diag.summary(),
        "top_criticality_node": headline_top,
        "resilience_index": headline_ri,
        "retuned_parameters": {
            "threshold": threshold, "threshold_low": threshold_low,
            "min_object_px": min_object_px, "min_branch_m": min_branch_m,
            "closing_iterations": closing_iterations, "dp_epsilon_m": dp_epsilon_m,
            "min_edge_mean_prob": min_edge_mean_prob,
            "heal_occlusion_min": heal_occlusion_min,
            "heal_angle_max_deg": heal_angle_max_deg,
            "heal_gap_max_m": heal_gap_max_m,
            "heal_max_search_radius_m": heal_max_search_radius_m,
        },
    }

    scenarios_for_report = {
        k: ({kk: vv for kk, vv in v.items() if kk != "polygon_coords"}
            if isinstance(v, dict) else v)
        for k, v in scenarios_out.items()
    }
    summary_for_report = {**summary, "scenarios": scenarios_for_report}
    with open(os.path.join(report_dir, "resilience_report.json"), "w") as f:
        json.dump(summary_for_report, f, indent=2, default=str)
    with open(os.path.join(report_dir, "reconstructed_graph.gpickle"), "wb") as f:
        pickle.dump(healed_graph, f)

    try:
        render_pipeline_result(
            prob_mask=prob_mask, transform=transform,
            healed_graph=healed_graph, healing_report=healing_report,
            criticality=criticality.get("composite"),
            out_path=os.path.join(report_dir, "visualization.png"),
            title=f"Re-tuned: {healed_graph.number_of_nodes()} nodes, "
                  f"{len(healing_report.bridges_added)} bridges, RI={headline_ri}",
        )
    except Exception as e:
        st.warning(f"Visualization skipped: {e}")

    return load_cached_result(report_dir)


def run_live_inference(image_path: str, checkpoint_path: str, out_dir: str,
                       source_gsd_m: float, hp: dict,
                       batch_size: int = 4, tile_size: int = 512,
                       use_fast_glcm_proxy: bool = True) -> dict:
    from route_resilience.run_real_inference import load_trained_model, run_one_tile

    if not os.path.exists(checkpoint_path):
        st.error(
            f"Checkpoint not found at:\n\n`{checkpoint_path}`\n\n"
            "This path is checked on the SAME machine the Streamlit app is "
            "running on -- not your local laptop. Find the real path with a "
            "terminal on that machine:\n\n"
            "`find / -iname '*.pth' 2>/dev/null`\n\n"
            "then paste the result into the 'Model checkpoint path' field above."
        )
        st.stop()

    if "model" not in st.session_state:
        with st.spinner("Setting things up (only happens once)..."):
            model, glcm_norm_value, global_stats = load_trained_model(checkpoint_path)
            st.session_state["model"] = model
            st.session_state["glcm_norm_value"] = glcm_norm_value
            st.session_state["global_stats"] = global_stats

    with st.spinner("Reading the roads from your image -- this takes a few minutes..."):
        run_one_tile(
            image_path=image_path,
            model=st.session_state["model"],
            glcm_norm_value=st.session_state["glcm_norm_value"],
            global_stats=st.session_state["global_stats"],
            out_dir=out_dir,
            source_gsd_m=source_gsd_m,
            batch_size=batch_size,
            tile_size=tile_size,
            use_fast_glcm_proxy=use_fast_glcm_proxy,
            **hp,
        )
    return load_cached_result(out_dir)


# ---------------------------------------------------------------------------
# Reusable slider block -- same knobs in the upload form and re-tune panel.
# Verified against run_one_tile()'s and heal()'s real parameter names.
# ---------------------------------------------------------------------------

def parameter_sliders(prefix: str, defaults: dict | None = None) -> dict:
    d = defaults or {}
    st.caption("Tighten these when the map shows too many bridges through empty space, "
              "or connections that don't make sense.")

    st.markdown("**Road detection**")
    c1, c2 = st.columns(2)
    threshold = c1.slider(
        "Detection threshold", 0.10, 0.90, float(d.get("threshold", 0.5)), 0.05,
        key=f"{prefix}_threshold",
        help="How confident the model needs to be before a pixel counts as road.",
    )
    threshold_low = c2.slider(
        "Faint-road recovery", 0.05, 0.50, float(d.get("threshold_low", 0.30)), 0.05,
        key=f"{prefix}_threshold_low",
        help="Recovers faint road continuations connected to a confident detection, "
             "without picking up random noise elsewhere.",
    )

    st.markdown("**Cleanup (removes noise before it becomes a bridge)**")
    c3, c4, c5 = st.columns(3)
    min_object_px = c3.slider(
        "Ignore specks smaller than (px)", 5, 200, int(d.get("min_object_px", 40)), 5,
        key=f"{prefix}_min_object_px",
    )
    min_branch_m = c4.slider(
        "Trim dead-ends shorter than (m)", 2, 50, int(d.get("min_branch_m", 15)), 1,
        key=f"{prefix}_min_branch_m",
    )
    closing_iterations = c5.slider(
        "Fill tiny gaps (iterations)", 0, 3, int(d.get("closing_iterations", 2)), 1,
        key=f"{prefix}_closing_iterations",
    )

    c6, c7 = st.columns(2)
    dp_epsilon_m = c6.slider(
        "Road-line smoothing (m)", 0.0, 5.0, float(d.get("dp_epsilon_m", 1.0)), 0.5,
        key=f"{prefix}_dp_epsilon_m",
    )
    min_edge_mean_prob = c7.slider(
        "Minimum confidence per road segment", 0.0, 0.9, float(d.get("min_edge_mean_prob", 0.0)), 0.05,
        key=f"{prefix}_min_edge_mean_prob",
    )

    st.markdown("**Bridging broken roads (three checks a bridge must pass)**")
    c8, c9 = st.columns(2)
    heal_occlusion_min = c8.slider(
        "Minimum visual evidence", 0.0, 0.9, float(d.get("heal_occlusion_min", 0.30)), 0.05,
        key=f"{prefix}_heal_occlusion_min",
        help="Higher = stricter. Prevents bridges through visibly empty space.",
    )
    heal_angle_max_deg = c9.slider(
        "Max direction change (deg)", 10, 90, int(d.get("heal_angle_max_deg", 35)), 5,
        key=f"{prefix}_heal_angle_max_deg",
        help="Lower = stricter. Prevents connecting roads that don't line up.",
    )

    c10, c11 = st.columns(2)
    heal_gap_max_m = c10.slider(
        "Longest bridge allowed (m)", 5, 200, int(d.get("heal_gap_max_m", 50)), 5,
        key=f"{prefix}_heal_gap_max_m",
    )
    heal_max_search_radius_m = c11.slider(
        "Search radius for candidates (m)", 10, 300, int(d.get("heal_max_search_radius_m", 60)), 5,
        key=f"{prefix}_heal_max_search_radius_m",
    )

    return {
        "threshold": threshold,
        "threshold_low": threshold_low,
        "min_object_px": min_object_px,
        "min_branch_m": float(min_branch_m),
        "closing_iterations": closing_iterations,
        "dp_epsilon_m": dp_epsilon_m,
        "min_edge_mean_prob": min_edge_mean_prob,
        "heal_occlusion_min": heal_occlusion_min,
        "heal_angle_max_deg": float(heal_angle_max_deg),
        "heal_gap_max_m": float(heal_gap_max_m),
        "heal_max_search_radius_m": float(heal_max_search_radius_m),
    }


def _signal_badge(signal: str) -> str:
    cls = {"STRONG": "signal-strong", "MODERATE": "signal-moderate",
           "WEAK": "signal-weak", "NEAR_ZERO": "signal-zero"}.get(signal, "")
    return f'<span class="{cls}">{signal}</span>'


# ---------------------------------------------------------------------------
# Graph visualization with numbered nodes, flow paths, and break highlighting
# ---------------------------------------------------------------------------

def _get_pos(G):
    return {n: (d.get("x", 0), d.get("y", 0)) for n, d in G.nodes(data=True)}


def _node_label_map(G):
    return {n: str(i + 1) for i, n in enumerate(sorted(G.nodes()))}


def _apply_disruption(G: nx.Graph, broken_nodes, broken_edges) -> nx.Graph:
    """Returns a copy of G with the given nodes AND edges removed. Local to
    app.py rather than a route_resilience.simulation change, since
    simulation.run_scenario only supports removing nodes -- roads (edges) as
    an independent, combinable disruption type is a UI-only extension for
    now. Edges are removed first so that removing a node afterward doesn't
    raise on an edge that's already gone."""
    G2 = G.copy()
    valid_edges = [(u, v) for (u, v) in broken_edges if G2.has_edge(u, v)]
    G2.remove_edges_from(valid_edges)
    G2.remove_nodes_from([n for n in broken_nodes if n in G2])
    return G2


def _largest_component_fraction(G_before: nx.Graph, G_after: nx.Graph) -> float:
    """Fraction of G_before's node count that survives in G_after's largest
    connected component. Mirrors simulation.ScenarioResult's
    largest_component_fraction but computed against a node+edge disruption
    rather than a node-only one."""
    total = G_before.number_of_nodes()
    if total == 0:
        return 1.0
    if G_after.number_of_nodes() == 0:
        return 0.0
    largest = max((len(c) for c in nx.connected_components(G_after)), default=0)
    return largest / total


def _unreachable_od_pairs(G_after: nx.Graph, od_pairs: list) -> tuple[int, int]:
    """(unreachable_count, total_count) over the given OD pairs against
    G_after. A pair counts as unreachable if either endpoint no longer
    exists in G_after (removed as a broken node) or no path connects them."""
    unreachable = 0
    for o, d in od_pairs:
        if o not in G_after or d not in G_after or not nx.has_path(G_after, o, d):
            unreachable += 1
    return unreachable, len(od_pairs)


def _graph_fingerprint(G: nx.Graph):
    """Cheap, hashable stand-in for a whole nx.Graph, for use as an
    st.cache_data hash_funcs key. Sorting nodes/edges is O(n log n) -- trivial
    next to the O(V*E)+ centrality algorithms below, so this buys real caching
    without needing the graph object itself to be hashable."""
    return (tuple(sorted(G.nodes())), tuple(sorted(G.edges())),
            G.number_of_nodes(), G.number_of_edges())


@st.cache_data(show_spinner=False, max_entries=4,
               hash_funcs={nx.Graph: _graph_fingerprint})
def _composite_criticality_cached(G: nx.Graph) -> dict:
    """Cached wrapper around centrality.composite_criticality.

    composite_criticality runs betweenness + current-flow betweenness +
    k-core + alpha centrality over the WHOLE healed_graph -- none of which
    depends on which nodes/edges are currently toggled broken in the sandbox.
    Before this cache, it was being recomputed from scratch on every single
    sandbox click (every click triggers st.rerun(), which re-executes the
    whole script top to bottom), even though its result is identical for the
    entire session unless the graph itself changes (e.g. via re-tuning).
    That was the single biggest contributor to sluggish node/road toggling --
    far more expensive than the plotting code downstream.
    """
    from route_resilience.centrality import composite_criticality
    return composite_criticality(G)


@st.cache_data(show_spinner=False, max_entries=4,
               hash_funcs={nx.Graph: _graph_fingerprint})
def _sample_od_pairs_cached(G: nx.Graph, max_pairs: int, seed: int) -> list:
    """Cached wrapper around scenarios._sample_od_pairs -- same reasoning as
    _composite_criticality_cached: deterministic given (G, max_pairs, seed),
    but was being resampled on every sandbox rerun."""
    from route_resilience.scenarios import _sample_od_pairs
    return _sample_od_pairs(G, max_pairs=max_pairs, seed=seed)


@st.cache_data(show_spinner=False, max_entries=64,
               hash_funcs={nx.Graph: _graph_fingerprint})
def _disruption_analysis_cached(
    G: nx.Graph, broken_nodes_fs: frozenset, broken_edges_fs: frozenset,
    origin, dest, od_pairs_t: tuple,
) -> dict:
    """Cached bundle of every metric that depends on a specific TOGGLE STATE
    (which nodes/edges are currently broken) -- the shortest route, largest-
    connected-component fraction, OD-pair reachability, and before/after
    route distances.

    The cache key is (graph, frozenset(broken_nodes), frozenset(broken_edges),
    origin, dest, od_pairs) -- frozensets so toggle ORDER doesn't matter and
    the key stays hashable. This is what makes toggling itself fast, not just
    the plotting: every one of these calls does a full shortest-path and/or
    connected-components pass over the graph, and the SAME toggle state comes
    up constantly in practice --
      - scrubbing the flood slider back to a step you already visited,
      - "Play flood" replaying the exact same step sequence from the top
        every time it's re-triggered,
      - toggling a node off then back on (returns to the empty-set state),
      - switching origin/destination while broken_nodes/broken_edges stay put.
    Any of those now hit this cache instead of re-running shortest_path /
    connected_components / has_path from scratch. Only a genuinely NEW toggle
    state (or a new graph, via re-tuning) actually recomputes.

    Deliberately returns a plain dict of numbers/lists rather than G_after
    itself -- G_after is cheap to rebuild locally wherever still needed
    (_apply_disruption is a single copy + remove_edges_from + remove_nodes_from),
    so there's no reason to pay to serialize a whole graph in/out of the cache.
    """
    broken_nodes = list(broken_nodes_fs)
    broken_edges = set(broken_edges_fs)
    od_pairs = list(od_pairs_t)

    G_after = _apply_disruption(G, broken_nodes, broken_edges)

    try:
        route_nodes = nx.shortest_path(G_after, origin, dest, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        route_nodes = []

    lcc_frac = _largest_component_fraction(G, G_after)
    unreachable, total_pairs = _unreachable_od_pairs(G_after, od_pairs)

    origin_in_after = origin in G_after
    dest_in_after = dest in G_after

    try:
        before_len = nx.shortest_path_length(G, origin, dest, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        before_len = None

    after_len = None
    if origin_in_after and dest_in_after:
        try:
            after_len = nx.shortest_path_length(G_after, origin, dest, weight="length")
        except nx.NetworkXNoPath:
            after_len = None

    return {
        "route_nodes": route_nodes,
        "lcc_frac": lcc_frac,
        "unreachable": unreachable,
        "total_pairs": total_pairs,
        "origin_in_after": origin_in_after,
        "dest_in_after": dest_in_after,
        "before_len": before_len,
        "after_len": after_len,
    }


def _flood_progression(G: nx.Graph, origin, n_steps: int = 8, weight: str = "length") -> list[set]:
    """Simulate a flood spreading outward from `origin` across the road
    network, without needing real elevation/DEM data.

    Approach: treat each edge's real-world `length` (meters) as the water's
    travel cost, and run a single-source Dijkstra frontier from `origin` --
    the same "expanding cost frontier" idea used for fire/contamination
    spread models when no terrain data is available. A junction is
    "underwater" at step i once the shortest cumulative distance to reach it
    from the origin falls within that step's threshold. This is a distance
    proxy, not real hydrology (real floods follow low ground, not shortest
    path) -- if a real DEM/elevation raster becomes available, only this
    function needs to change (e.g. seed multiple origins along a mapped
    waterway, weight edges by a slope-aware cost instead of raw length, or
    threshold directly against elevation rather than distance); every
    downstream caller (auto-toggling, playback, metrics) stays identical
    since they only consume the returned node sets.

    Returns a list of `n_steps` sets, each the CUMULATIVE set of node ids
    flooded by that step (monotonically growing, so step i is a superset of
    step i-1). Nodes with no path from `origin` (a separate component) never
    flood -- water can't reach them through this network either.
    """
    if n_steps < 1:
        return []
    if origin not in G:
        return [set() for _ in range(n_steps)]
    dist = nx.single_source_dijkstra_path_length(G, origin, weight=weight)
    if not dist:
        return [set() for _ in range(n_steps)]
    max_d = max(dist.values()) or 1.0
    thresholds = [max_d * (i + 1) / n_steps for i in range(n_steps)]
    return [{n for n, d in dist.items() if d <= t} for t in thresholds]


def _flood_edges_for_nodes(G: nx.Graph, flooded_nodes: set) -> set:
    """Edges with BOTH endpoints flooded -- used only for the flood-progress
    caption/count; _apply_disruption already drops these automatically as a
    side effect of removing the nodes, so this is display-only, not fed back
    into clicked_broken_edges."""
    return {(u, v) for u, v in G.edges() if u in flooded_nodes and v in flooded_nodes}


@st.cache_data(show_spinner=False, max_entries=16,
               hash_funcs={nx.Graph: _graph_fingerprint})
def _flood_progression_cached(G: nx.Graph, origin, n_steps: int, weight: str = "length") -> list:
    """Cached wrapper around _flood_progression -- during flood playback the
    fragment reruns itself every ~0.6s (see the autoplay loop below), and the
    Dijkstra frontier for a given (graph, origin, steps) is identical on
    every one of those reruns, so this avoids resolving it from scratch each
    frame."""
    return _flood_progression(G, origin, n_steps=n_steps, weight=weight)


def _edge_criticality_scores(G: nx.Graph, node_criticality: dict) -> dict:
    """Derives a per-edge criticality score from the two endpoint node scores.
    Uses the MAX of the two endpoints rather than the mean: an edge is only
    as safe as its most critical end -- a road segment leading into a major
    junction is itself a bottleneck approach, even if its other end is a
    quiet dead-end. Mean would wash that out.

    Returns {(u, v): score} for every edge, keyed exactly as G.edges() would
    iterate them (u, v as stored, not sorted) so callers can look up scores
    directly during iteration.
    """
    scores = {}
    for u, v in G.edges():
        su = node_criticality.get(u, 0.0)
        sv = node_criticality.get(v, 0.0)
        scores[(u, v)] = max(su, sv)
    return scores


@st.cache_data(show_spinner=False, max_entries=8)
def _density_grid_cached(
    node_order_t: tuple, xs_t: tuple, ys_t: tuple, scores_t: tuple,
    grid_res: int, sigma_frac: float, padding_frac: float,
):
    """Cached core of _criticality_density_grid. All args are plain tuples
    (hashable) so Streamlit can key the cache on them directly. This is the
    actual fix for sandbox lag: clicking a node/road to toggle it NEVER
    changes node positions or criticality scores -- only which are marked
    broken -- so this expensive Gaussian-splat computation is identical on
    every single rerun while playing with the sandbox, and was previously
    being recomputed from scratch on every click (twice per rerun, once for
    the static topology panel and once for the sandbox chart, since both
    call this with the same inputs). Caching it means it now runs ONCE per
    (graph, selected metric) combination for the whole session, not once
    per click.
    """
    xs = np.array(xs_t, dtype=np.float64)
    ys = np.array(ys_t, dtype=np.float64)
    weights = np.array(scores_t, dtype=np.float64)

    if len(xs) == 0 or weights.max() <= 0:
        gx = np.linspace(0, 1, grid_res)
        gy = np.linspace(0, 1, grid_res)
        return gx, gy, np.zeros((grid_res, grid_res))

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    diag = math.hypot(x_max - x_min, y_max - y_min) or 1.0
    pad = diag * padding_frac
    sigma = max(diag * sigma_frac, 1e-6)

    gx = np.linspace(x_min - pad, x_max + pad, grid_res)
    gy = np.linspace(y_min - pad, y_max + pad, grid_res)
    GX, GY = np.meshgrid(gx, gy)

    dx = GX[None, :, :] - xs[:, None, None]
    dy = GY[None, :, :] - ys[:, None, None]
    gauss = np.exp(-(dx**2 + dy**2) / (2 * sigma**2)) * weights[:, None, None]
    intensity = gauss.sum(axis=0)

    peak = intensity.max()
    if peak > 0:
        intensity = intensity / peak
    return gx, gy, intensity


def _criticality_density_grid(
    pos: dict, criticality: dict, node_order: list,
    grid_res: int = 90, sigma_frac: float = 0.05, padding_frac: float = 0.12,
):
    """Splats each node's criticality score onto a 2D grid as a Gaussian bump,
    then sums and normalizes -- the standard "heatmap glow" trick: instead of
    flat-colored dots, high-criticality junctions radiate outward into their
    surroundings and overlapping bumps from nearby critical junctions
    reinforce each other, producing the blurred red/orange halos a real
    heatmap has instead of a scatter of isolated colored circles.

    Thin wrapper around _density_grid_cached -- converts the dict/list
    inputs into hashable tuples so Streamlit's cache can key on them, then
    delegates the actual (expensive) computation. See that function's
    docstring for why this caching matters.

    grid_res lowered from an earlier 130 to 90 by default: visually still
    reads as a smooth heatmap at this resolution, and cuts the Gaussian
    broadcast's grid-cell count by more than half regardless of caching --
    helps first-render latency even before the cache has anything to serve.

    Returns (grid_x, grid_y, intensity) where intensity is normalized to
    [0, 1] by its own max (so the brightest spot on THIS graph is always full
    intensity -- relative, not absolute, since criticality scores are already
    normalized 0-1 individually but their Gaussian SUM is not).
    """
    node_order_t = tuple(node_order)
    xs_t = tuple(pos[n][0] for n in node_order)
    ys_t = tuple(pos[n][1] for n in node_order)
    scores_t = tuple(criticality.get(n, 0.0) for n in node_order)
    return _density_grid_cached(node_order_t, xs_t, ys_t, scores_t,
                                grid_res, sigma_frac, padding_frac)


def _flood_density_grid(
    pos: dict, flooded_nodes: set, node_order: list,
    grid_res: int = 90, sigma_frac: float = 0.075, padding_frac: float = 0.12,
):
    """Same Gaussian-splat mechanic as _criticality_density_grid, reusing the
    identical cached core (_density_grid_cached doesn't care what the
    'weight' at each node means) -- except the weight splatted per node is
    BINARY: 1.0 if currently flooded, 0.0 otherwise. That produces a pooling
    wash of color over the flooded area instead of a criticality gradient.

    sigma_frac is slightly wider than the criticality glow's 0.05 so
    adjacent flooded junctions bleed into one continuous body of water
    rather than reading as separate blue blobs -- closer to how a flood
    actually looks (a spreading puddle, not discrete dots).

    Returns (grid_x, grid_y, intensity) exactly like _criticality_density_grid.
    An empty flooded_nodes set produces an all-zero (invisible) grid, same
    "always emit, sometimes empty" pattern used everywhere else on this chart.
    """
    node_order_t = tuple(node_order)
    xs_t = tuple(pos[n][0] for n in node_order)
    ys_t = tuple(pos[n][1] for n in node_order)
    scores_t = tuple(1.0 if n in flooded_nodes else 0.0 for n in node_order)
    return _density_grid_cached(node_order_t, xs_t, ys_t, scores_t,
                                grid_res, sigma_frac, padding_frac)


def _prob_mask_background_uri(prob_mask: np.ndarray, dim_factor: float = 0.55,
                              max_dim_px: int = 900) -> str:
    """Renders prob_mask as a dim, warm duo-tone background image (base64 data
    URI) so the network graph can sit directly on top of the actual road
    probability map instead of floating on blank space.

    Duo-tone instead of plain grayscale: near-black base + warm parchment for
    road pixels, matching the app's cream/beige theme rather than a flat
    grayscale mask. dim_factor scales road brightness down so graph markers
    and edges stay legible on top -- lower = dimmer background, more contrast
    for the overlaid graph.

    Downsamples to max_dim_px on the long edge first: this is a decorative
    backdrop, not a measurement, so full resolution buys nothing and only
    slows down base64 encoding/page rendering.
    """
    m = np.clip(prob_mask.astype(np.float32), 0.0, 1.0)
    h, w = m.shape
    scale = min(1.0, max_dim_px / max(h, w))
    if scale < 1.0:
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        m = np.array(
            Image.fromarray((m * 255).astype(np.uint8)).resize(
                (new_w, new_h), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0

    m = m * dim_factor
    base = np.array([16, 13, 11], dtype=np.float32)      # near-black
    road = np.array([236, 222, 196], dtype=np.float32)   # warm parchment
    rgb = base[None, None, :] * (1 - m[..., None]) + road[None, None, :] * m[..., None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def plot_network_topology_interactive(
    G: nx.Graph, criticality: dict, broken_nodes: set,
    background_prob_mask: np.ndarray | None = None,
    pixel_size_m: float = 0.28,
    broken_edges: set | None = None,
    route_nodes: list | None = None,
    flooded_nodes: set | None = None,
) -> go.Figure:
    """Interactive, clickable network graph. Nodes colored on a continuous
    gradient by criticality (green = low, dark red = high), with a blurred
    heatmap glow radiating from each node underneath, instead of a
    flat single color -- click a node directly to toggle it as broken,
    replacing the old multiselect dropdown entirely. `criticality` is the
    FULL per-node composite score dict from centrality.composite_criticality
    (not just the top-10 subset the JSON report saves) -- computed live in
    the calling code so every node gets a real gradient value, not just the
    ones that happened to make the top 10.

    broken_edges : optional set of (u, v) tuples -- roads toggled off via the
    new edge-midpoint click targets (see the edge_midpoint_trace below).
    Order within each tuple doesn't matter; both (u, v) and (v, u) are
    checked when testing membership, since G is undirected.

    route_nodes : optional ordered list of node ids forming a path (e.g. from
    nx.shortest_path) -- drawn as a bold highlighted line on top of
    everything else so a selected origin/destination route is visible at a
    glance, including how it reroutes or breaks as nodes/edges get toggled.
    None or a list shorter than 2 elements means "no route to draw."

    flooded_nodes : optional set of node ids currently "underwater" in a
    flood simulation (see _flood_progression). Rendered as a soft blue
    Gaussian-splat wash (_flood_density_grid) beneath the edges/nodes, same
    mechanic as the criticality glow, so the flooded area reads as a
    spreading body of water rather than a scatter of individually toggled
    junctions. None or empty means no water layer (a harmless all-zero
    heatmap is still emitted -- see the fixed-trace-order note below).

    background_prob_mask : optional (H, W) prob_mask array. When given, the
    graph is drawn directly on top of a dim, warm duo-tone rendering of the
    actual road-probability map (see _prob_mask_background_uri) instead of
    floating on blank cream space -- same visual language as the segmentation
    panel on the left (roads visible, background dark), so the two panels
    read as one coherent image rather than a diagram plus an unrelated photo.
    pixel_size_m : world-units-per-pixel used to size the background image so
    it lines up with the graph's node x/y coordinates (both come from the
    same SimpleAffine transform at inference time).
    """
    pos = _get_pos(G)
    labels = _node_label_map(G)
    node_order = sorted(G.nodes())  # fixed order -- point_index from click
    # events maps back to this exact list, must never silently reorder.

    has_background = background_prob_mask is not None
    # NOTE: as of this pass, background_prob_mask is accepted for backward
    # compatibility with existing call sites but is NO LONGER rendered as an
    # actual image layer -- overlaying the real road-probability raster
    # under the graph caused enough visual/rendering problems in practice
    # (misaligned extents, washed-out heatmap glow, slow re-renders) that it
    # was pulled. The dark theme it introduced is kept unconditionally
    # though, since the heatmap glow reads far better against a dark
    # backdrop than the plain cream page background -- so callers get the
    # same dark look whether or not they still pass a mask.
    node_outline_color = "#FFFDF2"
    plot_bg = "#100D0B"

    # Heatmap "glow" layer: a blurred criticality density field radiating
    # from each node, sitting BELOW everything else. This is what makes the
    # chart read as a heatmap rather than a scatter of flat-colored dots --
    # see _criticality_density_grid for the Gaussian-splat mechanics.
    # ALWAYS emitted (even for an all-zero criticality dict, which just
    # produces an invisible zero-intensity layer) for the same reason edges
    # are always emitted in fixed tiers below: downstream code identifies
    # the node trace by a FIXED curve_number, so trace count must never
    # depend on the data.
    grid_x, grid_y, density = _criticality_density_grid(pos, criticality, node_order)
    heatmap_trace = go.Heatmap(
        x=grid_x, y=grid_y, z=density,
        colorscale=[
            [0.0, "rgba(76,163,107,0)"], [0.15, "rgba(76,163,107,0.55)"],
            [0.45, "rgba(221,161,94,0.7)"], [0.75, "rgba(188,108,37,0.8)"],
            [1.0, "rgba(122,30,30,0.9)"],
        ],
        zmin=0, zmax=1, showscale=False, hoverinfo="skip",
        zsmooth="best",
    )

    # Flood water overlay -- ALWAYS emitted (same fixed-slot reasoning as the
    # criticality heatmap above: an empty flooded_nodes set just produces an
    # invisible all-zero layer, but the SLOT in the trace list must never
    # disappear or every curve_number below it would shift under the
    # hardcoded click-handler constants in app.py). Drawn immediately after
    # the criticality glow and BEFORE edges/nodes, so it renders as a wash of
    # blue sitting under the road network -- roads and junctions stay crisp
    # on top, exactly like real floodwater over a street grid.
    _flooded = set(flooded_nodes) if flooded_nodes else set()
    flood_gx, flood_gy, flood_density = _flood_density_grid(pos, _flooded, node_order)
    water_trace = go.Heatmap(
        x=flood_gx, y=flood_gy, z=flood_density,
        colorscale=[
            [0.0, "rgba(20,90,190,0)"], [0.25, "rgba(30,110,210,0.35)"],
            [0.6, "rgba(25,95,205,0.55)"], [1.0, "rgba(10,55,150,0.75)"],
        ],
        zmin=0, zmax=1, showscale=False, hoverinfo="skip",
        zsmooth="best",
    )

    # Trace order is now FIXED as:
    #   0:      heatmap (criticality glow)
    #   1:      water_trace (flood overlay)
    #   2-5:    edge tiers (4, criticality-colored)
    #   6:      node_trace           <- node click target
    #   7:      edge_midpoint_trace  <- road click target
    #   8:      route_trace          <- selected-route highlight (always present, may be empty)
    #   9:      broken-node X markers (OPTIONAL, display-only, not a click target)
    # app.py's sandbox click handler hardcodes 6 and 7 as
    # _NODE_TRACE_CURVE_NUMBER / _EDGE_TRACE_CURVE_NUMBER -- if this ordering
    # ever changes, both constants must change with it.

    # Edges colored by criticality (max of their two endpoints -- see
    # _edge_criticality_scores), bucketed into 4 tiers so this stays ONE
    # Scatter trace per tier instead of one trace per edge (a real road
    # network can have hundreds of edges; one trace per edge would make the
    # chart sluggish and bloat the legend for no visual benefit at this
    # resolution). Same green-to-red ramp as the node markers and the new
    # heatmap glow, so all three encodings read as one consistent language.
    edge_scores = _edge_criticality_scores(G, criticality)
    _edge_tiers = [
        (0.75, "#7A1E1E", "Edge: critical (>=0.75)"),
        (0.50, "#BC6C25", "Edge: high (0.50-0.75)"),
        (0.25, "#DDA15E", "Edge: moderate (0.25-0.50)"),
        (0.00, "#4CA36B", "Edge: low (<0.25)"),
    ]
    # IMPORTANT: always emit exactly 4 edge tiers, even if a tier has no
    # matching edges (empty x/y is a valid, harmless Scatter) -- keeps every
    # downstream curve_number fixed regardless of what's in the data.
    edge_traces = []
    for i, (lo, color, tier_name) in enumerate(_edge_tiers):
        hi = 1.01 if lo == 0.75 else _edge_tiers[i - 1][0]
        tx, ty = [], []
        for u, v in G.edges():
            s = edge_scores.get((u, v), 0.0)
            if lo <= s < hi:
                tx += [pos[u][0], pos[v][0], None]
                ty += [pos[u][1], pos[v][1], None]
        width = 2.6 if lo >= 0.75 else (2.0 if lo >= 0.50 else 1.6)
        edge_traces.append(go.Scatter(
            x=tx, y=ty, mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="skip", showlegend=False, name=tier_name,
        ))
    # edge_traces now always has exactly 4 entries.

    scores = [criticality.get(n, 0.0) for n in node_order]
    node_x = [pos[n][0] for n in node_order]
    node_y = [pos[n][1] for n in node_order]
    node_text = [labels[n] for n in node_order]
    node_hover = [f"Junction {labels[n]}<br>Criticality: {criticality.get(n, 0.0):.3f}" for n in node_order]

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="middle center",
        textfont=dict(size=9, color="#FFFFFF", family="Arial Black, Arial, sans-serif"),
        hovertext=node_hover, hoverinfo="text",
        marker=dict(
            size=22, color=scores, colorscale=[
                [0.0, "#4CA36B"], [0.35, "#DDA15E"], [0.7, "#BC6C25"], [1.0, "#7A1E1E"],
            ],
            cmin=0, cmax=1, line=dict(width=1.4, color=node_outline_color),
            colorbar=dict(title="Criticality", thickness=14, len=0.6,
                          tickfont=dict(color="#000000"), title_font=dict(color="#000000")),
        ),
        showlegend=False,
    )

    traces = [heatmap_trace, water_trace, *edge_traces, node_trace]

    # ---------------------------------------------------------------------
    # Road (edge) toggling: one small marker at the MIDPOINT of every edge,
    # always emitted (same "fixed slot" principle as the edge-color tiers
    # above -- a road network's edges themselves don't change count between
    # renders, but the CLICKABILITY needs a stable marker at a stable
    # curve_number, which a variable-length line trace can't give us: a
    # multi-segment `mode="lines"` trace has no per-segment point_index, so
    # clicking a road couldn't be resolved to WHICH road without this).
    # Default appearance: small, translucent white dot -- deliberately
    # unobtrusive so it doesn't compete visually with the heatmap/edges/
    # nodes. Toggled-off roads become a bold red X, mirroring the existing
    # broken-node treatment for visual consistency.
    #
    # edge_order is recomputed the same way by the caller (sorted(G.edges())
    # on the same G) to map a click's point_index back to a real (u, v) --
    # this mirrors node_order/labels above, which app.py's click handler
    # already independently recomputes rather than having this function
    # return it. Keep both in sync if either ordering convention changes.
    _broken_edges = set()
    if broken_edges:
        for e in broken_edges:
            _broken_edges.add(tuple(e))
            _broken_edges.add(tuple(reversed(e)))

    edge_order = sorted(G.edges())
    mid_x, mid_y, mid_color, mid_symbol, mid_size, mid_hover = [], [], [], [], [], []
    for (u, v) in edge_order:
        mid_x.append((pos[u][0] + pos[v][0]) / 2)
        mid_y.append((pos[u][1] + pos[v][1]) / 2)
        is_broken = (u, v) in _broken_edges
        mid_color.append("#C0392B" if is_broken else "rgba(255,255,255,0.32)")
        mid_symbol.append("x" if is_broken else "circle")
        mid_size.append(15 if is_broken else 9)
        mid_hover.append(
            f"Road {labels[u]}-{labels[v]}<br>Criticality: {edge_scores.get((u, v), 0.0):.3f}"
            + ("<br>BROKEN" if is_broken else "")
        )
    edge_midpoint_trace = go.Scatter(
        x=mid_x, y=mid_y, mode="markers",
        marker=dict(size=mid_size, color=mid_color, symbol=mid_symbol,
                    line=dict(width=1.2, color="#FFFFFF")),
        hovertext=mid_hover, hoverinfo="text", showlegend=False,
    )
    traces.append(edge_midpoint_trace)
    # edge_midpoint_trace is therefore always at curve_number == 6
    # (1 heatmap + 4 edge tiers + 1 node_trace).

    # ---------------------------------------------------------------------
    # Selected route highlight -- ALWAYS emitted (empty when no route), same
    # fixed-slot reasoning as everything else on this chart. Drawn using
    # each edge's real traced geometry when available so it follows the
    # actual road curve rather than cutting a straight chord through
    # buildings/blocks, falling back to a straight line for edges without
    # geometry (e.g. a healing-added bridge, which legitimately IS straight).
    route_x, route_y = [], []
    route_nodes = route_nodes or []
    if len(route_nodes) >= 2:
        for a, b in zip(route_nodes[:-1], route_nodes[1:]):
            geom = None
            if G.has_edge(a, b):
                geom = G.edges[a, b].get("geometry")
            if geom and len(geom) >= 2:
                pts = geom
            else:
                pts = [(pos[a][0], pos[a][1]), (pos[b][0], pos[b][1])]
            for (x, y) in pts:
                route_x.append(x)
                route_y.append(y)
            route_x.append(None)
            route_y.append(None)
    route_trace = go.Scatter(
        x=route_x, y=route_y, mode="lines",
        line=dict(color="#00D4FF", width=4),
        hoverinfo="skip", showlegend=False, name="Selected route",
    )
    traces.append(route_trace)
    # route_trace is therefore always at curve_number == 7.

    # broken-node X markers, if any, land at curve_number == 8 (OPTIONAL --
    # only appended when broken_nodes is non-empty, but nothing else keys
    # off this trace's position since it's display-only, not a click target).
    if broken_nodes:
        bx = [pos[n][0] for n in broken_nodes if n in pos]
        by = [pos[n][1] for n in broken_nodes if n in pos]
        traces.append(go.Scatter(
            x=bx, y=by, mode="markers", marker=dict(symbol="x", size=22, color="#C0392B",
                                                     line=dict(width=2, color="#FFFFFF")),
            hoverinfo="skip", showlegend=False,
        ))

    fig = go.Figure(data=traces)

    # Deliberately NOT adding a layout image here anymore -- see the note by
    # has_background above. The heatmap glow + colored edges/nodes on a plain
    # dark background is the entire visual now; no road-mask raster underneath.
    fig.update_layout(
        plot_bgcolor=plot_bg, paper_bgcolor="#FFFDF2",
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=10, b=10),
        height=480, clickmode="event+select",
    )
    return fig


def plot_road_break(G: nx.Graph, broken_nodes: list, od_pairs: list) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("#FFFDF2")
    pos = _get_pos(G)
    labels = _node_label_map(G)

    ax1.set_facecolor("#FFFDF2")
    ax1.set_title("Before (all roads open)", fontsize=11, color="#000000", pad=8)
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color="#DDD0C8", width=1.8)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=120, node_color="#000000",
                           edgecolors="#DDD0C8", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax1, font_size=6,
                            font_color="#FFFDF2", font_weight="bold")
    for (o, d) in od_pairs[:6]:
        try:
            path = nx.shortest_path(G, o, d, weight="length")
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax1,
                                  edge_color="#4A90D9", width=3.0, alpha=0.7)
        except nx.NetworkXNoPath:
            pass
    ax1.axis("off")

    ax2.set_facecolor("#FFFDF2")
    ax2.set_title("After (selected junctions broken)", fontsize=11, color="#000000", pad=8)
    G_after = G.copy()
    G_after.remove_nodes_from(broken_nodes)
    reachable = set()
    if G_after.number_of_nodes() > 0:
        largest = max(nx.connected_components(G_after), key=len)
        reachable = largest

    still_edges = [(u, v) for u, v in G.edges()
                   if u in reachable and v in reachable
                   and u not in broken_nodes and v not in broken_nodes]
    nx.draw_networkx_edges(G, pos, edgelist=still_edges, ax=ax2,
                          edge_color="#4CA36B", width=1.8)
    cutoff_edges = [(u, v) for u, v in G.edges()
                   if (u, v) not in still_edges
                   and u not in broken_nodes and v not in broken_nodes]
    nx.draw_networkx_edges(G, pos, edgelist=cutoff_edges, ax=ax2,
                          edge_color="#9B8F82", width=1.5, style="dashed", alpha=0.6)

    surviving = [n for n in G.nodes() if n not in broken_nodes]
    surv_colors = ["#000000" if n in reachable else "#BBBBBB" for n in surviving]
    nx.draw_networkx_nodes(G, pos, nodelist=surviving, ax=ax2, node_size=120,
                           node_color=surv_colors, edgecolors="#DDD0C8", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, labels={n: labels[n] for n in surviving},
                           ax=ax2, font_size=6, font_color="#FFFDF2", font_weight="bold")

    if broken_nodes:
        xs = [pos[n][0] for n in broken_nodes if n in pos]
        ys = [pos[n][1] for n in broken_nodes if n in pos]
        ax2.scatter(xs, ys, marker="X", s=280, color="#C0392B", zorder=5,
                   edgecolors="white", linewidths=1.5)

    for (o, d) in od_pairs[:6]:
        if o in broken_nodes or d in broken_nodes:
            continue
        try:
            path = nx.shortest_path(G_after, o, d, weight="length")
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax2,
                                  edge_color="#4A90D9", width=3.0, alpha=0.7)
        except nx.NetworkXNoPath:
            if o in pos and d in pos:
                ax2.plot([pos[o][0], pos[d][0]], [pos[o][1], pos[d][1]],
                        color="#E74C3C", linewidth=1.5, linestyle=":", alpha=0.6)

    legend_elements = [
        mpatches.Patch(color="#4CA36B", label="Still connected"),
        mpatches.Patch(color="#9B8F82", label="Cut off"),
        mpatches.Patch(color="#C0392B", label="Broken junction"),
        mpatches.Patch(color="#4A90D9", label="Sample route (ok)"),
        mpatches.Patch(color="#E74C3C", label="Sample route (blocked)"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=8,
              facecolor="#FFFDF2", edgecolor="#DDD0C8")
    ax2.axis("off")
    fig.tight_layout()
    return fig


def plot_after_only(G: nx.Graph, broken_nodes: list, od_pairs: list) -> plt.Figure:
    """Single-panel 'after the break' view -- the RIGHT half of the sandbox's
    before/after pair. Deliberately separate from the interactive click-source
    chart (LEFT half, plot_network_topology_interactive with on_select) --
    keeping click-source and click-triggered-output as two genuinely
    different components avoids a feedback loop where the same chart both
    reads its own selection AND gets rebuilt with a new overlay based on
    that same selection, which is what caused the earlier glitching."""
    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor("#FFFDF2")
    ax.set_facecolor("#FFFDF2")
    pos = _get_pos(G)
    labels = _node_label_map(G)

    G_after = G.copy()
    G_after.remove_nodes_from(broken_nodes)
    reachable = set()
    if G_after.number_of_nodes() > 0:
        largest = max(nx.connected_components(G_after), key=len)
        reachable = largest

    still_edges = [(u, v) for u, v in G.edges()
                   if u in reachable and v in reachable
                   and u not in broken_nodes and v not in broken_nodes]
    nx.draw_networkx_edges(G, pos, edgelist=still_edges, ax=ax,
                          edge_color="#4CA36B", width=1.8)
    cutoff_edges = [(u, v) for u, v in G.edges()
                   if (u, v) not in still_edges
                   and u not in broken_nodes and v not in broken_nodes]
    nx.draw_networkx_edges(G, pos, edgelist=cutoff_edges, ax=ax,
                          edge_color="#9B8F82", width=1.5, style="dashed", alpha=0.6)

    surviving = [n for n in G.nodes() if n not in broken_nodes]
    surv_colors = ["#000000" if n in reachable else "#BBBBBB" for n in surviving]
    nx.draw_networkx_nodes(G, pos, nodelist=surviving, ax=ax, node_size=120,
                           node_color=surv_colors, edgecolors="#DDD0C8", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, labels={n: labels[n] for n in surviving},
                           ax=ax, font_size=6, font_color="#FFFDF2", font_weight="bold")

    if broken_nodes:
        xs = [pos[n][0] for n in broken_nodes if n in pos]
        ys = [pos[n][1] for n in broken_nodes if n in pos]
        ax.scatter(xs, ys, marker="X", s=280, color="#C0392B", zorder=5,
                  edgecolors="white", linewidths=1.5)

    for (o, d) in od_pairs[:6]:
        if o in broken_nodes or d in broken_nodes:
            continue
        try:
            path = nx.shortest_path(G_after, o, d, weight="length")
            edges = list(zip(path, path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                                  edge_color="#4A90D9", width=3.0, alpha=0.7)
        except nx.NetworkXNoPath:
            if o in pos and d in pos:
                ax.plot([pos[o][0], pos[d][0]], [pos[o][1], pos[d][1]],
                       color="#E74C3C", linewidth=1.5, linestyle=":", alpha=0.6)

    legend_elements = [
        mpatches.Patch(color="#4CA36B", label="Still connected"),
        mpatches.Patch(color="#9B8F82", label="Cut off"),
        mpatches.Patch(color="#C0392B", label="Broken junction"),
        mpatches.Patch(color="#4A90D9", label="Sample route (ok)"),
        mpatches.Patch(color="#E74C3C", label="Sample route (blocked)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7,
             facecolor="#FFFDF2", edgecolor="#DDD0C8")
    ax.set_title("After the break", fontsize=11, color="#000000", pad=8)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# STEP 1 -- front page. Upload/browse now expand INLINE on this same page
# (session_state["front_mode"]) instead of navigating to a separate
# intermediate page -- real navigation only happens once "Run analysis" or
# "Load analysis" is actually clicked.
# ---------------------------------------------------------------------------

_DECORATIVE_SVG = """
<svg width="100%" height="130" viewBox="0 0 900 130" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 1rem;">
  <path d="M 0 100 L 150 60 L 320 90 L 480 40 L 650 70 L 900 30" fill="none" stroke="#DDD0C8" stroke-width="4"/>
  <path d="M 320 90 L 340 20" fill="none" stroke="#DDD0C8" stroke-width="4"/>
  <path d="M 480 40 L 500 110" fill="none" stroke="#DDD0C8" stroke-width="4"/>
  <path d="M 150 60 L 130 10" fill="none" stroke="#DDD0C8" stroke-width="3"/>
  <circle cx="150" cy="60" r="7" fill="#BC6C25"/>
  <circle cx="320" cy="90" r="9" fill="#7A1E1E"/>
  <circle cx="480" cy="40" r="8" fill="#DDA15E"/>
  <circle cx="650" cy="70" r="6" fill="#DDD0C8"/>
  <circle cx="340" cy="20" r="5" fill="#DDD0C8"/>
  <circle cx="500" cy="110" r="5" fill="#DDD0C8"/>
</svg>
"""

if "step" not in st.session_state:
    st.session_state["step"] = "choose"
if "front_mode" not in st.session_state:
    st.session_state["front_mode"] = None

if st.session_state["step"] == "choose":
    st.markdown('<div class="team-name">MADRAS INTELLIGENCE</div>', unsafe_allow_html=True)
    st.markdown('<div class="route-title">RouteResilience</div>', unsafe_allow_html=True)
    st.markdown(_DECORATIVE_SVG, unsafe_allow_html=True)
    st.markdown(
        '<div class="hint-box">See how a road network holds up when disaster strikes -- '
        'a flood, a collapsed bridge, a blocked junction. Pick a satellite image below '
        'to get started.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        # Single unified markdown call for the whole card -- splitting this
        # across separate st.markdown/st.caption calls was what caused the
        # phantom empty white box (each Streamlit call gets its own wrapper,
        # so a div opened in one call and closed in another never actually
        # wraps anything -- it just renders as its own empty block).
        st.markdown(
            '<div class="choice-card"><h3>Upload a new image</h3>'
            '<p>Runs the full analysis pipeline from scratch. Takes a few minutes.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Upload image", key="go_upload", use_container_width=True):
            st.session_state["front_mode"] = "upload"
            st.rerun()
    with c2:
        st.markdown(
            '<div class="choice-card"><h3>Browse processed results</h3>'
            '<p>View results from images already analyzed. Instant.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Browse results", key="go_cached", use_container_width=True):
            st.session_state["front_mode"] = "cached"
            st.rerun()

    # -- Inline expansion, same page, no navigation yet --
    if st.session_state["front_mode"] == "upload":
        st.divider()
        st.subheader("Upload a new image")
        checkpoint_path = st.text_input(
            "Model checkpoint path",
            "/home/outputs/stage_c_cartosat_rgbfinal_snapshot.pth",
            help="Absolute path to the .pth checkpoint on THIS machine. "
                 "If unsure, run `find / -iname '*.pth' 2>/dev/null` in a terminal "
                 "on the same machine this Streamlit app is running on.",
        )
        uploaded = st.file_uploader("Satellite image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        source_gsd_m = st.number_input(
            "Ground sample distance (meters per pixel)",
            value=0.28, step=0.01,
            help="Cartosat-3: 0.28m, SpaceNet: 0.3m, DeepGlobe: 0.5m, Massachusetts: 1.0m",
        )
        with st.expander("Advanced parameters (optional)", expanded=False):
            hp = parameter_sliders("upload")
            st.markdown("**GPU memory safety**")
            st.caption("Lower these if inference silently fails/crashes -- especially if "
                      "something else (e.g. a training run) is using this GPU at the same time.")
            imc1, imc2 = st.columns(2)
            infer_batch_size = imc1.slider(
                "Inference batch size", 1, 8, 4, 1, key="upload_batch_size",
                help="Lower = less GPU memory per step, slower overall. "
                     "Drop to 1-2 if you suspect CUDA out-of-memory.",
            )
            infer_tile_size = imc2.select_slider(
                "Tile size (px)", options=[256, 384, 512, 640], value=512, key="upload_tile_size",
                help="Smaller tiles use less GPU memory per step but need more tiles overall.",
            )
            st.markdown("**CPU speed safety**")
            use_fast_glcm = st.checkbox(
                "Use fast GLCM proxy", value=True, key="upload_fast_glcm",
                help="The GLCM occlusion-texture channel (PathMamba's 2nd input) is computed "
                     "via skimage's graycomatrix, which is CPU-only and can be extremely slow "
                     "on a large real satellite image -- often SLOWER than the GPU inference "
                     "itself, with ZERO GPU usage the whole time (looks exactly like a hang). "
                     "The fast proxy trades a little accuracy for a much cheaper texture-variance "
                     "approximation. Leave this ON unless you have a specific reason to compare "
                     "against the exact GLCM computation.",
            )
        if uploaded is not None and st.button("Run analysis", key="run_upload"):
            tmp_path = f"/tmp/{uploaded.name}"
            with open(tmp_path, "wb") as f:
                f.write(uploaded.getbuffer())
            out_dir = f"./live_runs/{os.path.splitext(uploaded.name)[0]}"
            try:
                result = run_live_inference(
                    tmp_path, checkpoint_path, out_dir, source_gsd_m, hp,
                    batch_size=infer_batch_size, tile_size=infer_tile_size,
                    use_fast_glcm_proxy=use_fast_glcm,
                )
                st.session_state["result"] = result
                st.session_state["step"] = "results"  # real navigation, only now
                st.rerun()
            except Exception as e:
                # If this box appears, it's a catchable Python-level error --
                # the traceback below is the real cause, act on it directly.
                # If instead the page just silently resets to this same
                # screen with NO error shown at all, that's a different,
                # more serious failure mode: the Python process itself
                # likely crashed (most commonly a CUDA out-of-memory kill,
                # which terminates the whole worker rather than raising a
                # catchable exception) and got auto-restarted, wiping
                # session_state. Check `nvidia-smi` on this machine BEFORE
                # retrying -- if something else (e.g. this same repo's
                # training script) is already using most of the GPU's VRAM,
                # that contention is almost certainly the cause, not a bug
                # in this app.
                st.error(
                    "Inference failed. See the full traceback below. If you "
                    "instead saw NO error and just landed back here silently, "
                    "the Python process likely crashed (commonly CUDA OOM) -- "
                    "run `nvidia-smi` on this machine and check whether "
                    "something else (e.g. a training run) is already using "
                    "most of the GPU memory before retrying."
                )
                st.exception(e)

    elif st.session_state["front_mode"] == "cached":
        st.divider()
        st.subheader("Browse processed results")
        cache_root = "./cached_runs"
        if os.path.isdir(cache_root):
            available = sorted(
                d for d in os.listdir(cache_root)
                if os.path.isdir(os.path.join(cache_root, d))
                and _find_report_dir(os.path.join(cache_root, d))
            )
            if available:
                choice = st.selectbox("Available analyses", available)
                if st.button("Load analysis", key="load_cached"):
                    st.session_state["result"] = load_cached_result(os.path.join(cache_root, choice))
                    st.session_state["step"] = "results"  # real navigation, only now
                    st.rerun()
            else:
                st.info("No analyses found yet. Try uploading a new image instead.")
        else:
            st.info(f"Results folder not found ({cache_root}). Try uploading a new image.")

    st.stop()


# ---------------------------------------------------------------------------
# STEP 2 -- results page
# ---------------------------------------------------------------------------

if "result" not in st.session_state:
    st.session_state["step"] = "choose"
    st.rerun()

result = st.session_state["result"]
report = result["report"]
healed_graph = result["healed_graph"]
node_labels = _node_label_map(healed_graph)
reverse_labels = {v: k for k, v in node_labels.items()}

top_l, top_r = st.columns([4, 1])
with top_l:
    st.title("Analysis results")
with top_r:
    if st.button("Start over"):
        for k in ("step", "result"):
            st.session_state.pop(k, None)
        st.rerun()

# -- Model-output health band (from pipeline_diagnostics, when present) --
diag = report.get("prediction_diagnostics")
if diag:
    signal = diag.get("signal_strength", "UNKNOWN")
    st.markdown(f"**Model confidence on this image:** {_signal_badge(signal)}",
               unsafe_allow_html=True)
    if signal != "STRONG":
        warnings = diag.get("warnings") or []
        suggestions = diag.get("suggestions") or []
        with st.expander("Why this matters", expanded=(signal in ("WEAK", "NEAR_ZERO"))):
            for w in warnings:
                st.warning(w)
            for s in suggestions:
                st.info(s)

# -- Key metrics, compact row --
m1, m2, m3 = st.columns(3)
m1.metric("Road segments", report['graph_stats']['reconstructed_nodes'])
m2.metric("Gaps bridged", report["healing"]["bridges_added"],
         help="Where the road appeared broken (tree cover, shadow) but was reconnected "
              "because it clearly continues on the other side.")
ri = report.get("resilience_index")
m3.metric("Resilience score", f"{ri:.0%}" if ri is not None else "n/a",
         help="How well the network holds up if the single most critical junction is removed.")

# -- Full per-node criticality, computed live (NOT the top-10-only subset
# saved in the JSON report) -- needed so every node gets a real gradient
# color, not just the ten that happened to rank highest. Cheap: pure
# networkx/scipy on an already-built graph, no GPU, same pattern already
# used by the re-tuning panel and the sandbox below.
#
# composite_criticality already returns all four individual metrics
# alongside the composite -- keeping them all gives a "colour by" toggle
# for free, no extra compute.
_crit_all = _composite_criticality_cached(healed_graph)
_crit_full = _crit_all.get("composite", {})  # kept: sandbox code below refers to this name

_metric_options = {
    "Composite (all 4 combined)": _crit_all.get("composite", {}),
    "Betweenness centrality": _crit_all.get("betweenness", {}),
    "K-core number": _crit_all.get("k_core", {}),
    "Current-flow betweenness": _crit_all.get("cfbc", {}),
    "Alpha centrality": _crit_all.get("alpha", {}),
}
_metric_help = (
    "Composite blends all four -- best default single view. "
    "Betweenness: junctions on many shortest paths (evacuation-route value). "
    "K-core: junctions in the densest sub-network (structurally central). "
    "Current-flow BC: alternative-path resilience (rerouting under stress). "
    "Alpha centrality: influence through neighbors-of-neighbors."
)

if "clicked_broken_nodes" not in st.session_state:
    _default_top = report.get("top_criticality_node")
    st.session_state["clicked_broken_nodes"] = {_default_top} if _default_top is not None else set()

if "clicked_broken_edges" not in st.session_state:
    st.session_state["clicked_broken_edges"] = set()

# -- Side by side: pipeline visualization (left) + STATIC gradient-colored
# network graph (right, no clicking here -- clicking happens in the
# What-if sandbox below instead) -- both height-constrained to fit together.
st.divider()
st.subheader("Pipeline stages + network topology")
st.caption("Input tile -> model's predicted road probability -> final healed network -> "
          "criticality heatmap, all in one row.")

gcol1, gcol2, gcol3, topo_col = st.columns([1, 1, 1, 3], gap="medium")
with gcol1:
    st.markdown("**Input**")
    if result.get("input_image"):
        st.image(result["input_image"], use_container_width=True)
    else:
        # Older runs (before this feature existed) never saved a copy of
        # the raw tile -- graceful fallback rather than a crash, since
        # nothing about the rest of the report depends on this panel.
        st.info("Not saved for this run. Reprocess with the "
               "updated run_real_inference.py to see it here.")
with gcol2:
    st.markdown("**Predicted (prob mask)**")
    if result.get("prob_mask") is not None:
        st.image(np.clip(result["prob_mask"], 0.0, 1.0),
                 use_container_width=True, clamp=True)
    else:
        st.info("No probability mask available for this run.")
with gcol3:
    st.markdown("**Healed network**")
    if result["viz"]:
        st.image(result["viz"], use_container_width=True)
    else:
        st.info("No visualization available for this run.")
with topo_col:
    st.markdown("**Network topology (criticality heatmap)**")
    _selected_metric_name = st.radio(
        "Colour by",
        options=list(_metric_options.keys()),
        index=0, horizontal=True, key="topology_metric_choice",
        help=_metric_help,
    )
    _selected_crit = _metric_options[_selected_metric_name]
    st.caption("Darker red = more critical junction or road segment (each metric "
              "normalized to 0-1; segment color = the higher of its two endpoints).")
    static_topo_fig = plot_network_topology_interactive(
        healed_graph, _selected_crit, set(),
    )
    static_topo_fig.update_layout(dragmode=False)
    st.plotly_chart(
        static_topo_fig, use_container_width=True, key="static_topology_display",
        config={"displayModeBar": False, "staticPlot": False},
    )

# -- All scenarios, not just the single-top one --
st.divider()
st.subheader("Disaster scenario comparison")
st.caption("How the network responds to different scales of disruption.")
scenarios = report.get("scenarios", {})
scenario_rows = []
for name, sc in scenarios.items():
    if name.startswith("_") or not isinstance(sc, dict) or "resilience_index" not in sc:
        continue
    scenario_rows.append({
        "Scenario": name.replace("_", " ").title(),
        "Junctions removed": sc.get("n_disabled", "?"),
        "Resilience": f"{sc['resilience_index']:.1%}",
        "Routes blocked": sc.get("od_pairs_disconnected_after", "?"),
        "Routes degraded": sc.get("od_pairs_degraded_after", "?"),
    })
if scenario_rows:
    st.table(pd.DataFrame(scenario_rows))
else:
    st.caption("Scenario breakdown not available for this run.")

st.divider()
st.subheader("Most critical junctions")
st.caption("Removing any of these has the largest impact on network connectivity.")
crit_data = report.get("criticality_top10", [])
if crit_data:
    crit_rows = [{"Junction": node_labels.get(e.get("node"), str(e.get("node"))),
                 "Criticality score": f"{e.get('score', 0):.3f}"} for e in crit_data]
    st.table(pd.DataFrame(crit_rows))

# ---------------------------------------------------------------------------
# Live re-tuning panel -- works whenever both masks were saved
# ---------------------------------------------------------------------------

if result.get("prob_mask") is not None and result.get("confidence_mask") is not None:
    with st.expander("Adjust and re-run (fast, no GPU needed)", expanded=False):
        st.caption("Reruns road-tracing and bridging on the same image with different "
                  "settings. Takes seconds, not minutes.")
        prev_hp = report.get("retuned_parameters") or {}
        hp2 = parameter_sliders("retune", defaults=prev_hp)
        pixel_size_m = st.number_input("Ground sample distance (m/px)", value=0.28,
                                       step=0.01, key="retune_pixel_size_m")
        if st.button("Apply and re-run"):
            with st.spinner("Re-running..."):
                new_result = retune_graph_from_saved_masks(
                    result["report_dir"], pixel_size_m=pixel_size_m, **hp2,
                )
                st.session_state["result"] = new_result
                st.success("Updated.")
                st.rerun()

# ---------------------------------------------------------------------------
# Disaster sandbox -- driven directly by clicks on the topology graph above,
# no separate selection widget needed anymore.
# ---------------------------------------------------------------------------

@st.fragment
def _render_whatif_sandbox(healed_graph, node_labels, _selected_crit):
    """Everything below is click-driven (toggle a junction/road, pick an
    origin/destination) and reruns on every interaction. Scoped as its own
    st.fragment (Streamlit >= 1.37) so a click only re-executes THIS function,
    not the whole page -- the pipeline images, metrics, scenario table, and
    criticality table above no longer redraw on every toggle. Internal
    reruns use st.rerun(scope="fragment") for the same reason: a plain
    st.rerun() here would still force a full-page rerun and defeat the
    point of the fragment.
    """
    st.divider()
    st.subheader("What-if sandbox")
    st.markdown(
        '<div class="hint-box">Click a junction to simulate removing it, or click a small dot '
        'along a road to simulate blocking that road instead -- shift-click or drag-select for '
        'more than one of either. Pick an origin and destination below to see the current route '
        'highlighted live, including how it reroutes or breaks as you toggle junctions/roads.</div>',
        unsafe_allow_html=True,
    )

    _sandbox_node_order = sorted(healed_graph.nodes())
    _sandbox_edge_order = sorted(healed_graph.edges())

    # ---------------------------------------------------------------------------
    # Origin/destination picker -- placed BEFORE the chart since the route it
    # defines gets drawn ON that chart, live, reflecting whatever junctions/
    # roads are currently toggled off.
    # ---------------------------------------------------------------------------

    st.markdown("**Route: pick two junctions**")
    _route_labels = [node_labels[n] for n in _sandbox_node_order]
    _label_to_node = {node_labels[n]: n for n in _sandbox_node_order}

    rc1, rc2 = st.columns(2)
    _origin_label = rc1.selectbox("Origin junction", options=_route_labels,
                                  index=0, key="route_origin_label")
    _dest_default_idx = min(1, len(_route_labels) - 1)
    _dest_label = rc2.selectbox("Destination junction", options=_route_labels,
                                index=_dest_default_idx, key="route_dest_label")
    _origin_node = _label_to_node[_origin_label]
    _dest_node = _label_to_node[_dest_label]

    # ---------------------------------------------------------------------------
    # Flood scenario -- algorithmically decides which junctions go underwater
    # over time (see _flood_progression) and auto-toggles them into the SAME
    # clicked_broken_nodes state a manual click would set. Because it just
    # writes to that one shared piece of state, every panel below (chart,
    # "currently broken" caption, before/after metrics) reacts exactly as it
    # already does to a manual toggle -- no separate rendering path needed.
    # ---------------------------------------------------------------------------

    st.markdown("**Simulate a flood**")
    if st.session_state.get("flood_origin_label") not in _route_labels:
        st.session_state["flood_origin_label"] = _origin_label
    if "flood_step" not in st.session_state:
        st.session_state["flood_step"] = 0
    if "flood_playing" not in st.session_state:
        st.session_state["flood_playing"] = False
    if "flood_n_steps" not in st.session_state:
        st.session_state["flood_n_steps"] = 8

    fc1, fc2, fc3, fc4 = st.columns([2, 1, 1, 1])
    fc1.selectbox("Flood origin junction", options=_route_labels, key="flood_origin_label")
    fc2.slider("Steps", min_value=3, max_value=15, key="flood_n_steps")
    _flood_n_steps = st.session_state["flood_n_steps"]

    if fc3.button("🎲 Randomize origin"):
        st.session_state["flood_origin_label"] = random.choice(_route_labels)
        st.session_state["flood_step"] = 0
        st.rerun(scope="fragment")

    if fc4.button("⏸ Pause" if st.session_state["flood_playing"] else "▶ Play flood"):
        st.session_state["flood_playing"] = not st.session_state["flood_playing"]
        if st.session_state["flood_playing"] and st.session_state["flood_step"] >= _flood_n_steps:
            st.session_state["flood_step"] = 0  # replay from the start once finished
        st.rerun(scope="fragment")

    _flood_origin_node = _label_to_node[st.session_state["flood_origin_label"]]
    _flood_steps = _flood_progression_cached(healed_graph, _flood_origin_node, n_steps=_flood_n_steps)

    # Clamp BEFORE creating the keyed slider below -- Streamlit errors if a
    # widget's session_state value falls outside the min/max it's created
    # with (can happen after adjusting "Steps" mid-playback).
    if st.session_state["flood_step"] > _flood_n_steps:
        st.session_state["flood_step"] = _flood_n_steps

    st.slider(
        "Flood progress (step)", min_value=0, max_value=_flood_n_steps, key="flood_step",
        help="0 = dry network. Drag to scrub through the flood manually, or use Play.",
    )

    _current_flood_nodes = set()
    if st.session_state["flood_step"] > 0:
        _flooded_now = _flood_steps[st.session_state["flood_step"] - 1]
        _current_flood_nodes = set(_flooded_now)
        st.session_state["clicked_broken_nodes"] = set(_flooded_now)
        st.session_state["clicked_broken_edges"] = set()
        st.caption(
            f"Flood step {st.session_state['flood_step']}/{_flood_n_steps} -- "
            f"{len(_flooded_now)} junction(s) underwater, spreading from junction "
            f"{st.session_state['flood_origin_label']}."
        )
        if st.button("Reset flood", key="flood_reset"):
            st.session_state["flood_step"] = 0
            st.session_state["flood_playing"] = False
            st.session_state["clicked_broken_nodes"] = set()
            st.session_state["clicked_broken_edges"] = set()
            st.rerun(scope="fragment")

    # ---------------------------------------------------------------------------
    # Current disruption state (read BEFORE building the chart, so the chart's
    # road markers and route highlight reflect it immediately -- this is normal
    # state-driven redraw from a PREVIOUS render's click, not the circular
    # same-render feedback loop the "broken_nodes=set() always" note below is
    # actually about).
    # ---------------------------------------------------------------------------

    broken_nodes = list(st.session_state["clicked_broken_nodes"])
    broken_edges = set(st.session_state["clicked_broken_edges"])

    od_pairs = _sample_od_pairs_cached(healed_graph, max_pairs=40, seed=0)
    _analysis = _disruption_analysis_cached(
        healed_graph, frozenset(broken_nodes), frozenset(broken_edges),
        _origin_node, _dest_node, tuple(od_pairs),
    )
    route_nodes = _analysis["route_nodes"]

    sandbox_left, sandbox_right = st.columns(2, gap="medium")

    with sandbox_left:
        st.caption("Click to select (before)")
        # NOTE on broken_nodes=set() here specifically: this chart's OWN node
        # coloring never reflects its own click output within the same render
        # -- that specific circular pattern (chart reads its own output, then
        # gets redrawn from that output, on every rerun) caused real glitching
        # before and is deliberately avoided by always passing an empty node set
        # here. broken_edges and route_nodes below are NOT that pattern -- they
        # come from session_state set on a PRIOR render's click, which is
        # ordinary, safe Streamlit state-driven redraw, so the road markers and
        # route highlight ARE allowed to reflect current state.
        sandbox_fig = plot_network_topology_interactive(
            healed_graph, _selected_crit, set(),
            broken_edges=broken_edges,
            route_nodes=route_nodes,
            flooded_nodes=_current_flood_nodes,
        )
        sandbox_click = st.plotly_chart(sandbox_fig, on_select="rerun", key="sandbox_topology_chart",
                                        use_container_width=True)
        if sandbox_click and sandbox_click.get("selection", {}).get("points"):
            # Node trace and edge-midpoint trace are always at curve_number 6
            # and 7 respectively (shifted from 5/6 by the water-overlay trace
            # inserted at slot 1) -- see plot_network_topology_interactive's
            # fixed trace-ordering comment. A single click/box-select event can
            # contain points from BOTH traces at once (e.g. a drag-select
            # spanning a junction and a nearby road marker), so both are
            # checked independently against the SAME points list, not as an
            # either/or.
            _NODE_TRACE_CURVE_NUMBER = 6
            _EDGE_TRACE_CURVE_NUMBER = 7
            pts = sandbox_click["selection"]["points"]

            new_node_selection = {
                _sandbox_node_order[pt["point_index"]]
                for pt in pts
                if pt.get("curve_number") == _NODE_TRACE_CURVE_NUMBER and pt.get("point_index") is not None
                and pt["point_index"] < len(_sandbox_node_order)
            }
            if new_node_selection:
                st.session_state["clicked_broken_nodes"] = new_node_selection

            new_edge_selection = {
                _sandbox_edge_order[pt["point_index"]]
                for pt in pts
                if pt.get("curve_number") == _EDGE_TRACE_CURVE_NUMBER and pt.get("point_index") is not None
                and pt["point_index"] < len(_sandbox_edge_order)
            }
            if new_edge_selection:
                st.session_state["clicked_broken_edges"] = new_edge_selection

            if new_node_selection or new_edge_selection:
                st.rerun(scope="fragment")

    # Re-read after the possible rerun above, so the rest of this section always
    # reflects the LATEST click, not a stale value captured before the rerun.
    broken_nodes = list(st.session_state["clicked_broken_nodes"])
    broken_edges = set(st.session_state["clicked_broken_edges"])
    has_disruption = bool(broken_nodes) or bool(broken_edges)

    if has_disruption:
        node_labels_str = ", ".join(sorted((node_labels.get(n, str(n)) for n in broken_nodes), key=int))
        edge_labels_str = ", ".join(
            f"{node_labels.get(u, u)}-{node_labels.get(v, v)}" for u, v in sorted(broken_edges)
        )
        with sandbox_left:
            parts = []
            if broken_nodes:
                parts.append(f"junction(s) {node_labels_str}")
            if broken_edges:
                parts.append(f"road(s) {edge_labels_str}")
            st.caption(f"Currently broken: {' and '.join(parts)}")
            if st.button("Clear selection"):
                st.session_state["clicked_broken_nodes"] = set()
                st.session_state["clicked_broken_edges"] = set()
                st.rerun(scope="fragment")

        _analysis = _disruption_analysis_cached(
            healed_graph, frozenset(broken_nodes), frozenset(broken_edges),
            _origin_node, _dest_node, tuple(od_pairs),
        )

        with sandbox_right:
            st.caption("Result (after)")
            st.caption("Visual reflects removed junctions only; the metrics below "
                      "include both removed junctions AND removed roads.")
            after_fig = plot_after_only(healed_graph, broken_nodes, od_pairs)
            st.pyplot(after_fig, use_container_width=True)
            plt.close(after_fig)

        lcc_frac = _analysis["lcc_frac"]
        unreachable, total_pairs = _analysis["unreachable"], _analysis["total_pairs"]
        s1, s2 = st.columns(2)
        s1.metric("Network intact", f"{lcc_frac:.0%}")
        s2.metric("Routes blocked", f"{unreachable} / {total_pairs}")

        # Route distance delta for the specific origin/destination chosen above.
        # This is deliberately a DISTANCE metric, not a travel-TIME metric -- the
        # graph has no speed attribute per edge (only `length` in meters), so any
        # time number would need an assumed speed that isn't measured anywhere in
        # the pipeline. % increase in distance and % increase in time would be
        # IDENTICAL under a flat assumed speed anyway, so nothing informational
        # is lost by reporting distance directly.
        st.markdown("**Selected route: before vs after**")
        if not _analysis["origin_in_after"] or not _analysis["dest_in_after"]:
            missing = _origin_label if not _analysis["origin_in_after"] else _dest_label
            st.warning(f"Junction {missing} is itself one of the disabled junctions -- "
                      "no route is possible.")
        else:
            before_len = _analysis["before_len"]
            after_len = _analysis["after_len"]

            rd1, rd2, rd3 = st.columns(3)
            rd1.metric("Distance before", f"{before_len:.0f} m" if before_len is not None else "n/a")
            if after_len is None:
                rd2.metric("Distance after", "No route")
                rd3.metric("Change", "Fully cut off")
            elif before_len is None:
                rd2.metric("Distance after", f"{after_len:.0f} m")
                rd3.metric("Change", "n/a (no baseline route)")
            else:
                pct = (after_len - before_len) / before_len * 100 if before_len > 0 else 0.0
                rd2.metric("Distance after", f"{after_len:.0f} m")
                rd3.metric("Change", f"+{pct:.0f}%" if pct >= 0 else f"{pct:.0f}%",
                          delta=f"{after_len - before_len:+.0f} m")
    else:
        with sandbox_right:
            st.caption("Result (after)")
            st.info("Click one or more junctions or roads on the left to simulate a disruption.")

        st.markdown("**Selected route: baseline**")
        try:
            baseline_len = nx.shortest_path_length(
                healed_graph, _origin_node, _dest_node, weight="length"
            )
            st.metric("Distance (no disruption)", f"{baseline_len:.0f} m")
        except nx.NetworkXNoPath:
            st.metric("Distance (no disruption)", "No route exists")

    # ---------------------------------------------------------------------------
    # Flood autoplay -- placed LAST so every panel above (chart, metrics,
    # before/after) has already fully rendered for the CURRENT step before
    # this schedules the next one. st.rerun(scope="fragment") halts this
    # function immediately, so nothing after this block would run anyway --
    # the already-rendered elements stay visible until the next frame
    # replaces them, which is what produces the "toggles itself" animation.
    # ---------------------------------------------------------------------------
    if st.session_state["flood_playing"]:
        if st.session_state["flood_step"] < _flood_n_steps:
            time.sleep(0.6)
            st.session_state["flood_step"] += 1
            st.rerun(scope="fragment")
        else:
            st.session_state["flood_playing"] = False


_render_whatif_sandbox(healed_graph, node_labels, _selected_crit)
