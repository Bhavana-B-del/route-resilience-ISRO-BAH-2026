"""
scenarios.py
============
Multi-scenario disaster harness. Replaces the "disable the single top-criticality
node" placeholder every runner script uses with a small named set of standard
scenarios that produce a comparable resilience story:

  * single_top_node    -- disable the highest composite-criticality node (the old
                          default; kept so numbers stay comparable across runs).
  * top_5_nodes        -- disable the top-5 composite-criticality nodes together.
                          Answers "what if the five worst points of failure all
                          go at once."
  * random_10_percent  -- disable a random 10% of nodes (seeded for reproducibility).
                          A generic stress test that doesn't privilege any
                          particular topology assumption -- useful as a floor.

Each scenario returns the same shape of dict, so the runners can loop and dump
them side by side. When a WorldPopRaster is supplied, population impact is added
per scenario too.

This module is deliberately dependency-light: takes an already-computed graph +
criticality dict + optional worldpop raster, returns plain dicts. No I/O, no
argparse, no plots.
"""

from __future__ import annotations

import random
from typing import Optional

import networkx as nx

from .simulation import run_scenario, shortest_path_deltas
from .resilience_index import compute_resilience_index


def _sample_od_pairs(G: nx.Graph, max_pairs: int = 40, seed: int = 0) -> list[tuple]:
    """Pick a small, deterministic set of origin/destination pairs for the
    shortest-path deltas. Uses degree-1 nodes when available (real trip
    endpoints in a road graph), otherwise random node pairs.

    The old runners used just (nodes[0], nodes[-1]) -- a single pair, which
    made the resilience index quantize to 0.0 or 1.0. Sampling ~40 pairs gives
    a continuous score that actually responds to the scenario.
    """
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return []
    endpoints = [n for n in nodes if G.degree[n] == 1]
    rng = random.Random(seed)
    if len(endpoints) >= 4:
        # Prefer real trip endpoints -- pair them up randomly
        rng.shuffle(endpoints)
        pairs = []
        for i in range(0, len(endpoints) - 1, 2):
            pairs.append((endpoints[i], endpoints[i + 1]))
            if len(pairs) >= max_pairs:
                break
        return pairs
    # Fallback: random node pairs
    pairs = []
    for _ in range(max_pairs):
        u, v = rng.sample(nodes, 2)
        pairs.append((u, v))
    return pairs


def _run_one_scenario(
    G: nx.Graph,
    disabled_nodes: list,
    od_pairs: list,
    worldpop=None,
) -> dict:
    """Core: apply a scenario, compute the resilience index and (optionally)
    population impact. Returns a dict that can be dropped straight into JSON.
    """
    if not disabled_nodes:
        return {
            "disabled_nodes": [],
            "n_disabled": 0,
            "resilience_index": None,
            "od_pairs_evaluated": 0,
            "population_impact": None,
        }

    result = run_scenario(G, disabled_nodes, od_pairs)
    deltas = shortest_path_deltas(G, result.remaining_graph, od_pairs)
    ri = compute_resilience_index(deltas["baseline_lengths"], deltas["perturbed_lengths"])

    out = {
        "disabled_nodes": list(disabled_nodes),
        "n_disabled": len(disabled_nodes),
        "resilience_index": round(float(ri.index), 4),
        "od_pairs_evaluated": int(ri.total_pairs),
        "od_pairs_disconnected_after": int(ri.fully_cut_off_pairs),
        "od_pairs_degraded_after": int(ri.degraded_pairs),
        "population_impact": None,
    }

    if worldpop is not None:
        try:
            from .population_impact import isochrone_population_impact
            imp = isochrone_population_impact(G, result.remaining_graph, worldpop)
            out["population_impact"] = {
                "total_population": round(imp.total_population, 1),
                "main_component_population": round(imp.main_component_population, 1),
                "isolated_component_population": round(imp.isolated_component_population, 1),
                "disabled_area_population": round(imp.disabled_area_population, 1),
                "fraction_isolated_or_disabled": round(imp.fraction_isolated_or_disabled, 4),
                "n_surviving_components": imp.n_components,
            }
        except Exception as e:
            out["population_impact"] = {"error": f"{type(e).__name__}: {e}"}

    return out


def run_default_scenarios(
    G: nx.Graph,
    criticality: dict,
    worldpop=None,
    random_fraction: float = 0.10,
    seed: int = 0,
    flood_polygon=None,
    flood_target_crs=None,
    flood_source_crs=None,
) -> dict:
    """Run the three standard scenarios against a healed graph.

    Parameters
    ----------
    G : the healed graph (post skeletonization + healing).
    criticality : `composite_criticality(G)` output. Must have a
                  'composite' sub-dict mapping node -> [0,1] score.
    worldpop : optional WorldPopRaster (ArrayWorldPopRaster or
               RasterioWorldPopRaster). When supplied, per-scenario
               population impact is included.
    random_fraction : fraction of nodes to disable in random_10_percent
                      (0.10 by default; name is illustrative).
    seed : reproducibility.
    flood_polygon : optional flood-extent polygon (path to GeoJSON, GeoJSON
                    dict, or list of (x,y) tuples in the graph's coord frame).
                    When supplied, a fourth 'flood' scenario is added.

    Returns
    -------
    dict of {scenario_name: scenario_result_dict}. See _run_one_scenario for
    the per-scenario shape.
    """
    composite = criticality.get("composite", {}) if criticality else {}
    if not composite:
        return {"note": "No criticality scores available; scenarios skipped."}

    od_pairs = _sample_od_pairs(G, max_pairs=40, seed=seed)

    ranked = sorted(composite.items(), key=lambda kv: kv[1], reverse=True)
    top_1 = [ranked[0][0]]
    top_5 = [n for n, _ in ranked[:5]]

    all_nodes = list(G.nodes())
    rng = random.Random(seed)
    k = max(1, int(round(len(all_nodes) * random_fraction)))
    random_nodes = rng.sample(all_nodes, k)

    out = {
        "single_top_node": _run_one_scenario(G, top_1, od_pairs, worldpop),
        "top_5_nodes": _run_one_scenario(G, top_5, od_pairs, worldpop),
        "random_10_percent": _run_one_scenario(G, random_nodes, od_pairs, worldpop),
    }

    if flood_polygon is not None:
        try:
            from .flood_simulation import flood_scenario
            out["flood"] = flood_scenario(
                G, flood_polygon, od_pairs=od_pairs, worldpop=worldpop,
                target_crs=flood_target_crs, source_crs=flood_source_crs,
            )
        except Exception as e:
            out["flood"] = {"error": f"{type(e).__name__}: {e}"}

    out["_meta"] = {
        "n_od_pairs_sampled": len(od_pairs),
        "random_fraction": random_fraction,
        "seed": seed,
        "flood_polygon_provided": flood_polygon is not None,
    }
    return out


def load_worldpop_if_available(path: Optional[str]):
    """Convenience: return a RasterioWorldPopRaster when `path` is given and
    rasterio is installed; otherwise None. Runner scripts call this so they
    don't have to import rasterio themselves.
    """
    if not path:
        return None
    try:
        from .population_impact import RasterioWorldPopRaster
        return RasterioWorldPopRaster(path)
    except Exception as e:
        print(f"[warn] could not load WorldPop raster '{path}': "
              f"{type(e).__name__}: {e} -- population impact will be skipped.")
        return None
