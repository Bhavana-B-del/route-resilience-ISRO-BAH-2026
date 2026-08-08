"""
run_demo.py
===========
Runs the FULL pipeline end-to-end on synthetic data only. This exists to prove
every module is wired correctly and bug-free RIGHT NOW, with zero dependency on
the PathMamba training finishing. When the trained checkpoint exists, only the
first ~15 lines of `main()` change (swap synthetic graph/prediction for real
ones) -- everything after that line is identical.

Run with:  python -m route_resilience.run_demo
"""

from __future__ import annotations

import networkx as nx

from .synthetic_occlusion import (
    build_synthetic_city_graph,
    inject_occlusion,
    rasterize_graph_to_prediction,
)
from .healing import score_against_ground_truth
from .pipeline import run_pipeline


def main() -> None:
    # ------------------------------------------------------------------
    # THIS BLOCK is what gets replaced once PathMamba training finishes.
    # See pipeline.run_pipeline()'s docstring for the real-data equivalent.
    # cell_size_m is kept below the deck's 50m gap-gate threshold on purpose: a
    # single occlusion patch (canopy/shadow) breaks roads at the scale of tens
    # of meters, not full city-block spacing -- using 90m blocks here (a more
    # "realistic" city scale) made every cross-component gap exceed 50m and
    # healing never fired, which is what caught this in the first place.
    G_full = build_synthetic_city_graph(rows=6, cols=6, cell_size_m=30.0, seed=42)
    G_broken, gap_records = inject_occlusion(G_full, cluster_size=4, seed=7)
    prediction = rasterize_graph_to_prediction(G_full, G_broken, gap_records, pixel_size_m=2.0)
    # ------------------------------------------------------------------

    print(f"Ground-truth graph : {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")
    print(f"Broken graph        : {nx.number_connected_components(G_broken)} connected components "
          f"({G_broken.number_of_edges()} edges remain of {G_full.number_of_edges()})")
    print(f"Injected gaps       : {len(gap_records)}")
    print(f"Prediction raster   : prob_mask {prediction.shape}, is_synthetic={prediction.is_synthetic}")
    print()

    all_nodes = list(G_full.nodes())
    disaster_scenarios = {
        "flood_north_edge": [n for n in all_nodes if G_full.nodes[n]["y"] < 90],
        "single_hub_failure": [max(all_nodes, key=lambda n: G_full.degree[n])],
    }
    od_pairs = [(all_nodes[0], all_nodes[-1]), (all_nodes[5], all_nodes[-6])]

    result = run_pipeline(G_broken, prediction, disaster_scenarios, od_pairs)

    print("=== Healing ===")
    print(result.healing_report.summary())
    gt_score = score_against_ground_truth(result.healing_report, gap_records)
    print("Ground-truth check (synthetic-only sanity check):", gt_score)
    print()

    print("=== Criticality (top 5 nodes by composite score) ===")
    ranked = sorted(result.criticality["composite"].items(), key=lambda kv: kv[1], reverse=True)
    for node, score in ranked[:5]:
        print(f"  node {node}: composite={score:.3f}")
    print()

    print("=== Disaster scenarios ===")
    for name, sim in result.scenarios.items():
        print(f"  [{name}]")
        print(f"    disabled nodes            : {sim.disabled_nodes}")
        print(f"    largest component         : {sim.largest_component_size}/"
              f"{G_full.number_of_nodes()} nodes ({sim.largest_component_fraction:.1%})")
        print(f"    population isolated       : {sim.isolated_population}/{sim.total_population} "
              f"({sim.isolated_population_fraction:.1%})")
        print(f"    unreachable OD pairs       : {sim.unreachable_od_pairs}/{sim.total_od_pairs}")
        print(f"    resilience index           : {result.resilience[name].index:.3f}")
    print()

    print("=== Scenario comparison (worst first) ===")
    from .resilience_index import compare_scenarios
    print(compare_scenarios(result.resilience))


if __name__ == "__main__":
    main()
