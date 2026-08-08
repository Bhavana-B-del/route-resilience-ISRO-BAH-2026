"""
pipeline.py
===========
Orchestrates Stages 5-9 of the architecture (Z-Level -> Healing -> Centrality
-> Simulation -> Dashboard-ready output) behind one function, `run_pipeline()`.

Takes a `RoadPrediction` (model_io.py) and a broken graph as input -- the
broken graph now comes from either synthetic_occlusion.py (synthetic testing)
or skeletonization.graph_from_prediction() (real PathMamba output, built and
proven this session via run_real_inference.py). Zero change needed to this
file either way; see run_pipeline()'s docstring for both call sites.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .centrality import composite_criticality
from .healing import HealingReport, heal
from .model_io import RoadPrediction
from .resilience_index import ResilienceIndexResult, compute_resilience_index
from .simulation import SimulationResult, run_scenario
from .zlevel import assign_z_levels


@dataclass
class PipelineResult:
    healed_graph: nx.Graph
    healing_report: HealingReport
    criticality: dict
    scenarios: dict[str, SimulationResult]
    resilience: dict[str, ResilienceIndexResult]


def run_pipeline(
    broken_graph: nx.Graph,
    prediction: RoadPrediction,
    disaster_scenarios: dict[str, list],
    od_pairs: list[tuple],
    population_attr: str = "population",
    heal_max_iterations: int = 2,
) -> PipelineResult:
    """
    Real call site, now that PathMamba is trained, looks like:

        prediction = model_io.load_pathmamba_and_infer(
            "stage_b_checkpoint.pth", tile_array, PathMamba,
            transform=RasterioAffineAdapter(tile.transform), crs="EPSG:32644",
            source_gsd_m=0.28,
        )
        broken_graph = skeletonization.graph_from_prediction(prediction)  # built,
            # see skeletonization.py -- vectorizes prediction.prob_mask into a
            # networkx graph with the same node/edge attribute schema this
            # module expects (x, y, length, bridge, layer, population).
        result = run_pipeline(broken_graph, prediction, scenarios, od_pairs)

    NOTE: run_real_inference.py's run_one_tile() inlines this same flow
    directly rather than calling run_pipeline() -- both exist; run_pipeline()
    is the documented, reusable entry point, run_one_tile() is the proven,
    battle-tested CLI path actually used for every real test this session.
    Keep both in sync if either changes.

    `disaster_scenarios` : {scenario_name: [node_ids_to_disable]}
    `od_pairs`           : [(origin_node, dest_node), ...] fixed across scenarios
                           so resilience scores are comparable to each other
    """
    z_leveled = assign_z_levels(broken_graph)
    healing_report = heal(z_leveled, prediction, max_iterations=heal_max_iterations)
    healed = healing_report.healed_graph

    criticality = composite_criticality(healed)

    scenarios: dict[str, SimulationResult] = {}
    resilience: dict[str, ResilienceIndexResult] = {}
    for name, disabled_nodes in disaster_scenarios.items():
        sim_result = run_scenario(healed, disabled_nodes, od_pairs, population_attr)
        scenarios[name] = sim_result

        from .simulation import shortest_path_deltas  # local import: avoids a cycle at module load

        deltas = shortest_path_deltas(healed, sim_result.remaining_graph, od_pairs)
        resilience[name] = compute_resilience_index(deltas["baseline_lengths"], deltas["perturbed_lengths"])

    return PipelineResult(
        healed_graph=healed,
        healing_report=healing_report,
        criticality=criticality,
        scenarios=scenarios,
        resilience=resilience,
    )
