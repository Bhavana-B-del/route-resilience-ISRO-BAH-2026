"""
simulation.py
=============
Stage 8 (Disaster Simulation) of the pipeline: toggles nodes off (the deck's
`is_disabled` flag) and recomputes network state instantly -- no graph rebuild,
matching the "Computational Scalability" design decision.

Population impact uses a `population` node attribute as the WorldPop-100m
stand-in (synthetic_occlusion.py assigns one per node already). Swapping in a
real WorldPop raster later means replacing how that attribute gets populated
during graph construction -- this module doesn't change.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx


@dataclass
class SimulationResult:
    disabled_nodes: list
    remaining_graph: nx.Graph
    largest_component_size: int
    largest_component_fraction: float
    isolated_population: int
    total_population: int
    isolated_population_fraction: float
    unreachable_od_pairs: int
    total_od_pairs: int


def disable_nodes(G: nx.Graph, nodes_to_disable: list) -> nx.Graph:
    """Returns a copy with the given nodes removed -- the `is_disabled` toggle
    from the deck, implemented as an actual node removal for simplicity. (A
    live dashboard would instead flag `is_disabled=True` and filter at query
    time for true instant-rollback; this function is the "what would that
    query return" logic underneath either implementation.)
    """
    remaining = G.copy()
    remaining.remove_nodes_from(nodes_to_disable)
    return remaining


def largest_connected_component_stats(G: nx.Graph, original_node_count: int) -> tuple[int, float]:
    if G.number_of_nodes() == 0:
        return 0, 0.0
    largest = max(nx.connected_components(G), key=len)
    return len(largest), len(largest) / original_node_count


def population_impact(
    G_original: nx.Graph, G_remaining: nx.Graph, population_attr: str = "population"
) -> tuple[int, int, float]:
    """Population attached to nodes that are no longer in the same component
    as the single largest surviving component (i.e., cut off from the main
    network, not merely removed by the disaster itself)."""
    total_population = sum(d.get(population_attr, 0) for _, d in G_original.nodes(data=True))

    if G_remaining.number_of_nodes() == 0:
        return total_population, total_population, 1.0

    components = list(nx.connected_components(G_remaining))
    largest = max(components, key=len)
    isolated_population = 0
    for comp in components:
        if comp is largest:
            continue
        isolated_population += sum(
            G_remaining.nodes[n].get(population_attr, 0) for n in comp
        )
    # Population on directly-disabled nodes counts as affected too.
    disabled_nodes = set(G_original.nodes()) - set(G_remaining.nodes())
    isolated_population += sum(
        G_original.nodes[n].get(population_attr, 0) for n in disabled_nodes
    )

    fraction = isolated_population / total_population if total_population else 0.0
    return isolated_population, total_population, fraction


def shortest_path_deltas(
    G_baseline: nx.Graph, G_perturbed: nx.Graph, od_pairs: list[tuple], weight: str = "length"
) -> dict:
    """Baseline vs. perturbed shortest-path length for a fixed set of
    origin-destination pairs. Feeds directly into resilience_index.py."""
    baseline_lengths = {}
    perturbed_lengths = {}
    unreachable = 0
    for o, d in od_pairs:
        try:
            baseline_lengths[(o, d)] = nx.shortest_path_length(G_baseline, o, d, weight=weight)
        except nx.NetworkXNoPath:
            baseline_lengths[(o, d)] = float("inf")
        if o not in G_perturbed or d not in G_perturbed:
            perturbed_lengths[(o, d)] = float("inf")
            unreachable += 1
            continue
        try:
            perturbed_lengths[(o, d)] = nx.shortest_path_length(G_perturbed, o, d, weight=weight)
        except nx.NetworkXNoPath:
            perturbed_lengths[(o, d)] = float("inf")
            unreachable += 1
    return {
        "baseline_lengths": baseline_lengths,
        "perturbed_lengths": perturbed_lengths,
        "unreachable_od_pairs": unreachable,
        "total_od_pairs": len(od_pairs),
    }


def run_scenario(
    G: nx.Graph, nodes_to_disable: list, od_pairs: list[tuple], population_attr: str = "population"
) -> SimulationResult:
    """One-call disaster scenario: disable nodes, recompute LCC, population
    impact, and OD-pair reachability -- everything the dashboard's "Disaster
    Sandbox" view (View 4) needs in one object."""
    remaining = disable_nodes(G, nodes_to_disable)
    lcc_size, lcc_fraction = largest_connected_component_stats(remaining, G.number_of_nodes())
    isolated_pop, total_pop, isolated_frac = population_impact(G, remaining, population_attr)
    deltas = shortest_path_deltas(G, remaining, od_pairs)

    return SimulationResult(
        disabled_nodes=list(nodes_to_disable),
        remaining_graph=remaining,
        largest_component_size=lcc_size,
        largest_component_fraction=lcc_fraction,
        isolated_population=isolated_pop,
        total_population=total_pop,
        isolated_population_fraction=isolated_frac,
        unreachable_od_pairs=deltas["unreachable_od_pairs"],
        total_od_pairs=deltas["total_od_pairs"],
    )
