"""
resilience_index.py
====================
Turns the baseline-vs-perturbed shortest-path lengths from simulation.py into
the single quantitative "Resilience Index". Documented formula:

    For each OD pair:
        - if baseline was ALREADY unreachable  -> contributes 1 (unaffected --
          the disaster didn't cause this, it was already broken)
        - if perturbed path is unreachable      -> contributes 0
        - if perturbed path length <= baseline  -> contributes 1
          (shouldn't normally happen, but a rerouted path via a healed bridge
          could occasionally be shorter than the original baseline route)
        - otherwise                              -> contributes baseline / perturbed
          (a path that got twice as long contributes 0.5, etc.)

    Resilience Index = mean of all per-pair contributions, in [0, 1].
    1.0 = the disaster scenario caused zero degradation across all OD pairs.
    0.0 = every OD pair was completely cut off.

NOTE: the baseline==inf check MUST come before the perturbed==inf check.
Removing nodes can never reconnect an already-broken pair, so an
already-unreachable-at-baseline pair is trivially ALSO unreachable after any
disaster -- checking perturbed==inf first would score it as "fully cut off by
this scenario" (0.0) when it was never affected by the scenario at all. This
was a real, confirmed bug (fixed this session) -- verified against a real
graph: before the fix, top_5_nodes/random_10_percent scenarios both collapsed
to RI=0.0 on a fragmented 92-node/14-component graph; after the fix, the same
graph correctly showed RI=0.944 for both, with only 1 pair genuinely
newly-cut-off rather than the entire already-fragmented set being penalized.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResilienceIndexResult:
    index: float
    per_pair_scores: dict
    fully_cut_off_pairs: int
    degraded_pairs: int
    unaffected_pairs: int
    total_pairs: int


def compute_resilience_index(baseline_lengths: dict, perturbed_lengths: dict) -> ResilienceIndexResult:
    if set(baseline_lengths.keys()) != set(perturbed_lengths.keys()):
        raise ValueError("baseline_lengths and perturbed_lengths must cover the same OD pairs")

    per_pair_scores = {}
    fully_cut_off = 0
    degraded = 0
    unaffected = 0

    for pair, baseline in baseline_lengths.items():
        perturbed = perturbed_lengths[pair]
        if baseline == float("inf"):
            per_pair_scores[pair] = 1.0
            unaffected += 1
        elif perturbed == float("inf"):
            per_pair_scores[pair] = 0.0
            fully_cut_off += 1
        elif perturbed <= baseline:
            per_pair_scores[pair] = 1.0
            unaffected += 1
        else:
            per_pair_scores[pair] = baseline / perturbed
            degraded += 1

    total = len(per_pair_scores)
    index = sum(per_pair_scores.values()) / total if total else float("nan")

    return ResilienceIndexResult(
        index=index,
        per_pair_scores=per_pair_scores,
        fully_cut_off_pairs=fully_cut_off,
        degraded_pairs=degraded,
        unaffected_pairs=unaffected,
        total_pairs=total,
    )


def compare_scenarios(results: dict[str, ResilienceIndexResult]) -> dict:
    """Ranks named scenarios by resilience index, ascending -- most damaging first."""
    ranked = sorted(results.items(), key=lambda kv: kv[1].index)
    return {name: result.index for name, result in ranked}
