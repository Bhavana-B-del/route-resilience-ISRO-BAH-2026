"""
zlevel.py
=========
Stage 5 of the pipeline: keeps flyovers/underpasses structurally distinct from
surface roads using OSM shared-node topology (layer/bridge/tunnel tags), never
by rasterizing grade -- matches the "vector-first Z-levels" design decision in
the submission deck.

Works identically on the synthetic graph (which carries the same `bridge`/`layer`
attributes a real OSM extract would) and on a real OSM pull later -- no change
needed when real OSM data replaces the synthetic graph, as long as the loader
populates the same edge attributes.
"""

from __future__ import annotations

import networkx as nx


def assign_z_levels(G: nx.Graph) -> nx.Graph:
    """Reads existing `bridge` (bool) / `tunnel` (bool) / `layer` (int) edge
    attributes -- defaulting any missing ones to surface-level -- and writes a
    normalized `z_level` attribute: -1 = tunnel/underpass, 0 = surface,
    1+ = elevated/flyover (from OSM `layer` when present).
    """
    out = G.copy()
    for u, v, data in out.edges(data=True):
        if data.get("tunnel", False):
            z = -1
        elif data.get("bridge", False):
            z = max(1, data.get("layer", 1))
        else:
            z = data.get("layer", 0)
        out.edges[u, v]["z_level"] = z
    return out


def z_level_summary(G: nx.Graph) -> dict:
    counts: dict[int, int] = {}
    for _, _, data in G.edges(data=True):
        z = data.get("z_level", 0)
        counts[z] = counts.get(z, 0) + 1
    return dict(sorted(counts.items()))
