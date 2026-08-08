"""
visualize.py
============
Renders the pipeline's output as an actual image -- the piece that was
missing: everything else in the pipeline produces data (.npy, .gpickle,
.json), nothing produced something you could look at.

Draws, on top of the prob_mask raster:
  * surviving/reconstructed edges (from the mask) in one color
  * healed bridges (added by healing.py) in a distinct color + dashed
  * the true occlusion gaps (if known -- synthetic runs only) in another
    color, so you can see what SHOULD have been healed vs what WAS
  * nodes colored by composite criticality (optional -- pass a criticality
    dict) so this doubles as a rough stand-in for the deck's "Criticality
    Heatmap" view

Works for both the synthetic test (which has known ground truth / gaps) and
real inference (which doesn't -- gap_records and true_graph are optional).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # no display needed -- just write a PNG
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def render_pipeline_result(
    prob_mask: np.ndarray,
    transform,
    healed_graph: nx.Graph,
    healing_report=None,
    criticality: Optional[dict] = None,
    gap_records: Optional[list] = None,
    true_graph: Optional[nx.Graph] = None,
    out_path: str = "pipeline_visualization.png",
    title: str = "Road network reconstruction + healing",
    dpi: int = 150,
    flood_polygon: Optional[list] = None,
    flooded_nodes: Optional[list] = None,
):
    """Render one PNG showing the mask, reconstructed graph, and healing result.

    Parameters
    ----------
    prob_mask : (H, W) the model's road-probability raster (background image).
    transform : the same transform used to build the graph (world -> pixel),
                so node x,y can be plotted in the raster's pixel space.
    healed_graph : the final graph after healing.py ran.
    healing_report : optional HealingReport -- if given, bridges_added are
                drawn as dashed highlighted edges distinct from the rest.
    criticality : optional dict (e.g. composite_criticality(...)["composite"])
                -- if given, nodes are colored/sized by criticality score.
    gap_records : optional list of GapRecord (synthetic runs only) -- if
                given, draws the TRUE occlusion gap locations as red X's so
                you can see what healing should have found.
    true_graph : optional ground-truth graph (synthetic runs only) -- if
                given, drawn faintly underneath for comparison.
    flood_polygon : optional list of rings, where each ring is a list of
                (x, y) tuples in the graph's world frame. Matches the
                ``polygon_coords`` field returned by flood_simulation.py --
                a single Polygon gives one ring, a MultiPolygon / FeatureCollection
                gives multiple. Each ring is drawn as a translucent blue fill
                + dotted outline; they are drawn independently so disjoint
                flood extents all appear correctly.
    flooded_nodes : optional list of node ids that fall inside any flood ring.
                Drawn in blue, overriding the criticality gradient for those
                nodes so the flooded region is visually distinct.
    """
    H, W = prob_mask.shape
    fig, ax = plt.subplots(figsize=(12, 12 * H / W))
    ax.imshow(prob_mask, cmap="gray", vmin=0, vmax=1, origin="upper")

    def world_to_px(x, y):
        r, c = transform.world_to_pixel(x, y)
        return c, r  # matplotlib wants (x_pixel, y_pixel) = (col, row)

    def _edge_path_px(G, u, v):
        """Return (xs_px, ys_px) polyline for an edge. Uses edge['geometry']
        (the real traced polyline from skeletonization, Douglas-Peucker
        simplified) when present; falls back to a straight chord for edges
        without geometry (e.g. healing-added synthetic bridges, which
        legitimately are straight)."""
        geom = G.edges[u, v].get("geometry")
        if geom and len(geom) >= 2:
            pts = [world_to_px(x, y) for (x, y) in geom]
        else:
            pu = world_to_px(G.nodes[u]["x"], G.nodes[u]["y"])
            pv = world_to_px(G.nodes[v]["x"], G.nodes[v]["y"])
            pts = [pu, pv]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return xs, ys

    # 1. faint ground truth underneath, if given (synthetic runs only)
    if true_graph is not None:
        for u, v in true_graph.edges():
            xs, ys = _edge_path_px(true_graph, u, v)
            ax.plot(xs, ys, color="cyan", alpha=0.25, linewidth=1.5, zorder=1)

    # 1b. flood polygon overlay -- translucent blue fill + outline per ring.
    # Sits above the ground-truth cyan lines but below the reconstructed edges.
    # flood_polygon is a LIST OF RINGS (from flood_simulation.load_polygon),
    # so MultiPolygon / FeatureCollection flood extents all render correctly.
    flooded_set = set(flooded_nodes) if flooded_nodes else set()
    if flood_polygon:
        for ring_idx, ring in enumerate(flood_polygon):
            if not ring:
                continue
            poly_px = [world_to_px(x, y) for (x, y) in ring]
            poly_xs = [p[0] for p in poly_px] + [poly_px[0][0]]
            poly_ys = [p[1] for p in poly_px] + [poly_px[0][1]]
            # Only add the legend label on the first ring to avoid duplicates
            label = f"Flood zone ({len(flooded_set)} nodes)" if ring_idx == 0 else None
            ax.fill(poly_xs, poly_ys, color="steelblue", alpha=0.25, zorder=1.5,
                    label=label)
            ax.plot(poly_xs, poly_ys, color="steelblue", linewidth=2.0,
                    alpha=0.9, zorder=1.6, linestyle=":")

    # 2. healed-bridge edge set, for exclusion from the "normal" edge pass
    bridge_pairs = set()
    if healing_report is not None:
        for c in healing_report.bridges_added:
            bridge_pairs.add(frozenset((c.u, c.v)))

    # 3. reconstructed graph edges (normal)
    for u, v in healed_graph.edges():
        if frozenset((u, v)) in bridge_pairs:
            continue  # drawn separately below, highlighted
        xs, ys = _edge_path_px(healed_graph, u, v)
        ax.plot(xs, ys, color="lime", linewidth=1.8, alpha=0.9, zorder=2)

    # 4. healed bridges -- highlighted, dashed, distinct color (always straight
    # -- healing.py adds them without a geometry attribute, so the fallback
    # chord is the correct thing here)
    for u, v in bridge_pairs:
        if u not in healed_graph or v not in healed_graph:
            continue
        xs, ys = _edge_path_px(healed_graph, u, v)
        ax.plot(xs, ys, color="yellow", linewidth=3.0,
                linestyle="--", zorder=4, solid_capstyle="round")

    # 5. true gap locations, if known (synthetic only) -- red X where healing
    # SHOULD have connected something, whether or not it actually did
    if gap_records:
        for gap in gap_records:
            mx, my = (gap.x0 + gap.x1) / 2, (gap.y0 + gap.y1) / 2
            pm = world_to_px(mx, my)
            ax.plot(pm[0], pm[1], marker="x", color="red", markersize=14, markeredgewidth=3, zorder=5)

    # 6. nodes, colored/sized by criticality if given
    xs, ys, sizes, colors = [], [], [], []
    for n, d in healed_graph.nodes(data=True):
        px, py = world_to_px(d["x"], d["y"])
        xs.append(px)
        ys.append(py)
        if criticality:
            score = criticality.get(n, 0.0)
            sizes.append(30 + 120 * score)
            colors.append(score)
        else:
            sizes.append(25)
            colors.append(0.5)

    sc = None  # may be used by colorbar below; initialize here so always defined
    if criticality:
        sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap="autumn_r", vmin=0, vmax=1,
                        edgecolors="black", linewidths=0.5, zorder=6)

    # 6b. flooded nodes -- overlay in blue on top of the criticality scatter,
    # so the flooded region is unmistakable even when it overlaps high-
    # criticality nodes.
    if flooded_set:
        flood_xs, flood_ys, flood_sz = [], [], []
        for n, d in healed_graph.nodes(data=True):
            if n not in flooded_set:
                continue
            px, py = world_to_px(d["x"], d["y"])
            flood_xs.append(px)
            flood_ys.append(py)
            s = 30 + 120 * (criticality.get(n, 0.5) if criticality else 0.5)
            flood_sz.append(s)
        ax.scatter(flood_xs, flood_ys, s=flood_sz, c="steelblue",
                   edgecolors="white", linewidths=1.2, zorder=7,
                   label=f"Flooded nodes ({len(flooded_set)})")
        # Only add colorbar when a criticality scatter exists
        if sc is not None:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Composite criticality")
    else:
        ax.scatter(xs, ys, s=sizes, c="white", edgecolors="black", linewidths=0.5, zorder=6)

    legend_elems = [
        Line2D([0], [0], color="lime", lw=2, label="Reconstructed road"),
        Line2D([0], [0], color="yellow", lw=3, linestyle="--", label="Healed bridge"),
    ]
    if true_graph is not None:
        legend_elems.append(Line2D([0], [0], color="cyan", lw=1.5, alpha=0.5, label="Ground truth (faint)"))
    if gap_records:
        legend_elems.append(Line2D([0], [0], marker="x", color="red", lw=0, markersize=10,
                                   markeredgewidth=3, label="True gap location"))
    ax.legend(handles=legend_elems, loc="upper right", framealpha=0.85, fontsize=9)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
