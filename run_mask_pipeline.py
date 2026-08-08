"""
run_mask_pipeline.py
=====================
Runs the full downstream pipeline (skeletonization -> healing -> z-levels ->
criticality -> disaster simulation -> resilience index -> visualization) on
a mask image YOU ALREADY HAVE -- no trained PathMamba checkpoint needed.

USE THIS WHEN
-------------
Your PNG is already a road mask: roads highlighted against background,
whether that's a ground-truth mask (e.g. a DeepGlobe/SpaceNet/Massachusetts
*_mask.png file), a hand-drawn test mask, or output from some other
segmentation you already ran. This script treats the image directly as
`prob_mask` -- it skips PathMamba, GLCM, and Hann tiling entirely, because
none of those are needed once you already have a mask.

Use run_real_inference.py instead if your PNG is a raw satellite/aerial
photo that still needs to be SEGMENTED into a mask first -- that path
requires an actual trained checkpoint.

IMPORTANT LIMITATION: NO REAL CONFIDENCE MAP
---------------------------------------------
healing.py's gap-closing gate needs a `confidence_mask` -- PathMamba's own
belief that a road continues through an occluded area. A plain mask PNG has
no such signal (it's binary/grayscale road presence, not model uncertainty).
This script defaults confidence_mask to all-zeros, which means healing's
occlusion_min gate (default 0.3) will reject EVERY candidate -- healing
becomes a no-op unless you override the gate.

Two ways to use this honestly:
  1. Leave the default (confidence=0, occlusion gate active) -- appropriate
     when you just want to see what the mask vectorizes to, gaps and all,
     with no fabricated healing.
  2. Pass --heal-occlusion-min 0 to let healing connect stub pairs based
     purely on distance + angle (no occlusion evidence). This is a coarser
     approximation -- it will connect any nearby, well-aligned stub pair,
     whether or not a road plausibly continues there. Fine for a quick look,
     not something to trust as "verified reconnection."

USAGE
-----
    python -m route_resilience.run_mask_pipeline \\
        --mask /path/to/your_mask.png \\
        --out-dir ./mask_run \\
        --pixel-size-m 0.5

    # if your mask has values other than plain 0/255 (e.g. antialiased edges,
    # or a genuine grayscale probability image), --threshold controls the
    # binarization cutoff used before thinning (default 0.5 on a 0-1 scale):
    python -m route_resilience.run_mask_pipeline \\
        --mask /path/to/your_mask.png --threshold 0.4

    # to see healing actually reconnect nearby gaps (coarse, no confidence
    # evidence -- see limitation above):
    python -m route_resilience.run_mask_pipeline \\
        --mask /path/to/your_mask.png --heal-occlusion-min 0 --heal-angle-max-deg 60
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter

import numpy as np
import networkx as nx


def _tally_reasons(rejected) -> dict:
    if not rejected:
        return {}
    return dict(Counter(getattr(c, "reject_reason", "unknown") or "unknown" for c in rejected))


def run_mask_pipeline(
    mask_path: str,
    out_dir: str = "./mask_run",
    pixel_size_m: float = 0.5,
    threshold: float = 0.5,
    threshold_low: float | None = None,
    min_object_px: int = 12,
    min_branch_m: float = 8.0,
    closing_iterations: int = 1,
    dp_epsilon_m: float = 1.0,
    min_edge_mean_prob: float = 0.0,
    heal_max_search_radius_m: float | None = None,
    heal_occlusion_min: float | None = None,
    heal_angle_max_deg: float | None = None,
    heal_gap_max_m: float | None = None,
    heal_max_iterations: int = 2,
    ground_truth_graph_path: str | None = None,
    worldpop_raster: str | None = None,
    flood_polygon: str | None = None,
    flood_target_crs: str | None = None,
    flood_source_crs: str | None = None,
) -> dict:
    from PIL import Image
    from .geo_utils import SimpleAffine
    from .model_io import RoadPrediction
    from .skeletonization import graph_from_prediction
    from .zlevel import assign_z_levels
    from .healing import heal
    from .centrality import composite_criticality
    from .scenarios import run_default_scenarios, load_worldpop_if_available
    from .visualize import render_pipeline_result
    from .pipeline_diagnostics import diagnose_prediction

    os.makedirs(out_dir, exist_ok=True)

    # 1. load the mask, normalize to [0, 1] float regardless of source format
    # (plain black/white PNG, antialiased mask, or genuine grayscale probs).
    raw = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
    prob_mask = raw / 255.0
    H, W = prob_mask.shape

    # 2. confidence_mask: no real signal available from a plain mask -- see
    # module docstring. Defaults to all-zeros so healing's occlusion gate is
    # honest about having no evidence, unless the caller overrides the gate.
    confidence_mask = np.zeros_like(prob_mask)

    transform = SimpleAffine(origin_x=0.0, origin_y=0.0, pixel_size=pixel_size_m, height_px=H)
    prediction = RoadPrediction(
        prob_mask=prob_mask, confidence_mask=confidence_mask,
        transform=transform, source_gsd_m=pixel_size_m, is_synthetic=False,
    )

    # 2b. pipeline diagnostics
    diag = diagnose_prediction(prediction.prob_mask, prediction.confidence_mask)
    print(f"[diagnostics] signal={diag.signal_strength} "
          f"prob_p95={diag.prob_p95:.3f} coverage@0.5={diag.fraction_above_0_5:.2%}")
    for w in diag.warnings:
        print(f"  [warn] {w}")
    for s in diag.suggestions:
        print(f"  [hint] {s}")

    # 3. skeletonize -> graph (this is the real module, same as real inference)
    reconstructed = graph_from_prediction(
        prediction,
        threshold=threshold,
        threshold_low=threshold_low,
        min_object_px=min_object_px,
        min_branch_m=min_branch_m,
        closing_iterations=closing_iterations,
        dp_epsilon_m=dp_epsilon_m,
        min_edge_mean_prob=min_edge_mean_prob,
    )
    with open(os.path.join(out_dir, "reconstructed_graph_pre_healing.gpickle"), "wb") as f:
        pickle.dump(reconstructed, f)

    # 4. z-levels + healing
    heal_kwargs = {}
    if heal_max_search_radius_m is not None:
        heal_kwargs["max_search_radius_m"] = heal_max_search_radius_m
    if heal_occlusion_min is not None:
        heal_kwargs["occlusion_min"] = heal_occlusion_min
    if heal_angle_max_deg is not None:
        heal_kwargs["angle_max_deg"] = heal_angle_max_deg
    if heal_gap_max_m is not None:
        heal_kwargs["gap_max_m"] = heal_gap_max_m
        # Auto-expand KDTree search radius to match the gap gate when the user
        # only set --heal-gap-max-m. Otherwise the candidate search still uses
        # the 50 m default and finds nothing beyond it -- silent 0 healed.
        if heal_max_search_radius_m is None:
            heal_kwargs["max_search_radius_m"] = heal_gap_max_m
    heal_kwargs["max_iterations"] = heal_max_iterations

    z_graph = assign_z_levels(reconstructed)
    healing_report = heal(z_graph, prediction, **heal_kwargs)
    healed_graph = healing_report.healed_graph

    if healing_report.rejected:
        print(f"[healing] {len(healing_report.rejected)} candidate(s) rejected:")
        for c in healing_report.rejected[:10]:
            print(f"  dist={c.distance_m:.1f}m angle_dev={c.angle_deviation_deg:.1f}deg "
                  f"occl_conf={c.occlusion_confidence:.3f} reason={c.reject_reason}")

    with open(os.path.join(out_dir, "reconstructed_graph.gpickle"), "wb") as f:
        pickle.dump(healed_graph, f)

    # 5. criticality
    criticality = composite_criticality(healed_graph)

    # 6. multi-scenario disaster harness
    worldpop = load_worldpop_if_available(worldpop_raster)
    scenarios_out = run_default_scenarios(
        healed_graph, criticality, worldpop=worldpop, seed=0,
        flood_polygon=flood_polygon,
        flood_target_crs=flood_target_crs,
        flood_source_crs=flood_source_crs,
    )

    # 7. evaluation, only if you supply a ground-truth graph (optional --
    # a plain mask usually has none)
    eval_summary = None
    if ground_truth_graph_path:
        from .evaluation import evaluate
        with open(ground_truth_graph_path, "rb") as f:
            gt_graph = pickle.load(f)
        eval_summary = evaluate(gt_graph, healed_graph).summary()

    composite = criticality.get("composite", {})
    top10 = sorted(composite.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top10_dump = [
        {
            "node": int(n),
            "score": round(float(s), 4),
            "x": float(healed_graph.nodes[n]["x"]),
            "y": float(healed_graph.nodes[n]["y"]),
        }
        for n, s in top10 if n in healed_graph.nodes
    ]

    headline_ri = scenarios_out.get("single_top_node", {}).get("resilience_index")
    headline_top = (
        scenarios_out.get("single_top_node", {}).get("disabled_nodes", [None]) or [None]
    )[0]

    summary = {
        "mask": mask_path,
        "graph_stats": {
            "reconstructed_nodes": healed_graph.number_of_nodes(),
            "reconstructed_edges": healed_graph.number_of_edges(),
            "reconstructed_components": nx.number_connected_components(healed_graph),
        },
        "healing": {
            "bridges_added": len(healing_report.bridges_added),
            "candidates_rejected": len(healing_report.rejected),
            "rejection_reasons": _tally_reasons(healing_report.rejected),
        },
        "criticality_top10": top10_dump,
        "scenarios": scenarios_out,
        "evaluation": eval_summary,
        "prediction_diagnostics": diag.summary(),
        # Flat convenience fields
        "top_criticality_node": headline_top,
        "resilience_index": headline_ri,
    }

    scenarios_for_report = {
        k: ({kk: vv for kk, vv in v.items() if kk != "polygon_coords"}
            if isinstance(v, dict) else v)
        for k, v in scenarios_out.items()
    }
    summary_for_report = {**summary, "scenarios": scenarios_for_report}

    with open(os.path.join(out_dir, "resilience_report.json"), "w") as f:
        json.dump(summary_for_report, f, indent=2, default=str)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"Mask:               {mask_path}\n")
        gs = summary["graph_stats"]
        f.write(f"Nodes / Edges:      {gs['reconstructed_nodes']} / {gs['reconstructed_edges']}\n")
        f.write(f"Components:         {gs['reconstructed_components']}\n")
        f.write(f"Bridges healed:     {summary['healing']['bridges_added']}\n")
        f.write(f"Top criticality:    {headline_top}\n\n")
        f.write("Scenarios:\n")
        for name, sc in scenarios_out.items():
            if name.startswith("_") or not isinstance(sc, dict) or "resilience_index" not in sc:
                continue
            ri = sc.get("resilience_index")
            n_dis = sc.get("n_disabled", 0)
            line = f"  {name:20s} disable={n_dis:>3}  RI={ri}"
            pi = sc.get("population_impact")
            if pi and "fraction_isolated_or_disabled" in pi:
                line += f"  pop_isolated={pi['fraction_isolated_or_disabled']:.1%}"
            f.write(line + "\n")
        if eval_summary:
            f.write(f"\nEvaluation vs ground truth:\n")
            for k, v in eval_summary.items():
                f.write(f"  {k}: {v}\n")

    flood_sc = scenarios_out.get("flood") or {}
    flood_poly_viz = flood_sc.get("polygon_coords")
    flooded_nodes_viz = flood_sc.get("disabled_nodes")

    png_path = render_pipeline_result(
        prob_mask=prediction.prob_mask,
        transform=prediction.transform,
        healed_graph=healed_graph,
        healing_report=healing_report,
        criticality=criticality.get("composite"),
        out_path=os.path.join(out_dir, "visualization.png"),
        title=f"{os.path.basename(mask_path)}: {summary['graph_stats']['reconstructed_nodes']} nodes, "
              f"{summary['healing']['bridges_added']} bridges healed",
        flood_polygon=flood_poly_viz,
        flooded_nodes=flooded_nodes_viz,
    )
    summary["visualization"] = png_path

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask", required=True, help="Path to a mask PNG (roads highlighted)")
    ap.add_argument("--out-dir", default="./mask_run")
    ap.add_argument("--pixel-size-m", type=float, default=0.5,
                    help="Meters per pixel -- only affects reported lengths, not topology")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Binarization cutoff on [0,1]. With --threshold-low, this is the HIGH cutoff of hysteresis.")
    ap.add_argument("--threshold-low", type=float, default=None,
                    help="Optional LOW cutoff for hysteresis thresholding "
                         "(e.g. 0.3 with --threshold 0.5). Recommended for real inference output.")
    ap.add_argument("--min-object-px", type=int, default=12, help="Drop connected blobs smaller than this")
    ap.add_argument("--min-branch-m", type=float, default=8.0, help="Prune spurs shorter than this (meters)")
    ap.add_argument("--closing-iterations", type=int, default=1,
                    help="Binary closing iterations before thinning (bridges 1-2px pinholes)")
    ap.add_argument("--dp-epsilon-m", type=float, default=1.0,
                    help="Douglas-Peucker simplification tolerance (meters) on edge geometries. 0 to disable.")
    ap.add_argument("--min-edge-mean-prob", type=float, default=0.0,
                    help="Drop traced edges whose mean prob along their polyline is below this. "
                         "Try `threshold + 0.1`. 0 to disable.")
    ap.add_argument("--heal-max-search-radius-m", type=float, default=None)
    ap.add_argument("--heal-occlusion-min", type=float, default=None,
                    help="Set to 0 to let healing connect gaps with NO confidence evidence "
                         "(see module docstring limitation)")
    ap.add_argument("--heal-angle-max-deg", type=float, default=None)
    ap.add_argument("--heal-gap-max-m", type=float, default=None)
    ap.add_argument("--heal-max-iterations", type=int, default=2,
                    help="Number of iterative healing passes to run (stop early if no new bridges are added).")
    ap.add_argument("--ground-truth-graph", default=None, help="Optional gpickle for evaluation.evaluate()")
    ap.add_argument("--worldpop-raster", default=None,
                    help="Path to a WorldPop GeoTIFF. When provided, per-scenario "
                         "population impact is computed via isochrone intersection. "
                         "Requires rasterio + shapely installed. Omit to skip silently.")
    ap.add_argument("--flood-polygon", default=None,
                    help="Path to a GeoJSON flood-extent polygon in the graph's coord frame. "
                         "Adds a 'flood' scenario that disables all nodes inside the polygon.")
    ap.add_argument("--flood-polygon-crs", default=None,
                    help="EPSG code of the graph's coord frame. When set, polygon is auto-reprojected.")
    ap.add_argument("--flood-polygon-source-crs", default=None,
                    help="Override polygon source CRS (default: auto-detect EPSG:4326 for lon/lat).")
    args = ap.parse_args()

    summary = run_mask_pipeline(
        mask_path=args.mask, out_dir=args.out_dir, pixel_size_m=args.pixel_size_m,
        threshold=args.threshold, threshold_low=args.threshold_low,
        min_object_px=args.min_object_px, min_branch_m=args.min_branch_m,
        closing_iterations=args.closing_iterations,
        dp_epsilon_m=args.dp_epsilon_m,
        min_edge_mean_prob=args.min_edge_mean_prob,
        heal_max_search_radius_m=args.heal_max_search_radius_m,
        heal_occlusion_min=args.heal_occlusion_min,
        heal_angle_max_deg=args.heal_angle_max_deg,
        heal_gap_max_m=args.heal_gap_max_m,
        heal_max_iterations=args.heal_max_iterations,
        ground_truth_graph_path=args.ground_truth_graph,
        worldpop_raster=args.worldpop_raster,
        flood_polygon=args.flood_polygon,
        flood_target_crs=args.flood_polygon_crs,
        flood_source_crs=args.flood_polygon_source_crs,
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nFull report written to {args.out_dir}/")


if __name__ == "__main__":
    main()
