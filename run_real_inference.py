from __future__ import annotations
import argparse
import json
import os
import pickle
import re
from collections import Counter

import numpy as np
import networkx as nx
import torch

from route_resilience.train_pathmamba_v2_final import build_model, GLCMOcclusionMap

def _tally_reasons(rejected) -> dict:
    if not rejected:
        return {}
    return dict(Counter(getattr(c, "reject_reason", "unknown") or "unknown" for c in rejected))


# ---------------------------------------------------------------------------
# FIX (this session): two bugs, both in this file, both fixed here.
#
# BUG 1 -- silent scoping shadow. `run_one_tile` used to do
#     from .inference import tiled_infer_from_grayscale, make_torch_model_fn
#   as a LOCAL import inside the function. In Python, `from x import name`
#   anywhere inside a function body makes `name` local to the WHOLE function
#   -- so the call to make_torch_model_fn(...) later in the same function
#   silently resolved to .inference's version, never the careful one defined
#   at module level here (which does real per-channel normalization). No
#   error, no warning -- the correct code was just dead. Fixed by not
#   depending on .inference or a same-named import at all; this file now owns
#   its own self-contained tiling+inference loop end to end.
#
# BUG 2 -- real RGB was discarded before reaching the model. The pipeline
#   loaded genuine RGB (img_rgb), collapsed it to a single grayscale
#   luminance value, and only that grayscale value ever reached the model --
#   via a hack that replicated it into 3 identical "R=G=B" channels. The
#   model was TRAINED on real, distinct R/G/B channels; feeding it a
#   hue-less, zero-saturation image is a severe distribution shift on its
#   own, independent of Bug 1. Fixed by keeping img_rgb alive and building a
#   genuine 4-channel [R, G, B, GLCM] tensor -- GLCM is still computed from
#   luminance (that part was always correct), but R/G/B are now the real
#   source channels, each normalized with its own real per-channel
#   global_stats mean/std, not a single averaged grayscale approximation.
#
# Together these fully explain the reported symptom: blank prob_mask / a
# single surviving node after skeletonization is exactly what a
# severely-out-of-distribution, unnormalized (or wrongly normalized) input
# produces -- the model's activations saturate/collapse rather than
# genuinely finding "no roads here".
# ---------------------------------------------------------------------------


def _hann_window_2d(size: int) -> np.ndarray:
    """2D raised-cosine window: peak 1.0 at center, ~0 at edges. A tiny floor
    (1e-6) keeps every output pixel's total blend weight strictly > 0, even
    at tile-plan corners, so normalization never divides by exactly zero."""
    w1 = np.hanning(size).astype(np.float32)
    w1 = np.clip(w1, 1e-6, None)
    return np.outer(w1, w1)


def _plan_tiles(H: int, W: int, tile_size: int, exclusion_band_px: int):
    """Hann-tiling plan: stride = tile_size - 2*exclusion_band_px (50% overlap
    at the default 512/128). Pads top/left by exclusion_band_px so the FIRST
    tile's high-weight center covers the image's true (0,0) corner, and pads
    bottom/right so the last tile reaches the far edge. Returns
    (positions, pad_top, pad_left, H_pad, W_pad)."""
    stride = tile_size - 2 * exclusion_band_px
    if stride <= 0:
        raise ValueError(f"tile_size ({tile_size}) too small for exclusion_band_px ({exclusion_band_px})")
    pad_top = pad_left = exclusion_band_px
    n_rows = max(1, int(np.ceil((H + pad_top + exclusion_band_px - tile_size) / stride)) + 1)
    n_cols = max(1, int(np.ceil((W + pad_left + exclusion_band_px - tile_size) / stride)) + 1)
    H_pad = (n_rows - 1) * stride + tile_size
    W_pad = (n_cols - 1) * stride + tile_size
    positions = [(r * stride, c * stride) for r in range(n_rows) for c in range(n_cols)]
    return positions, pad_top, pad_left, H_pad, W_pad


def tiled_infer_rgb_glcm(
    img_rgb: np.ndarray,
    glcm_channel: np.ndarray,
    model,
    device: str,
    global_stats: dict | None,
    tile_size: int = 512,
    exclusion_band_px: int = 128,
    batch_size: int = 4,
):
    """Hann-blended tiled inference over a genuine 4-channel [R, G, B, GLCM]
    input. Replaces the old tiled_infer_from_grayscale + forward_wrapper
    2-channel-replication path (see the FIX note above) -- this is the ONLY
    inference path now, self-contained so it doesn't depend on .inference's
    internals.

    img_rgb        : (H, W, 3) uint8, the REAL source RGB (not luminance).
    glcm_channel    : (H, W) float32 in [0, 1], already normalized (unchanged
                      from before -- GLCM was never part of either bug).
    global_stats    : {"mean": [r_mean, g_mean, b_mean], "std": [r_std, g_std, b_std]}
                      persisted in the checkpoint by training. Required for
                      correct predictions -- see the loud warning below if
                      it's missing.

    Returns (prob_mask, confidence_mask), both (H, W) float32 in [0, 1].
    """
    H, W = img_rgb.shape[:2]
    positions, pad_top, pad_left, H_pad, W_pad = _plan_tiles(H, W, tile_size, exclusion_band_px)

    # Real per-channel normalization -- NOT the old "average of 3 stats
    # applied once to a replicated gray channel" approximation. If
    # global_stats is missing (older checkpoint), fall back to a neutral
    # 0-centered scaling and warn loudly, same spirit as before but now
    # actually reachable code (Bug 1 made the old warning path dead too,
    # since it lived inside the shadowed function).
    if global_stats is not None:
        mean = np.asarray(global_stats["mean"], dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(global_stats["std"], dtype=np.float32).reshape(3, 1, 1)
    else:
        print("[warn] no global_stats in checkpoint -- this checkpoint was saved before "
              "per-channel RGB normalization stats were persisted. Falling back to a "
              "generic /255, 0.5-centered scaling. Predictions will likely be degraded. "
              "Re-save/re-run training with the updated script to fix this permanently.")
        mean = np.array([127.5, 127.5, 127.5], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([127.5, 127.5, 127.5], dtype=np.float32).reshape(3, 1, 1)

    rgb_chw = np.transpose(img_rgb.astype(np.float32), (2, 0, 1))  # (3, H, W)
    rgb_norm = (rgb_chw - mean) / std
    full_4ch = np.concatenate([rgb_norm, glcm_channel[np.newaxis, :, :].astype(np.float32)], axis=0)  # (4, H, W)

    padded = np.pad(
        full_4ch,
        ((0, 0), (pad_top, H_pad - H - pad_top), (pad_left, W_pad - W - pad_left)),
        mode="reflect",
    )

    road_acc = np.zeros((H_pad, W_pad), dtype=np.float32)
    conf_acc = np.zeros((H_pad, W_pad), dtype=np.float32)
    weight_acc = np.zeros((H_pad, W_pad), dtype=np.float32)
    hann = _hann_window_2d(tile_size)

    model.eval()
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]
            batch_np = np.stack(
                [padded[:, r:r + tile_size, c:c + tile_size] for r, c in chunk], axis=0
            ).astype(np.float32)
            x_tensor = torch.from_numpy(batch_np).to(device)

            road_out, conf_out = model(x_tensor)
            # Same conditional-sigmoid heuristic as before this fix -- unchanged
            # on purpose, not part of either diagnosed bug: if the raw head
            # output already looks like it's in [0, 1] treat it as
            # probabilities, otherwise treat it as logits and apply sigmoid.
            road_prob = torch.sigmoid(road_out) if (road_out.min() < 0 or road_out.max() > 1) else road_out
            conf_prob = torch.sigmoid(conf_out) if (conf_out.min() < 0 or conf_out.max() > 1) else conf_out

            road_prob = road_prob.squeeze(1).cpu().numpy()
            conf_prob = conf_prob.squeeze(1).cpu().numpy()

            for k, (r, c) in enumerate(chunk):
                road_acc[r:r + tile_size, c:c + tile_size] += road_prob[k] * hann
                conf_acc[r:r + tile_size, c:c + tile_size] += conf_prob[k] * hann
                weight_acc[r:r + tile_size, c:c + tile_size] += hann

    weight_acc = np.maximum(weight_acc, 1e-8)
    prob_full = (road_acc / weight_acc)[pad_top:pad_top + H, pad_left:pad_left + W]
    conf_full = (conf_acc / weight_acc)[pad_top:pad_top + H, pad_left:pad_left + W]
    return prob_full.astype(np.float32), conf_full.astype(np.float32)


def load_trained_model(checkpoint_path, device="cuda", **kwargs):
    ckpt = torch.load(checkpoint_path, map_location=device)
    glcm_norm_value = ckpt.get("glcm_norm_value", 1.0) if isinstance(ckpt, dict) else 1.0
    global_stats = ckpt.get("global_stats") if isinstance(ckpt, dict) else None
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    use_mamba = kwargs.get("use_mamba", True)
    model = build_model(use_mamba=use_mamba)

    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load checkpoint '{checkpoint_path}' into the current model "
            f"architecture. This is almost always because the checkpoint was trained "
            f"with an OLDER/DIFFERENT version of build_model() (e.g. a different "
            f"number of input channels, or a different Mamba layer count) than the "
            f"one in the code right now -- it is NOT related to the uploaded image's "
            f"channel count. Point the checkpoint path at a checkpoint saved by the "
            f"CURRENT training script instead. Original error:\n{e}"
        ) from e
    model.to(device)
    model.eval()

    # NOTE: the old forward_wrapper (2-channel -> replicated-4-channel
    # expansion) is REMOVED. It existed only because the old pipeline never
    # gave the model real RGB in the first place -- see Bug 2 above. Now that
    # tiled_infer_rgb_glcm always builds genuine 4-channel [R,G,B,GLCM]
    # input, the model's own unmodified forward() is called directly.
    return model, glcm_norm_value, global_stats

def run_one_tile(
    image_path: str,
    model,
    glcm_norm_value,
    global_stats,
    out_dir: str,
    device: str = "cuda",
    tile_size: int = 512,
    exclusion_band_px: int = 128,
    batch_size: int = 4,
    source_gsd_m: float = 0.28,
    pixel_size_m: float | None = None,
    # skeletonization knobs
    threshold: float = 0.5,
    threshold_low: float | None = None,
    min_object_px: int = 12,
    min_branch_m: float = 10.0,
    closing_iterations: int = 1,
    dp_epsilon_m: float = 1.0,
    min_edge_mean_prob: float = 0.0,
    # healing knobs
    heal_max_search_radius_m: float | None = None,
    heal_occlusion_min: float | None = None,
    heal_angle_max_deg: float | None = None,
    heal_gap_max_m: float | None = None,
    heal_max_iterations: int = 2,
    # scenario knobs
    worldpop_raster: str | None = None,
    flood_polygon: str | None = None,
    flood_target_crs: str | None = None,
    flood_source_crs: str | None = None,
    # evaluation
    ground_truth_graph_path: str | None = None,
    use_fast_glcm_proxy: bool = False,
) -> dict:
    from PIL import Image
    from .glcm import GLCMOcclusionMap
    from .skeletonization import graph_from_prediction
    from .healing import heal
    from .zlevel import assign_z_levels
    from .centrality import composite_criticality
    from .geo_utils import SimpleAffine
    from .scenarios import run_default_scenarios, load_worldpop_if_available
    from .visualize import render_pipeline_result
    from .model_io import RoadPrediction

    os.makedirs(out_dir, exist_ok=True)

    # 1. load image as REAL RGB -- kept alive end to end now (see FIX note at
    # top of file). Luminance is still computed, but ONLY as GLCM's input
    # (GLCM was never part of either bug -- unchanged behavior).
    img_rgb = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    luminance = (0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]).astype(np.uint8)
    H, W = luminance.shape

    # Persist a copy of the raw input tile alongside prob_mask.npy /
    # confidence_mask.npy. Nothing downstream reads this file -- it exists
    # purely so a later dashboard session (possibly on a different machine,
    # long after the original upload path is gone) can still show the full
    # "input -> predicted -> healed" pipeline story instead of just the last
    # two stages. Cheap: one small PNG per tile, written once.
    try:
        Image.open(image_path).convert("RGB").save(os.path.join(out_dir, "input_image.png"))
    except Exception as e:
        print(f"[warn] could not save a copy of the input image ({type(e).__name__}: {e}); "
              f"the dashboard's 'Input' panel will be empty for this run.")

    # 2. transform. Without a real georeferenced input (see the "coordinate
    # frame" discussion in the docs), a SimpleAffine gives meter-accurate
    # lengths but no CRS anchor. For real georeferenced Cartosat/Sentinel
    # tiles you'd load a real affine here instead.
    transform = SimpleAffine(
        origin_x=0.0, origin_y=0.0,
        pixel_size=pixel_size_m or source_gsd_m,
        height_px=H,
    )

    # 3. GLCM tool -- reuse training's fixed norm value if the checkpoint saved
    # it; otherwise the tool computes per-image (fine for single-tile inference).
    glcm_tool = GLCMOcclusionMap(norm_value=glcm_norm_value, use_fast_proxy=use_fast_glcm_proxy)
    glcm_channel = glcm_tool.compute(luminance)

    # 4. tiled Hann-blended inference over the REAL 4-channel [R,G,B,GLCM]
    # input (see FIX note at top of file) -> a full RoadPrediction with REAL
    # prob_mask AND confidence_mask. This is where the 3-gate healing check
    # gets its occlusion evidence.
    prob_mask, confidence_mask = tiled_infer_rgb_glcm(
        img_rgb, glcm_channel, model, device=device, global_stats=global_stats,
        tile_size=tile_size, exclusion_band_px=exclusion_band_px, batch_size=batch_size,
    )
    prediction = RoadPrediction(
        prob_mask=prob_mask, confidence_mask=confidence_mask,
        transform=transform, source_gsd_m=source_gsd_m, is_synthetic=False,
    )
    np.save(os.path.join(out_dir, "prob_mask.npy"), prediction.prob_mask)
    np.save(os.path.join(out_dir, "confidence_mask.npy"), prediction.confidence_mask)

    # 5. skeletonize -> graph (with all the new knobs)
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

    # 6. z-levels + healing with 3 real gates (occlusion signal is real now!)
    heal_kwargs = {}
    if heal_max_search_radius_m is not None:
        heal_kwargs["max_search_radius_m"] = heal_max_search_radius_m
    if heal_occlusion_min is not None:
        heal_kwargs["occlusion_min"] = heal_occlusion_min
    if heal_angle_max_deg is not None:
        heal_kwargs["angle_max_deg"] = heal_angle_max_deg
    if heal_gap_max_m is not None:
        heal_kwargs["gap_max_m"] = heal_gap_max_m
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

    # 7. criticality
    criticality = composite_criticality(healed_graph)

    # 8. multi-scenario disaster harness (default 3, optional flood adds 4th)
    worldpop = load_worldpop_if_available(worldpop_raster)
    scenarios_out = run_default_scenarios(
        healed_graph, criticality, worldpop=worldpop, seed=0,
        flood_polygon=flood_polygon,
        flood_target_crs=flood_target_crs,
        flood_source_crs=flood_source_crs,
    )

    # 9. evaluation vs GT graph if provided
    eval_summary = None
    if ground_truth_graph_path:
        from .evaluation import evaluate
        with open(ground_truth_graph_path, "rb") as f:
            gt_graph = pickle.load(f)
        eval_summary = evaluate(gt_graph, healed_graph).summary()

    # Structured report assembly
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
        "image": image_path,
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
        "top_criticality_node": headline_top,
        "resilience_index": headline_ri,
    }

    # Strip large viz-only fields before JSON dump
    scenarios_for_report = {
        k: ({kk: vv for kk, vv in v.items() if kk != "polygon_coords"}
            if isinstance(v, dict) else v)
        for k, v in scenarios_out.items()
    }
    summary_for_report = {**summary, "scenarios": scenarios_for_report}

    with open(os.path.join(out_dir, "resilience_report.json"), "w") as f:
        json.dump(summary_for_report, f, indent=2, default=str)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        gs = summary["graph_stats"]
        f.write(f"Image:              {image_path}\n")
        f.write(f"Nodes / Edges:      {gs['reconstructed_nodes']} / {gs['reconstructed_edges']}\n")
        f.write(f"Components:         {gs['reconstructed_components']}\n")
        f.write(f"Bridges healed:     {summary['healing']['bridges_added']}\n")
        f.write(f"Top criticality:    {headline_top}\n\n")
        f.write("Scenarios (RI closer to 1.0 = more resilient):\n")
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
            f.write("\nEvaluation vs ground truth:\n")
            for k, v in eval_summary.items():
                f.write(f"  {k}: {v}\n")

    # Visualization -- pull flood overlay coords/nodes if a flood scenario ran
    flood_sc = scenarios_out.get("flood") or {}
    flood_poly_viz = flood_sc.get("polygon_coords")
    flooded_nodes_viz = flood_sc.get("disabled_nodes")

    gt_graph_for_viz = None
    if ground_truth_graph_path:
        with open(ground_truth_graph_path, "rb") as f:
            gt_graph_for_viz = pickle.load(f)

    try:
        png_path = render_pipeline_result(
            prob_mask=prediction.prob_mask,
            transform=prediction.transform,
            healed_graph=healed_graph,
            healing_report=healing_report,
            criticality=criticality.get("composite"),
            true_graph=gt_graph_for_viz,
            out_path=os.path.join(out_dir, "visualization.png"),
            title=f"{os.path.basename(image_path)}: "
                  f"{summary['graph_stats']['reconstructed_nodes']} nodes, "
                  f"{summary['healing']['bridges_added']} bridges, RI={headline_ri}",
            flood_polygon=flood_poly_viz,
            flooded_nodes=flooded_nodes_viz,
        )
        summary["visualization"] = png_path
    except Exception as e:
        # matplotlib backend fails (Windows Application Control on _backend_agg
        # is a real thing on locked-down laptops) shouldn't kill the run --
        # everything else has already been written.
        print(f"[warn] visualization skipped ({type(e).__name__}: {e}). "
              f"Try setting $env:MPLBACKEND='svg' and re-running.")
        summary["visualization"] = None

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True, help="Path to trained PathMamba checkpoint (.pth)")
    ap.add_argument("--image", help="Single grayscale image tile to run inference on")
    ap.add_argument("--image-dir", help="Directory of tiles to run in batch (one report per tile)")
    ap.add_argument("--out-dir", required=True, help="Where to write per-tile outputs")
    ap.add_argument("--source-gsd-m", type=float, default=0.28, help="Ground sample distance (m/px)")
    ap.add_argument("--pixel-size-m", type=float, default=None,
                    help="Override pixel size for the transform (defaults to --source-gsd-m)")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--exclusion-band-px", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--use-mamba", action="store_true", default=True)
    ap.add_argument("--no-mamba", dest="use_mamba", action="store_false",
                    help="Use the GRU bottleneck fallback instead of mamba-ssm")
    ap.add_argument("--use-fast-glcm-proxy", action="store_true", default=False)
    ap.add_argument("--device", default="cuda")

    # Skeletonization
    ap.add_argument("--threshold", type=float, default=0.5, help="Prob-mask binarization cutoff")
    ap.add_argument("--threshold-low", type=float, default=None,
                    help="Lower cutoff for hysteresis thresholding (e.g. 0.3 with --threshold 0.5). "
                         "Recommended for real PathMamba output.")
    ap.add_argument("--min-object-px", type=int, default=12)
    ap.add_argument("--min-branch-m", type=float, default=10.0)
    ap.add_argument("--closing-iterations", type=int, default=1)
    ap.add_argument("--dp-epsilon-m", type=float, default=1.0)
    ap.add_argument("--min-edge-mean-prob", type=float, default=0.0)

    # Healing
    ap.add_argument("--heal-max-search-radius-m", type=float, default=None)
    ap.add_argument("--heal-occlusion-min", type=float, default=None)
    ap.add_argument("--heal-angle-max-deg", type=float, default=None)
    ap.add_argument("--heal-gap-max-m", type=float, default=None,
                    help="Override healing's max gap length. When set, also auto-expands search radius.")
    ap.add_argument("--heal-max-iterations", type=int, default=2,
                    help="Number of iterative healing passes to run (stop early if no new bridges are added).")

    # Scenarios
    ap.add_argument("--worldpop-raster", default=None,
                    help="Path to WorldPop GeoTIFF for population impact (needs rasterio + shapely)")
    ap.add_argument("--flood-polygon", default=None,
                    help="Path to GeoJSON flood polygon; adds a 'flood' scenario")
    ap.add_argument("--flood-polygon-crs", default=None,
                    help="EPSG code of the graph's frame (e.g. EPSG:32643) for auto-reprojection")
    ap.add_argument("--flood-polygon-source-crs", default=None,
                    help="Override polygon source CRS (default: auto-detect EPSG:4326 for lon/lat)")

    # Evaluation
    ap.add_argument("--ground-truth-graph", default=None,
                    help="Optional gpickle of a known-correct graph for evaluation.evaluate()")

    args = ap.parse_args()

    if not args.image and not args.image_dir:
        ap.error("pass either --image or --image-dir")

    print(f"Loading checkpoint: {args.checkpoint}")
    model, glcm_norm_value, global_stats = load_trained_model(
        args.checkpoint, use_mamba=args.use_mamba, device=args.device
    )
    if glcm_norm_value is not None:
        print(f"  glcm_norm_value from checkpoint: {glcm_norm_value:.4f}")
    else:
        print("  no glcm_norm_value in checkpoint -- computed per-tile from each image")
    if global_stats is not None:
        print(f"  global_stats from checkpoint: mean={global_stats['mean']} std={global_stats['std']}")
    else:
        print("  WARNING: no global_stats in checkpoint -- this checkpoint was saved before "
              "normalization stats were persisted. Predictions will likely be near-empty. "
              "Re-train/re-save with the updated script to fix this permanently.")

    if args.image:
        images = [args.image]
    else:
        images = []
        for fn in sorted(os.listdir(args.image_dir)):
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                images.append(os.path.join(args.image_dir, fn))
        print(f"Found {len(images)} tiles in {args.image_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    all_summaries = []
    for img_path in images:
        name = os.path.splitext(os.path.basename(img_path))[0]
        tile_out = os.path.join(args.out_dir, name)
        print(f"\n=== {name} ===")
        summary = run_one_tile(
            image_path=img_path,
            model=model,
            glcm_norm_value=glcm_norm_value,
            global_stats=global_stats,
            out_dir=tile_out,
            device=args.device,
            tile_size=args.tile_size,
            exclusion_band_px=args.exclusion_band_px,
            batch_size=args.batch_size,
            source_gsd_m=args.source_gsd_m,
            pixel_size_m=args.pixel_size_m,
            threshold=args.threshold,
            threshold_low=args.threshold_low,
            min_object_px=args.min_object_px,
            min_branch_m=args.min_branch_m,
            closing_iterations=args.closing_iterations,
            dp_epsilon_m=args.dp_epsilon_m,
            min_edge_mean_prob=args.min_edge_mean_prob,
            heal_max_search_radius_m=args.heal_max_search_radius_m,
            heal_occlusion_min=args.heal_occlusion_min,
            heal_angle_max_deg=args.heal_angle_max_deg,
            heal_gap_max_m=args.heal_gap_max_m,
            heal_max_iterations=args.heal_max_iterations,
            worldpop_raster=args.worldpop_raster,
            flood_polygon=args.flood_polygon,
            flood_target_crs=args.flood_polygon_crs,
            flood_source_crs=args.flood_polygon_source_crs,
            ground_truth_graph_path=args.ground_truth_graph,
            use_fast_glcm_proxy=args.use_fast_glcm_proxy,
        )
        # Print a compact summary rather than the full nested dict
        gs = summary["graph_stats"]
        print(f"  nodes={gs['reconstructed_nodes']} edges={gs['reconstructed_edges']} "
              f"components={gs['reconstructed_components']} "
              f"bridges={summary['healing']['bridges_added']} "
              f"RI={summary['resilience_index']}")
        all_summaries.append(summary)

    with open(os.path.join(args.out_dir, "batch_summary.json"), "w") as f:
        # strip polygon_coords across all tiles for the batch dump too
        clean = []
        for s in all_summaries:
            s2 = dict(s)
            if "scenarios" in s2:
                s2["scenarios"] = {
                    k: ({kk: vv for kk, vv in v.items() if kk != "polygon_coords"}
                        if isinstance(v, dict) else v)
                    for k, v in s2["scenarios"].items()
                }
            clean.append(s2)
        json.dump(clean, f, indent=2, default=str)
    print(f"\nDone. {len(all_summaries)} tiles processed. See {args.out_dir}/batch_summary.json")


if __name__ == "__main__":
    main()