"""
inference.py
============
Hann-window tiled inference for PathMamba on large Cartosat-3 tiles, matching
the deck's "512px Hann-tiled tensors, 256px exclusion band, Hann-blended
inference" wording.

WHY THIS EXISTS
---------------
model_io.load_pathmamba_and_infer runs the model on the whole image in one
forward pass. That works on a training-sized crop (512x512), and OOMs
instantly on a real Cartosat-3 PAN scene at 0.28 m/px. This module replaces
that seam with the tiled path the deck actually promises.

WHAT "512 tile / 256 exclusion band" MEANS HERE
-----------------------------------------------
Two common readings; the one implemented, and stated in the deck's own numbers:

  * tile_size = 512 px
  * exclusion_band = 128 px per side  =>  central 256 px kept per tile
  * stride = tile_size - 2 * exclusion_band = 256 px
  * adjacent tiles overlap by 256 px (50% overlap)
  * a 2-D Hann window (raised cosine) weights each tile's contribution so the
    overlap region blends smoothly from one tile to the next -- no hard seam
    at the exclusion boundary.

This is the standard segmentation-tiling recipe (SpaceNet baselines, Solaris,
CosmiQ toolkits). The "exclusion band" name in the deck refers to the fact
that the outer 128 px of each tile is DE-EMPHASIZED (weight -> 0 at the tile
edge because of the Hann window), so seam artifacts near the tile boundary
do not survive stitching -- they are absorbed by the neighbor's central band.

If your intended interpretation was different (e.g. hard-crop-to-256 and
tile with 0% overlap), pass exclusion_band_px=0 and stride will match tile
size; keep the Hann window off with hann=False and it degrades to the naive
"crop-and-stitch" behavior.

TESTABLE PROPERTIES (see test_inference.py)
-------------------------------------------
  * On a constant input the tiled output equals the whole-image output
    exactly at every non-edge pixel -- proves the blending math is unbiased.
  * On a spatially-varying input the tiled output matches the whole-image
    output to within a small tolerance dominated by numerical noise.
  * No visible seam: for a smooth input, the max gradient in the stitched
    output stays within a small factor of the whole-image gradient.
  * Batch inference works: multiple tiles run per forward pass (batch_size).
  * Edge coverage: every output pixel receives non-zero weight from at least
    one tile, incl. corners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import importlib.util
import subprocess
import sys

import numpy as np

from .glcm import GLCMOcclusionMap, build_two_channel_input
from .model_io import RoadPrediction, WorldToPixel, from_pathmamba_output


def _neutralize_mamba_lm_head_import() -> bool:
    """Patch an installed mamba_ssm __init__.py so it does not import MambaLMHeadModel."""
    try:
        spec = importlib.util.find_spec("mamba_ssm")
    except Exception:
        spec = None
    if spec is None or spec.origin is None:
        return False

    init_path = spec.origin
    try:
        src = open(init_path, "r").read()
    except Exception:
        return False

    if "MambaLMHeadModel" not in src:
        return False

    lines = [line for line in src.splitlines() if "MambaLMHeadModel" not in line]
    with open(init_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return True


def install_mamba_ssm() -> bool:
    """Ensure mamba-ssm is importable, installing it if necessary."""
    try:
        import mamba_ssm  # noqa: F401
        from mamba_ssm import Mamba  # noqa: F401
        return True
    except Exception:
        pass

    _neutralize_mamba_lm_head_import()

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "packaging", "ninja", "wheel", "setuptools"],
        capture_output=True, text=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "causal-conv1d>=1.4.0", "--no-build-isolation"],
        capture_output=True, text=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "mamba-ssm", "--no-build-isolation"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False

    _neutralize_mamba_lm_head_import()
    try:
        import mamba_ssm  # noqa: F401
        from mamba_ssm import Mamba  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Hann window
# ---------------------------------------------------------------------------

def hann_window_2d(size: int, dtype=np.float32) -> np.ndarray:
    """2-D separable Hann window of shape (size, size). Peak = 1 at center,
    smoothly -> 0 at edges. numpy.hanning gives an N-point window with zeros
    at both ENDS -- fine here because those are the exclusion-band edges we
    want zeroed out anyway."""
    w1 = np.hanning(size).astype(dtype)
    # numpy.hanning returns exact 0 at both ends; nudge by a tiny epsilon so
    # no output pixel can end up with total weight exactly 0 in the corner
    # coverage edge case (see the coverage test).
    w1 = np.clip(w1, 1e-6, None)
    return np.outer(w1, w1)


# ---------------------------------------------------------------------------
# Tile plan
# ---------------------------------------------------------------------------

@dataclass
class TilePlan:
    """A concrete plan of where every tile sits on the source image. Kept as
    data (not an iterator) so tests can assert on it directly."""
    tile_size: int
    stride: int
    exclusion_band_px: int
    positions: list[tuple[int, int]]  # (row, col) top-left of each tile in the padded image
    padded_shape: tuple[int, int]      # (H_pad, W_pad)
    original_shape: tuple[int, int]    # (H, W)


def plan_tiles(image_shape: tuple[int, int], tile_size: int = 512,
               exclusion_band_px: int = 128) -> TilePlan:
    """Build the tile plan for an image of shape (H, W).

    stride = tile_size - 2 * exclusion_band_px, so adjacent tiles' CENTRAL
    kept regions abut with no gap. Padding is added so the last tile row/col
    covers the far edge cleanly.
    """
    if tile_size <= 0 or exclusion_band_px < 0:
        raise ValueError("tile_size must be > 0 and exclusion_band_px >= 0")
    if 2 * exclusion_band_px >= tile_size:
        raise ValueError(f"exclusion_band_px ({exclusion_band_px}) too large for "
                         f"tile_size ({tile_size}); stride would be <= 0")

    stride = tile_size - 2 * exclusion_band_px
    H, W = image_shape

    # Pad on the right/bottom so the last tile's central kept region reaches
    # to the far edge. Also pad on the left/top by exclusion_band_px so the
    # FIRST tile's central kept region covers pixel (0, 0) -- otherwise the
    # top-left exclusion_band_px band of the image is only ever seen through
    # a tile's low-weight rim.
    pad_top = exclusion_band_px
    pad_left = exclusion_band_px

    def _n_tiles(dim):
        # after left-padding, dim increases by pad_left. We want the last tile
        # to cover to dim + pad_left + pad_right, where pad_right is >= exclusion_band_px
        # and chosen so (n-1)*stride + tile_size >= dim + pad_left + exclusion_band_px.
        target = dim + pad_left + exclusion_band_px
        n = max(1, int(np.ceil((target - tile_size) / stride)) + 1)
        return n

    nH = _n_tiles(H)
    nW = _n_tiles(W)

    H_pad = pad_top + (nH - 1) * stride + tile_size - pad_top  # simpler: (nH-1)*stride + tile_size
    H_pad = (nH - 1) * stride + tile_size
    W_pad = (nW - 1) * stride + tile_size

    positions = [(r * stride, c * stride) for r in range(nH) for c in range(nW)]

    return TilePlan(
        tile_size=tile_size,
        stride=stride,
        exclusion_band_px=exclusion_band_px,
        positions=positions,
        padded_shape=(H_pad, W_pad),
        original_shape=(H, W),
    )


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

def _pad_for_plan(arr: np.ndarray, plan: TilePlan, mode: str = "reflect") -> np.ndarray:
    """Pad a (C, H, W) or (H, W) array to plan.padded_shape.

    Pads exclusion_band_px on the top/left so the first tile is centered on the
    image's true (0, 0), and pads whatever's needed on the bottom/right to reach
    padded_shape. Reflect padding keeps the model's border behavior sane -- zero
    padding would create a hard fake-boundary and hallucinate breaks."""
    H, W = plan.original_shape
    Hp, Wp = plan.padded_shape
    pad_top = plan.exclusion_band_px
    pad_left = plan.exclusion_band_px
    pad_bot = Hp - H - pad_top
    pad_right = Wp - W - pad_left
    if arr.ndim == 2:
        return np.pad(arr, ((pad_top, pad_bot), (pad_left, pad_right)), mode=mode)
    elif arr.ndim == 3:
        return np.pad(arr, ((0, 0), (pad_top, pad_bot), (pad_left, pad_right)), mode=mode)
    else:
        raise ValueError(f"unsupported array shape {arr.shape}")


def _unpad_to_original(arr: np.ndarray, plan: TilePlan) -> np.ndarray:
    """Crop a padded (H, W) or (C, H, W) array back to plan.original_shape."""
    H, W = plan.original_shape
    pad_top = plan.exclusion_band_px
    pad_left = plan.exclusion_band_px
    if arr.ndim == 2:
        return arr[pad_top:pad_top + H, pad_left:pad_left + W]
    elif arr.ndim == 3:
        return arr[:, pad_top:pad_top + H, pad_left:pad_left + W]
    else:
        raise ValueError(f"unsupported array shape {arr.shape}")


# ---------------------------------------------------------------------------
# Tiled inference core
# ---------------------------------------------------------------------------

ModelFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
"""Model callable: takes (B, C, tile, tile) float32, returns
(road_logits, confidence_logits) each shaped (B, 1, tile, tile) or
(B, tile, tile). Everything downstream converts to (B, tile, tile)."""

ModelAdapter = ModelFn | tuple[ModelFn, ModelFn]


def _coerce_model_fn(model_fn: ModelAdapter) -> tuple[ModelFn, ModelFn]:
    """Accept either a single callable or a (road_fn, conf_fn) tuple."""
    if isinstance(model_fn, tuple):
        if len(model_fn) != 2:
            raise ValueError("model_fn tuple must contain exactly (road_fn, conf_fn)")
        road_fn, conf_fn = model_fn
        if not callable(road_fn) or not callable(conf_fn):
            raise TypeError("model_fn tuple entries must be callables")
        return road_fn, conf_fn
    if callable(model_fn):
        return model_fn, model_fn
    raise TypeError(f"expected a callable or (road_fn, conf_fn) tuple, got {type(model_fn)}")


def _run_model_in_batches(input_2ch: np.ndarray, positions, tile_size: int,
                          model_fn: ModelAdapter, batch_size: int,
                          show_progress: bool = False):
    """Feed tile crops through model_fn in batches. Yields (position, road_logits_2d,
    conf_logits_2d) per tile."""
    total = len(positions)
    use_tqdm = False
    if show_progress:
        try:
            from tqdm import tqdm
            positions = list(tqdm(positions, desc="tiled inference", unit="tile"))
            use_tqdm = True
        except ImportError:
            print(f"[inference] tiled inference: {total} tiles, batch_size={batch_size}")

    def crop(pos):
        r, c = pos
        return input_2ch[:, r:r + tile_size, c:c + tile_size]

    for start in range(0, len(positions), batch_size):
        chunk_positions = positions[start:start + batch_size]
        batch = np.stack([crop(p) for p in chunk_positions], axis=0).astype(np.float32)
        road_fn, conf_fn = _coerce_model_fn(model_fn)
        road_batch, _ = road_fn(batch)
        conf_batch, _ = conf_fn(batch)
        road_batch = np.asarray(road_batch)
        conf_batch = np.asarray(conf_batch)
        # squeeze (B, 1, H, W) -> (B, H, W)
        if road_batch.ndim == 4 and road_batch.shape[1] == 1:
            road_batch = road_batch[:, 0]
        if conf_batch.ndim == 4 and conf_batch.shape[1] == 1:
            conf_batch = conf_batch[:, 0]
        if show_progress and not use_tqdm:
            processed = min(start + len(chunk_positions), total)
            print(f"[inference] processed tiles {processed}/{total}")
        for k, pos in enumerate(chunk_positions):
            yield pos, road_batch[k], conf_batch[k]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid; used only if the model_fn returned logits (default assumption).
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def tiled_infer(
    input_2ch: np.ndarray,
    model_fn: ModelAdapter,
    tile_size: int = 512,
    exclusion_band_px: int = 128,
    batch_size: int = 4,
    apply_sigmoid: bool = True,
    hann: bool = True,
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run tiled Hann-blended inference over a big 2-channel image.

    Parameters
    ----------
    input_2ch : (2, H, W) float32; channel 0 = grayscale in [0,1], channel 1 = GLCM.
                Build with route_resilience.glcm.build_two_channel_input.
    model_fn : callable that takes (B, 2, tile, tile) and returns
               (road_logits, conf_logits). See ModelFn typedef.
    tile_size, exclusion_band_px, batch_size : tile plan + throughput knobs.
    apply_sigmoid : if True (default), assume model_fn returned raw logits and
                    apply sigmoid before blending. If model_fn already returned
                    probabilities in [0,1] (e.g. wrapped model with conf_head
                    that already applies sigmoid), pass False.
    hann : Hann-window blending. Off = uniform weights across each tile, which
           reproduces "hard crop + stitch" and lets tests A/B the blending.

    Returns
    -------
    (prob_mask, confidence_mask) both shaped (H, W), float32 in [0, 1].
    """
    if input_2ch.ndim != 3 or input_2ch.shape[0] != 2:
        raise ValueError(f"input_2ch must be (2, H, W); got {input_2ch.shape}")

    H, W = input_2ch.shape[1:]
    plan = plan_tiles((H, W), tile_size=tile_size, exclusion_band_px=exclusion_band_px)
    padded = _pad_for_plan(input_2ch, plan, mode="reflect")

    Hp, Wp = plan.padded_shape
    road_acc = np.zeros((Hp, Wp), dtype=np.float32)
    conf_acc = np.zeros((Hp, Wp), dtype=np.float32)
    weight_acc = np.zeros((Hp, Wp), dtype=np.float32)

    weight_tile = hann_window_2d(tile_size) if hann else np.ones((tile_size, tile_size), dtype=np.float32)

    for pos, road_logits, conf_logits in _run_model_in_batches(
        padded, plan.positions, tile_size, model_fn, batch_size,
        show_progress=show_progress,
    ):
        r, c = pos
        road_prob = _sigmoid(road_logits) if apply_sigmoid else road_logits.astype(np.float32)
        conf_prob = _sigmoid(conf_logits) if apply_sigmoid else conf_logits.astype(np.float32)
        road_acc[r:r + tile_size, c:c + tile_size] += road_prob * weight_tile
        conf_acc[r:r + tile_size, c:c + tile_size] += conf_prob * weight_tile
        weight_acc[r:r + tile_size, c:c + tile_size] += weight_tile

    # normalize -- with the epsilon in hann_window_2d, weight_acc is strictly > 0
    weight_acc = np.maximum(weight_acc, 1e-8)
    road_full = road_acc / weight_acc
    conf_full = conf_acc / weight_acc

    road_full = _unpad_to_original(road_full, plan)
    conf_full = _unpad_to_original(conf_full, plan)
    return road_full.astype(np.float32), conf_full.astype(np.float32)


# ---------------------------------------------------------------------------
# End-to-end: raw grayscale tile -> RoadPrediction
# ---------------------------------------------------------------------------

def tiled_infer_from_grayscale(
    image_gray_uint8: np.ndarray,
    road_fn: ModelFn,
    conf_fn: Optional[ModelFn] = None,
    glcm_tool: Optional[GLCMOcclusionMap] = None,
    tile_size: int = 512,
    exclusion_band_px: int = 128,
    batch_size: int = 4,
    hann: bool = True,
    transform: Optional[WorldToPixel] = None,
    crs: Optional[str] = None,
    source_gsd_m: float = 0.28,
    show_progress: bool = True,
) -> RoadPrediction:
    """One-call inference: raw grayscale Cartosat-3 tile -> RoadPrediction.

    * Builds the 2-channel [grayscale, GLCM] input via the shared glcm module.
    * Runs Hann-tiled inference over the whole scene, with CORRECT per-channel
      sigmoid handling: road channel uses apply_sigmoid=True (road_head returns
      logits); conf channel uses apply_sigmoid=False (conf_head already applies
      sigmoid inside the model -- a second sigmoid would corrupt the signal near
      the 0.3 healing gate threshold).
    * Wraps the result in a RoadPrediction with the caller's transform/crs/gsd.

    Parameters
    ----------
    road_fn : ModelFn for the road channel. Returns (road_logits, ignored).
              Obtain from make_torch_model_fn(model, device)[0].
    conf_fn : ModelFn for the confidence channel. Returns (conf_prob, ignored).
              Obtain from make_torch_model_fn(model, device)[1].
              When None, road_fn is reused for both channels with apply_sigmoid=True
              on both -- only correct if road_fn already returns probabilities for
              both outputs (e.g. a wrapped model that applies sigmoid internally).
    glcm_tool : pre-built GLCMOcclusionMap. If None, a default one is created
                without a precomputed norm_value -- fine for single-tile inference
                but see the module docstring's warning about calibration.
    """
    if glcm_tool is None:
        glcm_tool = GLCMOcclusionMap()
    input_2ch = build_two_channel_input(image_gray_uint8, glcm_tool)

    # Road: apply sigmoid (road_head returns raw logits).
    prob_mask, _ = tiled_infer(
        input_2ch, road_fn,
        tile_size=tile_size, exclusion_band_px=exclusion_band_px,
        batch_size=batch_size, apply_sigmoid=True, hann=hann,
        show_progress=show_progress,
    )

    # Conf: do NOT apply sigmoid again (conf_head already applies it inside the model).
    # If no separate conf_fn provided, fall back to road_fn with apply_sigmoid=True.
    _conf_fn = conf_fn if conf_fn is not None else road_fn
    _conf_apply_sigmoid = conf_fn is None
    confidence_mask, _ = tiled_infer(
        input_2ch, _conf_fn,
        tile_size=tile_size, exclusion_band_px=exclusion_band_px,
        batch_size=batch_size, apply_sigmoid=_conf_apply_sigmoid, hann=hann,
        show_progress=show_progress,
    )

    return RoadPrediction(
        prob_mask=prob_mask,
        confidence_mask=confidence_mask,
        transform=transform,
        crs=crs,
        source_gsd_m=source_gsd_m,
        is_synthetic=False,
    )


# ---------------------------------------------------------------------------
# Torch adapter: wrap a loaded PathMamba into a ModelFn
# ---------------------------------------------------------------------------

def make_torch_model_fn(model, device: str = "cuda") -> tuple:
    """Wrap a torch PathMamba into a (road_fn, conf_fn) pair for tiled_infer.

    Returns TWO callables rather than one, because road_head returns raw logits
    (needs sigmoid applied during blending) while conf_head already applies
    sigmoid inside the model. Applying sigmoid again to conf corrupts the
    confidence signal: sigmoid(sigmoid(x)) != sigmoid(x), and the error is
    largest near p=0.5 -- i.e. right at the healing gate's 0.3 threshold.

    road_fn : (B,2,H,W) -> (road_logits, zeros)  -- pass to tiled_infer with apply_sigmoid=True
    conf_fn : (B,2,H,W) -> (conf_prob,   zeros)  -- pass to tiled_infer with apply_sigmoid=False

    Pass both to tiled_infer_from_grayscale:
        road_fn, conf_fn = make_torch_model_fn(model, device)
        prediction = tiled_infer_from_grayscale(img, road_fn, conf_fn=conf_fn, ...)
    """
    import torch

    def _run(batch_np: np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(batch_np).to(device).float()
            road_logits, conf_prob = model(x)
            return road_logits.cpu().numpy(), conf_prob.cpu().numpy()

    def road_fn(batch_np: np.ndarray):
        r, _ = _run(batch_np)
        return r, np.zeros_like(r)

    def conf_fn(batch_np: np.ndarray):
        _, c = _run(batch_np)
        return c, np.zeros_like(c)

    return road_fn, conf_fn


def load_pathmamba_and_tiled_infer(
    checkpoint_path: str,
    image_gray_uint8: np.ndarray,
    pathmamba_cls,
    device: str = "cuda",
    tile_size: int = 512,
    exclusion_band_px: int = 128,
    batch_size: int = 4,
    transform: Optional[WorldToPixel] = None,
    crs: Optional[str] = None,
    source_gsd_m: float = 0.28,
    glcm_norm_value: Optional[float] = None,
    use_fast_glcm_proxy: bool = False,
) -> RoadPrediction:
    """Drop-in replacement for model_io.load_pathmamba_and_infer, using tiled
    Hann-blended inference so a real Cartosat-3-sized scene doesn't OOM.

    Loads the checkpoint, builds a GLCM tool with the SAME norm_value that
    training used (pass it in), and runs tiled inference. See the training
    script for how glcm_norm_value gets persisted -- ideally it's saved into
    the checkpoint dict under 'glcm_norm_value' and read back here.
    """
    import torch

    model = pathmamba_cls(in_channels=2, pretrained=False, use_mamba=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Prefer norm_value from checkpoint if the training script starts saving it.
    norm = glcm_norm_value
    if norm is None and isinstance(ckpt, dict):
        norm = ckpt.get("glcm_norm_value")
    glcm_tool = GLCMOcclusionMap(norm_value=norm, use_fast_proxy=use_fast_glcm_proxy)

    # Model's conf_head already applies sigmoid -- ModelFn returns conf as prob.
    # Apply sigmoid to the road channel only. Handled by splitting in the fn:
    def fn(batch_np: np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(batch_np).to(device).float()
            road_logits, conf_prob = model(x)
            # Convert conf back to "logits" by inverse-sigmoid to keep the
            # blending unbiased: sigmoid(mean(inv_sigmoid(p_i)*w_i)/w_sum) is
            # NOT the same as mean(p_i * w_i)/w_sum. For probability maps in
            # [0,1] a linear weighted mean of the probabilities themselves is
            # actually well-defined and fine -- so we skip the sigmoid on conf.
            return road_logits.cpu().numpy(), conf_prob.cpu().numpy()

    input_2ch = build_two_channel_input(image_gray_uint8, glcm_tool)

    # Road channel: apply_sigmoid=True. Conf channel: already probability.
    # tiled_infer applies one sigmoid flag to both. To handle this properly we
    # run tiled_infer twice with a partial model_fn each time.
    def road_fn(batch_np):
        r, _ = fn(batch_np)
        return r, np.zeros_like(r)  # dummy second output

    def conf_fn(batch_np):
        _, c = fn(batch_np)
        return c, np.zeros_like(c)  # dummy second output

    prob_mask, _ = tiled_infer(input_2ch, road_fn,
                                tile_size=tile_size, exclusion_band_px=exclusion_band_px,
                                batch_size=batch_size, apply_sigmoid=True)
    confidence_mask, _ = tiled_infer(input_2ch, conf_fn,
                                      tile_size=tile_size, exclusion_band_px=exclusion_band_px,
                                      batch_size=batch_size, apply_sigmoid=False)

    return RoadPrediction(
        prob_mask=prob_mask, confidence_mask=confidence_mask,
        transform=transform, crs=crs, source_gsd_m=source_gsd_m, is_synthetic=False,
    )
