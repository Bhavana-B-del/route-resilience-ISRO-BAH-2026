"""
glcm.py
=======
Shared GLCM (Gray-Level Co-occurrence Matrix) occlusion-map computation.

This is the SAME implementation currently living inside train_pathmamba_v2.py,
extracted here so hackathon-day inference can build the 2-channel
[grayscale, GLCM] input without importing the training script (which pulls in
timm/torch/mamba-ssm and a whole environment we don't want at inference).

MIGRATION NOTE FOR TRAINING SCRIPT
----------------------------------
Once verified, train_pathmamba_v2.py should replace its local definitions of:
    GLCMOcclusionMap
    GLCM_FIXED_WINDOW
    compute_glcm_norm_value
    precompute_glcm_cache
    load_glcm_from_cache_or_compute
with `from route_resilience.glcm import ...`. Until then, this module and the
training script hold identical logic -- do NOT diverge them silently. The
class contract (window sizes, four-angle GLCM, stride-4 grid, 32 gray levels,
percentile-99 shared normalization) is locked, per the "prior decisions" notes.

WHY THE SHARED, PRECOMPUTED NORM MATTERS
----------------------------------------
_norm_value MUST be a shared, precomputed constant passed to every worker
(see compute_glcm_norm_value). Never let it default to per-instance lazy
init: DataLoader workers fork their own copies, each anchoring its own norm
to whatever image it happened to see first, which silently drifts the
"same" transform across workers. At inference this matters less (single
process) but keep the discipline for consistency.
"""

from __future__ import annotations

import os
import random
from typing import Optional, Callable

import numpy as np

# The one training-time constant that's also relevant at inference: the fixed
# window used for the shared-norm computation. Kept as a module-level constant
# so callers can reference it symbolically instead of magic-numbering 5.
GLCM_FIXED_WINDOW = 5


class GLCMOcclusionMap:
    """Multi-angle GLCM occlusion score (contrast and inverse-homogeneity),
    averaged over 4 angles (0, 45, 90, 135 deg), on a stride-4 sliding window
    over a 32-gray-level quantized image, upsampled bilinearly.

    Parameters
    ----------
    window : sliding-window size in pixels.
    norm_value : precomputed 99th-percentile constant (see
                 compute_glcm_norm_value). If None, the first call to compute()
                 sets it from that first image's own 99th percentile -- OK for
                 inference on a single tile, DANGEROUS in multi-worker training.
    gray_levels : quantization levels for the co-occurrence matrix (32 -> 32x32
                  GLCMs; the speed win vs. the naive 256x256).
    use_fast_proxy : emergency fallback that skips real GLCM in favor of the
                     old uniform-filter local-variance proxy. Off by default.
                     Controlled at the caller level by USE_FAST_GLCM_PROXY=1.
    """

    def __init__(self, window: int = 11, norm_value: Optional[float] = None,
                 gray_levels: int = 32, use_fast_proxy: bool = False):
        self.window = window
        self._norm_value = norm_value
        self.gray_levels = gray_levels
        self.use_fast_proxy = use_fast_proxy

    def compute(self, image_gray_uint8: np.ndarray) -> np.ndarray:
        result = self.compute_raw(image_gray_uint8)
        if self._norm_value is None:
            self._norm_value = float(np.percentile(result, 99)) if result.max() > 0 else 1.0
        return np.clip(result / max(self._norm_value, 1e-6), 0, 1).astype(np.float32)

    def compute_raw(self, image_gray_uint8: np.ndarray) -> np.ndarray:
        """Unclipped, unnormalized signal -- for calibration ONLY
        (compute_glcm_norm_value). Bypasses norm_value/clipping entirely.

        BUG THIS EXISTS TO FIX: compute_glcm_norm_value() used to call
        compute() with _norm_value temporarily set to 1.0, intending to get
        "raw" output. But compute() always clips to [0, norm_value] as its
        last step -- setting norm_value=1.0 doesn't disable that clip, it
        just makes the clip ceiling 1.0. Real GLCM contrast values routinely
        exceed 1.0 in raw units, so nearly every sampled pixel got slammed to
        exactly 1.0 BEFORE the percentile was computed -- meaning the
        "calibration" trivially returned ~1.0 every time, regardless of image
        content. This module's docstring says to keep this in sync with
        train_pathmamba_v2.py's local copy of the same logic -- that copy was
        fixed first; this was the un-synced half of the same bug.
        """
        if self.use_fast_proxy:
            return self._compute_fast_proxy_raw(image_gray_uint8)
        return self._compute_real_glcm_raw(image_gray_uint8)

    def _compute_fast_proxy(self, image_gray_uint8: np.ndarray) -> np.ndarray:
        result = self._compute_fast_proxy_raw(image_gray_uint8)
        if self._norm_value is None:
            self._norm_value = float(np.percentile(result, 99)) if result.max() > 0 else 1.0
        return np.clip(result / max(self._norm_value, 1e-6), 0, 1).astype(np.float32)

    def _compute_fast_proxy_raw(self, image_gray_uint8: np.ndarray) -> np.ndarray:
        from scipy.ndimage import uniform_filter

        img = image_gray_uint8.astype(np.float32)
        mean = uniform_filter(img, size=self.window)
        mean_sq = uniform_filter(img ** 2, size=self.window)
        return np.clip(mean_sq - mean ** 2, 0, None)

    def _compute_real_glcm(self, image_gray_uint8: np.ndarray) -> np.ndarray:
        result = self._compute_real_glcm_raw(image_gray_uint8)
        if self._norm_value is None:
            self._norm_value = float(np.percentile(result, 99)) if result.max() > 0 else 1.0
        return np.clip(result / max(self._norm_value, 1e-6), 0, 1).astype(np.float32)

    def _compute_real_glcm_raw(self, image_gray_uint8: np.ndarray) -> np.ndarray:
        from skimage.feature import graycomatrix, graycoprops

        img = image_gray_uint8.astype(np.float32)
        img_q = (img / 256.0 * self.gray_levels).astype(np.uint8)
        img_q = np.clip(img_q, 0, self.gray_levels - 1)

        H, W = img_q.shape
        w = self.window
        pad = w // 2
        padded = np.pad(img_q, pad, mode="reflect")
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        # stride-4 sliding-window grid, then bilinear upsample. See the training
        # script's comment for the rationale: road-vs-canopy is a texture-scale
        # phenomenon, so a 4-px grid loses ~nothing while running 16x faster.
        stride = 4
        grid_h = (H + stride - 1) // stride
        grid_w = (W + stride - 1) // stride
        out_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

        for gi, i in enumerate(range(0, H, stride)):
            for gj, j in enumerate(range(0, W, stride)):
                patch = padded[i:i + w, j:j + w]
                glcm = graycomatrix(patch, distances=[1], angles=angles,
                                     levels=self.gray_levels, symmetric=True, normed=True)
                contrast = graycoprops(glcm, "contrast").mean()
                homogeneity = graycoprops(glcm, "homogeneity").mean()
                out_grid[gi, gj] = float(contrast) * (1.0 - float(homogeneity))

        from scipy.ndimage import zoom
        result = zoom(out_grid, (H / grid_h, W / grid_w), order=1)
        result = result[:H, :W]
        if result.shape != (H, W):
            result = np.pad(result, ((0, H - result.shape[0]), (0, W - result.shape[1])), mode="edge")
        return result


# ---------------------------------------------------------------------------
# Batch helpers -- used at training time and (compute_glcm_norm_value) at
# inference time on a small calibration set if norm_value isn't already known.
# ---------------------------------------------------------------------------

def _default_logger(msg: str) -> None:
    print(msg)


def precompute_glcm_cache(pairs, glcm_tool: GLCMOcclusionMap, force: bool = False,
                          log: Callable[[str], None] = _default_logger) -> None:
    """Precompute and cache `<image>.glcm.npy` alongside each image so training
    workers load precomputed maps instead of recomputing. Idempotent."""
    from PIL import Image

    computed, skipped = 0, 0
    for img_path, _ in pairs:
        cache_path = img_path + ".glcm.npy"
        if os.path.exists(cache_path) and not force:
            skipped += 1
            continue
        try:
            arr = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
            o_map = glcm_tool.compute(arr)
            np.save(cache_path, o_map)
            computed += 1
            if computed % 100 == 0:
                log(f"  GLCM cache: {computed} images computed so far")
        except Exception as e:
            log(f"  GLCM cache: failed on {img_path}: {e!r}")
    log(f"GLCM cache: {computed} computed, {skipped} already cached ({computed + skipped} total).")


def load_glcm_from_cache_or_compute(img_path: str, image_gray_uint8: np.ndarray,
                                    glcm_tool: GLCMOcclusionMap) -> np.ndarray:
    cache_path = img_path + ".glcm.npy"
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path).astype(np.float32)
        except Exception:
            pass
    return glcm_tool.compute(image_gray_uint8)


def compute_glcm_norm_value(pairs, glcm_tool: GLCMOcclusionMap, n_samples: int = 50,
                            log: Callable[[str], None] = _default_logger,
                            rng: Optional[random.Random] = None) -> float:
    """One fixed 99th-percentile constant across a representative sample of the
    training pool. rng lets callers pass a seeded random.Random(42) to keep the
    sample deterministic across runs -- important for reproducibility, and
    consistent with the SpaceNet split policy."""
    from PIL import Image

    r = rng or random
    sample = r.sample(pairs, min(n_samples, len(pairs)))
    all_values = []
    for img_path, _ in sample:
        arr = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        arr_small = arr[::4, ::4]
        # compute_raw(), NOT compute() -- compute() always clips to
        # [0, norm_value], so it can never reveal the true dynamic range
        # calibration needs. See compute_raw()'s docstring.
        raw_output = glcm_tool.compute_raw(arr_small)
        all_values.append(raw_output.ravel())

    combined = np.concatenate(all_values)
    norm_value = float(np.percentile(combined, 99)) if combined.max() > 0 else 1.0
    log(f"Fixed GLCM norm_value = {norm_value:.4f} (computed over {len(sample)} sampled images)")
    return norm_value


# ---------------------------------------------------------------------------
# Inference-time convenience: build the 2-channel [grayscale, GLCM] input that
# PathMamba expects, from a raw grayscale or RGB tile.
# ---------------------------------------------------------------------------

def build_two_channel_input(image_gray_uint8: np.ndarray,
                             glcm_tool: GLCMOcclusionMap) -> np.ndarray:
    """Returns a (2, H, W) float32 array: channel 0 = grayscale in [0,1],
    channel 1 = GLCM occlusion map in [0,1]. This is the exact input shape
    PathMamba is trained on."""
    if image_gray_uint8.ndim != 2:
        raise ValueError(f"expected 2D grayscale uint8; got shape {image_gray_uint8.shape}")
    gray = image_gray_uint8.astype(np.float32) / 255.0
    occ = glcm_tool.compute(image_gray_uint8)
    return np.stack([gray, occ], axis=0).astype(np.float32)
