"""
geo_utils.py
============
Minimal world<->pixel transform so we can test sampling logic (healing.py's
occlusion-confidence gate) today, without a rasterio dependency and without a
real georeferenced tile. Real tiles later can use RasterioAffineAdapter, which
wraps an actual `rasterio` Affine so healing.py doesn't need to change at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimpleAffine:
    """Maps local (x, y) meters -> (row, col) pixel indices for a north-up raster.

    origin_x, origin_y : world coords of the array's bottom-left corner (row = height_px)
    pixel_size          : meters per pixel (isotropic)
    height_px           : raster height in pixels, needed to flip Y (world Y increases
                           upward, array rows increase downward)
    """

    origin_x: float
    origin_y: float
    pixel_size: float
    height_px: int

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        col = (x - self.origin_x) / self.pixel_size
        row = self.height_px - (y - self.origin_y) / self.pixel_size
        return row, col


class RasterioAffineAdapter:
    """Wraps a real rasterio.transform.Affine so it satisfies the same
    `.world_to_pixel(x, y)` protocol as SimpleAffine. Use this once real
    Cartosat-3 tiles with real georeferencing exist -- nothing else changes.
    """

    def __init__(self, rasterio_affine):
        self._inv = ~rasterio_affine  # rasterio Affine supports ~ for inverse

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        col, row = self._inv * (x, y)
        return row, col


def sample_bilinear(arr: np.ndarray, row: float, col: float) -> float:
    """Bilinear sample of a 2D array at fractional (row, col). Clamps to bounds
    instead of raising, since candidate bridge lines can graze a tile edge."""
    h, w = arr.shape
    row = min(max(row, 0.0), h - 1.0)
    col = min(max(col, 0.0), w - 1.0)
    r0, c0 = int(np.floor(row)), int(np.floor(col))
    r1, c1 = min(r0 + 1, h - 1), min(c0 + 1, w - 1)
    fr, fc = row - r0, col - c0
    top = arr[r0, c0] * (1 - fc) + arr[r0, c1] * fc
    bot = arr[r1, c0] * (1 - fc) + arr[r1, c1] * fc
    return float(top * (1 - fr) + bot * fr)


def sample_along_line(
    arr: np.ndarray, transform, x0: float, y0: float, x1: float, y1: float, n_samples: int = 10
) -> np.ndarray:
    """Samples `arr` at n_samples evenly spaced world-coord points between
    (x0, y0) and (x1, y1), using `transform.world_to_pixel`. If transform is
    None, assumes world coords already equal pixel coords (pure synthetic mode)."""
    xs = np.linspace(x0, x1, n_samples)
    ys = np.linspace(y0, y1, n_samples)
    out = np.empty(n_samples, dtype=np.float32)
    for i, (x, y) in enumerate(zip(xs, ys)):
        if transform is None:
            row, col = y, x
        else:
            row, col = transform.world_to_pixel(x, y)
        out[i] = sample_bilinear(arr, row, col)
    return out


def sample_along_corridor(
    arr: np.ndarray, transform, x0: float, y0: float, x1: float, y1: float,
    n_samples: int = 10, corridor_half_width_m: float = 2.0, n_cross: int = 3,
) -> np.ndarray:
    """Samples `arr` along a fat line: at each of n_samples points on the
    centerline, take n_cross perpendicular samples spanning +/- corridor_half_width_m,
    and reduce each cross-section by its MAX (per-column occlusion evidence
    -- if the model is confident at ANY point across the corridor, that counts).
    Returns an (n_samples,) array of per-centerline-point max values.

    Why: healing's occlusion gate on a 1-px-wide sample is a coinflip when the
    candidate skims the edge of a low-confidence pixel. Real occlusion (canopy)
    has non-trivial cross-road extent -- a 3-5 px wide corridor is a better
    match for what the model actually saw.

    NOTE: This is the ORIGINAL max-based sampler. See sample_along_corridor_robust
    for the newer median-based version that's less fooled by single-pixel spikes
    (e.g. a bright building edge crossing the bridge path).
    """
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    if L == 0.0:
        return sample_along_line(arr, transform, x0, y0, x1, y1, n_samples)
    # unit normal (perpendicular) to the centerline
    nx_u, ny_u = -dy / L, dx / L
    xs_c = np.linspace(x0, x1, n_samples)
    ys_c = np.linspace(y0, y1, n_samples)
    offsets = np.linspace(-corridor_half_width_m, corridor_half_width_m, n_cross)
    out = np.empty(n_samples, dtype=np.float32)
    for i, (xc, yc) in enumerate(zip(xs_c, ys_c)):
        cross_vals = []
        for off in offsets:
            xw, yw = xc + nx_u * off, yc + ny_u * off
            if transform is None:
                row, col = yw, xw
            else:
                row, col = transform.world_to_pixel(xw, yw)
            cross_vals.append(sample_bilinear(arr, row, col))
        out[i] = max(cross_vals)
    return out


def sample_along_corridor_robust(
    arr: np.ndarray, transform, x0: float, y0: float, x1: float, y1: float,
    n_samples: int = 10, corridor_half_width_m: float = 2.0, n_cross: int = 3,
) -> np.ndarray:
    """Robust variant of sample_along_corridor. Two differences that matter
    for the healing occlusion gate on real satellite imagery:

    1. Perpendicular reduction is MEDIAN, not MAX.
       On real data a single bright pixel in the corridor (a specular building
       corner, a bright car, a rooftop crossing the bridge line) can dominate
       MAX and make a candidate look supported even when the corridor is
       mostly empty. MEDIAN needs at least half of the perpendicular samples
       to agree -- kills those single-pixel outliers.

    2. Caller is expected to reduce the returned per-centerline-point array
       via FRACTION-ABOVE-THRESHOLD rather than MEAN.
       On the current (max, mean) recipe, one 0.9 point can pull the mean
       above the gate even if the other nine points are 0.05. On the new
       (median, fraction-above-threshold) recipe, that same corridor scores
       1/10 = 0.10, which correctly fails a reasonable gate.

    Combined effect: the score becomes "what fraction of the centerline has
    sustained evidence" instead of "how much bright-pixel noise did the
    corridor happen to catch." Same [0, 1] range so slider values still make
    sense, but the meaning is more defensible.

    Returns an (n_samples,) array of per-centerline-point median values.
    The caller (healing.heal) is expected to threshold each point and take
    the fraction above threshold as the final gate score.
    """
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    if L == 0.0:
        return sample_along_line(arr, transform, x0, y0, x1, y1, n_samples)
    nx_u, ny_u = -dy / L, dx / L
    xs_c = np.linspace(x0, x1, n_samples)
    ys_c = np.linspace(y0, y1, n_samples)
    offsets = np.linspace(-corridor_half_width_m, corridor_half_width_m, n_cross)
    out = np.empty(n_samples, dtype=np.float32)
    for i, (xc, yc) in enumerate(zip(xs_c, ys_c)):
        cross_vals = []
        for off in offsets:
            xw, yw = xc + nx_u * off, yc + ny_u * off
            if transform is None:
                row, col = yw, xw
            else:
                row, col = transform.world_to_pixel(xw, yw)
            cross_vals.append(sample_bilinear(arr, row, col))
        out[i] = float(np.median(cross_vals))
    return out
