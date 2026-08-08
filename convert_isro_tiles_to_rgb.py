"""
convert_isro_tiles_to_rgb.py
=============================
Converts patchify.py's raw GeoTIFF tile output (4-band NIR,R,G,B, UINT16)
into standard RGB PNG, dropping the NIR band, and discards tiles that
straddle the edge of the satellite swath (partial/incomplete coverage).
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import rasterio
from PIL import Image


def percentile_normalize_to_uint8(band: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    lo, hi = np.percentile(band, [p_low, p_high])
    if hi - lo < 1e-6:
        return np.zeros_like(band, dtype=np.uint8)
    return np.clip((band.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def tile_is_complete(raw: np.ndarray, max_nodata_fraction: float = 0.01) -> bool:
    """Discard tiles straddling the swath edge -- true nodata is exactly
    zero across ALL bands simultaneously; a genuinely dark real feature
    (shadow, water) still shows cross-band variation, so this won't
    accidentally discard legitimate dark tiles."""
    all_zero = np.all(raw == 0, axis=0)
    nodata_fraction = all_zero.mean()
    return nodata_fraction <= max_nodata_fraction


def convert_tile(tif_path: str, png_path: str, band_order: str = "nir_r_g_b") -> bool:
    with rasterio.open(tif_path) as src:
        raw = src.read()
        n_bands = raw.shape[0]

        if not tile_is_complete(raw):
            return False

        if band_order == "nir_r_g_b" and n_bands >= 4:
            r, g, b = raw[1], raw[2], raw[3]
        elif band_order == "nir_r_g" and n_bands == 3:
            r, g = raw[1], raw[2]
            b = ((raw[1].astype(np.float32) + raw[2].astype(np.float32)) / 2 * 0.7).astype(raw.dtype)
        elif n_bands >= 3:
            r, g, b = raw[0], raw[1], raw[2]
        else:
            raise ValueError(f"{tif_path}: only {n_bands} band(s), can't extract RGB")

        rgb = np.stack([
            percentile_normalize_to_uint8(r),
            percentile_normalize_to_uint8(g),
            percentile_normalize_to_uint8(b),
        ], axis=-1)

    Image.fromarray(rgb, mode="RGB").save(png_path)
    return True


def convert_mask_tile(tif_path: str, png_path: str):
    with rasterio.open(tif_path) as src:
        mask = src.read(1)
    Image.fromarray(mask, mode="L").save(png_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--band-order", default="nir_r_g_b", choices=["nir_r_g_b", "nir_r_g"])
    args = parser.parse_args()

    img_in = os.path.join(args.input_dir, "image")
    mask_in = os.path.join(args.input_dir, "mask")
    img_out = os.path.join(args.output_dir, "images")
    mask_out = os.path.join(args.output_dir, "masks")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    tif_files = sorted(f for f in os.listdir(img_in) if f.lower().endswith((".tif", ".tiff")))
    print(f"Converting {len(tif_files)} tiles (band_order={args.band_order})...")

    n_ok, n_skipped, n_incomplete = 0, 0, 0
    for fname in tif_files:
        base = os.path.splitext(fname)[0]
        img_tif = os.path.join(img_in, fname)
        mask_tif = os.path.join(mask_in, fname)
        if not os.path.exists(mask_tif):
            n_skipped += 1
            continue
        try:
            saved = convert_tile(img_tif, os.path.join(img_out, f"{base}.png"), args.band_order)
            if not saved:
                n_incomplete += 1
                continue
            convert_mask_tile(mask_tif, os.path.join(mask_out, f"{base}.png"))
            n_ok += 1
        except Exception as e:
            print(f"  SKIP {fname}: {e}")
            n_skipped += 1

    print(f"Done. {n_ok} converted, {n_incomplete} dropped (incomplete/edge tiles), "
         f"{n_skipped} skipped (errors). Output: {args.output_dir}")


if __name__ == "__main__":
    main()
