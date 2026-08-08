"""
model_io.py
===========
This is the ONE file that changes when the real trained PathMamba checkpoint is ready.
Every other module in this package consumes only the `RoadPrediction` object defined
here — never a raw tensor — so swapping synthetic data for real inference output is a
one-function change, not a refactor.

Confirmed PathMamba contract (from the Stage A/B pretraining notebook, Section 5):

    road_logits, confidence_map = model(x)      # x: (B, 2, H, W) float tensor
                                                 # channel 0 = grayscale PAN-proxy
                                                 # channel 1 = GLCM occlusion-texture
    # road_logits:      (B, 1, H, W) raw logits -- apply sigmoid yourself
    # confidence_map:   (B, 1, H, W) already sigmoided inside the model (Sigmoid in conf_head)

`confidence_map` is what the healing module (healing.py) samples for the "occlusion
confidence >= 0.3" gate -- it is the model's own belief that a low/no-road-probability
pixel is nonetheless a plausible occluded road, not a generic pixel-quality score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


class WorldToPixel(Protocol):
    """Anything with this method can georeference a RoadPrediction raster.

    Both SimpleAffine (synthetic testing, geo_utils.py) and a thin rasterio.Affine
    adapter (real georeferenced tiles) satisfy this protocol -- healing.py and
    friends don't care which one they get.
    """

    def world_to_pixel(self, x: float, y: float) -> tuple[float, float]: ...


@dataclass
class RoadPrediction:
    """Canonical container for one tile's worth of PathMamba output.

    prob_mask       : (H, W) float32 in [0, 1]  -- sigmoid(road_logits), squeezed to 2D
    confidence_mask : (H, W) float32 in [0, 1]  -- confidence_map, squeezed to 2D
    transform       : object satisfying WorldToPixel, or None if rasters are already
                      in a local/synthetic coordinate frame where world == pixel
    crs             : CRS string (e.g. "EPSG:32644"), None for synthetic data
    source_gsd_m    : ground sample distance in meters/pixel. Pass 0.28 for real
                      Cartosat-3 inference (per the notebook's Section 9 note).
    """

    prob_mask: np.ndarray
    confidence_mask: np.ndarray
    transform: Optional[WorldToPixel] = None
    crs: Optional[str] = None
    source_gsd_m: float = 0.28
    is_synthetic: bool = False

    def __post_init__(self) -> None:
        if self.prob_mask.shape != self.confidence_mask.shape:
            raise ValueError(
                f"prob_mask shape {self.prob_mask.shape} != "
                f"confidence_mask shape {self.confidence_mask.shape}"
            )
        if self.prob_mask.ndim != 2:
            raise ValueError(
                f"expected a squeezed (H, W) array, got ndim={self.prob_mask.ndim}. "
                "Squeeze batch/channel dims before constructing RoadPrediction."
            )
        self.prob_mask = self.prob_mask.astype(np.float32, copy=False)
        self.confidence_mask = self.confidence_mask.astype(np.float32, copy=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.prob_mask.shape  # type: ignore[return-value]


def _to_numpy(x) -> np.ndarray:
    """Accepts a torch tensor or numpy array; returns numpy without hard torch import."""
    if isinstance(x, np.ndarray):
        return x
    # Duck-type torch.Tensor without importing torch at module load time --
    # keeps this module importable in environments without torch/mamba_ssm installed.
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    raise TypeError(f"Expected numpy array or torch tensor, got {type(x)}")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _squeeze_to_hw(arr: np.ndarray) -> np.ndarray:
    """(B, 1, H, W) / (1, H, W) / (H, W) -> (H, W). Refuses batch size > 1 (call per-tile)."""
    arr = np.asarray(arr)
    while arr.ndim > 2:
        if arr.shape[0] != 1:
            raise ValueError(
                f"from_pathmamba_output expects a single tile at a time (batch size 1), "
                f"got leading dim {arr.shape[0]} on array with shape {arr.shape}. "
                "Loop over the batch and call this once per tile."
            )
        arr = arr[0]
    return arr


def from_pathmamba_output(
    road_logits,
    confidence_map,
    transform: Optional[WorldToPixel] = None,
    crs: Optional[str] = None,
    source_gsd_m: float = 0.28,
) -> RoadPrediction:
    """
    *** THE REAL HOOK POINT ***

    Call this exactly like:

        road_logits, confidence_map = model(x)   # x: (1, 2, H, W) tensor, one tile
        pred = from_pathmamba_output(
            road_logits, confidence_map,
            transform=tile_transform,             # e.g. a rasterio Affine wrapped in
                                                   # geo_utils.RasterioAffineAdapter
            crs="EPSG:32644",
            source_gsd_m=0.28,
        )

    Everything downstream (healing.py, centrality.py, simulation.py, pipeline.py)
    takes a RoadPrediction and does not know or care that this function exists.
    Accepts torch tensors or numpy arrays; applies sigmoid to road_logits only
    (confidence_map is already sigmoided inside the model, per Section 5 of the
    pretraining notebook -- do not double-sigmoid it).
    """
    prob = _sigmoid(_to_numpy(road_logits))
    conf = _to_numpy(confidence_map)
    prob = _squeeze_to_hw(prob)
    conf = _squeeze_to_hw(conf)
    return RoadPrediction(
        prob_mask=prob,
        confidence_mask=conf,
        transform=transform,
        crs=crs,
        source_gsd_m=source_gsd_m,
        is_synthetic=False,
    )


def load_pathmamba_and_infer(
    checkpoint_path: str,
    image_array: np.ndarray,
    pathmamba_cls,
    device: str = "cuda",
    transform: Optional[WorldToPixel] = None,
    crs: Optional[str] = None,
    source_gsd_m: float = 0.28,
) -> RoadPrediction:
    """
    Convenience end-to-end loader for once the checkpoint exists.

    `pathmamba_cls` is passed in rather than imported here on purpose: this module
    has zero hard dependency on torch/mamba_ssm, so the rest of the pipeline stays
    importable and testable today, in an environment that has never seen the
    training notebook's libraries. Call it like:

        from notebook_module import PathMamba   # wherever the trained class lives
        pred = load_pathmamba_and_infer("stage_b_checkpoint.pth", tile_array, PathMamba)

    `image_array` must already be the 2-channel (grayscale PAN-proxy + GLCM texture)
    input the model expects, shape (H, W, 2) or (2, H, W).
    """
    import torch  # deferred import -- only required on the machine that actually runs this

    model = pathmamba_cls(in_channels=2, pretrained=False, use_mamba=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()

    arr = np.asarray(image_array)
    if arr.shape[-1] == 2:  # (H, W, 2) -> (2, H, W)
        arr = np.moveaxis(arr, -1, 0)
    x = torch.from_numpy(arr).float().unsqueeze(0).to(device)  # (1, 2, H, W)

    with torch.no_grad():
        road_logits, confidence_map = model(x)

    return from_pathmamba_output(
        road_logits, confidence_map, transform=transform, crs=crs, source_gsd_m=source_gsd_m
    )
