"""
osm_labels.py
=============
OSM -> raster label generator, matching the deck's Stage 2 wording:

  * dynamic UTM CRS estimation per AOI (no hardcoded projection)
  * Douglas-Peucker geometry simplification
  * carriageway-aware buffer widths: single carriageway for oneway roads
    (halve the buffer), full carriageway for two-way roads

WHY THIS EXISTS
---------------
Training PathMamba on OSM-derived labels requires converting OSM way geometry
(lon/lat line-strings tagged with `highway`, `oneway`, `lanes`, etc.) into
binary road masks aligned to satellite imagery. The three deck alignments
above have big effects on label quality:

  * Dynamic UTM: buffering by "5 meters" in EPSG:4326 (degrees) is not 5m,
    and varies with latitude. Reprojecting to the local UTM zone gives us
    meter-accurate buffers everywhere.
  * Douglas-Peucker: OSM geometries have hundreds of near-collinear points
    per way; DP simplification cuts that 5-10x with essentially zero geometry
    loss and speeds rasterization proportionally.
  * Carriageway-aware widths: a divided highway ("dual carriageway") is
    mapped in OSM as TWO parallel oneway ways. Buffering both by the full
    road width doubles the label -- the correct answer is to halve the buffer
    for oneway ways so the two halves together reconstruct the full width.

DEPENDENCIES
------------
Uses osmnx (OSM ingest), geopandas + shapely (geometry), rasterio (mask
rasterization). All are deferred imports so this module can be inspected /
partially exercised without the geo stack installed. The buffer-width and
DP-simplification helpers are pure-Python and work without any of the above.

USAGE
-----
    from route_resilience.osm_labels import make_osm_label_raster
    mask, transform, crs = make_osm_label_raster(
        aoi_bbox=(min_lon, min_lat, max_lon, max_lat),
        pixel_size_m=0.3,          # match Cartosat-3 GSD
        simplify_tolerance_m=1.0,  # Douglas-Peucker
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math


# ---------------------------------------------------------------------------
# Carriageway-aware buffer widths (pure Python -- no deps)
# ---------------------------------------------------------------------------

# Default full-carriageway widths in meters by OSM highway tag. Values are
# for a two-way road -- oneway ways get HALVED (see buffer_width_for_way).
# Numbers are typical urban / suburban averages -- an academic-reference set
# is fine; on-the-ground widths vary considerably.
DEFAULT_HIGHWAY_WIDTHS_M: dict[str, float] = {
    "motorway": 12.0,
    "motorway_link": 6.0,
    "trunk": 10.0,
    "trunk_link": 5.0,
    "primary": 8.0,
    "primary_link": 5.0,
    "secondary": 7.0,
    "secondary_link": 4.5,
    "tertiary": 6.0,
    "tertiary_link": 4.0,
    "unclassified": 5.0,
    "residential": 5.0,
    "living_street": 4.0,
    "service": 3.5,
    "track": 3.0,
    "path": 1.5,
    "pedestrian": 3.0,
    "footway": 1.5,
}


def buffer_width_for_way(
    highway: str,
    oneway: bool = False,
    lanes: Optional[int] = None,
    width_table: Optional[dict] = None,
    lane_width_m: float = 3.5,
) -> float:
    """Return the *half-width* to buffer a way's centerline by (meters).

    The buffer is symmetric around the centerline, so this is HALF the total
    road width. A "half" wording avoids the common bug of buffering by the
    full carriageway (which doubles the label).

    Rules, in order of precedence:
      1. If `lanes` is set, use lanes * lane_width_m as the total width
         (regardless of highway class -- lane count is more specific).
      2. Otherwise use the highway-class default from width_table.
      3. If `oneway=True`, HALVE the result -- the deck alignment. A oneway
         way represents one carriageway of a divided road, not the whole road.

    Fallback: unrecognized highway tags default to 4.0m total width.
    """
    table = width_table if width_table is not None else DEFAULT_HIGHWAY_WIDTHS_M
    if lanes is not None and lanes > 0:
        total = lanes * lane_width_m
    else:
        total = table.get(highway, 4.0)
    if oneway:
        total *= 0.5
    return total / 2.0  # half-width for symmetric buffer


# ---------------------------------------------------------------------------
# Douglas-Peucker simplification (pure Python)
# ---------------------------------------------------------------------------

def _perp_distance(pt, a, b) -> float:
    """Perpendicular distance from pt to segment a-b."""
    (px, py), (ax, ay), (bx, by) = pt, a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    # projected fraction t on segment
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    fx, fy = ax + t * dx, ay + t * dy
    return math.hypot(px - fx, py - fy)


def douglas_peucker(points: list[tuple[float, float]], tolerance: float) -> list[tuple[float, float]]:
    """Douglas-Peucker line simplification.

    Recursively simplifies a polyline by keeping only the points whose
    perpendicular distance to the running best-fit segment exceeds `tolerance`.
    Returns a subset of the input with the same first and last point.

    tolerance is in the same units as `points` -- pass a meter value if points
    are in meters (i.e. after UTM projection). Pass 0.0 to disable.
    """
    if tolerance <= 0 or len(points) < 3:
        return list(points)

    def _simplify(pts):
        if len(pts) < 3:
            return list(pts)
        a, b = pts[0], pts[-1]
        max_d = -1.0
        max_i = -1
        for i in range(1, len(pts) - 1):
            d = _perp_distance(pts[i], a, b)
            if d > max_d:
                max_d, max_i = d, i
        if max_d < tolerance:
            return [a, b]
        left = _simplify(pts[:max_i + 1])
        right = _simplify(pts[max_i:])
        return left[:-1] + right

    return _simplify(list(points))


# ---------------------------------------------------------------------------
# Dynamic UTM CRS estimation
# ---------------------------------------------------------------------------

def estimate_utm_crs(lon: float, lat: float) -> str:
    """Estimate the UTM EPSG code (as an "EPSG:xxxxx" string) for a WGS84
    lon/lat.

    UTM zones are 6 degrees wide, indexed 1-60 starting at 180W. Northern
    zones: EPSG 32601 + (zone-1). Southern zones: EPSG 32701 + (zone-1).

    For an AOI use its CENTROID lon/lat; UTM is only meaningful for AOIs
    that fit within one zone (~ 600 km EW). Larger AOIs need a projected
    equal-area CRS instead -- out of scope here.
    """
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"lon {lon} out of range")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"lat {lat} out of range")
    zone = int((lon + 180.0) // 6.0) + 1
    zone = max(1, min(60, zone))
    if lat >= 0:
        return f"EPSG:{32600 + zone}"
    else:
        return f"EPSG:{32700 + zone}"


# ---------------------------------------------------------------------------
# High-level generator (deferred imports on the geo stack)
# ---------------------------------------------------------------------------

@dataclass
class OSMLabelConfig:
    """Config for OSM label rasterization."""
    pixel_size_m: float = 0.3                          # match Cartosat-3 GSD
    simplify_tolerance_m: float = 1.0                  # Douglas-Peucker tolerance in meters
    highway_filter: tuple = (                          # only rasterize these highway classes
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "service",
    )
    width_table: dict = field(default_factory=lambda: dict(DEFAULT_HIGHWAY_WIDTHS_M))
    lane_width_m: float = 3.5


def _dp_simplify_linestring(coords_utm, tol_m):
    """Wrapper so callers with shapely LineStrings can simplify via our
    dependency-free DP. Falls back to shapely's built-in .simplify() if
    shapely is available -- shapely's is faster and identical in output."""
    try:
        from shapely.geometry import LineString
        return list(LineString(coords_utm).simplify(tol_m, preserve_topology=False).coords)
    except Exception:
        return douglas_peucker(list(coords_utm), tol_m)


def make_osm_label_raster(
    aoi_bbox: tuple[float, float, float, float],  # (min_lon, min_lat, max_lon, max_lat)
    config: Optional[OSMLabelConfig] = None,
    osm_gdf=None,  # optional: pre-fetched geopandas GeoDataFrame with 'geometry','highway','oneway','lanes'
):
    """Fetch OSM roads for an AOI, project to UTM, DP-simplify, buffer with
    carriageway-aware widths, rasterize to a binary mask.

    Returns
    -------
    (mask, transform, crs) where mask is a numpy uint8 array (1=road, 0=bg),
    transform is a rasterio Affine mapping pixel -> UTM meters, and crs is
    an "EPSG:xxxxx" string.

    Notes
    -----
    Deferred imports: osmnx, geopandas, shapely, rasterio. If any are missing,
    the caller can bypass OSM fetch by passing `osm_gdf` -- a GeoDataFrame in
    EPSG:4326 with columns geometry, highway, oneway (bool), lanes (int|None).
    Then only shapely + rasterio + numpy are needed.
    """
    import numpy as np

    cfg = config or OSMLabelConfig()

    # 1. get OSM ways
    if osm_gdf is None:
        import osmnx as ox
        import geopandas as gpd
        graph = ox.graph_from_bbox(
            bbox=(aoi_bbox[3], aoi_bbox[1], aoi_bbox[2], aoi_bbox[0]),  # (N, S, E, W)
            network_type="drive", simplify=False,
        )
        _, edges = ox.graph_to_gdfs(graph)
        gdf = edges[["geometry", "highway", "oneway", "lanes"]].copy()
        gdf["highway"] = gdf["highway"].apply(lambda h: h[0] if isinstance(h, list) else h)
    else:
        gdf = osm_gdf.copy()

    # 2. filter by highway class
    gdf = gdf[gdf["highway"].isin(cfg.highway_filter)]
    if len(gdf) == 0:
        raise ValueError("no OSM ways after highway filter")

    # 3. estimate UTM CRS from AOI centroid, reproject
    cx = 0.5 * (aoi_bbox[0] + aoi_bbox[2])
    cy = 0.5 * (aoi_bbox[1] + aoi_bbox[3])
    crs = estimate_utm_crs(cx, cy)
    import geopandas as gpd
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    gdf = gdf.to_crs(crs)

    # 4. simplify + buffer each way with carriageway-aware width
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    buffered_geoms = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            oneway = bool(row.get("oneway", False))
        except Exception:
            oneway = False
        try:
            lanes = row.get("lanes")
            lanes = int(lanes) if lanes is not None else None
        except Exception:
            lanes = None
        half_w = buffer_width_for_way(
            highway=row["highway"], oneway=oneway, lanes=lanes,
            width_table=cfg.width_table, lane_width_m=cfg.lane_width_m,
        )
        # simplify then buffer
        coords = _dp_simplify_linestring(list(geom.coords), cfg.simplify_tolerance_m)
        if len(coords) < 2:
            continue
        buffered_geoms.append(LineString(coords).buffer(half_w, cap_style=2, join_style=2))

    if not buffered_geoms:
        raise ValueError("no geometries after simplify+buffer")

    combined = unary_union(buffered_geoms)

    # 5. rasterize into a mask aligned to the UTM bbox
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    minx, miny, maxx, maxy = combined.bounds
    W = int(math.ceil((maxx - minx) / cfg.pixel_size_m))
    H = int(math.ceil((maxy - miny) / cfg.pixel_size_m))
    transform = from_origin(minx, maxy, cfg.pixel_size_m, cfg.pixel_size_m)
    mask = rasterize(
        [(combined, 1)], out_shape=(H, W), transform=transform,
        fill=0, dtype="uint8", all_touched=True,
    )
    return mask, transform, crs
