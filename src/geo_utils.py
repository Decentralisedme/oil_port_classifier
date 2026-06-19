"""
geo_utils.py
------------
Lightweight, dependency-safe geospatial helpers.

Deliberately avoids GDAL/geopandas (heavy system deps, slow to install,
occasionally fragile). Everything here uses numpy + scipy only, which
covers >95% of what we need for global point matching and proximity
queries:

  - Great-circle (haversine) distance
  - Nearest-neighbour lookups between two point sets, anywhere on Earth,
    via a KD-tree over Earth-Centered-Earth-Fixed (ECEF) unit-sphere
    coordinates (handles the antimeridian / poles correctly, unlike
    naive lat/lon KD-trees).
  - "Is point within radius of any reference point" geofence checks.

All distances returned in kilometres unless otherwise noted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Vectorised great-circle distance between two sets of points.
    Inputs may be scalars or numpy arrays (broadcastable).
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c


def latlon_to_ecef(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert lat/lon (degrees) to unit-sphere ECEF xyz coordinates."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    return np.column_stack([x, y, z])


def chord_to_km(chord: np.ndarray) -> np.ndarray:
    """
    Convert a Euclidean chord distance on the unit sphere into a
    great-circle distance in km.
    """
    chord = np.clip(chord, 0, 2.0)
    central_angle = 2 * np.arcsin(chord / 2.0)
    return EARTH_RADIUS_KM * central_angle


def build_kdtree(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude") -> cKDTree:
    """Build a KD-tree over a dataframe's lat/lon columns (ECEF unit sphere)."""
    xyz = latlon_to_ecef(df[lat_col].to_numpy(), df[lon_col].to_numpy())
    return cKDTree(xyz)


def nearest_neighbours(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    k: int = 1,
) -> pd.DataFrame:
    """
    For each row in source_df, find the nearest row(s) in target_df.

    Returns source_df with extra columns:
      - nearest_idx_<i>   : positional index into target_df (0..k-1)
      - nearest_dist_km_<i>
    """
    tree = build_kdtree(target_df, lat_col, lon_col)
    src_xyz = latlon_to_ecef(source_df[lat_col].to_numpy(), source_df[lon_col].to_numpy())
    dist_chord, idx = tree.query(src_xyz, k=k)

    out = source_df.copy()
    if k == 1:
        out["nearest_idx_0"] = idx
        out["nearest_dist_km_0"] = chord_to_km(dist_chord)
    else:
        for i in range(k):
            out[f"nearest_idx_{i}"] = idx[:, i]
            out[f"nearest_dist_km_{i}"] = chord_to_km(dist_chord[:, i])
    return out


def within_radius(lat, lon, ref_lat, ref_lon, radius_km: float) -> np.ndarray:
    """Boolean mask: is (lat, lon) within radius_km of (ref_lat, ref_lon)?"""
    return haversine_km(lat, lon, ref_lat, ref_lon) <= radius_km
