"""California boundary helpers — keep the grid to actual state territory.

The model grid is a rectangle over California's bounding box, so ~37% of cells
land in Nevada / Arizona / offshore. These helpers filter cells to the real
(irregular) state polygon via a point-in-polygon test on each cell center, using
the cartographic boundary cached at data/external/california_boundary.geojson.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.prepared import prep

from src.data_acquisition.config import EXTERNAL_DIR

BOUNDARY_PATH = EXTERNAL_DIR / "california_boundary.geojson"


@lru_cache(maxsize=1)
def load_california_polygon(path: Path = BOUNDARY_PATH) -> BaseGeometry:
    """Load the California boundary polygon (Feature or bare geometry GeoJSON)."""
    if not path.exists():
        raise FileNotFoundError(
            f"California boundary not found at {path}. Download a US-states "
            "cartographic GeoJSON and extract the California feature."
        )
    data = json.loads(path.read_text())
    geom = data["geometry"] if data.get("type") == "Feature" else data
    return shape(geom)


def filter_to_california(
    df: pd.DataFrame, lon_col: str = "lon_center", lat_col: str = "lat_center"
) -> pd.DataFrame:
    """Return only the rows whose cell center lies inside California."""
    poly = prep(load_california_polygon())
    mask = [poly.contains(Point(lon, lat)) for lon, lat in zip(df[lon_col], df[lat_col])]
    return df[mask].copy()
