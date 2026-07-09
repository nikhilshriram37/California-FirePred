"""Regenerate the canonical scoring grid: FULL California-land 0.1-degree coverage.

The training dataset uses only *fire cells + a ~3x sample of non-fire cells* (a
class-balance trick for training). The live pipeline, however, should score
EVERY California land cell — otherwise large stretches of the state have no cell
at all and read as misleading "green" holes on the risk map.

This builds the complete 0.1-degree lattice over California's bounding box,
clips it to the real (irregular) state polygon via `filter_to_california`, and
writes `data/reference/grid_cells.json` (grid_id, lat_center, lon_center).

`build_grid` numbers cells row-major over the FULL bbox lattice and is
deterministic, so the grid_ids here are a *superset* of the previous grid: every
existing cell keeps its exact id (existing Supabase rows stay valid) and the
gap-filling cells simply take their previously-unused ids.

After running this, re-run `scripts/build_topography.py`,
`scripts/build_population.py`, and `scripts/extend_dryness_climatology.py` so the
per-cell static + seasonal reference features cover the new cells too.

One-time / on-demand (the grid is static).
"""

from __future__ import annotations

import pandas as pd

from src.data_acquisition.config import REGIONS, REFERENCE_DIR
from src.pipeline.geo import filter_to_california
from src.preprocessing.build_dataset import build_grid

COLS = ["grid_id", "lat_center", "lon_center"]


def build_california_land_grid(resolution_deg: float = 0.1) -> pd.DataFrame:
    """Full CA-land grid at the given resolution, with stable row-major grid_ids."""
    full = build_grid(REGIONS["california"], resolution_deg)
    land = filter_to_california(full).reset_index(drop=True)
    return land[COLS].astype({"grid_id": int}).round({"lat_center": 4, "lon_center": 4})


def main() -> None:
    path = REFERENCE_DIR / "grid_cells.json"
    prev = pd.read_json(path) if path.exists() else pd.DataFrame(columns=COLS)

    grid = build_california_land_grid()

    prev_ids, new_ids = set(prev.get("grid_id", [])), set(grid["grid_id"])
    added = len(new_ids - prev_ids)
    dropped = len(prev_ids - new_ids)  # should be 0: new grid is a superset
    if dropped:
        print(f"WARNING: {dropped} previously-present cells are NOT in the new grid")

    path.write_text(grid.to_json(orient="records"))
    print(f"wrote {path} | {len(prev)} -> {len(grid)} cells (+{added} new, -{dropped})")
    print(f"  lat {grid.lat_center.min()}..{grid.lat_center.max()}  "
          f"lon {grid.lon_center.min()}..{grid.lon_center.max()}")


if __name__ == "__main__":
    main()
