"""Extend the per-cell seasonal dryness climatology to newly-added grid cells.

`data/reference/dryness_climatology.json` holds per-cell monthly seasonal normals
for aet + water_deficit (TerraClimate has no live feed, so live scoring uses these
as a stand-in). It was built from the historical parquet, which only covers the
old sampled grid. When the scoring grid is expanded to full CA-land coverage (see
`scripts/build_grid_cells.py`), the new gap cells have no climatology.

This fills each new cell from its NEAREST existing cell (great-circle-ish nearest
neighbour on lat/lon), copying all 12 monthly aet/water_deficit values. Dryness is
a slow-moving, spatially-smooth seasonal field, so a nearest-neighbour fill is far
more faithful than the statewide-median fallback that `build_live_features` would
otherwise apply — and it keeps every scored cell on equal footing.

Run AFTER regenerating grid_cells.json. Idempotent (re-derives new cells each run).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_acquisition.config import REFERENCE_DIR


def main() -> None:
    grid = pd.read_json(REFERENCE_DIR / "grid_cells.json")
    clim = pd.read_json(REFERENCE_DIR / "dryness_climatology.json")

    # Donors = cells that have real climatology AND are in the (CA-clipped) grid,
    # so we have both their seasonal normals and their coordinates. Climatology
    # rows for cells no longer in the grid (old out-of-CA cells) are dropped.
    have = set(clim["grid_id"])
    src = grid[grid["grid_id"].isin(have)].reset_index(drop=True)
    keep = clim[clim["grid_id"].isin(set(grid["grid_id"]))]  # existing cells, real values
    new = grid[~grid["grid_id"].isin(have)].reset_index(drop=True)
    if new.empty:
        print(f"dryness_climatology.json already covers all {len(grid)} cells — nothing to do")
        return

    src_xy = src[["lon_center", "lat_center"]].to_numpy()
    new_xy = new[["lon_center", "lat_center"]].to_numpy()

    # Nearest donor cell for each new cell (small grids -> plain broadcast).
    d2 = ((new_xy[:, None, :] - src_xy[None, :, :]) ** 2).sum(axis=2)
    nearest_gid = src["grid_id"].to_numpy()[d2.argmin(axis=1)]

    # Copy the donor's 12 monthly rows to each new cell.
    donor = clim.rename(columns={"grid_id": "src_gid"})
    mapping = pd.DataFrame({"grid_id": new["grid_id"].to_numpy(), "src_gid": nearest_gid})
    filled = mapping.merge(donor, on="src_gid").drop(columns="src_gid")

    out = pd.concat([keep, filled[clim.columns]], ignore_index=True)
    out = out.sort_values(["grid_id", "month"]).reset_index(drop=True)

    assert out.isnull().sum().sum() == 0, "NaN in extended climatology"
    assert out.grid_id.nunique() == len(grid), "climatology cell count != grid cell count"
    assert (out.groupby("grid_id").size() == 12).all(), "every cell needs 12 monthly rows"

    path = REFERENCE_DIR / "dryness_climatology.json"
    path.write_text(out.to_json(orient="records"))
    print(f"wrote {path} | {src.grid_id.nunique()} real -> {out.grid_id.nunique()} cells "
          f"(+{len(new)} nearest-neighbour filled), {len(out)} rows")


if __name__ == "__main__":
    main()
