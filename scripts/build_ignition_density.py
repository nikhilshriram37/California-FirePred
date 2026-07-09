"""Per-cell historical ignition-density prior (leakage-safe).

~90% of California ignitions are human/undetermined, and the risk model leans on
raw lat/lon as a crude "where do fires happen" proxy (see the spatial-CV finding).
This formalizes that into a real feature: log1p(count of FPA-FOD fires per 0.1-deg
cell) over 1992-2017 ONLY.

Using pre-2018 years keeps it strictly leakage-safe — the model's dataset is
2018-2020, so no in-sample (or test-year) fire informs the prior. The 0.1-deg
binning matches training (`assign_fires_to_grid`), so grid_ids line up.

Writes data/reference/ignition_density.json (grid_id, ignition_density). Static.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_acquisition.config import RAW_DIR, REFERENCE_DIR, REGIONS
from src.data_acquisition.fetch_fire_history import load_fpa_fod
from src.preprocessing.build_dataset import assign_fires_to_grid, build_grid

PRIOR_YEARS = (1992, 2017)  # strictly before the 2018-2020 modeling dataset


def main() -> None:
    bbox = REGIONS["california"]
    grid = build_grid(bbox, 0.1)

    fires = load_fpa_fod(RAW_DIR / "fpa_fod" / "FPA_FOD.sqlite")
    fires = fires[
        (fires["STATE"] == "CA")
        & (fires["discovery_date"].dt.year >= PRIOR_YEARS[0])
        & (fires["discovery_date"].dt.year <= PRIOR_YEARS[1])
    ].copy()
    print(f"CA fires {PRIOR_YEARS[0]}-{PRIOR_YEARS[1]}: {len(fires):,}")

    fires = assign_fires_to_grid(fires, grid, 0.1, bbox)
    counts = (fires.dropna(subset=["grid_id"])
              .groupby("grid_id").size().rename("n_fires").reset_index())
    counts["grid_id"] = counts["grid_id"].astype(int)

    # Score every canonical grid cell; cells with no historical fire -> 0.
    cells = pd.read_json(REFERENCE_DIR / "grid_cells.json")[["grid_id"]]
    out = cells.merge(counts, on="grid_id", how="left")
    out["n_fires"] = out["n_fires"].fillna(0)
    out["ignition_density"] = np.log1p(out["n_fires"]).round(4)
    out = out[["grid_id", "ignition_density"]]

    path = REFERENCE_DIR / "ignition_density.json"
    path.write_text(out.to_json(orient="records"))
    nz = (out["ignition_density"] > 0).sum()
    print(f"wrote {path} | {len(out)} cells, {nz} with prior fires "
          f"({nz/len(out)*100:.0f}%), max log1p={out.ignition_density.max():.2f}")


if __name__ == "__main__":
    main()
