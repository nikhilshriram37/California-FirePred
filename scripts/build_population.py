"""Add a per-cell human-exposure feature from WorldPop (US 1km population count).

Most California ignitions are human-caused, but the model has no human features.
This aggregates WorldPop's 1km population-count raster to each 0.1-degree cell and
stores log1p(total population) — a proxy for human presence / ignition pressure.

Merges `log_pop` into data/reference/static_features.json. One-time (static).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.windows import from_bounds

from src.data_acquisition.config import RAW_DIR, REFERENCE_DIR

TIF_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/USA/usa_ppp_2020_1km_Aggregated.tif"
HALF = 0.05  # half of a 0.1deg cell


def _download(path):
    if path.exists() and path.stat().st_size > 1_000_000:
        print(f"using cached {path.name}")
        return
    print(f"downloading {path.name} (~53MB)...")
    r = requests.get(TIF_URL, stream=True, timeout=600)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192 * 16):
            f.write(chunk)
    print(f"  saved {path.stat().st_size/1e6:.0f}MB")


def main() -> None:
    tif = RAW_DIR / "worldpop_usa_2020_1km.tif"
    _download(tif)
    grid = pd.read_json(REFERENCE_DIR / "grid_cells.json")

    rows = []
    with rasterio.open(tif) as src:
        nodata = src.nodata
        for r in grid.itertuples():
            win = from_bounds(r.lon_center - HALF, r.lat_center - HALF,
                              r.lon_center + HALF, r.lat_center + HALF, src.transform)
            arr = src.read(1, window=win, boundless=True, fill_value=0).astype("float64")
            if nodata is not None:
                arr[arr == nodata] = 0
            arr[arr < 0] = 0  # WorldPop uses negative sentinels for no-data
            rows.append({"grid_id": int(r.grid_id), "log_pop": float(np.log1p(arr.sum()))})
    pop = pd.DataFrame(rows)

    path = REFERENCE_DIR / "static_features.json"
    if path.exists():
        existing = pd.read_json(path)
        existing = existing.drop(columns=["log_pop"], errors="ignore")
        pop = existing.merge(pop, on="grid_id", how="outer")
    pop = pop.round(3)
    path.write_text(pop.to_json(orient="records"))
    print(f"wrote {path} | cols: {list(pop.columns)}")
    print(pop["log_pop"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
