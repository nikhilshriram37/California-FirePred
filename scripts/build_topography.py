"""Compute per-cell topography features from Open-Meteo's elevation API (keyless).

Fetches elevation at each cell CENTER (few requests -> robust), then derives terrain
features from each cell's 3x3 neighbourhood on the 0.1-degree grid:
    elev_mean   cell-center elevation (m)
    ruggedness  std of the 3x3 neighbourhood elevations (terrain roughness)
    slope_deg   regional slope (degrees)
    northness   cos(aspect): +1 = north-facing (cooler/wetter), -1 = south-facing
    eastness    sin(aspect)

Merges these into data/reference/static_features.json. One-time (static).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from src.data_acquisition.config import REFERENCE_DIR

ELEV_URL = "https://api.open-meteo.com/v1/elevation"
STEP = 0.1     # grid spacing (cell centers are 0.1deg apart)
BATCH = 100    # Open-Meteo elevation: max 100 coords per request


def _fetch_elevations(lats: list[float], lons: list[float]) -> list[float]:
    out: list[float] = []
    for s in range(0, len(lats), BATCH):
        params = {"latitude": ",".join(f"{v:.5f}" for v in lats[s:s + BATCH]),
                  "longitude": ",".join(f"{v:.5f}" for v in lons[s:s + BATCH])}
        for attempt in range(8):
            try:
                r = requests.get(ELEV_URL, params=params, timeout=(15, 60))
                if r.status_code == 429:
                    time.sleep(10 * (attempt + 1)); continue
                r.raise_for_status()
                out.extend(r.json()["elevation"])
                break
            except requests.exceptions.RequestException as e:
                print(f"  retry {attempt+1}: {str(e)[:60]}")
                time.sleep(6 * (attempt + 1))
        else:
            raise RuntimeError("Open-Meteo elevation failed after retries")
        time.sleep(0.6)  # gentle: only ~37 requests total
    return out


def main() -> None:
    grid = pd.read_json(REFERENCE_DIR / "grid_cells.json")
    print(f"cells: {len(grid)} -> {len(grid)//BATCH + 1} requests")
    elev = _fetch_elevations(grid["lat_center"].tolist(), grid["lon_center"].tolist())
    grid["elev"] = elev
    lut = {(round(r.lat_center, 2), round(r.lon_center, 2)): r.elev for r in grid.itertuples()}

    rows = []
    for r in grid.itertuples():
        lat, lon, z0 = r.lat_center, r.lon_center, r.elev
        # 3x3 neighbourhood (row 0 = north, col 0 = west), missing -> center value
        z = np.array([[lut.get((round(lat + dy, 2), round(lon + dx, 2)), z0)
                       for dx in (-STEP, 0.0, STEP)]
                      for dy in (STEP, 0.0, -STEP)], dtype=float)
        dy_m = STEP * 111_000
        dx_m = STEP * 111_000 * np.cos(np.radians(lat))
        dz_dx = (z[:, 2].mean() - z[:, 0].mean()) / (2 * dx_m)
        dz_dy = (z[0, :].mean() - z[2, :].mean()) / (2 * dy_m)
        mag = float(np.hypot(dz_dx, dz_dy))
        rows.append({
            "grid_id": int(r.grid_id),
            "elev_mean": float(z0),
            "ruggedness": float(z.std()),
            "slope_deg": float(np.degrees(np.arctan(mag))),
            "northness": float(-dz_dy / mag) if mag > 1e-9 else 0.0,
            "eastness": float(-dz_dx / mag) if mag > 1e-9 else 0.0,
        })
    topo = pd.DataFrame(rows)

    path = REFERENCE_DIR / "static_features.json"
    if path.exists():
        existing = pd.read_json(path)
        existing = existing.drop(columns=[c for c in topo.columns if c != "grid_id"],
                                 errors="ignore")
        topo = existing.merge(topo, on="grid_id", how="outer")
    topo = topo.round(3)
    path.write_text(topo.to_json(orient="records"))
    print(f"wrote {path} | cols: {list(topo.columns)}")
    print(topo[["elev_mean", "ruggedness", "slope_deg", "northness"]].describe().round(1).to_string())


if __name__ == "__main__":
    main()
