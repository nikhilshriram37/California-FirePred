"""Pre-download full-year gridMET NetCDF files for emulator training (Route A).

Downloads {var}_{year}.nc into data/raw/gridmet/ (the cache fetch_gridmet_for_grid
reads), so the index emulator can train on multiple full years incl. peak fire
season. Idempotent: skips files already present.
"""
import sys
from pathlib import Path
import requests

from src.data_acquisition.config import RAW_DIR

VARS = ["tmmx", "rmin", "vs", "pr", "erc", "vpd", "fm100", "bi"]
YEARS = [2023, 2024, 2025]
BASE = "https://www.northwestknowledge.net/metdata/data"


def main():
    out = RAW_DIR / "gridmet"
    out.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        for var in VARS:
            p = out / f"{var}_{year}.nc"
            if p.exists() and p.stat().st_size > 1_000_000:
                print(f"skip {p.name} ({p.stat().st_size/1e6:.0f}MB)", flush=True)
                continue
            url = f"{BASE}/{var}_{year}.nc"
            print(f"downloading {var}_{year}.nc ...", flush=True)
            try:
                r = requests.get(url, stream=True, timeout=600)
                r.raise_for_status()
                with open(p, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192 * 16):
                        f.write(chunk)
                print(f"  saved {p.stat().st_size/1e6:.0f}MB", flush=True)
            except Exception as e:
                print(f"  FAILED {var}_{year}: {e}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
