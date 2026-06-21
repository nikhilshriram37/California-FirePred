"""Fetch GOES-R GLM lightning flashes and aggregate to grid cells (live feed).

GOES-GLM (Geostationary Lightning Mapper) publishes 20-second granules of detected
flashes to NOAA's public S3 bucket — no credentials required. We scan a recent
window, keep flashes inside the California bbox, and count flashes per grid cell.

This is the *live* lightning equivalent. Note: the model was trained on NOAA Storm
Events counts, so absolute magnitudes differ from GLM flash counts (a documented
train/serve skew on this single, low-importance feature).
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import logging
import os
import tempfile
import threading

import boto3
import numpy as np
import pandas as pd
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
from scipy.spatial import cKDTree

from src.data_acquisition.config import PROCESSED_DIR, REGIONS

logger = logging.getLogger(__name__)

_S3 = boto3.client(
    "s3", config=Config(signature_version=UNSIGNED, max_pool_connections=32)
)
GLM_BUCKET = "noaa-goes18"  # GOES-West — best view of California
GLM_PRODUCT = "GLM-L2-LCFA"

# HDF5 (under netCDF4) is not thread-safe — concurrent opens segfault. Downloads
# run in parallel, but the actual file reads are serialized through this lock.
_HDF5_LOCK = threading.Lock()


def _hour_prefixes(end: dt.datetime, hours: int):
    for h in range(hours):
        t = end - dt.timedelta(hours=h)
        yield f"{GLM_PRODUCT}/{t.year}/{t.timetuple().tm_yday:03d}/{t.hour:02d}/"


def _list_keys(prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        kw = dict(Bucket=GLM_BUCKET, Prefix=prefix)
        if token:
            kw["ContinuationToken"] = token
        r = _S3.list_objects_v2(**kw)
        keys += [o["Key"] for o in r.get("Contents", [])]
        token = r.get("NextContinuationToken")
        if not token:
            break
    return keys


def _flashes_in_bbox(key: str, bbox: dict) -> tuple[np.ndarray, np.ndarray]:
    """Download one granule and return (lat, lon) of flashes inside the bbox."""
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".nc")
        os.close(fd)
        _S3.download_file(GLM_BUCKET, key, tmp)  # parallel I/O
        with _HDF5_LOCK:                          # serialized HDF5 read
            ds = xr.open_dataset(tmp)
            la, lo = ds["flash_lat"].values, ds["flash_lon"].values
            ds.close()
        m = (la >= bbox["south"]) & (la < bbox["north"]) & (lo >= bbox["west"]) & (lo < bbox["east"])
        return la[m], lo[m]
    except Exception as e:  # a corrupt/missing granule shouldn't sink the run
        logger.debug("skip %s: %s", key, e)
        return np.array([]), np.array([])
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def fetch_glm_lightning(
    grid: pd.DataFrame,
    hours: int = 24,
    end: dt.datetime | None = None,
    sample_every: int = 1,
    max_workers: int = 16,
) -> pd.DataFrame:
    """Count GLM flashes per grid cell over the last ``hours``.

    Args:
        grid: DataFrame with grid_id, lat_center, lon_center (the model's cells).
        hours: Look-back window (24 = one day of lightning).
        end: Window end (UTC); defaults to now.
        sample_every: Process every Nth granule and scale counts up — trade
            accuracy for speed (1 = full scan).
        max_workers: Parallel granule downloads.

    Returns:
        DataFrame with columns grid_id, lightning_count (cells with >0 only).
    """
    bbox = REGIONS["california"]
    end = end or dt.datetime.now(dt.UTC)

    keys: list[str] = []
    for p in _hour_prefixes(end, hours):
        keys += _list_keys(p)
    if sample_every > 1:
        keys = keys[::sample_every]
    logger.info("GLM: scanning %d granules (%dh window, every %d)", len(keys), hours, sample_every)

    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for la, lo in ex.map(lambda k: _flashes_in_bbox(k, bbox), keys):
            if len(la):
                lats.append(la)
                lons.append(lo)

    if not lats:
        logger.info("GLM: no flashes found in window")
        return pd.DataFrame(columns=["grid_id", "lightning_count"])

    flat_la = np.concatenate(lats)
    flat_lo = np.concatenate(lons)

    # Assign each flash to its nearest grid-cell center (matches the training
    # pipeline's nearest-cell mapping).
    tree = cKDTree(np.c_[grid["lon_center"].to_numpy(), grid["lat_center"].to_numpy()])
    _, idx = tree.query(np.c_[flat_lo, flat_la])
    gid = grid["grid_id"].to_numpy()[idx]

    counts = pd.Series(gid).value_counts()
    out = pd.DataFrame({
        "grid_id": counts.index.astype(int),
        "lightning_count": (counts.to_numpy() * sample_every).astype(int),
    })
    logger.info("GLM: %d CA flashes -> %d cells with lightning", len(flat_la), len(out))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cells = (
        pd.read_parquet(
            PROCESSED_DIR / "california_dataset.parquet",
            columns=["grid_id", "lat_center", "lon_center"],
        )
        .drop_duplicates("grid_id")
        .reset_index(drop=True)
    )
    df = fetch_glm_lightning(cells, hours=3, sample_every=2)
    print(df.sort_values("lightning_count", ascending=False).head())
    print("total flashes mapped:", int(df["lightning_count"].sum()))
