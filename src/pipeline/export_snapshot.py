"""Export a historical risk snapshot as GeoJSON for the dashboard (replay mode).

Scores every California grid cell for one past date with the exported model and
writes the dashboard snapshot via the shared :mod:`src.pipeline.snapshot` writer.
For the *live* equivalent, see :mod:`src.pipeline.score_daily`.

Run:  python -m src.pipeline.export_snapshot               # peak 2020 fire day
      python -m src.pipeline.export_snapshot --date 2020-09-09
      python -m src.pipeline.export_snapshot --latest       # most recent date
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data_acquisition.config import PROCESSED_DIR
from src.models.predict import load_model
from src.pipeline.geo import filter_to_california
from src.pipeline.snapshot import SNAPSHOT_DIR, build_meta, day_to_feature_collection, write_snapshot

logger = logging.getLogger(__name__)


def build_snapshot(date: str | None = None, latest: bool = False,
                   dataset_path: Path | None = None) -> tuple[dict, dict]:
    """Score one historical date and return (geojson, meta)."""
    dataset_path = dataset_path or PROCESSED_DIR / "california_dataset.parquet"
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])

    if latest:
        target = df["date"].max()
    elif date:
        target = pd.Timestamp(date)
    else:  # default: the most fire-active day (a striking, representative map)
        target = df[df["date"].dt.year == 2020].groupby("date")["has_fire"].sum().idxmax()

    day = df[df["date"] == target].copy()
    if day.empty:
        raise ValueError(f"no rows for {target.date()} in {dataset_path}")

    # Clip to the real state border — the grid is a bbox, so ~37% of cells are
    # actually in Nevada/Arizona/offshore and shouldn't be drawn.
    before = len(day)
    day = filter_to_california(day)
    logger.info("Clipped to California: %s -> %s cells", before, len(day))

    day = day.join(load_model().predict(day))
    logger.info("Scored %s cells for %s (%s actual fires)",
                f"{len(day):,}", target.date(), int(day["has_fire"].sum()))

    geojson = day_to_feature_collection(day)
    meta = build_meta(day, target.strftime("%Y-%m-%d"), mode="replay",
                      source="model-snapshot", actual_fires=int(day["has_fire"].sum()))
    return geojson, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD date present in the dataset")
    ap.add_argument("--latest", action="store_true", help="use the most recent date")
    ap.add_argument("--out", type=Path, default=SNAPSHOT_DIR)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    geojson, meta = build_snapshot(date=args.date, latest=args.latest)
    write_snapshot(geojson, meta, args.out)
    logger.info("Wrote %s snapshot for %s -> %s  (%s)",
                meta["mode"], meta["data_date"], args.out, meta["tier_counts"])


if __name__ == "__main__":
    main()
