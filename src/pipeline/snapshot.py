"""Shared GeoJSON snapshot format for the dashboard.

Both the historical replay (:mod:`src.pipeline.export_snapshot`) and the live
scorer (:mod:`src.pipeline.score_daily`) emit the *same* shape via these helpers,
so the dashboard reads one format regardless of source.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data_acquisition.config import PROJECT_ROOT
from src.models.predict import load_model

SNAPSHOT_DIR = PROJECT_ROOT / "dashboard" / "public" / "data"
CELL_DEG = 0.1  # grid resolution (matches build_dataset.build_grid)

# Interpretable drivers surfaced in the cell click-through panel (operator trust).
DETAIL_FEATURES = [
    "vpd", "fm100", "dry_streak", "bi_7d", "erc_7d",
    "tmmx_c", "rmin", "pr_14d", "lightning_count",
]


def _cell_polygon(lon: float, lat: float, half: float = CELL_DEG / 2) -> list:
    return [[
        [lon - half, lat - half], [lon + half, lat - half],
        [lon + half, lat + half], [lon - half, lat + half],
        [lon - half, lat - half],
    ]]


def day_to_feature_collection(day: pd.DataFrame) -> dict:
    """Scored day -> GeoJSON FeatureCollection of cell polygons.

    ``day`` must carry grid_id, lat_center, lon_center, risk, tier (+ optionally
    has_fire and the DETAIL_FEATURES).
    """
    features = []
    for _, r in day.iterrows():
        props = {
            "grid_id": int(r["grid_id"]),
            "risk": round(float(r["risk"]), 5),
            "tier": str(r["tier"]),
            "lat": round(float(r["lat_center"]), 4),
            "lon": round(float(r["lon_center"]), 4),
        }
        if "has_fire" in r and pd.notna(r["has_fire"]):
            props["has_fire"] = int(r["has_fire"])
        for f in DETAIL_FEATURES:
            if f in r and pd.notna(r[f]):
                props[f] = round(float(r[f]), 3)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": _cell_polygon(r["lon_center"], r["lat_center"])},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}


def build_meta(day: pd.DataFrame, data_date, mode: str, source: str, **extra) -> dict:
    """Snapshot metadata: date, freshness, tier counts, model version."""
    counts = day["tier"].value_counts().reindex(["Red", "Yellow", "Green"]).fillna(0).astype(int)
    model = load_model()
    meta = {
        "data_date": str(data_date),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "mode": mode,
        "model_version": model.version,
        "n_cells": int(len(day)),
        "tier_counts": {k: int(v) for k, v in counts.items()},
        "thresholds": model.thresholds,
    }
    meta.update(extra)
    return meta


def write_snapshot(geojson: dict, meta: dict, out_dir: Path = SNAPSHOT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "risk_snapshot.geojson").write_text(json.dumps(geojson))
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
