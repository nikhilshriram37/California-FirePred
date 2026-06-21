"""Persist daily scores + streamed features to Supabase (optional).

Graceful by design: if SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY aren't set (or the
`supabase` package isn't installed), :func:`persist_daily` logs and returns without
error — the local snapshot still gets written by the caller. This lets the pipeline
run identically with or without a database.

Schema: see supabase/migrations/0001_init.sql.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

from src.models.features import FEATURE_COLS, select_features

logger = logging.getLogger(__name__)

_CHUNK = 1000  # rows per upsert request


def get_client():
    """Return a service-role Supabase client, or None if unavailable."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        return None
    try:
        from supabase import create_client
    except ImportError:
        logger.warning("supabase package not installed (pip install supabase) — skipping persistence")
        return None
    return create_client(url, key)


def _chunked(rows: list[dict]):
    for i in range(0, len(rows), _CHUNK):
        yield rows[i:i + _CHUNK]


def _upsert(client, table: str, rows: list[dict], on_conflict: str | None = None) -> None:
    for chunk in _chunked(rows):
        q = client.table(table).upsert(chunk, on_conflict=on_conflict) if on_conflict \
            else client.table(table).upsert(chunk)
        q.execute()


def persist_daily(day: pd.DataFrame, meta: dict, client=None) -> bool:
    """Write grid_cells, risk_scores, feature_history, and risk_meta for one day.

    ``day`` must carry grid_id, lat_center, lon_center, risk, raw_probability, tier
    plus the model feature columns. Returns True if written, False if skipped.
    """
    client = client or get_client()
    if client is None:
        logger.info("Supabase not configured — skipping persistence (local snapshot still written)")
        return False

    date = meta["data_date"]
    version = meta.get("model_version")

    # 1. Static grid (idempotent upsert).
    cells = day[["grid_id", "lat_center", "lon_center"]].drop_duplicates("grid_id")
    _upsert(client, "grid_cells", [
        {"grid_id": int(r.grid_id), "lat_center": float(r.lat_center), "lon_center": float(r.lon_center)}
        for r in cells.itertuples()
    ], on_conflict="grid_id")

    # 2. Risk scores for the day.
    _upsert(client, "risk_scores", [
        {"grid_id": int(r.grid_id), "date": date, "raw_probability": float(r.raw_probability),
         "risk": float(r.risk), "tier": str(r.tier), "model_version": version}
        for r in day.itertuples()
    ], on_conflict="grid_id,date")

    # 3. Streamed features → grows the dataset (labels backfilled later).
    feats = select_features(day, strict=False)
    feat_rows = []
    for gid, (_, frow) in zip(day["grid_id"].to_numpy(), feats.iterrows()):
        feat_rows.append({
            "grid_id": int(gid), "date": date,
            "features": {c: (None if pd.isna(frow[c]) else float(frow[c])) for c in FEATURE_COLS},
        })
    _upsert(client, "feature_history", feat_rows, on_conflict="grid_id,date")

    # 4. Snapshot metadata.
    client.table("risk_meta").insert({
        "data_date": date, "source": meta.get("source"), "mode": meta.get("mode"),
        "model_version": version, "n_cells": meta.get("n_cells"),
        "tier_counts": meta.get("tier_counts"), "thresholds": meta.get("thresholds"),
        "actual_fires": meta.get("actual_fires"), "lightning_cells": meta.get("lightning_cells"),
    }).execute()

    logger.info("Persisted %s cells to Supabase for %s", len(day), date)
    return True
