"""Assemble live model features for the latest available day.

Merges recent gridMET + seasonal dryness + GOES-GLM lightning, runs the *exact*
training feature engineering (:func:`src.preprocessing.build_dataset.engineer_features`),
then returns the target (latest) day clipped to California — ready for
:func:`src.models.predict.predict`.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.models.features import FEATURE_COLS, merge_static_features
from src.preprocessing.build_dataset import engineer_features
from src.pipeline.geo import filter_to_california

logger = logging.getLogger(__name__)


def build_live_features(
    grid: pd.DataFrame,
    gridmet_recent: pd.DataFrame,
    dryness: pd.DataFrame,
    lightning: pd.DataFrame,
    target_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return (features_for_target_day, target_date).

    Args:
        grid: canonical cells (grid_id, lat_center, lon_center).
        gridmet_recent: recent gridMET (grid_id, date, 8 raw vars).
        dryness: per-cell aet, water_deficit (seasonal normal for the month).
        lightning: per-cell lightning_count for the target day (GOES-GLM).
        target_date: day to score; defaults to the latest gridMET date.
    """
    df = gridmet_recent.merge(grid[["grid_id", "lat_center", "lon_center"]], on="grid_id", how="left")
    df = df.merge(dryness, on="grid_id", how="left")

    # Fill dryness gaps (coastal cells) with the statewide median, as in training.
    for col in ["aet", "water_deficit"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # engineer_features computes tmmx_c, the 7d/14d rollings, dry_streak, the 3-day
    # changes, and the temporal encodings — identical to the training pipeline.
    df = engineer_features(df)

    target = pd.Timestamp(target_date) if target_date is not None else df["date"].max()
    day = df[df["date"] == target].copy()
    if day.empty:
        raise ValueError(f"no gridMET rows for target date {target.date()}")

    # Apply same-day lightning from GOES-GLM (0 where no flashes).
    day = day.drop(columns=["lightning_count"], errors="ignore").merge(lightning, on="grid_id", how="left")
    day["lightning_count"] = day["lightning_count"].fillna(0).astype(int)

    day = merge_static_features(day)  # topography + human-exposure (per-cell, static)

    before = len(day)
    day = filter_to_california(day)

    # Drop cells that can't be fully scored — any NaN in the model's feature
    # vector. In practice these are remote offshore islands (San Clemente, San
    # Nicolas) whose center is inside the CA polygon but whose nearest gridMET
    # pixel is ocean, leaving weather (and its rolling derivatives) undefined.
    # Training likewise dropped weatherless ocean cells; scoring them would
    # fabricate a tier from XGBoost's missing-value default paths.
    incomplete = day[FEATURE_COLS].isna().any(axis=1)
    if incomplete.any():
        logger.info("dropping %s cell(s) with incomplete features: %s",
                    int(incomplete.sum()), day.loc[incomplete, "grid_id"].tolist())
        day = day[~incomplete].copy()

    logger.info("live features: %s cells for %s (CA-clipped from %s)",
                len(day), target.date(), before)
    return day, target
