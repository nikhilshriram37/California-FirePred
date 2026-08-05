"""Load the exported model and score feature rows -> probability + risk tier.

The single inference path shared by the daily scoring pipeline and any on-demand
API. Artifacts (written by :mod:`src.models.train`) are loaded once and cached.

    from src.models.predict import predict
    scored = predict(features_df)   # -> raw_probability, risk, tier
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data_acquisition.config import PROJECT_ROOT
from src.models.features import select_features
from src.models.recency import RECENCY_FEATURES, is_quiet

MODELS_DIR = PROJECT_ROOT / "models"


class RiskModel:
    """Bundles the booster, isotonic calibrator, and tier thresholds."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.model = XGBClassifier()
        self.model.load_model(str(self.models_dir / "xgb_model.json"))
        self.calibrator = joblib.load(self.models_dir / "calibrator.joblib")
        self.thresholds = json.loads((self.models_dir / "thresholds.json").read_text())
        self.features = json.loads((self.models_dir / "feature_list.json").read_text())
        self.card = json.loads((self.models_dir / "model_card.json").read_text())

    @property
    def version(self) -> str:
        return self.card.get("version", "unknown")

    def to_tier(self, risk: np.ndarray, quiet: np.ndarray | None = None) -> np.ndarray:
        """Map calibrated risk to Red / Yellow / Green.

        Red uses one statewide cutoff, so the colour means the same thing everywhere.
        Yellow uses a per-regime cutoff when ``quiet`` is supplied and the artifacts
        carry regime thresholds: a single global Yellow boundary leaves areas with no
        recent fire nearby almost uniformly Green, because the model concentrates its
        probability mass where fires cluster. Measured on the live record, splitting
        the Yellow boundary raises the share of quiet-area ignitions carrying any
        warning from 19.7% to 65.8% without moving Red at all.

        Falls back to the global Yellow cutoff when the regime is unknown or the
        artifacts predate this change, so an older models/ directory still scores.
        """
        red = self.thresholds["red"]
        yq, ya = self.thresholds.get("yellow_quiet"), self.thresholds.get("yellow_active")
        if quiet is None or yq is None or ya is None:
            yellow = np.full(len(risk), self.thresholds["yellow"], dtype=float)
        else:
            yellow = np.where(quiet, yq, ya)
        return np.where(risk >= red, "Red", np.where(risk >= yellow, "Yellow", "Green"))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a feature frame.

        Returns a frame (aligned to ``df.index``) with:
            raw_probability  uncalibrated XGBoost output
            risk             calibrated fire probability
            tier             "Red" / "Yellow" / "Green"
        """
        X = select_features(df, strict=False)
        raw = self.model.predict_proba(X)[:, 1]
        risk = self.calibrator.transform(raw)
        # The regime is read from the recency features, which both the nowcast and every
        # forecast horizon already carry. If they are absent the tiering silently falls
        # back to the global cutoff rather than guessing a regime.
        quiet = is_quiet(df) if all(c in df.columns for c in RECENCY_FEATURES[:2]) else None
        return pd.DataFrame(
            {"raw_probability": raw, "risk": risk, "tier": self.to_tier(risk, quiet)},
            index=df.index,
        )


@lru_cache(maxsize=1)
def load_model(models_dir: str | None = None) -> RiskModel:
    """Load (and cache) the risk model. Pass a dir to override the default."""
    return RiskModel(Path(models_dir) if models_dir else MODELS_DIR)


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper using the cached default model."""
    return load_model().predict(df)
