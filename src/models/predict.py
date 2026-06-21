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

    def to_tier(self, risk: np.ndarray) -> np.ndarray:
        red, yellow = self.thresholds["red"], self.thresholds["yellow"]
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
        return pd.DataFrame(
            {"raw_probability": raw, "risk": risk, "tier": self.to_tier(risk)},
            index=df.index,
        )


@lru_cache(maxsize=1)
def load_model(models_dir: str | None = None) -> RiskModel:
    """Load (and cache) the risk model. Pass a dir to override the default."""
    return RiskModel(Path(models_dir) if models_dir else MODELS_DIR)


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper using the cached default model."""
    return load_model().predict(df)
