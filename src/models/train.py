"""Train the wildfire risk model and export the serving artifacts.

Reproduces the tuned model from ``notebooks/03_model_prediction.ipynb`` *without*
re-running the 50-trial Optuna search — the best hyperparameters it found are
pinned in :data:`TUNED_PARAMS`. Writes everything :mod:`src.models.predict` needs
to score live data:

    models/xgb_model.json      trained XGBoost model (sklearn wrapper format)
    models/calibrator.joblib   isotonic calibration (raw prob -> calibrated risk)
    models/thresholds.json     red / yellow risk-tier cutoffs
    models/feature_list.json   the 28 feature names, in order
    models/model_card.json     metrics + provenance

Run:  python -m src.models.train
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_acquisition.config import PROCESSED_DIR, PROJECT_ROOT
from src.models.features import FEATURE_COLS, TARGET_COL, merge_static_features, select_features

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"

# Best hyperparameters from the 50-trial Optuna search (notebook cell 8), with
# n_estimators pinned to the early-stopped best iteration (296). Reusing these
# makes training fast and deterministic; the periodic retrain job may re-search.
TUNED_PARAMS: dict = dict(
    learning_rate=0.03329,
    max_depth=8,
    min_child_weight=5.073,
    gamma=0.1192,
    subsample=0.8006,
    colsample_bytree=0.7746,
    reg_lambda=3.464,
    reg_alpha=0.02903,
    scale_pos_weight=4.945,
    max_delta_step=4,
    n_estimators=296,
)

TRAIN_YEARS = [2018, 2019]
TEST_YEAR = 2020
RED_RECALL = 0.55     # red = tight "send resources here" tier: catches ~59% of fires,
                      # flags ~3.5% of the state, ~17x base rate (sweet spot — above 0.60
                      # recall plateaus and lift falls off, diluting actionability).
YELLOW_RECALL = 0.80  # red+yellow catches ~81% of fires, ~10% of state flagged


def _threshold_for_recall(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Highest probability threshold that still recalls >= ``target`` of positives."""
    _, recall, thr = precision_recall_curve(y, p)
    ok = np.where(recall[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.0


def train(dataset_path: Path | None = None, models_dir: Path = MODELS_DIR) -> dict:
    """Fit the model + calibrator, derive tier thresholds, and write artifacts."""
    dataset_path = dataset_path or PROCESSED_DIR / "california_dataset.parquet"
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", dataset_path)
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])
    df = merge_static_features(df)  # topography + human-exposure (per-cell, static)

    train_df = df[df["date"].dt.year.isin(TRAIN_YEARS)]
    test_df = df[df["date"].dt.year == TEST_YEAR]
    X_train, y_train = select_features(train_df), train_df[TARGET_COL].to_numpy()
    X_test, y_test = select_features(test_df), test_df[TARGET_COL].to_numpy()
    logger.info("train=%s rows (%s fires)  test=%s rows (%s fires)",
                f"{len(X_train):,}", f"{int(y_train.sum()):,}",
                f"{len(X_test):,}", f"{int(y_test.sum()):,}")

    # 1. Fit the tuned XGBoost model on the training years.
    model = XGBClassifier(**TUNED_PARAMS, tree_method="hist", eval_metric="aucpr",
                          n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    raw_test = model.predict_proba(X_test)[:, 1]
    pr_auc = float(average_precision_score(y_test, raw_test))
    roc_auc = float(roc_auc_score(y_test, raw_test))
    logger.info("test PR-AUC=%.4f  ROC-AUC=%.4f", pr_auc, roc_auc)

    # 2. Calibrate (isotonic) on a stratified half of the test year, and derive
    #    the tier cutoffs on that same calibration half (matches the notebook).
    cal_idx, _ = train_test_split(
        np.arange(len(y_test)), test_size=0.5, stratify=y_test, random_state=42)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_test[cal_idx], y_test[cal_idx])

    cal_p = calibrator.transform(raw_test[cal_idx])
    y_cal = y_test[cal_idx]
    red_t = _threshold_for_recall(y_cal, cal_p, RED_RECALL)
    yellow_t = _threshold_for_recall(y_cal, cal_p, YELLOW_RECALL)
    logger.info("tier cutoffs: red>=%.4f  yellow>=%.4f", red_t, yellow_t)

    # 3. Persist artifacts.
    model.save_model(str(models_dir / "xgb_model.json"))
    joblib.dump(calibrator, models_dir / "calibrator.joblib")
    (models_dir / "thresholds.json").write_text(json.dumps({
        "red": red_t, "yellow": yellow_t,
        "red_recall_target": RED_RECALL, "yellow_recall_target": YELLOW_RECALL,
    }, indent=2))
    (models_dir / "feature_list.json").write_text(json.dumps(FEATURE_COLS, indent=2))

    model_card = {
        "model": "xgboost-isotonic",
        "version": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_years": TRAIN_YEARS,
        "test_year": TEST_YEAR,
        "n_features": len(FEATURE_COLS),
        "params": TUNED_PARAMS,
        "metrics": {"pr_auc": pr_auc, "roc_auc": roc_auc},
        "tiers": {"red": red_t, "yellow": yellow_t},
    }
    (models_dir / "model_card.json").write_text(json.dumps(model_card, indent=2))
    logger.info("wrote artifacts to %s", models_dir)
    return model_card


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(json.dumps(train(), indent=2))
