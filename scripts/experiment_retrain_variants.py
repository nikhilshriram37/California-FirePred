"""Retrain experiment: train feature-set variants into models/candidate/<name>/.
Live models/ is untouched. Same params/split/calibration as src.models.train."""
from __future__ import annotations
import json, logging
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_acquisition.config import PROCESSED_DIR, PROJECT_ROOT, REFERENCE_DIR
from src.models.features import FEATURE_COLS, TARGET_COL, merge_static_features
from src.models.train import TUNED_PARAMS, TRAIN_YEARS, TEST_YEAR, RED_RECALL, YELLOW_RECALL, _threshold_for_recall

logging.basicConfig(level=logging.WARNING)
CAND = PROJECT_ROOT / "models" / "candidate"

BASE = list(FEATURE_COLS)
NO_AET = [c for c in BASE if c not in ("aet", "water_deficit")]
VARIANTS = {
    "A_baseline":       (BASE, False),
    "B_noaet":          (NO_AET, False),
    "C_ignition":       (BASE + ["ignition_density"], True),
    "D_noaet_ignition": (NO_AET + ["ignition_density"], True),
}


def load_dataset(with_ignition: bool) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "california_dataset.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = merge_static_features(df)
    if with_ignition:
        ig = pd.read_json(REFERENCE_DIR / "ignition_density.json")
        df = df.merge(ig, on="grid_id", how="left")
        df["ignition_density"] = df["ignition_density"].fillna(0.0)
    return df


def train_variant(name: str, feats: list[str], with_ignition: bool) -> dict:
    df = load_dataset(with_ignition)
    tr = df[df["date"].dt.year.isin(TRAIN_YEARS)]
    te = df[df["date"].dt.year == TEST_YEAR]
    Xtr, ytr = tr[feats], tr[TARGET_COL].to_numpy()
    Xte, yte = te[feats], te[TARGET_COL].to_numpy()

    model = XGBClassifier(**TUNED_PARAMS, tree_method="hist", eval_metric="aucpr",
                          n_jobs=-1, random_state=42)
    model.fit(Xtr, ytr)
    raw = model.predict_proba(Xte)[:, 1]
    pr, roc = float(average_precision_score(yte, raw)), float(roc_auc_score(yte, raw))

    cal_idx, _ = train_test_split(np.arange(len(yte)), test_size=0.5, stratify=yte, random_state=42)
    calib = IsotonicRegression(out_of_bounds="clip"); calib.fit(raw[cal_idx], yte[cal_idx])
    cp, yc = calib.transform(raw[cal_idx]), yte[cal_idx]
    red_t, yel_t = _threshold_for_recall(yc, cp, RED_RECALL), _threshold_for_recall(yc, cp, YELLOW_RECALL)

    d = CAND / name; d.mkdir(parents=True, exist_ok=True)
    model.save_model(str(d / "xgb_model.json"))
    joblib.dump(calib, d / "calibrator.joblib")
    (d / "thresholds.json").write_text(json.dumps(
        {"red": red_t, "yellow": yel_t, "red_recall_target": RED_RECALL, "yellow_recall_target": YELLOW_RECALL}, indent=2))
    (d / "feature_list.json").write_text(json.dumps(feats, indent=2))
    card = {"model": "xgboost-isotonic", "variant": name,
            "version": "cand-" + datetime.now(timezone.utc).strftime("%Y%m%d"),
            "n_features": len(feats), "features": feats,
            "metrics": {"pr_auc": pr, "roc_auc": roc}, "tiers": {"red": red_t, "yellow": yel_t}}
    (d / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"{name:18s} nfeat={len(feats):2d}  backtest(2020) PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}")
    return card


if __name__ == "__main__":
    print(f"train={TRAIN_YEARS} test={TEST_YEAR}\n")
    for nm, (fs, wi) in VARIANTS.items():
        train_variant(nm, fs, wi)
    print("\nartifacts -> models/candidate/<variant>/ (live models/ untouched)")
