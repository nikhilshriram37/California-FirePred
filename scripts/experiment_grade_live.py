"""Grade candidate models vs the current LIVE model on the accumulated live eval.

Re-scores the labeled live days from stored feature_history (the exact features the
live pipeline computed) with each model's own artifacts, so differences are purely
the model. Reports live ROC-AUC / PR-AUC and recall at an equal alarm budget."""
from __future__ import annotations
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import src.data_acquisition.config  # loads .env.local
from src.data_acquisition.config import PROJECT_ROOT, REFERENCE_DIR
from src.pipeline.supabase_io import get_client

# ---- pull stored live features + outcomes -------------------------------------
c = get_client()
def pull(table, cols):
    rows = []
    for frm in range(0, 200_000, 1000):
        d = c.table(table).select(cols).range(frm, frm+999).execute().data
        if not d: break
        rows += d
        if len(d) < 1000: break
    return pd.DataFrame(rows)

truth = pull("feature_history", "grid_id,date,has_fire,label_source,features")
truth = truth[truth["has_fire"].notna()].copy()
feats = pd.json_normalize(truth["features"]).set_index(truth.index)
data = pd.concat([truth[["grid_id", "date", "has_fire"]], feats], axis=1)

ig = pd.read_json(REFERENCE_DIR / "ignition_density.json")
data = data.merge(ig, on="grid_id", how="left")
data["ignition_density"] = data["ignition_density"].fillna(0.0)
y = (data["has_fire"] == 1).to_numpy().astype(int)
nf, n = int(y.sum()), len(data)
print(f"live eval: {n} cell-days, {nf} fires, base {nf/n*100:.2f}%\n")

# ---- score a model dir --------------------------------------------------------
def score(models_dir: Path) -> np.ndarray:
    m = XGBClassifier(); m.load_model(str(models_dir / "xgb_model.json"))
    calib = joblib.load(models_dir / "calibrator.joblib")
    fl = json.loads((models_dir / "feature_list.json").read_text())
    X = data.reindex(columns=fl)
    return calib.transform(m.predict_proba(X)[:, 1])

def recall_at_budget(r, budget_frac):
    """Recall when flagging the top `budget_frac` of cells by risk."""
    k = int(np.ceil(budget_frac * n))
    thr = np.sort(r)[::-1][min(k, n-1)]
    fl = r >= thr
    return y[fl].sum()/nf*100, fl.mean()*100

def recall_for_target(r, target):
    order = np.argsort(-r); cum = np.cumsum(y[order])
    k = np.searchsorted(cum, target*nf)
    thr = r[order][min(k, n-1)]; fl = r >= thr
    return (r >= thr).mean()*100  # % of state flagged to hit target recall

CAND = PROJECT_ROOT / "models" / "candidate"
models = {
    "LIVE (current)": PROJECT_ROOT / "models",
    "A_baseline":     CAND / "A_baseline",
    "B_noaet":        CAND / "B_noaet",
    "C_ignition":     CAND / "C_ignition",
    "D_noaet_ignition": CAND / "D_noaet_ignition",
}

# equal alarm budget = what current LIVE red flags on this data
r_live = score(models["LIVE (current)"])
budget = (r_live >= json.loads((PROJECT_ROOT/"models"/"thresholds.json").read_text())["red"]).mean()
print(f"equal-alarm budget = {budget*100:.1f}% of state (current red flag rate)\n")

print(f"{'model':18s} {'ROC':>6s} {'PR-AUC':>7s} {'recall@budget':>14s} {'%state@55%rec':>13s}")
for name, d in models.items():
    r = score(d)
    roc, pr = roc_auc_score(y, r), average_precision_score(y, r)
    rb, fb = recall_at_budget(r, budget)
    st55 = recall_for_target(r, 0.55)
    print(f"{name:18s} {roc:6.3f} {pr:7.4f} {rb:9.0f}% @{fb:3.0f}%   {st55:11.0f}%")
