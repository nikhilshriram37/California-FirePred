"""What, exactly, turned the Gann Fire's cell Green?

The Gann Fire (2026-08-04, Calaveras County, 9,363 acres) is the case that exposed the
quiet-cell regression: cell 5917 was flagged Yellow for about ten days running by the
pre-recency model and Green by the model deployed 2026-08-05. Three things changed at
once on that date, and an aggregate cannot separate them:

    the model        34 features -> 37 (the recency block)
    the prior        production's fire history begins 2026-06-19, so the block was cold
    the cutoffs      thresholds re-derived on 2020 for a differently-shaped score

This walks the cell through each configuration in turn, holding the others fixed, and
finishes with the stratified tiering the remedy experiment recommends. The cell burned
one day after the evaluation panel ends, so every tier shown is a genuine forecast:
no configuration here can see the fire it is being judged on.

Run:  .venv/bin/python -m scripts.experiment_gann_tracer
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from src.data_acquisition.config import PROJECT_ROOT
from src.models.slice_eval import (QUIET_EPS, assign_tiers, coverage_tiers,
                                   hybrid_tiers, quiet_mask)
from scripts.analyze_feature_ablation import RED_COV, YELLOW_COV
from scripts.experiment_tiering_remedies import stratified_tiers

logger = logging.getLogger(__name__)
OUT_DIR = PROJECT_ROOT / "data" / "eval" / "ablation"
CELL = 5917
DAYS = 14


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    thr = json.loads((OUT_DIR / "thresholds.json").read_text())
    warm = pd.read_parquet(OUT_DIR / "scores_live2026.parquet")
    cold = pd.read_parquet(OUT_DIR / "scores_live2026_cold.parquet")
    q = quiet_mask(warm, QUIET_EPS)

    configs = []

    def add(label, panel, combo, tiers):
        sub = panel.assign(tier=tiers)
        sub = sub[sub.grid_id == CELL].sort_values("date").tail(DAYS)
        configs.append((label, sub.set_index(sub.date.dt.strftime("%m-%d"))["tier"]))

    def own(panel, combo):
        cal = joblib.load(OUT_DIR / f"calibrator_{combo}.joblib")
        return assign_tiers(cal.transform(panel[f"s_{combo}"].to_numpy()),
                            thr[combo]["red"], thr[combo]["yellow"])

    # 1. the pre-recency model, as the dashboard ran it before 2026-08-05
    add("old 34f, own cutoffs", warm, "WGCL", own(warm, "WGCL"))
    # 2. what actually shipped: new model reading a fire history that starts 06-19
    add("new 37f, own, COLD prior", cold, "WGCLR", own(cold, "WGCLR"))
    # 3. same model and cutoffs, but given the fire history that really existed
    add("new 37f, own, WARM prior", warm, "WGCLR", own(warm, "WGCLR"))
    # 4. cutoffs removed as a variable
    add("new 37f, matched coverage", warm, "WGCLR",
        coverage_tiers(warm["s_WGCLR"].to_numpy(), RED_COV, YELLOW_COV))
    # 5. cutoffs computed entirely within regime
    add("new 37f, STRATIFIED", warm, "WGCLR",
        stratified_tiers(warm["s_WGCLR"].to_numpy(), q, RED_COV, YELLOW_COV))
    # 6. the recommendation: Red global (unchanged meaning), Yellow within regime
    add("new 37f, HYBRID", warm, "WGCLR",
        hybrid_tiers(warm["s_WGCLR"].to_numpy(), q, RED_COV, YELLOW_COV))

    tbl = pd.DataFrame(dict(configs))
    print("\n" + "=" * 96)
    print(f"CELL {CELL} — Gann Fire, ignited 2026-08-04. Every column is a forecast:")
    print("  the panel ends 08-03, so no configuration can see the fire it is judged on.")
    print("=" * 96)
    print(tbl.to_string())

    r = warm[warm.grid_id == CELL].sort_values("date").iloc[-1]
    print(f"\ncell state on 08-03: fire_recency_cell {r.fire_recency_cell:.4f}  "
          f"fire_recency_nbr {r.fire_recency_nbr:.4f}  "
          f"days_since_fire_cell {r.days_since_fire_cell:.0f}  -> quiet={bool(q[warm.grid_id.eq(CELL).to_numpy() & warm.date.eq(r.date).to_numpy()][0])}")

    tbl.to_csv(OUT_DIR / "gann_tracer.csv")
    logger.info("wrote gann_tracer.csv")


if __name__ == "__main__":
    main()
