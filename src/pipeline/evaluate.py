"""Real-world evaluation: compare stored predictions against what actually burned.

Joins the daily nowcast predictions (risk_scores) with the backfilled outcomes
(feature_history.has_fire + label_source) and reports recall / precision / lift by
risk tier. Reports under two label definitions so results aren't distorted by the
label-source mismatch:

  * fused     — any has_fire=1 (IRWIN + CAL FIRE + FIRMS); high recall, but includes
                FIRMS-only heat detections and small/human-caused ignitions a
                weather model can't predict.
  * confirmed — only cells confirmed by an incident record (IRWIN or CAL FIRE);
                a stricter, cleaner ground truth.

Run:  python -m src.pipeline.evaluate                 # all labeled dates
      python -m src.pipeline.evaluate --date 2026-06-19
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import src.data_acquisition.config  # noqa: F401 — loads .env.local (SUPABASE_* etc.)
from src.pipeline.supabase_io import get_client

logger = logging.getLogger(__name__)


def _pull(client, table, cols, date=None):
    rows = []
    for frm in range(0, 200_000, 1000):
        q = client.table(table).select(cols)
        if date:
            q = q.eq("date", date)
        data = q.range(frm, frm + 999).execute().data
        if not data:
            break
        rows += data
        if len(data) < 1000:
            break
    return pd.DataFrame(rows)


def load(date: str | None = None) -> pd.DataFrame:
    """Merged predictions + outcomes for the given date (or all labeled dates)."""
    c = get_client()
    pred = _pull(c, "risk_scores", "grid_id,date,tier,risk", date)
    truth = _pull(c, "feature_history", "grid_id,date,has_fire,label_source", date)
    if pred.empty or truth.empty:
        return pd.DataFrame()
    m = pred.merge(truth, on=["grid_id", "date"])
    return m[m["has_fire"].notna()].copy()


def _report(m: pd.DataFrame, label_name: str, fire_mask: np.ndarray) -> None:
    y = fire_mask.astype(int)
    n, nf = len(m), int(y.sum())
    base = nf / n * 100 if n else 0
    print(f"\n--- {label_name} labels: {nf} fires / {n} cell-days (base rate {base:.2f}%) ---")
    if nf == 0:
        print("  (no fires under this definition)")
        return
    tier = m["tier"].to_numpy()
    for name, mask in [("Red", tier == "Red"), ("Red+Yellow", np.isin(tier, ["Red", "Yellow"]))]:
        flagged = int(mask.sum())
        hit = int((mask & (y == 1)).sum())
        recall = hit / nf * 100
        prec = hit / flagged * 100 if flagged else 0
        lift = (prec / base) if base else 0
        print(f"  {name:11s}: {flagged:5d} flagged | {hit:3d}/{nf} caught = {recall:4.0f}% recall | "
              f"{prec:4.1f}% burned ({lift:.1f}x base)")


def evaluate(date: str | None = None) -> None:
    m = load(date)
    if m.empty:
        print("No labeled prediction data yet (need risk_scores + backfilled has_fire).")
        return
    dates = sorted(m["date"].unique())
    print(f"REAL-WORLD EVALUATION | {len(m):,} cell-days over {len(dates)} day(s): "
          f"{dates[0]}..{dates[-1]}")

    fused = m["has_fire"].to_numpy() == 1
    confirmed = fused & m["label_source"].fillna("").str.contains("irwin|calfire").to_numpy()
    _report(m, "FUSED (IRWIN+CALFIRE+FIRMS)", fused)
    _report(m, "CONFIRMED-incident only (IRWIN/CALFIRE)", confirmed)
    print("\nNote: 'confirmed' is the fairer test for a weather-driven model — it drops "
          "FIRMS-only heat detections (industrial/ag/artifacts).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="evaluate a single YYYY-MM-DD date")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    evaluate(args.date)


if __name__ == "__main__":
    main()
