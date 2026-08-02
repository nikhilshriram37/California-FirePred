"""Materialise per-day model performance into ``model_metrics``.

Joins each day's stored predictions (``risk_scores``) with the backfilled outcomes
(``feature_history.has_fire``) and records how the deployed model actually did. This
is the retrain loop's memory: the drift time series that decides when live data has
matured enough to be worth training on, and the per-version baseline that
auto-rollback compares a newly promoted model against.

Every day is scored under both label definitions, which are *not* interchangeable:

  * fused     — any has_fire=1 (IRWIN + CAL FIRE + FIRMS). Runs ~2.65x the historical
                base rate, because FIRMS catches industrial and agricultural heat that
                no weather model can predict. Useful for recall monitoring only.
  * confirmed — only cells backed by an incident record (IRWIN / CAL FIRE). Runs ~1.11x
                the historical base rate, so it is the one comparable to backtest
                numbers and the one every promotion gate reads.

Dates whose labels are known-untrustworthy (``label_health.healthy = false``) are
skipped: a day we could not see is not a day with no fires.

Run:  python -m src.pipeline.metrics --days 7
      python -m src.pipeline.metrics --all --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import src.data_acquisition.config  # noqa: F401 — loads .env.local (SUPABASE_* etc.)
from src.pipeline.supabase_io import get_client

logger = logging.getLogger(__name__)

_PAGE = 1000
LABEL_DEFS = ("fused", "confirmed")


def _pull_date(client, table: str, cols: str, ds: str) -> pd.DataFrame:
    """Paginate one date's rows.

    Ordered deliberately: PostgREST ``.range()`` without an ``order_by`` returns
    inconsistent pages across a large table, which silently drops and duplicates rows.
    """
    rows: list[dict] = []
    frm = 0
    while True:
        page = (client.table(table).select(cols).eq("date", ds)
                .order("grid_id").range(frm, frm + _PAGE - 1).execute().data)
        rows += page
        if len(page) < _PAGE:
            break
        frm += _PAGE
    return pd.DataFrame(rows)


def load_day(client, ds: str) -> pd.DataFrame:
    """Predictions joined to outcomes for one date; empty if either side is missing."""
    pred = _pull_date(client, "risk_scores", "grid_id,tier,risk,raw_probability,model_version", ds)
    truth = _pull_date(client, "feature_history", "grid_id,has_fire,label_source", ds)
    if pred.empty or truth.empty:
        return pd.DataFrame()
    day = pred.merge(truth, on="grid_id")
    return day[day["has_fire"].notna()].copy()


def _tier_stats(prefix: str, flagged: np.ndarray, y: np.ndarray, base: float) -> dict:
    """Operational numbers for one tier mask: how much was flagged, and how much burned."""
    n_flag, n_fire = int(flagged.sum()), int(y.sum())
    hits = int((flagged & (y == 1)).sum())
    prec = hits / n_flag if n_flag else None
    return {
        f"{prefix}_flagged": n_flag,
        f"{prefix}_hits": hits,
        f"{prefix}_recall": hits / n_fire if n_fire else None,
        f"{prefix}_precision": prec,
        f"{prefix}_lift": (prec / base) if (prec is not None and base) else None,
    }


def compute(day: pd.DataFrame, ds: str, label_def: str) -> dict | None:
    """One scorecard row. Returns None if the day carries no usable predictions."""
    if day.empty:
        return None
    fused = day["has_fire"].to_numpy() == 1
    if label_def == "confirmed":
        y = fused & day["label_source"].fillna("").str.contains("irwin|calfire").to_numpy()
    else:
        y = fused
    y = y.astype(int)

    risk = day["risk"].to_numpy(dtype=float)
    tier = day["tier"].to_numpy()
    n, n_fire = len(day), int(y.sum())
    base = n_fire / n if n else 0.0

    # Ranking metrics need both classes present; a day with no fires under this
    # definition still yields valid calibration and tier counts, so keep the row.
    ranking = {"pr_auc": None, "roc_auc": None}
    if 0 < n_fire < n:
        ranking = {"pr_auc": float(average_precision_score(y, risk)),
                   "roc_auc": float(roc_auc_score(y, risk))}

    versions = day["model_version"].dropna().unique()
    if len(versions) > 1:
        logger.warning("%s: %d model versions in one day (%s) — recording the modal one",
                       ds, len(versions), ", ".join(map(str, versions)))

    return {
        "date": ds,
        "model_version": str(day["model_version"].mode().iat[0]) if len(versions) else "unknown",
        "label_def": label_def,
        "n_cells": n,
        "n_fires": n_fire,
        "base_rate": base,
        **_tier_stats("red", tier == "Red", y, base),
        **_tier_stats("ry", np.isin(tier, ["Red", "Yellow"]), y, base),
        **ranking,
        "brier": float(np.mean((risk - y) ** 2)),
        "mean_predicted": float(risk.mean()),
    }


def unhealthy_dates(client) -> set[str]:
    """Dates whose labels are known-bad. Empty if migration 0005 has not been run."""
    try:
        rows = client.table("label_health").select("date").eq("healthy", False).execute().data
        return {r["date"] for r in rows}
    except Exception as e:
        logger.warning("label_health unavailable (%s) — cannot filter untrustworthy dates; "
                       "run migration 0005", str(e)[:120])
        return set()


def persist(client, rows: list[dict]) -> bool:
    """Upsert scorecards. Returns False (and warns) if migration 0006 has not run."""
    if not rows:
        return False
    try:
        client.table("model_metrics").upsert(rows, on_conflict="date,model_version,label_def").execute()
    except Exception as e:
        msg = str(e)
        hint = " — run migration 0006" if "does not exist" in msg or "schema cache" in msg else ""
        logger.warning("model_metrics write skipped (%d rows): %s%s", len(rows), msg[:160], hint)
        return False
    return True


def run(dates: list[str], write: bool = True) -> list[dict]:
    """Compute (and optionally persist) scorecards for the given dates."""
    client = get_client()
    if client is None:
        logger.warning("Supabase not configured — nothing to do")
        return []
    skip = unhealthy_dates(client)
    out: list[dict] = []
    for ds in dates:
        if ds in skip:
            logger.info("%s: SKIPPED — label_health marks this date untrustworthy", ds)
            continue
        day = load_day(client, ds)
        if day.empty:
            logger.info("%s: no joined prediction/outcome rows", ds)
            continue
        rows = [r for ld in LABEL_DEFS if (r := compute(day, ds, ld))]
        out += rows
        c = {r["label_def"]: r for r in rows}
        logger.info("%s [%s] %d cells | fused %d fires (red recall %s) | confirmed %d fires (red recall %s)",
                    ds, rows[0]["model_version"], rows[0]["n_cells"],
                    c["fused"]["n_fires"], _pct(c["fused"]["red_recall"]),
                    c["confirmed"]["n_fires"], _pct(c["confirmed"]["red_recall"]))
    if write and out:
        persist(client, out)
    return out


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v * 100:.0f}%"


def _date_range(days: int, end: dt.date) -> list[str]:
    return [(end - dt.timedelta(days=d)).isoformat() for d in range(days, 0, -1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="trailing days to score")
    ap.add_argument("--date", help="score a single YYYY-MM-DD date instead")
    ap.add_argument("--end", help="end the trailing window at YYYY-MM-DD (default: today)")
    ap.add_argument("--all", action="store_true", help="score every date present in risk_meta")
    ap.add_argument("--dry-run", action="store_true", help="compute and print, but do not write")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.date:
        dates = [args.date]
    elif args.all:
        client = get_client()
        rows = client.table("risk_meta").select("data_date").execute().data
        dates = sorted({r["data_date"] for r in rows})
    else:
        end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
        dates = _date_range(args.days, end)

    rows = run(dates, write=not args.dry_run)
    print(f"\n{len(rows)} scorecard row(s) over {len(set(r['date'] for r in rows))} date(s)"
          f"{' (dry run, nothing written)' if args.dry_run else ''}")


if __name__ == "__main__":
    main()
