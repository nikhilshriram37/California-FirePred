"""Recent confirmed ignitions, for the serving side of the fire-recency prior.

Scoring now depends on the label record, not just the weather feed: the model carries
a decaying prior over where fires have recently been (see src/models/recency.py), and
that prior has to be rebuilt at score time from what the label job has written.

Only *confirmed* ignitions count — IRWIN and CAL FIRE incidents, which are agency
records of a fire starting. FIRMS-only detections are excluded: they mark thermal
activity anywhere a fire is burning, not where one began, and the model is trained on
FPA-FOD ignition records. Feeding it fire *presence* where it learned fire *starts*
would be exactly the train/serve mismatch this feature was added to remove.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# How much history to pull. The prior decays to well under 1% of its initial weight
# over this span at the default tau, so nothing meaningful is lost by cutting here.
LOOKBACK_DAYS = 120

# Serving degrades quietly if the label job stalls: the prior simply decays toward
# zero and the model reverts to weather-only behaviour, under-predicting. It cannot be
# detected from the scores themselves, so it is checked explicitly.
MAX_ACCEPTABLE_STALENESS_DAYS = 5


def fetch_recent_fires(client, end: dt.date,
                       lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Confirmed ignition cell-days in the trailing window, as (grid_id, date).

    Returns an empty frame — never raises — when Supabase is unavailable or the
    label_source column predates migration 0002. The caller decides what an empty
    history means; recency features then evaluate to "nothing burned recently",
    which is a real (if pessimistic) belief rather than a missing value.
    """
    if client is None:
        logger.warning("no Supabase client — fire-recency prior will be empty")
        return pd.DataFrame(columns=["grid_id", "date"])

    start = (end - dt.timedelta(days=lookback_days)).isoformat()
    rows, offset = [], 0
    try:
        while True:
            page = (client.table("feature_history")
                    .select("grid_id,date,label_source")
                    .eq("has_fire", 1)
                    .gte("date", start).lte("date", end.isoformat())
                    .order("date").order("grid_id")
                    .range(offset, offset + 999).execute().data)
            rows += page
            if len(page) < 1000:
                break
            offset += 1000
    except Exception as e:
        logger.error("could not read fire history (%s) — recency prior will be empty", e)
        return pd.DataFrame(columns=["grid_id", "date"])

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("no confirmed fires found in the last %d days — is the label "
                       "job healthy?", lookback_days)
        return pd.DataFrame(columns=["grid_id", "date"])

    # label_source is absent pre-0002 and NULL for FIRMS-only rows written before the
    # column existed; treat a missing value as unconfirmed rather than assuming.
    src = df.get("label_source", pd.Series([None] * len(df))).fillna("")
    df = df[src != "firms"]
    df["date"] = pd.to_datetime(df["date"])
    return df[["grid_id", "date"]].reset_index(drop=True)


def check_freshness(fires: pd.DataFrame, target: pd.Timestamp | dt.date,
                    lag_days: int) -> int | None:
    """Log how stale the fire history is relative to the day being scored.

    Returns the staleness in days, or None if there is no history at all. Staleness up
    to ``lag_days`` is expected and costs nothing, because the feature deliberately
    ignores anything fresher than that.
    """
    if fires.empty:
        logger.error("fire-recency prior is EMPTY — the model will score as if nothing "
                     "has burned recently and will under-predict. Check the label job.")
        return None
    newest = pd.Timestamp(fires["date"].max())
    stale = (pd.Timestamp(target) - newest).days
    if stale > MAX_ACCEPTABLE_STALENESS_DAYS:
        logger.error("fire history is %d days behind %s (newest label %s) — beyond the "
                     "%d-day tolerance; the recency prior is decaying toward zero and "
                     "risk will be understated", stale, pd.Timestamp(target).date(),
                     newest.date(), MAX_ACCEPTABLE_STALENESS_DAYS)
    elif stale > lag_days:
        logger.warning("fire history is %d days behind %s (feature lag is %d)",
                       stale, pd.Timestamp(target).date(), lag_days)
    else:
        logger.info("fire history current to %s (%d confirmed cell-days in window)",
                    newest.date(), len(fires))
    return stale
