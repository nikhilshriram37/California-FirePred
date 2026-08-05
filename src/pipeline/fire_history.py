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
import time

import pandas as pd

logger = logging.getLogger(__name__)

# How much history to pull. The prior decays to under 2% of its initial weight over
# this span at the default tau, so nothing meaningful is lost by cutting here.
LOOKBACK_DAYS = 120

# Fetched in date slices rather than one range query. `feature_history` is indexed on
# date but not on has_fire, so a 120-day scan for confirmed fires reads ~500k rows and
# trips Postgres' statement timeout — intermittently, which is worse than reliably.
# Slicing keeps each scan small enough to complete. Migration 0006 adds the partial
# index that makes this cheap; the slicing stays regardless, as it costs nothing.
CHUNK_DAYS = 15
RETRIES = 3
# ~35 confirmed ignitions a day statewide in peak season, so a 15-day slice runs to a
# few hundred rows. The cap is a tripwire against silent truncation, not a page size.
CHUNK_ROW_CAP = 5000

# Serving degrades quietly if the label job stalls: the prior decays toward zero and
# the model reverts to weather-only behaviour, under-predicting. That cannot be seen
# in the scores themselves, so it is checked explicitly.
MAX_ACCEPTABLE_STALENESS_DAYS = 5


class FireHistoryUnavailable(RuntimeError):
    """The fire-recency prior could not be read.

    Raised rather than returning an empty frame, because the two are not remotely the
    same thing. An empty prior tells the model nothing has burned recently, which
    produces a confident all-green map — the single most dangerous way for a fire
    dashboard to fail. It happened: on 2026-08-05 a statement timeout published a
    forecast of 4,169 green cells across all six horizons while the nowcast for the
    same day had 143 red. Callers must let this propagate and leave the previous
    run's output standing.
    """


def fetch_recent_fires(client, end: dt.date,
                       lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Confirmed ignition cell-days in the trailing window, as (grid_id, date).

    Raises:
        FireHistoryUnavailable: if the record cannot be read. A genuinely empty
            result (no fires in the window — real in winter) is returned as an empty
            frame, and is not an error.
    """
    if client is None:
        raise FireHistoryUnavailable("no Supabase client — cannot build the recency prior")

    rows = []
    start = end - dt.timedelta(days=lookback_days)
    lo = start
    while lo <= end:
        hi = min(lo + dt.timedelta(days=CHUNK_DAYS - 1), end)
        for attempt in range(RETRIES):
            try:
                page = (client.table("feature_history")
                        .select("grid_id,date,label_source")
                        .eq("has_fire", 1)
                        .gte("date", lo.isoformat()).lte("date", hi.isoformat())
                        .order("grid_id")
                        .limit(CHUNK_ROW_CAP).execute().data)
                if len(page) >= CHUNK_ROW_CAP:
                    # Never let the prior be silently truncated — that understates
                    # recent fire and quietly lowers risk, the same failure this
                    # module exists to prevent.
                    raise FireHistoryUnavailable(
                        f"fire history {lo}..{hi} hit the {CHUNK_ROW_CAP}-row cap; "
                        f"the prior would be truncated. Lower CHUNK_DAYS.")
                rows += page
                break
            except FireHistoryUnavailable:
                raise
            except Exception as e:
                if attempt == RETRIES - 1:
                    raise FireHistoryUnavailable(
                        f"could not read fire history {lo}..{hi}: {e}") from e
                time.sleep(2 * (attempt + 1))
        lo = hi + dt.timedelta(days=1)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("no confirmed fires in the last %d days — legitimate in winter, "
                       "but check the label job if this is fire season", lookback_days)
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
                     "has burned recently. Legitimate in deep winter; in fire season it "
                     "means the label job has stopped. Check label_health.")
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
