"""Close the prediction -> outcome loop: backfill has_fire from fused fire sources.

For each recent prediction day, determine where fires actually occurred and set
feature_history.has_fire, fusing three sources by their strengths:

  * IRWIN / WFIGS  — interagency confirmed incidents (discovery date, location,
                     cause, size). FPA-FOD-aligned ground truth; primary positive.
  * CAL FIRE       — California official incidents; CA-specific confirmation.
  * NASA FIRMS     — satellite active-fire detections; supplementary recall
                     (catches small/unreported fires the agencies don't log).

A cell is has_fire=1 if a confirmed incident started there OR FIRMS detected fire.
label_source records which source(s) confirmed it ('irwin', 'calfire', 'firms',
or '+'-joined) so retraining can prefer high-fidelity labels. Re-labels a trailing
window each run to absorb late-arriving incidents/detections. FIRMS detections are
also archived to active_fires.

Failure contract: a source that cannot be read raises :class:`SourceUnavailable`
rather than returning an empty result, and no date is relabelled unless all three
sources answered for it. This is not defensive padding — the trailing re-label
window means a silently-empty source overwrites *known-good* labels with zeros, and
it did: IRWIN and FIRMS both failed from CI from 2026-07-04, erasing ~85% of that
month's positives while every run reported success.

Reporting contract: the run's exit status tracks *data loss*, not this run's luck.
FIRMS times out intermittently from GitHub Actions, so a failed date is normal and
the trailing window heals it on a later run — failing loudly every time trains the
alarm to be ignored, which is exactly how the corruption above went unnoticed for a
month. The run therefore stays green while affected dates are still inside the
window, and fails only when a date is about to leave it unlabelled, which no later
run can undo. A date that has not been scored yet is reported as unlabelled rather
than as a success; it previously recorded healthy=true after updating zero rows.

Run:  python -m src.pipeline.backfill_labels                 # last 10 days
      python -m src.pipeline.backfill_labels --days 14
      python -m src.pipeline.backfill_labels --date 2026-06-18
      python -m src.pipeline.backfill_labels --days 30 --end 2026-08-01
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import logging
import urllib.parse
from functools import lru_cache

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data_acquisition.config import NASA_FIRMS_MAP_KEY, REFERENCE_DIR, REGIONS
from src.pipeline.supabase_io import get_client

logger = logging.getLogger(__name__)

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
FIRMS_SOURCES = ["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT"]
IRWIN_URL = ("https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
             "WFIGS_Incident_Locations_YearToDate/FeatureServer/0/query")
CALFIRE_URL = "https://incidents.fire.ca.gov/umbraco/api/IncidentApi/List"
CELL_DEG = 0.1
HALF = CELL_DEG / 2

# Both remote sources answer failures with a 200 and a human-readable body rather
# than an error status, and both were observed returning nothing at all from GitHub
# Actions while working fine locally. A statewide year-to-date query can never
# legitimately be empty, so a count below these floors is an outage, not a quiet day.
MIN_IRWIN_YTD = 1
MIN_CALFIRE_YTD = 1
FIRMS_TIMEOUT = 30             # per request; the retry budget multiplies this
USER_AGENT = ("FireProject/1.0 (wildfire risk research; "
              "+https://california-firepred.vercel.app)")

# A date can only be labelled once scoring has written its feature_history rows.
# Scoring and labelling are independent crons, and gridMET's ~1-day lag means the
# day the labeller wants is often not scored until hours later — on 2026-08-04 the
# labeller "labelled" 2026-08-03 six hours before it was scored, updating zero rows
# and recording healthy=true. Routine, but it must never read as success. The
# fraction (rather than >0) also rejects a partially-written day if scoring is
# still in flight.
MIN_SCORED_FRACTION = 0.9

# A date drops out of the trailing re-label window after this many days; once it
# does, an unlabelled day is unlabellable forever. The margin escalates one run
# early, so the alarm fires while there is still a run left to fix it.
AGE_OUT_MARGIN = 1

# Wider than the 7 days this used to re-label, and deliberately wider than
# score_daily's catch-up window: a day the scorer recovers late must still be inside
# the label window when it lands, or it arrives scored and permanently unlabellable.
LABEL_WINDOW_DAYS = 10

# Labelling normally trails scoring by about a day (gridMET's lag means the newest
# scored day is often scored after the labeller last ran). A gap wider than this is
# not the usual offset — it means labelling has stopped keeping up, and waiting for
# the age-out alarm would sit silent for over a week while a permanent breakage (a
# revoked key, a moved endpoint) looks exactly like a run of bad luck.
MAX_LABEL_LAG_DAYS = 3


class SourceUnavailable(RuntimeError):
    """A ground-truth source failed.

    Raised instead of returning an empty frame: an empty frame is indistinguishable
    from "no fires today", and writing labels from it silently marks a real fire day
    as quiet. Callers must leave existing labels untouched when this is raised.
    """


def _force_ipv4() -> None:
    """Make outbound HTTPS resolve A records only.

    GitHub Actions runners advertise IPv6 but frequently have no working IPv6 egress,
    which surfaces as ``[Errno 101] Network is unreachable`` the moment a host publishes
    an AAAA record — exactly what FIRMS does. Locally, where IPv6 works or is absent,
    this is a no-op.
    """
    import socket
    import urllib3.util.connection as u3
    u3.allowed_gai_family = lambda: socket.AF_INET


@lru_cache(maxsize=1)
def _session() -> requests.Session:
    """Shared session with a bounded retry budget and a real User-Agent.

    The retry budget is deliberately small. An earlier version used 4 retries with a
    1.5s backoff factor and a 120s timeout, which turned an unreachable host into ~8
    minutes per satellite per date and blew through the workflow's 20-minute cap after
    two days' work. Connection failures in particular get a single retry: a network
    that is unreachable now will still be unreachable three seconds from now.
    """
    _force_ipv4()
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    retry = Retry(total=2, connect=1, backoff_factor=1.0,
                  status_forcelist=(408, 429, 500, 502, 503, 504),
                  allowed_methods=frozenset(["GET"]), raise_on_status=False)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def _grid() -> pd.DataFrame:
    """Canonical grid from the small committed reference file (no heavy deps)."""
    return pd.read_json(REFERENCE_DIR / "grid_cells.json")


# --------------------------------------------------------------------------- #
# Source fetchers — each returns a DataFrame with latitude, longitude, date
# --------------------------------------------------------------------------- #

def _parse_firms_csv(text: str, src: str, ds: str) -> pd.DataFrame:
    """Parse a FIRMS CSV body, rejecting error/notice text masquerading as data.

    FIRMS reports an invalid key or exhausted quota with HTTP 200 and a plain-text
    body. ``pd.read_csv`` turns that into a junk one-column frame, which the old
    column check discarded without a word — the silent path that zeroed a month of
    labels. A header-only body is left alone: that is a genuine no-detection day.
    """
    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as e:
        raise SourceUnavailable(f"FIRMS {src} {ds}: unparseable body: {text[:200]!r}") from e
    if not {"latitude", "longitude"} <= set(df.columns):
        raise SourceUnavailable(f"FIRMS {src} {ds}: unexpected body: {text[:200]!r}")
    return df


def fetch_firms_for_date(date: dt.date, map_key: str | None = None) -> pd.DataFrame:
    """CA FIRMS detections for one date, across satellites (+ raw cols for archive).

    Raises:
        SourceUnavailable: if any satellite feed fails. A partial read understates
            fire activity, so it is rejected outright rather than merged.
    """
    map_key = map_key or NASA_FIRMS_MAP_KEY
    if not map_key:
        raise SourceUnavailable("NASA_FIRMS_MAP_KEY is not set")
    b = REGIONS["california"]
    bbox = f"{b['west']},{b['south']},{b['east']},{b['north']}"
    ds = date.strftime("%Y-%m-%d")
    frames = []
    for src in FIRMS_SOURCES:
        try:
            r = _session().get(f"{FIRMS_BASE}/{map_key}/{src}/{bbox}/1/{ds}", timeout=FIRMS_TIMEOUT)
            r.raise_for_status()
        except SourceUnavailable:
            raise
        except Exception as e:
            raise SourceUnavailable(f"FIRMS {src} {ds} request failed: {e}") from e
        df = _parse_firms_csv(r.text, src, ds)
        if "satellite" not in df.columns:
            df["satellite"] = src
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fetch_irwin_ca() -> pd.DataFrame:
    """All CA interagency (IRWIN/WFIGS) incidents year-to-date: lat, lon, discovery date.

    Raises:
        SourceUnavailable: on transport failure, an ArcGIS error payload, or a result
            below :data:`MIN_IRWIN_YTD`. ArcGIS signals throttling with a 200 and an
            ``error`` object, which the previous code read as "no incidents".
    """
    rows, offset = [], 0
    while True:
        q = {"where": "POOState='US-CA'",
             "outFields": "FireDiscoveryDateTime,InitialLatitude,InitialLongitude",
             "resultOffset": offset, "resultRecordCount": 2000, "f": "json"}
        try:
            r = _session().get(f"{IRWIN_URL}?{urllib.parse.urlencode(q)}", timeout=90)
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            raise SourceUnavailable(f"IRWIN request failed at offset {offset}: {e}") from e
        if isinstance(d, dict) and d.get("error"):
            raise SourceUnavailable(f"IRWIN error payload: {str(d['error'])[:300]}")
        feats = d.get("features", [])
        for f in feats:
            a = f["attributes"]
            ts, la, lo = a.get("FireDiscoveryDateTime"), a.get("InitialLatitude"), a.get("InitialLongitude")
            if ts and la and lo:
                rows.append({"latitude": la, "longitude": lo,
                             "date": dt.datetime.fromtimestamp(ts / 1000, dt.UTC).date()})
        if len(feats) < 2000:
            break
        offset += 2000
    df = pd.DataFrame(rows)
    if len(df) < MIN_IRWIN_YTD:
        raise SourceUnavailable(
            f"IRWIN returned {len(df)} CA incidents YTD (floor {MIN_IRWIN_YTD}) — treating as an outage")
    logger.info("IRWIN: %d CA incidents YTD", len(df))
    return df


def fetch_calfire_ca() -> pd.DataFrame:
    """CA official incidents (CAL FIRE) for the year: lat, lon, start date.

    Raises:
        SourceUnavailable: on transport failure or a result below
        :data:`MIN_CALFIRE_YTD`.
    """
    try:
        r = _session().get(f"{CALFIRE_URL}?inactive=true&year={dt.date.today().year}", timeout=90)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise SourceUnavailable(f"CAL FIRE request failed: {e}") from e
    rows = data if isinstance(data, list) else data.get("Incidents", [])
    out = []
    for r_ in rows:
        la, lo, st = r_.get("Latitude"), r_.get("Longitude"), r_.get("Started")
        if la and lo and st:
            try:
                out.append({"latitude": float(la), "longitude": float(lo),
                            "date": pd.to_datetime(st).date()})
            except Exception:
                pass
    df = pd.DataFrame(out)
    if len(df) < MIN_CALFIRE_YTD:
        raise SourceUnavailable(
            f"CAL FIRE returned {len(df)} CA incidents (floor {MIN_CALFIRE_YTD}) — treating as an outage")
    logger.info("CAL FIRE: %d CA incidents this year", len(df))
    return df


# --------------------------------------------------------------------------- #
# Mapping + persistence
# --------------------------------------------------------------------------- #

def points_to_grid_ids(pts: pd.DataFrame, grid: pd.DataFrame) -> set[int]:
    """Map (latitude, longitude) points to grid cells via the training flooring rule."""
    if pts is None or pts.empty:
        return set()
    lookup = {(round(r.lat_center, 2), round(r.lon_center, 2)): int(r.grid_id) for r in grid.itertuples()}
    ids: set[int] = set()
    for lat, lon in zip(pts["latitude"].to_numpy(), pts["longitude"].to_numpy()):
        clat = round(np.floor(lat / CELL_DEG) * CELL_DEG + HALF, 2)
        clon = round(np.floor(lon / CELL_DEG) * CELL_DEG + HALF, 2)
        gid = lookup.get((clat, clon))
        if gid is not None:
            ids.add(gid)
    return ids


def _archive_fires(client, det: pd.DataFrame, ds: str) -> None:
    client.table("active_fires").delete().eq("acq_date", ds).execute()
    if det.empty:
        return
    rows = [{
        "latitude": float(r.latitude), "longitude": float(r.longitude),
        "frp": float(getattr(r, "frp")) if pd.notna(getattr(r, "frp", None)) else None,
        "confidence": str(getattr(r, "confidence", "")) or None, "acq_date": ds,
        "acq_time": str(getattr(r, "acq_time", "")) or None,
        "satellite": str(getattr(r, "satellite", "")) or None,
    } for r in det.itertuples()]
    for i in range(0, len(rows), 1000):
        client.table("active_fires").insert(rows[i:i + 1000]).execute()


def _record_health(client, result: dict) -> dict:
    """Upsert one date's source-health record; no-op if migration 0005 hasn't run.

    Written for every date, healthy or not, so the retrain gates can tell "no fires
    that day" apart from "we could not see that day".
    """
    s = result.get("sources", {})
    row = {"date": result["date"], "healthy": bool(result.get("healthy")),
           "irwin_ok": bool(s.get("irwin")), "calfire_ok": bool(s.get("calfire")),
           "firms_ok": bool(s.get("firms")), "fire_cells": result.get("fire_cells"),
           "confirmed_cells": result.get("confirmed"),
           "firms_only_cells": result.get("firms_only"), "error": result.get("error")}
    try:
        client.table("label_health").upsert(row, on_conflict="date").execute()
    except Exception as e:
        msg = str(e)
        hint = " — run migration 0005" if "does not exist" in msg else ""
        logger.warning("label_health write skipped for %s: %s%s", result["date"], msg[:150], hint)
    return result


def _row_counts(client, ds: str) -> tuple[int, int]:
    """(rows scored, rows carrying a label) in feature_history for one date.

    Server-side exact counts, not client-side pagination: PostgREST ``.range()``
    without an explicit order returns inconsistent pages over a table this size.
    """
    q = client.table("feature_history").select("grid_id", count="exact").eq("date", ds)
    scored = q.limit(1).execute().count or 0
    labelled = (client.table("feature_history").select("grid_id", count="exact")
                .eq("date", ds).not_.is_("has_fire", "null").limit(1).execute().count or 0)
    return scored, labelled


def _has_label_source(client) -> bool:
    try:
        client.table("feature_history").select("label_source").limit(1).execute()
        return True
    except Exception:
        logger.info("label_source column absent — run migration 0002 to enable source tracking")
        return False


def backfill_date(client, grid, date, irwin, calfire, with_source: bool) -> dict:
    """Label one date's feature_history rows from fused sources; archive FIRMS.

    FIRMS is fetched per-date, so a FIRMS outage aborts *this date only*. The day's
    existing labels are left exactly as they were: a stale label is recoverable on a
    later run, whereas resetting to zero destroys ground truth permanently and looks
    identical to a genuinely quiet day.

    Returns a result carrying ``status``, one of:
      ``labelled``       — all three sources answered and the day was written;
      ``unscored``       — scoring has not produced this day's rows yet, so there is
                           nothing to label (no fault, but not a success either);
      ``source_outage``  — a source could not be read; existing labels left intact.
    """
    ds = date.strftime("%Y-%m-%d")

    # Checked before FIRMS so an unscored day costs nothing against a host that has
    # been intermittently timing out from CI.
    scored, _ = _row_counts(client, ds)
    if scored < MIN_SCORED_FRACTION * len(grid):
        logger.warning("%s: NOT LABELLED — only %d of %d cells scored; nothing to label yet",
                       ds, scored, len(grid))
        return _record_health(client, {
            "date": ds, "status": "unscored", "healthy": False,
            "error": f"not scored yet ({scored}/{len(grid)} cells present)",
            "sources": {"irwin": True, "calfire": True, "firms": False}})

    try:
        firms = fetch_firms_for_date(date)
    except SourceUnavailable as e:
        logger.error("%s: SKIPPED — FIRMS unavailable (%s); existing labels left untouched", ds, e)
        return _record_health(client, {
            "date": ds, "status": "source_outage", "healthy": False, "error": str(e)[:300],
            "sources": {"irwin": True, "calfire": True, "firms": False}})

    irwin_ids = points_to_grid_ids(irwin[irwin["date"] == date] if not irwin.empty else irwin, grid)
    calfire_ids = points_to_grid_ids(calfire[calfire["date"] == date] if not calfire.empty else calfire, grid)
    firms_ids = points_to_grid_ids(firms, grid)
    all_ids = irwin_ids | calfire_ids | firms_ids

    # Safe to reset now: all three sources answered for this date.
    client.table("feature_history").update(
        {"has_fire": 0, **({"label_source": None} if with_source else {})}).eq("date", ds).execute()

    by_source: dict[str, list[int]] = {}
    for gid in all_ids:
        src = "+".join(s for s, ids in
                       (("irwin", irwin_ids), ("calfire", calfire_ids), ("firms", firms_ids)) if gid in ids)
        by_source.setdefault(src, []).append(gid)
    for src, ids in by_source.items():
        payload = {"has_fire": 1, **({"label_source": src} if with_source else {})}
        for i in range(0, len(ids), 500):
            client.table("feature_history").update(payload).eq("date", ds).in_("grid_id", ids[i:i + 500]).execute()

    _archive_fires(client, firms, ds)
    confirmed = len(irwin_ids | calfire_ids)
    logger.info("%s: fire cells=%d (confirmed=%d, firms-only=%d)",
                ds, len(all_ids), confirmed, len(all_ids - irwin_ids - calfire_ids))
    return _record_health(client, {
        "date": ds, "status": "labelled", "healthy": True,
        "sources": {"irwin": True, "calfire": True, "firms": True},
        "fire_cells": len(all_ids), "confirmed": confirmed,
        "irwin_cells": len(irwin_ids), "calfire_cells": len(calfire_ids),
        "firms_cells": len(firms_ids), "firms_only": len(all_ids - irwin_ids - calfire_ids)})


def backfill_range(days: int = LABEL_WINDOW_DAYS, end: dt.date | None = None, client=None) -> list[dict]:
    """Label a trailing window, aborting before any write if a YTD source is down."""
    client = client if client is not None else get_client()
    if client is None:
        logger.warning("Supabase not configured — nothing to backfill")
        return []
    grid = _grid()
    # Fetched once per run and *before* any write: if either feed is down, every date
    # in the window would be mislabelled, so raise rather than corrupt the window.
    irwin, calfire = fetch_irwin_ca(), fetch_calfire_ca()
    ws = _has_label_source(client)
    end = end or dt.date.today()

    # Stop at the first FIRMS *outage* instead of retrying it once per date. An
    # unreachable host is unreachable for the whole window, and grinding through every
    # date against it is what previously consumed the workflow's entire time budget.
    # An unscored date is not an outage and must not stop the walk: the newest days
    # are routinely unscored while older ones still need labelling.
    results = []
    for d in range(1, days + 1):
        r = backfill_date(client, grid, end - dt.timedelta(days=d), irwin, calfire, ws)
        results.append(r)
        if r.get("status") == "source_outage":
            remaining = days - d
            if remaining:
                logger.error("FIRMS unreachable — abandoning the remaining %d date(s) rather "
                             "than retrying a dead host; labels are left intact", remaining)
            break
    return results


def critical_dates(client, grid, days: int, end: dt.date) -> list[str]:
    """Dates that become permanently unlabellable unless the next run fixes them.

    The trailing window gives every date several attempts, so a single failed run is
    routine and self-healing — alarming on it trains the alarm to be ignored, which is
    how a month of label corruption went unnoticed. Only the tail of the window is
    urgent: past it, the date is never revisited and its ground truth is lost for good.
    """
    urgent = []
    for d in range(max(1, days - AGE_OUT_MARGIN), days + 1):
        date = end - dt.timedelta(days=d)
        ds = date.strftime("%Y-%m-%d")
        scored, labelled = _row_counts(client, ds)
        if scored < MIN_SCORED_FRACTION * len(grid):
            urgent.append(f"{ds}: never scored ({scored}/{len(grid)} cells) and leaving the "
                          f"{days}-day label window — this day's ground truth will be lost")
        elif labelled < scored:
            urgent.append(f"{ds}: scored ({scored} cells) but only {labelled} labelled, and "
                          f"leaving the {days}-day label window — ground truth will be lost")
    return urgent


def label_lag(client, grid, days: int, end: dt.date) -> str | None:
    """Complain if labelling has fallen too far behind scoring.

    The age-out alarm alone is not enough: it only fires at the far end of the
    window, so a source that breaks for good — a revoked key, a moved endpoint —
    would look identical to a run of bad luck for over a week before anything said
    so. This catches that within a few days, while still tolerating the routine
    one-day offset between the two jobs.
    """
    newest_scored = newest_labelled = None
    for d in range(1, days + 1):
        ds = (end - dt.timedelta(days=d)).strftime("%Y-%m-%d")
        scored, labelled = _row_counts(client, ds)
        if scored >= MIN_SCORED_FRACTION * len(grid):
            newest_scored = newest_scored or ds
            if labelled >= scored:
                newest_labelled = ds
                break          # walking backwards, so the first hit is the newest
    if newest_scored is None:
        return None            # nothing scored in the window; a scoring problem, not ours
    if newest_labelled is None:
        return (f"no labelled day anywhere in the last {days} days (newest scored is "
                f"{newest_scored}) — labelling has stopped, not just stumbled")
    lag = (dt.date.fromisoformat(newest_scored) - dt.date.fromisoformat(newest_labelled)).days
    if lag > MAX_LABEL_LAG_DAYS:
        return (f"labels are {lag} days behind scoring (newest scored {newest_scored}, "
                f"newest labelled {newest_labelled}, tolerance {MAX_LABEL_LAG_DAYS}) — "
                f"treating as a persistent failure, not a transient outage")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=LABEL_WINDOW_DAYS,
                    help="trailing days to (re)label")
    ap.add_argument("--date", help="label a single YYYY-MM-DD date instead")
    ap.add_argument("--end", help="end the trailing window at YYYY-MM-DD (default: today)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    client = get_client()
    if client is None:
        print("Supabase not configured")
        return
    grid = _grid()
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()

    try:
        if args.date:
            d = dt.date.fromisoformat(args.date)
            results = [backfill_date(client, grid, d, fetch_irwin_ca(), fetch_calfire_ca(),
                                     _has_label_source(client))]
        else:
            results = backfill_range(days=args.days, end=end, client=client)
    except SourceUnavailable as e:
        # A year-to-date feed being down aborts the whole window before any write, so
        # nothing was even attempted. Always loud.
        logger.error("ABORTED before any write — %s", e)
        raise SystemExit(1)

    for r in results:
        print(r)

    outages = [r["date"] for r in results if r.get("status") == "source_outage"]
    unscored = [r["date"] for r in results if r.get("status") == "unscored"]
    if unscored:
        logger.info("%d date(s) not scored yet, nothing to label: %s",
                    len(unscored), ", ".join(unscored))
    if outages:
        logger.warning("%d date(s) skipped on a source outage, labels left intact: %s",
                       len(outages), ", ".join(outages))

    # Exit status is about *data loss*, not about whether this particular run had a
    # clean path. A transient outage inside the self-heal window is expected and stays
    # green; a date about to age out unlabelled is the one thing no later run can undo.
    if args.date:
        if outages:
            raise SystemExit(1)
        return
    urgent = critical_dates(client, grid, args.days, end)
    lag = label_lag(client, grid, args.days, end)
    if lag:
        urgent.append(lag)
    if urgent:
        for msg in urgent:
            logger.error("AT RISK — %s", msg)
        raise SystemExit(1)
    if outages:
        logger.warning("Not failing the run: every affected date is still inside the "
                       "%d-day self-heal window and will be retried.", args.days)


if __name__ == "__main__":
    main()
