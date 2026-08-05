"""Operational reading of the ablation: lead time, a named-fire tracer, and CIs.

PR-AUC is a ranking summary; it is not what the dashboard promises. What a user of the
map actually gets is: *was this cell flagged before it burned, and how far before?*
``CLAUDE.md`` lists "lead time before fire detection" as an operational metric, and no
prior analysis in this project has computed it. It is the measure that most directly
distinguishes a forecast from a fire-persistence map, so it is the one most worth having
when deciding which feature blocks to keep.

Three things here:

  1. **Lead time.** For every ignition, was its cell flagged on the day itself, or on
     any day in the preceding window? Reported separately for quiet and active cells,
     at matched coverage, so a model cannot buy lead time by flagging more of the state.
  2. **Named-fire tracer.** The day-by-day tier history of specific cells, including
     5917 — the Gann Fire (2026-08-04, Calaveras, 9,363 ac), the case that exposed the
     regression. An aggregate that disagrees with the fire a human went and looked at
     is an aggregate to distrust.
  3. **Day-level bootstrap** on the handful of comparisons the conclusions rest on.
     Only those: 31 combinations x several metrics x several slices would otherwise
     manufacture significant-looking differences by sheer multiplicity.

Run:  .venv/bin/python -m scripts.analyze_ablation_operational
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.data_acquisition.config import PROJECT_ROOT
from src.models.slice_eval import QUIET_EPS, coverage_tiers, day_block_bootstrap, quiet_mask
from scripts.analyze_feature_ablation import RED_COV, YELLOW_COV, load_panel

logger = logging.getLogger(__name__)
OUT_DIR = PROJECT_ROOT / "data" / "eval" / "ablation"

LEAD_WINDOWS = (0, 3, 7)
GANN_CELL = 5917          # Calaveras County; the Gann Fire ignited here 2026-08-04

# The comparisons the conclusions actually rest on. Kept deliberately short.
HEADLINE = [("WGCLR", "WGCL", "recency block: does it help?"),
            ("WGCLR", "G", "full model vs static geography alone"),
            ("WGCLR", "GCLR", "weather block: does it help?"),
            ("WGCL", "G", "pre-recency model vs geography alone")]


def lead_time(df: pd.DataFrame, tiers: np.ndarray, mask: np.ndarray) -> dict:
    """Was each ignition's cell flagged on the day, or in the days before it?

    Looks back from the ignition day, so lead 0 is the nowcast (identical to tier
    recall) and lead 7 asks whether the cell was flagged at any point in the week
    leading up — the question a fire manager is actually asking of the map.
    """
    d = df[["grid_id", "date"]].copy()
    d["tier"] = tiers
    d["fire"] = df["confirmed"].to_numpy()
    d["q"] = mask
    d["red"] = (d.tier == "red").astype(np.int8)
    d["amber"] = (d.tier != "green").astype(np.int8)

    ti, dates = pd.factorize(d["date"], sort=True)
    ci, cells = pd.factorize(d["grid_id"], sort=True)
    d["ti"], d["ci"] = ti, ci
    assert (ti >= 0).all() and (ci >= 0).all(), "unindexable date or cell"

    red = np.zeros((len(cells), len(dates)), np.int8)
    amb = np.zeros_like(red)
    red[d.ci, d.ti] = d.red.to_numpy()
    amb[d.ci, d.ti] = d.amber.to_numpy()

    out = {}
    for slab, sel in (("quiet", d.q.to_numpy()), ("active", ~d.q.to_numpy())):
        f = d[(d.fire == 1) & sel]
        out[f"n_fires_{slab}"] = len(f)
        if not len(f):
            continue
        cc, tt = f.ci.to_numpy(), f.ti.to_numpy()
        for w in LEAD_WINDOWS:
            lo = np.maximum(tt - w, 0)
            hit_r = np.array([red[c, a:b + 1].any() for c, a, b in zip(cc, lo, tt)])
            hit_a = np.array([amb[c, a:b + 1].any() for c, a, b in zip(cc, lo, tt)])
            out[f"red_lead{w}_{slab}"] = float(hit_r.mean())
            out[f"amber_lead{w}_{slab}"] = float(hit_a.mean())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    warm_ref, _ = load_panel("live2026")

    rows = []
    for panel in ("live2026", "autumn2025"):
        df, q = load_panel(panel)
        for combo in ("WGCLR", "WGCL", "G", "GCLR", "WG", "GR"):
            tiers = coverage_tiers(df[f"s_{combo}"].to_numpy(), RED_COV, YELLOW_COV)
            rows.append({"panel": panel, "combo": combo, **lead_time(df, tiers, q)})

    lt = pd.DataFrame(rows)
    lt.to_csv(OUT_DIR / "lead_time.csv", index=False)

    for panel in ("live2026", "autumn2025"):
        sub = lt[lt.panel == panel]
        print("\n" + "=" * 92)
        print(f"LEAD TIME — {panel}, matched coverage (red {RED_COV:.1%} of cell-days)")
        print("  'red L7' = flagged Red on the ignition day or any of the 7 days before")
        print("=" * 92)
        print(f"{'combo':7s}{'QUIET: red L0':>14s}{'red L3':>9s}{'red L7':>9s}"
              f"{'amb L7':>9s}{'| ACTIVE: red L0':>18s}{'red L7':>9s}{'amb L7':>9s}")
        print("-" * 92)
        for _, r in sub.iterrows():
            print(f"{r.combo:7s}{r.red_lead0_quiet:>14.1%}{r.red_lead3_quiet:>9.1%}"
                  f"{r.red_lead7_quiet:>9.1%}{r.amber_lead7_quiet:>9.1%}"
                  f"{r.red_lead0_active:>18.1%}{r.red_lead7_active:>9.1%}"
                  f"{r.amber_lead7_active:>9.1%}")

    # --- named-fire tracer -------------------------------------------------------
    df, q = load_panel("live2026")
    print("\n" + "=" * 92)
    print(f"TRACER — cell {GANN_CELL} (Gann Fire, ignited 2026-08-04, just past the panel's end)")
    print("  tier on each of the last 12 scored days, at matched coverage")
    print("=" * 92)
    tr = {}
    for combo in ("WGCLR", "WGCL", "G", "GR"):
        t = coverage_tiers(df[f"s_{combo}"].to_numpy(), RED_COV, YELLOW_COV)
        sub = df.assign(tier=t)
        sub = sub[sub.grid_id == GANN_CELL].sort_values("date").tail(12)
        tr[combo] = sub.set_index(sub.date.dt.strftime("%m-%d"))["tier"]
    tracer = pd.DataFrame(tr)
    print(tracer.to_string())

    # --- bootstrap on the headline comparisons only ------------------------------
    boot = []
    for panel in ("live2026", "autumn2025"):
        d, qq = load_panel(panel)
        y, dates = d["confirmed"].to_numpy(), d["date"].to_numpy()
        for a, b, why in HEADLINE:
            for slab, m in (("all", np.ones(len(d), bool)), ("quiet", qq)):
                r = day_block_bootstrap(dates[m], y[m], d[f"s_{a}"].to_numpy()[m],
                                        d[f"s_{b}"].to_numpy()[m], n=400)
                boot.append({"panel": panel, "cmp": f"{a} - {b}", "slice": slab,
                             "why": why, **r})
    bt = pd.DataFrame(boot)
    bt.to_csv(OUT_DIR / "bootstrap_headline.csv", index=False)
    print("\n" + "=" * 92)
    print("DAY-LEVEL BOOTSTRAP — PR-AUC difference, 400 resamples of whole days")
    print("  significant only when the interval excludes zero")
    print("=" * 92)
    print(f"{'panel':12s}{'comparison':16s}{'slice':7s}{'delta':>9s}{'95% CI':>22s}  verdict")
    print("-" * 92)
    for _, r in bt.iterrows():
        sig = "SIGNIFICANT" if (r.lo > 0 or r.hi < 0) else "tie"
        print(f"{r.panel:12s}{r['cmp']:16s}{r['slice']:7s}{r['median']:>9.4f}"
              f"  [{r.lo:+.4f}, {r.hi:+.4f}]  {sig}")

    json.dump({"lead_time": lt.to_dict("records"), "bootstrap": bt.to_dict("records"),
               "tracer": tracer.to_dict()}, open(OUT_DIR / "operational.json", "w"),
              indent=2, default=str)
    logger.info("wrote lead_time.csv, bootstrap_headline.csv, operational.json")


if __name__ == "__main__":
    main()
