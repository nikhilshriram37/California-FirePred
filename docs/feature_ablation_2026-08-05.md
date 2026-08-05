# Feature-block ablation and slice-based evaluation

Run 2026-08-05. Diagnostic study prompted by the quiet-cell tier collapse found after the
recency retrain (model `20260805`). Production was not modified.

## What was run

31 non-empty combinations of five feature blocks, 3 seeds each (93 fits, 18 min), all
holding the tuned hyperparameters, the 2018-19 fitting window, the calibration procedure
and the tier-derivation procedure fixed. Only the feature set varies.

| block | n | contents |
|---|---|---|
| W weather | 22 | gridMET feed + rolling derivatives + dry_streak |
| G geography | 8 | lat/lon, elevation, ruggedness, slope, northness, eastness, log_pop |
| C calendar | 3 | month, month_sin, doy_cos |
| L lightning | 1 | lightning_count |
| R recency | 3 | fire_recency_cell, fire_recency_nbr, days_since_fire_cell |

Evaluated on four panels (`data/eval/panel_*.parquet`):

| panel | window | rows | ignitions | role |
|---|---|---|---|---|
| holdout2020 | 2020 | 1,342,488 | 4,185 | era reference (the flattering one) |
| live2026 | 2026-06-19..08-03 | 184,314 | 1,613 confirmed | peak summer, the live regime |
| live2026_cold | same | 184,314 | 1,613 | recency prior as production actually had it |
| autumn2025 | 2025-09-01..11-30 | 379,379 | 2,242 | shoulder season, 8x the weather variance |

`quiet` = no ignition in the cell or its 8 neighbours within roughly 90 days
(`fire_recency_cell < 0.05` and `fire_recency_nbr < 0.05`), computed from the data so the
same split applies to every candidate. Tiers are compared at **matched coverage**
(red 5.80% of cell-days, yellow the next 22.45% — what production actually paid).

**Seed noise: PR-AUC sd 0.0004–0.0005 per combination.** The handoff assumed a ±0.02 band;
that remains right for comparing across data regimes but is ~40x too conservative within a
panel. Differences above ~0.001 here are outside fit randomness.

## Result 1 — the collapse is a tiering failure, and it is not specific to recency

Marginal contribution of each block, averaged over the 15 combination pairs that differ
only in that block, on each panel:

| block | PR-AUC pooled | PR-AUC quiet | Red share of quiet ignitions |
|---|---|---|---|
| R recency | +0.053 … +0.093 (100% of pairs +) | +0.0006 … +0.0020 | **−0.112 … −0.141 (0% of pairs +)** |
| G geography | +0.002 … +0.038 | **+0.0021 … +0.0043 (100% +)** | mixed (+0.056 on 2020) |
| W weather | −0.005 (summer) … +0.010 | +0.0011 … +0.0013 | −0.028 … −0.060 |
| C calendar | −0.002 … +0.007 | ~+0.001 | −0.047 … −0.065 |
| L lightning | −0.0003 … −0.0001 (~50% of pairs) | ~0 | ~0 |

Recency **improves quiet-cell ranking and degrades quiet-cell tiering** — on all four
panels, in all 60 matched pairs, without exception. Weather and calendar do the same thing
more weakly. **This is a property of a single global cutoff serving two regimes, not a
defect of the recency block.** Recency is simply the strongest active-cell signal, so it
reallocates the most.

Day-level bootstrap confirms the ranking half: recency's quiet-cell PR-AUC gain is
+0.0019 [+0.0004, +0.0042] on live summer and +0.0060 [+0.0006, +0.0220] in autumn.

## Result 2 — per-block verdicts

- **Lightning: drop it.** Marginal effect is a coin flip on every panel (≈50% of pairs
  positive, |Δ| ≤ 0.0003). Confirms the earlier note; 37 → 36 features.
- **Geography: the quiet-cell workhorse.** The only block positive for quiet-cell PR-AUC in
  100% of pairs on all four panels.
- **Weather: the "when" carrier, seasonally.** Marginal detrended temporal correlation
  +0.3212 in autumn (100% of pairs) and +0.1997 on 2020, but only +0.0444 in live summer,
  where it *costs* pooled PR-AUC (−0.0046, 27% of pairs). Keep it — the product runs
  year-round and autumn is when day-to-day weather signal exists.
- **On live summer only, the whole 34-feature pre-recency model is statistically
  indistinguishable from the 8 geography features alone** (ΔPR −0.0000 [−0.0012, +0.0016]).
  In autumn the same comparison is a real win for the full set (+0.0058 [+0.0034, +0.0095]).

## Result 3 — two independent causes, separated

Production's fire history begins 2026-06-19, so at launch `days_since_fire_cell` sat at its
365-day cap for **88.7%** of cell-days while 4,298 real CA ignitions since January were
invisible to the model.

Sliced by the cold prior (the definition the original regression was recorded against —
reproduced here exactly: 77.6% of rows, 641 of 1,613 ignitions):

| configuration | Red | Yellow | Green |
|---|---|---|---|
| old + corrected features | 15.1% | 49.3% | 35.6% |
| new + COLD prior (what shipped) | 0.2% | 1.2% | 98.6% |
| new + WARM prior | 20.7% | 31.5% | 47.7% |

On genuinely quiet cells (193 ignitions) at matched coverage the regression is still real
— 5.2% Red under the old model, 0.5% under the new — and warming the prior does not fix
that half. **Both causes are real and independent.**

## Result 4 — the Gann Fire, traced

Cell 5917 (Calaveras, ignited 2026-08-04, one day past the panel's end, so every tier below
is a genuine forecast). Its own last ignition was 107 days earlier: genuinely quiet.

| configuration | tier over the 10 days before ignition |
|---|---|
| old 34f, own cutoffs | Yellow (intermittent) |
| new 37f, own cutoffs, COLD prior — **what shipped** | **Green throughout** |
| new 37f, own cutoffs, WARM prior | Green (Yellow only on 08-03) |
| new 37f, matched coverage | Yellow throughout |
| new 37f, hybrid tiering | **Yellow throughout** |

For this cell the dominant cause of Green was the **re-derived cutoffs**, not the cold
prior: removing the threshold change alone restores Yellow.

## Result 5 — remedies

Tested at identical coverage on both seasons.

- **Dropping `days_since_fire_cell`: no effect.** Quiet R+Y 20.2% vs 19.7% (live). The
  hypothesis that this feature causes the collapse is **refuted**.
- **Monotone constraints on recency: mildly harmful.** Improves pooled recall slightly
  (44.3% → 45.4%) and *reduces* quiet R+Y (19.7% → 16.1%). Not the fix.
- **Full stratified tiering** (cutoffs within regime) works but changes what Red means: a
  stratified Red in a quiet area carries a 0.86% observed fire rate against 10.8% in an
  active one — the same colour for a 12.6x difference in risk. It also costs 8.5 points of
  overall red recall.
- **Hybrid tiering — recommended.** Red stays global (unchanged statewide meaning and
  unchanged recall); Yellow is computed within regime.

| rule (live2026, coverage identical) | red recall | Red % quiet ign | R+Y % quiet ign |
|---|---|---|---|
| old model, global | 28.4% [25.8, 31.3] | 5.2% | 40.9% [31.7, 50.3] |
| new model, global (deployed) | 44.1% [41.0, 47.1] | 0.5% | 19.7% [12.8, 25.5] |
| new model, fully stratified | 35.5% [33.2, 38.1] | 25.9% | 65.8% [58.1, 74.4] |
| **new model, hybrid** | **44.1% [41.0, 47.1]** | 0.5% | **65.8% [58.1, 74.4]** |

Hybrid minus deployed, paired day-level bootstrap: overall red recall **+0.0% (exactly
unchanged, by construction)**; R+Y share of quiet ignitions **+46.4 pts [+39.2, +54.4]**
on live summer and **+41.5 pts [+30.6, +53.1]** in autumn. Both significant.

It beats the *pre-recency* model on both axes simultaneously: 44.1% vs 28.4% overall red
recall, and 65.8% vs 40.9% quiet-area warning coverage.

## What shipped, and how it differs from the measurement above

Deployed 2026-08-05: `lightning_count` dropped (36 features, 2020 PR-AUC 0.1096 → 0.1107)
and the Yellow cutoff made per-regime while Red stays global.

**The numbers above were measured at fixed coverage; production sets tiers from absolute
thresholds derived on the 2020 calibration year, and those do not transfer to 2026.** The
same idea therefore behaves differently in production than in the study, and the honest
figures for what actually shipped are:

| | live summer | autumn 2025 |
|---|---|---|
| quiet-area R+Y recall | 8.8% → **33.2%** | 2.3% → **14.3%** |
| overall R+Y recall | 73.5% → 62.9% | 65.5% → 53.2% |
| share of state flagged | 20.7% → 16.6% | 10.9% → 6.1% |
| Red tier | unchanged | unchanged |

So the deployed change flags *less* of the state, catches substantially more quiet-area
ignitions, leaves Red untouched, and costs about 11 points of overall red+yellow recall
against the 80% design target. It moves the Gann Fire cell from Green to Yellow for the
nine days before ignition. Realising the full measured gain (+46 pts at no cost) would
require moving from 2020-derived absolute thresholds to coverage-based ones — a change to
how thresholds are set, deliberately deferred to a separate decision.

Two earlier derivations of the per-regime cutoff were tried and rejected, recorded in
`train.py` so they are not retried: targeting 80% recall within regime does not transfer
(it paints 51% of quiet cells Yellow on the live record), and taking the top slice of each
regime outright puts the active cutoff *above* Red, silently emptying that tier.

## Remaining recommendations

1. **Give scoring a real fire history.** Backfill `feature_history` or query IRWIN
   year-to-date at score time. Self-heals as the table fills, but is degrading output now.
   **Not yet done.**
2. **Revisit threshold policy** — coverage-based vs 2020-absolute. This is what limits the
   tiering fix, and it is also why the 55%/80% recall targets no longer hold live.
3. **Keep weather**, despite its summer cost — it is the only source of day-to-day temporal
   skill, and that skill is real in shoulder season.
4. **Do not pursue** monotone constraints or removing `days_since_fire_cell`; both measured
   and rejected.

## Limitations

- 46 live days and only 193 genuinely-quiet ignitions; quiet-slice intervals are wide.
- Hyperparameters were tuned for the 37-feature set, which mildly favours it.
- The quiet/active split is a hard threshold at `eps=0.05`; cells near it flip regimes
  discontinuously. A smooth local-percentile formulation would avoid that and is untested.
- `lightning_count` forced to 0 on the autumn 2025 panel (no GLM archive).
- Era decay is not addressed here — the fitting window stays 2018-19 throughout.

## Reproducing

```bash
.venv/bin/python scripts/build_ablation_evalsets.py        # panels
.venv/bin/python scripts/run_feature_ablation.py           # 93 fits, ~18 min
.venv/bin/python -m scripts.analyze_feature_ablation       # main tables
.venv/bin/python -m scripts.analyze_ablation_operational   # lead time, tracer, CIs
.venv/bin/python -m scripts.experiment_recency_coldstart   # cold vs warm prior
.venv/bin/python -m scripts.experiment_tiering_remedies    # remedies
.venv/bin/python -m scripts.experiment_stratified_ci       # CIs on the tiering rules
.venv/bin/python -m scripts.experiment_gann_tracer         # the Gann Fire cell
```

Outputs land in `data/eval/ablation/` (gitignored): per-panel metrics CSVs, raw scores,
calibrators, bootstrap results, `gann_tracer.csv`.
