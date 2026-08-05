# Evaluation datasets

Held-out data for grading the live model. Gitignored (derived + large), but kept on
disk because rebuilding means paged Supabase exports and rate-limited API paging.

All of it is **recent and real** — the point of these files is that the 2020 backtest
is not a trustworthy guide to current performance. Measured 2026-08-05: the deployed
model scores ROC 0.97 on its training years, 0.92 on held-out 2020, and ~0.81 live.
Grade on these, not on 2020.

| file | what it is |
|---|---|
| `live_record.parquet` | 2026-06-19..08-03 exported from Supabase: 184,314 cell-days, the features **as served**, predictions, `has_fire`, `label_source`. Use this to see what the pipeline actually did. |
| `live_features_corrected.parquet` | Same period and labels, but features **recomputed** from the gridMET archive with full history. Use this for modelling — the served features were corrupted (see below). |
| `labels_2025.parquet` | 6,921 confirmed CA ignitions, 2025-07-01..11-30, from the WFIGS incident archive. Covers the Sep-Nov shoulder season. July onward so the recency feature has warm-up. |
| `models_pre_recency/` | The 34-feature model deployed before 2026-08-05. Kept for before/after comparison. Feed it its own `feature_list.json` — `select_features()` will hand it today's 37 and fail. |

## Why the corrected features exist

The served features do not match the gridMET archive. `dry_streak` is a cumulative
counter with no history bound, but live scoring only fetched a ~26-day window, so it
was capped at ~27 where training reached 216; stored precipitation was also wrong on
days scored from a stale cache. An adversarial classifier separates served-vs-training
data at AUC 0.9998. `live_features_corrected.parquet` rebuilds the whole period from
the archive so modelling work is not fitted to that corruption.

## Confirmed vs fused labels

`confirmed` = IRWIN/CAL FIRE agency ignition records, which is what the model is
trained on (FPA-FOD ignitions). `has_fire` additionally counts FIRMS satellite
detections, which mark where fire is *burning*, not where it *began* — about 48% of
positives. **Grade on `confirmed`.**

## Regenerating

- `live_record.parquet` — page `feature_history` + `risk_scores` per date, ordered by
  `grid_id` (PostREST `.range()` without an order returns inconsistent pages).
- `live_features_corrected.parquet` — `fetch_gridmet_for_grid(2026)` with a 365-day
  lookback, `engineer_features`, `merge_static_features`, `merge_recency`, then join
  live labels and lightning.
- `labels_2025.parquet` — `WFIGS_Incident_Locations` archive (the YearToDate service is
  current-year only; the archive has identical fields), 1000/page, back off on HTTP 429.
