"""Contract tests for regime-aware tiering.

These guard the change made 2026-08-05, where the Yellow cutoff became per-regime while
Red stayed global. The failure modes worth catching are quiet ones: a Yellow cutoff that
drifts above Red silently empties a tier, and a missing recency column silently changes
every cell's tier. Both produce a plausible-looking map.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data_acquisition.config import PROJECT_ROOT
from src.models.features import FEATURE_COLS
from src.models.predict import RiskModel
from src.models.recency import QUIET_EPS, RECENCY_FEATURES, is_quiet

THRESHOLDS = json.loads((PROJECT_ROOT / "models" / "thresholds.json").read_text())


def test_regime_yellow_cutoffs_sit_below_red():
    """A Yellow cutoff above Red would empty that regime's Yellow tier entirely.

    This is not hypothetical: deriving the cutoff as the top slice of each regime
    outright put the active regime's Yellow above Red, because active cells are far
    riskier than the statewide population Red is calibrated on.
    """
    red = THRESHOLDS["red"]
    for key in ("yellow_quiet", "yellow_active"):
        assert THRESHOLDS[key] < red, f"{key} ({THRESHOLDS[key]}) must sit below red ({red})"


def test_quiet_cutoff_is_looser_than_active():
    """Quiet areas must clear a lower bar for Yellow — that is the whole point."""
    assert THRESHOLDS["yellow_quiet"] < THRESHOLDS["yellow_active"]


def test_lightning_is_not_a_model_input():
    """Removed 2026-08-05 after measuring a coin-flip contribution on four panels."""
    assert "lightning_count" not in FEATURE_COLS
    assert len(FEATURE_COLS) == 36


def test_is_quiet_reads_both_recency_features():
    df = pd.DataFrame({
        "fire_recency_cell": [0.0, 0.0, 1.0, 1.0],
        "fire_recency_nbr": [0.0, 1.0, 0.0, 1.0],
    })
    assert is_quiet(df).tolist() == [True, False, False, False]


def test_quiet_boundary_is_exclusive_at_eps():
    df = pd.DataFrame({"fire_recency_cell": [QUIET_EPS - 1e-9, QUIET_EPS],
                       "fire_recency_nbr": [0.0, 0.0]})
    assert is_quiet(df).tolist() == [True, False]


def _model() -> RiskModel:
    return RiskModel()


def test_to_tier_falls_back_to_global_without_regime():
    """An unknown regime must not silently pick one — it falls back to the old rule."""
    m = _model()
    risk = np.array([0.9, THRESHOLDS["yellow"], 0.0])
    assert m.to_tier(risk, None).tolist() == ["Red", "Yellow", "Green"]


def test_to_tier_falls_back_when_artifacts_predate_the_change():
    """An older models/ directory has no regime keys; scoring must still work."""
    m = _model()
    m.thresholds = {k: v for k, v in m.thresholds.items()
                    if k not in ("yellow_quiet", "yellow_active")}
    risk = np.array([0.9, THRESHOLDS["yellow"], 0.0])
    quiet = np.array([True, True, True])
    assert m.to_tier(risk, quiet).tolist() == ["Red", "Yellow", "Green"]


def test_to_tier_applies_the_looser_cutoff_only_to_quiet_cells():
    m = _model()
    # A risk between the two cutoffs: Yellow when quiet, Green when active.
    mid = (THRESHOLDS["yellow_quiet"] + THRESHOLDS["yellow_active"]) / 2
    risk = np.array([mid, mid])
    assert m.to_tier(risk, np.array([True, False])).tolist() == ["Yellow", "Green"]


def test_red_is_regime_independent():
    """Red must mean the same thing statewide, quiet or not."""
    m = _model()
    risk = np.full(2, THRESHOLDS["red"] + 1e-6)
    assert m.to_tier(risk, np.array([True, False])).tolist() == ["Red", "Red"]


def test_predict_without_recency_columns_does_not_crash():
    """A feature frame missing the recency block scores on the global cutoff."""
    m = _model()
    df = pd.DataFrame(np.zeros((3, len(FEATURE_COLS))), columns=FEATURE_COLS)
    df = df.drop(columns=RECENCY_FEATURES)
    out = m.predict(df)
    assert set(out.columns) == {"raw_probability", "risk", "tier"}
    assert out["tier"].isin(["Red", "Yellow", "Green"]).all()


@pytest.mark.skipif(not (PROJECT_ROOT / "data" / "eval" / "panel_live2026.parquet").exists(),
                    reason="evaluation panel not built")
def test_regime_tiering_raises_quiet_area_coverage_on_the_live_panel():
    """The change has to actually do its job on real data, not just in the unit sense."""
    m = _model()
    d = pd.read_parquet(PROJECT_ROOT / "data" / "eval" / "panel_live2026.parquet")
    y, q = d["confirmed"].to_numpy(), is_quiet(d)
    out = m.predict(d)
    new = out["tier"].to_numpy() != "Green"
    old = m.to_tier(out["risk"].to_numpy(), None) != "Green"
    quiet_new = y[new & q].sum() / y[q].sum()
    quiet_old = y[old & q].sum() / y[q].sum()
    assert quiet_new > 2 * quiet_old, f"quiet coverage {quiet_old:.1%} -> {quiet_new:.1%}"
    # Red is untouched by construction; assert it rather than trust it.
    assert ((out["tier"] == "Red").to_numpy()
            == (m.to_tier(out["risk"].to_numpy(), None) == "Red")).all()
