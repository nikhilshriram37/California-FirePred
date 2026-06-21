"use client";

import { TIER_COLORS, type RiskMeta, type Tier } from "@/lib/types";

const TIERS: { tier: Tier; label: string; desc: string }[] = [
  { tier: "Red", label: "High risk", desc: "≥45% of fires" },
  { tier: "Yellow", label: "Elevated", desc: "≥80% cumulative" },
  { tier: "Green", label: "Clear", desc: "low risk" },
];

interface Props {
  counts: Record<Tier, number>;
  nCells: number;
  fireCount: number;
  meta: RiskMeta | null;
  visible: Set<Tier>;
  onToggleTier: (t: Tier) => void;
}

export default function ControlPanel({ counts, nCells, fireCount, meta, visible, onToggleTier }: Props) {
  const flagged = counts.Red + counts.Yellow;

  return (
    <aside className="sidebar">
      <div className="section">
        <h2>Risk tiers — click to filter</h2>
        {TIERS.map(({ tier, label, desc }) => (
          <div
            key={tier}
            className={`tier-row ${visible.has(tier) ? "" : "off"}`}
            onClick={() => onToggleTier(tier)}
          >
            <span className="swatch" style={{ background: TIER_COLORS[tier] }} />
            <span className="label">
              {label} <span style={{ color: "var(--muted)", fontWeight: 400 }}>· {desc}</span>
            </span>
            <span className="count">{counts[tier].toLocaleString()}</span>
          </div>
        ))}
      </div>

      <div className="section">
        <h2>Situation</h2>
        <div className="stat-grid">
          <div className="stat">
            <div className="v">{flagged.toLocaleString()}</div>
            <div className="k">cells flagged (red+yellow)</div>
          </div>
          <div className="stat">
            <div className="v" style={{ color: "var(--fire)" }}>{fireCount.toLocaleString()}</div>
            <div className="k">active fire detections (48h)</div>
          </div>
          <div className="stat">
            <div className="v">{nCells.toLocaleString()}</div>
            <div className="k">grid cells scored</div>
          </div>
          <div className="stat">
            <div className="v">{flagged && nCells ? Math.round((flagged / nCells) * 100) : 0}%</div>
            <div className="k">of state flagged</div>
          </div>
        </div>
        <div className="fire-legend">
          <span className="fire-dot" /> live VIIRS satellite fire detection
        </div>
      </div>

      <div className="section">
        <h2>How to read this</h2>
        <p className="legend-note">
          A <b>5-day wildfire ignition-risk forecast.</b> Each ~10&nbsp;km cell is
          scored by a calibrated XGBoost model for the selected day
          (<b>Today</b> through <b>+5d</b>) and binned into recall-targeted tiers, so
          flagging red+yellow catches most fires while keeping the search area
          manageable. Forecast skill is strongest in the near term and softens toward
          day&nbsp;5. Click any cell for the drivers behind its score.
        </p>
        {meta && (
          <p className="legend-note" style={{ marginTop: 8 }}>
            Forecast updated {new Date(meta.generated_at).toLocaleString()}
          </p>
        )}
      </div>
    </aside>
  );
}
