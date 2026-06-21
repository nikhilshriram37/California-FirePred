"use client";

import { TIER_COLORS, type RiskMeta, type Tier } from "@/lib/types";

const TIERS: { tier: Tier; label: string; desc: string }[] = [
  { tier: "Red", label: "High risk", desc: "≥60% of fires" },
  { tier: "Yellow", label: "Elevated", desc: "≥90% cumulative" },
  { tier: "Green", label: "Clear", desc: "low risk" },
];

interface Props {
  meta: RiskMeta | null;
  fireCount: number;
  visible: Set<Tier>;
  onToggleTier: (t: Tier) => void;
}

export default function ControlPanel({ meta, fireCount, visible, onToggleTier }: Props) {
  const counts = meta?.tier_counts;
  const flagged = counts ? counts.Red + counts.Yellow : 0;

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
            <span className="count">{counts ? counts[tier].toLocaleString() : "—"}</span>
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
            <div className="k">active fire detections (24h)</div>
          </div>
          <div className="stat">
            <div className="v">{meta ? meta.n_cells.toLocaleString() : "—"}</div>
            <div className="k">grid cells scored</div>
          </div>
          <div className="stat">
            <div className="v">{meta?.actual_fires != null ? meta.actual_fires : "—"}</div>
            <div className="k">{meta?.mode === "replay" ? "actual fires (replay)" : "—"}</div>
          </div>
        </div>
        <div className="fire-legend">
          <span className="fire-dot" /> live VIIRS satellite fire detection
        </div>
      </div>

      <div className="section">
        <h2>How to read this</h2>
        <p className="legend-note">
          Each ~10&nbsp;km cell is scored for next-day ignition risk by a calibrated
          XGBoost model and binned into recall-targeted tiers: flagging red+yellow
          catches the large majority of fires while keeping the search area
          manageable for crews. Click any cell for the drivers behind its score.
        </p>
        {meta && (
          <p className="legend-note" style={{ marginTop: 8 }}>
            Updated {new Date(meta.generated_at).toLocaleString()} · source{" "}
            <b>{meta.source}</b>
          </p>
        )}
      </div>
    </aside>
  );
}
