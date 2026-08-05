"use client";

import { TIER_COLORS, type CellProperties, type Tier } from "@/lib/types";

// Human-friendly labels + units for the model drivers surfaced per cell.
// Fire recency leads because it leads the model: since the recency features were
// added it accounts for ~41% of total gain, more than every weather feature combined.
// Showing only weather here would misrepresent why a cell is red.
const FEATURE_LABELS: Record<string, string> = {
  days_since_fire_cell: "Days since last fire here",
  fire_recency_nbr: "Recent fire activity nearby",
  vpd: "Vapor pressure deficit",
  fm100: "100-hr fuel moisture (%)",
  dry_streak: "Consecutive dry days",
  bi_7d: "Burning index (7-day)",
  erc_7d: "Energy release (7-day)",
  tmmx_c: "Max temperature (°C)",
  rmin: "Min humidity (%)",
  pr_14d: "Precip, 14-day (mm)",
  lightning_count: "Lightning strikes",
};

// days_since_fire_cell saturates rather than growing without bound, so the cap is a
// floor on "a long time ago", not a measurement.
const DAYS_SINCE_CAP = 365;

function formatDriver(key: string, value: number): string {
  if (key === "days_since_fire_cell") {
    return value >= DAYS_SINCE_CAP ? `${DAYS_SINCE_CAP}+` : Math.round(value).toLocaleString();
  }
  if (key === "fire_recency_nbr") return value.toFixed(2);
  return value.toLocaleString();
}

interface Props {
  cell: CellProperties;
  onClose: () => void;
}

export default function CellDetail({ cell, onClose }: Props) {
  const tier = cell.tier as Tier;
  const riskPct = (Number(cell.risk) * 100).toFixed(2);
  const drivers = Object.keys(FEATURE_LABELS).filter((k) => cell[k] != null);

  return (
    <div className="detail">
      <span className="close" onClick={onClose}>✕</span>
      <h3>Cell #{cell.grid_id}</h3>
      <div className="sub">
        {Number(cell.lat).toFixed(3)}, {Number(cell.lon).toFixed(3)}
      </div>

      <span className="tier-pill" style={{ background: TIER_COLORS[tier] }}>
        {tier} · {riskPct}% risk
      </span>
      <div className="riskbar">
        <div
          style={{
            width: `${Math.min(100, Number(cell.risk) * 100 * 6)}%`,
            background: TIER_COLORS[tier],
          }}
        />
      </div>

      {cell.has_fire != null && (
        <div className="sub" style={{ marginBottom: 10 }}>
          Ground truth: {Number(cell.has_fire) ? "🔥 fire occurred" : "no fire"}
        </div>
      )}

      <table>
        <tbody>
          {drivers.map((k) => (
            <tr key={k}>
              <td style={{ color: "var(--muted)" }}>{FEATURE_LABELS[k]}</td>
              <td>{formatDriver(k, Number(cell[k]))}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
