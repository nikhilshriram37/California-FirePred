"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

import ControlPanel from "./ControlPanel";
import CellDetail from "./CellDetail";
import DaySelector from "./DaySelector";
import type { CellProperties, FeatureCollection, RiskMeta, Tier } from "@/lib/types";

// MapLibre touches `window`, so load the map only on the client.
const MapView = dynamic(() => import("./MapView"), {
  ssr: false,
  loading: () => <div className="loading">Loading map…</div>,
});

const ALL_TIERS: Tier[] = ["Red", "Yellow", "Green"];

// "5m ago" / "3h ago" / "2d ago" — conveys how current the assessment is.
function timeAgo(iso?: string): string {
  if (!iso) return "";
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

function isoDate(daysAhead: number): string {
  const d = new Date();
  d.setUTCDate(d.getUTCDate() + daysAhead);
  return d.toISOString().slice(0, 10);
}

function tierCounts(fc: FeatureCollection | null): Record<Tier, number> {
  const c: Record<Tier, number> = { Red: 0, Yellow: 0, Green: 0 };
  for (const f of fc?.features ?? []) {
    const t = (f.properties as any)?.tier as Tier;
    if (t in c) c[t]++;
  }
  return c;
}

export default function Dashboard() {
  const [meta, setMeta] = useState<RiskMeta | null>(null);
  const [fires, setFires] = useState<FeatureCollection | null>(null);
  const [selected, setSelected] = useState<CellProperties | null>(null);
  const [visible, setVisible] = useState<Set<Tier>>(new Set(ALL_TIERS));

  // horizon 0 = today; 1..5 = forecast days ahead. Cache layers by horizon.
  const [horizon, setHorizon] = useState(0);
  const [layers, setLayers] = useState<Record<number, FeatureCollection>>({});

  useEffect(() => {
    let alive = true;
    Promise.all([
      fetch("/api/meta").then((r) => r.json()),
      fetch("/api/risk").then((r) => r.json()),
      fetch("/api/fires").then((r) => r.json()),
    ]).then(([m, rk, fr]) => {
      if (!alive) return;
      setMeta(m);
      setLayers((prev) => ({ ...prev, 0: rk }));
      setFires(fr);
    });
    return () => { alive = false; };
  }, []);

  // Lazily fetch a forecast horizon the first time it's selected.
  useEffect(() => {
    if (horizon === 0 || layers[horizon]) return;
    let alive = true;
    fetch(`/api/forecast?h=${horizon}`)
      .then((r) => r.json())
      .then((fc) => { if (alive) setLayers((prev) => ({ ...prev, [horizon]: fc })); });
    return () => { alive = false; };
  }, [horizon, layers]);

  const toggleTier = useCallback((t: Tier) => {
    setVisible((prev) => {
      const next = new Set(prev);
      next.has(t) ? next.delete(t) : next.add(t);
      return next;
    });
  }, []);

  const risk = layers[horizon] ?? null;
  const counts = useMemo(() => tierCounts(risk), [risk]);
  const fireCount = fires?.features.length ?? 0;
  const visibleTiers = useMemo(() => Array.from(visible), [visible]);
  const nCells = risk?.features.length ?? 0;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          <div>
            <h1>California Wildfire Risk — 5-Day Forecast</h1>
            <div className="subtitle">ML ignition-risk forecast · updated daily</div>
          </div>
        </div>
        <div className="status">
          <span className="badge forecast">5-DAY FORECAST</span>
          <span>
            {horizon === 0 ? "Today" : `Day +${horizon}`} <b>{isoDate(horizon)}</b>
          </span>
          {meta && <span>Updated {timeAgo(meta.generated_at)}</span>}
        </div>
      </header>

      <ControlPanel
        counts={counts}
        nCells={nCells}
        fireCount={fireCount}
        meta={meta}
        visible={visible}
        onToggleTier={toggleTier}
      />

      <div className="map-wrap">
        <MapView
          risk={risk}
          fires={fires}
          visibleTiers={visibleTiers}
          onSelectCell={setSelected}
        />
        <DaySelector horizon={horizon} onSelect={setHorizon} />
        {selected && <CellDetail cell={selected} onClose={() => setSelected(null)} />}
      </div>
    </div>
  );
}
