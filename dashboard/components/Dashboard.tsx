"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

import ControlPanel from "./ControlPanel";
import CellDetail from "./CellDetail";
import DaySelector from "./DaySelector";
import { dayDiff, formatMD, pacificToday, relLabel, timeAgo } from "@/lib/dates";
import type { CellProperties, FeatureCollection, Tier } from "@/lib/types";

const MapView = dynamic(() => import("./MapView"), {
  ssr: false,
  loading: () => <div className="loading">Loading map…</div>,
});

const ALL_TIERS: Tier[] = ["Red", "Yellow", "Green"];

interface ForecastInfo {
  run_date: string | null;
  generated_at: string | null;
  dates: Record<number, string>;
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
  const [info, setInfo] = useState<ForecastInfo | null>(null);
  const [fires, setFires] = useState<FeatureCollection | null>(null);
  const [selected, setSelected] = useState<CellProperties | null>(null);
  const [visible, setVisible] = useState<Set<Tier>>(new Set(ALL_TIERS));

  // horizon 0 = today; 1..5 = forecast days ahead. Active fires is an independent
  // overlay (showFires), so you can see fires and risk on the same map.
  const [horizon, setHorizon] = useState(0);
  const [showFires, setShowFires] = useState(false);
  const [layers, setLayers] = useState<Record<number, FeatureCollection>>({});

  useEffect(() => {
    let alive = true;
    Promise.all([
      fetch("/api/forecast/info").then((r) => r.json()),
      fetch("/api/forecast?h=0").then((r) => r.json()),
      fetch("/api/fires").then((r) => r.json()),
    ]).then(([inf, f0, fr]) => {
      if (!alive) return;
      setInfo(inf);
      setLayers((prev) => ({ ...prev, 0: f0 }));
      setFires(fr);
    });
    return () => { alive = false; };
  }, []);

  // Lazily fetch a forecast horizon (1..5) the first time it's selected.
  useEffect(() => {
    if (horizon <= 0 || layers[horizon]) return;
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

  const selectHorizon = useCallback((h: number) => {
    setHorizon(h);
    setSelected(null);
  }, []);

  const toggleFires = useCallback(() => setShowFires((v) => !v), []);

  const today = pacificToday();
  const risk = layers[horizon] ?? null;
  const displayedFires = showFires ? fires : null;
  const counts = useMemo(() => tierCounts(risk), [risk]);
  const fireCount = fires?.features.length ?? 0;
  const visibleTiers = useMemo(() => Array.from(visible), [visible]);
  const nCells = risk?.features.length ?? 0;
  const dates = info?.dates ?? {};
  const activeDate = dates[horizon];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          <div>
            <h1>California Wildfire Risk — 5-Day Forecast</h1>
            <div className="subtitle">ML ignition-risk forecast · updated daily ~6am PT</div>
          </div>
        </div>
        <div className="status">
          <span className="badge forecast">FORECAST</span>
          <span>
            {activeDate ? relLabel(dayDiff(activeDate, today)) : `Day +${horizon}`}
            {" · "}<b>{formatMD(activeDate)}</b>
          </span>
          <span>Updated {timeAgo(info?.generated_at)}</span>
          {showFires && (
            <span className="badge fires">🔥 {fireCount.toLocaleString()} active · last 48h</span>
          )}
        </div>
      </header>

      <ControlPanel
        counts={counts}
        nCells={nCells}
        fireCount={fireCount}
        showFires={showFires}
        generatedAt={info?.generated_at ?? null}
        visible={visible}
        onToggleTier={toggleTier}
      />

      <div className="map-wrap">
        <MapView
          risk={risk}
          fires={displayedFires}
          visibleTiers={visibleTiers}
          onSelectCell={setSelected}
        />
        <DaySelector
          horizon={horizon}
          dates={dates}
          today={today}
          onSelect={selectHorizon}
          showFires={showFires}
          onToggleFires={toggleFires}
          fireCount={fireCount}
        />
        {selected && <CellDetail cell={selected} onClose={() => setSelected(null)} />}
      </div>
    </div>
  );
}
