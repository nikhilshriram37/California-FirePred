"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

import ControlPanel from "./ControlPanel";
import CellDetail from "./CellDetail";
import type { CellProperties, FeatureCollection, RiskMeta, Tier } from "@/lib/types";

// MapLibre touches `window`, so load the map only on the client.
const MapView = dynamic(() => import("./MapView"), {
  ssr: false,
  loading: () => <div className="loading">Loading map…</div>,
});

const ALL_TIERS: Tier[] = ["Red", "Yellow", "Green"];

export default function Dashboard() {
  const [meta, setMeta] = useState<RiskMeta | null>(null);
  const [risk, setRisk] = useState<FeatureCollection | null>(null);
  const [fires, setFires] = useState<FeatureCollection | null>(null);
  const [selected, setSelected] = useState<CellProperties | null>(null);
  const [visible, setVisible] = useState<Set<Tier>>(new Set(ALL_TIERS));

  useEffect(() => {
    let alive = true;
    Promise.all([
      fetch("/api/meta").then((r) => r.json()),
      fetch("/api/risk").then((r) => r.json()),
      fetch("/api/fires").then((r) => r.json()),
    ]).then(([m, rk, fr]) => {
      if (!alive) return;
      setMeta(m);
      setRisk(rk);
      setFires(fr);
    });
    return () => {
      alive = false;
    };
  }, []);

  const toggleTier = useCallback((t: Tier) => {
    setVisible((prev) => {
      const next = new Set(prev);
      next.has(t) ? next.delete(t) : next.add(t);
      return next;
    });
  }, []);

  const fireCount = fires?.features.length ?? 0;
  const visibleTiers = useMemo(() => Array.from(visible), [visible]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="dot" />
          <h1>California Wildfire Risk — Operations Dashboard</h1>
        </div>
        <div className="status">
          {meta && (
            <>
              <span>
                Forecast date <b>{meta.data_date}</b>
              </span>
              <span className={`badge ${meta.mode === "live" ? "live" : "replay"}`}>
                {meta.mode === "live" ? "LIVE FEED" : "REPLAY"}
              </span>
              <span>
                Model <b>{meta.model_version}</b>
              </span>
            </>
          )}
        </div>
      </header>

      <ControlPanel
        meta={meta}
        fireCount={fireCount}
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
        {selected && (
          <CellDetail cell={selected} onClose={() => setSelected(null)} />
        )}
      </div>
    </div>
  );
}
