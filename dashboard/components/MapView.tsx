"use client";

import { useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";

import { TIER_COLORS, type CellProperties, type FeatureCollection, type Tier } from "@/lib/types";

const CALIFORNIA_BOUNDS: [number, number, number, number] = [-124.5, 32.3, -114.0, 42.1];
const EMPTY: FeatureCollection = { type: "FeatureCollection", features: [] };
// Free CARTO basemap (no API token required); dark theme for an ops-center feel.
const BASEMAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

interface Props {
  risk: FeatureCollection | null;
  fires: FeatureCollection | null;
  visibleTiers: Tier[];
  onSelectCell: (cell: CellProperties | null) => void;
}

export default function MapView({ risk, fires, visibleTiers, onSelectCell }: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const [ready, setReady] = useState(false);

  // Initialize the map once.
  useEffect(() => {
    if (map.current || !container.current) return;
    const m = new maplibregl.Map({
      container: container.current,
      style: BASEMAP,
      bounds: CALIFORNIA_BOUNDS,
      fitBoundsOptions: { padding: 24 },
      attributionControl: { compact: true },
    });
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");

    m.on("load", () => {
      m.addSource("risk", { type: "geojson", data: EMPTY as never });
      m.addSource("fires", { type: "geojson", data: EMPTY as never });

      m.addLayer({
        id: "risk-fill",
        type: "fill",
        source: "risk",
        paint: {
          "fill-color": [
            "match", ["get", "tier"],
            "Red", TIER_COLORS.Red,
            "Yellow", TIER_COLORS.Yellow,
            "Green", TIER_COLORS.Green,
            "#555",
          ],
          "fill-opacity": ["match", ["get", "tier"], "Green", 0.35, 0.6],
        },
      });
      m.addLayer({
        id: "risk-outline",
        type: "line",
        source: "risk",
        paint: { "line-color": "#000", "line-opacity": 0.15, "line-width": 0.3 },
      });

      // Active fire detections (live FIRMS): glow + core.
      m.addLayer({
        id: "fires-glow",
        type: "circle",
        source: "fires",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["coalesce", ["get", "frp"], 1], 0, 6, 50, 16],
          "circle-color": "#ff5a1f",
          "circle-blur": 1,
          "circle-opacity": 0.5,
        },
      });
      m.addLayer({
        id: "fires-core",
        type: "circle",
        source: "fires",
        paint: { "circle-radius": 2.5, "circle-color": "#ffd24a", "circle-opacity": 0.95 },
      });

      m.on("click", "risk-fill", (e) => {
        const f = e.features?.[0];
        if (f) onSelectCell(f.properties as unknown as CellProperties);
      });
      m.on("mouseenter", "risk-fill", () => (m.getCanvas().style.cursor = "pointer"));
      m.on("mouseleave", "risk-fill", () => (m.getCanvas().style.cursor = ""));

      setReady(true);
    });

    map.current = m;
    return () => {
      m.remove();
      map.current = null;
    };
  }, [onSelectCell]);

  // Push risk data when it arrives/changes.
  useEffect(() => {
    if (!ready || !map.current) return;
    const src = map.current.getSource("risk") as maplibregl.GeoJSONSource | undefined;
    src?.setData((risk ?? EMPTY) as never);
  }, [ready, risk]);

  // Push fire data.
  useEffect(() => {
    if (!ready || !map.current) return;
    const src = map.current.getSource("fires") as maplibregl.GeoJSONSource | undefined;
    src?.setData((fires ?? EMPTY) as never);
  }, [ready, fires]);

  // Apply the tier visibility filter.
  useEffect(() => {
    if (!ready || !map.current) return;
    map.current.setFilter("risk-fill", ["in", ["get", "tier"], ["literal", visibleTiers]]);
    map.current.setFilter("risk-outline", ["in", ["get", "tier"], ["literal", visibleTiers]]);
  }, [ready, visibleTiers]);

  return <div id="map" ref={container} />;
}
