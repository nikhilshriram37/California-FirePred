import "server-only";
import fs from "node:fs/promises";
import path from "node:path";

import { getServerSupabase } from "./supabase";
import type { FeatureCollection, RiskMeta } from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");
const CELL_HALF = 0.05; // half of the 0.1-degree grid cell

// Drivers surfaced in the cell click panel (mirror src/pipeline/snapshot.py).
const DETAIL_FEATURES = [
  "vpd", "fm100", "dry_streak", "bi_7d", "erc_7d",
  "tmmx_c", "rmin", "pr_14d", "lightning_count",
];

function cellPolygon(lon: number, lat: number) {
  return [[
    [lon - CELL_HALF, lat - CELL_HALF], [lon + CELL_HALF, lat - CELL_HALF],
    [lon + CELL_HALF, lat + CELL_HALF], [lon - CELL_HALF, lat + CELL_HALF],
    [lon - CELL_HALF, lat - CELL_HALF],
  ]];
}

async function readLocal<T>(file: string): Promise<T> {
  return JSON.parse(await fs.readFile(path.join(DATA_DIR, file), "utf8")) as T;
}

/** Latest snapshot metadata: date, mode, tier counts, model version. */
export async function getMeta(): Promise<RiskMeta> {
  const sb = getServerSupabase();
  if (sb) {
    // gridMET lags ~2 days, so many runs share the same data_date and risk_meta
    // is insert-only — tie-break on generated_at so we return the *newest* run's
    // metadata (n_cells, model_version), not an arbitrary same-date row.
    const { data } = await sb
      .from("risk_meta")
      .select("*")
      .order("data_date", { ascending: false })
      .order("generated_at", { ascending: false })
      .limit(1)
      .maybeSingle();
    if (data) return data as RiskMeta;
  }
  return readLocal<RiskMeta>("meta.json");
}

/** Risk choropleth as a polygon FeatureCollection for the latest scored date. */
export async function getRiskGeoJSON(): Promise<FeatureCollection> {
  const sb = getServerSupabase();
  if (sb) {
    // Latest scored date, then all cells for that date joined to their geometry.
    const { data: latest } = await sb
      .from("risk_scores")
      .select("date")
      .order("date", { ascending: false })
      .limit(1)
      .maybeSingle();
    if (latest?.date) {
      // Supabase caps responses at 1000 rows by default — page through them all.
      const PAGE = 1000;
      const pageAll = async (build: (from: number) => any) => {
        const acc: any[] = [];
        for (let from = 0; ; from += PAGE) {
          const { data, error } = await build(from).range(from, from + PAGE - 1);
          if (error || !data || data.length === 0) break;
          acc.push(...data);
          if (data.length < PAGE) break;
        }
        return acc;
      };

      const scores = await pageAll((from) =>
        sb.from("risk_scores")
          .select("grid_id, risk, tier, grid_cells(lat_center, lon_center)")
          .eq("date", latest.date));

      // Driver features for the click panel, keyed by grid_id.
      const featRows = await pageAll((from) =>
        sb.from("feature_history").select("grid_id, features").eq("date", latest.date));
      const drivers = new Map<number, Record<string, number>>();
      for (const r of featRows) drivers.set(r.grid_id, r.features ?? {});

      if (scores.length) {
        return {
          type: "FeatureCollection",
          features: scores.map((r: any) => {
            const lat = r.grid_cells?.lat_center;
            const lon = r.grid_cells?.lon_center;
            const f = drivers.get(r.grid_id) ?? {};
            const props: Record<string, unknown> = { grid_id: r.grid_id, risk: r.risk, tier: r.tier, lat, lon };
            for (const k of DETAIL_FEATURES) if (f[k] != null) props[k] = f[k];
            return {
              type: "Feature" as const,
              geometry: { type: "Polygon", coordinates: cellPolygon(lon, lat) },
              properties: props,
            };
          }),
        };
      }
    }
  }
  return readLocal<FeatureCollection>("risk_snapshot.geojson");
}

/** Forecast choropleth for a horizon (1..5 days ahead), latest run. */
export async function getForecast(horizon: number): Promise<FeatureCollection> {
  const sb = getServerSupabase();
  if (sb) {
    const { data: latest } = await sb
      .from("forecast_scores")
      .select("run_date")
      .order("run_date", { ascending: false })
      .limit(1)
      .maybeSingle();
    if (latest?.run_date) {
      const PAGE = 1000;
      const rows: any[] = [];
      for (let from = 0; ; from += PAGE) {
        const { data, error } = await sb
          .from("forecast_scores")
          .select("grid_id, risk, tier, grid_cells(lat_center, lon_center)")
          .eq("run_date", latest.run_date)
          .eq("horizon", horizon)
          .range(from, from + PAGE - 1);
        if (error || !data || data.length === 0) break;
        rows.push(...data);
        if (data.length < PAGE) break;
      }
      if (rows.length) {
        return {
          type: "FeatureCollection",
          features: rows.map((r: any) => {
            const lat = r.grid_cells?.lat_center;
            const lon = r.grid_cells?.lon_center;
            return {
              type: "Feature" as const,
              geometry: { type: "Polygon", coordinates: cellPolygon(lon, lat) },
              properties: { grid_id: r.grid_id, risk: r.risk, tier: r.tier, lat, lon },
            };
          }),
        };
      }
    }
  }
  // Local dev fallback: per-horizon snapshot written by the forecast pipeline.
  try {
    return await readLocal<FeatureCollection>(path.join("forecast", `h${horizon}`, "risk_snapshot.geojson"));
  } catch {
    return { type: "FeatureCollection", features: [] };
  }
}

export interface ForecastInfo {
  run_date: string | null;
  generated_at: string | null;
  /** horizon (0..5) -> real target date (YYYY-MM-DD, Pacific) */
  dates: Record<number, string>;
}

/** Real target dates + freshness for the latest forecast run (so the UI shows
 *  the dates the data is actually for, not a client-computed today+N). */
export async function getForecastInfo(): Promise<ForecastInfo> {
  const sb = getServerSupabase();
  if (sb) {
    const { data: latest } = await sb
      .from("forecast_scores")
      .select("run_date")
      .order("run_date", { ascending: false })
      .limit(1)
      .maybeSingle();
    if (latest?.run_date) {
      // One row per horizon (a plain select hits the 1000-row cap — 13k+ rows
      // per run — and would only return horizon 0).
      const dates: Record<number, string> = {};
      for (let h = 0; h <= 5; h++) {
        const { data } = await sb
          .from("forecast_scores")
          .select("target_date")
          .eq("run_date", latest.run_date)
          .eq("horizon", h)
          .limit(1)
          .maybeSingle();
        if (data?.target_date) dates[h] = data.target_date;
      }
      // Freshness: the daily run timestamp (nowcast + forecast run together).
      const { data: meta } = await sb
        .from("risk_meta").select("generated_at")
        .order("generated_at", { ascending: false }).limit(1).maybeSingle();
      return { run_date: latest.run_date, generated_at: meta?.generated_at ?? null, dates };
    }
  }
  // Local dev fallback: read each horizon's meta.json.
  const dates: Record<number, string> = {};
  let generated_at: string | null = null;
  let run_date: string | null = null;
  for (let h = 0; h <= 5; h++) {
    try {
      const m = await readLocal<RiskMeta & { horizon: number; run_date: string }>(
        path.join("forecast", `h${h}`, "meta.json"));
      dates[h] = m.data_date;
      generated_at = m.generated_at;
      run_date = (m as any).run_date ?? run_date;
    } catch { /* horizon not present locally */ }
  }
  return { run_date, generated_at, dates };
}

/**
 * Active fire detections (NASA FIRMS VIIRS NRT) for California, last 24h.
 * Returns an empty collection when no FIRMS key is set.
 */
export async function getActiveFires(): Promise<FeatureCollection> {
  const key = process.env.NASA_FIRMS_MAP_KEY;
  const empty: FeatureCollection = { type: "FeatureCollection", features: [] };
  if (!key) return empty;

  const bbox = "-124.5,32.5,-114.0,42.0"; // west,south,east,north (California)
  const days = 2; // last 48h — a single satellite/day can be empty between passes
  const sources = ["VIIRS_NOAA20_NRT", "VIIRS_SNPP_NRT", "MODIS_NRT"];

  const results = await Promise.allSettled(
    sources.map(async (src) => {
      const url = `https://firms.modaps.eosdis.nasa.gov/api/area/csv/${key}/${src}/${bbox}/${days}`;
      const res = await fetch(url, { next: { revalidate: 900 } }); // cache 15 min
      if (!res.ok) return [];
      return parseFirmsCsv(await res.text());
    }),
  );

  const features = results.flatMap((r) => (r.status === "fulfilled" ? r.value : []));
  return { type: "FeatureCollection", features };
}

function parseFirmsCsv(text: string): FeatureCollection["features"] {
  const lines = text.trim().split("\n");
  if (lines.length < 2) return [];
  const header = lines[0].split(",");
  const idx = (name: string) => header.indexOf(name);
  const iLat = idx("latitude"), iLon = idx("longitude");
  const iFrp = idx("frp"), iConf = idx("confidence");
  const iDate = idx("acq_date"), iTime = idx("acq_time"), iSat = idx("satellite");
  if (iLat < 0 || iLon < 0) return [];

  return lines.slice(1).map((line) => {
    const c = line.split(",");
    const lat = parseFloat(c[iLat]), lon = parseFloat(c[iLon]);
    return {
      type: "Feature" as const,
      geometry: { type: "Point", coordinates: [lon, lat] },
      properties: {
        frp: iFrp >= 0 ? parseFloat(c[iFrp]) : null,
        confidence: iConf >= 0 ? c[iConf] : null,
        acq_date: iDate >= 0 ? c[iDate] : null,
        acq_time: iTime >= 0 ? c[iTime] : null,
        satellite: iSat >= 0 ? c[iSat] : null,
      },
    };
  }).filter((f) => Number.isFinite((f.geometry.coordinates as number[])[0]));
}
