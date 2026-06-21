import "server-only";
import fs from "node:fs/promises";
import path from "node:path";

import { getServerSupabase } from "./supabase";
import type { FeatureCollection, RiskMeta } from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");
const CELL_HALF = 0.05; // half of the 0.1-degree grid cell

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
    const { data } = await sb
      .from("risk_meta")
      .select("*")
      .order("data_date", { ascending: false })
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
      const { data: rows } = await sb
        .from("risk_scores")
        .select("grid_id, risk, tier, grid_cells(lat_center, lon_center)")
        .eq("date", latest.date);
      if (rows) {
        return {
          type: "FeatureCollection",
          features: rows.map((r: any) => {
            const lat = r.grid_cells?.lat_center;
            const lon = r.grid_cells?.lon_center;
            return {
              type: "Feature" as const,
              geometry: { type: "Polygon", coordinates: cellPolygon(lon, lat) },
              properties: {
                grid_id: r.grid_id, risk: r.risk, tier: r.tier, lat, lon,
              },
            };
          }),
        };
      }
    }
  }
  return readLocal<FeatureCollection>("risk_snapshot.geojson");
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
  const url = `https://firms.modaps.eosdis.nasa.gov/api/area/csv/${key}/VIIRS_NOAA20_NRT/${bbox}/1`;

  try {
    const res = await fetch(url, { next: { revalidate: 900 } }); // cache 15 min
    if (!res.ok) return empty;
    const text = await res.text();
    return { type: "FeatureCollection", features: parseFirmsCsv(text) };
  } catch {
    return empty;
  }
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
