export type Tier = "Red" | "Yellow" | "Green";

export const TIER_COLORS: Record<Tier, string> = {
  Red: "#d6604d",
  Yellow: "#e6b800",
  Green: "#4d9221",
};

export interface RiskMeta {
  data_date: string;
  generated_at: string;
  source: string;
  mode: "replay" | "live" | string;
  model_version: string;
  n_cells: number;
  tier_counts: Record<Tier, number>;
  actual_fires?: number;
  thresholds: Record<string, number>;
}

export interface CellProperties {
  grid_id: number;
  risk: number;
  tier: Tier;
  lat: number;
  lon: number;
  has_fire?: number;
  [feature: string]: number | string | undefined;
}

export interface FeatureCollection {
  type: "FeatureCollection";
  features: Array<{
    type: "Feature";
    geometry: { type: string; coordinates: unknown };
    properties: Record<string, unknown>;
  }>;
}
