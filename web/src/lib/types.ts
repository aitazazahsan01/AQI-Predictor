/**
 * The shape of public/data/forecast.json.
 *
 * Mirrors `src/inference/snapshot.py` on the Python side. `schemaVersion` is
 * carried through so a future payload change is a visible mismatch rather than
 * a page that silently renders nothing.
 */

export interface Snapshot {
  schema_version: number;
  generated_at: string;
  city: City;
  feature_source: string;
  observed_days: number;
  latest: Latest;
  station: Station | null;
  forecast: ForecastDay[];
  alert: Alert | null;
  history: HistoryPoint[];
  drivers: HorizonDrivers[];
  models: ModelSummary[];
  scale: ScaleBand[];
}

export interface City {
  slug: string;
  name: string;
  latitude: number;
  longitude: number;
}

export interface Condition {
  label: string;
  value: number | null;
  unit: string;
}

export interface Latest {
  date: string;
  aqi: number | null;
  aqi_max: number | null;
  aqi_min: number | null;
  category: string | null;
  color: string | null;
  advice: string | null;
  conditions: Condition[];
}

/** Live monitoring-station reading. Display only - never training data. */
export interface Station {
  aqi?: number | null;
  station?: string | null;
  observed_at?: string | null;
  [key: string]: unknown;
}

export interface ForecastDay {
  horizon: number;
  date: string;
  aqi: number;
  category: string;
  color: string;
  model_type: string;
}

export interface Alert {
  level: "none" | "warning" | "critical";
  days: string[];
  worst_day: string;
  worst_aqi: number;
  category: string;
  color: string;
  advice: string;
}

export interface HistoryPoint {
  date: string;
  aqi: number | null;
}

export interface Driver {
  feature: string;
  label: string;
  value: number | null;
  contribution: number | null;
}

export interface HorizonDrivers {
  horizon: number;
  model_type: string;
  /** Non-null when SHAP could not explain this model family, with the reason. */
  unavailable: string | null;
  features: Driver[];
}

export interface ModelSummary {
  horizon: number;
  model_type: string;
  source: "registry" | "local" | string;
  metrics: { rmse?: number | null; mae?: number | null; r2?: number | null };
  n_features: number;
}

export interface ScaleBand {
  name: string;
  lower: number;
  /** null on the open-ended top band. */
  upper: number | null;
  color: string;
  alert_level: "none" | "warning" | "critical" | string;
  advice: string;
}
