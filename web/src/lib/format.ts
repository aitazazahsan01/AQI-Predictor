/**
 * Formatting helpers.
 *
 * All dates are formatted in UTC. The pipeline aggregates on UTC days, so
 * rendering in the reader's local zone would shift labels by a day for anyone
 * east or west of Greenwich and quietly disagree with the AQI it sits next to.
 */

const UTC = { timeZone: "UTC" } as const;

export function formatDay(iso: string): string {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", {
    ...UTC,
    weekday: "short",
    day: "numeric",
    month: "short",
  });
}

export function formatWeekday(iso: string): string {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", {
    ...UTC,
    weekday: "long",
  });
}

export function formatShortDay(iso: string): string {
  return new Date(`${iso}T00:00:00Z`).toLocaleDateString("en-GB", {
    ...UTC,
    day: "numeric",
    month: "short",
  });
}

export function formatTimestamp(iso: string): string {
  return `${new Date(iso).toLocaleString("en-GB", {
    ...UTC,
    dateStyle: "medium",
    timeStyle: "short",
  })} UTC`;
}

/** AQI values are integers on screen; the decimals are false precision. */
export function formatAqi(value: number | null | undefined): string {
  return value == null ? "--" : Math.round(value).toString();
}

export function formatNumber(value: number | null | undefined, digits = 1): string {
  return value == null ? "--" : value.toFixed(digits);
}

/** "random_forest" -> "Random forest". Model names are stored as slugs. */
export function humanizeModel(name: string): string {
  const special: Record<string, string> = {
    xgboost: "XGBoost",
    lstm: "LSTM",
    sarimax: "SARIMAX",
  };
  if (special[name]) return special[name];
  const words = name.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function horizonLabel(horizon: number): string {
  return horizon === 1 ? "Tomorrow" : `In ${horizon} days`;
}
