import { formatShortDay } from "@/lib/format";
import type { ForecastDay, HistoryPoint } from "@/lib/types";
import styles from "./TrendChart.module.css";

/* Drawn in a fixed coordinate space and scaled by CSS. Strokes carry
   vector-effect="non-scaling-stroke" so a 2px rule stays 2px at every width,
   which is the whole point of a system built on rules. */
const VIEW_W = 1000;
const VIEW_H = 380;
const PAD = { top: 24, right: 104, bottom: 34, left: 46 };

const PLOT_W = VIEW_W - PAD.left - PAD.right;
const PLOT_H = VIEW_H - PAD.top - PAD.bottom;

/** Boundaries worth a gridline, with the band they open. */
const THRESHOLDS = [
  { value: 50, label: "Good" },
  { value: 100, label: "Moderate" },
  { value: 150, label: "Sensitive" },
  { value: 200, label: "Unhealthy" },
  { value: 300, label: "Very unhealthy" },
];

interface Props {
  history: HistoryPoint[];
  forecast: ForecastDay[];
}

interface Point {
  x: number;
  y: number;
}

export function TrendChart({ history, forecast }: Props) {
  const observed = history.filter((point) => point.aqi != null) as { date: string; aqi: number }[];
  if (observed.length === 0) return null;

  const series = [
    ...observed.map((point) => ({ date: point.date, aqi: point.aqi })),
    ...forecast.map((day) => ({ date: day.date, aqi: day.aqi })),
  ];

  const peak = Math.max(...series.map((point) => point.aqi));
  // Round the axis up to the next band boundary so the gridlines mean
  // something, instead of ending on an arbitrary maximum.
  const ceiling = Math.max(100, THRESHOLDS.find((t) => t.value >= peak * 1.08)?.value ?? peak * 1.1);

  const x = (index: number) => PAD.left + (index / (series.length - 1)) * PLOT_W;
  const y = (value: number) => PAD.top + PLOT_H - (value / ceiling) * PLOT_H;

  const toPath = (points: Point[]) =>
    points.map((point, i) => `${i === 0 ? "M" : "L"}${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(" ");

  const observedPoints: Point[] = observed.map((point, i) => ({ x: x(i), y: y(point.aqi) }));
  // The forecast line starts at the last observed value so the join is a
  // continuous reading rather than a floating second series.
  const forecastPoints: Point[] = [
    observedPoints[observedPoints.length - 1],
    ...forecast.map((day, i) => ({ x: x(observed.length + i), y: y(day.aqi) })),
  ];

  const visibleThresholds = THRESHOLDS.filter((t) => t.value <= ceiling);
  const tickEvery = Math.max(1, Math.ceil(observed.length / 6));

  return (
    <figure className={styles.figure}>
      <svg
        className={styles.svg}
        viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
        role="img"
        aria-label={`Daily average AQI over the last ${observed.length} days, followed by a three-day forecast.`}
      >
        {/* gridlines */}
        {visibleThresholds.map((threshold) => (
          <g key={threshold.value}>
            <line
              x1={PAD.left}
              x2={VIEW_W - PAD.right}
              y1={y(threshold.value)}
              y2={y(threshold.value)}
              className={styles.grid}
              vectorEffect="non-scaling-stroke"
            />
            <text x={PAD.left - 8} y={y(threshold.value) + 4} className={styles.axisLabel} textAnchor="end">
              {threshold.value}
            </text>
            <text x={VIEW_W - PAD.right + 8} y={y(threshold.value) + 4} className={styles.bandLabel}>
              {threshold.label}
            </text>
          </g>
        ))}

        {/* baseline */}
        <line
          x1={PAD.left}
          x2={VIEW_W - PAD.right}
          y1={PAD.top + PLOT_H}
          y2={PAD.top + PLOT_H}
          className={styles.axis}
          vectorEffect="non-scaling-stroke"
        />

        {/* the boundary between what happened and what is predicted */}
        <line
          x1={x(observed.length - 1)}
          x2={x(observed.length - 1)}
          y1={PAD.top}
          y2={PAD.top + PLOT_H}
          className={styles.divider}
          vectorEffect="non-scaling-stroke"
        />
        <text x={x(observed.length - 1) + 6} y={PAD.top + 11} className={styles.marker}>
          FORECAST
        </text>

        <path d={toPath(observedPoints)} className={styles.observed} vectorEffect="non-scaling-stroke" />
        <path d={toPath(forecastPoints)} className={styles.forecast} vectorEffect="non-scaling-stroke" />

        {forecastPoints.slice(1).map((point, i) => (
          <rect
            key={i}
            x={point.x - 4}
            y={point.y - 4}
            width={8}
            height={8}
            className={styles.forecastPoint}
            vectorEffect="non-scaling-stroke"
          />
        ))}

        {/* date ticks */}
        {observed.map((point, i) =>
          i % tickEvery === 0 ? (
            <text key={point.date} x={x(i)} y={VIEW_H - 12} className={styles.axisLabel} textAnchor="middle">
              {formatShortDay(point.date)}
            </text>
          ) : null,
        )}
      </svg>

      <figcaption className={styles.legend}>
        <span className={styles.key}>
          <span className={styles.keyObserved} aria-hidden="true" />
          Observed daily average
        </span>
        <span className={styles.key}>
          <span className={styles.keyForecast} aria-hidden="true" />
          Forecast
        </span>
      </figcaption>
    </figure>
  );
}
