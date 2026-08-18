import { formatAqi, formatDay, formatNumber } from "@/lib/format";
import type { Latest, ScaleBand } from "@/lib/types";
import { AqiBandStrip } from "./AqiBandStrip";
import styles from "./HeroNow.module.css";

interface Props {
  cityName: string;
  latest: Latest;
  scale: ScaleBand[];
  observedDays: number;
}

export function HeroNow({ cityName, latest, scale, observedDays }: Props) {
  return (
    <section className={`shell section ${styles.hero}`} id="top">
      <div className={styles.top}>
        <div className={styles.headline}>
          <p className="kicker">{cityName}</p>
          <h1 className={styles.title}>
            Three-day air quality forecast, with the reasoning shown.
          </h1>
          <p className="lede">
            Every number below was produced by an automated pipeline that ingests air quality
            hourly, engineers the same features it was trained on, and re-selects the best model
            each night from {observedDays.toLocaleString("en-GB")} days of history.
          </p>
        </div>

        <div className={styles.reading}>
          <p className="label">Observed {formatDay(latest.date)}</p>
          <div className={styles.value}>
            <span className={`display ${styles.number}`}>{formatAqi(latest.aqi)}</span>
            <span className={styles.unit}>AQI</span>
          </div>
          {latest.category && (
            <p className={styles.category}>
              <span className={styles.swatch} style={{ background: latest.color ?? undefined }} />
              {latest.category}
            </p>
          )}
          <p className={styles.range}>
            Ranged {formatAqi(latest.aqi_min)} to {formatAqi(latest.aqi_max)} across the day
          </p>
        </div>
      </div>

      <AqiBandStrip scale={scale} value={latest.aqi} />

      {latest.advice && <p className={styles.advice}>{latest.advice}</p>}

      {latest.conditions.length > 0 && (
        <dl className={styles.conditions}>
          {latest.conditions.map((condition) => (
            <div key={condition.label} className={styles.condition}>
              <dt className="label">{condition.label}</dt>
              <dd className={styles.conditionValue}>
                {formatNumber(condition.value)}
                <span className={styles.conditionUnit}>{condition.unit}</span>
              </dd>
            </div>
          ))}
        </dl>
      )}
    </section>
  );
}
