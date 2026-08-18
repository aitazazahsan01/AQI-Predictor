import { formatAqi, formatDay, formatWeekday, horizonLabel, humanizeModel } from "@/lib/format";
import type { ForecastDay, ScaleBand } from "@/lib/types";
import styles from "./ForecastGrid.module.css";

const SCALE_CEILING = 400;

interface Props {
  forecast: ForecastDay[];
  scale: ScaleBand[];
}

export function ForecastGrid({ forecast, scale }: Props) {
  return (
    <section className="shell section" id="forecast">
      <div className="section-head">
        <div>
          <p className="kicker">The forecast</p>
          <h2>The next three days</h2>
        </div>
        <p className={styles.note}>
          Each day is predicted by its own model, chosen on hold-out error.
        </p>
      </div>

      <div className="grid grid-3">
        {forecast.map((day) => (
          <article key={day.horizon} className={styles.cell}>
            <p className="label">{horizonLabel(day.horizon)}</p>
            <h3 className={styles.day}>{formatWeekday(day.date)}</h3>
            <p className={styles.date}>{formatDay(day.date)}</p>

            <div className={styles.value}>
              <span className={`display ${styles.number}`}>{formatAqi(day.aqi)}</span>
              <span className={styles.unit}>AQI</span>
            </div>

            {/* A bar rather than a badge: the same visual language as the
                scale strip above, so the three panels read as one ruler. */}
            <div className={styles.meter} aria-hidden="true">
              <div
                className={styles.fill}
                style={{
                  width: `${Math.min((day.aqi / SCALE_CEILING) * 100, 100)}%`,
                  background: day.color,
                }}
              />
            </div>

            <p className={styles.category}>
              <span className={styles.swatch} style={{ background: day.color }} />
              {day.category}
            </p>

            <p className={styles.model}>Predicted by {humanizeModel(day.model_type)}</p>
          </article>
        ))}
      </div>

      <p className={styles.footnote}>
        Values are daily averages on the US EPA scale, the same measure the models were trained on.
        {scale.length > 0 && " Bands are shown in full further down the page."}
      </p>
    </section>
  );
}
