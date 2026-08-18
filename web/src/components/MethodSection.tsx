import { Brain, Database, MonitorPlay, Timer } from "lucide-react";

import { formatTimestamp } from "@/lib/format";
import styles from "./MethodSection.module.css";

const STEPS = [
  {
    icon: Timer,
    title: "Ingest, hourly",
    body: "A scheduled job pulls air quality and weather for Islamabad from Open-Meteo and appends the raw hourly rows to a Hopsworks feature group.",
  },
  {
    icon: Database,
    title: "Engineer, daily",
    body: "Those hours are aggregated into one daily row: means and extremes, lags, rolling windows, change rates and calendar features. One set of pure functions does this for both live and historical data.",
  },
  {
    icon: Brain,
    title: "Train, nightly",
    body: "Six candidates - persistence, ridge, random forest, XGBoost, SARIMAX and an LSTM - are refit and scored on a chronological hold-out. The winner per horizon goes to the Model Registry.",
  },
  {
    icon: MonitorPlay,
    title: "Publish",
    body: "The pipeline runs the models, computes SHAP explanations and writes this page's data as a single JSON file. The site is static: no server, no database, no credentials in the browser.",
  },
];

interface Props {
  featureSource: string;
  observedDays: number;
  generatedAt: string;
}

export function MethodSection({ featureSource, observedDays, generatedAt }: Props) {
  return (
    <section className="shell section" id="method">
      <div className="section-head">
        <div>
          <p className="kicker">How it works</p>
          <h2>A pipeline, not a page</h2>
        </div>
        <p className={styles.note}>
          Everything runs on scheduled GitHub Actions. Nothing here is generated on request.
        </p>
      </div>

      <div className={styles.steps}>
        {STEPS.map((step, index) => {
          const Icon = step.icon;
          return (
            <article key={step.title} className={styles.step}>
              <div className={styles.stepHead}>
                <span className={styles.index}>{String(index + 1).padStart(2, "0")}</span>
                <Icon size={18} strokeWidth={2} aria-hidden="true" />
              </div>
              <h3 className={styles.stepTitle}>{step.title}</h3>
              <p className={styles.stepBody}>{step.body}</p>
            </article>
          );
        })}
      </div>

      {/* Provenance, stated rather than implied: a reader should be able to
          tell whether they are looking at stored features or a live refetch. */}
      <dl className={styles.provenance}>
        <div>
          <dt className="label">Feature source</dt>
          <dd>{featureSource}</dd>
        </div>
        <div>
          <dt className="label">Days of history</dt>
          <dd className="mono-nums">{observedDays.toLocaleString("en-GB")}</dd>
        </div>
        <div>
          <dt className="label">Snapshot written</dt>
          <dd className="mono-nums">{formatTimestamp(generatedAt)}</dd>
        </div>
      </dl>
    </section>
  );
}
