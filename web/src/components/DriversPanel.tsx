"use client";

import { useState } from "react";

import { formatNumber, horizonLabel, humanizeModel } from "@/lib/format";
import type { HorizonDrivers } from "@/lib/types";
import styles from "./DriversPanel.module.css";

/**
 * SHAP contributions, precomputed per horizon by the pipeline.
 *
 * The values are already in the payload, so switching horizons is instant and
 * needs no server - the only reason this is a client component is the toggle.
 */
export function DriversPanel({ drivers }: { drivers: HorizonDrivers[] }) {
  const [horizon, setHorizon] = useState(drivers[0]?.horizon ?? 1);
  const active = drivers.find((entry) => entry.horizon === horizon) ?? drivers[0];

  if (!active) return null;

  const largest = Math.max(
    ...active.features.map((feature) => Math.abs(feature.contribution ?? 0)),
    1,
  );

  return (
    <section className="shell section" id="drivers">
      <div className="section-head">
        <div>
          <p className="kicker">Explainability</p>
          <h2>Why this forecast?</h2>
        </div>

        <div className={`seg ${styles.toggle}`} role="group" aria-label="Forecast day">
          {drivers.map((entry) => (
            <label key={entry.horizon} className="seg-opt">
              <input
                type="radio"
                name="horizon"
                value={entry.horizon}
                checked={entry.horizon === horizon}
                onChange={() => setHorizon(entry.horizon)}
              />
              Day {entry.horizon}
            </label>
          ))}
        </div>
      </div>

      <p className="lede">
        How much each input moved the {horizonLabel(active.horizon).toLowerCase()} forecast, in AQI
        points, relative to a typical recent day. Computed with SHAP against the{" "}
        {humanizeModel(active.model_type)} model that actually produced the number.
      </p>

      {active.unavailable ? (
        <p className={styles.unavailable}>
          No explanation is available for this model family.
          <span className={styles.reason}>{active.unavailable}</span>
        </p>
      ) : (
        <ol className={styles.list}>
          {active.features.map((feature) => {
            const contribution = feature.contribution ?? 0;
            const width = (Math.abs(contribution) / largest) * 50;
            const pushesUp = contribution >= 0;

            return (
              <li key={feature.feature} className={styles.row}>
                <div className={styles.labels}>
                  <span className={styles.name}>{feature.label}</span>
                  <span className={styles.observed}>{formatNumber(feature.value)}</span>
                </div>

                <div className={styles.track}>
                  <span className={styles.centre} aria-hidden="true" />
                  <span
                    className={pushesUp ? styles.barUp : styles.barDown}
                    style={{
                      width: `${width}%`,
                      left: pushesUp ? "50%" : `${50 - width}%`,
                    }}
                  />
                </div>

                <span className={pushesUp ? styles.valueUp : styles.valueDown}>
                  {pushesUp ? "+" : "-"}
                  {formatNumber(Math.abs(contribution))}
                </span>
              </li>
            );
          })}
        </ol>
      )}

      {!active.unavailable && (
        <p className={styles.axis}>
          <span>Pushes the forecast down</span>
          <span>Pushes it up</span>
        </p>
      )}
    </section>
  );
}
