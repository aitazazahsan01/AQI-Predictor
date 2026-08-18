import type { City } from "@/lib/types";
import styles from "./SiteFooter.module.css";

const REPO_URL = "https://github.com/aitazazahsan01/AQI-Predictor";

export function SiteFooter({ city }: { city: City }) {
  return (
    <footer className={styles.footer}>
      <div className={`shell ${styles.inner}`}>
        <div className={styles.about}>
          <p className={styles.brand}>Pearls AQI Predictor</p>
          <p className={styles.blurb}>
            A serverless machine learning pipeline forecasting air quality for {city.name} three
            days ahead. Built as a university project, with the failures documented alongside the
            results.
          </p>
        </div>

        <nav className={styles.links} aria-label="Further reading">
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            Source code
          </a>
          <a href={`${REPO_URL}/blob/main/REPORT.md`} target="_blank" rel="noreferrer">
            Project report
          </a>
          <a href={`${REPO_URL}/blob/main/EDA.md`} target="_blank" rel="noreferrer">
            Exploratory analysis
          </a>
          <a href="https://open-meteo.com/" target="_blank" rel="noreferrer">
            Data: Open-Meteo
          </a>
        </nav>
      </div>

      <div className={`shell ${styles.legal}`}>
        <p>
          Forecasts are model estimates, not official measurements, and should not be used for
          medical decisions.
        </p>
        <p className="mono-nums">
          {city.latitude.toFixed(4)}, {city.longitude.toFixed(4)} &middot; MIT licensed
        </p>
      </div>
    </footer>
  );
}
