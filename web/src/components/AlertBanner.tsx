import { CircleCheck, TriangleAlert } from "lucide-react";

import { formatAqi } from "@/lib/format";
import type { Alert } from "@/lib/types";
import styles from "./AlertBanner.module.css";

/**
 * The poster statement: the one place the accent runs as a full field.
 *
 * Modernist reserves that treatment for a single moment on a page, which makes
 * it exactly right for a health warning - it is loud because the content is,
 * not for decoration. When nothing is wrong the banner drops to a plain rule.
 */
export function AlertBanner({ alert }: { alert: Alert | null }) {
  if (!alert) {
    return (
      <div className={styles.clear}>
        <div className={`shell ${styles.clearInner}`}>
          <CircleCheck size={18} strokeWidth={2} aria-hidden="true" />
          <p className={styles.clearText}>
            <strong>No air quality alerts.</strong> Nothing in the next three days is forecast to
            reach unhealthy levels.
          </p>
        </div>
      </div>
    );
  }

  return (
    <aside className={`poster ${styles.alert}`} role="alert">
      <div className={`shell ${styles.inner}`}>
        <div className={styles.icon} aria-hidden="true">
          <TriangleAlert size={28} strokeWidth={2.5} />
        </div>

        <div className={styles.body}>
          <p className={styles.kicker}>
            {alert.level === "critical" ? "Health alert" : "Health warning"}
          </p>
          <h2 className={styles.headline}>{alert.category} air expected</h2>
          <p className={styles.days}>
            {alert.days.join(", ")} &middot; peaking at AQI {formatAqi(alert.worst_aqi)} on{" "}
            {alert.worst_day}
          </p>
          <p className={styles.advice}>{alert.advice}</p>
        </div>
      </div>
    </aside>
  );
}
