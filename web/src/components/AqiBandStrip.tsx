import type { ScaleBand } from "@/lib/types";
import styles from "./AqiBandStrip.module.css";

/** The open-ended "Hazardous" band needs a finite width to be drawable. */
const SCALE_CEILING = 400;

interface Props {
  scale: ScaleBand[];
  value: number | null;
  label?: string;
}

/**
 * The EPA scale as a ruler, with a marker where the reading falls.
 *
 * Bands are drawn to their true numeric width rather than as equal blocks, so
 * the strip doubles as an honest picture of how wide "Good" is compared with
 * "Unhealthy" - the eye should not be told that a 50-point band and a
 * 100-point band are the same size.
 */
export function AqiBandStrip({ scale, value, label = "Current reading" }: Props) {
  const bands = scale.map((band) => ({
    ...band,
    width: ((band.upper ?? SCALE_CEILING) - band.lower + 1) / SCALE_CEILING,
  }));

  const position = value == null ? null : Math.min(value / SCALE_CEILING, 1);

  return (
    <figure className={styles.wrap}>
      <div
        className={styles.strip}
        role="img"
        aria-label={
          value == null
            ? "Air quality index scale"
            : `${label}: ${Math.round(value)} on a scale where 0 is good and above 300 is hazardous`
        }
      >
        {bands.map((band) => (
          <div
            key={band.name}
            className={styles.band}
            style={{ flexGrow: band.width, background: band.color }}
            title={`${band.name}: ${band.lower}${band.upper == null ? "+" : `-${band.upper}`}`}
          />
        ))}

        {position != null && (
          <div className={styles.marker} style={{ left: `${position * 100}%` }}>
            <span className={styles.markerValue}>{Math.round(value as number)}</span>
          </div>
        )}
      </div>

      <div className={styles.ticks} aria-hidden="true">
        {[0, 50, 100, 150, 200, 300].map((tick) => (
          <span key={tick} className={styles.tick} style={{ left: `${(tick / SCALE_CEILING) * 100}%` }}>
            {tick}
          </span>
        ))}
      </div>
    </figure>
  );
}
