import type { ScaleBand } from "@/lib/types";
import styles from "./ScaleTable.module.css";

export function ScaleTable({ scale }: { scale: ScaleBand[] }) {
  return (
    <section className="shell section" id="scale">
      <div className="section-head">
        <div>
          <p className="kicker">Reference</p>
          <h2>What the numbers mean</h2>
        </div>
        <p className={styles.note}>
          US EPA breakpoints. One table in the codebase defines these, and the pipeline, the alerts
          and this page all read from it.
        </p>
      </div>

      <div className="scroll-x">
        <table className="table">
          <thead>
            <tr>
              <th scope="col" className={styles.swatchCol}>
                <span className="sr-only">Colour</span>
              </th>
              <th scope="col">Range</th>
              <th scope="col">Category</th>
              <th scope="col">Health advice</th>
            </tr>
          </thead>
          <tbody>
            {scale.map((band) => (
              <tr key={band.name}>
                <td className={styles.swatchCol}>
                  <span className={styles.swatch} style={{ background: band.color }} aria-hidden="true" />
                </td>
                <td className={styles.range}>
                  {band.lower}
                  {band.upper == null ? "+" : `–${band.upper}`}
                </td>
                <td className={styles.name}>
                  {band.name}
                  {band.alert_level !== "none" && (
                    <span className={styles.flag}>
                      {band.alert_level === "critical" ? "Alert" : "Warning"}
                    </span>
                  )}
                </td>
                <td className={styles.advice}>{band.advice}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
