import { formatNumber, humanizeModel } from "@/lib/format";
import type { ModelSummary } from "@/lib/types";
import styles from "./ModelTable.module.css";

export function ModelTable({ models }: { models: ModelSummary[] }) {
  return (
    <section className="shell section" id="models">
      <div className="section-head">
        <div>
          <p className="kicker">Under the hood</p>
          <h2>The models serving each day</h2>
        </div>
        <p className={styles.note}>
          Six candidates are retrained nightly and compared on a held-out window that always sits
          after the training data. The lowest RMSE wins each horizon, so the three days are often
          served by different models.
        </p>
      </div>

      <div className="scroll-x">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Horizon</th>
              <th scope="col">Winning model</th>
              <th scope="col">RMSE</th>
              <th scope="col">MAE</th>
              <th scope="col">R&sup2;</th>
              <th scope="col">Features</th>
              <th scope="col">Loaded from</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model.horizon}>
                <th scope="row" className={styles.horizon}>
                  +{model.horizon}d
                </th>
                <td className={styles.model}>{humanizeModel(model.model_type)}</td>
                <td className="mono-nums">{formatNumber(model.metrics.rmse, 2)}</td>
                <td className="mono-nums">{formatNumber(model.metrics.mae, 2)}</td>
                <td className="mono-nums">{formatNumber(model.metrics.r2, 3)}</td>
                <td className="mono-nums">{model.n_features}</td>
                <td>
                  <span className={model.source === "registry" ? "tag tag-accent" : "tag tag-neutral"}>
                    {model.source === "registry" ? "Model Registry" : "Local bundle"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className={styles.footnote}>
        RMSE and MAE are in AQI points, so an RMSE of 9 means the typical miss is about nine points.
        Error grows with the horizon: three days out is genuinely harder than one, and the numbers
        say so rather than hiding it.
      </p>
    </section>
  );
}
