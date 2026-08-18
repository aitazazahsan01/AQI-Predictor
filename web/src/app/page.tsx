import { AlertBanner } from "@/components/AlertBanner";
import { DriversPanel } from "@/components/DriversPanel";
import { ForecastGrid } from "@/components/ForecastGrid";
import { HeroNow } from "@/components/HeroNow";
import { MethodSection } from "@/components/MethodSection";
import { ModelTable } from "@/components/ModelTable";
import { ScaleTable } from "@/components/ScaleTable";
import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";
import { TrendChart } from "@/components/TrendChart";
import { loadSnapshot } from "@/lib/data";

export default async function Home() {
  const snapshot = await loadSnapshot();

  return (
    <>
      <SiteHeader generatedAt={snapshot.generated_at} />

      <main>
        <AlertBanner alert={snapshot.alert} />

        <HeroNow
          cityName={snapshot.city.name}
          latest={snapshot.latest}
          scale={snapshot.scale}
          observedDays={snapshot.observed_days}
        />

        <ForecastGrid forecast={snapshot.forecast} scale={snapshot.scale} />

        <section className="shell section" id="trend">
          <div className="section-head">
            <div>
              <p className="kicker">History</p>
              <h2>Where the forecast comes from</h2>
            </div>
            <p className="lede">
              The last {snapshot.history.length} observed days, then the three predicted ones.
            </p>
          </div>
          <TrendChart history={snapshot.history} forecast={snapshot.forecast} />
        </section>

        {snapshot.drivers.length > 0 && <DriversPanel drivers={snapshot.drivers} />}

        <ModelTable models={snapshot.models} />

        <ScaleTable scale={snapshot.scale} />

        <MethodSection
          featureSource={snapshot.feature_source}
          observedDays={snapshot.observed_days}
          generatedAt={snapshot.generated_at}
        />
      </main>

      <SiteFooter city={snapshot.city} />
    </>
  );
}
