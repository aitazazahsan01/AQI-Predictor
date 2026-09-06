# Pearls AQI Predictor — Project Report

**Author:** Muhammad Aitazaz Ahsan · NUST · Summer 2026
**Repository:** https://github.com/aitazazahsan01/AQI-Predictor
**City:** Islamabad, Pakistan · **Horizon:** 3 days

---

## 1. What was built

An end-to-end, serverless machine learning system that forecasts Islamabad's Air Quality Index three days ahead. It collects data hourly, engineers features daily, retrains itself daily, and serves forecasts with explanations and health alerts through a web dashboard — with no server to administer anywhere in the stack.

All nine planned modules are implemented, plus a public website built after them:

| # | Module | What it does |
|:--:|---|---|
| M1 | Hourly ingestion | Fetches live AQI, 6 pollutants and 6 weather variables into `aqi_hourly_raw` |
| M2 | Daily feature engineering | Collapses 24 hourly rows into one engineered daily row (27 features) |
| M3 | Historical backfill | Loads ~4 years of history so training doesn't wait a year for data |
| M4 | Training pipeline | Trains 6 model families, selects the best per forecast horizon |
| M5 | CI/CD automation | Scheduled GitHub Actions for ingestion, aggregation, training and tests |
| M6 | Dashboard | Streamlit app: forecast, trends, explanations, alerts |
| M7 | Explainability | SHAP attribution per forecast |
| M8 | Alerts | EPA-scale categorisation with hazardous-air warnings |
| M9 | EDA + report | Generated data analysis ([EDA.md](EDA.md)) and this document |
| M10 | Public website | Statically exported Next.js front end, fed by a JSON snapshot the pipeline publishes |

**190 unit tests** cover the feature logic, metrics, alert thresholds, inference contracts, snapshot serialisation, feature-view reads and data-quality guards.

---

## 2. Architecture

```
Open-Meteo (AQI + pollutants + weather)  ─┐
AQICN (station reading, display only)    ─┴─►  M1 hourly ingestion
                                                      │
                                                      ▼
                                          Hopsworks: aqi_hourly_raw
                                                      │
                                    M2 daily aggregation + feature engineering
                                                      │
                                                      ▼
                                        Hopsworks: aqi_daily_features
                                                      │
                                        aqi_daily_features_view
                                              │              │
                          M4 training ────────┘              └──── M6 dashboard
                                │                                        ▲
                                ▼                                        │
                     Hopsworks Model Registry ───────────────────────────┤
                                │                                        │
                                └──► M10 snapshot export ──► forecast.json
                                                                  │
                                                                  ▼
                                                          Next.js website
```

GitHub Actions drives ingestion hourly and aggregation-then-training daily. Both front ends are read-only consumers: the dashboard reads the feature store and registry live, the website reads a JSON snapshot the pipeline publishes for it.

Training and inference read through a **Feature View** (`aqi_daily_features_view`) rather than the feature group directly, so which columns constitute "the training data" is declared once on the read side instead of re-decided by each consumer.

### Key design decision: pure feature functions

All feature engineering lives in pure functions (`src/features/feature_engineering.py`) with no network or database access. The live daily job and the historical backfill call **the same functions**.

This structurally prevents **training/serving skew** — where a model learns from features computed one way, then receives features computed slightly differently in production, and silently degrades in ways that are very hard to trace. Because both paths share one implementation, they cannot diverge. A unit test asserts the contract directly.

---

## 3. Data

### Sources and why

| Source | Role | Rationale |
|---|---|---|
| **Open-Meteo Air Quality** | Training target + pollutant features | Provides `us_aqi` directly **and** a 4-year historical archive — the only free source that supports both live ingestion and backfill |
| **Open-Meteo Weather** | Weather features | Same provider, so live and historical stay methodologically consistent |
| **AQICN** | Display only | Official station readings shown as a real-world reference; deliberately **not** training data |

The brief suggested AQICN or OpenWeather. Neither works for backfill: AQICN's free tier has no deep historical endpoint, so it cannot produce training data at all.

That decision was vindicated during development. AQICN's only Islamabad station (*Islamabad US Embassy*) **had not reported since February 2026** — months of silence. Had it been the training source, the pipeline would have been broken from day one. The client now treats readings older than 6 hours as unavailable, because showing nothing beats showing a months-old number as current.

### Dataset

- **1,469 daily rows**, 2022-08-05 → present (~4 years)
- **1,454/1,454 complete 24-hour days** at the time of backfill — zero gaps
- Mean AQI **112** ("Unhealthy for Sensitive Groups"), range 14–180

### Features (27)

| Family | Features |
|---|---|
| Daily aggregates | `aqi_mean/max/min`, 6 pollutant means, 4 weather means, `precipitation_sum` |
| Calendar | `day_of_week`, `day_of_month`, `month`, `is_weekend` |
| Trend | `aqi_lag1/2/3/7`, `aqi_change_rate`, `aqi_roll3_mean`, `aqi_roll7_mean`, `aqi_roll3_std` |
| Data quality | `hours_observed` |

Every feature describes information **known as of that date**. Forecast targets are derived at training time by shifting `aqi_mean`, deliberately not stored — so the newest row stays valid for live prediction even though its "tomorrow" doesn't exist yet.

---

## 4. Exploratory analysis

Full output in [EDA.md](EDA.md). The findings that mattered:

**Seasonality dominates.** January averages AQI 144; April averages 84 — a 1.7× swing. This is why `month` is a feature: without it, a model cannot know whether 120 is unusually bad for the season or unusually good.

**Weekday effects are negligible.** The entire Monday–Sunday spread is 1.4 AQI points (1.2% of the mean). `is_weekend` earns its place but was never going to be a strong signal — worth knowing before over-interpreting it.

**Weather correlates as physics predicts.** Wind speed −0.29, rainfall −0.08: wind disperses pollution, rain washes it out. This is the justification for fetching weather at all. PM2.5's +0.88 correlation is closer to a definition than a discovery, since AQI is largely derived from pollutant concentrations.

**Predictability decays fast — and this explains everything downstream:**

| Lag | Autocorrelation |
|---|--:|
| 1 day | 0.841 |
| 2 days | 0.678 |
| 3 days | 0.600 |
| 7 days | 0.481 |

**Typical daily movement is 7.8 AQI points** (median absolute change). This sets the noise floor: a day-1 model with RMSE ~9 is already operating at roughly the natural limit of the data.

---

## 5. Modelling

### Candidates

Six families, spanning the statistical-to-deep-learning range the brief asked for:

| Model | Family |
|---|---|
| Persistence ("tomorrow = today") | Baseline |
| Ridge Regression | Linear |
| Random Forest | Tree ensemble |
| XGBoost | Gradient boosting |
| SARIMAX | Classical time series |
| LSTM | Deep learning |

### Evaluation method

- **Chronological split**, never shuffled — the most recent 90 days are the test set. A random split would let the model train on data from after the days it is tested on, inflating scores enormously and measuring nothing.
- **RMSE, MAE, R²** reported; **RMSE selects** the winner because large misses matter disproportionately — being 60 points wrong on a hazardous day is far worse than 6 points wrong ten times.
- **Per-horizon selection**, because the best model for day 1 is not the best for day 3.

### The persistence baseline

Included deliberately. Air quality is strongly autocorrelated, so a model can post a respectable R² while being *worse than repeating yesterday's number*. Reporting every winner's lift over this baseline is the difference between an honest evaluation and a flattering one.

### Results

Trained on 1,469 days, scored on a 90-day chronological hold-out:

| Horizon | Winner | RMSE | MAE | R² | Lift over baseline |
|:--:|---|--:|--:|--:|--:|
| Day 1 | **Ridge** | 8.87 | 6.72 | 0.843 | **+29.1%** |
| Day 2 | **XGBoost** | 17.47 | 14.38 | 0.389 | **+8.5%** |
| Day 3 | **Random Forest** | 20.72 | 16.98 | 0.141 | **+12.9%** |

Full comparison:

```
 horizon                model      family      rmse       mae        r2
       1 persistence_baseline    baseline 12.519142  9.841204  0.686369
       1                ridge     sklearn  8.871098  6.719025  0.842520
       1        random_forest     sklearn  9.659741  7.452965  0.813275
       1              xgboost     sklearn  9.922773  7.674252  0.802968
       1              sarimax statsmodels 12.510215  9.944189  0.686816
       2 persistence_baseline    baseline 19.091922 14.384722  0.270593
       2                ridge     sklearn 17.718916 13.842527  0.371732
       2        random_forest     sklearn 17.631019 14.098580  0.377950
       2              xgboost     sklearn 17.471751 14.376156  0.389138
       2              sarimax statsmodels 19.096155 14.467750  0.270270
       3 persistence_baseline    baseline 23.778795 18.196296 -0.131489
       3                ridge     sklearn 21.519701 17.314082  0.073292
       3        random_forest     sklearn 20.717934 16.980757  0.141059
       3              xgboost     sklearn 21.317890 17.646211  0.090592
       3              sarimax statsmodels 23.713939 18.262940 -0.125325
```

### Interpretation

**A different model wins at each horizon**, which directly validates per-horizon selection — a single global choice would be wrong two-thirds of the time.

**The simplest model wins day 1.** Ridge beats both tree ensembles. At one day out the relationship is close to linear (tomorrow ≈ today plus adjustments), and the flexible models slightly overfit. "More sophisticated" is not automatically better.

**Accuracy degrades sharply and honestly.** R² falls 0.84 → 0.39 → 0.14. This tracks the autocorrelation decay measured in the EDA almost exactly. Day-3 AQI is genuinely hard; RMSE ~21 can straddle a health-category boundary. Any comparable project claiming R² > 0.9 at three days is almost certainly leaking.

**The baseline goes negative at day 3** (R² −0.13) — "tomorrow equals today" becomes worse than guessing the long-run average. The model still delivers +12.9% over it.

**SARIMAX ≈ persistence** at every horizon. Pure univariate time-series modelling extracts about as much as assuming no change; the useful signal lives in the pollutant and weather features SARIMAX deliberately doesn't see.

---

## 6. Problems encountered and how they were handled

Three of these produced results that looked fine and were not.

### 6.1 A data leak that made a model look good

The most instructive bug in the project. SARIMAX initially scored R² = −8.4, which was an unfair comparison: the tabular models receive each test day's real features (including yesterday's AQI), while SARIMAX was forecasting all 90 test days blind.

The fix — walk-forward evaluation — made SARIMAX suddenly **win** at days 2 and 3. That was the actual bug. The tell was that it scored *better at 3 days than at 1 day*, and was nearly flat across horizons. Forecasting further ahead cannot be easier.

The cause: walking forward over the *shifted target* series fed the model the actual AQI from h days later, so it was performing 1-day forecasts wearing a 3-day label. It now models the base series and forecasts genuine h-step-ahead values, and degrades properly with distance.

**Lesson:** the suspicious result was the *good* one. Bad results get investigated; good results get accepted. That asymmetry is how leaks survive into published work. A regression test now pins this down.

### 6.2 An API that returns HTTP 200 for missing data

Open-Meteo returns a normal-looking 200 response with the correct row count — and every value `null` — for dates before its archive begins. Without a guard, backfilling from 2020 would have loaded ~20,000 rows of pure nulls. Nothing would crash; the data would simply be garbage, discovered much later as unexplained poor model performance.

Guarded by clamping the start date to 2022-08-05 and dropping null-AQI rows.

### 6.3 Forecasts stored as observations

Open-Meteo fills the remaining hours of the *current* day with forecast values. Backfilling through "today" would store predictions as measured fact — and land them on the newest row, the one the model leans on most for lag features. The backfill now defaults to ending yesterday.

### 6.4 Stale data presented as current

The dashboard initially read whatever the feature store held, which was a single leftover row from an early pipeline test. It displayed a month-old AQI as "latest". Stored features are now rejected if empty, shorter than a lag window, or more than 3 days old, with a visible note in the UI naming the data source.

### 6.5 Environment and network constraints

- **`hopsworks` cannot pip-install on native Windows** (`pyjks` → `twofish` needs a C compiler, no prebuilt wheel). Resolved by developing inside WSL, which also matches the Linux CI runners.
- **The development network permits only port 443.** Hopsworks *reads* work (REST over 443), but *writes* need HopsFS (8020) and Kafka (9092), both blocked. Diagnosed by testing raw TCP against an unrelated host to prove it was the network rather than Hopsworks, and confirmed identically from Windows to rule out a WSL quirk. The pipelines are designed to run on GitHub Actions, which has unrestricted egress, so this is an environment limitation rather than a system defect.

### 6.6 Two hosting platforms, two runtime refusals

Neither front end failed on its own code. Both were rejected by the platform hosting it, for reasons unrelated to whether the application worked.

**Streamlit Cloud** defaults to a very recent Python (3.14 at the time). TensorFlow publishes no wheels for it, so dependency resolution failed outright and nothing installed. TensorFlow is only needed to *train* the LSTM, never to serve a forecast, so the fix was to split the dependency file in two: `requirements.txt` for the pipelines, dashboard and tests, and `requirements-train.txt` adding TensorFlow for the training job alone. The LSTM is skipped automatically wherever TensorFlow is absent, so training still works under the base requirements — it just compares five candidates instead of six.

**Vercel** compiled and exported the site successfully, then refused to publish it: the pinned Next.js version carried a published CVE. Upgrading within the same major line was not enough — the advisory covered the entire 15.x range and named a major upgrade as the only remedy, with two transitive advisories in `postcss` and `sharp` resolving the same way. Moving to Next 16 cleared all three.

**Lesson:** a hosted platform pins the runtime and the security floor; the project does not. Both problems were invisible locally, where the interpreter was older and no advisory gate existed, and both surfaced only at deploy time. The defence is the same in each case — keep the heavy or fast-moving dependency out of the path that has to run in production.

### 6.7 The pipeline was quietly losing a day at a time

Building the backtest surfaced a defect nothing else had: **8 of 53 recent days were missing from the feature store** — 23, 25, 27, 28, 29, 31 August and 2, 3 September.

Nothing had failed loudly. `run_daily_aggregation.py` computed *yesterday* and wrote only that row, so a scheduled run that never fired left a permanent hole. GitHub skips scheduled workflow runs routinely under load, and each skip cost exactly one day, forever. Every missing feature day mapped precisely to a day the daily workflow had not run.

The damage compounds. A missing day is not just one absent training row: it breaks the lag and rolling features of the days *after* it, and it removes up to three forecasts from any evaluation, because a forecast can only be scored against a day that was recorded.

The aggregation now rewrites a trailing seven-day window (`--catch-up 7`) rather than a single day. The feature group is keyed on `(city, date)`, so re-inserting a day that already exists is an upsert — the repair costs nothing and a skipped run heals itself on the next one.

**Lesson:** the failure was invisible precisely because the job succeeded. It did exactly what it was told, and what it was told was subtly wrong. A pipeline that writes only the newest record silently depends on never missing a run — an assumption no scheduler honours. It surfaced only because something finally tried to *use* the history rather than append to it.

---

## 7. Explainability

SHAP attributes each forecast across its inputs, turning "tomorrow will be 150" into a ranked account of why. The explainer is matched to the winning model family, since the exact methods are family-specific: `TreeExplainer` for ensembles, `LinearExplainer` for Ridge, and a hard-sampled `KernelExplainer` otherwise.

A real day-1 explanation:

```
PM2.5 (56.9)              increases the forecast by 37.3 AQI points
Today's peak AQI (208.0)  increases the forecast by 13.7 AQI points
Today's average AQI (160) decreases the forecast by  6.8 AQI points
7-day average AQI (128.4) increases the forecast by  3.0 AQI points
```

PM2.5 dominating is consistent with its +0.88 correlation in the EDA — the explanations agree with the data rather than contradicting it, which is the basic sanity check for any attribution method.

---

## 8. Alerts

A single module (`src/alerts/aqi_scale.py`) owns the EPA breakpoints, colours and health advice, so the dashboard and any future notifier can never disagree about where "unhealthy" starts.

- **≥ 151** ("Unhealthy") → warning
- **≥ 201** ("Very Unhealthy") → critical

An alert fires if **any** of the next three days crosses the threshold — advance notice is the entire point. Missing values never raise an alert, and negative model extrapolations clamp to "Good" rather than reading as unknown.

---

## 9. The public website

The Streamlit dashboard is a live consumer: it boots a Python process, holds
Hopsworks credentials, loads models and runs SHAP on request. That is the right
shape for an internal tool and the wrong shape for a public one.

So the website inverts it. The pipeline publishes a single JSON snapshot; the
site is a static Next.js export that draws it and nothing else.

```
GitHub Actions (holds the secrets)        The website (holds none)
Hopsworks -> models -> SHAP
         └-> forecast.json  ------------> renders forecast.json
```

Three properties follow, and each was the reason for the choice:

- **The trust boundary does not move.** Credentials stay inside GitHub Actions,
  where they already were. There is no browser-side key, no proxy, no API to
  secure.
- **Skew remains impossible.** The published numbers come from the same
  `load_models` -> `build_forecast` -> `explain_prediction` path the dashboard
  uses. A TypeScript reimplementation of feature loading or SHAP would have
  reintroduced exactly the divergence section 2 exists to prevent - and neither
  the Hopsworks client nor SHAP exists in JavaScript anyway.
- **Hosting is a folder of HTML.** Deployed on Vercel, which redeploys from the
  push webhook that the nightly snapshot commit already produces. No deploy
  hook, no scheduled build, no runtime.

The cost is that the site is only as fresh as its last build. For a forecast
that regenerates once a night, that is not a cost.

Two guards protect it. `export_web_data.py` refuses to publish an empty feature
frame or a model-less snapshot, so a bad run leaves the previous good snapshot
serving rather than replacing it with nothing. And the front end fails its
build on a `schema_version` mismatch rather than shipping a page that silently
renders half its panels.

The interface is built on **Modernist**, a design system whose constraints
happened to suit the content: flat, ruled, flush-left, with the accent colour
permitted to run as a full field exactly once per page. That one place is the
health alert - the loudest element on the page is loud because the content is.
The EPA category colours are the deliberate exception to the palette: they are
a published standard, and recolouring them to match a brand would misinform the
reader.

---

## 10. Measured in operation

Every accuracy figure above comes from a chronological hold-out: fit on old days, score on recent ones. That is the correct way to *select* a model. It is not the same as measuring one, because it is computed once, by the process that chose the model, on data that was already on disk.

The deployed system makes a different claim every night, in public, about days that have not happened. Those claims were never checked — so I checked them.

No new infrastructure was needed. Each nightly run had already committed a snapshot recording what the system predicted, and later runs recorded what the air actually did, so the git history of `web/public/data/forecast.json` is a forecast ledger. `scripts/run_backtest.py` reads every historical version out of git, joins each forecast to the day it targeted, and writes [BACKTEST.md](BACKTEST.md). A forecast is therefore scored against an observation that did not exist when it was made — there is no way to leak.

The persistence baseline is reconstructed the same way, from the last AQI observed at the moment each forecast was issued: genuinely what a person could have said for free, at the time, with no model.

| Horizon | n | RMSE (operational) | RMSE (hold-out) | Skill vs persistence |
|:--:|--:|--:|--:|--:|
| +1d | 6 | 15.02 | 9.13 | −0.1% |
| +2d | 7 | 14.89 | 17.76 | −2.3% |
| +3d | 5 | 14.86 | 20.76 | −16.8% |

Read carefully, because the honest reading is uncomfortable and the sample is small.

**Day 1 is roughly two-thirds worse in operation than on the hold-out** (15.02 vs 9.13), and dead level with persistence. On the hold-out, day 1 was the system's strongest result and beat persistence by 29%. That gap is the entire reason this measurement was worth building: the number that was quoted everywhere is the number that degraded most.

Days 2 and 3 look *better* than their hold-out figures, which is not evidence of anything good — it reflects a quiet fortnight rather than improved modelling. Day 3 still lost to persistence by 17%.

Every horizon shows negative bias (−3.7 to −7.7), meaning the system consistently forecast **cleaner air than arrived** across this window. A one-directional error is more interesting than a large one, because it suggests something correctable rather than irreducible noise.

The caveats are load-bearing:

- **n is 6, 7 and 5.** A handful of unusual days moves these numbers substantially. This is a signal that something is worth investigating, not a verdict on the models.
- **The sample is small partly because of the bug in §6.7.** Eight missing days removed up to 24 forecasts from the scoreable set. The next run of this backtest, after the catch-up fix has been live for a week, will rest on a materially larger sample.
- **The window is one stretch of late-summer weather**, not a range of conditions. The hold-out deliberately spans 90 days; this does not.

What it establishes is the thing a hold-out cannot: whether the deployed system, fed by the live pipeline, performs as its selection process promised. On this evidence, at day 1, it does not — and that is worth knowing.

---

## 11. Honest limitations

- **Day-3 forecasts are weak** (R² 0.14). The autocorrelation analysis shows the signal genuinely isn't there at that range; this is a property of the problem, not a fixable defect.
- **The LSTM competes nightly and has not won a horizon.** TensorFlow could not be installed on the development network, so the LSTM was never scored locally. It is installed in the training workflow (`requirements-train.txt`) and has competed in every scheduled run since 19 August; Ridge still wins days 1 and 2, and Random Forest day 3. Sequence modelling has not recovered anything at day 3 on this dataset — a real result, but one produced by defaults rather than by a tuned architecture.
- **The dashboard cannot serve an LSTM even if one wins.** TensorFlow is excluded from the base requirements because it has no wheels for the Python version the hosting platforms default to. The loaders degrade per horizon rather than failing outright, but that horizon would fall back rather than serve the winner.
- **Single city.** The schema, config and pipelines are city-agnostic (adding one is a four-line config change), but only Islamabad has been run.
- **The operational sample is small, and it disagrees with the hold-out.** Section 10 scores 18 resolved forecasts and finds day-1 RMSE of 15.02 against the hold-out's 9.13. Whether that is a real degradation or a short unlucky window cannot be settled at this sample size, and the honest position is that it is unresolved rather than explained.
- **No hyperparameter tuning.** Models use sensible defaults. Given day 1 is already near the noise floor and day 3 is signal-limited, tuning would likely yield marginal gains — but this is an assumption, not a measured result.
- **Backfilled "observations" are themselves reanalysis output**, not physical sensor readings. Open-Meteo's archive is model-based, so the ground truth is itself an estimate.

---

## 12. What I would do next

1. Re-run the backtest once the catch-up fix has been live for a few weeks, on a sample large enough to say whether the day-1 gap in section 10 is real.
2. Read off whether the LSTM, now competing in the scheduled runs, recovers anything at day 3 - and drop it from the candidate set if it does not, since it costs the most to train by a wide margin.
3. Predict *categories* rather than values at longer horizons — "will tomorrow be unhealthy?" is both more useful and more tractable than an exact number when R² is 0.14.
4. Add a second city to prove the multi-city path.
5. Add prediction intervals. A day-3 forecast of 120 ± 40 is more honest, and more useful, than a bare 120.

---

## 13. Conclusion

The system meets its objective: an automated, serverless pipeline that ingests data hourly, retrains daily, and serves explained 3-day AQI forecasts with health alerts. Every module is implemented and tested.

The results are modest where the data is modest. Day-1 forecasting works well (R² 0.84, RMSE 8.87 against a ~7.8-point natural noise floor). Day-3 forecasting is weak, and the exploratory analysis explains why rather than leaving it unexplained.

The most valuable outcome was not the accuracy figure but the evaluation discipline: a persistence baseline that reveals when machine learning is not earning its keep, a chronological split that refuses to flatter the model, and a caught data leak that would otherwise have produced an impressive and entirely false result.
