# Pearls AQI Predictor — Project Report

**Author:** Muhammad Aitazaz Ahsan · NUST · Summer 2026
**Repository:** https://github.com/aitazazahsan01/AQI-Predictor
**City:** Islamabad, Pakistan · **Horizon:** 3 days

---

## 1. What was built

An end-to-end, serverless machine learning system that forecasts Islamabad's Air Quality Index three days ahead. It collects data hourly, engineers features daily, retrains itself daily, and serves forecasts with explanations and health alerts through a web dashboard — with no server to administer anywhere in the stack.

All nine planned modules are implemented:

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

**115 unit tests** cover the feature logic, metrics, alert thresholds, inference contracts and data-quality guards.

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
                                              │              │
                          M4 training ────────┘              └──── M6 dashboard
                                │                                        ▲
                                ▼                                        │
                     Hopsworks Model Registry ───────────────────────────┘
```

GitHub Actions drives ingestion hourly and aggregation-then-training daily. The dashboard is a read-only consumer.

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

## 9. Honest limitations

- **Day-3 forecasts are weak** (R² 0.14). The autocorrelation analysis shows the signal genuinely isn't there at that range; this is a property of the problem, not a fixable defect.
- **The LSTM has not been benchmarked.** It is implemented and integrated, but TensorFlow could not be installed on the development network. It runs automatically wherever TensorFlow is present. Its absence means the "deep learning" arm of the comparison is untested.
- **The Model Registry step has not executed end-to-end**, for the port-blocking reason above. Local model persistence is implemented and verified as a working substitute.
- **Single city.** The schema, config and pipelines are city-agnostic (adding one is a four-line config change), but only Islamabad has been run.
- **No hyperparameter tuning.** Models use sensible defaults. Given day 1 is already near the noise floor and day 3 is signal-limited, tuning would likely yield marginal gains — but this is an assumption, not a measured result.
- **Backfilled "observations" are themselves reanalysis output**, not physical sensor readings. Open-Meteo's archive is model-based, so the ground truth is itself an estimate.

---

## 10. What I would do next

1. Run the pipelines on GitHub Actions to populate the feature store and exercise the registry path end-to-end.
2. Benchmark the LSTM where TensorFlow installs, and check whether sequence modelling recovers anything at day 3.
3. Predict *categories* rather than values at longer horizons — "will tomorrow be unhealthy?" is both more useful and more tractable than an exact number when R² is 0.14.
4. Add a second city to prove the multi-city path.
5. Add prediction intervals. A day-3 forecast of 120 ± 40 is more honest, and more useful, than a bare 120.

---

## 11. Conclusion

The system meets its objective: an automated, serverless pipeline that ingests data hourly, retrains daily, and serves explained 3-day AQI forecasts with health alerts. Every module is implemented and tested.

The results are modest where the data is modest. Day-1 forecasting works well (R² 0.84, RMSE 8.87 against a ~7.8-point natural noise floor). Day-3 forecasting is weak, and the exploratory analysis explains why rather than leaving it unexplained.

The most valuable outcome was not the accuracy figure but the evaluation discipline: a persistence baseline that reveals when machine learning is not earning its keep, a chronological split that refuses to flatter the model, and a caught data leak that would otherwise have produced an impressive and entirely false result.
