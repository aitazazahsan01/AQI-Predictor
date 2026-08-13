# Exploratory Data Analysis — Pearls AQI Predictor

_Generated 14 August 2026._

## Dataset

- **Source:** Open-Meteo
- **Rows:** 1,470 daily observations
- **Period:** 2022-08-05 to 2026-08-13 (4.0 years)
- **AQI:** mean 112.1, median 108.0, min 14.3, max 180.5
- **Missing daily AQI:** 0
- **Complete 24-hour days:** 1,470 / 1,470

## How often is the air actually bad?

| Category | Days | Share | |
|---|--:|--:|---|
| Good | 2 | 0.1% |  |
| Moderate | 586 | 39.9% | ████████████████████ |
| Unhealthy for Sensitive Groups | 698 | 47.5% | ████████████████████████ |
| Unhealthy | 184 | 12.5% | ██████ |
| Very Unhealthy | 0 | 0.0% |  |
| Hazardous | 0 | 0.0% |  |

**12.5%** of days reach 'Unhealthy' or worse — the days this project exists to warn about.

## Seasonal pattern

| Month | Mean AQI | Days | |
|---|--:|--:|---|
| Jan | 143.6 | 124 | ██████████████████████████████ |
| Feb | 108.0 | 113 | ███████████████████████ |
| Mar | 91.3 | 124 | ███████████████████ |
| Apr | 83.9 | 120 | ██████████████████ |
| May | 101.7 | 124 | █████████████████████ |
| Jun | 114.9 | 120 | ████████████████████████ |
| Jul | 122.6 | 124 | ██████████████████████████ |
| Aug | 114.1 | 133 | ████████████████████████ |
| Sep | 116.3 | 120 | ████████████████████████ |
| Oct | 100.9 | 124 | █████████████████████ |
| Nov | 117.4 | 120 | █████████████████████████ |
| Dec | 129.7 | 124 | ███████████████████████████ |

Worst month is **Jan** (144), best is **Apr** (84) — a **1.7x** swing across the year.

This is why `month` is a model feature: without it, a model has no way to know whether an AQI of 120 is unusually bad for the season or unusually good.

## Weekly pattern

| Day | Mean AQI | |
|---|--:|---|
| Mon | 112.1 | ██████████████████████████████ |
| Tue | 111.9 | ██████████████████████████████ |
| Wed | 112.0 | ██████████████████████████████ |
| Thu | 111.3 | ██████████████████████████████ |
| Fri | 112.5 | ██████████████████████████████ |
| Sat | 112.6 | ██████████████████████████████ |
| Sun | 112.5 | ██████████████████████████████ |

Spread across the week is only **1.4 AQI points** (1.2% of the weekly mean).

Weekday-versus-weekend traffic matters far less here than season does. Useful to know: `is_weekend` earns its place as a feature, but it was never going to be a strong one.

## What moves air quality?

| Feature | Correlation with AQI |
|---|--:|
| `pm2_5_mean` | +0.883 |
| `pm10_mean` | +0.742 |
| `so2_mean` | +0.586 |
| `co_mean` | +0.494 |
| `no2_mean` | +0.380 |
| `wind_speed_mean` | -0.286 |
| `humidity_mean` | +0.203 |
| `temp_mean` | -0.138 |
| `precipitation_sum` | -0.076 |
| `pressure_mean` | +0.056 |
| `o3_mean` | +0.021 |

The strongest pollutant link is `pm2_5_mean` (+0.88), which is expected — the AQI is largely *derived* from pollutant concentrations, so this is closer to a definition than a discovery.

The interesting column is weather, which is genuinely independent:

- `wind_speed_mean`: -0.286
- `humidity_mean`: +0.203
- `temp_mean`: -0.138
- `precipitation_sum`: -0.076
- `pressure_mean`: +0.056

Negative correlations for wind and rain match the physics — wind disperses pollution and rain washes it out. That is why weather is worth fetching at all.

## How predictable is it?

| Lag | Autocorrelation |
|---|--:|
| 1 day(s) | 0.841 |
| 2 day(s) | 0.678 |
| 3 day(s) | 0.600 |
| 7 day(s) | 0.481 |
| 14 day(s) | 0.397 |
| 30 day(s) | 0.235 |

Day-to-day correlation is strong (0.84) but decays quickly — by day 3 it is 0.60.

This single table explains the model results better than anything else: it is why the day-1 forecast reaches R² 0.84 while day 3 struggles past 0.14, and why a naive 'tomorrow equals today' baseline is hard to beat at day 1 and useless by day 3.

## Day-to-day swings

- Median absolute change: **7.8 AQI points**
- 90th percentile: **24.5**
- Largest single-day change: **78.9**

A typical day moves about 8 points. Any forecast with an error much below that is doing genuinely well; the day-1 model's RMSE of ~9 sits right around this natural day-to-day noise floor, which is roughly the best that can be expected.
