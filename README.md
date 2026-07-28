# Pearls AQI Predictor 🌫️

Forecasting Islamabad's Air Quality Index (AQI) for the next 3 days, using a 100% serverless pipeline: automated hourly data collection → daily feature engineering → daily model retraining → a live dashboard.

- **New here?** Read [Project_Explanation.md](Project_Explanation.md) for a plain-language walkthrough of what this is and why.
- **Building this?** Read [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full technical spec — module breakdown, data schemas, and API contracts.

## Status

🚧 **In progress.** This section is updated as each module lands.

| Module | Status |
|---|---|
| Repo + docs | ✅ Done |
| M1 — Hourly raw data ingestion | ✅ Working — verified end-to-end against the live Hopsworks project (`aqi_hourly_raw` feature group) |
| M2 — Daily feature engineering | ✅ Working — `aqi_daily_features` feature group, verified on 10 days of real data (26 unit tests) |
| M3 — Historical backfill | ⬜ Not started |
| M4 — Training pipeline (multi-model) | ⬜ Not started |
| M5 — CI/CD automation (GitHub Actions) | ⬜ Not started |
| M6 — Streamlit dashboard | ⬜ Not started |
| M7 — SHAP explainability | ⬜ Not started |
| M8 — Hazardous AQI alerts | ⬜ Not started |
| M9 — EDA notebook + final report | ⬜ Not started |

## Tech stack

Python · Open-Meteo & AQICN APIs · Hopsworks (Feature Store + Model Registry) · scikit-learn · XGBoost · Statsmodels (SARIMAX) · TensorFlow (LSTM) · SHAP · Streamlit · GitHub Actions

(See [Project_Explanation.md](Project_Explanation.md#4-the-technology-stack--what-each-tool-is-and-why-we-chose-it) for what each of these is and why it was chosen.)

## Setup

**Note (Windows users):** the `hopsworks` package depends on `pyjks`/`twofish`, which needs a C compiler to build and has no prebuilt wheel for Windows. Either install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), or do what this project's development so far has used: run everything inside **WSL** (`wsl --install`, then any Ubuntu distro), where `pip install -r requirements.txt` works out of the box.

```bash
python3 -m venv .venv
source .venv/bin/activate   # WSL/Linux/macOS
pip install -r requirements.txt

cp .env.example .env   # then fill in AQICN_TOKEN, HOPSWORKS_API_KEY, HOPSWORKS_PROJECT_NAME

python scripts/run_feature_pipeline.py    # Module 1: fetch + store one hourly row
python scripts/run_daily_aggregation.py   # Module 2: aggregate hourly -> daily features
python -m pytest tests/                   # run unit tests
```

### Pipelines

| Script | Runs | What it does |
|---|---|---|
| `scripts/run_feature_pipeline.py` | hourly | Fetches current air quality + weather, writes one row to `aqi_hourly_raw`. |
| `scripts/run_daily_aggregation.py` | daily | Aggregates hourly rows into one engineered row per day in `aqi_daily_features` (lags, rolling stats, AQI change rate, calendar features). Accepts `--date YYYY-MM-DD` or `--all`. |

## License

TBD.
