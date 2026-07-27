# Pearls AQI Predictor 🌫️

Forecasting Islamabad's Air Quality Index (AQI) for the next 3 days, using a 100% serverless pipeline: automated hourly data collection → daily feature engineering → daily model retraining → a live dashboard.

- **New here?** Read [Project_Explanation.md](Project_Explanation.md) for a plain-language walkthrough of what this is and why.
- **Building this?** Read [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full technical spec — module breakdown, data schemas, and API contracts.

## Status

🚧 **In progress.** This section is updated as each module lands.

| Module | Status |
|---|---|
| Repo + docs | ✅ Done |
| M1 — Hourly raw data ingestion | ⬜ Not started |
| M2 — Daily feature engineering | ⬜ Not started |
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

Setup instructions (environment variables, Hopsworks/AQICN account requirements, how to run each pipeline locally) will be added here as those pieces are built.

## License

TBD.
