# Pearls AQI Predictor — Project Explained (Plain Words)

This document explains, in plain language, what we're building, why each piece exists, and what you'll actually see/use at the end. For the detailed technical spec (schemas, exact API contracts), see [PROJECT_PLAN.md](PROJECT_PLAN.md). This file answers *"what is this and why"*; that file answers *"exactly how"*.

---

## 1. What is this project, really?

Think of it like a weather forecast, but for air pollution instead of rain. Every hour, our system quietly checks how polluted the air is in Islamabad and what the weather looks like, remembers that history, and uses it to teach a machine learning model to answer one question:

> **"Given what's happened recently, how polluted will the air be tomorrow, the day after, and the day after that?"**

It does this completely automatically — no server we have to babysit, no manual button-pressing. It fetches data on a timer, retrains itself on a timer, and shows the results on a website anyone can open.

"100% serverless" just means: we never rent or manage our own always-on computer. Everything runs on free, on-demand cloud services that wake up, do their job, and go back to sleep.

---

## 2. The big picture, in one flow

```
[Internet weather/pollution data]
        ↓ (fetched automatically, every hour)
[Our scripts turn raw numbers into "features" a model can learn from]
        ↓
[Hopsworks — a cloud filing cabinet for ML data and models]
        ↓
[Training scripts read that history and teach several models to predict AQI]
        ↓ (the best model per forecast day gets saved)
[Hopsworks Model Registry — stores the trained models]
        ↓
[A website (Streamlit dashboard) loads the latest data + models]
        ↓
[You open the website and see today's AQI + the next 3 days forecast]
```

Everything before the website runs on a schedule in the background (GitHub Actions, which is free automation GitHub provides). The website is the only part a human actually looks at.

---

## 3. Module by module — what each piece does and why

### Module 1 — Raw Data Ingestion *(runs every hour)*
**What it does:** Calls two outside services (Open-Meteo and AQICN) and asks, "what's the pollution and weather like in Islamabad right now?" Saves the answer.
**Why it exists:** A model can only be as good as its data. We need a continuous, reliable stream of real numbers — you can't predict tomorrow's air quality without knowing today's.
**Output / use case:** One new row of raw numbers (pollution levels + weather) added to storage every hour. Nobody looks at this directly — it's raw material for the next module.

### Module 2 — Daily Feature Engineering *(runs once a day)*
**What it does:** Takes that day's 24 hourly rows and boils them down into one smart daily summary — not just averages, but things like "how much worse is the air today compared to yesterday?" (the AQI *change rate*), "what's the 7-day trend?", and "is today a weekday or weekend?" (pollution patterns often differ).
**Why it exists:** Raw numbers alone are weak predictors. A model that also knows *trends* and *patterns* (is pollution rising? is it a weekday? what season is it?) predicts much better than one that only sees a single snapshot. This is the difference between showing a model "72" versus "72, up from 65 yesterday, on a Tuesday in July, still rising for the 3rd day straight."
**Output / use case:** One clean, information-rich row per day, ready to train a model on — this is the actual training data for everything downstream.

### Module 3 — Historical Backfill *(run once, manually, at the start)*
**What it does:** Same logic as Module 2, but instead of "today," it fetches 2–3 years of *past* weather/pollution data in one go and processes all of it at once.
**Why it exists:** Module 1 + 2 only start collecting data from the day we turn them on. But a model needs hundreds of past examples to learn from before it's useful — we can't wait 2 years for enough real-time data to accumulate. Backfilling instantly gives us years of "training examples" to learn from.
**Output / use case:** A large historical dataset — this is what actually makes the very first model good, before real-time data has had time to build up.

### Module 4 — Training Pipeline *(runs once a day)*
**What it does:** Reads all the historical daily rows, and tries several different modeling approaches — simple statistical ones and complex machine-learning/deep-learning ones — to see which one predicts AQI most accurately for each of the 3 forecast days. Keeps the best one for each day.
**Why it exists:** No single algorithm is always best. A simple method might do fine predicting tomorrow but fail badly predicting 3 days out, while a more complex model might be the opposite. We test a *range* (see §4 below) and let the evidence — not a guess — decide, which is exactly the "build from everything and choose the best" approach you asked for.
**Output / use case:** Three trained, tested models (one that's best at predicting Day 1, one for Day 2, one for Day 3), each saved with its accuracy scores so we always know how much to trust it.

### Module 5 — Automation (CI/CD) *(always running in the background)*
**What it does:** Tells GitHub, "run Module 1 every hour, run Module 2 and Module 4 every day, forever, without anyone touching a keyboard."
**Why it exists:** This is what makes the system actually "serverless" and self-sustaining. Without it, someone would have to manually run these scripts every single hour — impossible to keep up, and the whole point of the project is that it *doesn't* need a human in the loop.
**Output / use case:** You'll see this as a "Actions" tab on the GitHub repo with a history of green checkmarks — visible proof the system is alive and updating itself.

### Module 6 — The Dashboard (Web App) *(the only part people actually see)*
**What it does:** A website that loads the latest data and the trained models, and shows: today's actual air quality, the forecast for the next 3 days, how AQI has trended recently, and a warning banner if pollution is forecast to reach dangerous levels.
**Why it exists:** This is the actual deliverable — a human being (you, a classmate, an evaluator) needs a simple way to *see* the forecast without reading code or database tables.
**Output / use case:** A live, public webpage. Open it, pick "today," and see: "AQI is 145 (Unhealthy for Sensitive Groups) today, forecast to rise to 165 tomorrow" — in plain cards and charts.

### Module 7 — Explainability (SHAP)
**What it does:** For any given prediction, shows *which factors* pushed the number up or down — e.g., "today's forecast is high mainly because of low wind speed and high PM2.5 from yesterday."
**Why it exists:** A model that just spits out a number is a black box — nobody can trust or learn from a number they can't question. This makes the model's reasoning visible, which is also a specific requirement of the assignment.
**Output / use case:** A bar chart on the dashboard showing the top factors behind today's forecast — useful for a report, a demo, or just understanding *why* the model thinks what it thinks.

### Module 8 — Alerts
**What it does:** Checks every forecast number against standard air-quality danger thresholds and shows a clear red warning banner if any of the next 3 days crosses into "Unhealthy" or worse.
**Why it exists:** A forecast nobody notices is useless if it's genuinely dangerous. This turns "165" (just a number) into "⚠️ Unhealthy air expected tomorrow — sensitive groups should limit outdoor activity," which is the entire *point* of predicting AQI in the first place.
**Output / use case:** The visible warning banner on the dashboard whenever pollution is forecast to be genuinely bad.

### Module 9 — EDA & Report
**What it does:** A notebook exploring the data (what patterns exist? does pollution spike on certain days/seasons? which weather factors correlate with bad air?) and a written report documenting everything we built, learned, and achieved.
**Why it exists:** This is required deliverable #4 — the write-up that explains our reasoning and results to someone who didn't watch us build it (a grader, in this case).
**Output / use case:** Charts and findings that also inform which features matter (feeding back into Module 2/4), plus the final report document itself.

---

## 4. The technology stack — what each tool is, and why we chose it

| Technology | What it actually is, in plain words | Why we're using it here |
|---|---|---|
| **Python** | A widely-used, beginner-friendly programming language with the richest ecosystem of data/ML libraries. | It's the industry-standard language for data science and ML — every other tool below has first-class Python support. |
| **Open-Meteo (Air Quality + Weather APIs)** | A free weather/pollution data provider — no signup, no API key, no credit card. | It's the only source in our stack that gives us both live *and* multi-year historical pollution + weather data for free, which we need to backfill (Module 3) *and* keep updating live (Module 1). |
| **AQICN** | A real-time air-quality monitoring network that publishes readings from actual physical monitoring stations worldwide. | It gives us the "official," human-recognizable AQI reading people are used to seeing — used on the dashboard as a trustworthy live reference point, and as a bonus comparison against our own forecast. |
| **Hopsworks (Feature Store + Model Registry)** | A free cloud service purpose-built for ML pipelines — it's essentially an organized, versioned filing cabinet: one section for processed data ("features"), one section for trained models. | Instead of us building our own database + file storage + versioning system from scratch, Hopsworks gives us all of that out of the box, for free, and is specifically designed for exactly the "collect data → train → serve" pattern this project needs. |
| **GitHub Actions** | GitHub's built-in automation robot — you give it a schedule and a script, and it runs that script for you, on GitHub's computers, for free. | This is what lets Module 1/2/4 run "by themselves" every hour/day, with zero servers for us to maintain — the core of what "serverless automation" means here. |
| **Scikit-learn** | The standard, beginner-friendly Python library for classic machine learning algorithms (Random Forest, Ridge Regression, etc.). | Fast to train, easy to interpret, and a strong baseline — often good enough on its own, and essential as a comparison point for fancier models. |
| **XGBoost** | A more powerful, competition-winning variant of "tree-based" machine learning (similar family to Random Forest, but usually more accurate). | Typically one of the strongest performers on structured/tabular data like ours — a natural "step up" from plain scikit-learn models to include in the comparison. |
| **Statsmodels (SARIMAX)** | A classical statistics-based time-series forecasting method (the "old-school" way to forecast a number over time, before machine learning). | The assignment specifically asks for a *range* from statistical to deep learning — this represents the statistical end, and time-series-specific methods sometimes beat ML on smooth, seasonal data like pollution. |
| **TensorFlow (LSTM)** | A deep learning library; LSTM is a type of neural network specifically built to learn from *sequences* (like "the last 14 days of data"). | Represents the deep-learning end of the model comparison — good at picking up complex patterns across time that simpler models might miss. |
| **SHAP** | A library that explains *why* a machine learning model made a specific prediction, by scoring how much each input factor contributed. | Turns our model from a black box into something explainable — a specific requirement, and genuinely useful for the report and dashboard. |
| **Streamlit** | A Python library that turns a plain script into an interactive website, without needing to know web development (HTML/CSS/JavaScript). | The fastest way for us to build a real, good-looking, interactive dashboard using only Python — matches the "simple and descriptive dashboard" requirement with the least overhead. |
| **Streamlit Community Cloud** | Free hosting, made by the Streamlit team, that runs a Streamlit app straight from a GitHub repo. | Keeps the dashboard "serverless" too — we never manage a web server; we just push code to GitHub and the site updates itself. |
| **Git & GitHub** | Git tracks every change to our code over time; GitHub is where that history lives online and where our automation (Actions) runs. | Lets us build incrementally with a full commit history (as you asked), collaborate, and is also required as the trigger for GitHub Actions and Streamlit Cloud deployment. |

---

## 5. What you'll actually see and use at the end

- A **public GitHub repo** with a running commit history showing the project being built step by step.
- A **live website** (Streamlit) you can open on your phone or laptop that shows: today's real AQI in Islamabad, a 3-day forecast, a trend chart, a "why" explanation panel, and a red warning banner on bad-air days.
- A **GitHub Actions tab** showing the pipelines running automatically, hour after hour, day after day, with no manual intervention.
- A **report** summarizing the data patterns we found, which model won for each forecast day and why, and what the system can and can't do well.

---

## 6. Two things you'll need to set up yourself (can't be automated for you)

Both are free, but require a human to sign up (email verification, accepting terms):

1. **Hopsworks account + API key** — [hopsworks.ai](https://www.hopsworks.ai) (free tier). This is where all our processed data and trained models will live.
2. **AQICN API token** — [aqicn.org/data-platform/token](https://aqicn.org/data-platform/token/) (free, just needs an email). This powers the "live official reading" shown on the dashboard.

Once you have both, we'll store them as GitHub repo secrets (so the automation can use them) and as Streamlit Cloud secrets (so the dashboard can use them) — never committed into the code itself, for security.
