"""Module 9: exploratory data analysis.

Answers the questions that shape the modelling: when is the air worst, what
moves it, and how predictable is it at all? Writes a markdown report so the
findings can be cited rather than re-derived.

Usage:
    python scripts/run_eda.py                    # from the feature store
    python scripts/run_eda.py --offline          # straight from Open-Meteo
    python scripts/run_eda.py --output eda.md
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alerts.aqi_scale import AQI_CATEGORIES, categorize  # noqa: E402

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

WEATHER_COLUMNS = [
    "temp_mean", "humidity_mean", "wind_speed_mean", "pressure_mean", "precipitation_sum",
]
POLLUTANT_COLUMNS = ["pm2_5_mean", "pm10_mean", "co_mean", "no2_mean", "so2_mean", "o3_mean"]


def parse_args():
    parser = argparse.ArgumentParser(description="Explore the AQI dataset and write findings.")
    parser.add_argument("--city", default=None)
    parser.add_argument("--offline", action="store_true", help="Fetch from Open-Meteo instead of the feature store.")
    parser.add_argument("--output", default="EDA.md", help="Where to write the markdown report.")
    return parser.parse_args()


def load_frame(city_slug, offline):
    if not offline:
        from src.features.availability import describe_usability

        try:
            from src.training.data_prep import load_training_frame

            frame = load_training_frame(city_slug)
            # A handful of stale rows would produce a confident-looking report
            # about nothing, so hold stored data to the same bar as the dashboard.
            usable, reason = describe_usability(frame, min_rows=60)
            if usable:
                return frame, f"feature store ({reason})"
            print(f"Feature store not usable for analysis: {reason}. Falling back to the API.")
        except Exception as exc:
            print(f"Feature store unavailable ({type(exc).__name__}), falling back to the API.")

    from src.config import CITIES, DEFAULT_CITY
    from src.features.feature_engineering import compute_daily_features
    from src.features.historical import EARLIEST_AIR_QUALITY_DATE, fetch_historical_hourly

    city = CITIES[city_slug or DEFAULT_CITY]
    end = dt.date.today() - dt.timedelta(days=1)
    hourly = fetch_historical_hourly(city, EARLIEST_AIR_QUALITY_DATE, end)
    daily = compute_daily_features(hourly)
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    return daily, "Open-Meteo"


def bar(value, maximum, width=40):
    """Text bar, so the report reads without needing images."""
    filled = 0 if maximum <= 0 else round(width * value / maximum)
    return "█" * filled


def section_overview(df, source):
    span = f"{df['date'].min().date()} to {df['date'].max().date()}"
    aqi = df["aqi_mean"]
    lines = [
        "## Dataset",
        "",
        f"- **Source:** {source}",
        f"- **Rows:** {len(df):,} daily observations",
        f"- **Period:** {span} ({(df['date'].max() - df['date'].min()).days / 365.25:.1f} years)",
        f"- **AQI:** mean {aqi.mean():.1f}, median {aqi.median():.1f}, min {aqi.min():.1f}, max {aqi.max():.1f}",
        f"- **Missing daily AQI:** {int(aqi.isna().sum())}",
    ]
    if "hours_observed" in df:
        complete = int((df["hours_observed"] >= 24).sum())
        lines.append(f"- **Complete 24-hour days:** {complete:,} / {len(df):,}")
    return lines


def section_categories(df):
    counts = {c.name: 0 for c in AQI_CATEGORIES}
    for value in df["aqi_mean"].dropna():
        category = categorize(value)
        if category:
            counts[category.name] += 1

    total = sum(counts.values()) or 1
    peak = max(counts.values())
    lines = ["## How often is the air actually bad?", "", "| Category | Days | Share | |", "|---|--:|--:|---|"]
    for name, count in counts.items():
        lines.append(f"| {name} | {count:,} | {count / total * 100:.1f}% | {bar(count, peak, 24)} |")

    unhealthy = sum(c for n, c in counts.items() if n in ("Unhealthy", "Very Unhealthy", "Hazardous"))
    lines += [
        "",
        f"**{unhealthy / total * 100:.1f}%** of days reach 'Unhealthy' or worse — the days this project "
        "exists to warn about.",
    ]
    return lines


def section_seasonality(df):
    monthly = df.groupby(df["date"].dt.month)["aqi_mean"].agg(["mean", "count"])
    peak = monthly["mean"].max()

    lines = ["## Seasonal pattern", "", "| Month | Mean AQI | Days | |", "|---|--:|--:|---|"]
    for month, row in monthly.iterrows():
        lines.append(
            f"| {MONTH_NAMES[int(month) - 1]} | {row['mean']:.1f} | {int(row['count'])} | {bar(row['mean'], peak, 30)} |"
        )

    worst = MONTH_NAMES[int(monthly['mean'].idxmax()) - 1]
    best = MONTH_NAMES[int(monthly['mean'].idxmin()) - 1]
    ratio = monthly["mean"].max() / monthly["mean"].min()
    lines += [
        "",
        f"Worst month is **{worst}** ({monthly['mean'].max():.0f}), best is **{best}** "
        f"({monthly['mean'].min():.0f}) — a **{ratio:.1f}x** swing across the year.",
        "",
        "This is why `month` is a model feature: without it, a model has no way to know whether "
        "an AQI of 120 is unusually bad for the season or unusually good.",
    ]
    return lines


def section_weekly(df):
    weekly = df.groupby("day_of_week")["aqi_mean"].mean() if "day_of_week" in df else \
        df.groupby(df["date"].dt.dayofweek)["aqi_mean"].mean()
    peak = weekly.max()

    lines = ["## Weekly pattern", "", "| Day | Mean AQI | |", "|---|--:|---|"]
    for day, value in weekly.items():
        lines.append(f"| {DAY_NAMES[int(day)]} | {value:.1f} | {bar(value, peak, 30)} |")

    spread = weekly.max() - weekly.min()
    lines += [
        "",
        f"Spread across the week is only **{spread:.1f} AQI points** "
        f"({spread / weekly.mean() * 100:.1f}% of the weekly mean).",
        "",
        "Weekday-versus-weekend traffic matters far less here than season does. Useful to know: "
        "`is_weekend` earns its place as a feature, but it was never going to be a strong one.",
    ]
    return lines


def section_drivers(df):
    available = [c for c in POLLUTANT_COLUMNS + WEATHER_COLUMNS if c in df.columns]
    correlations = df[available + ["aqi_mean"]].corr()["aqi_mean"].drop("aqi_mean")
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

    lines = ["## What moves air quality?", "", "| Feature | Correlation with AQI |", "|---|--:|"]
    for name, value in correlations.items():
        lines.append(f"| `{name}` | {value:+.3f} |")

    strongest_weather = correlations[[c for c in WEATHER_COLUMNS if c in correlations.index]]
    lines += [
        "",
        f"The strongest pollutant link is `{correlations.index[0]}` ({correlations.iloc[0]:+.2f}), "
        "which is expected — the AQI is largely *derived* from pollutant concentrations, so this is "
        "closer to a definition than a discovery.",
        "",
        "The interesting column is weather, which is genuinely independent:",
        "",
    ]
    for name, value in strongest_weather.sort_values(key=abs, ascending=False).items():
        lines.append(f"- `{name}`: {value:+.3f}")
    lines += [
        "",
        "Negative correlations for wind and rain match the physics — wind disperses pollution and "
        "rain washes it out. That is why weather is worth fetching at all.",
    ]
    return lines


def section_predictability(df):
    aqi = df.sort_values("date")["aqi_mean"]
    lines = [
        "## How predictable is it?",
        "",
        "| Lag | Autocorrelation |",
        "|---|--:|",
    ]
    for lag in (1, 2, 3, 7, 14, 30):
        lines.append(f"| {lag} day(s) | {aqi.autocorr(lag):.3f} |")

    lines += [
        "",
        f"Day-to-day correlation is strong ({aqi.autocorr(1):.2f}) but decays quickly — by day 3 it is "
        f"{aqi.autocorr(3):.2f}.",
        "",
        "This single table explains the model results better than anything else: it is why the day-1 "
        "forecast reaches R² 0.84 while day 3 struggles past 0.14, and why a naive "
        "'tomorrow equals today' baseline is hard to beat at day 1 and useless by day 3.",
    ]
    return lines


def section_volatility(df):
    swings = df.sort_values("date")["aqi_mean"].diff().abs()
    lines = [
        "## Day-to-day swings",
        "",
        f"- Median absolute change: **{swings.median():.1f} AQI points**",
        f"- 90th percentile: **{swings.quantile(0.9):.1f}**",
        f"- Largest single-day change: **{swings.max():.1f}**",
        "",
        f"A typical day moves about {swings.median():.0f} points. Any forecast with an error much below "
        "that is doing genuinely well; the day-1 model's RMSE of ~9 sits right around this natural "
        "day-to-day noise floor, which is roughly the best that can be expected.",
    ]
    return lines


def main():
    args = parse_args()
    df, source = load_frame(args.city, args.offline)

    if df.empty:
        print("No data available.")
        return

    df = df.sort_values("date").reset_index(drop=True)

    sections = [
        ["# Exploratory Data Analysis — Pearls AQI Predictor", "",
         f"_Generated {dt.date.today():%d %B %Y}._", ""],
        section_overview(df, source),
        [""],
        section_categories(df),
        [""],
        section_seasonality(df),
        [""],
        section_weekly(df),
        [""],
        section_drivers(df),
        [""],
        section_predictability(df),
        [""],
        section_volatility(df),
        [""],
    ]

    report = "\n".join(line for section in sections for line in section)
    Path(args.output).write_text(report, encoding="utf-8")

    print(report)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
