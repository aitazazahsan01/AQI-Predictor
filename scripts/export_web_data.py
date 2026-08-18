"""Publishes the forecast as a static JSON file for the Next.js front end.

The website is a static build: no server, no database, no credentials. This
script is the only thing that talks to Hopsworks and the models, and it runs
where the secrets already live - inside GitHub Actions, right after training.
The site then renders whatever this wrote.

Usage:
    python scripts/export_web_data.py
    python scripts/export_web_data.py --out web/public/data/forecast.json
    python scripts/export_web_data.py --skip-drivers   # faster; omits SHAP

Requires HOPSWORKS_API_KEY and HOPSWORKS_PROJECT_NAME unless the feature store
is unreachable, in which case it falls back to a live Open-Meteo fetch and says
so in the published payload.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_CITY  # noqa: E402
from src.inference.predict import build_forecast, load_models  # noqa: E402
from src.inference.snapshot import build_snapshot, load_daily_features  # noqa: E402

DEFAULT_OUTPUT = Path("web/public/data/forecast.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Export the forecast as static JSON.")
    parser.add_argument("--city", default=DEFAULT_CITY, help="City slug to export.")
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT, help="Where to write the JSON."
    )
    parser.add_argument(
        "--skip-drivers",
        action="store_true",
        help="Skip the SHAP explanations, which dominate the runtime.",
    )
    return parser.parse_args()


def load_station(city_slug: str) -> dict | None:
    """The live station reading, display only - never training data.

    Absent or stale is normal (the Islamabad station goes quiet for months), so
    every failure here degrades to None rather than failing the export.
    """
    import os

    from src.config import CITIES
    from src.data_sources import aqicn_client

    token = os.environ.get("AQICN_TOKEN", "")
    if not token:
        return None
    try:
        return aqicn_client.fetch_live_reading(CITIES[city_slug].aqicn_station_uid, token)
    except Exception as exc:
        print(f"Station reading unavailable ({type(exc).__name__}: {exc}).")
        return None


def main() -> int:
    args = parse_args()

    print(f"Loading features for {args.city}...")
    features, feature_source = load_daily_features(args.city)
    if features.empty:
        print("No features available - refusing to publish an empty snapshot.")
        return 1
    print(f"  {len(features)} rows from {feature_source}")

    print("Loading models...")
    models = load_models()
    if not models:
        print(
            "No models available - refusing to publish a forecast-less snapshot.\n"
            "Run the training pipeline first so the Model Registry has something to serve."
        )
        return 1
    print("  " + ", ".join(f"h{h}={m.model_type} ({m.source})" for h, m in sorted(models.items())))

    forecasts = build_forecast(features.tail(1), models)
    station = load_station(args.city)

    print("Building snapshot" + ("" if args.skip_drivers else " (computing SHAP explanations)") + "...")
    snapshot = build_snapshot(
        args.city,
        features=features,
        forecasts=forecasts,
        models=models,
        feature_source=feature_source,
        station=station,
        include_drivers=not args.skip_drivers,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    size_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out} ({size_kb:.1f} KB)")
    for entry in snapshot["forecast"]:
        print(f"  +{entry['horizon']}d {entry['date']}: AQI {entry['aqi']} ({entry['category']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
