"""Scores published forecasts against what actually happened.

Reads every historical version of `web/public/data/forecast.json` out of git,
joins each published forecast to the day it predicted, and writes BACKTEST.md.

The git history is the forecast ledger: each nightly run committed what the
system predicted, and later runs committed what the air actually did. Nothing
extra had to be recorded for this to be possible.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --out BACKTEST.md
    python scripts/run_backtest.py --quiet

Needs no credentials and no network - only the repository.
"""

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.backtest import (  # noqa: E402
    HorizonScore,
    Matched,
    build_ledger,
    coverage_gaps,
    match,
    score_by_horizon,
    unresolved,
)

SNAPSHOT_PATH = "web/public/data/forecast.json"
DEFAULT_OUTPUT = Path("BACKTEST.md")


def parse_args():
    parser = argparse.ArgumentParser(description="Score published forecasts against outcomes.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Where to write the report.")
    parser.add_argument("--quiet", action="store_true", help="Write the file without printing a summary.")
    return parser.parse_args()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, encoding="utf-8"
    ).stdout


def load_snapshots_from_git() -> list[dict]:
    """Every committed version of the snapshot, oldest first, plus the working copy.

    A revision that fails to parse is skipped rather than fatal: one malformed
    historical commit should not stop the other twenty being scored.
    """
    revisions = _git("log", "--format=%H", "--reverse", "--", SNAPSHOT_PATH).split()

    snapshots: list[dict] = []
    for revision in revisions:
        try:
            snapshots.append(json.loads(_git("show", f"{revision}:{SNAPSHOT_PATH}")))
        except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            print(f"  skipping {revision[:8]}: {type(exc).__name__}")

    working = Path(SNAPSHOT_PATH)
    if working.exists():
        try:
            snapshots.append(json.loads(working.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass

    return snapshots


def _fmt(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _signed(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "--"
    return f"{value:+.{digits}f}"


def render_report(
    scores: list[HorizonScore],
    matched: list[Matched],
    pending: list,
    snapshot_count: int,
    gaps: list[str] | None = None,
) -> str:
    dates = sorted({m.prediction.target_date for m in matched})
    window = f"{dates[0]} to {dates[-1]}" if dates else "no resolved days yet"
    gaps = gaps or []

    lines = [
        "# Backtest — published forecasts vs. what actually happened",
        "",
        f"*Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC by "
        "`scripts/run_backtest.py`.*",
        "",
        "Every accuracy figure in [REPORT.md](REPORT.md) comes from a chronological hold-out:",
        "the model is fitted on old days and scored on recent ones. That is the right way to",
        "*select* a model, but it is scored once, by the same process that chose it, on data",
        "that was already on disk.",
        "",
        "This is the other measurement. Each nightly run committed a snapshot recording what",
        "the system predicted; later runs recorded what the air did. Joining the git history",
        "of the snapshot file to itself scores forecasts against observations that **did not",
        "exist when the forecast was made**.",
        "",
        f"- **{snapshot_count}** published snapshots read from git history",
        f"- **{len(matched)}** forecasts resolved and scored, covering {window}",
        f"- **{len(pending)}** forecasts unresolved — see the coverage note below",
        "",
    ]

    if gaps:
        lines += [
            "> ### ⚠️ Coverage gap",
            "> ",
            f"> The feature store is missing **{len(gaps)} day(s)** inside the observed range:",
            f"> {', '.join(gaps)}.",
            "> ",
            "> A forecast can only be scored against a day that was actually recorded, so each",
            "> gap silently removes up to three forecasts from the sample. The `n` below is",
            "> therefore smaller than the elapsed time suggests — the missing days are a data",
            "> collection failure, not a short history.",
            "",
        ]

    lines += [
        "---",
        "",
        "## Results",
        "",
        "`Bias` is the mean signed error: positive means the system forecast dirtier air than",
        "arrived. `Category` is how often the forecast landed in the correct EPA band, which is",
        "what the alert logic actually acts on. `Skill` is the percentage reduction in RMSE",
        "against persistence — negative means persistence won.",
        "",
        "| Horizon | n | RMSE | MAE | Bias | Category | Persistence RMSE | Skill | Beat baseline |",
        "|:--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]

    for score in scores:
        skill = score.skill
        lines.append(
            f"| +{score.horizon}d | {score.n} | {_fmt(score.rmse)} | {_fmt(score.mae)} | "
            f"{_signed(score.bias)} | {_fmt(score.category_accuracy, 0)}% | "
            f"{_fmt(score.baseline_rmse)} | "
            f"{'--' if skill is None else f'{skill:+.1f}%'} | "
            f"{_fmt(score.beat_baseline_rate, 0)}% |"
        )

    lines += [
        "",
        "### How to read this",
        "",
        "These numbers are **not** comparable to the hold-out figures in REPORT.md, and should",
        "not be quoted as if they were. The hold-out spans 90 days across a full range of",
        "conditions; this window is short, recent, and whatever weather happened to occur in it.",
        "A handful of unusual days moves it substantially.",
        "",
        "What it does establish is the thing a hold-out cannot: that the deployed system, fed by",
        "the live pipeline, produces forecasts of roughly the quality its selection process",
        "promised — or that it does not.",
        "",
        "---",
        "",
        "## The ledger",
        "",
        "Every resolved forecast, newest first. `Persistence` is the last AQI observed when the",
        "forecast was issued — what you could have said for free, at the time, with no model.",
        "",
        "| Target day | H | Forecast | Actual | Error | Persistence | Model |",
        "|---|:--:|--:|--:|--:|--:|---|",
    ]

    for m in sorted(matched, key=lambda m: (m.prediction.target_date, m.prediction.horizon), reverse=True):
        p = m.prediction
        lines.append(
            f"| {p.target_date} | +{p.horizon}d | {p.predicted:.1f} | {m.actual:.1f} | "
            f"{m.error:+.1f} | {_fmt(p.baseline, 1)} | {p.model_type} |"
        )

    if pending:
        lines += [
            "",
            "---",
            "",
            "## Still open",
            "",
            "Forecasts whose target day has not been observed yet.",
            "",
            "| Target day | Horizon | Forecast | Model |",
            "|---|:--:|--:|---|",
        ]
        for p in sorted(pending, key=lambda p: (p.target_date, p.horizon)):
            lines.append(f"| {p.target_date} | +{p.horizon}d | {p.predicted:.1f} | {p.model_type} |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()

    print("Reading published snapshots from git history...")
    snapshots = load_snapshots_from_git()
    if not snapshots:
        print(f"No versions of {SNAPSHOT_PATH} found. Has the pipeline published anything yet?")
        return 1
    print(f"  {len(snapshots)} snapshots")

    predictions, observations = build_ledger(snapshots)
    matched = match(predictions, observations)
    pending = unresolved(predictions, observations)

    print(f"  {len(predictions)} distinct forecasts, {len(observations)} observed days")

    if not matched:
        print(
            "No forecast has been resolved yet: every predicted day is still in the future.\n"
            "There is nothing to score until the pipeline has run past its own horizon."
        )
        return 1

    scores = score_by_horizon(matched)
    gaps = coverage_gaps(observations)
    if gaps:
        print(f"  WARNING: {len(gaps)} day(s) missing from the feature store: {', '.join(gaps)}")
    args.out.write_text(
        render_report(scores, matched, pending, len(snapshots), gaps), encoding="utf-8"
    )
    print(f"Wrote {args.out}")

    if not args.quiet:
        print()
        print(f"{'':<8}{'n':>5}{'RMSE':>9}{'MAE':>8}{'bias':>8}{'persist':>9}{'skill':>9}")
        for score in scores:
            skill = score.skill
            print(
                f"  +{score.horizon}d   {score.n:>4}{score.rmse:>9.2f}{score.mae:>8.2f}"
                f"{score.bias:>+8.2f}{_fmt(score.baseline_rmse):>9}"
                f"{'--' if skill is None else f'{skill:+.1f}%':>9}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
