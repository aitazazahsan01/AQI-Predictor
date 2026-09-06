# Backtest — published forecasts vs. what actually happened

*Generated 2026-09-06 19:18 UTC by `scripts/run_backtest.py`.*

Every accuracy figure in [REPORT.md](REPORT.md) comes from a chronological hold-out:
the model is fitted on old days and scored on recent ones. That is the right way to
*select* a model, but it is scored once, by the same process that chose it, on data
that was already on disk.

This is the other measurement. Each nightly run committed a snapshot recording what
the system predicted; later runs recorded what the air did. Joining the git history
of the snapshot file to itself scores forecasts against observations that **did not
exist when the forecast was made**.

- **13** published snapshots read from git history
- **18** forecasts resolved and scored, covering 2026-08-18 to 2026-09-05
- **18** forecasts unresolved — see the coverage note below

> ### ⚠️ Coverage gap
> 
> The feature store is missing **8 day(s)** inside the observed range:
> 2026-08-23, 2026-08-25, 2026-08-27, 2026-08-28, 2026-08-29, 2026-08-31, 2026-09-02, 2026-09-03.
> 
> A forecast can only be scored against a day that was actually recorded, so each
> gap silently removes up to three forecasts from the sample. The `n` below is
> therefore smaller than the elapsed time suggests — the missing days are a data
> collection failure, not a short history.

---

## Results

`Bias` is the mean signed error: positive means the system forecast dirtier air than
arrived. `Category` is how often the forecast landed in the correct EPA band, which is
what the alert logic actually acts on. `Skill` is the percentage reduction in RMSE
against persistence — negative means persistence won.

| Horizon | n | RMSE | MAE | Bias | Category | Persistence RMSE | Skill | Beat baseline |
|:--:|--:|--:|--:|--:|--:|--:|--:|--:|
| +1d | 6 | 15.02 | 12.90 | -7.66 | 100% | 15.01 | -0.1% | 33% |
| +2d | 7 | 14.89 | 12.44 | -3.68 | 100% | 14.56 | -2.3% | 57% |
| +3d | 5 | 14.86 | 13.15 | -6.87 | 100% | 12.73 | -16.8% | 40% |

### How to read this

These numbers are **not** comparable to the hold-out figures in REPORT.md, and should
not be quoted as if they were. The hold-out spans 90 days across a full range of
conditions; this window is short, recent, and whatever weather happened to occur in it.
A handful of unusual days moves it substantially.

What it does establish is the thing a hold-out cannot: that the deployed system, fed by
the live pipeline, produces forecasts of roughly the quality its selection process
promised — or that it does not.

---

## The ledger

Every resolved forecast, newest first. `Persistence` is the last AQI observed when the
forecast was issued — what you could have said for free, at the time, with no model.

| Target day | H | Forecast | Actual | Error | Persistence | Model |
|---|:--:|--:|--:|--:|--:|---|
| 2026-09-05 | +1d | 138.0 | 148.6 | -10.6 | 137.8 | ridge |
| 2026-09-04 | +3d | 115.6 | 137.8 | -22.2 | 113.3 | random_forest |
| 2026-09-01 | +2d | 116.5 | 113.3 | +3.2 | 126.0 | ridge |
| 2026-08-26 | +2d | 128.3 | 124.7 | +3.6 | 136.1 | ridge |
| 2026-08-24 | +3d | 118.7 | 136.1 | -17.4 | 130.5 | random_forest |
| 2026-08-24 | +2d | 115.8 | 136.1 | -20.3 | 126.2 | ridge |
| 2026-08-22 | +3d | 122.9 | 126.2 | -3.3 | 122.3 | random_forest |
| 2026-08-22 | +2d | 110.0 | 126.2 | -16.2 | 103.1 | ridge |
| 2026-08-22 | +1d | 116.7 | 126.2 | -9.5 | 130.5 | ridge |
| 2026-08-21 | +3d | 123.4 | 130.5 | -7.1 | 123.7 | random_forest |
| 2026-08-21 | +2d | 114.0 | 130.5 | -16.5 | 122.3 | ridge |
| 2026-08-21 | +1d | 100.8 | 130.5 | -29.7 | 103.1 | ridge |
| 2026-08-20 | +3d | 118.8 | 103.1 | +15.7 | 113.9 | random_forest |
| 2026-08-20 | +2d | 127.0 | 103.1 | +23.9 | 123.7 | ridge |
| 2026-08-20 | +1d | 109.8 | 103.1 | +6.7 | 122.3 | ridge |
| 2026-08-19 | +2d | 118.9 | 122.3 | -3.4 | 113.9 | ridge |
| 2026-08-19 | +1d | 131.3 | 122.3 | +9.0 | 123.7 | ridge |
| 2026-08-18 | +1d | 111.8 | 123.7 | -11.9 | 113.9 | ridge |

---

## Still open

Forecasts whose target day has not been observed yet.

| Target day | Horizon | Forecast | Model |
|---|:--:|--:|---|
| 2026-08-23 | +1d | 107.6 | ridge |
| 2026-08-23 | +2d | 117.5 | ridge |
| 2026-08-23 | +3d | 107.4 | random_forest |
| 2026-08-25 | +1d | 135.9 | ridge |
| 2026-08-25 | +3d | 115.7 | random_forest |
| 2026-08-27 | +1d | 126.4 | ridge |
| 2026-08-27 | +3d | 123.2 | random_forest |
| 2026-08-28 | +2d | 125.8 | ridge |
| 2026-08-29 | +3d | 114.3 | random_forest |
| 2026-08-31 | +1d | 106.6 | ridge |
| 2026-09-02 | +1d | 100.3 | ridge |
| 2026-09-02 | +3d | 118.4 | random_forest |
| 2026-09-03 | +2d | 109.2 | ridge |
| 2026-09-06 | +1d | 142.9 | ridge |
| 2026-09-06 | +2d | 132.9 | ridge |
| 2026-09-07 | +2d | 132.3 | ridge |
| 2026-09-07 | +3d | 125.1 | random_forest |
| 2026-09-08 | +3d | 128.2 | random_forest |
