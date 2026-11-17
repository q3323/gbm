from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from app.services.ensemble import EnsembleConfig, GBMSeedEnsembleService


def seed_frame(seed: int, days: int = 50, stocks: int = 10) -> pl.DataFrame:
    rows = []
    start = date(2025, 1, 2)
    for day_index in range(days):
        trade_date = start + timedelta(days=day_index)
        for stock_index in range(stocks):
            score = ((day_index * 2 + stock_index * 5 + seed) % 20) / 20.0
            future_return = score * 0.03 - 0.01
            rows.append(
                {
                    "stock_code": f"{stock_index + 1:06d}",
                    "trade_date": trade_date,
                    "future_return": future_return,
                    "future_up": int(future_return > 0),
                    "predicted_return": future_return * 0.9,
                    "up_probability": score,
                    "baseline_probability": score * 0.95,
                    "logistic_probability": score * 0.9,
                    "fold": 1,
                    "horizon_days": 10,
                }
            )
    return pl.DataFrame(rows)


def test_seed_ensemble_averages_and_backtests() -> None:
    frames = [(7, seed_frame(7)), (42, seed_frame(42))]
    result = GBMSeedEnsembleService().run(
        frames,
        "000001",
        EnsembleConfig(
            horizon=10,
            top_fraction=0.2,
            random_trials=2,
            seed=1,
        ),
    )
    report = result.report
    assert report["metadata"]["seeds"] == [7, 42]
    assert report["ensemble"]["backtest"]["gbm"]["periods"] >= 5
    assert report["ensemble"]["cross_sectional"]["dates"] > 0
    assert len(report["seeds"]) == 2
    first_row = result.out_of_sample.head(1).to_dicts()[0]
    assert 0 <= first_row["up_probability"] <= 1
