from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from app.services.backtest import (
    BacktestConfig,
    StrategyBacktestService,
)


def synthetic_oos(days: int = 40, stocks: int = 10) -> pl.DataFrame:
    rows = []
    start = date(2026, 1, 5)
    for day_index in range(days):
        trade_date = start + timedelta(days=day_index)
        for stock_index in range(stocks):
            score = ((day_index * 3 + stock_index * 7) % 10) / 10.0
            future_return = score * 0.02 - 0.009
            rows.append(
                {
                    "stock_code": f"{stock_index + 1:06d}",
                    "trade_date": trade_date,
                    "future_return": future_return,
                    "future_up": int(future_return > 0),
                    "predicted_return": future_return * 0.8,
                    "up_probability": score,
                    "logistic_probability": score * 0.9,
                    "fold": day_index // 10 + 1,
                }
            )
    return pl.DataFrame(rows)


def test_strategy_backtest_reports_costs_and_baselines() -> None:
    result = StrategyBacktestService().run(
        synthetic_oos(),
        BacktestConfig(
            top_fraction=0.2,
            rebalance_days=5,
            random_trials=5,
            seed=7,
        ),
    )
    report = result.report
    assert report["metadata"]["periods"] >= 5
    assert len(report["equity_curve"]) == report["metadata"]["periods"]
    assert report["gbm"]["total_cost_rate"] > 0
    assert (
        report["gbm"]["gross_total_return"]
        > report["gbm"]["total_return"]
    )
    assert report["random_baseline"]["trials"] == 5
    assert report["probability_analysis"]["buckets"]
    assert report["probability_analysis"]["thresholds"]


def test_strategy_backtest_uses_holding_fraction() -> None:
    result = StrategyBacktestService().run(
        synthetic_oos(stocks=20),
        BacktestConfig(
            top_fraction=0.1,
            rebalance_days=5,
            random_trials=2,
        ),
    )
    assert all(
        item["holding_count"] == 2
        for item in result.report["equity_curve"]
    )
