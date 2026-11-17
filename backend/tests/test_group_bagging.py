from __future__ import annotations

import math
from datetime import UTC, datetime

import pandas as pd
import polars as pl

from app.services.group_bagging import (
    GroupBaggingConfig,
    GroupedGBMBaggingService,
)
from app.services.index_gbm import IndexBaggingConfig, IndexGBMBaggingService


def group_daily(stock_count: int = 8, days: int = 420) -> pl.DataFrame:
    dates = [item.date() for item in pd.bdate_range("2025-01-02", periods=days)]
    rows = []
    for stock_index in range(stock_count):
        code = f"{stock_index + 1:06d}"
        for day_index, trade_date in enumerate(dates):
            close = (
                10.0
                + stock_index
                + day_index * (0.01 + stock_index * 0.001)
                + math.sin((day_index + stock_index) / 8) * 0.2
            )
            rows.append(
                {
                    "stock_code": code,
                    "trade_date": trade_date,
                    "adjust_type": "qfq",
                    "open": close - 0.05,
                    "high": close + 0.15,
                    "low": close - 0.15,
                    "close": close,
                    "pre_close": close - 0.01,
                    "volume": 1000.0 + day_index + stock_index * 20,
                    "amount": 10000.0 + day_index * 10,
                    "turnover": 1.0 + stock_index * 0.1,
                    "pct_change": 0.1,
                    "source": "baostock",
                    "ingested_at": datetime.now(UTC).replace(tzinfo=None),
                }
            )
    return pl.DataFrame(rows)


def test_group_bagging_averages_seed_predictions() -> None:
    codes = tuple(f"{index:06d}" for index in range(1, 9))
    result = GroupedGBMBaggingService().run(
        group_daily(),
        GroupBaggingConfig(
            group_id="test_group",
            name="Test",
            market="CN",
            stock_codes=codes,
            seeds=(7, 42, 2024),
            sample_fraction=0.8,
            horizon_days=5,
        ),
            min_training_rows=500,
        walk_forward_folds=2,
        min_test_rows_per_fold=15,
    )
    report = result.report
    assert report["metadata"]["seeds"] == [7, 42, 2024]
    assert len(report["rankings"]) == 8
    assert all(item["seed_count"] == 3 for item in report["rankings"])
    assert report["rankings"][0]["rank_score"] == 100.0
    assert report["rankings"][-1]["rank_score"] == 0.0
    assert "raw_up_probability" in report["rankings"][0]
    assert "calibration_shift" in report["rankings"][0]
    assert "trend_score" in report["rankings"][0]
    assert report["evaluation"]["rank_ic_mode"] == "cross_sectional"
    assert "icir" in report["evaluation"]
    assert "max_drawdown" in report["evaluation"]
    assert "raw_brier_score" in report["evaluation"]
    assert "calibrated_brier_score" in report["evaluation"]
    assert "momentum_rank_ic" in report["evaluation"]["baselines"]


def test_nasdaq_index_bagging_builds_prediction() -> None:
    days = 340
    dates = [item.date() for item in pd.bdate_range("2025-01-02", periods=days)]
    rows = []
    for index, trade_date in enumerate(dates):
        close = 20000 + index * 8 + math.sin(index / 9) * 180
        rows.append(
            {
                "date": trade_date.isoformat(),
                "open": close - 40,
                "high": close + 90,
                "low": close - 100,
                "close": close,
                "volume": 1_000_000_000 + index * 1_000_000,
                "amount": 0.0,
            }
        )
    result = IndexGBMBaggingService().run(
        pl.DataFrame(rows),
        IndexBaggingConfig(
            group_id="us_ndx_index",
            name="Nasdaq",
            symbol=".NDX",
            seeds=(7, 42, 2024),
            sample_fraction=0.8,
            horizon_days=10,
        ),
    )
    report = result.report
    assert report["metadata"]["group_id"] == "us_ndx_index"
    assert len(report["seeds"]) == 3
    assert 0 <= report["group_prediction"]["up_probability"] <= 1
    assert report["rankings"][0]["stock_code"] == ".NDX"
    assert report["evaluation"]["rank_ic_mode"] == "time_series"
    assert "signal_correlation" in report["evaluation"]
    assert "max_drawdown" in report["evaluation"]
    assert "raw_up_probability" in report["group_prediction"]
    assert "calibration_shift" in report["group_prediction"]
    assert report["seeds"][0]["selected_features"]
