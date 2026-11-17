from __future__ import annotations

import math
from datetime import date
from datetime import UTC, datetime

import pandas as pd
import polars as pl
import pytest

from app.services.gbm import GBMDataError, GBMFeatureBuilder, GBMService


def multi_stock_frame(
    stock_count: int = 5,
    days: int = 220,
) -> pl.DataFrame:
    dates = [item.date() for item in pd.bdate_range("2025-01-02", periods=days)]
    rows = []
    for stock_index in range(stock_count):
        code = f"{stock_index + 1:06d}"
        for day_index, trade_date in enumerate(dates):
            close = (
                10.0
                + stock_index
                + day_index * (0.01 + stock_index * 0.002)
                + math.sin((day_index + stock_index) / 7) * 0.2
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


def test_feature_builder_creates_forward_labels() -> None:
    dataset = GBMFeatureBuilder().build(multi_stock_frame())
    assert dataset.stock_count == 5
    assert dataset.training.height > 500
    assert "future_return_5d" in dataset.training.columns
    assert dataset.training["future_return_5d"].null_count() == 0
    assert {
        "market_positive_ratio_1d",
        "market_dispersion_1d",
        "industry_return_5d",
        "industry_relative_rank_5d",
    }.issubset(dataset.training.columns)
    assert dataset.latest.height == 5


def test_gbm_trains_backtests_and_predicts() -> None:
    result = GBMService().analyze(
        multi_stock_frame(),
        "000001",
        seed=7,
        min_training_rows=200,
        min_trading_days=100,
        walk_forward_folds=2,
        min_test_rows_per_fold=15,
    )
    report = result.report
    assert report["metadata"]["seed"] == 7
    assert report["dataset"]["stock_count"] == 5
    assert report["regression"]["test_rows"] >= 50
    assert report["classification"]["test_rows"] >= 50
    assert report["walk_forward"]["folds"][0]["fold"] == 1
    assert report["dataset"]["walk_forward_folds"] == 2
    assert report["reliability"]["status"] in {
        "acceptable",
        "not_reliable",
    }
    assert report["reliability"]["regression_status"] in {
        "acceptable",
        "not_reliable",
    }
    assert report["reliability"]["classification_status"] in {
        "weak_signal",
        "not_reliable",
    }
    assert 0 <= report["prediction"]["up_probability_5d"] <= 1
    assert report["prediction"]["direction"] in {"up", "down"}
    assert result.features.height > report["dataset"]["training_rows"]


def test_gbm_analyze_accepts_financial_frame() -> None:
    financial = pl.DataFrame(
        {
            "stock_code": ["000001"],
            "report_date": [date(2024, 12, 31)],
            "announced_date": [date(2025, 1, 2)],
            "roe": [0.08],
            "net_margin": [0.30],
            "net_profit_yoy": [0.05],
            "eps_yoy": [0.04],
            "debt_ratio": [0.45],
            "equity_yoy": [0.10],
            "asset_turnover": [0.22],
            "cfo_to_revenue": [1.20],
        }
    )
    result = GBMService().analyze(
        multi_stock_frame(),
        "000001",
        seed=7,
        min_training_rows=200,
        min_trading_days=100,
        walk_forward_folds=2,
        min_test_rows_per_fold=15,
        financial=financial,
    )
    assert result.report["metadata"]["fundamental_features_enabled"] is True
    assert result.report["metadata"]["financial_report_rows"] == 1
    assert "fund_roe" in result.report["features"]

def test_gbm_rejects_single_stock_data() -> None:
    with pytest.raises(GBMDataError, match="at least 3 stocks"):
        GBMService().analyze(
            multi_stock_frame(stock_count=1),
            "000001",
            min_training_rows=100,
            min_trading_days=100,
        )


def test_fundamental_features_are_joined_without_lookahead() -> None:
    daily = multi_stock_frame(stock_count=1)
    financial = pl.DataFrame(
        {
            "stock_code": ["000001", "000001"],
            "report_date": [date(2025, 3, 31), date(2025, 6, 30)],
            "announced_date": [date(2025, 4, 21), date(2025, 8, 11)],
            "roe": [0.08, 0.09],
            "net_margin": [0.30, 0.31],
            "net_profit_yoy": [0.05, 0.06],
            "eps_yoy": [0.04, 0.05],
            "debt_ratio": [0.45, 0.44],
            "equity_yoy": [0.10, 0.11],
            "asset_turnover": [0.22, 0.23],
            "cfo_to_revenue": [1.20, 1.25],
        }
    )
    joined = GBMFeatureBuilder._join_financial_features(daily, financial)

    by_date = dict(
        zip(
            joined["trade_date"].to_list(),
            joined["fund_roe"].to_list(),
        )
    )
    assert by_date[date(2025, 4, 17)] == 0.0
    assert by_date[date(2025, 4, 21)] == pytest.approx(0.08)
    assert by_date[date(2025, 7, 1)] == pytest.approx(0.08)
    assert by_date[date(2025, 8, 8)] == pytest.approx(0.08)
    assert by_date[date(2025, 8, 11)] == pytest.approx(0.09)

    for column in {
        "fund_net_margin",
        "fund_net_profit_yoy",
        "fund_eps_yoy",
        "fund_debt_ratio",
        "fund_equity_yoy",
        "fund_asset_turnover",
        "fund_cfo_to_revenue",
    }:
        assert joined[column].null_count() == 0

