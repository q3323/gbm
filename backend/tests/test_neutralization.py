from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import polars as pl

from app.industries import (
    INDUSTRY_TO_SECTOR,
    MIN_SECTOR_SIZE,
    load_industry_map,
    load_sector_map,
)
from app.services.gbm import (
    FEATURE_COLUMNS,
    NEUTRALIZABLE_FEATURES,
    GBMFeatureBuilder,
)


def _multi_stock_frame(
    stock_count: int = 10,
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


def test_industry_to_sector_covers_all_config_industries() -> None:
    industry_map = load_industry_map()
    unmapped = [
        industry
        for industry in set(industry_map.values())
        if industry not in INDUSTRY_TO_SECTOR and industry != "Unknown"
    ]
    assert unmapped == []


def test_load_sector_map_returns_banking_codes() -> None:
    sector_map = load_sector_map()
    assert sector_map
    assert sector_map["000001"] == "banking"
    assert sector_map["601318"] == "non_bank_financials"
    assert sector_map["600519"] == "consumer"


def test_neutralized_stock_features_are_cross_sectional_ranks() -> None:
    daily = _multi_stock_frame(stock_count=10)
    industry_map = {
        f"{index + 1:06d}": "银行" if index < 6 else "证券"
        for index in range(10)
    }
    dataset = GBMFeatureBuilder().build_neutralized(
        daily,
        industry_map=industry_map,
    )
    assert dataset.stock_count == 10
    for feature in NEUTRALIZABLE_FEATURES:
        values = dataset.training[feature].to_numpy()
        assert np.all(np.isfinite(values)), feature
        assert np.all(values > 0.0), feature
        assert np.all(values <= 1.0 + 1e-9), feature


def test_market_aggregates_are_not_neutralized() -> None:
    daily = _multi_stock_frame(stock_count=10)
    industry_map = {
        f"{index + 1:06d}": "银行" if index < 6 else "证券"
        for index in range(10)
    }
    plain = GBMFeatureBuilder().build(daily, industry_map=industry_map)
    neutral = GBMFeatureBuilder().build_neutralized(
        daily,
        industry_map=industry_map,
    )
    aggregate_features = [
        name for name in FEATURE_COLUMNS if name not in NEUTRALIZABLE_FEATURES
    ]
    assert aggregate_features
    for feature in aggregate_features:
        assert np.allclose(
            plain.training[feature].to_numpy(),
            neutral.training[feature].to_numpy(),
        ), feature


def test_neutralization_differs_from_plain_stock_features() -> None:
    daily = _multi_stock_frame(stock_count=10)
    industry_map = {
        f"{index + 1:06d}": "银行" if index < 6 else "证券"
        for index in range(10)
    }
    plain = GBMFeatureBuilder().build(daily, industry_map=industry_map)
    neutral = GBMFeatureBuilder().build_neutralized(
        daily,
        industry_map=industry_map,
    )
    changed = [
        feature
        for feature in NEUTRALIZABLE_FEATURES
        if not np.allclose(
            plain.training[feature].to_numpy(),
            neutral.training[feature].to_numpy(),
        )
    ]
    assert len(changed) >= len(NEUTRALIZABLE_FEATURES) - 1


def test_small_sector_falls_back_to_global_rank() -> None:
    daily = _multi_stock_frame(stock_count=8)
    industry_map = {
        f"{index + 1:06d}": "银行" if index < MIN_SECTOR_SIZE else "证券"
        for index in range(8)
    }
    # Non-bank sector has 3 stocks (< MIN_SECTOR_SIZE) so uses global rank.
    dataset = GBMFeatureBuilder().build_neutralized(
        daily,
        industry_map=industry_map,
    )
    for feature in NEUTRALIZABLE_FEATURES:
        values = dataset.training[feature].to_numpy()
        assert np.all(values > 0.0), feature
        assert np.all(values <= 1.0 + 1e-9), feature
