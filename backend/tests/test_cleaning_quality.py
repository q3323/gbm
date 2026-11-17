from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from app.cleaning import clean_daily_bars
from app.quality import (
    compare_daily_sources,
    daily_quality_report,
    source_comparison_report,
)


def sample_bars() -> pl.DataFrame:
    now = datetime.now(UTC).replace(tzinfo=None)
    return pl.DataFrame(
        {
            "stock_code": ["000001", "000001", "000001", "000001"],
            "trade_date": ["2026-11-10", "2026-11-10", "2026-11-11", "2026-11-12"],
            "adjust_type": ["qfq", "qfq", "qfq", "qfq"],
            "open": [10.0, 10.0, 10.1, 10.2],
            "high": [10.5, 10.5, 10.2, 9.7],
            "low": [9.9, 9.9, 9.9, 9.8],
            "close": [10.2, 10.2, 10.0, 9.9],
            "pre_close": [10.0, 10.0, 10.2, 10.0],
            "volume": [1000.0, 1000.0, 1100.0, 900.0],
            "amount": [10000.0, 10000.0, 11000.0, 9000.0],
            "turnover": [1.0, 1.0, 1.1, 0.9],
            "pct_change": [2.0, 2.0, -1.96, -1.0],
            "source": ["akshare", "akshare", "akshare", "akshare"],
            "ingested_at": [now, now, now, now],
        }
    )


def test_cleaning_deduplicates_and_flags_invalid_price() -> None:
    cleaned = clean_daily_bars(sample_bars())
    assert cleaned.height == 3
    assert "is_duplicate" in cleaned.columns
    assert cleaned.filter(pl.col("invalid_price_range")).height == 1


def test_quality_report_detects_duplicate_and_coverage() -> None:
    report = daily_quality_report(sample_bars(), expected_trading_days=4)
    assert report["duplicate_count"] == 1
    assert report["anomaly_count"]["invalid_price_range"] == 1
    assert report["coverage"]["observed_trading_days"] == 3
    assert report["coverage"]["coverage_ratio"] == 0.75
    assert report["status"] == "error"


def test_source_comparison_flags_price_difference() -> None:
    primary = sample_bars().head(1)
    comparison = primary.with_columns(
        pl.lit(11.0).alias("close"),
        pl.lit("tushare").alias("source"),
    )
    result = compare_daily_sources(primary, comparison)
    assert result.height == 1
    assert result[0, "requires_review"] is True


def test_quality_report_accepts_raw_string_numbers() -> None:
    raw = sample_bars().with_columns(
        pl.col("open").cast(pl.String),
        pl.col("high").cast(pl.String),
        pl.col("low").cast(pl.String),
        pl.col("close").cast(pl.String),
        pl.col("volume").cast(pl.String),
    )
    report = daily_quality_report(raw)
    assert report["anomaly_count"]["invalid_close"] == 0
    assert report["status"] == "error"


def test_adjustment_types_are_separate_duplicate_keys() -> None:
    raw = sample_bars().head(1)
    combined = pl.concat(
        [
            raw,
            raw.with_columns(pl.lit("raw").alias("adjust_type")),
        ]
    )
    assert daily_quality_report(combined)["duplicate_count"] == 0
    cleaned = clean_daily_bars(combined)
    assert set(cleaned["adjust_type"].to_list()) == {"qfq", "raw"}


def test_source_comparison_report_flags_review_rows() -> None:
    primary = sample_bars().head(1)
    comparison = primary.with_columns(
        pl.lit(10.5).alias("close"),
        pl.lit("tushare").alias("source"),
    )
    rows = compare_daily_sources(primary, comparison)
    report = source_comparison_report(
        rows,
        primary_source="akshare",
        comparison_source="tushare",
        primary_rows=1,
        comparison_rows=1,
    )
    assert report["status"] == "warning"
    assert report["review_row_count"] == 1

