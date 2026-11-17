from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl

from app.cleaning import DAILY_KEY_COLUMNS, REQUIRED_DAILY_COLUMNS


PRICE_COLUMNS = ["open", "high", "low", "close"]
FINANCIAL_KEY_FIELDS = [
    "revenue",
    "net_profit",
    "roe",
    "debt_ratio",
    "eps",
]


def _missing_rates(frame: pl.DataFrame) -> dict[str, float]:
    if frame.height == 0:
        return {name: 1.0 for name in frame.columns}
    return {
        name: round(frame[name].null_count() / frame.height, 6)
        for name in frame.columns
    }


def _status(report: dict[str, Any]) -> str:
    missing_required = any(
        report["missing_rate"].get(name, 1.0) > 0
        for name in REQUIRED_DAILY_COLUMNS
    )
    fatal = (
        report["duplicate_count"] > 0
        or report["anomaly_count"]["invalid_price_range"] > 0
        or report["anomaly_count"]["invalid_close"] > 0
        or report["anomaly_count"]["invalid_volume"] > 0
        or missing_required
    )
    if fatal:
        return "error"
    if any(rate > 0.05 for rate in report["missing_rate"].values()):
        return "warning"
    return "good"


def daily_quality_report(
    frame: pl.DataFrame,
    *,
    expected_trading_days: int | None = None,
) -> dict[str, Any]:
    missing_columns = [
        name for name in REQUIRED_DAILY_COLUMNS if name not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    duplicate_count = 0
    if frame.height:
        unique_rows = frame.unique(
            subset=DAILY_KEY_COLUMNS,
            keep="first",
        ).height
        duplicate_count = frame.height - unique_rows

    anomaly_count = {
        "invalid_price_range": 0,
        "invalid_close": 0,
        "invalid_volume": 0,
        "volume_outlier": 0,
    }
    if frame.height:
        numeric = frame.with_columns(
            [
                pl.col(name).cast(pl.Float64, strict=False)
                for name in ["open", "high", "low", "close", "volume", "amount"]
                if name in frame.columns
            ]
        )
        if {"high", "low"}.issubset(frame.columns):
            anomaly_count["invalid_price_range"] = numeric.filter(
                pl.col("high").is_not_null()
                & pl.col("low").is_not_null()
                & (pl.col("high") < pl.col("low"))
            ).height
        if "close" in frame.columns:
            anomaly_count["invalid_close"] = numeric.filter(
                pl.col("close").is_not_null() & (pl.col("close") <= 0)
            ).height
        if "volume" in frame.columns:
            anomaly_count["invalid_volume"] = numeric.filter(
                pl.col("volume").is_not_null() & (pl.col("volume") < 0)
            ).height
            volumes = numeric["volume"].drop_nulls()
            if volumes.len() >= 5:
                median = volumes.median()
                mad = (volumes - median).abs().median()
                if median is not None and mad is not None and mad > 0:
                    anomaly_count["volume_outlier"] = numeric.filter(
                        pl.col("volume").is_not_null()
                        & ((pl.col("volume") - median).abs() > 10 * mad)
                    ).height

    coverage_start: str | None = None
    coverage_end: str | None = None
    observed_days = 0
    coverage_ratio: float | None = None
    if frame.height and "trade_date" in frame.columns:
        dates = frame["trade_date"].drop_nulls().unique().sort()
        observed_days = dates.len()
        if observed_days:
            coverage_start = str(dates[0])
            coverage_end = str(dates[-1])
        if expected_trading_days and expected_trading_days > 0:
            coverage_ratio = round(observed_days / expected_trading_days, 6)

    report: dict[str, Any] = {
        "row_count": frame.height,
        "stock_count": (
            frame["stock_code"].n_unique()
            if frame.height and "stock_code" in frame.columns
            else 0
        ),
        "missing_rate": _missing_rates(frame),
        "duplicate_count": int(duplicate_count or 0),
        "coverage": {
            "start": coverage_start,
            "end": coverage_end,
            "observed_trading_days": observed_days,
            "expected_trading_days": expected_trading_days,
            "coverage_ratio": coverage_ratio,
        },
        "anomaly_count": anomaly_count,
        "checked_on": date.today().isoformat(),
    }
    report["status"] = _status(report)
    return report


def financial_quality_report(frame: pl.DataFrame) -> dict[str, Any]:
    missing = _missing_rates(frame)
    key_missing = {
        name: missing.get(name, 1.0)
        for name in FINANCIAL_KEY_FIELDS
        if name in frame.columns
    }
    status = "good"
    if any(rate > 0 for rate in key_missing.values()):
        status = "error"
    elif any(rate > 0.1 for rate in missing.values()):
        status = "warning"
    return {
        "row_count": frame.height,
        "missing_rate": missing,
        "key_field_missing_rate": key_missing,
        "status": status,
    }


def compare_daily_sources(
    primary: pl.DataFrame,
    comparison: pl.DataFrame,
    *,
    close_tolerance: float = 0.01,
    volume_tolerance: float = 0.02,
) -> pl.DataFrame:
    keys = [
        name
        for name in ["stock_code", "trade_date", "adjust_type"]
        if name in primary.columns and name in comparison.columns
    ]
    if primary.is_empty() or comparison.is_empty():
        return pl.DataFrame()

    left = primary.select(
        *keys,
        pl.col("close").alias("primary_close"),
        pl.col("volume").alias("primary_volume"),
        pl.col("source").alias("primary_source"),
    )
    right = comparison.select(
        *keys,
        pl.col("close").alias("comparison_close"),
        pl.col("volume").alias("comparison_volume"),
        pl.col("source").alias("comparison_source"),
    )
    joined = left.join(right, on=keys, how="inner")
    result = joined.with_columns(
        (pl.col("primary_close") - pl.col("comparison_close")).alias("close_diff"),
        (
            (pl.col("primary_close") - pl.col("comparison_close")).abs()
            / pl.when(pl.col("primary_close").abs() > 0)
            .then(pl.col("primary_close").abs())
            .otherwise(None)
        ).alias("close_diff_ratio"),
        (pl.col("primary_volume") - pl.col("comparison_volume")).alias("volume_diff"),
        (
            (pl.col("primary_volume") - pl.col("comparison_volume")).abs()
            / pl.when(pl.col("primary_volume").abs() > 0)
            .then(pl.col("primary_volume").abs())
            .otherwise(None)
        ).alias("volume_diff_ratio"),
    )
    return result.with_columns(
        (
            (pl.col("close_diff_ratio").fill_null(0) > close_tolerance)
            | (pl.col("volume_diff_ratio").fill_null(0) > volume_tolerance)
        ).alias("requires_review")
    )


def source_comparison_report(
    rows: pl.DataFrame,
    *,
    primary_source: str,
    comparison_source: str,
    primary_rows: int,
    comparison_rows: int,
) -> dict[str, Any]:
    matched_rows = rows.height
    review_rows = (
        rows.filter(pl.col("requires_review")).height if matched_rows else 0
    )
    if matched_rows:
        close_diff_ratio = rows["close_diff_ratio"].drop_nulls()
        volume_diff_ratio = rows["volume_diff_ratio"].drop_nulls()
        max_close_diff_ratio = (
            round(float(close_diff_ratio.max()), 6)
            if close_diff_ratio.len()
            else None
        )
        max_volume_diff_ratio = (
            round(float(volume_diff_ratio.max()), 6)
            if volume_diff_ratio.len()
            else None
        )
        dates = rows["trade_date"].drop_nulls().unique().sort()
        coverage_start = str(dates[0]) if dates.len() else None
        coverage_end = str(dates[-1]) if dates.len() else None
    else:
        max_close_diff_ratio = None
        max_volume_diff_ratio = None
        coverage_start = None
        coverage_end = None

    status = "good"
    if not matched_rows:
        status = "error"
    elif review_rows:
        status = "warning"

    return {
        "status": status,
        "primary_source": primary_source,
        "comparison_source": comparison_source,
        "primary_row_count": primary_rows,
        "comparison_row_count": comparison_rows,
        "matched_row_count": matched_rows,
        "review_row_count": review_rows,
        "max_close_diff_ratio": max_close_diff_ratio,
        "max_volume_diff_ratio": max_volume_diff_ratio,
        "coverage": {"start": coverage_start, "end": coverage_end},
        "checked_on": date.today().isoformat(),
    }

