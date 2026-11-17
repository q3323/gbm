from __future__ import annotations

import math

import polars as pl

from app.adjustments import normalize_adjust_type


PRIMARY_SOURCE_ORDER = {"tushare": 0, "baostock": 1, "akshare": 2}


def select_primary_daily(
    frame: pl.DataFrame,
    *,
    adjust_type: str = "qfq",
) -> pl.DataFrame:
    if frame.is_empty() or "source" not in frame.columns:
        return frame
    requested_adjust = normalize_adjust_type(adjust_type)
    if "adjust_type" not in frame.columns:
        frame = frame.with_columns(pl.lit(requested_adjust).alias("adjust_type"))
    frame = frame.with_columns(
        pl.col("adjust_type")
        .cast(pl.String)
        .map_elements(
            lambda value: normalize_adjust_type(value),
            return_dtype=pl.String,
        )
        .alias("adjust_type")
    ).filter(pl.col("adjust_type") == requested_adjust)
    if frame.is_empty():
        return frame
    priority = pl.col("source").replace_strict(
        PRIMARY_SOURCE_ORDER,
        default=99,
        return_dtype=pl.Int8,
    )
    return (
        frame.with_columns(priority.alias("_source_priority"))
        .sort(["stock_code", "trade_date", "adjust_type", "_source_priority"])
        .unique(
            subset=["stock_code", "trade_date", "adjust_type"],
            keep="first",
            maintain_order=True,
        )
        .drop("_source_priority")
    )


def compute_indicators(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"stock_code", "trade_date", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing indicator columns: {', '.join(sorted(missing))}")
    if frame.is_empty():
        return frame

    primary = select_primary_daily(frame, adjust_type="qfq").sort(
        ["stock_code", "trade_date"]
    )
    close = pl.col("close").cast(pl.Float64)
    high = pl.col("high").cast(pl.Float64)
    low = pl.col("low").cast(pl.Float64)
    volume = (
        pl.col("volume").cast(pl.Float64)
        if "volume" in primary.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    turnover = (
        pl.col("turnover").cast(pl.Float64)
        if "turnover" in primary.columns
        else pl.lit(None, dtype=pl.Float64)
    )

    previous_close = close.shift(1)
    true_range = pl.max_horizontal(
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    )

    enriched = primary.with_columns(
        close.pct_change().alias("return_1d"),
        (close / previous_close).log().alias("log_return_1d"),
        close.rolling_mean(window_size=5).alias("ma_5"),
        close.rolling_mean(window_size=10).alias("ma_10"),
        close.rolling_mean(window_size=20).alias("ma_20"),
        close.rolling_mean(window_size=60).alias("ma_60"),
        close.ewm_mean(span=5, adjust=False).alias("ema_5"),
        close.ewm_mean(span=10, adjust=False).alias("ema_10"),
        close.ewm_mean(span=20, adjust=False).alias("ema_20"),
        close.ewm_mean(span=12, adjust=False).alias("ema_12"),
        close.ewm_mean(span=26, adjust=False).alias("ema_26"),
        true_range.alias("true_range"),
    )

    enriched = enriched.with_columns(
        (pl.col("ema_12") - pl.col("ema_26")).alias("macd"),
        pl.col("return_1d").rolling_std(window_size=20).mul(math.sqrt(252)).alias(
            "volatility_20"
        ),
        (volume / volume.rolling_mean(window_size=5)).alias("volume_ratio_5"),
        volume.rolling_mean(window_size=5).alias("volume_ma_5"),
        volume.rolling_mean(window_size=20).alias("volume_ma_20"),
        turnover.rolling_mean(window_size=5).alias("turnover_ma_5"),
        pl.col("true_range").ewm_mean(alpha=1 / 14, adjust=False).alias("atr_14"),
    )
    enriched = enriched.with_columns(
        pl.col("macd").ewm_mean(span=9, adjust=False).alias("macd_signal")
    ).with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_histogram")
    )

    delta = close - previous_close
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.ewm_mean(alpha=1 / 14, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1 / 14, adjust=False)
    relative_strength = avg_gain / avg_loss
    rsi = pl.when(avg_loss == 0)
    rsi = rsi.then(
        pl.when(avg_gain > 0).then(100.0).otherwise(50.0)
    ).otherwise(100.0 - 100.0 / (1.0 + relative_strength))

    rolling_low = low.rolling_min(window_size=9)
    rolling_high = high.rolling_max(window_size=9)
    rsv = (
        (close - rolling_low)
        / pl.when(rolling_high > rolling_low)
        .then(rolling_high - rolling_low)
        .otherwise(None)
        * 100.0
    )
    k_value = rsv.ewm_mean(alpha=1 / 3, adjust=False)
    d_value = k_value.ewm_mean(alpha=1 / 3, adjust=False)
    j_value = 3 * k_value - 2 * d_value

    middle = close.rolling_mean(window_size=20)
    std = close.rolling_std(window_size=20, ddof=0)
    enriched = enriched.with_columns(
        rsi.alias("rsi_14"),
        rsv.alias("kdj_rsv"),
        k_value.alias("kdj_k"),
        d_value.alias("kdj_d"),
        j_value.alias("kdj_j"),
        middle.alias("boll_mid_20"),
        (middle + 2 * std).alias("boll_upper_20"),
        (middle - 2 * std).alias("boll_lower_20"),
    )
    return enriched.with_columns(
        pl.lit("dws_v1").alias("indicator_version")
    )

