from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from app.adjustments import DEFAULT_ADJUST_TYPE, normalize_adjust_type


CANONICAL_DAILY_COLUMNS = [
    "stock_code",
    "trade_date",
    "adjust_type",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "turnover",
    "pct_change",
    "source",
    "ingested_at",
]

NUMERIC_DAILY_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "volume",
    "amount",
    "turnover",
    "pct_change",
]

REQUIRED_DAILY_COLUMNS = [
    "stock_code",
    "trade_date",
    "close",
    "source",
    "adjust_type",
]
DAILY_KEY_COLUMNS = ["stock_code", "trade_date", "source", "adjust_type"]


def ensure_daily_columns(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [name for name in CANONICAL_DAILY_COLUMNS if name not in frame.columns]
    if missing:
        frame = frame.with_columns(
            [
                pl.lit(DEFAULT_ADJUST_TYPE if name == "adjust_type" else None).alias(
                    name
                )
                for name in missing
            ]
        )
    frame = frame.with_columns(
        pl.col("adjust_type")
        .cast(pl.String)
        .map_elements(
            lambda value: normalize_adjust_type(
                value,
                default=DEFAULT_ADJUST_TYPE,
            ),
            return_dtype=pl.String,
        )
        .alias("adjust_type")
    )
    return frame.select(CANONICAL_DAILY_COLUMNS)


def clean_daily_bars(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return ensure_daily_columns(frame)

    cleaned = ensure_daily_columns(frame)
    cleaned = cleaned.with_columns(
        pl.col("stock_code").cast(pl.String).str.zfill(6),
        pl.col("trade_date")
        .cast(pl.String)
        .str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("source").cast(pl.String),
        pl.lit(datetime.now(timezone.utc).replace(tzinfo=None))
        .cast(pl.Datetime)
        .alias("ingested_at"),
        *[
            pl.col(name).cast(pl.Float64, strict=False)
            for name in NUMERIC_DAILY_COLUMNS
        ],
    )

    cleaned = cleaned.with_columns(
        (
            pl.col("high").is_not_null()
            & pl.col("low").is_not_null()
            & (pl.col("high") < pl.col("low"))
        ).alias("invalid_price_range"),
        (
            pl.col("close").is_not_null()
            & (pl.col("close") <= 0)
        ).alias("invalid_close"),
        (
            pl.col("volume").is_not_null()
            & (pl.col("volume") < 0)
        ).alias("invalid_volume"),
    )

    cleaned = cleaned.with_columns(
        pl.struct(DAILY_KEY_COLUMNS)
        .is_duplicated()
        .alias("is_duplicate")
    )

    return (
        cleaned.sort([*DAILY_KEY_COLUMNS, "ingested_at"])
        .unique(
            subset=DAILY_KEY_COLUMNS,
            keep="last",
            maintain_order=True,
        )
        .with_columns(
            pl.lit("dwd_v2").alias("rule_version"),
            pl.lit(True).alias("is_cleaned"),
        )
    )

