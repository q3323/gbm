from datetime import UTC, datetime, timedelta

import polars as pl

from app.indicators import compute_indicators, select_primary_daily


def trend_frame(days: int = 80) -> pl.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    closes = [10.0 + index * 0.1 for index in range(days)]
    return pl.DataFrame(
        {
            "stock_code": ["000001"] * days,
            "trade_date": [
                (start + timedelta(days=index)).date() for index in range(days)
            ],
            "adjust_type": ["qfq"] * days,
            "open": [value - 0.05 for value in closes],
            "high": [value + 0.15 for value in closes],
            "low": [value - 0.15 for value in closes],
            "close": closes,
            "volume": [1000.0 + index * 10 for index in range(days)],
            "turnover": [1.0] * days,
            "source": ["baostock"] * days,
            "ingested_at": [datetime.now(UTC)] * days,
        }
    )


def test_indicator_columns_and_ranges() -> None:
    result = compute_indicators(trend_frame())
    expected = {
        "return_1d",
        "ma_5",
        "ma_20",
        "ema_12",
        "macd",
        "macd_signal",
        "rsi_14",
        "kdj_k",
        "boll_upper_20",
        "atr_14",
        "volatility_20",
        "indicator_version",
    }
    assert expected.issubset(result.columns)
    latest = result.tail(1).to_dicts()[0]
    assert 0 <= latest["rsi_14"] <= 100
    assert latest["ma_5"] > latest["ma_20"]
    assert latest["indicator_version"] == "dws_v1"


def test_primary_source_prefers_tushare() -> None:
    frame = trend_frame(3)
    tushare = frame.with_columns(pl.lit("tushare").alias("source"))
    combined = pl.concat([frame, tushare])
    selected = select_primary_daily(combined)
    assert selected.height == 3
    assert selected["source"].unique().to_list() == ["tushare"]


def test_primary_source_only_uses_qfq_for_indicators() -> None:
    frame = trend_frame(3)
    raw = frame.with_columns(pl.lit("raw").alias("adjust_type"))
    combined = pl.concat([frame, raw])
    selected = select_primary_daily(combined)
    assert selected.height == 3
    assert selected["adjust_type"].unique().to_list() == ["qfq"]

