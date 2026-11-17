from __future__ import annotations

import polars as pl

from app.providers.tushare import TushareProvider


def test_tushare_daily_units_and_fields_are_normalized() -> None:
    raw = pl.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20261112"],
            "open": ["10.0"],
            "high": ["10.5"],
            "low": ["9.9"],
            "close": ["10.2"],
            "pre_close": ["10.0"],
            "pct_chg": ["2.0"],
            "vol": ["1234"],
            "amount": ["5678"],
        }
    )
    frame = TushareProvider("token")._normalize_daily_frame(
        raw,
        "000001",
        "raw",
    )
    row = frame.row(0, named=True)
    assert row["adjust_type"] == "raw"
    assert row["pct_change"] == 2.0
    assert row["volume"] == 123400.0
    assert row["amount"] == 5678000.0
    assert str(row["trade_date"]) == "2026-11-12"
