from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import polars as pl

from app.services.time_series import TimeSeriesService


def analysis_frame(days: int = 140) -> pl.DataFrame:
    dates = pl.date_range(
        start=datetime(2026, 1, 2).date(),
        end=datetime(2026, 6, 30).date(),
        interval="1d",
        eager=True,
    ).head(days)
    closes = [
        10.0 + index * 0.02 + math.sin(index / 5) * 0.15
        for index in range(days)
    ]
    return pl.DataFrame(
        {
            "stock_code": ["000001"] * days,
            "trade_date": dates,
            "adjust_type": ["qfq"] * days,
            "close": closes,
            "volume": [1000.0 + index for index in range(days)],
            "source": ["baostock"] * days,
            "ingested_at": [datetime.now(UTC).replace(tzinfo=None)] * days,
        }
    )


def test_time_series_analysis_has_core_sections() -> None:
    result = TimeSeriesService().analyze(
        "000001",
        analysis_frame(),
        horizon=3,
        seed=7,
        max_arima_order=1,
        include_volatility=False,
    )
    report = result.report
    assert report["metadata"]["seed"] == 7
    assert report["metadata"]["observations"] == 140
    assert "adf" in report["stationarity"]
    assert len(report["autocorrelation"]["acf"]) > 10
    assert report["decomposition"]["available"] is True
    assert report["forecast"]["order"]["d"] in {0, 1}
    assert len(report["forecast"]["points"]) == 3
    assert np.isfinite(report["summary"]["total_return"])
