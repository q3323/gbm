from datetime import UTC, datetime

import polars as pl

from app.config import Settings
from app.storage import ParquetStore, new_run_record


def test_store_ingest_and_load(tmp_path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        tushare_token="",
        api_token="",
        host="127.0.0.1",
        port=8000,
        log_level="INFO",
    )
    store = ParquetStore(settings)
    frame = pl.DataFrame(
        {
            "stock_code": ["000001", "000001"],
            "trade_date": ["2026-11-10", "2026-11-11"],
            "adjust_type": ["qfq", "qfq"],
            "open": [10.0, 10.2],
            "high": [10.5, 10.4],
            "low": [9.9, 10.0],
            "close": [10.2, 10.1],
            "pre_close": [10.0, 10.2],
            "volume": [1000.0, 900.0],
            "amount": [10000.0, 9000.0],
            "turnover": [1.0, 0.9],
            "pct_change": [2.0, -0.98],
            "source": ["baostock", "baostock"],
            "ingested_at": [datetime.now(UTC), datetime.now(UTC)],
        }
    )

    result = store.ingest_daily("000001", "baostock", frame)
    loaded = store.load_dwd_daily("000001", "2026-11-10", "2026-11-11")

    assert result.ods_path.exists()
    assert result.dwd_path.exists()
    assert result.dws_path.exists()
    assert loaded.height == 2
    assert store.load_dwd_daily("000001", "20261110", "20261111").height == 2
    assert store.load_quality_report("000001") is not None
    indicators = store.load_dws_indicators("000001")
    assert indicators.height == 2
    assert indicators.schema["trade_date"] == pl.Date


def test_pipeline_run_history(tmp_path) -> None:
    settings = Settings(tmp_path, "", "", "127.0.0.1", 8000, "INFO")
    store = ParquetStore(settings)
    now = datetime.now(UTC)
    store.record_pipeline_run(
        new_run_record(
            run_id="run-1",
            code="000001",
            source="baostock",
            status="success",
            started_at=now,
            finished_at=now,
            row_count=10,
        )
    )

    runs = store.load_pipeline_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "run-1"

