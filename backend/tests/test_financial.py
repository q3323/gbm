from datetime import UTC, datetime

import polars as pl

from app.api.stocks import _latest_report_periods
from app.config import Settings
from app.models import FinancialReport
from app.quality import financial_quality_report
from app.storage import ParquetStore


def test_financial_quality_and_storage(tmp_path) -> None:
    report = FinancialReport(
        stock_code="000001",
        report_date="2025-12-31",
        announced_date="2026-03-21",
        roe=0.0815,
        net_margin=0.3243,
        gross_margin=None,
        net_profit=42_633_000_000,
        eps=2.1969,
        revenue=131_442_000_000,
        total_shares=19_405_918_198,
        current_ratio=None,
        quick_ratio=None,
        cash_ratio=None,
        debt_ratio=0.0091,
        asset_to_equity=10.75,
        asset_turnover=0.0225,
        inventory_turnover=None,
        receivable_turnover=None,
        equity_yoy=0.1139,
        asset_yoy=0.0271,
        net_profit_yoy=-0.0421,
        eps_yoy=-0.0372,
        cfo_to_revenue=2.403,
        cfo_to_net_profit=7.409,
        dupont_roe=0.0815,
        dupont_asset_turnover=0.0225,
        dupont_net_margin=0.3243,
        source="baostock",
        ingested_at=datetime.now(UTC).replace(tzinfo=None),
    )
    settings = Settings(tmp_path, "", "", "127.0.0.1", 8000, "INFO")
    store = ParquetStore(settings)
    path = store.save_financial_report(report)

    assert path.exists()
    frame = store.load_financial_reports("000001")
    assert frame.height == 1
    quality = financial_quality_report(frame)
    assert quality["key_field_missing_rate"]["revenue"] == 0
    assert quality["key_field_missing_rate"]["roe"] == 0


def test_financial_quality_flags_missing_key_fields() -> None:
    frame = pl.DataFrame(
        {
            "revenue": [None],
            "net_profit": [10.0],
            "roe": [None],
            "debt_ratio": [0.2],
            "eps": [1.0],
        }
    )
    report = financial_quality_report(frame)
    assert report["status"] == "error"
    assert report["key_field_missing_rate"]["revenue"] == 1.0


def test_latest_report_periods_skip_incomplete_quarter() -> None:
    assert list(_latest_report_periods(datetime(2026, 11, 14).date())) == [
        (2026, 3),
        (2026, 2),
        (2026, 1),
        (2025, 4),
    ]

