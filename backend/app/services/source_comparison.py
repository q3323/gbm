from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from app.adjustments import normalize_adjust_type
from app.cleaning import clean_daily_bars
from app.providers.base import ProviderError
from app.quality import compare_daily_sources, source_comparison_report
from app.services.market_data import MarketDataService
from app.storage import ParquetStore
from app.symbols import normalize_stock_code


@dataclass(frozen=True)
class SourceComparisonResult:
    report: dict[str, Any]
    rows: pl.DataFrame
    path: Path


class SourceComparisonService:
    def __init__(
        self,
        market_data: MarketDataService,
        store: ParquetStore,
    ) -> None:
        self.market_data = market_data
        self.store = store

    async def compare(
        self,
        code: str,
        start_date: str,
        end_date: str,
        *,
        primary_source: str = "tushare",
        comparison_source: str = "baostock",
        adjust_type: str = "raw",
        close_tolerance: float = 0.01,
        volume_tolerance: float = 0.02,
    ) -> SourceComparisonResult:
        normalized_code = normalize_stock_code(code)
        normalized_adjust = normalize_adjust_type(adjust_type)
        if primary_source == comparison_source:
            raise ValueError("Primary and comparison sources must be different")

        errors: list[str] = []
        primary = await self._fetch(
            normalized_code,
            start_date,
            end_date,
            primary_source,
            normalized_adjust,
            errors,
        )
        comparison = await self._fetch(
            normalized_code,
            start_date,
            end_date,
            comparison_source,
            normalized_adjust,
            errors,
        )
        if primary.is_empty() or comparison.is_empty():
            detail = "; ".join(errors) or "Both providers returned empty data"
            raise ProviderError(detail)

        primary = clean_daily_bars(primary)
        comparison = clean_daily_bars(comparison)
        rows = compare_daily_sources(
            primary,
            comparison,
            close_tolerance=close_tolerance,
            volume_tolerance=volume_tolerance,
        )
        if rows.is_empty():
            raise ValueError("No overlapping trading dates for the selected sources")
        report = source_comparison_report(
            rows,
            primary_source=primary_source,
            comparison_source=comparison_source,
            primary_rows=primary.height,
            comparison_rows=comparison.height,
        )
        report["adjust_type"] = normalized_adjust
        report["close_tolerance"] = close_tolerance
        report["volume_tolerance"] = volume_tolerance
        path = self.store.save_source_comparison(
            normalized_code,
            normalized_adjust,
            rows,
            report,
        )
        return SourceComparisonResult(report=report, rows=rows, path=path)

    async def _fetch(
        self,
        code: str,
        start_date: str,
        end_date: str,
        source: str,
        adjust_type: str,
        errors: list[str],
    ) -> pl.DataFrame:
        try:
            return await self.market_data.daily_bars(
                code,
                start_date,
                end_date,
                source=source,
                adjust="none" if adjust_type == "raw" else adjust_type,
            )
        except ProviderError as exc:
            errors.append(f"{source}: {exc}")
            return pl.DataFrame()
