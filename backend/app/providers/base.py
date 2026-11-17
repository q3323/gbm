from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

from app.models import FinancialReport, StockSummary


class ProviderError(RuntimeError):
    pass


class MarketDataProvider(ABC):
    name: str
    token_required: bool = False

    @property
    @abstractmethod
    def configured(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def search_stocks(self, query: str, limit: int = 20) -> list[StockSummary]:
        raise NotImplementedError

    @abstractmethod
    async def daily_bars(
        self,
        code: str,
        start_date: str,
        end_date: str,
        *,
        adjust: str = "qfq",
    ) -> pl.DataFrame:
        raise NotImplementedError

    async def financial_report(
        self,
        code: str,
        year: int,
        quarter: int,
    ) -> FinancialReport:
        raise ProviderError(f"{self.name} does not support financial reports")

