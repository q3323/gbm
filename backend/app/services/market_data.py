from __future__ import annotations

from app.config import Settings
from app.models import FinancialReport, ProviderStatus, StockSummary
from app.providers.akshare import AkshareProvider
from app.providers.baostock import BaostockProvider
from app.providers.base import MarketDataProvider, ProviderError
from app.providers.tushare import TushareProvider


class MarketDataService:
    def __init__(self, settings: Settings) -> None:
        self.providers: dict[str, MarketDataProvider] = {
            "tushare": TushareProvider(settings.tushare_token),
            "baostock": BaostockProvider(),
            "akshare": AkshareProvider(),
        }

    def statuses(self) -> list[ProviderStatus]:
        return [
            ProviderStatus(
                name=provider.name,
                configured=provider.configured,
                token_required=provider.token_required,
            )
            for provider in self.providers.values()
        ]

    async def search(
        self,
        query: str,
        *,
        source: str = "auto",
        limit: int = 20,
    ) -> list[StockSummary]:
        providers = self._provider_order(source)
        errors: list[str] = []
        results: list[StockSummary] = []
        seen: set[str] = set()
        for provider in providers:
            if not provider.configured:
                continue
            try:
                matches = await provider.search_stocks(query, limit)
            except (ProviderError, ValueError) as exc:
                errors.append(f"{provider.name}: {exc}")
                continue
            for item in matches:
                if item.code in seen:
                    continue
                seen.add(item.code)
                results.append(item)
                if len(results) >= limit:
                    return results
        if results:
            return results
        raise ProviderError("; ".join(errors) or "No configured stock provider")

    async def daily_bars(
        self,
        code: str,
        start_date: str,
        end_date: str,
        *,
        source: str = "auto",
        adjust: str = "qfq",
    ):
        providers = self._provider_order(source)
        errors: list[str] = []
        for provider in providers:
            if not provider.configured:
                continue
            try:
                frame = await provider.daily_bars(
                    code, start_date, end_date, adjust=adjust
                )
                if frame.height:
                    return frame
            except ProviderError as exc:
                errors.append(f"{provider.name}: {exc}")
        raise ProviderError("; ".join(errors) or "No daily data available")

    async def financial_report(
        self,
        code: str,
        year: int,
        quarter: int,
        *,
        source: str = "auto",
    ) -> FinancialReport:
        providers = self._provider_order(source)
        errors: list[str] = []
        for provider in providers:
            if not provider.configured:
                continue
            try:
                return await provider.financial_report(code, year, quarter)
            except ProviderError as exc:
                errors.append(f"{provider.name}: {exc}")
        raise ProviderError("; ".join(errors) or "No financial report available")

    def _provider_order(self, source: str) -> list[MarketDataProvider]:
        if source != "auto":
            provider = self.providers.get(source)
            if provider is None:
                raise ProviderError(f"Unknown provider: {source}")
            return [provider]
        return [
            self.providers["tushare"],
            self.providers["baostock"],
            self.providers["akshare"],
        ]

