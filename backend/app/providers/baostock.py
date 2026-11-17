from __future__ import annotations

import asyncio
import re
import socket
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from types import ModuleType

import polars as pl

from app.adjustments import normalize_adjust_type
from app.models import FinancialReport, StockSummary
from app.providers.base import MarketDataProvider, ProviderError
from app.symbols import baostock_symbol, exchange_for_code, normalize_stock_code


class BaostockProvider(MarketDataProvider):
    name = "baostock"
    catalog_ttl_seconds = 6 * 60 * 60
    socket_timeout_seconds = 20

    def __init__(self) -> None:
        self._catalog: list[StockSummary] | None = None
        self._catalog_loaded_at = 0.0
        self._code_cache: dict[str, StockSummary | None] = {}
        self._client: ModuleType | None = None
        self._persistent_session = False

    @property
    def configured(self) -> bool:
        return True

    @contextmanager
    def persistent_session(self):
        """Reuse one BaoStock login across a batch of queries."""
        self._login()
        self._persistent_session = True
        try:
            yield
        finally:
            self._persistent_session = False
            self._logout()

    async def search_stocks(self, query: str, limit: int = 20) -> list[StockSummary]:
        return await asyncio.to_thread(self._search_stocks_sync, query, limit)

    async def daily_bars(
        self,
        code: str,
        start_date: str,
        end_date: str,
        *,
        adjust: str = "qfq",
    ) -> pl.DataFrame:
        return await asyncio.to_thread(
            self._daily_bars_sync, code, start_date, end_date, adjust
        )

    async def financial_report(
        self,
        code: str,
        year: int,
        quarter: int,
    ) -> FinancialReport:
        return await asyncio.to_thread(
            self._financial_report_sync,
            code,
            year,
            quarter,
        )

    def _login(self):
        import baostock as bs

        socket.setdefaulttimeout(self.socket_timeout_seconds)
        if self._client is not None:
            return self._client
        last_error = "unknown error"
        for attempt in range(3):
            try:
                result = bs.login()
                if result.error_code == "0":
                    self._client = bs
                    return bs
                last_error = result.error_msg
            except Exception as exc:
                last_error = str(exc)
            if attempt == 2:
                continue
            if attempt < 2:
                time.sleep(attempt + 1)
        raise ProviderError(f"BaoStock login failed after retries: {last_error}")

    def _logout(self) -> None:
        if self._persistent_session:
            return
        if self._client is None:
            return
        try:
            self._client.logout()
        finally:
            self._client = None

    def _search_stocks_sync(self, query: str, limit: int) -> list[StockSummary]:
        needle = query.strip()
        if re.fullmatch(r"\d{6}", needle):
            item = self._search_by_code_sync(needle)
            return [item] if item is not None else []

        catalog = self._catalog_sync()
        matched = [
            item
            for item in catalog
            if not needle or needle in item.code or needle in item.name
        ]
        return matched[:limit]

    def _search_by_code_sync(self, code: str) -> StockSummary | None:
        if code in self._code_cache:
            return self._code_cache[code]

        bs = self._login()
        try:
            result = bs.query_stock_basic(code=baostock_symbol(code))
            if result.error_code != "0":
                raise ProviderError(result.error_msg)
            rows: list[list[str]] = []
            while result.next():
                rows.append(result.get_row_data())
            columns = list(result.fields)
        finally:
            self._logout()

        if not rows:
            self._code_cache[code] = None
            return None
        raw = pl.DataFrame(rows, schema=columns, orient="row")
        if "type" in raw.columns:
            raw = raw.filter(pl.col("type").cast(pl.String) == "1")
        if raw.is_empty():
            self._code_cache[code] = None
            return None
        row = raw.row(0, named=True)
        item = StockSummary(
            code=code,
            name=row.get("code_name") or code,
            exchange=exchange_for_code(code),
            source=self.name,
        )
        self._code_cache[code] = item
        return item

    def _catalog_sync(self) -> list[StockSummary]:
        if (
            self._catalog is not None
            and time.monotonic() - self._catalog_loaded_at < self.catalog_ttl_seconds
        ):
            return self._catalog

        bs = self._login()
        try:
            result = bs.query_stock_basic()
            if result.error_code != "0":
                raise ProviderError(result.error_msg)
            rows: list[dict[str, str]] = []
            while result.next():
                rows.append(result.get_row_data())
            columns = list(result.fields)
        finally:
            self._logout()

        raw = pl.DataFrame(rows, schema=columns, orient="row")
        if "type" in raw.columns:
            raw = raw.filter(pl.col("type").cast(pl.String) == "1")
        if "status" in raw.columns:
            raw = raw.filter(
                pl.col("status").cast(pl.String).is_in(["1", ""])
            )

        catalog: list[StockSummary] = []
        for row in raw.to_dicts():
            code = normalize_stock_code(row["code"])
            catalog.append(
                StockSummary(
                    code=code,
                    name=row.get("code_name") or code,
                    exchange=exchange_for_code(code),
                    source=self.name,
                )
            )
        self._catalog = catalog
        self._catalog_loaded_at = time.monotonic()
        return catalog

    def _financial_report_sync(
        self,
        code: str,
        year: int,
        quarter: int,
    ) -> FinancialReport:
        normalized_code = normalize_stock_code(code)
        symbol = baostock_symbol(normalized_code)
        bs = self._login()
        try:
            profit = self._financial_rows(
                bs.query_profit_data,
                symbol,
                year,
                quarter,
            )
            operation = self._financial_rows(
                bs.query_operation_data,
                symbol,
                year,
                quarter,
            )
            growth = self._financial_rows(
                bs.query_growth_data,
                symbol,
                year,
                quarter,
            )
            balance = self._financial_rows(
                bs.query_balance_data,
                symbol,
                year,
                quarter,
            )
            cash_flow = self._financial_rows(
                bs.query_cash_flow_data,
                symbol,
                year,
                quarter,
            )
            dupont = self._financial_rows(
                bs.query_dupont_data,
                symbol,
                year,
                quarter,
            )
        finally:
            self._logout()

        if not profit:
            raise ProviderError(
                f"No BaoStock financial report for {normalized_code} "
                f"{year}Q{quarter}"
            )

        merged: dict[str, str] = {}
        for row in (operation, growth, balance, cash_flow, dupont):
            merged.update(
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"code", "pubDate", "statDate"}
                }
            )
        merged.update(profit)

        def number(key: str) -> float | None:
            raw = merged.get(key, "")
            if raw in {"", None}:
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None

        return FinancialReport(
            stock_code=normalized_code,
            report_date=merged["statDate"],
            announced_date=merged.get("pubDate") or None,
            roe=number("roeAvg"),
            net_margin=number("npMargin"),
            gross_margin=number("gpMargin"),
            net_profit=number("netProfit"),
            eps=number("epsTTM"),
            revenue=number("MBRevenue"),
            total_shares=number("totalShare"),
            current_ratio=number("currentRatio"),
            quick_ratio=number("quickRatio"),
            cash_ratio=number("cashRatio"),
            debt_ratio=number("liabilityToAsset"),
            asset_to_equity=number("assetToEquity"),
            asset_turnover=number("AssetTurnRatio"),
            inventory_turnover=number("INVTurnRatio"),
            receivable_turnover=number("NRTurnRatio"),
            equity_yoy=number("YOYEquity"),
            asset_yoy=number("YOYAsset"),
            net_profit_yoy=number("YOYNI"),
            eps_yoy=number("YOYEPSBasic"),
            cfo_to_revenue=number("CFOToOR"),
            cfo_to_net_profit=number("CFOToNP"),
            dupont_roe=number("dupontROE"),
            dupont_asset_turnover=number("dupontAssetTurn"),
            dupont_net_margin=number("dupontNitogr"),
            source=self.name,
            ingested_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )

    @staticmethod
    def _financial_rows(method, symbol: str, year: int, quarter: int) -> dict[str, str]:
        result = method(code=symbol, year=year, quarter=quarter)
        if result.error_code != "0":
            raise ProviderError(result.error_msg)
        if not result.next():
            return {}
        return dict(zip(result.fields, result.get_row_data(), strict=False))

    def _daily_bars_sync(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pl.DataFrame:
        normalized_code = normalize_stock_code(code)
        normalized_adjust = normalize_adjust_type(adjust)
        bs = self._login()
        adjust_flag = {"raw": "3", "qfq": "2", "hfq": "1"}[normalized_adjust]
        fields = "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg"
        try:
            result = bs.query_history_k_data_plus(
                baostock_symbol(normalized_code),
                fields,
                start_date=f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}",
                end_date=f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}",
                frequency="d",
                adjustflag=adjust_flag,
            )
            if result.error_code != "0":
                raise ProviderError(result.error_msg)
            rows: list[list[str]] = []
            while result.next():
                rows.append(result.get_row_data())
            columns = list(result.fields)
        finally:
            self._logout()

        raw = pl.DataFrame(rows, schema=columns, orient="row")
        if raw.is_empty():
            return raw
        renamed = raw.rename(
            {
                "date": "trade_date",
                "preclose": "pre_close",
                "turn": "turnover",
                "pctChg": "pct_change",
            }
        )
        return renamed.with_columns(
            pl.lit(normalized_code).alias("stock_code"),
            pl.lit(normalized_adjust).alias("adjust_type"),
            pl.lit(self.name).alias("source"),
            pl.lit(datetime.now(timezone.utc).replace(tzinfo=None)).alias("ingested_at"),
        )

