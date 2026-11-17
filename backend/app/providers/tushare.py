from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import polars as pl

from app.adjustments import normalize_adjust_type
from app.models import StockSummary
from app.providers.base import MarketDataProvider, ProviderError
from app.symbols import exchange_for_code, normalize_stock_code, tushare_symbol


class TushareProvider(MarketDataProvider):
    name = "tushare"
    token_required = True

    def __init__(self, token: str) -> None:
        self.token = token.strip()

    @property
    def configured(self) -> bool:
        return bool(self.token)

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

    def _pro(self):
        if not self.configured:
            raise ProviderError("Tushare token is not configured")
        try:
            import tushare as ts

            return ts.pro_api(self.token)
        except Exception as exc:
            raise ProviderError(f"Tushare initialization failed: {exc}") from exc

    def _search_stocks_sync(self, query: str, limit: int) -> list[StockSummary]:
        pro = self._pro()
        try:
            raw = pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,name",
            )
        except Exception as exc:
            raise ProviderError(f"Tushare search failed: {exc}") from exc
        frame = pl.from_pandas(raw)
        needle = query.strip()
        if needle:
            frame = frame.filter(
                pl.col("ts_code").str.contains(needle, literal=True)
                | pl.col("name").str.contains(needle, literal=True)
            )
        stocks: list[StockSummary] = []
        for row in frame.head(limit).to_dicts():
            code = normalize_stock_code(row["ts_code"])
            stocks.append(
                StockSummary(
                    code=code,
                    name=row["name"] or code,
                    exchange=exchange_for_code(code),
                    source=self.name,
                )
            )
        return stocks

    def _daily_bars_sync(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pl.DataFrame:
        normalized_adjust = normalize_adjust_type(adjust)
        try:
            if normalized_adjust == "raw":
                raw = self._pro().daily(
                    ts_code=tushare_symbol(code),
                    start_date=start_date,
                    end_date=end_date,
                )
            else:
                import tushare as ts

                raw = ts.pro_bar(
                    ts_code=tushare_symbol(code),
                    adj=normalized_adjust,
                    start_date=start_date,
                    end_date=end_date,
                    asset="E",
                )
        except Exception as exc:
            raise ProviderError(f"Tushare daily data failed: {exc}") from exc
        return self._normalize_daily_frame(raw, code, normalized_adjust)

    def _normalize_daily_frame(
        self,
        raw,
        code: str,
        adjust_type: str,
    ) -> pl.DataFrame:
        frame = raw if isinstance(raw, pl.DataFrame) else pl.from_pandas(raw)
        if frame.is_empty():
            return frame
        frame = frame.rename(
            {
                "vol": "volume",
                "pct_chg": "pct_change",
            }
        )
        if "trade_date" in frame.columns:
            frame = frame.with_columns(
                pl.col("trade_date")
                .cast(pl.String)
                .str.strptime(pl.Date, "%Y%m%d", strict=False)
                .dt.strftime("%Y-%m-%d")
            )
        numeric_columns = [
            name
            for name in [
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "pct_change",
                "volume",
                "amount",
            ]
            if name in frame.columns
        ]
        if numeric_columns:
            frame = frame.with_columns(
                [
                    pl.col(name).cast(pl.Float64, strict=False)
                    for name in numeric_columns
                ]
            )
        unit_columns = []
        if "volume" in frame.columns:
            unit_columns.append(
                (pl.col("volume") * 100.0).alias("volume")
            )
        if "amount" in frame.columns:
            unit_columns.append(
                (pl.col("amount") * 1000.0).alias("amount")
            )
        if unit_columns:
            frame = frame.with_columns(unit_columns)
        return frame.with_columns(
            pl.lit(normalize_stock_code(code)).alias("stock_code"),
            pl.lit(adjust_type).alias("adjust_type"),
            pl.lit(self.name).alias("source"),
            pl.lit(datetime.now(timezone.utc).replace(tzinfo=None)).alias("ingested_at"),
        )

