from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import polars as pl

from app.adjustments import normalize_adjust_type
from app.models import StockSummary
from app.providers.base import MarketDataProvider, ProviderError
from app.symbols import exchange_for_code, is_fund_code, normalize_stock_code


def _first_column(frame: pl.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _normalize_catalog(raw: pl.DataFrame, asset_type: str) -> pl.DataFrame:
    code_column = _first_column(raw, ["code", "代码", "证券代码"])
    name_column = _first_column(raw, ["name", "名称", "证券简称"])
    if not code_column or not name_column:
        raise ProviderError(f"Unexpected AKShare columns: {raw.columns}")

    normalized = raw.select(
        pl.col(code_column).cast(pl.String).alias("code"),
        pl.col(name_column).cast(pl.String).alias("name"),
    ).drop_nulls()
    return (
        normalized.with_columns(
            pl.col("code").map_elements(
                normalize_stock_code,
                return_dtype=pl.String,
            )
        )
        .with_columns(pl.lit(asset_type).alias("asset_type"))
        .unique("code", keep="first")
    )


class AkshareProvider(MarketDataProvider):
    name = "akshare"
    catalog_ttl_seconds = 6 * 60 * 60

    def __init__(self) -> None:
        self._catalog: pl.DataFrame | None = None
        self._catalog_loaded_at = 0.0

    @property
    def configured(self) -> bool:
        return True

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

    def _search_stocks_sync(self, query: str, limit: int) -> list[StockSummary]:
        raw = self._catalog_sync()
        normalized = raw
        needle = query.strip()
        if needle:
            normalized = normalized.filter(
                pl.col("code").str.contains(needle, literal=True)
                | pl.col("name").str.contains(needle, literal=True)
            )
        return [
            StockSummary(
                code=row["code"],
                name=row["name"],
                exchange=exchange_for_code(row["code"]),
                source=self.name,
                asset_type=row["asset_type"],
            )
            for row in normalized.head(limit).to_dicts()
        ]

    def _catalog_sync(self) -> pl.DataFrame:
        if (
            self._catalog is not None
            and time.monotonic() - self._catalog_loaded_at < self.catalog_ttl_seconds
        ):
            return self._catalog
        import akshare as ak

        catalogs: list[pl.DataFrame] = []
        errors: list[str] = []
        try:
            catalogs.append(
                _normalize_catalog(
                    pl.from_pandas(ak.stock_info_a_code_name()),
                    "stock",
                )
            )
        except Exception as exc:
            errors.append(f"stock catalog: {exc}")
        try:
            catalogs.append(
                _normalize_catalog(
                    pl.from_pandas(
                        ak.fund_etf_category_sina(symbol="ETF基金")
                    ),
                    "fund",
                )
            )
        except Exception as exc:
            errors.append(f"ETF catalog: {exc}")
        if not catalogs:
            raise ProviderError(
                "AKShare search failed: " + "; ".join(errors)
            )

        self._catalog = pl.concat(
            catalogs,
            how="vertical_relaxed",
        ).unique("code", keep="first")
        self._catalog_loaded_at = time.monotonic()
        return self._catalog

    def _daily_bars_sync(
        self,
        code: str,
        start_date: str,
        end_date: str,
        adjust: str,
    ) -> pl.DataFrame:
        normalized_code = normalize_stock_code(code)
        if is_fund_code(normalized_code):
            return self._fund_daily_bars_sync(
                normalized_code,
                start_date,
                end_date,
            )
        normalized_adjust = normalize_adjust_type(adjust)
        akshare_adjust = "" if normalized_adjust == "raw" else normalized_adjust
        try:
            import akshare as ak

            raw = pl.from_pandas(
                ak.stock_zh_a_hist(
                    symbol=normalized_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=akshare_adjust,
                )
            )
        except Exception as exc:
            raise ProviderError(f"AKShare daily data failed: {exc}") from exc
        if raw.is_empty():
            return raw

        column_map = {
            "日期": "trade_date",
            "股票代码": "stock_code",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
            "涨跌幅": "pct_change",
        }
        renamed = raw.rename(
            {source: target for source, target in column_map.items() if source in raw.columns}
        )
        renamed = renamed.with_columns(
            pl.lit(normalized_code).alias("stock_code"),
            pl.lit(normalized_adjust).alias("adjust_type"),
            pl.lit(self.name).alias("source"),
            pl.lit(datetime.now(timezone.utc).replace(tzinfo=None)).alias("ingested_at"),
        )
        return renamed

    def _fund_daily_bars_sync(
        self,
        code: str,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        symbol = f"{exchange_for_code(code).lower()}{code}"
        try:
            import akshare as ak

            raw = pl.from_pandas(ak.fund_etf_hist_sina(symbol=symbol))
        except Exception as exc:
            raise ProviderError(f"AKShare ETF daily data failed: {exc}") from exc
        if raw.is_empty():
            return raw
        return (
            raw.select(
                pl.col("date").cast(pl.Date).alias("trade_date"),
                *[
                    pl.col(name).cast(pl.Float64)
                    for name in ("open", "high", "low", "close", "volume", "amount")
                ],
            )
            .drop_nulls(["trade_date", "close"])
            .filter(
                pl.col("trade_date")
                >= pl.lit(start_date).str.strptime(pl.Date, "%Y%m%d")
            )
            .filter(
                pl.col("trade_date")
                <= pl.lit(end_date).str.strptime(pl.Date, "%Y%m%d")
            )
            .sort("trade_date")
            .unique("trade_date", keep="last")
            .with_columns(
                pl.lit(code).alias("stock_code"),
                pl.lit("raw").alias("adjust_type"),
                pl.col("close").shift(1).alias("pre_close"),
                pl.lit(None, dtype=pl.Float64).alias("turnover"),
                pl.lit(self.name).alias("source"),
                pl.lit(datetime.now(timezone.utc).replace(tzinfo=None)).alias(
                    "ingested_at"
                ),
            )
            .with_columns(
                (
                    pl.col("close") / pl.col("pre_close") - 1.0
                ).mul(100.0).alias("pct_change")
            )
        )

