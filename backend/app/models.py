from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class StockSummary(BaseModel):
    code: str = Field(min_length=6, max_length=6)
    name: str
    exchange: str
    source: str
    asset_type: Literal["stock", "fund"] = "stock"


class DailyBar(BaseModel):
    stock_code: str = Field(min_length=6, max_length=6)
    trade_date: date
    adjust_type: Literal["raw", "qfq", "hfq"]
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    pre_close: float | None = None
    volume: float | None = None
    amount: float | None = None
    turnover: float | None = None
    pct_change: float | None = None
    source: str
    ingested_at: datetime


class FinancialReport(BaseModel):
    stock_code: str = Field(min_length=6, max_length=6)
    report_date: date
    announced_date: date | None = None
    roe: float | None = None
    net_margin: float | None = None
    gross_margin: float | None = None
    net_profit: float | None = None
    eps: float | None = None
    revenue: float | None = None
    total_shares: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    cash_ratio: float | None = None
    debt_ratio: float | None = None
    asset_to_equity: float | None = None
    asset_turnover: float | None = None
    inventory_turnover: float | None = None
    receivable_turnover: float | None = None
    equity_yoy: float | None = None
    asset_yoy: float | None = None
    net_profit_yoy: float | None = None
    eps_yoy: float | None = None
    cfo_to_revenue: float | None = None
    cfo_to_net_profit: float | None = None
    dupont_roe: float | None = None
    dupont_asset_turnover: float | None = None
    dupont_net_margin: float | None = None
    source: str
    ingested_at: datetime


class ProviderStatus(BaseModel):
    name: str
    configured: bool
    token_required: bool

