from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import date, timedelta

import polars as pl
from fastapi import APIRouter, HTTPException, Query, Request

from app.indicators import compute_indicators
from app.models import DailyBar, FinancialReport, StockSummary
from app.providers.base import ProviderError
from app.services.backtest import BacktestConfig, StrategyBacktestService
from app.services.gbm import GBMDataError, GBMService
from app.services.market_data import MarketDataService
from app.services.time_series import TimeSeriesService


router = APIRouter(prefix="/api/stocks", tags=["stocks"])


def _service(request: Request) -> MarketDataService:
    return request.app.state.market_data


@router.get("/search", response_model=list[StockSummary])
async def search_stocks(
    request: Request,
    q: str = Query(default="", max_length=50),
    source: str = Query(default="auto"),
    limit: int = Query(default=20, ge=1, le=50),
) -> list[StockSummary]:
    try:
        return await _service(request).search(q, source=source, limit=limit)
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/{code}/daily", response_model=list[DailyBar])
async def daily_bars(
    request: Request,
    code: str,
    start_date: str = Query(min_length=8, max_length=8),
    end_date: str = Query(min_length=8, max_length=8),
    source: str = Query(default="auto"),
    adjust: str = Query(default="qfq", pattern="^(qfq|hfq|none)$"),
) -> list[DailyBar]:
    try:
        frame = await _service(request).daily_bars(
            code,
            start_date,
            end_date,
            source=source,
            adjust="" if adjust == "none" else adjust,
        )
    except (ProviderError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [DailyBar.model_validate(row) for row in frame.to_dicts()]


@router.get("/{code}/indicators")
async def indicators(
    request: Request,
    code: str,
    start_date: str | None = Query(default=None, min_length=8, max_length=8),
    end_date: str | None = Query(default=None, min_length=8, max_length=8),
) -> list[dict]:
    try:
        normalized_start = _iso_date(start_date) if start_date else None
        normalized_end = _iso_date(end_date) if end_date else None
        frame = request.app.state.store.load_dws_indicators(
            code,
            normalized_start,
            normalized_end,
        )
        if frame.is_empty():
            today = date.today()
            fetch_start = start_date or (
                today - timedelta(days=730)
            ).strftime("%Y%m%d")
            fetch_end = end_date or today.strftime("%Y%m%d")
            daily = await _service(request).daily_bars(
                code,
                fetch_start,
                fetch_end,
                source="auto",
                adjust="qfq",
            )
            if daily.is_empty():
                raise HTTPException(
                    status_code=404,
                    detail="No market data available for indicators",
                )
            computed = await asyncio.to_thread(
                compute_indicators,
                daily,
            )
            if computed.is_empty():
                raise HTTPException(
                    status_code=404,
                    detail="Unable to compute indicators",
                )
            request.app.state.store.save_dws_indicators(code, computed)
            frame = request.app.state.store.load_dws_indicators(
                code,
                normalized_start,
                normalized_end,
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if frame.is_empty():
        raise HTTPException(
            status_code=404,
            detail="No indicator data for this stock",
        )
    return frame.to_dicts()


@router.get("/{code}/analysis")
async def time_series_analysis(
    request: Request,
    code: str,
    start_date: str | None = Query(default=None, min_length=8, max_length=8),
    end_date: str | None = Query(default=None, min_length=8, max_length=8),
    horizon: int = Query(default=5, ge=1, le=30),
    seed: int = Query(default=42, ge=0, le=2_147_483_647),
    seasonal_period: int = Query(default=5, ge=2, le=60),
    model: str = Query(default="arima", pattern="^(arima|gbm|compare)$"),
    fundamentals: bool = Query(default=True),
) -> dict:
    try:
        normalized_code = code.zfill(6)
        normalized_start = _iso_date(start_date) if start_date else None
        normalized_end = _iso_date(end_date) if end_date else None
        arima_report: dict | None = None
        gbm_report: dict | None = None

        if model in {"arima", "compare"}:
            daily = request.app.state.store.load_dwd_daily(
                normalized_code,
                normalized_start,
                normalized_end,
            )
            if daily.is_empty():
                raise HTTPException(
                    status_code=404,
                    detail="No daily data for this stock",
                )
            arima_result = await asyncio.to_thread(
                TimeSeriesService().analyze,
                normalized_code,
                daily,
                start_date=normalized_start,
                end_date=normalized_end,
                horizon=horizon,
                seed=seed,
                seasonal_period=seasonal_period,
            )
            arima_report = arima_result.report
            if model == "arima":
                path = request.app.state.store.save_analysis_report(
                    normalized_code,
                    arima_report,
                )
                return {**arima_report, "report_path": str(path)}

        all_daily = request.app.state.store.load_all_dwd_daily(
            normalized_start,
            normalized_end,
        )
        financial = (
            request.app.state.store.load_all_financial_reports()
            if fundamentals
            else None
        )
        if all_daily.is_empty():
            if model == "gbm":
                raise HTTPException(
                    status_code=404,
                    detail="No multi-stock daily data available for GBM",
                )
            gbm_report = {
                "available": False,
                "reason": "No multi-stock daily data available for GBM",
                "details": {},
            }
        try:
            if all_daily.is_empty():
                raise GBMDataError(
                    "No multi-stock daily data available for GBM",
                    {},
                )
            gbm_result = await asyncio.to_thread(
                GBMService().analyze,
                all_daily,
                normalized_code,
                seed=seed,
                horizon=horizon,
                financial=financial,
            )
            gbm_report = gbm_result.report
        except GBMDataError as exc:
            if model == "gbm":
                raise HTTPException(
                    status_code=422,
                    detail={
                        "message": str(exc),
                        "details": exc.details,
                    },
                ) from exc
            gbm_report = {
                "available": False,
                "reason": str(exc),
                "details": exc.details,
            }
        except ValueError as exc:
            if model == "gbm":
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            gbm_report = {
                "available": False,
                "reason": str(exc),
                "details": {},
            }

        if model == "gbm":
            path = request.app.state.store.save_gbm_report(
                normalized_code,
                gbm_report,
            )
            return {**gbm_report, "report_path": str(path)}

        return {
            "metadata": {
                "stock_code": normalized_code,
                "model": "compare",
                "generated_on": date.today().isoformat(),
            },
            "arima": arima_report,
            "gbm": gbm_report,
        }
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Time-series analysis failed: {exc}",
        ) from exc


@router.get("/{code}/analysis/backtest")
async def strategy_backtest(
    request: Request,
    code: str,
    start_date: str | None = Query(default=None, min_length=8, max_length=8),
    end_date: str | None = Query(default=None, min_length=8, max_length=8),
    model: str = Query(default="gbm", pattern="^gbm$"),
    top_fraction: float = Query(default=0.2, gt=0, le=1),
    rebalance_days: int = Query(default=5, ge=1, le=30),
    seed: int = Query(default=42, ge=0, le=2_147_483_647),
    commission_each_side: float = Query(default=0.0003, ge=0, le=0.01),
    stamp_duty_sell: float = Query(default=0.0005, ge=0, le=0.01),
    slippage_each_side: float = Query(default=0.001, ge=0, le=0.01),
    fundamentals: bool = Query(default=True),
) -> dict:
    try:
        normalized_code = code.zfill(6)
        normalized_start = _iso_date(start_date) if start_date else None
        normalized_end = _iso_date(end_date) if end_date else None
        all_daily = request.app.state.store.load_all_dwd_daily(
            normalized_start,
            normalized_end,
        )
        financial = (
            request.app.state.store.load_all_financial_reports()
            if fundamentals
            else None
        )
        if all_daily.is_empty():
            raise HTTPException(
                status_code=404,
                detail="No multi-stock daily data available for backtest",
            )
        gbm_result = await asyncio.to_thread(
            GBMService().analyze,
            all_daily,
            normalized_code,
            seed=seed,
            horizon=rebalance_days,
            financial=financial,
        )
        backtest_result = await asyncio.to_thread(
            StrategyBacktestService().run,
            gbm_result.out_of_sample,
            BacktestConfig(
                top_fraction=top_fraction,
                holding_days=rebalance_days,
                rebalance_days=rebalance_days,
                commission_each_side=commission_each_side,
                stamp_duty_sell=stamp_duty_sell,
                slippage_each_side=slippage_each_side,
                seed=seed,
            ),
        )
        report = {
            **backtest_result.report,
            "model_evaluation": gbm_result.report,
        }
        path = request.app.state.store.save_backtest_report(
            normalized_code,
            model,
            report,
        )
        return {**report, "report_path": str(path)}
    except HTTPException:
        raise
    except GBMDataError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(exc),
                "details": exc.details,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Backtest failed: {exc}",
        ) from exc


@router.get("/{code}/model/ensemble")
async def ensemble_model_result(
    request: Request,
    code: str,
    horizon: int = Query(default=10, ge=1, le=60),
    model: str = Query(default="gbm_fund", pattern="^(gbm|gbm_fund|gbm_v3|gbm_v3_fund)$"),
) -> dict:
    report = request.app.state.store.load_ensemble_result(
        code,
        model,
        horizon,
    )
    if report is None:
        raise HTTPException(status_code=404, detail="No ensemble model result for this stock")
    return {**report, "report_path": str(
        request.app.state.store.data_dir
        / "ads"
        / "ensemble"
        / model
        / f"stock_code={code.zfill(6)}"
        / f"horizon={horizon}"
        / "latest.json"
    )}


def _iso_date(value: str) -> str:
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"Invalid date: {value}")
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


@router.get("/{code}/financial", response_model=FinancialReport)
async def financial_report(
    request: Request,
    code: str,
    year: int = Query(ge=1990, le=2100),
    quarter: int = Query(ge=1, le=4),
    source: str = Query(default="auto"),
) -> FinancialReport:
    try:
        cached = request.app.state.store.load_financial_reports(code)
        target_date = _quarter_end_date(year, quarter)
        if not cached.is_empty():
            selected = cached.filter(
                pl.col("report_date") == pl.lit(target_date)
            )
            if source != "auto":
                selected = selected.filter(pl.col("source") == source)
            if not selected.is_empty():
                return FinancialReport.model_validate(selected.row(0, named=True))

        report = await _service(request).financial_report(
            code,
            year,
            quarter,
            source=source,
        )
        request.app.state.store.save_financial_report(report)
        return report
    except (ProviderError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{code}/financial/latest", response_model=FinancialReport)
async def latest_financial_report(
    request: Request,
    code: str,
    source: str = Query(default="auto"),
) -> FinancialReport:
    cached = request.app.state.store.load_financial_reports(code)
    if not cached.is_empty():
        selected = cached
        if source != "auto":
            selected = selected.filter(pl.col("source") == source)
        if not selected.is_empty():
            return FinancialReport.model_validate(selected.row(0, named=True))

    for year, quarter in _latest_report_periods(date.today()):
        try:
            report = await _service(request).financial_report(
                code,
                year,
                quarter,
                source=source,
            )
        except (ProviderError, ValueError):
            continue
        request.app.state.store.save_financial_report(report)
        return report

    raise HTTPException(
        status_code=404,
        detail="No financial report found in the recent quarters",
    )


def _quarter_end_date(year: int, quarter: int) -> date:
    return {
        1: date(year, 3, 31),
        2: date(year, 6, 30),
        3: date(year, 9, 30),
        4: date(year, 12, 31),
    }[quarter]


def _latest_report_periods(today: date, limit: int = 4) -> Iterator[tuple[int, int]]:
    absolute_quarter = today.year * 4 + (today.month - 1) // 3
    for offset in range(1, limit + 1):
        period = absolute_quarter - offset
        yield period // 4, period % 4 + 1






