from __future__ import annotations

import polars as pl
from fastapi import APIRouter, HTTPException, Query, Request, UploadFile

from app.api.stocks import _service
from app.cleaning import clean_daily_bars
from app.providers.base import ProviderError
from app.quality import daily_quality_report
from app.services.source_comparison import SourceComparisonService


router = APIRouter(prefix="/api/quality", tags=["quality"])


@router.get("/providers")
async def provider_status(request: Request):
    return _service(request).statuses()


@router.get("/stocks/{code}")
async def stock_quality(request: Request, code: str) -> dict:
    try:
        report = request.app.state.store.load_quality_report(code, "clean")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if report is None:
        raise HTTPException(status_code=404, detail="No quality report for this stock")
    return report


@router.get("/pipeline-runs")
async def pipeline_runs(request: Request, limit: int = 50) -> list[dict]:
    return request.app.state.store.load_pipeline_runs(max(1, min(limit, 200)))


@router.get("/sources/compare")
async def compare_sources(
    request: Request,
    code: str = Query(min_length=6, max_length=6),
    start: str = Query(min_length=8, max_length=8),
    end: str = Query(min_length=8, max_length=8),
    primary_source: str = Query(
        default="tushare",
        pattern="^(tushare|baostock|akshare)$",
    ),
    comparison_source: str = Query(
        default="baostock",
        pattern="^(tushare|baostock|akshare)$",
    ),
    adjust_type: str = Query(default="raw", pattern="^(raw|qfq|hfq)$"),
) -> dict:
    service = SourceComparisonService(
        request.app.state.market_data,
        request.app.state.store,
    )
    try:
        result = await service.compare(
            code,
            start,
            end,
            primary_source=primary_source,
            comparison_source=comparison_source,
            adjust_type=adjust_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "report": result.report,
        "rows": result.rows.to_dicts(),
        "path": str(result.path),
    }


@router.post("/daily/frame")
async def inspect_daily_frame(request: Request, file: UploadFile) -> dict:
    if not file.filename or not file.filename.endswith(".parquet"):
        raise HTTPException(status_code=415, detail="Upload a Parquet file")
    raw = await file.read()
    if len(raw) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Parquet file exceeds 50 MB")
    try:
        frame = pl.read_parquet(raw)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid Parquet: {exc}") from exc
    clean = clean_daily_bars(frame)
    return {
        "original": daily_quality_report(frame),
        "cleaned": daily_quality_report(clean),
    }

