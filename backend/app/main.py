from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from app.api.models import router as models_router
from app.api.quality import router as quality_router
from app.api.stocks import router as stocks_router
from app.config import ensure_data_dirs, get_settings
from app.services.market_data import MarketDataService
from app.storage import ParquetStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ensure_data_dirs(settings)
    app.state.settings = settings
    app.state.market_data = MarketDataService(settings)
    app.state.store = ParquetStore(settings)
    yield


app = FastAPI(
    title="Stock Analysis API",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(stocks_router)
app.include_router(quality_router)
app.include_router(models_router)


@app.get("/api/health", tags=["system"])
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "stock-analysis",
        "time": datetime.now(timezone.utc).isoformat(),
    }

