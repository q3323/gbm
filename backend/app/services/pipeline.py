from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from app.providers.base import ProviderError
from app.services.market_data import MarketDataService
from app.storage import ParquetStore, StoredDailyResult, new_run_record


class DailyPipeline:
    def __init__(
        self,
        market_data: MarketDataService,
        store: ParquetStore,
    ) -> None:
        self.market_data = market_data
        self.store = store

    async def sync_stock(
        self,
        code: str,
        start_date: str,
        end_date: str,
        *,
        source: str = "auto",
        adjust: str = "qfq",
    ) -> StoredDailyResult:
        run_id = uuid4().hex
        started_at = datetime.now(timezone.utc)
        try:
            frame = await self.market_data.daily_bars(
                code,
                start_date,
                end_date,
                source=source,
                adjust=adjust,
            )
            actual_source = (
                str(frame["source"][0])
                if frame.height and "source" in frame.columns
                else source
            )
            result = self.store.ingest_daily(code, actual_source, frame)
            self.store.record_pipeline_run(
                new_run_record(
                    run_id=run_id,
                    code=code,
                    source=actual_source,
                    status="success",
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                    row_count=frame.height,
                )
            )
            return result
        except Exception as exc:
            self.store.record_pipeline_run(
                new_run_record(
                    run_id=run_id,
                    code=code,
                    source=source,
                    status="failed",
                    started_at=started_at,
                    finished_at=datetime.now(timezone.utc),
                    row_count=0,
                    error=str(exc),
                )
            )
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(str(exc)) from exc

