from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.providers.base import ProviderError
from app.services.market_data import MarketDataService
from app.services.pipeline import DailyPipeline
from app.storage import ParquetStore


async def main() -> None:
    parser = argparse.ArgumentParser(description="Sync one stock into Parquet")
    parser.add_argument("--code", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--source", default="auto")
    parser.add_argument("--adjust", default="qfq")
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    pipeline = DailyPipeline(
        MarketDataService(settings),
        ParquetStore(settings),
    )
    try:
        result = await pipeline.sync_stock(
            args.code,
            args.start,
            args.end,
            source=args.source,
            adjust=args.adjust,
        )
    except ProviderError as exc:
        print(f"sync failed: {exc}")
        raise SystemExit(1) from exc

    print(f"ODS: {result.ods_path}")
    print(f"DWD: {result.dwd_path}")
    print(f"DWS: {result.dws_path}")
    print(f"raw rows: {result.raw_report['row_count']}")
    print(f"clean rows: {result.clean_report['row_count']}")
    print(f"quality: {result.clean_report['status']}")


if __name__ == "__main__":
    asyncio.run(main())

