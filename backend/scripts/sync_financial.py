from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.providers.base import ProviderError
from app.services.market_data import MarketDataService
from app.storage import ParquetStore


async def main() -> None:
    parser = argparse.ArgumentParser(description="Sync one financial report")
    parser.add_argument("--code", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--quarter", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--source", default="auto")
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    service = MarketDataService(settings)
    store = ParquetStore(settings)
    try:
        report = await service.financial_report(
            args.code,
            args.year,
            args.quarter,
            source=args.source,
        )
    except ProviderError as exc:
        print(f"financial sync failed: {exc}")
        raise SystemExit(1) from exc
    path = store.save_financial_report(report)
    print(report.model_dump_json(indent=2))
    print(f"Financial report: {path}")


if __name__ == "__main__":
    asyncio.run(main())

