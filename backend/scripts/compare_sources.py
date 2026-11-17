from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.providers.base import ProviderError
from app.services.market_data import MarketDataService
from app.services.source_comparison import SourceComparisonService
from app.storage import ParquetStore


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two daily data sources")
    parser.add_argument("--code", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--primary-source", default="tushare")
    parser.add_argument("--comparison-source", default="baostock")
    parser.add_argument("--adjust", default="raw")
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    service = SourceComparisonService(
        MarketDataService(settings),
        ParquetStore(settings),
    )
    try:
        result = await service.compare(
            args.code,
            args.start,
            args.end,
            primary_source=args.primary_source,
            comparison_source=args.comparison_source,
            adjust_type=args.adjust,
        )
    except (ProviderError, ValueError) as exc:
        print(f"comparison failed: {exc}")
        raise SystemExit(1) from exc

    print(json.dumps(result.report, ensure_ascii=False, indent=2))
    print(f"comparison rows: {result.rows.height}")
    print(f"comparison file: {result.path}")


if __name__ == "__main__":
    asyncio.run(main())
