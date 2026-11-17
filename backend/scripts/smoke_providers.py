from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.akshare import AkshareProvider
from app.providers.baostock import BaostockProvider
from app.providers.base import ProviderError
from app.providers.tushare import TushareProvider


async def main() -> None:
    parser = argparse.ArgumentParser(description="Check provider connectivity")
    parser.add_argument("--query", default="000001")
    parser.add_argument("--tushare-token", default="")
    args = parser.parse_args()

    providers = [
        TushareProvider(args.tushare_token),
        BaostockProvider(),
        AkshareProvider(),
    ]
    for provider in providers:
        if not provider.configured:
            print(f"{provider.name}: not configured")
            continue
        try:
            stocks = await provider.search_stocks(args.query, limit=5)
            print(f"{provider.name}: {[(item.code, item.name) for item in stocks]}")
        except ProviderError as exc:
            print(f"{provider.name}: ERROR: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

