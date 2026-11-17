from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.akshare import AkshareProvider
from app.providers.baostock import BaostockProvider
from app.providers.base import ProviderError


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a small daily bar sample")
    parser.add_argument("--provider", choices=["akshare", "baostock"], required=True)
    parser.add_argument("--code", default="000001")
    parser.add_argument("--start", default="20261101")
    parser.add_argument("--end", default="20261113")
    args = parser.parse_args()

    provider = (
        AkshareProvider()
        if args.provider == "akshare"
        else BaostockProvider()
    )
    try:
        frame = await provider.daily_bars(
            args.code,
            args.start,
            args.end,
            adjust="qfq",
        )
    except ProviderError as exc:
        print(f"{provider.name}: ERROR: {exc}")
        raise SystemExit(1) from exc
    print(f"{provider.name}: rows={frame.height}")
    print(frame.head(3))


if __name__ == "__main__":
    asyncio.run(main())

