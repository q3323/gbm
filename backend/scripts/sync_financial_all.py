from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl

from app.config import get_settings
from app.providers.baostock import BaostockProvider
from app.storage import ParquetStore


START_YEAR = 2022
START_QUARTER = 4
END_YEAR = 2026
END_QUARTER = 3


def _quarter_list() -> list[tuple[int, int]]:
    quarters = []
    year, quarter = START_YEAR, START_QUARTER
    while (year, quarter) <= (END_YEAR, END_QUARTER):
        quarters.append((year, quarter))
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1
    return quarters


def _already_synced(store: ParquetStore, code: str) -> set[tuple[int, int]]:
    frame = store.load_financial_reports(code)
    if frame.is_empty():
        return set()
    result = set()
    for row in frame["report_date"].to_list():
        result.add((row.year, (row.month - 1) // 3 + 1))
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-sync BaoStock quarterly financial reports"
    )
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--max-failures", type=int, default=100)
    args = parser.parse_args()

    settings = get_settings()
    store = ParquetStore(settings)
    provider = BaostockProvider()

    with open(Path(__file__).resolve().parents[1] / "config/stock_universe.json", encoding="utf-8") as fh:
        import json
        universe = json.load(fh)
    codes = [
        item["code"] if isinstance(item, dict) else item
        for item in (universe.get("stocks") or universe.get("universe") or universe)
    ]

    quarters = _quarter_list()
    total = len(codes) * len(quarters)
    print(f"syncing {len(codes)} stocks x {len(quarters)} quarters = {total} reports")

    done = 0
    failed = 0
    skipped = 0
    started = time.monotonic()
    with provider.persistent_session():
        for code in codes:
            existing = _already_synced(store, code)
            for year, quarter in quarters:
                if (year, quarter) in existing:
                    skipped += 1
                    done += 1
                    continue
                try:
                    report = await provider.financial_report(code, year, quarter)
                    store.save_financial_report(report)
                except Exception as exc:
                    failed += 1
                    if failed >= args.max_failures:
                        print(f"aborting: {failed} failures")
                        return
                    print(f"FAIL {code} {year}Q{quarter}: {exc}")
                done += 1
                if done % 50 == 0:
                    elapsed = time.monotonic() - started
                    rate = done / max(elapsed, 1)
                    remaining = (total - done) / max(rate, 0.01)
                    print(
                        f"progress {done}/{total} "
                        f"(ok={done - failed - skipped} fail={failed} skip={skipped}) "
                        f"eta={remaining:.0f}s"
                    )
                if args.delay > 0:
                    await asyncio.sleep(args.delay)

    elapsed = time.monotonic() - started
    print(
        f"done in {elapsed:.0f}s: ok={done - failed - skipped} "
        f"fail={failed} skip={skipped}"
    )


if __name__ == "__main__":
    asyncio.run(main())
