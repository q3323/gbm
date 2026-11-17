from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.providers.base import ProviderError
from app.services.gbm import GBMFeatureBuilder
from app.services.market_data import MarketDataService
from app.services.pipeline import DailyPipeline
from app.storage import ParquetStore


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/stock_universe.json"


def load_universe(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    stocks = raw.get("stocks", [])
    if not stocks:
        raise ValueError(f"No stocks configured in {path}")
    seen = set()
    normalized = []
    for item in stocks:
        code = str(item["code"]).strip().zfill(6)
        if code in seen:
            raise ValueError(f"Duplicate stock code in universe: {code}")
        if len(code) != 6 or not code.isdigit():
            raise ValueError(f"Invalid stock code in universe: {code}")
        seen.add(code)
        normalized.append({"code": code, "name": str(item.get("name", code))})
    return normalized


def compact_date(value: str) -> date:
    if len(value) != 8 or not value.isdigit():
        raise ValueError(f"Invalid date: {value}")
    return datetime.strptime(value, "%Y%m%d").date()


def incremental_start(
    existing_max: date | None,
    fallback: str,
    end: str,
) -> str | None:
    fallback_date = compact_date(fallback)
    end_date = compact_date(end)
    if existing_max is None:
        return fallback if fallback_date <= end_date else None
    start_date = existing_max + timedelta(days=1)
    if start_date > end_date:
        return None
    return start_date.strftime("%Y%m%d")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync and incrementally update a configured stock universe"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start", default="20230101")
    parser.add_argument("--end", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--source", default="baostock")
    parser.add_argument("--adjust", default="qfq")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    args = parser.parse_args()
    compact_date(args.start)
    compact_date(args.end)

    settings = get_settings()
    ensure_data_dirs(settings)
    store = ParquetStore(settings)
    pipeline = DailyPipeline(
        MarketDataService(settings),
        store,
    )
    stocks = load_universe(args.config)
    started_at = datetime.now(timezone.utc)
    results = []

    for index, stock in enumerate(stocks, start=1):
        code = stock["code"]
        existing = store.load_dwd_daily(code)
        if (
            not args.full
            and not existing.is_empty()
            and "adjust_type" in existing.columns
        ):
            adjusted = existing.filter(pl.col("adjust_type") == args.adjust)
        else:
            adjusted = existing
        existing_max = (
            adjusted["trade_date"].max()
            if not adjusted.is_empty()
            else None
        )
        start = (
            args.start
            if args.full
            else incremental_start(existing_max, args.start, args.end)
        )
        if start is None:
            results.append(
                {
                    "code": code,
                    "name": stock["name"],
                    "status": "up_to_date",
                    "row_count": adjusted.height,
                    "start": None,
                    "end": args.end,
                    "error": "",
                }
            )
            print(f"[{index}/{len(stocks)}] {code} {stock['name']}: up to date")
            continue

        print(
            f"[{index}/{len(stocks)}] {code} {stock['name']}: "
            f"{start} -> {args.end}"
        )
        try:
            await pipeline.sync_stock(
                code,
                start,
                args.end,
                source=args.source,
                adjust=args.adjust,
            )
            refreshed = store.load_dwd_daily(code)
            if "adjust_type" in refreshed.columns:
                refreshed = refreshed.filter(
                    pl.col("adjust_type") == args.adjust
                )
            results.append(
                {
                    "code": code,
                    "name": stock["name"],
                    "status": "success",
                    "row_count": refreshed.height,
                    "start": start,
                    "end": args.end,
                    "error": "",
                }
            )
            print(
                f"[{index}/{len(stocks)}] {code}: success, "
                f"{refreshed.height} total rows"
            )
        except Exception as exc:
            results.append(
                {
                    "code": code,
                    "name": stock["name"],
                    "status": "failed",
                    "row_count": adjusted.height,
                    "start": start,
                    "end": args.end,
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(stocks)}] {code}: failed: {exc}")

        if index < len(stocks) and args.delay > 0:
            await asyncio.sleep(args.delay)

    feature_summary = None
    if not args.skip_features:
        all_daily = store.load_all_dwd_daily(args.start, args.end)
        if not all_daily.is_empty():
            try:
                dataset = GBMFeatureBuilder().build(all_daily)
                feature_paths = []
                for code in dataset.all_features["stock_code"].unique().sort().to_list():
                    stock_features = dataset.all_features.filter(
                        pl.col("stock_code") == code
                    )
                    feature_paths.append(
                        str(store.save_gbm_features(code, stock_features))
                    )
                feature_summary = {
                    "stock_count": dataset.stock_count,
                    "training_rows": dataset.training.height,
                    "trading_days": dataset.trading_day_count,
                    "files": feature_paths,
                }
            except Exception as exc:
                feature_summary = {"error": str(exc)}

    failed = [item for item in results if item["status"] == "failed"]
    report = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "requested_start": args.start,
        "requested_end": args.end,
        "source": args.source,
        "adjust": args.adjust,
        "stock_count": len(stocks),
        "success_count": len(
            [item for item in results if item["status"] == "success"]
        ),
        "up_to_date_count": len(
            [item for item in results if item["status"] == "up_to_date"]
        ),
        "failed_count": len(failed),
        "stocks": results,
        "features": feature_summary,
    }
    path = store.save_universe_sync_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"universe sync report: {path}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
