from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import akshare as ak
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import BACKEND_ROOT, ensure_data_dirs, get_settings
from app.services.group_bagging import (
    GroupBaggingConfig,
    GroupedGBMBaggingService,
    PHARMA_FACTOR_POOL,
    TECH_FACTOR_POOL,
    US_ETF_FACTOR_POOL,
    load_group_configs,
)
from app.services.index_gbm import IndexBaggingConfig, IndexGBMBaggingService
from app.storage import ParquetStore


GROUP_CONFIG_PATH = BACKEND_ROOT / "config" / "model_groups.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train configured grouped GBM models")
    parser.add_argument("--group", help="Train one configured group id")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train every configured group",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=(5, 10, 20),
        help="Override the configured prediction horizon",
    )
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    store = ParquetStore(settings)
    groups = load_group_configs(GROUP_CONFIG_PATH)
    selected = [
        group
        for group in groups
        if args.all
        or not args.group
        or group.get("id") == args.group
    ]
    if not selected:
        raise SystemExit("No matching model groups")

    daily = None
    financial = None
    for group in selected:
        group_type = group.get("type")
        horizon = args.horizon or int(group.get("horizon_days", 10))
        benchmark_symbol = str(group.get("benchmark_symbol", "")).strip()
        benchmark_daily = (
            _load_benchmark(store, benchmark_symbol)
            if benchmark_symbol
            else None
        )
        if group_type == "cross_section":
            if daily is None:
                daily = store.load_all_dwd_daily()
                if daily.is_empty():
                    raise SystemExit("No DWD daily data available")
                financial = store.load_all_financial_reports()
            config = GroupBaggingConfig(
                group_id=str(group["id"]),
                name=str(group.get("name", group["id"])),
                market=str(group.get("market", "CN")),
                stock_codes=tuple(group.get("stock_codes", [])),
                seeds=tuple(group.get("seeds", [7, 42, 2024, 123, 888])),
                sample_fraction=float(group.get("sample_fraction", 0.8)),
                horizon_days=horizon,
                factor_pool=_factor_pool(group.get("factor_pool_name")),
            )
            result = GroupedGBMBaggingService().run(
                daily,
                config,
                financial=financial,
                benchmark_daily=benchmark_daily,
            )
        elif group_type == "us_etf_group":
            etf_daily = _load_us_etf_group(
                store,
                group.get("stock_codes", []),
            )
            model_codes = [
                str(code)
                for code in group.get(
                    "model_stock_codes",
                    group.get("stock_codes", []),
                )
            ]
            config = GroupBaggingConfig(
                group_id=str(group["id"]),
                name=str(group.get("name", group["id"])),
                market="US",
                stock_codes=tuple(model_codes),
                seeds=tuple(group.get("seeds", [7, 42, 2024, 123, 888])),
                sample_fraction=float(group.get("sample_fraction", 0.8)),
                horizon_days=horizon,
                factor_pool=US_ETF_FACTOR_POOL,
                minimum_available_stocks=int(
                    group.get("minimum_available_stocks", len(model_codes))
                ),
            )
            result = GroupedGBMBaggingService().run(
                etf_daily,
                config,
                benchmark_daily=benchmark_daily,
                min_training_rows=800,
            )
            all_codes = [
                str(code) for code in group.get("stock_codes", [])
            ]
            supplemental_codes = [
                code for code in all_codes if code not in model_codes
            ]
            result.report["metadata"]["display_stock_codes"] = all_codes
            result.report["metadata"]["model_stock_codes"] = model_codes
            result.report["supplemental_products"] = (
                _supplemental_product_snapshots(
                    etf_daily,
                    supplemental_codes,
                    group.get("product_profiles", {}),
                    group.get("names", {}),
                )
            )
        elif group_type == "index":
            symbol = str(group.get("symbol", ".NDX"))
            raw = ak.index_us_stock_sina(symbol=symbol)
            index_daily = pl.from_pandas(raw)
            index_path = (
                settings.data_dir
                / "ods"
                / "us_index"
                / f"symbol={symbol.lstrip('.')}"
                / "daily.parquet"
            )
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_daily.write_parquet(index_path)
            config = IndexBaggingConfig(
                group_id=str(group["id"]),
                name=str(group.get("name", group["id"])),
                symbol=symbol,
                seeds=tuple(group.get("seeds", [7, 42, 2024, 123, 888])),
                sample_fraction=float(group.get("sample_fraction", 0.8)),
                horizon_days=horizon,
            )
            result = IndexGBMBaggingService().run(
                index_daily,
                config,
                benchmark_daily=benchmark_daily,
            )
        else:
            print(f"skip unsupported group type: {group_type}")
            continue

        path = store.save_group_bagging_result(
            config.group_id,
            horizon,
            result.report,
        )
        print(
            json.dumps(
                {
                    "group": config.group_id,
                    "prediction": result.report["group_prediction"],
                    "stocks": len(result.report["rankings"]),
                    "report": str(path),
                },
                ensure_ascii=False,
            )
        )

def _load_benchmark(store: ParquetStore, symbol: str) -> pl.DataFrame:
    path = (
        store.data_dir
        / "ods"
        / "benchmarks"
        / f"symbol={symbol.lstrip('.')}"
        / "daily.parquet"
    )
    if path.exists():
        return pl.read_parquet(path)
    if symbol.startswith("."):
        raw = ak.index_us_stock_sina(symbol=symbol)
    else:
        raw = ak.stock_zh_index_daily(symbol=symbol)
    frame = pl.from_pandas(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return frame


def _factor_pool(name: object) -> tuple[str, ...]:
    if name == "tech":
        return TECH_FACTOR_POOL
    if name == "pharma":
        return PHARMA_FACTOR_POOL
    if name == "us_etf":
        return US_ETF_FACTOR_POOL
    return ()


def _load_us_etf_group(
    store: ParquetStore,
    tickers: list[str],
) -> pl.DataFrame:
    frames = []
    for ticker in tickers:
        symbol = str(ticker).strip().upper()
        # Sina's US qfq mode can generate negative historical prices.
        # Raw US data is split-normalized and remains positive.
        raw = ak.stock_us_daily(symbol=symbol, adjust="")
        frame = pl.from_pandas(raw)
        if "date" not in frame.columns:
            raise ValueError(f"US ETF {symbol} missing date")
        date_expression = (
            pl.col("date").cast(pl.Date)
            if isinstance(frame.schema["date"], pl.Datetime)
            else pl.col("date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
        )
        frame = (
            frame.select(
                date_expression.alias("trade_date"),
                *[
                    pl.col(name).cast(pl.Float64)
                    for name in ("open", "high", "low", "close", "volume")
                ],
            )
            .drop_nulls(["trade_date", "open", "high", "low", "close"])
            .filter(
                (pl.col("open") > 0)
                & (pl.col("high") > 0)
                & (pl.col("low") > 0)
                & (pl.col("close") > 0)
                & (pl.col("volume") >= 0)
            )
            .sort("trade_date")
            .unique("trade_date", keep="last")
            .with_columns(
                pl.lit(symbol).alias("stock_code"),
                pl.lit("qfq").alias("adjust_type"),
                pl.col("close").shift(1).alias("pre_close"),
                (pl.col("volume") * pl.col("close")).alias("amount"),
                pl.lit(None, dtype=pl.Float64).alias("turnover"),
                pl.lit("sina_us_raw").alias("source"),
                pl.lit(datetime.now(timezone.utc).replace(tzinfo=None)).alias(
                    "ingested_at"
                ),
            )
            .with_columns(
                (
                    pl.col("close") / pl.col("pre_close") - 1.0
                ).mul(100.0).alias("pct_change")
            )
        )
        path = (
            store.data_dir
            / "ods"
            / "us_etf"
            / f"symbol={symbol}"
            / "bars.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)
        frames.append(frame)
    if not frames:
        raise ValueError("No US ETF daily data loaded")
    return pl.concat(frames, how="diagonal_relaxed")


def _supplemental_product_snapshots(
    daily: pl.DataFrame,
    codes: list[str],
    profiles: dict[str, dict],
    names: dict[str, str],
) -> list[dict]:
    snapshots = []
    for code in codes:
        frame = daily.filter(pl.col("stock_code") == code).sort("trade_date")
        if frame.is_empty():
            continue
        enriched = (
            frame.with_columns(
                pl.col("close").rolling_mean(5).alias("ma_5"),
                pl.col("close").rolling_mean(20).alias("ma_20"),
                pl.col("close").rolling_mean(60).alias("ma_60"),
            )
            .with_columns(
                (pl.col("close") / pl.col("close").shift(5) - 1.0).alias(
                    "return_5d"
                ),
                (pl.col("close") / pl.col("close").shift(20) - 1.0).alias(
                    "return_20d"
                ),
            )
            .with_columns(
                (
                    (pl.col("close") > pl.col("ma_5")).cast(pl.Int8)
                    + (pl.col("close") > pl.col("ma_20")).cast(pl.Int8)
                    + (pl.col("ma_5") > pl.col("ma_20")).cast(pl.Int8)
                    + (pl.col("ma_20") > pl.col("ma_60")).cast(pl.Int8)
                ).alias("trend_checks")
            )
        )
        latest = enriched.row(-1, named=True)
        trend_score = float(latest["trend_checks"]) / 4.0
        profile = profiles.get(code, {})
        snapshots.append(
            {
                "stock_code": code,
                "stock_name": names.get(code, code),
                "trade_date": str(latest["trade_date"]),
                "category": profile.get("category", "unknown"),
                "leverage": profile.get("leverage"),
                "trend_score": round(trend_score, 6),
                "direction": (
                    "up"
                    if trend_score >= 0.75
                    else "down"
                    if trend_score <= 0.25
                    else "mixed"
                ),
                "return_5d": round(float(latest["return_5d"]), 8),
                "return_20d": round(float(latest["return_20d"]), 8),
                "note": "not_in_cross_section_score",
            }
        )
    return snapshots


if __name__ == "__main__":
    main()
