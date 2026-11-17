from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.services.gbm import GBMFeatureBuilder
from app.storage import ParquetStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-stock GBM features")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    store = ParquetStore(settings)
    daily = store.load_all_dwd_daily(args.start, args.end)
    if daily.is_empty():
        print("No DWD daily data available")
        raise SystemExit(1)

    dataset = GBMFeatureBuilder().build(daily)
    paths = []
    for code in dataset.all_features["stock_code"].unique().sort().to_list():
        stock_features = dataset.all_features.filter(
            pl.col("stock_code") == code
        )
        paths.append(store.save_gbm_features(code, stock_features))

    print(f"stocks: {dataset.stock_count}")
    print(f"training rows: {dataset.training.height}")
    print(f"trading days: {dataset.trading_day_count}")
    print(f"feature files: {len(paths)}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
