from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.adjustments import normalize_adjust_type
from app.config import ensure_data_dirs, get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill adjust_type into existing daily Parquet files"
    )
    parser.add_argument("--adjust", default="qfq")
    args = parser.parse_args()
    adjustment = normalize_adjust_type(args.adjust)

    settings = get_settings()
    ensure_data_dirs(settings)
    paths = [
        *settings.data_dir.glob("ods/daily_price/**/bars.parquet"),
        *settings.data_dir.glob("dwd/daily_price/**/bars.parquet"),
        *settings.data_dir.glob("dws/technical_indicators/**/indicators.parquet"),
    ]
    for path in paths:
        frame = pl.read_parquet(path)
        if "adjust_type" not in frame.columns:
            frame = frame.with_columns(pl.lit(adjustment).alias("adjust_type"))
        else:
            frame = frame.with_columns(
                pl.col("adjust_type")
                .fill_null(adjustment)
                .replace({"": adjustment, "none": "raw"})
            )
        temp = path.with_suffix(".tmp.parquet")
        frame.write_parquet(temp)
        os.replace(temp, path)
        print(f"backfilled {adjustment}: {path}")
    print(f"updated {len(paths)} file(s)")


if __name__ == "__main__":
    main()
