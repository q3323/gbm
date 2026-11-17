from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.indicators import compute_indicators
from app.storage import ParquetStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute technical indicators from stored DWD data"
    )
    parser.add_argument("--code", required=True)
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    store = ParquetStore(settings)
    daily = store.load_dwd_daily(args.code)
    if daily.is_empty():
        print(f"No DWD daily data for {args.code}")
        raise SystemExit(1)

    indicators = compute_indicators(daily)
    path = store.save_dws_indicators(args.code, indicators)
    print(f"DWD rows: {daily.height}")
    print(f"DWS rows: {indicators.height}")
    print(f"DWS: {path}")


if __name__ == "__main__":
    main()

