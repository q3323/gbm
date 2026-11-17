from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.services.time_series import TimeSeriesService
from app.storage import ParquetStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one stored stock series")
    parser.add_argument("--code", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seasonal-period", type=int, default=5)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    store = ParquetStore(settings)
    daily = store.load_dwd_daily(args.code, args.start, args.end)
    if daily.is_empty():
        print(f"No DWD daily data for {args.code}")
        raise SystemExit(1)

    result = TimeSeriesService().analyze(
        args.code,
        daily,
        start_date=args.start,
        end_date=args.end,
        horizon=args.horizon,
        seed=args.seed,
        seasonal_period=args.seasonal_period,
    )
    path = store.save_analysis_report(args.code, result.report)
    if args.print_json:
        print(json.dumps(result.report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result.report["forecast"], ensure_ascii=False, indent=2))
        print(
            json.dumps(
                result.report["volatility"],
                ensure_ascii=False,
                indent=2,
            )
        )
    print(f"analysis report: {path}")


if __name__ == "__main__":
    main()
