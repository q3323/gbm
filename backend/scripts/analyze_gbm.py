from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.services.gbm import GBMDataError, GBMService
from app.storage import ParquetStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and backtest GBM")
    parser.add_argument("--code", required=True)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--no-fundamentals", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    store = ParquetStore(settings)
    daily = store.load_all_dwd_daily(args.start, args.end)
    financial = None if args.no_fundamentals else store.load_all_financial_reports()
    try:
        result = GBMService().analyze(
            daily,
            args.code,
            seed=args.seed,
            horizon=args.horizon,
            financial=financial,
        )
    except (GBMDataError, ValueError) as exc:
        details = exc.details if isinstance(exc, GBMDataError) else {}
        print(f"GBM analysis failed: {exc}")
        if details:
            print(json.dumps(details, ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc

    report_path = store.save_gbm_report(args.code, result.report)
    print(json.dumps(result.report, ensure_ascii=False, indent=2))
    print(f"GBM report: {report_path}")


if __name__ == "__main__":
    main()

