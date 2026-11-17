from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.services.sensitivity import (
    GBMSensitivityService,
    SensitivityConfig,
)
from app.storage import ParquetStore


def integer_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GBM sensitivity tests")
    parser.add_argument("--code", default="000001")
    parser.add_argument("--seeds", type=integer_tuple, default=(7, 42, 2024))
    parser.add_argument("--horizons", type=integer_tuple, default=(5, 10, 20))
    parser.add_argument("--top-fractions", type=float_tuple, default=(0.1, 0.2, 0.3))
    parser.add_argument(
        "--cost-multipliers",
        type=float_tuple,
        default=(0.0, 1.0, 2.0),
    )
    parser.add_argument("--no-fundamentals", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    store = ParquetStore(settings)
    daily = store.load_all_dwd_daily()
    financial = None if args.no_fundamentals else store.load_all_financial_reports()
    if daily.is_empty():
        print("No DWD daily data available")
        raise SystemExit(1)

    result = GBMSensitivityService().run(
        daily,
        args.code,
        SensitivityConfig(
            seeds=args.seeds,
            horizons=args.horizons,
            top_fractions=args.top_fractions,
            cost_multipliers=args.cost_multipliers,
        ),
        financial=financial,
    )
    path = store.save_sensitivity_report(args.code, "gbm", result.report)
    print(json.dumps(result.report["summary"], ensure_ascii=False, indent=2))
    print(f"sensitivity report: {path}")


if __name__ == "__main__":
    main()

