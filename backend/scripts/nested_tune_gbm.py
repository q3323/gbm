from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.services.nested_tuning import (
    NestedTuningConfig,
    NestedTuningService,
)
from app.storage import ParquetStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Nested TPE GBM tuning")
    parser.add_argument("--code", default="000001")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--outer-folds", type=int, default=4)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--embargo-days", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--fundamentals", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    store = ParquetStore(settings)
    daily = store.load_all_dwd_daily()
    financial = store.load_all_financial_reports() if args.fundamentals else None
    model_tag = "gbm_fund" if args.fundamentals else "gbm"
    result = NestedTuningService().run(
        daily,
        args.code,
        NestedTuningConfig(
            horizon=args.horizon,
            outer_folds=args.outer_folds,
            inner_splits=args.inner_splits,
            trials=args.trials,
            embargo_days=args.embargo_days,
            seed=args.seed,
            top_fraction=args.top_fraction,
        ),
        financial=financial,
    )
    path = store.save_nested_tuning_result(
        args.code,
        model_tag,
        result.report,
        result.out_of_sample,
    )
    print(
        json.dumps(
            {
                "metadata": result.report["metadata"],
                "dataset": result.report["dataset"],
                "aggregate_outer": result.report["aggregate_outer"],
                "folds": result.report["folds"],
                "backtest": {
                    "gbm": result.report["backtest"]["gbm"],
                    "benchmark": result.report["backtest"][
                        "equal_weight_benchmark"
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"nested tuning report: {path}")


if __name__ == "__main__":
    main()

