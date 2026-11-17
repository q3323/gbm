from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.services.backtest import BacktestConfig, StrategyBacktestService
from app.services.gbm import GBMService
from app.storage import ParquetStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest GBM stock selection")
    parser.add_argument("--code", default="000001")
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--rebalance-days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--commission", type=float, default=0.0003)
    parser.add_argument("--stamp-duty", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--no-fundamentals", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    store = ParquetStore(settings)
    all_daily = store.load_all_dwd_daily()
    financial = None if args.no_fundamentals else store.load_all_financial_reports()
    gbm_result = GBMService().analyze(
        all_daily,
        args.code,
        seed=args.seed,
        horizon=args.rebalance_days,
        financial=financial,
    )
    backtest_result = StrategyBacktestService().run(
        gbm_result.out_of_sample,
        BacktestConfig(
            top_fraction=args.top_fraction,
            holding_days=args.rebalance_days,
            rebalance_days=args.rebalance_days,
            commission_each_side=args.commission,
            stamp_duty_sell=args.stamp_duty,
            slippage_each_side=args.slippage,
            seed=args.seed,
        ),
    )
    report = {
        **backtest_result.report,
        "model_evaluation": gbm_result.report,
    }
    path = store.save_backtest_report(args.code, "gbm", report)
    print(
        json.dumps(
            {
                "metadata": report["metadata"],
                "costs": report["costs"],
                "gbm": report["gbm"],
                "logistic_baseline": report["logistic_baseline"],
                "equal_weight_benchmark": report["equal_weight_benchmark"],
                "random_baseline": report["random_baseline"],
                "probability_analysis": report["probability_analysis"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"backtest report: {path}")


if __name__ == "__main__":
    main()

