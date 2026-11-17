from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.services.nested_tuning import (
    NestedTuningConfig,
    NestedTuningService,
)
from app.storage import ParquetStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run nested TPE with multiple seeds and summarize stability"
    )
    parser.add_argument("--code", default="000001")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seeds", default="7,2024")
    parser.add_argument("--reuse-seed", type=int, default=42)
    parser.add_argument("--outer-folds", type=int, default=4)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--embargo-days", type=int, default=2)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument(
        "--neutralize",
        action="store_true",
        help="Enable cross-sectional sector neutralization (GBM v3)",
    )
    parser.add_argument(
        "--fundamentals",
        action="store_true",
        help="Enable as-of joined quarterly fundamental features",
    )
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    store = ParquetStore(settings)
    daily = store.load_all_dwd_daily()
    if daily.is_empty():
        raise SystemExit("No DWD daily data available")

    model_tag = "gbm_v3" if args.neutralize else "gbm"
    if args.fundamentals:
        model_tag += "_fund"
    target_dir = (
        settings.data_dir
        / "ads"
        / "nested_tuning"
        / model_tag
        / f"stock_code={args.code}"
        / f"horizon={args.horizon}"
    )
    if args.reuse_seed is not None and not args.neutralize:
        existing_seed_path = target_dir / f"seed={args.reuse_seed}" / "latest.json"
        top_level_path = (
            settings.data_dir
            / "ads"
            / "nested_tuning"
            / model_tag
            / f"stock_code={args.code}"
            / "latest.json"
        )
        existing_oos = (
            settings.data_dir
            / "ads"
            / "nested_tuning"
            / model_tag
            / f"stock_code={args.code}"
            / "out_of_sample.parquet"
        )
        if not existing_seed_path.exists() and top_level_path.exists() and existing_oos.exists():
            report = json.loads(top_level_path.read_text(encoding="utf-8"))
            if (
                report["metadata"]["seed"] == args.reuse_seed
                and report["metadata"]["target_horizon_days"] == args.horizon
            ):
                import polars as pl

                store.save_nested_tuning_seed_result(
                    args.code,
                    model_tag,
                    args.horizon,
                    args.reuse_seed,
                    report,
                    pl.read_parquet(existing_oos),
                )
                print(f"reused seed {args.reuse_seed}")

    seeds = [
        int(value.strip())
        for value in args.seeds.split(",")
        if value.strip()
    ]
    financial = store.load_all_financial_reports() if args.fundamentals else None
    if args.fundamentals and financial.is_empty():
        raise SystemExit("No financial report data available")
    for seed in seeds:
        print(f"running seed {seed}...")
        result = NestedTuningService().run(
            daily,
            args.code,
            NestedTuningConfig(
                horizon=args.horizon,
                outer_folds=args.outer_folds,
                inner_splits=args.inner_splits,
                trials=args.trials,
                embargo_days=args.embargo_days,
                seed=seed,
                top_fraction=args.top_fraction,
                neutralize=args.neutralize,
            ),
            financial=financial,
        )
        path = store.save_nested_tuning_seed_result(
            args.code,
            model_tag,
            args.horizon,
            seed,
            result.report,
            result.out_of_sample,
        )
        print(
            json.dumps(
                {
                    "seed": seed,
                    "rank_ic": result.report["aggregate_outer"][
                        "tuned_cross_sectional"
                    ]["mean_rank_ic"],
                    "auc": result.report["aggregate_outer"]["tuned"][
                        "roc_auc"
                    ],
                    "backtest_return": result.report["backtest"]["gbm"][
                        "total_return"
                    ],
                    "report": str(path),
                },
                ensure_ascii=False,
            )
        )

    reports = []
    for path in sorted(target_dir.glob("seed=*/latest.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["metadata"]["target_horizon_days"] != args.horizon:
            continue
        reports.append(report)
    summary = summarize_seed_reports(reports)
    summary_path = store.save_nested_tuning_summary(
        args.code,
        model_tag,
        args.horizon,
        summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"seed summary: {summary_path}")


def summarize_seed_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    runs = []
    for report in reports:
        tuned = report["aggregate_outer"]["tuned"]
        cross_sectional = report["aggregate_outer"]["tuned_cross_sectional"]
        backtest = report["backtest"]["gbm"]
        benchmark = report["backtest"]["equal_weight_benchmark"]
        runs.append(
            {
                "seed": report["metadata"]["seed"],
                "accuracy": tuned["accuracy"],
                "roc_auc": tuned["roc_auc"],
                "mean_rank_ic": cross_sectional["mean_rank_ic"],
                "positive_rank_ic_rate": cross_sectional[
                    "positive_rank_ic_rate"
                ],
                "mean_top_excess_return": cross_sectional[
                    "mean_top_excess_return"
                ],
                "backtest_total_return": backtest["total_return"],
                "benchmark_total_return": benchmark["total_return"],
                "excess_total_return": (
                    backtest["total_return"] - benchmark["total_return"]
                ),
                "sharpe": backtest["sharpe"],
                "max_drawdown": backtest["max_drawdown"],
            }
        )
    runs.sort(key=lambda item: item["seed"])
    if not runs:
        return {"status": "no_data", "seeds": []}

    rank_ics = [run["mean_rank_ic"] for run in runs]
    excess_returns = [run["excess_total_return"] for run in runs]
    sharpes = [run["sharpe"] for run in runs]
    criteria = {
        "all_rank_ic_positive": all(value > 0 for value in rank_ics),
        "minimum_rank_ic_above_0_03": min(rank_ics) > 0.03,
        "all_cost_adjusted_excess_positive": all(
            value > 0 for value in excess_returns
        ),
        "excess_return_range_below_0_15": (
            max(excess_returns) - min(excess_returns) < 0.15
        ),
        "median_sharpe_above_0_5": float(
            np.median(sharpes)
        )
        > 0.5,
        "minimum_sharpe_above_0_3": min(sharpes) > 0.3,
    }
    passed = sum(criteria.values())
    status = (
        "robust"
        if passed == len(criteria)
        else "promising"
        if passed >= 3
        else "not_robust"
    )
    return {
        "status": status,
        "criteria": criteria,
        "passed_criteria": passed,
        "seed_count": len(runs),
        "runs": runs,
        "median": {
            "rank_ic": _safe_float(np.median(rank_ics)),
            "excess_total_return": _safe_float(np.median(excess_returns)),
            "sharpe": _safe_float(
                np.median(sharpes)
            ),
            "max_drawdown": _safe_float(
                np.median([run["max_drawdown"] for run in runs])
            ),
        },
    }


def _safe_float(value: Any) -> float | None:
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 8)


if __name__ == "__main__":
    main()
