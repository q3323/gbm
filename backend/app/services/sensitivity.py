from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from app.services.backtest import BacktestConfig, StrategyBacktestService
from app.services.gbm import GBMService


@dataclass(frozen=True)
class SensitivityConfig:
    seeds: tuple[int, ...] = (7, 42, 2024)
    top_fractions: tuple[float, ...] = (0.1, 0.2, 0.3)
    horizons: tuple[int, ...] = (5, 10, 20)
    cost_multipliers: tuple[float, ...] = (0.0, 1.0, 2.0)
    commission_each_side: float = 0.0003
    stamp_duty_sell: float = 0.0005
    slippage_each_side: float = 0.001


@dataclass(frozen=True)
class SensitivityResult:
    report: dict[str, Any]


class GBMSensitivityService:
    def run(
        self,
        daily: pl.DataFrame,
        target_code: str,
        config: SensitivityConfig | None = None,
        financial: pl.DataFrame | None = None,
    ) -> SensitivityResult:
        cfg = config or SensitivityConfig()
        _validate_config(cfg)
        runs = []
        backtest_service = StrategyBacktestService()

        for seed in cfg.seeds:
            for horizon in cfg.horizons:
                gbm_result = GBMService().analyze(
                    daily,
                    target_code,
                    seed=seed,
                    horizon=horizon,
                    financial=financial,
                )
                for top_fraction in cfg.top_fractions:
                    for cost_multiplier in cfg.cost_multipliers:
                        backtest = backtest_service.run(
                            gbm_result.out_of_sample,
                            BacktestConfig(
                                top_fraction=top_fraction,
                                holding_days=horizon,
                                rebalance_days=horizon,
                                commission_each_side=(
                                    cfg.commission_each_side * cost_multiplier
                                ),
                                stamp_duty_sell=(
                                    cfg.stamp_duty_sell * cost_multiplier
                                ),
                                slippage_each_side=(
                                    cfg.slippage_each_side * cost_multiplier
                                ),
                                random_trials=1,
                                seed=seed,
                            ),
                        ).report
                        gbm_metrics = backtest["gbm"]
                        benchmark_metrics = backtest["equal_weight_benchmark"]
                        runs.append(
                            {
                                "seed": seed,
                                "horizon_days": horizon,
                                "top_fraction": top_fraction,
                                "cost_multiplier": cost_multiplier,
                                "periods": gbm_metrics["periods"],
                                "total_return": gbm_metrics["total_return"],
                                "annualized_return": gbm_metrics[
                                    "annualized_return"
                                ],
                                "sharpe": gbm_metrics["sharpe"],
                                "max_drawdown": gbm_metrics[
                                    "max_drawdown"
                                ],
                                "hit_rate": gbm_metrics["hit_rate"],
                                "total_cost_rate": gbm_metrics[
                                    "total_cost_rate"
                                ],
                                "benchmark_total_return": benchmark_metrics[
                                    "total_return"
                                ],
                                "excess_total_return": (
                                    gbm_metrics["total_return"]
                                    - benchmark_metrics["total_return"]
                                ),
                                "classification_status": gbm_result.report[
                                    "reliability"
                                ]["classification_status"],
                            }
                        )

        summary = _summarize_runs(
            runs,
            seeds=cfg.seeds,
            horizons=cfg.horizons,
            top_fractions=cfg.top_fractions,
            cost_multipliers=cfg.cost_multipliers,
        )
        return SensitivityResult(
            report={
                "metadata": {
                    "model": "HistGradientBoosting",
                    "generated_on": date.today().isoformat(),
                    "run_count": len(runs),
                    "seeds": list(cfg.seeds),
                    "horizons": list(cfg.horizons),
                    "top_fractions": list(cfg.top_fractions),
                    "cost_multipliers": list(cfg.cost_multipliers),
                },
                "base_costs": {
                    "commission_each_side": cfg.commission_each_side,
                    "stamp_duty_sell": cfg.stamp_duty_sell,
                    "slippage_each_side": cfg.slippage_each_side,
                },
                "summary": summary,
                "runs": runs,
            }
        )


def _summarize_runs(
    runs: list[dict[str, Any]],
    *,
    seeds: tuple[int, ...],
    horizons: tuple[int, ...],
    top_fractions: tuple[float, ...],
    cost_multipliers: tuple[float, ...],
) -> dict[str, Any]:
    base_top = 0.2 if 0.2 in top_fractions else min(top_fractions)
    by_horizon = []
    for horizon in horizons:
        horizon_runs = [
            run for run in runs if run["horizon_days"] == horizon
        ]
        by_horizon.append(
            {
                "horizon_days": horizon,
                "median_total_return": _median_metric(
                    horizon_runs,
                    "total_return",
                ),
                "median_excess_return": _median_metric(
                    horizon_runs,
                    "excess_total_return",
                ),
                "median_sharpe": _median_metric(horizon_runs, "sharpe"),
                "positive_excess_rate": _positive_rate(
                    horizon_runs,
                    "excess_total_return",
                ),
            }
        )

    cost_summary = []
    for multiplier in cost_multipliers:
        subset = [
            run
            for run in runs
            if run["cost_multiplier"] == multiplier
        ]
        cost_summary.append(
            {
                "cost_multiplier": multiplier,
                "median_total_return": _median_metric(
                    subset,
                    "total_return",
                ),
                "median_excess_return": _median_metric(
                    subset,
                    "excess_total_return",
                ),
                "positive_excess_rate": _positive_rate(
                    subset,
                    "excess_total_return",
                ),
                "positive_return_rate": _positive_rate(
                    subset,
                    "total_return",
                ),
            }
        )

    base_runs = [
        run
        for run in runs
        if run["top_fraction"] == base_top
        and run["cost_multiplier"] == 1.0
    ]
    seed_results = []
    for seed in seeds:
        seed_runs = [run for run in base_runs if run["seed"] == seed]
        seed_results.append(
            {
                "seed": seed,
                "positive_excess_count": sum(
                    1
                    for run in seed_runs
                    if run["excess_total_return"] > 0
                ),
                "run_count": len(seed_runs),
                "median_excess_return": _median_metric(
                    seed_runs,
                    "excess_total_return",
                ),
            }
        )

    cost_2x_positive = next(
        (
            item["positive_excess_rate"]
            for item in cost_summary
            if item["cost_multiplier"] == 2.0
        ),
        None,
    )
    cost_1x_positive = next(
        (
            item["positive_excess_rate"]
            for item in cost_summary
            if item["cost_multiplier"] == 1.0
        ),
        None,
    )
    if (
        cost_2x_positive is not None
        and cost_2x_positive >= 0.6
        and all(
            item["positive_excess_count"] >= 2
            for item in seed_results
        )
    ):
        status = "robust"
    elif cost_1x_positive is not None and cost_1x_positive >= 0.6:
        status = "cost_sensitive"
    else:
        status = "not_robust"

    cost_one_runs = [
        run for run in runs if run["cost_multiplier"] == 1.0
    ]
    best = max(
        cost_one_runs,
        key=lambda run: (
            run["sharpe"] if run["sharpe"] is not None else -math.inf,
            run["annualized_return"]
            if run["annualized_return"] is not None
            else -math.inf,
        ),
        default=None,
    )
    return {
        "status": status,
        "base_top_fraction": base_top,
        "by_horizon": by_horizon,
        "by_cost": cost_summary,
        "seed_results": seed_results,
        "best_1x_cost_run": best,
    }


def _positive_rate(
    runs: list[dict[str, Any]],
    field_name: str,
) -> float | None:
    values = [
        run[field_name]
        for run in runs
        if run.get(field_name) is not None
    ]
    if not values:
        return None
    return _safe_float(np.mean([value > 0 for value in values]))


def _median_metric(
    runs: list[dict[str, Any]],
    field_name: str,
) -> float | None:
    values = [
        float(run[field_name])
        for run in runs
        if run.get(field_name) is not None
    ]
    return _safe_float(np.median(values)) if values else None


def _validate_config(config: SensitivityConfig) -> None:
    if not config.seeds:
        raise ValueError("At least one seed is required")
    if not config.horizons:
        raise ValueError("At least one horizon is required")
    if any(horizon < 1 or horizon > 60 for horizon in config.horizons):
        raise ValueError("Horizons must be between 1 and 60 days")
    if any(not 0 < value <= 1 for value in config.top_fractions):
        raise ValueError("Top fractions must be between 0 and 1")
    if any(value < 0 for value in config.cost_multipliers):
        raise ValueError("Cost multipliers cannot be negative")


def _safe_float(value: Any) -> float | None:
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 8)

