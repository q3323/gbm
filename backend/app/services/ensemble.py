from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl

from app.services.backtest import BacktestConfig, StrategyBacktestService
from app.services.nested_tuning import (
    _classification_metrics,
    _cross_sectional_metrics,
)
from app.symbols import normalize_stock_code


@dataclass(frozen=True)
class EnsembleConfig:
    horizon: int = 10
    top_fraction: float = 0.1
    commission_each_side: float = 0.0003
    stamp_duty_sell: float = 0.0005
    slippage_each_side: float = 0.001
    random_trials: int = 10
    seed: int = 42
    model_tag: str = "gbm"


@dataclass(frozen=True)
class EnsembleResult:
    report: dict[str, Any]
    out_of_sample: pl.DataFrame


class GBMSeedEnsembleService:
    def run(
        self,
        seed_frames: list[tuple[int, pl.DataFrame]],
        target_code: str,
        config: EnsembleConfig | None = None,
    ) -> EnsembleResult:
        cfg = config or EnsembleConfig()
        _validate_config(cfg)
        target = normalize_stock_code(target_code)
        if len(seed_frames) < 2:
            raise ValueError("At least two seed OOS frames are required")

        base = seed_frames[0][1].select(
            "stock_code",
            "trade_date",
            "future_return",
            "future_up",
            "fold",
        )
        aligned = base
        for seed, frame in seed_frames:
            _validate_seed_frame(frame, seed, cfg.horizon)
            seed_suffix = str(seed)
            renamed = frame.select(
                "stock_code",
                "trade_date",
                pl.col("predicted_return").alias(
                    f"predicted_return_{seed_suffix}"
                ),
                pl.col("up_probability").alias(
                    f"up_probability_{seed_suffix}"
                ),
                pl.col("baseline_probability").alias(
                    f"baseline_probability_{seed_suffix}"
                ),
                pl.col("logistic_probability").alias(
                    f"logistic_probability_{seed_suffix}"
                ),
            )
            aligned = aligned.join(
                renamed,
                on=["stock_code", "trade_date"],
                how="inner",
            )

        if aligned is None or aligned.is_empty():
            raise ValueError("Seed OOS frames have no aligned rows")
        seed_values = [seed for seed, _ in seed_frames]
        ensemble = aligned.with_columns(
            pl.mean_horizontal(
                [
                    pl.col(f"up_probability_{seed}")
                    for seed in seed_values
                ]
            ).alias("up_probability"),
            pl.mean_horizontal(
                [
                    pl.col(f"predicted_return_{seed}")
                    for seed in seed_values
                ]
            ).alias("predicted_return"),
            pl.mean_horizontal(
                [
                    pl.col(f"baseline_probability_{seed}")
                    for seed in seed_values
                ]
            ).alias("baseline_probability"),
            pl.mean_horizontal(
                [
                    pl.col(f"logistic_probability_{seed}")
                    for seed in seed_values
                ]
            ).alias("logistic_probability"),
        ).select(
            "stock_code",
            "trade_date",
            "future_return",
            "future_up",
            "predicted_return",
            "up_probability",
            "baseline_probability",
            "logistic_probability",
            "fold",
            pl.lit(cfg.horizon).alias("horizon_days"),
            pl.lit(seed_values).alias("seeds"),
        )

        backtest_config = BacktestConfig(
            top_fraction=cfg.top_fraction,
            holding_days=cfg.horizon,
            rebalance_days=cfg.horizon,
            commission_each_side=cfg.commission_each_side,
            stamp_duty_sell=cfg.stamp_duty_sell,
            slippage_each_side=cfg.slippage_each_side,
            random_trials=cfg.random_trials,
            seed=cfg.seed,
        )
        backtest_service = StrategyBacktestService()
        ensemble_backtest = backtest_service.run(
            ensemble,
            backtest_config,
        ).report

        seed_reports = []
        for seed, frame in seed_frames:
            seed_backtest = backtest_service.run(
                frame,
                backtest_config,
            ).report
            seed_metrics = seed_backtest["gbm"]
            benchmark_metrics = seed_backtest["equal_weight_benchmark"]
            seed_reports.append(
                {
                    "seed": seed,
                    "classification": _classification_metrics(
                        frame["future_up"].to_numpy(),
                        frame["up_probability"].to_numpy(),
                    ),
                    "cross_sectional": _cross_sectional_metrics(
                        frame["trade_date"].to_list(),
                        frame["future_return"].to_numpy(),
                        frame["up_probability"].to_numpy(),
                    ),
                    "backtest": {
                        "total_return": seed_metrics["total_return"],
                        "benchmark_total_return": benchmark_metrics[
                            "total_return"
                        ],
                        "excess_total_return": (
                            seed_metrics["total_return"]
                            - benchmark_metrics["total_return"]
                        ),
                        "sharpe": seed_metrics["sharpe"],
                        "max_drawdown": seed_metrics["max_drawdown"],
                    },
                }
            )

        report = {
            "metadata": {
                "stock_code": target,
                "model": f"seed_averaged_{cfg.model_tag}",
                "model_tag": cfg.model_tag,
                "horizon_days": cfg.horizon,
                "seeds": seed_values,
                "generated_on": date.today().isoformat(),
                "rows": ensemble.height,
            },
            "ensemble": {
                "classification": _classification_metrics(
                    ensemble["future_up"].to_numpy(),
                    ensemble["up_probability"].to_numpy(),
                ),
                "cross_sectional": _cross_sectional_metrics(
                    ensemble["trade_date"].to_list(),
                    ensemble["future_return"].to_numpy(),
                    ensemble["up_probability"].to_numpy(),
                ),
                "backtest": {
                    "gbm": ensemble_backtest["gbm"],
                    "benchmark": ensemble_backtest[
                        "equal_weight_benchmark"
                    ],
                    "logistic_baseline": ensemble_backtest[
                        "logistic_baseline"
                    ],
                    "random_baseline": ensemble_backtest[
                        "random_baseline"
                    ],
                },
            },
            "seeds": seed_reports,
        }
        return EnsembleResult(report=report, out_of_sample=ensemble)


def _validate_seed_frame(
    frame: pl.DataFrame,
    seed: int,
    horizon: int,
) -> None:
    required = {
        "stock_code",
        "trade_date",
        "future_return",
        "future_up",
        "predicted_return",
        "up_probability",
        "baseline_probability",
        "logistic_probability",
        "fold",
        "horizon_days",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Seed {seed} OOS missing columns: {', '.join(sorted(missing))}"
        )
    if frame["horizon_days"].unique().to_list() != [horizon]:
        raise ValueError(f"Seed {seed} OOS horizon does not match")


def _validate_config(config: EnsembleConfig) -> None:
    if not 1 <= config.horizon <= 60:
        raise ValueError("horizon must be between 1 and 60")
    if not 0 < config.top_fraction <= 1:
        raise ValueError("top_fraction must be between 0 and 1")
    if config.random_trials < 1:
        raise ValueError("random_trials must be positive")


def _safe_float(value: Any) -> float | None:
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 8)

