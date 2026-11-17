from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True)
class BacktestConfig:
    rank_column: str = "up_probability"
    top_fraction: float = 0.2
    holding_days: int = 5
    rebalance_days: int = 5
    commission_each_side: float = 0.0003
    stamp_duty_sell: float = 0.0005
    slippage_each_side: float = 0.001
    random_trials: int = 30
    seed: int = 42


@dataclass(frozen=True)
class BacktestResult:
    report: dict[str, Any]


class StrategyBacktestService:
    def run(
        self,
        out_of_sample: pl.DataFrame,
        config: BacktestConfig | None = None,
    ) -> BacktestResult:
        cfg = config or BacktestConfig()
        self._validate_config(cfg)
        frame = self._prepare(out_of_sample)
        dates = frame["trade_date"].unique().sort()
        rebalance_dates = dates.gather(
            list(range(0, dates.len(), cfg.rebalance_days))
        )
        if rebalance_dates.len() < 5:
            raise ValueError("At least five rebalance dates are required")

        period_rows = []
        previous_gbm: dict[str, float] = {}
        previous_logistic: dict[str, float] = {}
        random_generator = np.random.default_rng(cfg.seed)
        random_period_returns = [[] for _ in range(cfg.random_trials)]
        random_period_costs = [[] for _ in range(cfg.random_trials)]

        for trade_date in rebalance_dates:
            candidates = frame.filter(pl.col("trade_date") == trade_date)
            if candidates.height < 2:
                continue
            holding_count = max(
                1,
                math.ceil(candidates.height * cfg.top_fraction),
            )
            rank_columns = [cfg.rank_column]
            descending = [True]
            if cfg.rank_column != "predicted_return":
                rank_columns.append("predicted_return")
                descending.append(True)
            gbm_selected = candidates.sort(
                rank_columns,
                descending=descending,
            ).head(holding_count)
            logistic_selected = candidates.sort(
                "logistic_probability",
                descending=True,
            ).head(holding_count)

            gbm_weights = _equal_weights(
                gbm_selected["stock_code"].to_list()
            )
            logistic_weights = _equal_weights(
                logistic_selected["stock_code"].to_list()
            )
            gbm_turnover = _turnover(previous_gbm, gbm_weights)
            logistic_turnover = _turnover(
                previous_logistic,
                logistic_weights,
            )
            previous_gbm = gbm_weights
            previous_logistic = logistic_weights

            gbm_gross = float(gbm_selected["future_return"].mean())
            logistic_gross = float(
                logistic_selected["future_return"].mean()
            )
            benchmark = float(candidates["future_return"].mean())
            gbm_cost_rate = _round_trip_cost(gbm_turnover, cfg)
            logistic_cost_rate = _round_trip_cost(
                logistic_turnover,
                cfg,
            )

            for trial in range(cfg.random_trials):
                selected_indices = random_generator.choice(
                    candidates.height,
                    size=holding_count,
                    replace=False,
                )
                selected_return = float(
                    candidates["future_return"].gather(
                        selected_indices
                    ).mean()
                )
                random_period_returns[trial].append(selected_return)
                random_period_costs[trial].append(
                    _round_trip_cost(1.0, cfg)
                )

            period_rows.append(
                {
                    "trade_date": trade_date,
                    "holding_count": holding_count,
                    "gbm_gross_return": gbm_gross,
                    "gbm_net_return": gbm_gross - gbm_cost_rate,
                    "logistic_gross_return": logistic_gross,
                    "logistic_net_return": (
                        logistic_gross - logistic_cost_rate
                    ),
                    "benchmark_return": benchmark,
                    "gbm_turnover": gbm_turnover,
                    "logistic_turnover": logistic_turnover,
                    "gbm_cost_rate": gbm_cost_rate,
                    "logistic_cost_rate": logistic_cost_rate,
                    "holdings": gbm_selected["stock_code"].to_list(),
                }
            )

        if len(period_rows) < 5:
            raise ValueError("Not enough valid rebalance periods")

        gbm_gross_curve = _equity_curve(
            [row["gbm_gross_return"] for row in period_rows]
        )
        gbm_net_curve = _equity_curve(
            [row["gbm_net_return"] for row in period_rows]
        )
        logistic_net_curve = _equity_curve(
            [row["logistic_net_return"] for row in period_rows]
        )
        benchmark_curve = _equity_curve(
            [row["benchmark_return"] for row in period_rows]
        )

        curve = []
        for index, row in enumerate(period_rows):
            curve.append(
                {
                    **row,
                    "trade_date": str(row["trade_date"]),
                    "gbm_gross_equity": gbm_gross_curve[index],
                    "gbm_net_equity": gbm_net_curve[index],
                    "logistic_net_equity": logistic_net_curve[index],
                    "benchmark_equity": benchmark_curve[index],
                }
            )

        gbm_metrics = _performance_metrics(
            [row["gbm_net_return"] for row in period_rows],
            [row["gbm_gross_return"] for row in period_rows],
            rebalance_days=cfg.rebalance_days,
            total_cost=sum(row["gbm_cost_rate"] for row in period_rows),
            average_turnover=float(
                np.mean([row["gbm_turnover"] for row in period_rows])
            ),
        )
        logistic_metrics = _performance_metrics(
            [row["logistic_net_return"] for row in period_rows],
            [row["logistic_gross_return"] for row in period_rows],
            rebalance_days=cfg.rebalance_days,
            total_cost=sum(
                row["logistic_cost_rate"] for row in period_rows
            ),
            average_turnover=float(
                np.mean([row["logistic_turnover"] for row in period_rows])
            ),
        )
        benchmark_metrics = _performance_metrics(
            [row["benchmark_return"] for row in period_rows],
            [row["benchmark_return"] for row in period_rows],
            rebalance_days=cfg.rebalance_days,
            total_cost=0.0,
            average_turnover=0.0,
        )
        random_metrics = []
        for trial in range(cfg.random_trials):
            net_returns = [
                gross - cost
                for gross, cost in zip(
                    random_period_returns[trial],
                    random_period_costs[trial],
                    strict=True,
                )
            ]
            random_metrics.append(
                _performance_metrics(
                    net_returns,
                    random_period_returns[trial],
                    rebalance_days=cfg.rebalance_days,
                    total_cost=sum(random_period_costs[trial]),
                    average_turnover=1.0,
                )
            )
        random_summary = {
            "trials": cfg.random_trials,
            "mean_total_return": _mean_metric(
                random_metrics,
                "total_return",
            ),
            "mean_annualized_return": _mean_metric(
                random_metrics,
                "annualized_return",
            ),
            "mean_sharpe": _mean_metric(random_metrics, "sharpe"),
            "mean_max_drawdown": _mean_metric(
                random_metrics,
                "max_drawdown",
            ),
        }

        return BacktestResult(
            report={
                "metadata": {
                    "model": "HistGradientBoosting",
                    "strategy": "top_probability_long_only",
                    "top_fraction": cfg.top_fraction,
                    "holding_days": cfg.holding_days,
                    "rebalance_days": cfg.rebalance_days,
                    "periods": len(period_rows),
                    "start_date": str(rebalance_dates[0]),
                    "end_date": str(rebalance_dates[-1]),
                },
                "costs": {
                    "commission_each_side": cfg.commission_each_side,
                    "stamp_duty_sell": cfg.stamp_duty_sell,
                    "slippage_each_side": cfg.slippage_each_side,
                    "full_turnover_round_trip": (
                        2 * cfg.commission_each_side
                        + cfg.stamp_duty_sell
                        + 2 * cfg.slippage_each_side
                    ),
                },
                "gbm": gbm_metrics,
                "logistic_baseline": logistic_metrics,
                "equal_weight_benchmark": benchmark_metrics,
                "random_baseline": random_summary,
                "probability_analysis": _probability_analysis(frame),
                "equity_curve": curve,
            }
        )

    @staticmethod
    def _prepare(frame: pl.DataFrame) -> pl.DataFrame:
        required = {
            "stock_code",
            "trade_date",
            "future_return",
            "future_up",
            "predicted_return",
            "up_probability",
            "logistic_probability",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Missing backtest columns: {', '.join(sorted(missing))}"
            )
        return frame.filter(
            pl.col("future_return").is_not_null()
            & pl.col("future_return").is_finite()
            & pl.col("up_probability").is_not_null()
            & pl.col("logistic_probability").is_not_null()
        ).sort(["trade_date", "stock_code"])

    @staticmethod
    def _validate_config(config: BacktestConfig) -> None:
        if not 0 < config.top_fraction <= 1:
            raise ValueError("top_fraction must be between 0 and 1")
        if config.rebalance_days < 1:
            raise ValueError("rebalance_days must be positive")
        if config.holding_days < 1:
            raise ValueError("holding_days must be positive")
        if config.holding_days > config.rebalance_days:
            raise ValueError("holding_days cannot exceed rebalance_days")
        if config.random_trials < 1:
            raise ValueError("random_trials must be positive")
        for cost in (
            config.commission_each_side,
            config.stamp_duty_sell,
            config.slippage_each_side,
        ):
            if cost < 0:
                raise ValueError("trading costs cannot be negative")


def _equal_weights(codes: list[str]) -> dict[str, float]:
    if not codes:
        return {}
    weight = 1.0 / len(codes)
    return {code: weight for code in codes}


def _turnover(
    previous: dict[str, float],
    current: dict[str, float],
) -> float:
    if not previous:
        return 1.0
    return 0.5 * sum(
        abs(current.get(code, 0.0) - previous.get(code, 0.0))
        for code in set(previous) | set(current)
    )


def _round_trip_cost(turnover: float, config: BacktestConfig) -> float:
    return turnover * (
        2 * config.commission_each_side
        + config.stamp_duty_sell
        + 2 * config.slippage_each_side
    )


def _equity_curve(returns: list[float]) -> list[float]:
    equity = [1.0]
    for value in returns:
        equity.append(equity[-1] * (1.0 + value))
    return equity[1:]


def _performance_metrics(
    net_returns: list[float],
    gross_returns: list[float],
    *,
    rebalance_days: int,
    total_cost: float,
    average_turnover: float,
) -> dict[str, Any]:
    net = np.asarray(net_returns, dtype=float)
    gross = np.asarray(gross_returns, dtype=float)
    periods_per_year = 252.0 / rebalance_days
    net_curve = np.asarray(_equity_curve(net_returns), dtype=float)
    running_peak = np.maximum.accumulate(net_curve)
    drawdown = net_curve / running_peak - 1.0
    standard_deviation = float(np.std(net, ddof=1)) if net.size > 1 else 0.0
    sharpe = (
        float(np.mean(net)) / standard_deviation * math.sqrt(periods_per_year)
        if standard_deviation > 0
        else None
    )
    annualized = (
        float(net_curve[-1] ** (periods_per_year / net.size) - 1.0)
        if net_curve[-1] > 0
        else -1.0
    )
    return {
        "total_return": _safe_float(net_curve[-1] - 1.0),
        "gross_total_return": _safe_float(np.prod(1.0 + gross) - 1.0),
        "annualized_return": _safe_float(annualized),
        "annualized_volatility": _safe_float(
            standard_deviation * math.sqrt(periods_per_year)
        ),
        "sharpe": _safe_float(sharpe),
        "max_drawdown": _safe_float(np.min(drawdown)),
        "hit_rate": _safe_float(np.mean(net > 0)),
        "average_period_return": _safe_float(np.mean(net)),
        "periods": int(net.size),
        "total_cost_rate": _safe_float(total_cost),
        "average_turnover": _safe_float(average_turnover),
    }


def _probability_analysis(frame: pl.DataFrame) -> dict[str, Any]:
    buckets = [
        ("<0.45", 0.0, 0.45),
        ("0.45-0.50", 0.45, 0.50),
        ("0.50-0.55", 0.50, 0.55),
        ("0.55-0.60", 0.55, 0.60),
        (">=0.60", 0.60, 1.000001),
    ]
    bucket_results = []
    for label, lower, upper in buckets:
        selected = frame.filter(
            (pl.col("up_probability") >= lower)
            & (pl.col("up_probability") < upper)
        )
        bucket_results.append(
            _subset_stats(selected, label, lower, upper)
        )
    thresholds = []
    for threshold in [0.50, 0.55, 0.60, 0.65]:
        selected = frame.filter(
            pl.col("up_probability") >= threshold
        )
        thresholds.append(
            _subset_stats(selected, f">={threshold:.2f}", threshold, None)
        )
    return {"buckets": bucket_results, "thresholds": thresholds}


def _subset_stats(
    frame: pl.DataFrame,
    label: str,
    lower: float,
    upper: float | None,
) -> dict[str, Any]:
    return {
        "label": label,
        "lower": lower,
        "upper": upper,
        "rows": frame.height,
        "actual_up_rate": (
            _safe_float(frame["future_up"].mean())
            if frame.height
            else None
        ),
        "average_future_return": (
            _safe_float(frame["future_return"].mean())
            if frame.height
            else None
        ),
    }


def _mean_metric(metrics: list[dict[str, Any]], name: str) -> float | None:
    values = [
        float(item[name])
        for item in metrics
        if item.get(name) is not None
    ]
    return _safe_float(np.mean(values)) if values else None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 8)
