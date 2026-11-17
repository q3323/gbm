from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from app.services.backtest import BacktestConfig, StrategyBacktestService
from app.services.calibration import fit_platt, prequential_platt
from app.services.gbm import GBMService
from app.services.nested_tuning import (
    _classification_metrics,
    _cross_sectional_metrics,
)
from app.symbols import normalize_model_code


TECH_FACTOR_POOL = (
    "return_1d",
    "log_return_1d",
    "close_to_ma_5",
    "close_to_ma_20",
    "ma_5_to_20",
    "ma_20_to_60",
    "macd_histogram_ratio",
    "rsi_14_scaled",
    "kdj_k_scaled",
    "kdj_d_scaled",
    "boll_position",
    "atr_14_ratio",
    "volatility_20",
    "volume_ratio_5",
    "volume_to_ma_20",
    "turnover",
    "return_5d",
    "return_20d",
    "excess_return_5d",
    "market_return_5d",
    "market_volatility_20",
    "relative_strength_20",
    "market_return_1d",
    "market_positive_ratio_1d",
    "market_above_ma20_ratio",
    "market_dispersion_1d",
    "industry_return_5d",
    "industry_momentum_20d",
    "industry_above_ma20_ratio",
    "excess_return_vs_industry_5d",
    "industry_relative_rank_5d",
    "fund_roe",
    "fund_net_margin",
    "fund_net_profit_yoy",
    "fund_eps_yoy",
    "fund_equity_yoy",
    "fund_asset_turnover",
    "fund_cfo_to_revenue",
)

PHARMA_FACTOR_POOL = (
    "return_1d",
    "log_return_1d",
    "close_to_ma_5",
    "close_to_ma_20",
    "ma_5_to_20",
    "ma_20_to_60",
    "macd_histogram_ratio",
    "rsi_14_scaled",
    "kdj_k_scaled",
    "kdj_d_scaled",
    "boll_position",
    "atr_14_ratio",
    "volatility_20",
    "volume_ratio_5",
    "volume_to_ma_20",
    "turnover",
    "return_5d",
    "return_20d",
    "excess_return_5d",
    "market_return_5d",
    "market_volatility_20",
    "relative_strength_20",
    "industry_return_5d",
    "industry_momentum_20d",
    "excess_return_vs_industry_5d",
    "fund_roe",
    "fund_net_margin",
    "fund_net_profit_yoy",
    "fund_eps_yoy",
    "fund_debt_ratio",
    "fund_equity_yoy",
    "fund_asset_turnover",
    "fund_cfo_to_revenue",
)

US_ETF_FACTOR_POOL = (
    "return_1d",
    "log_return_1d",
    "close_to_ma_5",
    "close_to_ma_20",
    "ma_5_to_20",
    "ma_20_to_60",
    "macd_histogram_ratio",
    "rsi_14_scaled",
    "kdj_k_scaled",
    "kdj_d_scaled",
    "boll_position",
    "atr_14_ratio",
    "volatility_20",
    "volume_ratio_5",
    "volume_to_ma_20",
    "turnover",
    "return_5d",
    "return_20d",
    "excess_return_5d",
    "market_return_5d",
    "market_volatility_20",
    "relative_strength_20",
    "market_return_1d",
    "market_positive_ratio_1d",
    "market_above_ma20_ratio",
    "market_dispersion_1d",
)


@dataclass(frozen=True)
class GroupBaggingConfig:
    group_id: str
    name: str
    market: str
    stock_codes: tuple[str, ...]
    seeds: tuple[int, ...] = (7, 42, 2024, 123, 888)
    sample_fraction: float = 0.8
    horizon_days: int = 10
    date_block_size: int = 10
    factor_pool: tuple[str, ...] = ()
    minimum_available_stocks: int = 3


@dataclass(frozen=True)
class GroupBaggingResult:
    report: dict[str, Any]
    out_of_sample: pl.DataFrame | None = None


def load_group_configs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("groups", [])
    if not isinstance(groups, list):
        raise ValueError("model_groups.json must contain a groups list")
    return groups


class GroupedGBMBaggingService:
    def run(
        self,
        daily: pl.DataFrame,
        config: GroupBaggingConfig,
        *,
        financial: pl.DataFrame | None = None,
        benchmark_daily: pl.DataFrame | None = None,
        min_training_rows: int = 800,
        walk_forward_folds: int = 4,
        min_test_rows_per_fold: int = 10,
    ) -> GroupBaggingResult:
        _validate_config(config)
        stock_codes = tuple(
            dict.fromkeys(normalize_model_code(code) for code in config.stock_codes)
        )
        if len(stock_codes) < 3:
            raise ValueError("At least three stocks are required for a group model")

        group_daily = daily.filter(
            pl.col("stock_code").is_in(stock_codes)
        )
        if group_daily.is_empty():
            raise ValueError(f"No daily data for group {config.group_id}")
        date_counts = (
            group_daily.group_by("trade_date")
            .agg(pl.col("stock_code").n_unique().alias("stock_count"))
            .filter(
                pl.col("stock_count") >= config.minimum_available_stocks
            )
        )
        if date_counts.is_empty():
            raise ValueError(
                f"Group {config.group_id} never reaches "
                f"{config.minimum_available_stocks} available stocks"
            )
        first_eligible_date = date_counts["trade_date"].min()
        group_daily = group_daily.filter(
            pl.col("trade_date") >= first_eligible_date
        )
        group_financial = None
        if financial is not None and not financial.is_empty():
            group_financial = financial.filter(
                pl.col("stock_code").is_in(stock_codes)
            )

        seed_predictions: dict[str, list[dict[str, float | str]]] = {
            code: [] for code in stock_codes
        }
        seed_details: list[dict[str, Any]] = []
        seed_oos_frames: list[pl.DataFrame] = []
        available_dates = (
            group_daily["trade_date"].unique().sort().to_list()
        )
        for seed in config.seeds:
            training_dates = _sample_training_dates(
                available_dates,
                config.sample_fraction,
                seed,
                config.date_block_size,
            )
            result = GBMService().analyze(
                group_daily,
                stock_codes[0],
                seed=seed,
                min_stocks=config.minimum_available_stocks,
                min_training_rows=min_training_rows,
                min_trading_days=120,
                horizon=config.horizon_days,
                walk_forward_folds=walk_forward_folds,
                min_test_rows_per_fold=min_test_rows_per_fold,
                financial=group_financial,
                benchmark_daily=benchmark_daily,
                training_dates=training_dates,
                feature_selection=True,
                candidate_features=list(config.factor_pool) or None,
                max_selected_features=20,
                minimum_feature_icir=0.05,
            )
            latest_predictions = result.report.get("latest_predictions", [])
            if not latest_predictions:
                raise ValueError(
                    f"Seed {seed} produced no latest group predictions"
                )
            seed_cross_sectional = _cross_sectional_metrics(
                result.out_of_sample["trade_date"].to_list(),
                result.out_of_sample["future_return"].to_numpy(),
                result.out_of_sample["predicted_return"].to_numpy(),
            )
            for prediction in latest_predictions:
                code = str(prediction["stock_code"])
                if code in seed_predictions:
                    seed_predictions[code].append(prediction)
            reliability = result.report["reliability"]
            seed_details.append(
                {
                    "seed": seed,
                    "training_date_count": len(training_dates),
                    "reliability": reliability["status"],
                    "regression_reliability": reliability[
                        "regression_status"
                    ],
                    "classification_reliability": reliability[
                        "classification_status"
                    ],
                    "reliability_reasons": reliability.get("reasons", []),
                    "rank_ic_mean": seed_cross_sectional.get(
                        "mean_rank_ic"
                    ),
                    "positive_rank_ic_rate": seed_cross_sectional.get(
                        "positive_rank_ic_rate"
                    ),
                    "selected_features": result.report.get("features", []),
                    "candidate_feature_count": result.report["metadata"].get(
                        "candidate_feature_count"
                    ),
                    "fold_features": [
                        {
                            "fold": fold.get("fold"),
                            "features": fold.get("selected_features", []),
                        }
                        for fold in result.report.get(
                            "walk_forward",
                            {},
                        ).get("folds", [])
                    ],
                    "auc": result.report["classification"].get("roc_auc"),
                    "accuracy": result.report["classification"].get("accuracy"),
                }
            )
            seed_oos_frames.append(
                result.out_of_sample.with_columns(
                    pl.lit(seed).alias("seed")
                )
            )

        aggregated_oos = _aggregate_group_oos(seed_oos_frames)
        raw_oos_probabilities = aggregated_oos["up_probability"].to_numpy()
        oos_actual_up = aggregated_oos["future_up"].to_numpy().astype(int)
        current_calibrator = fit_platt(
            raw_oos_probabilities,
            oos_actual_up,
            minimum_samples=60,
        )

        rankings = []
        for code in stock_codes:
            predictions = seed_predictions[code]
            if not predictions:
                continue
            expected_returns = np.asarray(
                [
                    float(item["expected_return"])
                    for item in predictions
                    if item.get("expected_return") is not None
                ],
                dtype=float,
            )
            probabilities = np.asarray(
                [
                    float(item["up_probability"])
                    for item in predictions
                    if item.get("up_probability") is not None
                ],
                dtype=float,
            )
            if expected_returns.size == 0 or probabilities.size == 0:
                continue
            mean_return = float(expected_returns.mean())
            raw_probability = float(probabilities.mean())
            calibrated_probability = (
                float(
                    current_calibrator.transform(
                        np.asarray([raw_probability])
                    )[0]
                )
                if current_calibrator is not None
                else raw_probability
            )
            trend_values = []
            for item in predictions:
                close_ma5 = item.get("close_to_ma_5")
                close_ma20 = item.get("close_to_ma_20")
                ma5_ma20 = item.get("ma_5_to_20")
                ma20_ma60 = item.get("ma_20_to_60")
                checks = [
                    value is not None and float(value) > 0
                    for value in (
                        close_ma5,
                        close_ma20,
                        ma5_ma20,
                        ma20_ma60,
                    )
                ]
                trend_values.append(sum(checks) / len(checks))
            trend_score = float(np.mean(trend_values))
            rankings.append(
                {
                    "stock_code": code,
                    "expected_return": round(mean_return, 8),
                    "raw_up_probability": round(raw_probability, 8),
                    "up_probability": round(calibrated_probability, 8),
                    "calibration_shift": round(
                        calibrated_probability - raw_probability,
                        8,
                    ),
                    "prediction_std": round(
                        float(expected_returns.std(ddof=1))
                        if expected_returns.size > 1
                        else 0.0,
                        8,
                    ),
                    "direction": (
                        "up"
                        if mean_return > 0
                        else "down"
                        if mean_return < 0
                        else "mixed"
                    ),
                    "trend_score": round(trend_score, 8),
                    "trend_eligible": trend_score >= 0.5,
                    "seed_count": expected_returns.size,
                }
            )

        if not rankings:
            raise ValueError(f"No aggregated predictions for {config.group_id}")
        rankings.sort(key=lambda item: item["expected_return"], reverse=True)
        last_index = max(1, len(rankings) - 1)
        for index, item in enumerate(rankings):
            item["rank"] = index + 1
            item["rank_score"] = round(
                (last_index - index) / last_index * 100.0,
                2,
            )
            item["final_score"] = round(
                0.7 * item["rank_score"]
                + 0.3 * item["trend_score"] * 100.0,
                2,
            )
            item["trade_signal"] = (
                item["trend_eligible"]
                and item["expected_return"] > 0
                and item["up_probability"] >= 0.55
            )

        all_returns = np.asarray(
            [item["expected_return"] for item in rankings],
            dtype=float,
        )
        all_probabilities = np.asarray(
            [item["up_probability"] for item in rankings],
            dtype=float,
        )
        all_raw_probabilities = np.asarray(
            [item["raw_up_probability"] for item in rankings],
            dtype=float,
        )
        all_std = np.asarray(
            [item["prediction_std"] for item in rankings],
            dtype=float,
        )
        all_trend_scores = np.asarray(
            [item["trend_score"] for item in rankings],
            dtype=float,
        )
        evaluation = _evaluate_group_oos(
            aggregated_oos,
            config.horizon_days,
        )
        report = {
            "metadata": {
                "group_id": config.group_id,
                "name": config.name,
                "market": config.market,
                "model": "grouped_gbm_bagging",
                "horizon_days": config.horizon_days,
                "seeds": list(config.seeds),
                "sample_fraction": config.sample_fraction,
                "calibration": "platt",
                "target": "future_excess_return",
                "stock_count": len(rankings),
                "generated_on": date.today().isoformat(),
                "as_of_date": max(
                    str(item.get("trade_date", ""))
                    for seed_items in seed_predictions.values()
                    for item in seed_items
                ),
            },
            "group_prediction": {
                "expected_return": round(float(all_returns.mean()), 8),
                "raw_up_probability": round(
                    float(all_raw_probabilities.mean()),
                    8,
                ),
                "up_probability": round(float(all_probabilities.mean()), 8),
                "calibration_shift": round(
                    float(all_probabilities.mean())
                    - float(all_raw_probabilities.mean()),
                    8,
                ),
                "prediction_std": round(float(all_std.mean()), 8),
                "trend_score": round(float(all_trend_scores.mean()), 8),
                "trend_eligible_ratio": round(
                    float(
                        np.mean(
                            [item["trend_eligible"] for item in rankings]
                        )
                    ),
                    8,
                ),
                "trade_signal_count": sum(
                    bool(item["trade_signal"]) for item in rankings
                ),
                "direction": (
                    "up"
                    if float(all_returns.mean()) > 0
                    else "down"
                    if float(all_returns.mean()) < 0
                    else "mixed"
                ),
            },
            "rankings": rankings,
            "evaluation": evaluation,
            "seeds": seed_details,
        }
        return GroupBaggingResult(
            report=report,
            out_of_sample=aggregated_oos,
        )


def _validate_config(config: GroupBaggingConfig) -> None:
    if not config.seeds:
        raise ValueError("At least one seed is required")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("Seeds must be unique")
    if not 0 < config.sample_fraction <= 1:
        raise ValueError("sample_fraction must be between 0 and 1")
    if not 1 <= config.horizon_days <= 60:
        raise ValueError("horizon_days must be between 1 and 60")


def _sample_training_dates(
    available_dates: list[date],
    fraction: float,
    seed: int,
    block_size: int,
) -> list[date]:
    blocks = [
        available_dates[index:index + max(2, block_size)]
        for index in range(0, len(available_dates), max(2, block_size))
    ]
    selected_count = max(1, math.ceil(len(blocks) * fraction))
    selected_blocks = random.Random(seed).sample(blocks, selected_count)
    return [
        trade_date
        for block in sorted(selected_blocks, key=lambda values: values[0])
        for trade_date in block
    ]


def _aggregate_group_oos(seed_frames: list[pl.DataFrame]) -> pl.DataFrame:
    oos = pl.concat(seed_frames, how="diagonal_relaxed")
    return (
        oos.group_by(["stock_code", "trade_date"])
        .agg(
            pl.col("predicted_return").mean(),
            pl.col("up_probability").mean(),
            pl.col("logistic_probability").mean(),
            pl.col("close_to_ma_5").mean(),
            pl.col("close_to_ma_20").mean(),
            pl.col("ma_5_to_20").mean(),
            pl.col("ma_20_to_60").mean(),
            pl.col("return_20d").mean(),
            pl.col("future_return").first(),
            pl.col("future_up").first(),
            pl.col("fold").first(),
        )
        .sort(["trade_date", "stock_code"])
    )


def _evaluate_group_oos(
    aggregated: pl.DataFrame,
    horizon: int,
) -> dict[str, Any]:
    actual_up = aggregated["future_up"].to_numpy().astype(int)
    raw_probabilities = aggregated["up_probability"].to_numpy()
    calibrated_probabilities = prequential_platt(
        raw_probabilities,
        actual_up,
        aggregated["trade_date"].to_list(),
        minimum_samples=60,
    )
    classification = _classification_metrics(
        actual_up,
        calibrated_probabilities,
    )
    raw_classification = _classification_metrics(
        actual_up,
        raw_probabilities,
    )
    cross_sectional = _cross_sectional_metrics(
        aggregated["trade_date"].to_list(),
        aggregated["future_return"].to_numpy(),
        aggregated["predicted_return"].to_numpy(),
    )
    momentum_cross_sectional = _cross_sectional_metrics(
        aggregated["trade_date"].to_list(),
        aggregated["future_return"].to_numpy(),
        aggregated["return_20d"].to_numpy(),
    )
    logistic_cross_sectional = _cross_sectional_metrics(
        aggregated["trade_date"].to_list(),
        aggregated["future_return"].to_numpy(),
        aggregated["logistic_probability"].to_numpy(),
    )
    rank_ic_std = cross_sectional.get("rank_ic_std")
    rank_ic_mean = cross_sectional.get("mean_rank_ic")
    icir = (
        rank_ic_mean / rank_ic_std
        if rank_ic_mean is not None
        and rank_ic_std is not None
        and rank_ic_std > 0
        else None
    )

    backtest_frame = aggregated.with_columns(
        pl.Series("up_probability", calibrated_probabilities)
    )
    backtest = StrategyBacktestService().run(
        backtest_frame,
        BacktestConfig(
            rank_column="predicted_return",
            top_fraction=0.2,
            holding_days=horizon,
            rebalance_days=horizon,
            seed=42,
        ),
    ).report["gbm"]
    trend_frame = backtest_frame.with_columns(
        _trend_score_expression().alias("trend_score")
    ).with_columns(
        (pl.col("trend_score") >= 0.5).alias("trend_eligible")
    )
    filtered_frame = trend_frame.filter(
        pl.col("trend_eligible")
        & (pl.col("up_probability") >= 0.55)
        & (pl.col("predicted_return") > 0)
    )
    trend_filtered_backtest = None
    trend_filter_error = None
    try:
        trend_filtered_backtest = StrategyBacktestService().run(
            filtered_frame,
            BacktestConfig(
                rank_column="predicted_return",
                top_fraction=0.2,
                holding_days=horizon,
                rebalance_days=horizon,
                seed=42,
            ),
        ).report["gbm"]
    except Exception as exc:
        trend_filter_error = str(exc)
    return {
        "rank_ic_mode": "cross_sectional",
        "rank_ic_mean": _round_or_none(rank_ic_mean),
        "rank_ic_std": _round_or_none(rank_ic_std),
        "icir": _round_or_none(icir),
        "positive_rank_ic_rate": _round_or_none(
            cross_sectional.get("positive_rank_ic_rate")
        ),
        "accuracy": _round_or_none(classification.get("accuracy")),
        "roc_auc": _round_or_none(classification.get("roc_auc")),
        "raw_brier_score": _round_or_none(
            raw_classification.get("brier_score")
        ),
        "calibrated_brier_score": _round_or_none(
            classification.get("brier_score")
        ),
        "mean_calibration_shift": _round_or_none(
            float(
                np.mean(
                    calibrated_probabilities - raw_probabilities
                )
            )
        ),
        "total_return": _round_or_none(backtest.get("total_return")),
        "sharpe": _round_or_none(backtest.get("sharpe")),
        "max_drawdown": _round_or_none(backtest.get("max_drawdown")),
        "hit_rate": _round_or_none(backtest.get("hit_rate")),
        "periods": backtest.get("periods"),
        "trend_eligible_ratio": _round_or_none(
            float(trend_frame["trend_eligible"].mean())
        ),
        "trend_filtered_total_return": _round_or_none(
            trend_filtered_backtest.get("total_return")
            if trend_filtered_backtest
            else None
        ),
        "trend_filtered_sharpe": _round_or_none(
            trend_filtered_backtest.get("sharpe")
            if trend_filtered_backtest
            else None
        ),
        "trend_filtered_max_drawdown": _round_or_none(
            trend_filtered_backtest.get("max_drawdown")
            if trend_filtered_backtest
            else None
        ),
        "trend_filtered_periods": (
            trend_filtered_backtest.get("periods")
            if trend_filtered_backtest
            else None
        ),
        "trend_filter_error": trend_filter_error,
        "baselines": {
            "momentum_rank_ic": _round_or_none(
                momentum_cross_sectional.get("mean_rank_ic")
            ),
            "momentum_icir": _icir(momentum_cross_sectional),
            "logistic_rank_ic": _round_or_none(
                logistic_cross_sectional.get("mean_rank_ic")
            ),
            "logistic_icir": _icir(logistic_cross_sectional),
            "gbm_beats_momentum": _beats_baseline(
                cross_sectional,
                momentum_cross_sectional,
            ),
            "gbm_beats_logistic": _beats_baseline(
                cross_sectional,
                logistic_cross_sectional,
            ),
        },
        "sample_rows": aggregated.height,
    }


def _trend_score_expression() -> pl.Expr:
    checks = [
        (pl.col("close_to_ma_5") > 0).cast(pl.Float64),
        (pl.col("close_to_ma_20") > 0).cast(pl.Float64),
        (pl.col("ma_5_to_20") > 0).cast(pl.Float64),
        (pl.col("ma_20_to_60") > 0).cast(pl.Float64),
    ]
    return sum(checks) / len(checks)


def _round_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 8)


def _icir(metrics: dict[str, Any]) -> float | None:
    mean_ic = metrics.get("mean_rank_ic")
    std_ic = metrics.get("rank_ic_std")
    if mean_ic is None or std_ic is None or std_ic <= 0:
        return None
    return round(float(mean_ic) / float(std_ic), 8)


def _beats_baseline(
    model_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
) -> bool:
    model_ic = model_metrics.get("mean_rank_ic")
    baseline_ic = baseline_metrics.get("mean_rank_ic")
    return (
        model_ic is not None
        and baseline_ic is not None
        and float(model_ic) > float(baseline_ic)
    )
