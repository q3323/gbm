from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)
from scipy.stats import spearmanr

from app.services.calibration import fit_platt, prequential_platt

INDEX_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "log_return_1d",
    "close_to_ma_5",
    "close_to_ma_20",
    "close_to_ma_60",
    "ma_5_to_20",
    "ma_20_to_60",
    "volatility_20",
    "volume_ratio_5",
    "volume_to_ma_20",
    "high_low_range",
    "close_position_20",
    "drawdown_20",
    "rsi_14_scaled",
    "macd_histogram_ratio",
]


@dataclass(frozen=True)
class IndexBaggingConfig:
    group_id: str
    name: str
    symbol: str
    seeds: tuple[int, ...] = (7, 42, 2024, 123, 888)
    sample_fraction: float = 0.8
    horizon_days: int = 10
    block_size: int = 10


@dataclass(frozen=True)
class IndexBaggingResult:
    report: dict[str, Any]


class IndexGBMBaggingService:
    def run(
        self,
        daily: pl.DataFrame,
        config: IndexBaggingConfig,
        *,
        benchmark_daily: pl.DataFrame | None = None,
    ) -> IndexBaggingResult:
        _validate_config(config)
        frame = _prepare_frame(
            daily,
            config.horizon_days,
            benchmark_daily=benchmark_daily,
        )
        training = frame.filter(
            pl.col("future_return").is_not_null()
            & pl.col("future_up").is_not_null()
        ).drop_nulls(subset=INDEX_FEATURES)
        latest = frame.drop_nulls(subset=INDEX_FEATURES).tail(1)
        if training.height < 180:
            raise ValueError("At least 180 labeled rows are required for index GBM")
        if latest.is_empty():
            raise ValueError("No current feature row is available for index GBM")

        split_index = max(120, int(training.height * 0.8))
        evaluation_train = training.head(
            max(1, split_index - config.horizon_days)
        )
        evaluation_test = training.slice(split_index)
        if evaluation_test.height < 20:
            raise ValueError("Not enough holdout rows for index GBM evaluation")

        seed_predictions = []
        seed_details = []
        evaluation_return_parts: list[np.ndarray] = []
        evaluation_probability_parts: list[np.ndarray] = []
        for seed in config.seeds:
            sampled_train = _sample_time_blocks(
                evaluation_train,
                config.sample_fraction,
                seed,
                config.block_size,
            )
            evaluation_features = _select_index_features(
                sampled_train,
                maximum_features=10,
                minimum_icir=0.05,
            )
            eval_regressor, eval_classifier = _fit_models(
                sampled_train,
                seed,
                evaluation_features,
            )
            test_x = evaluation_test.select(
                evaluation_features
            ).to_numpy()
            test_actual_return = evaluation_test["future_return"].to_numpy()
            test_actual_up = evaluation_test["future_up"].to_numpy()
            predicted_return = eval_regressor.predict(test_x)
            up_probability = eval_classifier.predict_proba(test_x)[:, 1]
            evaluation_return_parts.append(predicted_return)
            evaluation_probability_parts.append(up_probability)

            final_train = _sample_time_blocks(
                training,
                config.sample_fraction,
                seed + 10_000,
                config.block_size,
            )
            final_features = _select_index_features(
                final_train,
                maximum_features=10,
                minimum_icir=0.05,
            )
            final_regressor, final_classifier = _fit_models(
                final_train,
                seed,
                final_features,
            )
            final_latest_x = latest.select(final_features).to_numpy()
            expected_return = float(final_regressor.predict(final_latest_x)[0])
            expected_probability = float(
                final_classifier.predict_proba(final_latest_x)[0, 1]
            )
            seed_predictions.append(
                {
                    "expected_return": expected_return,
                    "up_probability": expected_probability,
                }
            )
            seed_details.append(
                {
                    "seed": seed,
                    "selected_features": final_features,
                    "rmse": round(
                        float(
                            np.sqrt(
                                np.mean(
                                    (test_actual_return - predicted_return) ** 2
                                )
                            )
                        ),
                        8,
                    ),
                    "mae": round(
                        float(
                            mean_absolute_error(
                                test_actual_return,
                                predicted_return,
                            )
                        ),
                        8,
                    ),
                    "accuracy": round(
                        float(
                            accuracy_score(
                                test_actual_up,
                                up_probability >= 0.5,
                            )
                        ),
                        8,
                    ),
                    "roc_auc": _safe_auc(test_actual_up, up_probability),
                }
            )

        returns = np.asarray(
            [item["expected_return"] for item in seed_predictions],
            dtype=float,
        )
        probabilities = np.asarray(
            [item["up_probability"] for item in seed_predictions],
            dtype=float,
        )
        expected_return = float(returns.mean())
        raw_up_probability = float(probabilities.mean())
        prediction_std = (
            float(returns.std(ddof=1)) if returns.size > 1 else 0.0
        )
        evaluation_dates = evaluation_test["trade_date"].to_list()
        evaluation_actual_up = evaluation_test["future_up"].to_numpy()
        evaluation_returns = np.vstack(evaluation_return_parts).mean(axis=0)
        evaluation_probabilities = np.vstack(
            evaluation_probability_parts
        ).mean(axis=0)
        calibrated_evaluation_probabilities = prequential_platt(
            evaluation_probabilities,
            evaluation_actual_up,
            evaluation_dates,
            minimum_samples=60,
        )
        current_calibrator = fit_platt(
            evaluation_probabilities,
            evaluation_actual_up,
            minimum_samples=60,
        )
        up_probability = (
            float(
                current_calibrator.transform(
                    np.asarray([raw_up_probability])
                )[0]
            )
            if current_calibrator is not None
            else raw_up_probability
        )
        calibration_shift = up_probability - raw_up_probability
        evaluation = _evaluate_time_series_predictions(
            actual_returns=evaluation_test["future_return"].to_numpy(),
            actual_up=evaluation_actual_up,
            predicted_returns=evaluation_returns,
            raw_probabilities=evaluation_probabilities,
            calibrated_probabilities=calibrated_evaluation_probabilities,
            trend_scores=_trend_scores(evaluation_test),
            horizon=config.horizon_days,
        )
        latest_row = latest.row(0, named=True)
        latest_trend_score = _trend_score_from_row(latest_row)
        report = {
            "metadata": {
                "group_id": config.group_id,
                "name": config.name,
                "market": "US",
                "model": "index_gbm_bagging",
                "symbol": config.symbol,
                "horizon_days": config.horizon_days,
                "seeds": list(config.seeds),
                "sample_fraction": config.sample_fraction,
                "calibration": "platt",
                "target": "future_excess_return",
                "stock_count": 1,
                "generated_on": date.today().isoformat(),
                "as_of_date": str(latest_row["trade_date"]),
            },
            "group_prediction": {
                "expected_return": round(expected_return, 8),
                "raw_up_probability": round(raw_up_probability, 8),
                "up_probability": round(up_probability, 8),
                "calibration_shift": round(calibration_shift, 8),
                "prediction_std": round(prediction_std, 8),
                "direction": (
                    "up"
                    if expected_return > 0
                    else "down"
                    if expected_return < 0
                    else "mixed"
                ),
                "trend_score": round(latest_trend_score, 8),
                "trend_eligible": latest_trend_score >= 0.5,
            },
            "rankings": [
                {
                    "stock_code": config.symbol,
                    "expected_return": round(expected_return, 8),
                    "raw_up_probability": round(raw_up_probability, 8),
                    "up_probability": round(up_probability, 8),
                    "calibration_shift": round(calibration_shift, 8),
                    "prediction_std": round(prediction_std, 8),
                    "direction": (
                        "up"
                        if expected_return > 0
                        else "down"
                        if expected_return < 0
                        else "mixed"
                    ),
                    "seed_count": len(config.seeds),
                    "rank": 1,
                    "rank_score": 100.0,
                    "trend_score": round(latest_trend_score, 8),
                    "trend_eligible": latest_trend_score >= 0.5,
                    "final_score": round(
                        70.0 + 30.0 * latest_trend_score,
                        2,
                    ),
                    "trade_signal": (
                        latest_trend_score >= 0.5
                        and expected_return > 0
                        and up_probability >= 0.55
                    ),
                }
            ],
            "evaluation": evaluation,
            "seeds": seed_details,
        }
        return IndexBaggingResult(report=report)


def _prepare_frame(
    daily: pl.DataFrame,
    horizon: int,
    *,
    benchmark_daily: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if "trade_date" in daily.columns:
        date_column = "trade_date"
    elif "date" in daily.columns:
        date_column = "date"
    else:
        raise ValueError("Index data requires date or trade_date")
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(
            f"Index data missing columns: {', '.join(sorted(missing))}"
        )

    frame = (
        daily.rename({date_column: "trade_date"})
        .with_columns(
            pl.col("trade_date")
            .cast(pl.Utf8)
            .str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            *[
                pl.col(name).cast(pl.Float64)
                for name in ("open", "high", "low", "close", "volume")
            ],
        )
        .drop_nulls(["trade_date", "close"])
        .sort("trade_date")
        .unique("trade_date", keep="last")
    )
    close = pl.col("close")
    return_1d = close / close.shift(1) - 1.0
    frame = frame.with_columns(
        return_1d.alias("return_1d"),
        (close / close.shift(5) - 1.0).alias("return_5d"),
        (close / close.shift(20) - 1.0).alias("return_20d"),
        (close.log() - close.log().shift(1)).alias("log_return_1d"),
        close.rolling_mean(5).alias("ma_5"),
        close.rolling_mean(20).alias("ma_20"),
        close.rolling_mean(60).alias("ma_60"),
        return_1d.rolling_std(20).alias("volatility_20"),
        (
            pl.col("volume")
            / pl.col("volume").shift(1).rolling_mean(5)
        ).alias("volume_ratio_5"),
        (
            pl.col("volume")
            / pl.col("volume").rolling_mean(20)
        ).alias("volume_to_ma_20"),
        ((pl.col("high") - pl.col("low")) / close).alias("high_low_range"),
        (
            (close - pl.col("low").rolling_min(20))
            / (
                pl.col("high").rolling_max(20)
                - pl.col("low").rolling_min(20)
            )
        ).alias("close_position_20"),
        (
            close / close.rolling_max(20) - 1.0
        ).alias("drawdown_20"),
        (
            close.shift(-horizon) / close - 1.0
        ).alias("_absolute_future_return"),
    )
    delta = close.diff()
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    average_gain = gain.ewm_mean(span=14, adjust=False)
    average_loss = loss.ewm_mean(span=14, adjust=False)
    rsi = 100.0 - 100.0 / (
        1.0 + average_gain / average_loss.clip(lower_bound=1e-12)
    )
    ema12 = close.ewm_mean(span=12, adjust=False)
    ema26 = close.ewm_mean(span=26, adjust=False)
    macd = ema12 - ema26
    macd_signal = macd.ewm_mean(span=9, adjust=False)
    frame = frame.with_columns(
        (rsi / 100.0).alias("rsi_14_scaled"),
        ((macd - macd_signal) / close).alias("macd_histogram_ratio"),
        pl.col("_absolute_future_return").alias("future_return"),
        (pl.col("_absolute_future_return") > 0).alias("future_up"),
        (close / pl.col("ma_5") - 1.0).alias("close_to_ma_5"),
        (close / pl.col("ma_20") - 1.0).alias("close_to_ma_20"),
        (close / pl.col("ma_60") - 1.0).alias("close_to_ma_60"),
        (pl.col("ma_5") / pl.col("ma_20") - 1.0).alias("ma_5_to_20"),
        (pl.col("ma_20") / pl.col("ma_60") - 1.0).alias("ma_20_to_60"),
    )
    if benchmark_daily is None:
        return frame.drop("_absolute_future_return")
    benchmark = benchmark_daily
    if "trade_date" in benchmark.columns:
        date_column = "trade_date"
    elif "date" in benchmark.columns:
        date_column = "date"
    else:
        raise ValueError("Benchmark data requires date or trade_date")
    benchmark = benchmark.rename({date_column: "trade_date"})
    benchmark_date_dtype = benchmark.schema["trade_date"]
    if benchmark_date_dtype == pl.Utf8:
        benchmark = benchmark.with_columns(
            pl.col("trade_date").str.strptime(
                pl.Date,
                "%Y-%m-%d",
                strict=False,
            )
        )
    elif isinstance(benchmark_date_dtype, pl.Datetime):
        benchmark = benchmark.with_columns(
            pl.col("trade_date").cast(pl.Date)
        )
    benchmark = (
        benchmark.select(
            "trade_date",
            pl.col("close").cast(pl.Float64).alias("_benchmark_close"),
        )
        .drop_nulls()
        .sort("trade_date")
        .unique("trade_date", keep="last")
        .with_columns(
            (
                pl.col("_benchmark_close").shift(-horizon)
                / pl.col("_benchmark_close")
                - 1.0
            ).alias("_benchmark_future_return")
        )
        .select("trade_date", "_benchmark_future_return")
    )
    frame = frame.join(benchmark, on="trade_date", how="left")
    excess = (
        pl.col("_absolute_future_return")
        - pl.col("_benchmark_future_return")
    )
    return frame.with_columns(
        excess.alias("future_return"),
        (excess > 0).alias("future_up"),
    ).drop("_absolute_future_return", "_benchmark_future_return")


def _fit_models(
    training: pl.DataFrame,
    seed: int,
    features: list[str],
) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingClassifier]:
    if training.height < 100:
        raise ValueError("Sampled training block is too small")
    x = training.select(features).to_numpy()
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
        early_stopping=False,
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
        early_stopping=False,
    )
    regressor.fit(x, training["future_return"].to_numpy())
    classifier.fit(x, training["future_up"].to_numpy())
    return regressor, classifier


def _select_index_features(
    training: pl.DataFrame,
    *,
    maximum_features: int,
    minimum_icir: float,
) -> list[str]:
    scored = []
    for feature in INDEX_FEATURES:
        data = training.select(
            pl.col(feature).alias("_feature"),
            pl.col("future_return").alias("_target"),
        ).drop_nulls()
        if data.height < 100:
            continue
        feature_values = data["_feature"].to_numpy()
        target_values = data["_target"].to_numpy()
        if np.std(feature_values) == 0 or np.std(target_values) == 0:
            continue
        overall = spearmanr(feature_values, target_values).statistic
        rolling = []
        window = 20
        for start in range(0, data.height - window + 1, window):
            value = spearmanr(
                feature_values[start:start + window],
                target_values[start:start + window],
            ).statistic
            if math.isfinite(float(value)):
                rolling.append(float(value))
        if not rolling:
            continue
        mean_ic = float(np.mean(rolling))
        std_ic = (
            float(np.std(rolling, ddof=1))
            if len(rolling) > 1
            else 0.0
        )
        icir = mean_ic / std_ic if std_ic > 0 else None
        scored.append(
            {
                "feature": feature,
                "overall_ic": float(overall) if math.isfinite(float(overall)) else None,
                "mean_ic": mean_ic,
                "icir": icir,
            }
        )
    qualified = [
        item
        for item in scored
        if item["mean_ic"] > 0
        and item["icir"] is not None
        and item["icir"] >= minimum_icir
    ]
    pool = qualified or [
        item for item in scored if item["mean_ic"] > 0
    ]
    pool.sort(
        key=lambda item: (
            item["icir"] if item["icir"] is not None else -math.inf,
            item["mean_ic"],
        ),
        reverse=True,
    )
    selected = [item["feature"] for item in pool[:maximum_features]]
    if selected:
        return selected
    scored.sort(key=lambda item: item["mean_ic"], reverse=True)
    return [item["feature"] for item in scored[:5]]


def _sample_time_blocks(
    frame: pl.DataFrame,
    fraction: float,
    seed: int,
    block_size: int,
) -> pl.DataFrame:
    count = frame.height
    indexes = list(range(0, count, max(2, block_size)))
    selected_count = max(1, math.ceil(len(indexes) * fraction))
    chosen = random.Random(seed).sample(indexes, selected_count)
    parts = [
        frame.slice(start, min(max(2, block_size), count - start))
        for start in sorted(chosen)
    ]
    return pl.concat(parts, how="vertical")


def _safe_auc(actual: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(actual)) < 2:
        return None
    return round(float(roc_auc_score(actual, probabilities)), 8)


def _trend_scores(frame: pl.DataFrame) -> np.ndarray:
    expression = (
        (pl.col("close_to_ma_5") > 0).cast(pl.Float64)
        + (pl.col("close_to_ma_20") > 0).cast(pl.Float64)
        + (pl.col("ma_5_to_20") > 0).cast(pl.Float64)
        + (pl.col("ma_20_to_60") > 0).cast(pl.Float64)
    ) / 4.0
    return frame.with_columns(expression.alias("_trend_score"))[
        "_trend_score"
    ].to_numpy()


def _trend_score_from_row(row: dict[str, Any]) -> float:
    values = (
        row.get("close_to_ma_5"),
        row.get("close_to_ma_20"),
        row.get("ma_5_to_20"),
        row.get("ma_20_to_60"),
    )
    return sum(
        value is not None and float(value) > 0
        for value in values
    ) / len(values)


def _evaluate_time_series_predictions(
    *,
    actual_returns: np.ndarray,
    actual_up: np.ndarray,
    predicted_returns: np.ndarray,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    trend_scores: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    correlation = spearmanr(actual_returns, predicted_returns).statistic
    window = 20
    rolling_ic = []
    for start in range(0, len(actual_returns) - window + 1, window):
        value = spearmanr(
            actual_returns[start:start + window],
            predicted_returns[start:start + window],
        ).statistic
        if math.isfinite(float(value)):
            rolling_ic.append(float(value))
    ic_mean = float(np.mean(rolling_ic)) if rolling_ic else None
    ic_std = (
        float(np.std(rolling_ic, ddof=1))
        if len(rolling_ic) > 1
        else None
    )
    icir = (
        ic_mean / ic_std
        if ic_mean is not None and ic_std is not None and ic_std > 0
        else None
    )

    period_indices = np.arange(0, len(actual_returns), max(1, horizon))
    period_returns = np.where(
        (predicted_returns[period_indices] > 0)
        & (calibrated_probabilities[period_indices] >= 0.55),
        actual_returns[period_indices],
        0.0,
    )
    equity = np.cumprod(1.0 + period_returns)
    total_return = float(equity[-1] - 1.0) if equity.size else None
    max_drawdown = _max_drawdown(equity)
    directions_match = (
        (predicted_returns[period_indices] > 0)
        == (actual_returns[period_indices] > 0)
    )
    trend_mask = (
        (trend_scores[period_indices] >= 0.5)
        & (predicted_returns[period_indices] > 0)
        & (calibrated_probabilities[period_indices] >= 0.55)
    )
    trend_period_returns = np.where(
        trend_mask,
        actual_returns[period_indices],
        0.0,
    )
    trend_equity = np.cumprod(1.0 + trend_period_returns)
    return {
        "rank_ic_mode": "time_series",
        "rank_ic_mean": _round_or_none(ic_mean),
        "rank_ic_std": _round_or_none(ic_std),
        "icir": _round_or_none(icir),
        "positive_rank_ic_rate": _round_or_none(
            sum(value > 0 for value in rolling_ic) / len(rolling_ic)
            if rolling_ic
            else None
        ),
        "signal_correlation": _round_or_none(correlation),
        "accuracy": _round_or_none(
            accuracy_score(actual_up, calibrated_probabilities >= 0.5)
        ),
        "roc_auc": _safe_auc(actual_up, calibrated_probabilities),
        "raw_brier_score": _round_or_none(
            brier_score_loss(actual_up, raw_probabilities)
        ),
        "calibrated_brier_score": _round_or_none(
            brier_score_loss(actual_up, calibrated_probabilities)
        ),
        "mean_calibration_shift": _round_or_none(
            np.mean(calibrated_probabilities - raw_probabilities)
        ),
        "rmse": _round_or_none(
            np.sqrt(np.mean((actual_returns - predicted_returns) ** 2))
        ),
        "mae": _round_or_none(
            mean_absolute_error(actual_returns, predicted_returns)
        ),
        "total_return": _round_or_none(total_return),
        "max_drawdown": _round_or_none(max_drawdown),
        "hit_rate": _round_or_none(
            float(directions_match.mean()) if directions_match.size else None
        ),
        "trend_eligible_ratio": _round_or_none(
            float(np.mean(trend_scores >= 0.5))
        ),
        "trend_filtered_total_return": _round_or_none(
            float(trend_equity[-1] - 1.0) if trend_equity.size else None
        ),
        "trend_filtered_max_drawdown": _round_or_none(
            _max_drawdown(trend_equity)
        ),
        "trend_filtered_periods": int(period_indices.size),
        "periods": int(period_indices.size),
        "sample_rows": int(len(actual_returns)),
    }


def _max_drawdown(equity: np.ndarray) -> float | None:
    if equity.size == 0:
        return None
    peak = 1.0
    worst = 0.0
    for value in np.concatenate(([1.0], equity)):
        peak = max(peak, float(value))
        worst = min(worst, float(value) / peak - 1.0)
    return worst


def _round_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 8)


def _validate_config(config: IndexBaggingConfig) -> None:
    if not config.seeds:
        raise ValueError("At least one seed is required")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("Seeds must be unique")
    if not 0 < config.sample_fraction <= 1:
        raise ValueError("sample_fraction must be between 0 and 1")
    if not 1 <= config.horizon_days <= 60:
        raise ValueError("horizon_days must be between 1 and 60")
