from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import optuna
import polars as pl
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

from app.services.backtest import BacktestConfig, StrategyBacktestService
from app.services.gbm import (
    FEATURE_COLUMNS,
    GBMFeatureBuilder,
    GBMService,
)
from app.symbols import normalize_stock_code


@dataclass(frozen=True)
class NestedTuningConfig:
    horizon: int = 10
    outer_folds: int = 4
    inner_splits: int = 3
    trials: int = 20
    embargo_days: int = 2
    min_test_rows_per_fold: int = 30
    seed: int = 42
    top_fraction: float = 0.1
    commission_each_side: float = 0.0003
    stamp_duty_sell: float = 0.0005
    slippage_each_side: float = 0.001
    neutralize: bool = False


@dataclass(frozen=True)
class NestedTuningResult:
    report: dict[str, Any]
    out_of_sample: pl.DataFrame


class NestedTuningService:
    def run(
        self,
        daily: pl.DataFrame,
        target_code: str,
        config: NestedTuningConfig | None = None,
        financial: pl.DataFrame | None = None,
    ) -> NestedTuningResult:
        cfg = config or NestedTuningConfig()
        _validate_config(cfg)
        target = normalize_stock_code(target_code)
        builder = GBMFeatureBuilder()
        if cfg.neutralize:
            dataset = builder.build_neutralized(
                daily,
                target_horizon=cfg.horizon,
                financial=financial,
            )
        else:
            dataset = builder.build(
                daily,
                target_horizon=cfg.horizon,
                financial=financial,
            )
        target_return_column = f"future_return_{cfg.horizon}d"
        target_up_column = f"future_up_{cfg.horizon}d"
        dates = dataset.training["trade_date"].unique().sort()
        outer_ranges = GBMService._fold_ranges(
            dates,
            folds=cfg.outer_folds,
            min_test_rows=cfg.min_test_rows_per_fold,
            target_horizon=cfg.horizon,
        )

        out_of_sample_parts = []
        fold_reports = []
        tuned_probability_parts = []
        baseline_probability_parts = []
        logistic_probability_parts = []
        actual_up_parts = []
        actual_return_parts = []
        actual_date_parts = []
        predicted_return_parts = []

        for fold_number, (
            outer_train_end,
            outer_test_start,
            outer_test_end,
        ) in enumerate(outer_ranges, start=1):
            outer_train = dataset.training.filter(
                pl.col("trade_date") <= outer_train_end
            )
            outer_test = dataset.training.filter(
                (pl.col("trade_date") >= outer_test_start)
                & (pl.col("trade_date") <= outer_test_end)
            )
            if outer_test.height < cfg.min_test_rows_per_fold:
                raise ValueError(
                    f"Outer fold {fold_number} has too few test rows"
                )

            outer_dates = outer_train["trade_date"].unique().sort()
            inner_folds = _inner_date_folds(
                outer_dates,
                n_splits=cfg.inner_splits,
                purge_days=cfg.horizon + 1,
                embargo_days=cfg.embargo_days,
            )
            if len(inner_folds) < 2:
                raise ValueError(
                    f"Outer fold {fold_number} has too few inner folds"
                )

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(
                    seed=cfg.seed + fold_number,
                    n_startup_trials=min(5, cfg.trials),
                    multivariate=True,
                    group=True,
                ),
                study_name=f"gbm-{target}-h{cfg.horizon}-outer-{fold_number}",
            )

            def objective(trial: optuna.Trial) -> float:
                params = _suggest_parameters(trial)
                inner_scores = []
                for train_end, validation_start, validation_end in inner_folds:
                    train = outer_train.filter(
                        pl.col("trade_date") <= train_end
                    )
                    validation = outer_train.filter(
                        (pl.col("trade_date") >= validation_start)
                        & (pl.col("trade_date") <= validation_end)
                    )
                    y_train = train[target_up_column].to_numpy()
                    y_validation = validation[target_up_column].to_numpy()
                    if len(np.unique(y_train)) < 2:
                        raise optuna.TrialPruned(
                            "Inner training fold has one class"
                        )
                    model = HistGradientBoostingClassifier(
                        **params,
                        random_state=cfg.seed + fold_number,
                        early_stopping=False,
                    )
                    model.fit(
                        train.select(FEATURE_COLUMNS).to_numpy(),
                        y_train,
                    )
                    probabilities = model.predict_proba(
                        validation.select(FEATURE_COLUMNS).to_numpy()
                    )[:, 1]
                    objective_metrics = _cross_sectional_metrics(
                        validation["trade_date"].to_list(),
                        validation[target_return_column].to_numpy(),
                        probabilities,
                    )
                    trial.set_user_attr(
                        f"fold_{len(inner_scores)}_rank_ic",
                        objective_metrics["mean_rank_ic"],
                    )
                    inner_scores.append(
                        _classification_objective(
                            y_validation,
                            probabilities,
                            objective_metrics,
                        )
                    )
                return float(np.mean(inner_scores))

            study.optimize(objective, n_trials=cfg.trials)
            best_params = study.best_params

            x_train = outer_train.select(FEATURE_COLUMNS).to_numpy()
            x_test = outer_test.select(FEATURE_COLUMNS).to_numpy()
            y_train_up = outer_train[target_up_column].to_numpy()
            y_test_up = outer_test[target_up_column].to_numpy()
            y_test_return = outer_test[target_return_column].to_numpy()

            tuned_model = HistGradientBoostingClassifier(
                **best_params,
                random_state=cfg.seed + fold_number,
                early_stopping=False,
            )
            tuned_model.fit(x_train, y_train_up)
            tuned_probabilities = tuned_model.predict_proba(x_test)[:, 1]

            baseline_model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                max_features=1.0,
                random_state=cfg.seed + fold_number,
                early_stopping=False,
            )
            baseline_model.fit(x_train, y_train_up)
            baseline_probabilities = baseline_model.predict_proba(x_test)[:, 1]

            logistic_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    random_state=cfg.seed + fold_number,
                ),
            )
            logistic_model.fit(x_train, y_train_up)
            logistic_probabilities = logistic_model.predict_proba(x_test)[:, 1]

            regressor = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=cfg.seed + fold_number,
                early_stopping=False,
            )
            regressor.fit(
                x_train,
                outer_train[target_return_column].to_numpy(),
            )
            predicted_returns = regressor.predict(x_test)

            fold_reports.append(
                {
                    "fold": fold_number,
                    "outer_train_end": str(outer_train_end),
                    "outer_test_start": str(outer_test_start),
                    "outer_test_end": str(outer_test_end),
                    "outer_train_rows": outer_train.height,
                    "outer_test_rows": outer_test.height,
                    "inner_folds": len(inner_folds),
                    "best_inner_score": _safe_float(study.best_value),
                    "best_params": best_params,
                    "tuned_outer": _classification_metrics(
                        y_test_up,
                        tuned_probabilities,
                    ),
                    "tuned_outer_cross_sectional": _cross_sectional_metrics(
                        outer_test["trade_date"].to_list(),
                        y_test_return,
                        tuned_probabilities,
                    ),
                    "baseline_outer": _classification_metrics(
                        y_test_up,
                        baseline_probabilities,
                    ),
                    "baseline_outer_cross_sectional": (
                        _cross_sectional_metrics(
                            outer_test["trade_date"].to_list(),
                            y_test_return,
                            baseline_probabilities,
                        )
                    ),
                    "logistic_outer": _classification_metrics(
                        y_test_up,
                        logistic_probabilities,
                    ),
                    "logistic_outer_cross_sectional": (
                        _cross_sectional_metrics(
                            outer_test["trade_date"].to_list(),
                            y_test_return,
                            logistic_probabilities,
                        )
                    ),
                }
            )
            out_of_sample_parts.append(
                outer_test.select(
                    "stock_code",
                    "trade_date",
                ).with_columns(
                    pl.Series("future_return", y_test_return),
                    pl.Series("future_up", y_test_up),
                    pl.Series("predicted_return", predicted_returns),
                    pl.Series("up_probability", tuned_probabilities),
                    pl.Series("baseline_probability", baseline_probabilities),
                    pl.Series("logistic_probability", logistic_probabilities),
                    pl.lit(fold_number).alias("fold"),
                    pl.lit(cfg.horizon).alias("horizon_days"),
                )
            )
            actual_up_parts.append(y_test_up)
            actual_return_parts.append(y_test_return)
            actual_date_parts.extend(outer_test["trade_date"].to_list())
            tuned_probability_parts.append(tuned_probabilities)
            baseline_probability_parts.append(baseline_probabilities)
            logistic_probability_parts.append(logistic_probabilities)
            predicted_return_parts.append(predicted_returns)

        out_of_sample = pl.concat(
            out_of_sample_parts,
            how="diagonal_relaxed",
        ).sort(["trade_date", "stock_code"])
        backtest = StrategyBacktestService().run(
            out_of_sample,
            BacktestConfig(
                top_fraction=cfg.top_fraction,
                holding_days=cfg.horizon,
                rebalance_days=cfg.horizon,
                commission_each_side=cfg.commission_each_side,
                stamp_duty_sell=cfg.stamp_duty_sell,
                slippage_each_side=cfg.slippage_each_side,
                seed=cfg.seed,
            ),
        ).report

        actual_up = np.concatenate(actual_up_parts)
        actual_returns = np.concatenate(actual_return_parts)
        tuned_probabilities = np.concatenate(tuned_probability_parts)
        baseline_probabilities = np.concatenate(baseline_probability_parts)
        logistic_probabilities = np.concatenate(logistic_probability_parts)
        report = {
            "metadata": {
                "stock_code": target,
                "model": (
                    "HistGradientBoosting_v3_neutralized"
                    if cfg.neutralize
                    else "HistGradientBoosting"
                ),
                "validation": "nested_walk_forward",
                "early_stopping": False,
                "generated_on": date.today().isoformat(),
                "seed": cfg.seed,
                "target_horizon_days": cfg.horizon,
                "outer_folds": cfg.outer_folds,
                "inner_splits": cfg.inner_splits,
                "tpe_trials_per_outer_fold": cfg.trials,
                "purged_trading_days": cfg.horizon + 1,
                "embargo_days": cfg.embargo_days,
                "neutralize": cfg.neutralize,
            },
            "dataset": {
                "stock_count": dataset.stock_count,
                "training_rows": dataset.training.height,
                "trading_days": dataset.trading_day_count,
                "out_of_sample_rows": out_of_sample.height,
            },
            "aggregate_outer": {
                "tuned": _classification_metrics(
                    actual_up,
                    tuned_probabilities,
                ),
                "tuned_cross_sectional": _cross_sectional_metrics(
                    actual_date_parts,
                    actual_returns,
                    tuned_probabilities,
                ),
                "baseline": _classification_metrics(
                    actual_up,
                    baseline_probabilities,
                ),
                "baseline_cross_sectional": _cross_sectional_metrics(
                    actual_date_parts,
                    actual_returns,
                    baseline_probabilities,
                ),
                "logistic": _classification_metrics(
                    actual_up,
                    logistic_probabilities,
                ),
                "logistic_cross_sectional": _cross_sectional_metrics(
                    actual_date_parts,
                    actual_returns,
                    logistic_probabilities,
                ),
            },
            "folds": fold_reports,
            "backtest": backtest,
        }
        return NestedTuningResult(
            report=report,
            out_of_sample=out_of_sample,
        )


def _suggest_parameters(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.10,
            log=True,
        ),
        "max_iter": trial.suggest_int(
            "max_iter",
            100,
            500,
            step=50,
        ),
        "max_leaf_nodes": trial.suggest_int(
            "max_leaf_nodes",
            7,
            31,
        ),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf",
            20,
            200,
            step=10,
        ),
        "l2_regularization": trial.suggest_float(
            "l2_regularization",
            0.1,
            20.0,
            log=True,
        ),
        "max_features": trial.suggest_float(
            "max_features",
            0.6,
            1.0,
        ),
    }


def _inner_date_folds(
    dates: pl.Series,
    *,
    n_splits: int,
    purge_days: int,
    embargo_days: int,
) -> list[tuple[date, date, date]]:
    values = dates.to_list()
    splitter = TimeSeriesSplit(n_splits=n_splits)
    folds = []
    for _, validation_indices in splitter.split(values):
        if validation_indices.size == 0:
            continue
        validation_start_index = int(validation_indices[0])
        training_end_index = (
            validation_start_index - purge_days - embargo_days
        )
        if training_end_index < 60:
            continue
        folds.append(
            (
                values[training_end_index],
                values[int(validation_indices[0])],
                values[int(validation_indices[-1])],
            )
        )
    return folds


def _classification_objective(
    actual: np.ndarray,
    probabilities: np.ndarray,
    cross_sectional: dict[str, Any],
) -> float:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    loss = log_loss(actual, clipped, labels=[0, 1])
    brier = brier_score_loss(actual, clipped)
    auc = (
        roc_auc_score(actual, clipped)
        if len(np.unique(actual)) == 2
        else 0.5
    )
    mean_ic = cross_sectional["mean_rank_ic"] or 0.0
    ic_volatility = cross_sectional["rank_ic_std"] or 0.0
    top_excess = cross_sectional["mean_top_excess_return"] or 0.0
    return float(
        0.1 * loss
        + 0.2 * brier
        - mean_ic
        + 0.5 * ic_volatility
        - 0.1 * top_excess
        - 0.01 * auc
    )


def _cross_sectional_metrics(
    dates: list[date],
    actual_returns: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    unique_dates = sorted(set(dates))
    rank_ics = []
    top_excess_returns = []
    for trade_date in unique_dates:
        mask = np.asarray(
            [value == trade_date for value in dates],
            dtype=bool,
        )
        date_returns = actual_returns[mask]
        date_probabilities = probabilities[mask]
        if date_returns.size < 5:
            continue
        if (
            np.std(date_returns) == 0
            or np.std(date_probabilities) == 0
        ):
            continue
        rank_ic = float(
            np.corrcoef(
                rankdata(date_probabilities),
                rankdata(date_returns),
            )[0, 1]
        )
        if math.isfinite(rank_ic):
            rank_ics.append(rank_ic)
        top_count = max(1, math.ceil(date_returns.size * 0.2))
        top_indices = np.argsort(date_probabilities)[-top_count:]
        top_excess_returns.append(
            float(
                np.mean(date_returns[top_indices])
                - np.mean(date_returns)
            )
        )
    return {
        "dates": len(rank_ics),
        "mean_rank_ic": _safe_float(np.mean(rank_ics)) if rank_ics else None,
        "rank_ic_std": _safe_float(np.std(rank_ics)) if rank_ics else None,
        "positive_rank_ic_rate": (
            _safe_float(np.mean([value > 0 for value in rank_ics]))
            if rank_ics
            else None
        ),
        "mean_top_excess_return": (
            _safe_float(np.mean(top_excess_returns))
            if top_excess_returns
            else None
        ),
    }


def _classification_metrics(
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": _safe_float(accuracy_score(actual, predicted)),
        "roc_auc": (
            _safe_float(roc_auc_score(actual, probabilities))
            if len(np.unique(actual)) == 2
            else None
        ),
        "log_loss": _safe_float(
            log_loss(
                actual,
                np.clip(probabilities, 1e-6, 1 - 1e-6),
                labels=[0, 1],
            )
        ),
        "brier_score": _safe_float(
            brier_score_loss(actual, probabilities)
        ),
        "rows": int(actual.size),
    }


def _validate_config(config: NestedTuningConfig) -> None:
    if not 1 <= config.horizon <= 60:
        raise ValueError("horizon must be between 1 and 60")
    if config.outer_folds < 2:
        raise ValueError("At least two outer folds are required")
    if config.inner_splits < 2:
        raise ValueError("At least two inner splits are required")
    if config.trials < 1:
        raise ValueError("At least one TPE trial is required")
    if config.embargo_days < 0:
        raise ValueError("embargo_days cannot be negative")
    if not 0 < config.top_fraction <= 1:
        raise ValueError("top_fraction must be between 0 and 1")


def _safe_float(value: Any) -> float | None:
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, 8)
