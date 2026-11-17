from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.industries import (
    INDUSTRY_TO_SECTOR,
    MIN_SECTOR_SIZE,
    load_industry_map,
)
from app.indicators import compute_indicators
from app.symbols import normalize_model_code


DEFAULT_TARGET_HORIZON = 5
FEATURE_COLUMNS = [
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
]

# Fundamental features derived from quarterly financial reports (as-of joined).
FUNDAMENTAL_FEATURES = [
    "fund_roe",
    "fund_net_margin",
    "fund_net_profit_yoy",
    "fund_eps_yoy",
    "fund_debt_ratio",
    "fund_equity_yoy",
    "fund_asset_turnover",
    "fund_cfo_to_revenue",
]

FUNDAMENTAL_SOURCE_MAP = {
    "fund_roe": "roe",
    "fund_net_margin": "net_margin",
    "fund_net_profit_yoy": "net_profit_yoy",
    "fund_eps_yoy": "eps_yoy",
    "fund_debt_ratio": "debt_ratio",
    "fund_equity_yoy": "equity_yoy",
    "fund_asset_turnover": "asset_turnover",
    "fund_cfo_to_revenue": "cfo_to_revenue",
}

FEATURE_COLUMNS = FEATURE_COLUMNS + FUNDAMENTAL_FEATURES
# Stock-specific features that vary across stocks on a given trade date.
# Market/industry aggregate features are excluded (rank-normalizing a constant
# produces a meaningless constant).
NEUTRALIZABLE_FEATURES = [
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
    "relative_strength_20",
    "excess_return_vs_industry_5d",
    "industry_relative_rank_5d",
]


class GBMDataError(ValueError):
    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


@dataclass(frozen=True)
class GBMFeatureDataset:
    training: pl.DataFrame
    latest: pl.DataFrame
    all_features: pl.DataFrame
    stock_count: int
    trading_day_count: int


@dataclass(frozen=True)
class GBMResult:
    report: dict[str, Any]
    features: pl.DataFrame
    out_of_sample: pl.DataFrame


class GBMFeatureBuilder:
    def build(
        self,
        daily: pl.DataFrame,
        *,
        target_horizon: int = DEFAULT_TARGET_HORIZON,
        industry_map: dict[str, str] | None = None,
        financial: pl.DataFrame | None = None,
    ) -> GBMFeatureDataset:
        return self._build_impl(
            daily,
            target_horizon=target_horizon,
            industry_map=industry_map,
            financial=financial,
            neutralize=False,
        )

    def build_neutralized(
        self,
        daily: pl.DataFrame,
        *,
        target_horizon: int = DEFAULT_TARGET_HORIZON,
        industry_map: dict[str, str] | None = None,
        financial: pl.DataFrame | None = None,
    ) -> GBMFeatureDataset:
        return self._build_impl(
            daily,
            target_horizon=target_horizon,
            industry_map=industry_map,
            financial=financial,
            neutralize=True,
        )

    def _build_impl(
        self,
        daily: pl.DataFrame,
        *,
        target_horizon: int = DEFAULT_TARGET_HORIZON,
        industry_map: dict[str, str] | None = None,
        financial: pl.DataFrame | None = None,
        neutralize: bool = False,
    ) -> GBMFeatureDataset:
        if not 1 <= target_horizon <= 60:
            raise ValueError("target_horizon must be between 1 and 60")
        target_return_column = f"future_return_{target_horizon}d"
        target_up_column = f"future_up_{target_horizon}d"
        required = {
            "stock_code",
            "trade_date",
            "adjust_type",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "volume",
            "turnover",
            "source",
            "ingested_at",
        }
        missing = required.difference(daily.columns)
        if missing:
            raise ValueError(
                f"Missing GBM daily columns: {', '.join(sorted(missing))}"
            )

        adjusted = daily.filter(pl.col("adjust_type") == "qfq")
        if adjusted.is_empty():
            raise ValueError("GBM requires forward-adjusted qfq daily data")

        adjusted = self._join_financial_features(adjusted, financial)

        indicator_frames = []
        for stock_code in adjusted["stock_code"].unique().sort().to_list():
            stock_frame = adjusted.filter(
                pl.col("stock_code") == stock_code
            ).sort("trade_date")
            if stock_frame.height < 80:
                continue
            indicator_frames.append(compute_indicators(stock_frame))
        if not indicator_frames:
            raise ValueError("No stock has at least 80 qfq trading days")

        combined = pl.concat(
            indicator_frames,
            how="diagonal_relaxed",
        ).sort(["stock_code", "trade_date"])
        mapping = industry_map or load_industry_map()
        industry_frame = pl.DataFrame(
            {
                "stock_code": list(mapping),
                "industry": list(mapping.values()),
            }
        )
        if industry_frame.is_empty():
            combined = combined.with_columns(
                pl.lit("Unknown").alias("industry")
            )
        else:
            combined = combined.join(
                industry_frame,
                on="stock_code",
                how="left",
            ).with_columns(
                pl.col("industry").fill_null("Unknown")
            )
        engineered = combined.with_columns(
            (pl.col("close") / pl.col("ma_5") - 1.0).alias("close_to_ma_5"),
            (pl.col("close") / pl.col("ma_20") - 1.0).alias("close_to_ma_20"),
            (pl.col("ma_5") / pl.col("ma_20") - 1.0).alias("ma_5_to_20"),
            (pl.col("ma_20") / pl.col("ma_60") - 1.0).alias("ma_20_to_60"),
            (
                pl.col("macd_histogram")
                / pl.when(pl.col("close").abs() > 0)
                .then(pl.col("close").abs())
                .otherwise(None)
            ).alias("macd_histogram_ratio"),
            (pl.col("rsi_14") / 100.0).alias("rsi_14_scaled"),
            (pl.col("kdj_k") / 100.0).alias("kdj_k_scaled"),
            (pl.col("kdj_d") / 100.0).alias("kdj_d_scaled"),
            (
                (pl.col("close") - pl.col("boll_lower_20"))
                / pl.when(pl.col("boll_upper_20") > pl.col("boll_lower_20"))
                .then(pl.col("boll_upper_20") - pl.col("boll_lower_20"))
                .otherwise(None)
            )
            .fill_null(0.5)
            .alias("boll_position"),
            (
                pl.col("atr_14")
                / pl.when(pl.col("close").abs() > 0)
                .then(pl.col("close").abs())
                .otherwise(None)
            ).alias("atr_14_ratio"),
            pl.col("volume_ratio_5").fill_null(1.0),
            (
                pl.col("volume")
                / pl.when(pl.col("volume_ma_20") > 0)
                .then(pl.col("volume_ma_20"))
                .otherwise(None)
            )
            .fill_null(1.0)
            .alias("volume_to_ma_20"),
            pl.col("turnover").fill_null(0.0),
            (
                pl.col("close").shift(-target_horizon).over("stock_code")
                / pl.col("close")
                - 1.0
            ).alias(target_return_column),
            (
                pl.col("close")
                / pl.col("close").shift(5).over("stock_code")
                - 1.0
            ).alias("return_5d"),
            (
                pl.col("close")
                / pl.col("close").shift(20).over("stock_code")
                - 1.0
            ).alias("return_20d"),
        ).with_columns(
            (pl.col(target_return_column) > 0)
            .cast(pl.Int8)
            .alias(target_up_column),
            pl.col("return_5d").mean().over("trade_date").alias(
                "market_return_5d"
            ),
            pl.col("volatility_20").mean().over("trade_date").alias(
                "market_volatility_20"
            ),
            pl.col("return_20d").mean().over("trade_date").alias(
                "_market_return_20d"
            ),
        ).with_columns(
            (
                pl.col("return_5d") - pl.col("market_return_5d")
            ).alias("excess_return_5d"),
            (
                pl.col("return_20d") - pl.col("_market_return_20d")
            ).alias("relative_strength_20"),
        ).drop(
            "_market_return_20d"
        ).with_columns(
            pl.col("return_1d").mean().over("trade_date").alias(
                "market_return_1d"
            ),
            (
                (pl.col("return_1d") > 0)
                .cast(pl.Float64)
                .mean()
                .over("trade_date")
            ).alias("market_positive_ratio_1d"),
            (
                (pl.col("close") > pl.col("ma_20"))
                .cast(pl.Float64)
                .mean()
                .over("trade_date")
            ).alias("market_above_ma20_ratio"),
            pl.col("return_1d")
            .std()
            .over("trade_date")
            .fill_null(0.0)
            .alias("market_dispersion_1d"),
            pl.col("return_5d")
            .mean()
            .over(["trade_date", "industry"])
            .alias("industry_return_5d"),
            pl.col("return_20d")
            .mean()
            .over(["trade_date", "industry"])
            .alias("industry_momentum_20d"),
            (
                (pl.col("close") > pl.col("ma_20"))
                .cast(pl.Float64)
                .mean()
                .over(["trade_date", "industry"])
            ).alias("industry_above_ma20_ratio"),
        ).with_columns(
            (
                pl.col("return_5d") - pl.col("industry_return_5d")
            ).alias("excess_return_vs_industry_5d"),
            (
                pl.col("return_5d")
                .rank("average")
                .over(["trade_date", "industry"])
                / pl.len().over(["trade_date", "industry"]).cast(pl.Float64)
            ).alias("industry_relative_rank_5d"),
        )

        finite_features = pl.all_horizontal(
            [
                pl.col(name).is_not_null() & pl.col(name).is_finite()
                for name in FEATURE_COLUMNS
            ]
        )
        valid = engineered.filter(finite_features).sort(
            ["stock_code", "trade_date"]
        )
        if neutralize:
            sector_mapping = (
                industry_map
                if industry_map is not None
                else load_industry_map()
            )
            valid = _apply_cross_sectional_neutralization(
                valid,
                sector_map={
                    code: INDUSTRY_TO_SECTOR.get(industry, "other")
                    for code, industry in sector_mapping.items()
                },
            )
        training = valid.filter(
            pl.col(target_return_column).is_not_null()
            & pl.col(target_return_column).is_finite()
        )
        latest = valid.group_by("stock_code", maintain_order=True).last()
        if training.is_empty():
            raise ValueError("No valid GBM training rows after feature generation")
        return GBMFeatureDataset(
            training=training,
            latest=latest,
            all_features=valid,
            stock_count=training["stock_code"].n_unique(),
            trading_day_count=training["trade_date"].n_unique(),
        )

    @staticmethod
    def _join_financial_features(
        daily: pl.DataFrame,
        financial: pl.DataFrame | None,
    ) -> pl.DataFrame:
        """As-of join quarterly fundamentals onto daily bars.

        Each daily row receives the most recently announced report so that
        no future information leaks into the feature set.
        """
        if financial is None or financial.is_empty():
            return daily.with_columns(
                [pl.lit(0.0).alias(name) for name in FUNDAMENTAL_FEATURES]
            )

        source_columns = list(FUNDAMENTAL_SOURCE_MAP.values())
        available = [c for c in source_columns if c in financial.columns]
        if not available:
            return daily.with_columns(
                [pl.lit(0.0).alias(name) for name in FUNDAMENTAL_FEATURES]
            )

        selected = financial.select(
            "stock_code",
            "report_date",
            *(
                [pl.col("announced_date")]
                if "announced_date" in financial.columns
                else []
            ),
            *(
                (
                    pl.col(FUNDAMENTAL_SOURCE_MAP[name])
                    if FUNDAMENTAL_SOURCE_MAP[name] in available
                    else pl.lit(None, dtype=pl.Float64)
                ).alias(name)
                for name in FUNDAMENTAL_FEATURES
            ),
        )
        if "announced_date" in financial.columns:
            selected = selected.with_columns(
                pl.col("announced_date")
                .fill_null(pl.col("report_date") + pl.duration(days=45))
                .alias("_effective_date")
            )
        else:
            selected = selected.with_columns(
                (pl.col("report_date") + pl.duration(days=45)).alias(
                    "_effective_date"
                )
            )
        selected = selected.filter(
            pl.col("_effective_date").is_not_null()
        ).sort(["stock_code", "_effective_date"])
        daily_sorted = daily.sort(["stock_code", "trade_date"])
        joined = daily_sorted.join_asof(
            selected,
            left_on="trade_date",
            right_on="_effective_date",
            by="stock_code",
            strategy="backward",
        )
        return joined.with_columns(
            [pl.col(name).fill_null(0.0) for name in FUNDAMENTAL_FEATURES]
        )


class GBMService:
    def analyze(
        self,
        daily: pl.DataFrame,
        target_code: str,
        *,
        seed: int = 42,
        min_stocks: int = 3,
        min_training_rows: int = 1000,
        min_trading_days: int = 120,
        horizon: int = DEFAULT_TARGET_HORIZON,
        walk_forward_folds: int = 4,
        min_test_rows_per_fold: int = 30,
        financial: pl.DataFrame | None = None,
        benchmark_daily: pl.DataFrame | None = None,
        training_stock_codes: list[str] | None = None,
        training_dates: list[date] | None = None,
        feature_selection: bool = False,
        candidate_features: list[str] | None = None,
        max_selected_features: int = 20,
        minimum_feature_icir: float = 0.05,
    ) -> GBMResult:
        target = normalize_model_code(target_code)
        dataset = GBMFeatureBuilder().build(
            daily,
            target_horizon=horizon,
            financial=financial,
        )
        training = dataset.training
        if training_stock_codes is not None:
            normalized_training_codes = {
                normalize_model_code(code)
                for code in training_stock_codes
            }
            training = training.filter(
                pl.col("stock_code").is_in(normalized_training_codes)
            )
        if training_dates is not None:
            training = training.filter(
                pl.col("trade_date").is_in(training_dates)
            )
        target_return_column = f"future_return_{horizon}d"
        target_up_column = f"future_up_{horizon}d"
        candidate_pool = (
            list(candidate_features)
            if candidate_features is not None
            else list(FEATURE_COLUMNS)
        )
        unknown_features = set(candidate_pool).difference(FEATURE_COLUMNS)
        if unknown_features:
            raise ValueError(
                "Unknown candidate features: "
                + ", ".join(sorted(unknown_features))
            )
        if not candidate_pool:
            raise ValueError("candidate_features must not be empty")
        if benchmark_daily is not None:
            training = _apply_excess_target(
                training,
                benchmark_daily,
                horizon=horizon,
                target_return_column=target_return_column,
                target_up_column=target_up_column,
            )
        gates = {
            "stock_count": dataset.stock_count,
            "training_stock_count": training["stock_code"].n_unique(),
            "minimum_stocks": min_stocks,
            "training_rows": training.height,
            "minimum_training_rows": min_training_rows,
            "trading_days": dataset.trading_day_count,
            "minimum_trading_days": min_trading_days,
            "target_horizon": horizon,
        }
        if dataset.stock_count < min_stocks:
            raise GBMDataError(
                f"GBM requires at least {min_stocks} stocks; "
                f"only {dataset.stock_count} available",
                gates,
            )
        if training.height < min_training_rows:
            raise GBMDataError(
                f"GBM requires at least {min_training_rows} training rows; "
                f"only {training.height} available",
                gates,
            )
        if dataset.trading_day_count < min_trading_days:
            raise GBMDataError(
                f"GBM requires at least {min_trading_days} trading days; "
                f"only {dataset.trading_day_count} available",
                gates,
            )
        if target not in set(dataset.latest["stock_code"].to_list()):
            raise ValueError(f"No current GBM features available for {target}")

        dates = training["trade_date"].unique().sort()
        try:
            fold_ranges = self._fold_ranges(
                dates,
                folds=walk_forward_folds,
                min_test_rows=min_test_rows_per_fold,
                target_horizon=horizon,
            )
        except ValueError as exc:
            raise GBMDataError(
                f"Not enough data for {walk_forward_folds} walk-forward folds: {exc}",
                gates,
            ) from exc

        y_test_parts: list[np.ndarray] = []
        gbm_return_parts: list[np.ndarray] = []
        y_up_parts: list[np.ndarray] = []
        gbm_probability_parts: list[np.ndarray] = []
        logistic_probability_parts: list[np.ndarray] = []
        out_of_sample_parts = []
        fold_reports = []

        for fold_number, (
            train_end,
            test_start,
            test_end,
        ) in enumerate(fold_ranges, start=1):
            train = training.filter(pl.col("trade_date") <= train_end)
            test = training.filter(
                (pl.col("trade_date") >= test_start)
                & (pl.col("trade_date") <= test_end)
            )
            if train.height < min_training_rows:
                raise GBMDataError(
                    f"Walk-forward fold {fold_number} has only "
                    f"{train.height} training rows",
                    {**gates, "fold": fold_number, "train_rows": train.height},
                )
            if test.height < min_test_rows_per_fold:
                raise GBMDataError(
                    f"Walk-forward fold {fold_number} has only "
                    f"{test.height} test rows",
                    {**gates, "fold": fold_number, "test_rows": test.height},
                )

            fold_features = (
                _select_point_in_time_features(
                    train,
                    target_return_column,
                    candidate_pool,
                    maximum_features=max_selected_features,
                    minimum_icir=minimum_feature_icir,
                )
                if feature_selection
                else candidate_pool
            )
            x_train = train.select(fold_features).to_numpy()
            y_train = train[target_return_column].to_numpy()
            y_train_up = train[target_up_column].to_numpy()
            x_test = test.select(fold_features).to_numpy()
            y_test = test[target_return_column].to_numpy()
            y_test_up = test[target_up_column].to_numpy()
            if len(np.unique(y_train_up)) < 2:
                raise GBMDataError(
                    f"Walk-forward fold {fold_number} has one training class",
                    {**gates, "fold": fold_number},
                )

            regressor = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=seed + fold_number,
                early_stopping=False,
            )
            regressor.fit(x_train, y_train)
            return_predictions = regressor.predict(x_test)

            classifier = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                random_state=seed + fold_number,
                early_stopping=False,
            )
            classifier.fit(x_train, y_train_up)
            up_probabilities = classifier.predict_proba(x_test)[:, 1]

            logistic = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    random_state=seed + fold_number,
                ),
            )
            logistic.fit(x_train, y_train_up)
            logistic_probabilities = logistic.predict_proba(x_test)[:, 1]

            fold_reports.append(
                {
                    "fold": fold_number,
                    "train_start": str(train["trade_date"].min()),
                    "train_end": str(train_end),
                    "test_start": str(test_start),
                    "test_end": str(test_end),
                    "train_rows": int(train.height),
                    "test_rows": int(test.height),
                    "selected_features": fold_features,
                    "regression": _regression_metrics(
                        y_test,
                        return_predictions,
                    ),
                    "classification": _classification_metrics(
                        y_test_up,
                        (up_probabilities >= 0.5).astype(int),
                        up_probabilities,
                    ),
                    "logistic_baseline": _classification_metrics(
                        y_test_up,
                        (logistic_probabilities >= 0.5).astype(int),
                        logistic_probabilities,
                    ),
                }
            )
            y_test_parts.append(y_test)
            gbm_return_parts.append(return_predictions)
            y_up_parts.append(y_test_up)
            gbm_probability_parts.append(up_probabilities)
            logistic_probability_parts.append(logistic_probabilities)
            out_of_sample_parts.append(
                test.select(
                    "stock_code",
                    "trade_date",
                    target_return_column,
                    target_up_column,
                    "close_to_ma_5",
                    "close_to_ma_20",
                    "ma_5_to_20",
                    "ma_20_to_60",
                    "return_20d",
                ).with_columns(
                    pl.col(target_return_column).alias("future_return"),
                    pl.col(target_up_column).alias("future_up"),
                    pl.Series(
                        "predicted_return",
                        return_predictions,
                    ),
                    pl.Series(
                        "up_probability",
                        up_probabilities,
                    ),
                    pl.Series(
                        "logistic_probability",
                        logistic_probabilities,
                    ),
                    pl.lit(fold_number).alias("fold"),
                    pl.lit(horizon).alias("horizon_days"),
                )
            )

        y_test_all = np.concatenate(y_test_parts)
        return_predictions_all = np.concatenate(gbm_return_parts)
        y_up_all = np.concatenate(y_up_parts)
        up_probabilities_all = np.concatenate(gbm_probability_parts)
        logistic_probabilities_all = np.concatenate(
            logistic_probability_parts
        )
        regression_metrics = _regression_metrics(
            y_test_all,
            return_predictions_all,
        )
        classification_metrics = _classification_metrics(
            y_up_all,
            (up_probabilities_all >= 0.5).astype(int),
            up_probabilities_all,
        )
        logistic_metrics = _classification_metrics(
            y_up_all,
            (logistic_probabilities_all >= 0.5).astype(int),
            logistic_probabilities_all,
        )
        better_than_zero = (
            regression_metrics["rmse"] is not None
            and regression_metrics["baseline_zero_rmse"] is not None
            and regression_metrics["rmse"]
            < regression_metrics["baseline_zero_rmse"]
        )
        better_than_majority = (
            classification_metrics["accuracy"] is not None
            and classification_metrics["majority_baseline_accuracy"] is not None
            and classification_metrics["accuracy"]
            > classification_metrics["majority_baseline_accuracy"]
        )
        auc_above_chance = (
            classification_metrics["roc_auc"] is not None
            and classification_metrics["roc_auc"] > 0.5
        )
        better_than_logistic = (
            classification_metrics["accuracy"] is not None
            and logistic_metrics["accuracy"] is not None
            and classification_metrics["accuracy"] > logistic_metrics["accuracy"]
        )
        improved_regression_fold_count = sum(
            1
            for fold in fold_reports
            if (
                fold["regression"]["rmse"]
                is not None
                and fold["regression"]["baseline_zero_rmse"] is not None
                and fold["regression"]["rmse"]
                < fold["regression"]["baseline_zero_rmse"]
            )
        )
        improved_classification_fold_count = sum(
            1
            for fold in fold_reports
            if (
                fold["classification"]["accuracy"] is not None
                and fold["classification"]["majority_baseline_accuracy"]
                is not None
                and fold["classification"]["accuracy"]
                > fold["classification"]["majority_baseline_accuracy"]
            )
        )
        stable_fold_count = math.ceil(len(fold_reports) * 0.6)
        regression_folds_stable = (
            improved_regression_fold_count >= stable_fold_count
        )
        classification_folds_stable = (
            improved_classification_fold_count >= stable_fold_count
        )
        regression_status = (
            "acceptable"
            if better_than_zero and regression_folds_stable
            else "not_reliable"
        )
        classification_status = (
            "weak_signal"
            if (
                better_than_majority
                and auc_above_chance
                and better_than_logistic
                and classification_folds_stable
            )
            else "not_reliable"
        )
        reliability_reasons = []
        if not better_than_zero:
            reliability_reasons.append("RMSE did not beat the zero-return baseline")
        if not better_than_majority:
            reliability_reasons.append("Accuracy did not beat the majority baseline")
        if not auc_above_chance:
            reliability_reasons.append("ROC AUC did not exceed 0.5")
        if not better_than_logistic:
            reliability_reasons.append(
                "Accuracy did not beat the logistic-regression baseline"
            )
        if not regression_folds_stable:
            reliability_reasons.append(
                "GBM did not beat the zero-return baseline in enough folds"
            )
        if not classification_folds_stable:
            reliability_reasons.append(
                "GBM did not beat the majority direction baseline in enough folds"
            )

        final_features = (
            _select_point_in_time_features(
                training,
                target_return_column,
                candidate_pool,
                maximum_features=max_selected_features,
                minimum_icir=minimum_feature_icir,
            )
            if feature_selection
            else candidate_pool
        )
        final_x = training.select(final_features).to_numpy()
        final_regressor = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
            early_stopping=False,
        )
        final_regressor.fit(
            final_x,
            training[target_return_column].to_numpy(),
        )
        final_classifier = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
            early_stopping=False,
        )
        final_up_values = training[target_up_column].to_numpy()
        final_classifier.fit(
            final_x,
            final_up_values,
        )

        latest_row = dataset.latest.filter(
            pl.col("stock_code") == target
        ).row(0, named=True)
        latest_frame = pl.DataFrame(
            [{name: latest_row[name] for name in final_features}]
        )
        latest_x = latest_frame.select(final_features).to_numpy()
        expected_return = float(final_regressor.predict(latest_x)[0])
        up_probability = float(final_classifier.predict_proba(latest_x)[0, 1])
        latest_feature_frame = dataset.latest.select(final_features)
        latest_expected_returns = final_regressor.predict(
            latest_feature_frame.to_numpy()
        )
        latest_up_probabilities = final_classifier.predict_proba(
            latest_feature_frame.to_numpy()
        )[:, 1]
        latest_predictions = [
            {
                "stock_code": row["stock_code"],
                "trade_date": str(row["trade_date"]),
                "expected_return": _safe_float(expected),
                "up_probability": _safe_float(probability),
                "direction": "up" if probability >= 0.5 else "down",
                "close_to_ma_5": _safe_float(row.get("close_to_ma_5")),
                "close_to_ma_20": _safe_float(row.get("close_to_ma_20")),
                "ma_5_to_20": _safe_float(row.get("ma_5_to_20")),
                "ma_20_to_60": _safe_float(row.get("ma_20_to_60")),
            }
            for row, expected, probability in zip(
                dataset.latest.to_dicts(),
                latest_expected_returns,
                latest_up_probabilities,
                strict=True,
            )
        ]
        report = {
            "metadata": {
                "stock_code": target,
                "model": "HistGradientBoosting",
                "target": f"future_{horizon}d_return",
                "seed": seed,
                "generated_on": date.today().isoformat(),
                "feature_count": len(final_features),
                "candidate_feature_count": len(candidate_pool),
                "fundamental_features_enabled": financial is not None and not financial.is_empty(),
                "financial_report_rows": 0 if financial is None else financial.height,
            },
            "dataset": {
                "stock_count": dataset.stock_count,
                "training_rows": training.height,
                "trading_days": dataset.trading_day_count,
                "train_start": str(training["trade_date"].min()),
                "train_end": str(training["trade_date"].max()),
                "target_horizon_days": horizon,
                "purged_trading_days": horizon + 1,
                "walk_forward_folds": len(fold_reports),
                "out_of_sample_rows": int(y_test_all.size),
            },
            "features": final_features,
            "regression": regression_metrics,
            "classification": classification_metrics,
            "logistic_baseline": logistic_metrics,
            "walk_forward": {
                "folds": fold_reports,
                "improved_regression_fold_count": (
                    improved_regression_fold_count
                ),
                "improved_classification_fold_count": (
                    improved_classification_fold_count
                ),
                "required_improved_folds": stable_fold_count,
                "regression_stable": regression_folds_stable,
                "classification_stable": classification_folds_stable,
            },
            "reliability": {
                "status": (
                    "acceptable" if not reliability_reasons else "not_reliable"
                ),
                "regression_status": regression_status,
                "classification_status": classification_status,
                "reasons": reliability_reasons,
                "better_than_zero_baseline": better_than_zero,
                "better_than_majority_baseline": better_than_majority,
                "auc_above_chance": auc_above_chance,
                "better_than_logistic_baseline": better_than_logistic,
                "regression_folds_stable": regression_folds_stable,
                "classification_folds_stable": classification_folds_stable,
            },
            "prediction": {
                "as_of_date": str(latest_row["trade_date"]),
                f"expected_return_{horizon}d": _safe_float(expected_return),
                f"up_probability_{horizon}d": _safe_float(up_probability),
                "expected_return": _safe_float(expected_return),
                "up_probability": _safe_float(up_probability),
                "direction": "up" if up_probability >= 0.5 else "down",
            },
            "latest_predictions": latest_predictions,
            "disclaimer": (
                "Statistical model output only; not a guarantee of future returns."
            ),
        }
        return GBMResult(
            report=report,
            features=dataset.all_features,
            out_of_sample=pl.concat(
                out_of_sample_parts,
                how="diagonal_relaxed",
            ).sort(["trade_date", "stock_code"]),
        )

    @staticmethod
    def _fold_ranges(
        dates: pl.Series,
        *,
        folds: int,
        min_test_rows: int,
        target_horizon: int,
    ) -> list[tuple[date, date, date]]:
        if folds < 2:
            raise ValueError("At least two folds are required")
        total_days = dates.len()
        initial_train_days = max(120, math.ceil(total_days * 0.5))
        remaining_days = total_days - initial_train_days
        if remaining_days < folds * min_test_rows:
            raise ValueError(
                f"only {remaining_days} days remain after initial training"
            )
        test_window = remaining_days // folds
        ranges = []
        for fold_index in range(folds):
            test_start_index = initial_train_days + fold_index * test_window
            test_end_index = (
                initial_train_days + (fold_index + 1) * test_window - 1
                if fold_index < folds - 1
                else total_days - 1
            )
            train_end_index = test_start_index - target_horizon - 1
            if train_end_index < 60:
                raise ValueError("initial training window is too short")
            ranges.append(
                (
                    dates[train_end_index],
                    dates[test_start_index],
                    dates[test_end_index],
                )
            )
        return ranges


def _regression_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    return {
        "rmse": _safe_float(math.sqrt(np.mean((actual - predicted) ** 2))),
        "mae": _safe_float(mean_absolute_error(actual, predicted)),
        "r2": _safe_float(r2_score(actual, predicted)),
        "baseline_zero_rmse": _safe_float(math.sqrt(np.mean(actual**2))),
        "test_rows": int(actual.size),
    }


def _classification_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    positive_rate = float(np.mean(actual))
    return {
        "accuracy": _safe_float(accuracy_score(actual, predicted)),
        "precision": _safe_float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": _safe_float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": _safe_float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": (
            _safe_float(roc_auc_score(actual, probabilities))
            if len(np.unique(actual)) == 2
            else None
        ),
        "positive_rate": _safe_float(positive_rate),
        "majority_baseline_accuracy": _safe_float(
            max(positive_rate, 1.0 - positive_rate)
        ),
        "test_rows": int(actual.size),
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 8)


def _apply_excess_target(
    training: pl.DataFrame,
    benchmark_daily: pl.DataFrame,
    *,
    horizon: int,
    target_return_column: str,
    target_up_column: str,
) -> pl.DataFrame:
    benchmark = benchmark_daily
    if "trade_date" in benchmark.columns:
        date_column = "trade_date"
    elif "date" in benchmark.columns:
        date_column = "date"
    else:
        raise ValueError("Benchmark data requires date or trade_date")
    if "close" not in benchmark.columns:
        raise ValueError("Benchmark data requires close")
    benchmark = benchmark.rename({date_column: "trade_date"})
    date_dtype = benchmark.schema["trade_date"]
    if date_dtype == pl.Utf8:
        benchmark = benchmark.with_columns(
            pl.col("trade_date").str.strptime(
                pl.Date,
                "%Y-%m-%d",
                strict=False,
            )
        )
    elif isinstance(date_dtype, pl.Datetime):
        benchmark = benchmark.with_columns(
            pl.col("trade_date").cast(pl.Date)
        )
    benchmark = (
        benchmark.select(
            "trade_date",
            pl.col("close").cast(pl.Float64).alias("benchmark_close"),
        )
        .drop_nulls()
        .sort("trade_date")
        .unique("trade_date", keep="last")
        .with_columns(
            (
                pl.col("benchmark_close").shift(-horizon)
                / pl.col("benchmark_close")
                - 1.0
            ).alias("benchmark_future_return")
        )
        .select("trade_date", "benchmark_future_return")
    )
    joined = training.join(
        benchmark,
        on="trade_date",
        how="left",
    ).drop_nulls(["benchmark_future_return"])
    excess = (
        pl.col(target_return_column) - pl.col("benchmark_future_return")
    )
    return joined.with_columns(
        excess.alias(target_return_column),
        (excess > 0).alias(target_up_column),
    ).drop("benchmark_future_return")


def _select_point_in_time_features(
    training: pl.DataFrame,
    target_return_column: str,
    candidate_features: list[str],
    *,
    maximum_features: int,
    minimum_icir: float,
    minimum_features: int = 8,
) -> list[str]:
    scored = []
    minimum_rows = max(30, len(candidate_features))
    for feature in candidate_features:
        data = training.select(
            "trade_date",
            pl.col(feature).alias("_feature"),
            pl.col(target_return_column).alias("_target"),
        ).drop_nulls()
        if data.height < minimum_rows:
            continue
        coverage = data.height / max(1, training.height)
        if coverage < 0.70:
            continue
        ranked = data.with_columns(
            pl.col("_feature").rank().over("trade_date").alias("_feature_rank"),
            pl.col("_target").rank().over("trade_date").alias("_target_rank"),
        )
        by_date = (
            ranked.group_by("trade_date")
            .agg(
                pl.len().alias("_count"),
                pl.corr("_feature_rank", "_target_rank").alias("_ic"),
            )
            .filter(pl.col("_count") >= 5)
            .drop_nulls("_ic")
        )
        if by_date.height < 20:
            continue
        values = by_date["_ic"].to_numpy()
        mean_ic = float(np.mean(values))
        std_ic = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        icir = mean_ic / std_ic if std_ic > 0 else None
        positive_rate = float(np.mean(values > 0))
        scored.append(
            {
                "feature": feature,
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "icir": icir,
                "positive_rate": positive_rate,
                "coverage": coverage,
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
    positive_candidates = [
        item
        for item in sorted(
            scored,
            key=lambda item: item["mean_ic"],
            reverse=True,
        )
        if item["mean_ic"] > 0
    ]
    for item in positive_candidates:
        if len(selected) >= minimum_features:
            break
        if item["feature"] not in selected:
            selected.append(item["feature"])

    technical = [
        item["feature"]
        for item in positive_candidates
        if not item["feature"].startswith("fund_")
    ]
    fundamental = [
        item["feature"]
        for item in positive_candidates
        if item["feature"].startswith("fund_")
    ]
    for feature in technical:
        if sum(not name.startswith("fund_") for name in selected) >= 4:
            break
        if feature not in selected:
            selected.append(feature)
    for feature in fundamental:
        if sum(name.startswith("fund_") for name in selected) >= 2:
            break
        if feature not in selected:
            selected.append(feature)

    if selected:
        return selected[:maximum_features]
    scored.sort(key=lambda item: item["mean_ic"], reverse=True)
    return [item["feature"] for item in scored[:min(5, len(scored))]]


def _apply_cross_sectional_neutralization(
    frame: pl.DataFrame,
    *,
    sector_map: dict[str, str],
) -> pl.DataFrame:
    """Rank-normalize stock-specific features within sector per trade date.

    Falls back to global rank normalization when a sector has fewer than
    MIN_SECTOR_SIZE stocks on a given date.
    """
    sector_frame = pl.DataFrame(
        {
            "stock_code": list(sector_map),
            "sector": list(sector_map.values()),
        }
    )
    if sector_frame.is_empty():
        frame = frame.with_columns(pl.lit("other").alias("sector"))
    else:
        frame = frame.join(
            sector_frame,
            on="stock_code",
            how="left",
        ).with_columns(pl.col("sector").fill_null("other"))

    sector_sizes = frame.group_by(["trade_date", "sector"]).len()
    frame = frame.join(
        sector_sizes.rename({"len": "_sector_size"}),
        on=["trade_date", "sector"],
        how="left",
    )

    for feature in NEUTRALIZABLE_FEATURES:
        sector_rank = (
            pl.col(feature)
            .rank("average")
            .over(["trade_date", "sector"])
            / pl.len()
            .over(["trade_date", "sector"])
            .cast(pl.Float64)
        )
        global_rank = (
            pl.col(feature)
            .rank("average")
            .over("trade_date")
            / pl.len()
            .over("trade_date")
            .cast(pl.Float64)
        )
        frame = frame.with_columns(
            sector_rank.alias(f"_sector_rank_{feature}"),
            global_rank.alias(f"_global_rank_{feature}"),
        ).with_columns(
            pl.when(pl.col("_sector_size") >= MIN_SECTOR_SIZE)
            .then(pl.col(f"_sector_rank_{feature}"))
            .otherwise(pl.col(f"_global_rank_{feature}"))
            .alias(feature)
        ).drop([f"_sector_rank_{feature}", f"_global_rank_{feature}"])

    return frame.drop("_sector_size")



