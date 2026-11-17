from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import BACKEND_ROOT, ensure_data_dirs, get_settings
from app.services.gbm import GBMFeatureBuilder
from app.services.group_bagging import (
    TECH_FACTOR_POOL,
    US_ETF_FACTOR_POOL,
    GroupBaggingConfig,
    GroupedGBMBaggingService,
    load_group_configs,
)
from app.services.nested_tuning import _cross_sectional_metrics
from app.storage import ParquetStore


GROUP_CONFIG_PATH = BACKEND_ROOT / "config" / "model_groups.json"
FACTOR_COVERAGE_THRESHOLD = 0.85


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit model group coverage, comparability, and OOS quality"
    )
    parser.add_argument("--group", required=True)
    parser.add_argument("--horizon", type=int)
    parser.add_argument(
        "--skip-oos-recompute",
        action="store_true",
        help="Use the existing report instead of rerunning walk-forward models",
    )
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    store = ParquetStore(settings)
    groups = load_group_configs(GROUP_CONFIG_PATH)
    group = next(
        (item for item in groups if item.get("id") == args.group),
        None,
    )
    if group is None:
        raise SystemExit(f"Unknown model group: {args.group}")

    group_type = str(group.get("type", ""))
    horizon = args.horizon or int(group.get("horizon_days", 10))
    display_codes = [
        str(code) for code in group.get("stock_codes", [])
    ]
    codes = [
        str(code)
        for code in group.get("model_stock_codes", display_codes)
    ]
    all_daily = _load_group_daily(store, group, display_codes)
    daily = all_daily.filter(pl.col("stock_code").is_in(codes))
    factor_pool = _factor_pool(group)
    dataset = GBMFeatureBuilder().build(
        daily,
        target_horizon=horizon,
    )
    coverage = _coverage_report(
        daily,
        dataset.all_features,
        codes,
        factor_pool,
        horizon,
        int(group.get("minimum_available_stocks", 5)),
    )
    comparability = _comparability_report(
        daily,
        {
            code: profile
            for code, profile in group.get("product_profiles", {}).items()
            if code in codes
        },
    )
    oos = _out_of_sample_report(
        store,
        group,
        daily,
        horizon,
        recompute=not args.skip_oos_recompute,
    )
    checks = _checks(coverage, comparability, oos)
    report = {
        "group_id": args.group,
        "name": group.get("name", args.group),
        "horizon_days": horizon,
        "generated_at": date.today().isoformat(),
        "factor_pool": list(factor_pool),
        "display_stock_codes": display_codes,
        "model_stock_codes": codes,
        "coverage": coverage,
        "comparability": comparability,
        "out_of_sample": oos,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }
    path = (
        store.data_dir
        / "quality"
        / "group_audits"
        / args.group
        / f"horizon={horizon}"
        / "latest.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": checks,
                "first_eligible_date": coverage["first_eligible_date"],
                "minimum_factor_coverage": coverage[
                    "minimum_factor_coverage"
                ],
                "product_categories": comparability[
                    "category_counts"
                ],
                "oos_rank_ic": oos["rank_ic_mean"],
                "oos_positive_rank_ic_rate": oos[
                    "positive_rank_ic_rate"
                ],
                "oos_total_return": oos["total_return"],
                "reliable_seed_count": oos["reliable_seed_count"],
                "ranking_reliable_seed_count": oos[
                    "ranking_reliable_seed_count"
                ],
                "report": str(path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["status"] != "pass":
        raise SystemExit(1)


def _load_group_daily(
    store: ParquetStore,
    group: dict[str, Any],
    codes: list[str],
) -> pl.DataFrame:
    group_type = str(group.get("type", ""))
    if group_type == "us_etf_group":
        frames = []
        for code in codes:
            path = (
                store.data_dir
                / "ods"
                / "us_etf"
                / f"symbol={code}"
                / "bars.parquet"
            )
            if not path.exists():
                raise SystemExit(f"Missing US ETF data: {path}")
            frames.append(pl.read_parquet(path))
        return pl.concat(frames, how="diagonal_relaxed").sort(
            ["stock_code", "trade_date"]
        )

    daily = store.load_all_dwd_daily()
    return daily.filter(pl.col("stock_code").is_in(codes)).sort(
        ["stock_code", "trade_date"]
    )


def _factor_pool(group: dict[str, Any]) -> tuple[str, ...]:
    name = group.get("factor_pool_name")
    if name == "tech":
        return TECH_FACTOR_POOL
    if name == "us_etf":
        return US_ETF_FACTOR_POOL
    return ()


def _coverage_report(
    daily: pl.DataFrame,
    features: pl.DataFrame,
    codes: list[str],
    factor_pool: tuple[str, ...],
    horizon: int,
    minimum_available_stocks: int,
) -> dict[str, Any]:
    data_coverage = []
    for code in codes:
        frame = daily.filter(pl.col("stock_code") == code)
        if frame.is_empty():
            continue
        data_coverage.append(
            {
                "code": code,
                "rows": frame.height,
                "unique_dates": frame["trade_date"].n_unique(),
                "duplicate_dates": (
                    frame.height - frame["trade_date"].n_unique()
                ),
                "first_date": str(frame["trade_date"].min()),
                "last_date": str(frame["trade_date"].max()),
                "null_close": frame["close"].null_count(),
                "null_volume": frame["volume"].null_count(),
                "nonpositive_close": frame.filter(
                    pl.col("close") <= 0
                ).height,
            }
        )

    date_counts = (
        features.group_by("trade_date")
        .agg(pl.col("stock_code").n_unique().alias("stock_count"))
        .sort("trade_date")
    )
    eligible_dates = date_counts.filter(
        pl.col("stock_count") >= minimum_available_stocks
    )
    first_eligible = (
        eligible_dates["trade_date"].min()
        if not eligible_dates.is_empty()
        else None
    )
    eligible = (
        features.filter(pl.col("trade_date") >= first_eligible)
        if first_eligible is not None
        else features.clear()
    )

    factor_coverage = []
    for feature in factor_pool:
        if feature not in eligible.columns:
            factor_coverage.append(
                {
                    "factor": feature,
                    "eligible_coverage": 0.0,
                    "stock_coverage": 0,
                    "latest_value_available": False,
                }
            )
            continue
        eligible_coverage = (
            1.0 - eligible[feature].null_count() / eligible.height
            if eligible.height
            else 0.0
        )
        stock_coverage = (
            eligible.select(
                pl.col("stock_code"),
                pl.col(feature).is_not_null().alias("present"),
            )
            .group_by("stock_code")
            .agg(pl.col("present").mean())
            .filter(pl.col("present") >= FACTOR_COVERAGE_THRESHOLD)
            .height
        )
        latest_value_available = (
            not features.sort("trade_date")
            .tail(1)[feature]
            .is_null()
            .all()
        )
        factor_coverage.append(
            {
                "factor": feature,
                "eligible_coverage": round(eligible_coverage, 6),
                "stock_coverage": stock_coverage,
                "latest_value_available": latest_value_available,
            }
        )

    cross_section_counts = (
        eligible["trade_date"].value_counts().sort("trade_date")
    )
    count_values = cross_section_counts["count"].to_numpy()
    dates_below_minimum = int(
        np.sum(count_values < minimum_available_stocks)
    )
    cross_section_coverage_ratio = (
        1.0 - dates_below_minimum / len(count_values)
        if len(count_values)
        else 0.0
    )
    all_members_date = (
        features.group_by("trade_date")
        .agg(pl.col("stock_code").n_unique().alias("stock_count"))
        .filter(pl.col("stock_count") >= len(codes))["trade_date"]
        .min()
    )
    label_column = f"future_return_{horizon}d"
    label_coverage = (
        1.0 - features[label_column].null_count() / features.height
        if label_column in features.columns and features.height
        else 0.0
    )
    return {
        "first_eligible_date": (
            str(first_eligible) if first_eligible is not None else None
        ),
        "data": data_coverage,
        "factor_coverage": factor_coverage,
        "minimum_factor_coverage": round(
            min(
                (
                    item["eligible_coverage"]
                    for item in factor_coverage
                ),
                default=0.0,
            ),
            6,
        ),
        "label": {
            "column": label_column,
            "coverage": round(label_coverage, 6),
        },
        "cross_section": {
            "minimum_available_stocks": minimum_available_stocks,
            "minimum": int(count_values.min()),
            "median": float(np.median(count_values)),
            "maximum": int(count_values.max()),
            "dates": int(len(count_values)),
            "dates_below_minimum": dates_below_minimum,
            "coverage_ratio": round(cross_section_coverage_ratio, 6),
            "all_members_available_date": (
                str(all_members_date)
                if all_members_date is not None
                else None
            ),
        },
    }


def _comparability_report(
    daily: pl.DataFrame,
    product_profiles: dict[str, Any],
) -> dict[str, Any]:
    returns = (
        daily.sort(["stock_code", "trade_date"])
        .with_columns(
            (
                pl.col("close") / pl.col("close").shift(1) - 1.0
            ).over("stock_code").alias("return")
        )
        .drop_nulls("return")
    )
    benchmark = returns.filter(
        pl.col("stock_code") == "QQQ"
    ).select(
        pl.col("trade_date"),
        pl.col("return").alias("benchmark_return"),
    )
    products = []
    for code in sorted(product_profiles):
        profile = product_profiles[code]
        frame = (
            returns.filter(pl.col("stock_code") == code)
            .join(benchmark, on="trade_date", how="inner")
            .drop_nulls("benchmark_return")
        )
        if frame.height < 2:
            beta = None
            correlation = None
        else:
            covariance = float(
                np.cov(
                    frame["return"].to_numpy(),
                    frame["benchmark_return"].to_numpy(),
                    ddof=1,
                )[0, 1]
            )
            benchmark_variance = float(
                np.var(
                    frame["benchmark_return"].to_numpy(),
                    ddof=1,
                )
            )
            beta = (
                covariance / benchmark_variance
                if benchmark_variance > 0
                else None
            )
            correlation = float(
                np.corrcoef(
                    frame["return"].to_numpy(),
                    frame["benchmark_return"].to_numpy(),
                )[0, 1]
            )
        products.append(
            {
                "code": code,
                "category": profile.get("category", "unknown"),
                "leverage": profile.get("leverage"),
                "annualized_volatility": round(
                    float(frame["return"].std(ddof=1)) * np.sqrt(252),
                    6,
                ),
                "beta_to_qqq": (
                    round(beta, 6) if beta is not None else None
                ),
                "correlation_to_qqq": (
                    round(correlation, 6)
                    if correlation is not None
                    else None
                ),
                "average_dollar_volume": round(
                    float(
                        (
                            frame["volume"] * frame["close"]
                        ).mean()
                    ),
                    2,
                ),
            }
        )
    category_counts: dict[str, int] = {}
    for product in products:
        category = product["category"]
        category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "products": products,
        "category_counts": category_counts,
        "comparable": len(category_counts) == 1,
        "reason": (
            "Single product category"
            if len(category_counts) == 1
            else "Leveraged, inverse, and 1x products have different risk exposures"
        ),
    }


def _out_of_sample_report(
    store: ParquetStore,
    group: dict[str, Any],
    daily: pl.DataFrame,
    horizon: int,
    *,
    recompute: bool,
) -> dict[str, Any]:
    report_path = (
        store.data_dir
        / "ads"
        / "group_bagging"
        / str(group["id"])
        / f"horizon={horizon}"
        / "latest.json"
    )
    if not report_path.exists():
        raise SystemExit(f"Missing model report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evaluation = report.get("evaluation", {})
    result: dict[str, Any] = {
        "source": "existing_report",
        "sample_rows": evaluation.get("sample_rows"),
        "rank_ic_mean": evaluation.get("rank_ic_mean"),
        "rank_ic_std": evaluation.get("rank_ic_std"),
        "icir": evaluation.get("icir"),
        "positive_rank_ic_rate": evaluation.get(
            "positive_rank_ic_rate"
        ),
        "total_return": evaluation.get("total_return"),
        "sharpe": evaluation.get("sharpe"),
        "max_drawdown": evaluation.get("max_drawdown"),
        "reliable_seed_count": sum(
            seed.get("reliability") == "acceptable"
            for seed in report.get("seeds", [])
        ),
        "regression_reliable_seed_count": sum(
            seed.get("regression_reliability") == "acceptable"
            for seed in report.get("seeds", [])
        ),
        "ranking_reliable_seed_count": sum(
            (seed.get("rank_ic_mean") or 0.0) > 0
            and (seed.get("positive_rank_ic_rate") or 0.0) >= 0.55
            for seed in report.get("seeds", [])
        ),
        "baselines": evaluation.get("baselines", {}),
        "seed_count": len(report.get("seeds", [])),
    }
    if not recompute:
        return result

    benchmark_symbol = str(group.get("benchmark_symbol", "")).strip()
    benchmark = None
    if benchmark_symbol:
        benchmark_path = (
            store.data_dir
            / "ods"
            / "benchmarks"
            / f"symbol={benchmark_symbol.lstrip('.')}"
            / "daily.parquet"
        )
        if benchmark_path.exists():
            benchmark = pl.read_parquet(benchmark_path)
    config = GroupBaggingConfig(
        group_id=str(group["id"]),
        name=str(group.get("name", group["id"])),
        market=str(group.get("market", "US")),
        stock_codes=tuple(group.get("stock_codes", [])),
        seeds=tuple(group.get("seeds", [7, 42, 2024, 123, 888])),
        sample_fraction=float(group.get("sample_fraction", 0.8)),
        horizon_days=horizon,
        factor_pool=_factor_pool(group),
        minimum_available_stocks=int(
            group.get("minimum_available_stocks", 5)
        ),
    )
    result_obj = GroupedGBMBaggingService().run(
        daily,
        config,
        benchmark_daily=benchmark,
        min_training_rows=800,
    )
    oos = result_obj.out_of_sample
    if oos is None:
        raise SystemExit("Model service did not return OOS rows")
    probability_metrics = _cross_sectional_metrics(
        oos["trade_date"].to_list(),
        oos["future_return"].to_numpy(),
        oos["up_probability"].to_numpy(),
    )
    expected_return_metrics = _cross_sectional_metrics(
        oos["trade_date"].to_list(),
        oos["future_return"].to_numpy(),
        oos["predicted_return"].to_numpy(),
    )
    folds = []
    for fold in sorted(oos["fold"].unique().to_list()):
        frame = oos.filter(pl.col("fold") == fold)
        probability_fold_metrics = _cross_sectional_metrics(
            frame["trade_date"].to_list(),
            frame["future_return"].to_numpy(),
            frame["up_probability"].to_numpy(),
        )
        expected_return_fold_metrics = _cross_sectional_metrics(
            frame["trade_date"].to_list(),
            frame["future_return"].to_numpy(),
            frame["predicted_return"].to_numpy(),
        )
        folds.append(
            {
                "fold": int(fold),
                "rows": frame.height,
                "dates": frame["trade_date"].n_unique(),
                "first_date": str(frame["trade_date"].min()),
                "last_date": str(frame["trade_date"].max()),
                "rank_ic": expected_return_fold_metrics.get(
                    "mean_rank_ic"
                ),
                "positive_rank_ic_rate": expected_return_fold_metrics.get(
                    "positive_rank_ic_rate"
                ),
                "probability_rank_ic": probability_fold_metrics.get(
                    "mean_rank_ic"
                ),
                "top_bottom_spread": round(
                    _top_bottom_spread(frame, "predicted_return"),
                    8,
                ),
                "probability_top_bottom_spread": round(
                    _top_bottom_spread(frame, "up_probability"),
                    8,
                ),
            }
        )
    result.update(
        {
            "source": "independent_walk_forward_recompute",
            "first_date": str(oos["trade_date"].min()),
            "last_date": str(oos["trade_date"].max()),
            "rows": oos.height,
            "rank_ic_mean": expected_return_metrics.get("mean_rank_ic"),
            "rank_ic_std": expected_return_metrics.get("rank_ic_std"),
            "positive_rank_ic_rate": expected_return_metrics.get(
                "positive_rank_ic_rate"
            ),
            "probability_rank_ic_mean": probability_metrics.get(
                "mean_rank_ic"
            ),
            "expected_return_rank_ic_mean": expected_return_metrics.get(
                "mean_rank_ic"
            ),
            "folds": folds,
        }
    )
    return result


def _top_bottom_spread(
    frame: pl.DataFrame,
    score_column: str,
    fraction: float = 0.2,
) -> float:
    spreads = []
    for _, group in frame.group_by("trade_date"):
        if group.height < 5:
            continue
        ordered = group.sort(score_column)
        bucket_size = max(1, int(group.height * fraction))
        spreads.append(
            float(
                ordered.tail(bucket_size)["future_return"].mean()
                - ordered.head(bucket_size)["future_return"].mean()
            )
        )
    return float(np.mean(spreads)) if spreads else 0.0


def _checks(
    coverage: dict[str, Any],
    comparability: dict[str, Any],
    oos: dict[str, Any],
) -> dict[str, bool]:
    folds = oos.get("folds", [])
    baselines = oos.get("baselines", {})
    return {
        "factor_coverage": (
            coverage["minimum_factor_coverage"]
            >= FACTOR_COVERAGE_THRESHOLD
        ),
        "label_coverage": coverage["label"]["coverage"] >= 0.8,
        "cross_section_breadth": (
            coverage["cross_section"]["coverage_ratio"] >= 0.98
        ),
        "risk_comparability": bool(comparability["comparable"]),
        "positive_rank_ic": (oos.get("rank_ic_mean") or 0.0) > 0,
        "positive_ic_rate": (
            oos.get("positive_rank_ic_rate") or 0.0
        ) >= 0.55,
        "positive_strategy_return": (
            oos.get("total_return") or 0.0
        ) > 0,
        "maximum_drawdown_within_limit": (
            (oos.get("max_drawdown") or -1.0) >= -0.35
        ),
        "all_oos_folds_positive_rank_ic": (
            bool(folds)
            and all(
                (fold.get("rank_ic") or 0.0) > 0
                for fold in folds
            )
        ),
        "beats_momentum_baseline": (
            (oos.get("rank_ic_mean") or 0.0)
            > (baselines.get("momentum_rank_ic") or 0.0)
        ),
        "beats_logistic_baseline": (
            (oos.get("rank_ic_mean") or 0.0)
            > (baselines.get("logistic_rank_ic") or 0.0)
        ),
        "majority_ranking_seeds_reliable": (
            oos.get("seed_count", 0) > 0
            and oos.get("ranking_reliable_seed_count", 0)
            >= math.ceil(oos.get("seed_count", 0) * 0.6)
        ),
    }


if __name__ == "__main__":
    main()
