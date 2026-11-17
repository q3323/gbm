from __future__ import annotations

from scripts.nested_tune_seeds import summarize_seed_reports


def report(seed: int, rank_ic: float, excess: float, sharpe: float) -> dict:
    return {
        "metadata": {"seed": seed},
        "aggregate_outer": {
            "tuned": {"accuracy": 0.52, "roc_auc": 0.53},
            "tuned_cross_sectional": {
                "mean_rank_ic": rank_ic,
                "positive_rank_ic_rate": 0.6,
                "mean_top_excess_return": 0.003,
            },
        },
        "backtest": {
            "gbm": {
                "total_return": 0.12 + excess,
                "sharpe": sharpe,
                "max_drawdown": -0.1,
            },
            "equal_weight_benchmark": {"total_return": 0.12},
        },
    }


def test_seed_summary_marks_stable_results_robust() -> None:
    result = summarize_seed_reports(
        [
            report(7, 0.05, 0.02, 0.7),
            report(42, 0.04, 0.01, 0.6),
            report(2024, 0.06, 0.03, 0.8),
        ]
    )
    assert result["status"] == "robust"
    assert result["passed_criteria"] == 6
