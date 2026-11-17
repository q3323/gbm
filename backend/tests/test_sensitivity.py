from __future__ import annotations

import pytest

from app.services.sensitivity import (
    SensitivityConfig,
    _summarize_runs,
    _validate_config,
)


def test_sensitivity_summary_detects_cost_robustness() -> None:
    runs = []
    for seed in (7, 42, 2024):
        for horizon in (5, 10):
            runs.append(
                {
                    "seed": seed,
                    "horizon_days": horizon,
                    "top_fraction": 0.2,
                    "cost_multiplier": 1.0,
                    "total_return": 0.12,
                    "annualized_return": 0.08,
                    "sharpe": 0.8,
                    "max_drawdown": -0.1,
                    "hit_rate": 0.56,
                    "total_cost_rate": 0.05,
                    "benchmark_total_return": 0.06,
                    "excess_total_return": 0.06,
                    "classification_status": "weak_signal",
                }
            )
            runs.append(
                {
                    **runs[-1],
                    "cost_multiplier": 2.0,
                    "total_return": 0.09,
                    "excess_total_return": 0.03,
                }
            )
    summary = _summarize_runs(
        runs,
        seeds=(7, 42, 2024),
        horizons=(5, 10),
        top_fractions=(0.2,),
        cost_multipliers=(1.0, 2.0),
    )
    assert summary["status"] == "robust"
    assert summary["by_cost"][1]["positive_excess_rate"] == 1.0


def test_sensitivity_config_rejects_empty_seeds() -> None:
    with pytest.raises(ValueError, match="seed"):
        _validate_config(SensitivityConfig(seeds=()))
