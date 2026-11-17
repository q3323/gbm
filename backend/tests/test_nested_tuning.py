from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from app.services.nested_tuning import (
    NestedTuningConfig,
    _cross_sectional_metrics,
    _inner_date_folds,
    _validate_config,
)


def test_inner_folds_apply_purge_and_embargo() -> None:
    start = date(2024, 1, 1)
    dates = pl.Series(
        "trade_date",
        [start + timedelta(days=index) for index in range(200)],
    )
    folds = _inner_date_folds(
        dates,
        n_splits=3,
        purge_days=11,
        embargo_days=2,
    )
    values = dates.to_list()
    assert len(folds) >= 2
    for train_end, validation_start, _ in folds:
        train_index = values.index(train_end)
        validation_index = values.index(validation_start)
        assert validation_index - train_index >= 13


def test_nested_tuning_config_validation() -> None:
    with pytest.raises(ValueError, match="inner"):
        _validate_config(NestedTuningConfig(inner_splits=1))
    with pytest.raises(ValueError, match="trial"):
        _validate_config(NestedTuningConfig(trials=0))


def test_cross_sectional_rank_ic_detects_perfect_ranking() -> None:
    dates = [date(2026, 1, 5)] * 10
    actual_returns = np.arange(10, dtype=float) / 100.0
    probabilities = np.arange(10, dtype=float) / 10.0
    metrics = _cross_sectional_metrics(
        dates,
        actual_returns,
        probabilities,
    )
    assert metrics["mean_rank_ic"] == 1.0
    assert metrics["mean_top_excess_return"] > 0
