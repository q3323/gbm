from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from app.config import get_settings
from app.services.gbm import (
    FEATURE_COLUMNS,
    NEUTRALIZABLE_FEATURES,
    GBMFeatureBuilder,
)
from app.storage import ParquetStore


def main() -> None:
    store = ParquetStore(get_settings())
    daily = store.load_all_dwd_daily()
    print(f"daily rows: {daily.height}")
    print(f"stocks: {daily['stock_code'].n_unique()}")

    plain = GBMFeatureBuilder().build(daily, target_horizon=10)
    print(f"plain: rows={plain.training.height} days={plain.trading_day_count}")

    neutral = GBMFeatureBuilder().build_neutralized(daily, target_horizon=10)
    print(
        f"neutral: rows={neutral.training.height} "
        f"days={neutral.trading_day_count}"
    )

    for feature in NEUTRALIZABLE_FEATURES:
        values = neutral.training[feature].to_numpy()
        assert np.all(np.isfinite(values)), feature
        assert values.min() > 0, feature
        assert values.max() <= 1.0 + 1e-9, feature
    print("all neutralizable features in (0, 1] OK")

    aggregate_features = [
        name for name in FEATURE_COLUMNS if name not in NEUTRALIZABLE_FEATURES
    ]
    for feature in aggregate_features:
        a = plain.training[feature].to_numpy()
        b = neutral.training[feature].to_numpy()
        assert np.allclose(a, b), feature
    print("market aggregates unchanged OK")


if __name__ == "__main__":
    main()
