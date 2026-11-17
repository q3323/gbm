from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ensure_data_dirs, get_settings
from app.services.ensemble import EnsembleConfig, GBMSeedEnsembleService
from app.storage import ParquetStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Average GBM seed OOS predictions")
    parser.add_argument("--code", default="000001")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        default="gbm",
        help="Model tag for seed OOS directory (gbm or gbm_v3)",
    )
    args = parser.parse_args()

    settings = get_settings()
    ensure_data_dirs(settings)
    base = (
        settings.data_dir
        / "ads"
        / "nested_tuning"
        / args.model
        / f"stock_code={args.code}"
        / f"horizon={args.horizon}"
    )
    seed_frames = []
    for path in sorted(base.glob("seed=*/out_of_sample.parquet")):
        seed = int(path.parent.name.split("=", 1)[1])
        seed_frames.append((seed, pl.read_parquet(path)))
    if len(seed_frames) < 2:
        raise SystemExit("At least two seed OOS files are required")

    result = GBMSeedEnsembleService().run(
        seed_frames,
        args.code,
        EnsembleConfig(
            horizon=args.horizon,
            top_fraction=args.top_fraction,
            random_trials=args.random_trials,
            seed=args.seed,
            model_tag=args.model,
        ),
    )
    store = ParquetStore(settings)
    path = store.save_ensemble_result(
        args.code,
        args.model,
        args.horizon,
        result.report,
        result.out_of_sample,
    )
    print(json.dumps(result.report, ensure_ascii=False, indent=2))
    print(f"ensemble report: {path}")


if __name__ == "__main__":
    main()
