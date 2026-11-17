from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_report(data_dir: Path, model: str, code: str, horizon: int) -> dict[str, Any]:
    path = (
        data_dir
        / "ads"
        / "ensemble"
        / model
        / f"stock_code={code}"
        / f"horizon={horizon}"
        / "latest.json"
    )
    if not path.exists():
        raise SystemExit(f"Ensemble report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_metrics(report: dict[str, Any]) -> dict[str, Any]:
    cross = report["ensemble"]["cross_sectional"]
    gbm = report["ensemble"]["backtest"]["gbm"]
    benchmark = report["ensemble"]["backtest"]["benchmark"]
    seed_rows = report.get("seeds", [])
    seed_returns = [row["backtest"]["excess_total_return"] for row in seed_rows]
    return {
        "mean_rank_ic": cross.get("mean_rank_ic"),
        "rank_ic_std": cross.get("rank_ic_std"),
        "positive_rank_ic_rate": cross.get("positive_rank_ic_rate"),
        "total_return": gbm.get("total_return"),
        "excess_total_return": (
            gbm.get("total_return") - benchmark.get("total_return")
            if gbm.get("total_return") is not None
            and benchmark.get("total_return") is not None
            else None
        ),
        "sharpe": gbm.get("sharpe"),
        "max_drawdown": gbm.get("max_drawdown"),
        "seed_excess_min": min(seed_returns) if seed_returns else None,
        "seed_excess_max": max(seed_returns) if seed_returns else None,
        "seed_excess_range": (
            max(seed_returns) - min(seed_returns) if seed_returns else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare v2 vs v3 ensemble results side by side"
    )
    parser.add_argument("--code", default="000001")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (default: project data dir)",
    )
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from app.config import get_settings

        data_dir = get_settings().data_dir

    v2 = _load_report(data_dir, "gbm", args.code, args.horizon)
    v3 = _load_report(data_dir, "gbm_v3", args.code, args.horizon)
    v2_metrics = _extract_metrics(v2)
    v3_metrics = _extract_metrics(v3)

    print(f"Stock {args.code} | horizon {args.horizon}d")
    print(f"{'metric':<28} {'v2':>10} {'v3':>10} {'delta':>10}")
    print("-" * 62)
    for name in v2_metrics:
        a = v2_metrics[name]
        b = v3_metrics[name]
        a_str = f"{a:.4f}" if a is not None else "-"
        b_str = f"{b:.4f}" if b is not None else "-"
        delta_str = (
            f"{b - a:+.4f}" if a is not None and b is not None else "-"
        )
        print(f"{name:<28} {a_str:>10} {b_str:>10} {delta_str:>10}")

    v2_meta = v2.get("metadata", {})
    v3_meta = v3.get("metadata", {})
    print()
    print(f"v2 model: {v2_meta.get('model', '-')}")
    print(f"v3 model: {v3_meta.get('model', '-')}")
    print(f"v2 seeds: {v2_meta.get('seeds', '-')}")
    print(f"v3 seeds: {v3_meta.get('seeds', '-')}")


if __name__ == "__main__":
    main()
