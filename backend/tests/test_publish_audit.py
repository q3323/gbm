from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.publish_predictions import _validate_group_audits


def test_publish_rejects_failed_required_audit(tmp_path: Path) -> None:
    audit_path = (
        tmp_path
        / "quality"
        / "group_audits"
        / "etf"
        / "horizon=20"
        / "latest.json"
    )
    audit_path.parent.mkdir(parents=True)
    audit_path.write_text(
        json.dumps(
            {
                "status": "fail",
                "checks": {
                    "risk_comparability": False,
                    "positive_strategy_return": False,
                },
            }
        ),
        encoding="utf-8",
    )
    groups = [
        {
            "id": "etf",
            "horizon_days": 20,
            "require_audit": True,
        }
    ]

    with pytest.raises(SystemExit, match="risk_comparability"):
        _validate_group_audits(
            tmp_path,
            groups,
            allow_failed=False,
        )

    _validate_group_audits(
        tmp_path,
        groups,
        allow_failed=True,
    )
