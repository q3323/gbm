from __future__ import annotations

import json
from datetime import date

import pytest

from scripts.sync_universe import incremental_start, load_universe


def test_load_universe_normalizes_and_rejects_duplicates(tmp_path) -> None:
    path = tmp_path / "universe.json"
    path.write_text(
        json.dumps(
            {
                "stocks": [
                    {"code": "1", "name": "One"},
                    {"code": "600000", "name": "Two"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_universe(path) == [
        {"code": "000001", "name": "One"},
        {"code": "600000", "name": "Two"},
    ]

    path.write_text(
        json.dumps({"stocks": [{"code": "000001"}] * 2}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_universe(path)


def test_incremental_start_advances_one_day() -> None:
    assert (
        incremental_start(date(2026, 11, 10), "20240101", "20261113")
        == "20261111"
    )
    assert incremental_start(None, "20240101", "20261113") == "20240101"
    assert incremental_start(date(2026, 11, 13), "20240101", "20261113") is None
