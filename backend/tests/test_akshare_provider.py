from __future__ import annotations

import polars as pl

from app.providers.akshare import _normalize_catalog


def test_normalize_catalog_accepts_prefixed_etf_codes() -> None:
    frame = pl.DataFrame(
        {
            "代码": ["sz159915", "sh510300"],
            "名称": ["创业板ETF", "沪深300ETF"],
        }
    )

    result = _normalize_catalog(frame, "fund")

    by_code = {row["code"]: row for row in result.to_dicts()}
    assert by_code["159915"]["name"] == "创业板ETF"
    assert by_code["510300"]["name"] == "沪深300ETF"
    assert {row["asset_type"] for row in result.to_dicts()} == {"fund"}
