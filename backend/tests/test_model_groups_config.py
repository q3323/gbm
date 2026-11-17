from __future__ import annotations

from app.config import BACKEND_ROOT
from app.services.group_bagging import load_group_configs


def test_configured_groups_have_expected_stock_counts() -> None:
    groups = load_group_configs(
        BACKEND_ROOT / "config" / "model_groups.json"
    )
    by_id = {group["id"]: group for group in groups}
    assert set(by_id) == {
        "us_ndx_index",
        "cn_growth_tech",
        "us_ndx_etfs",
    }
    assert len(by_id["cn_growth_tech"]["stock_codes"]) == 50
    assert len(by_id["us_ndx_etfs"]["stock_codes"]) == 11
    etf_group = by_id["us_ndx_etfs"]
    assert etf_group["model_stock_codes"] == [
        "QQQ",
        "QQQM",
        "QQQE",
        "QTEC",
        "QQQI",
        "QQMG",
    ]
    assert etf_group["minimum_available_stocks"] == 6
    assert set(etf_group["product_profiles"]) == set(etf_group["stock_codes"])
    assert {
        profile["category"]
        for profile in etf_group["product_profiles"].values()
    } == {"long_1x", "leveraged_long", "inverse"}
    assert etf_group["require_audit"] is True
    assert by_id["cn_growth_tech"]["factor_pool_name"] == "tech"
    assert by_id["us_ndx_etfs"]["factor_pool_name"] == "us_etf"
    assert by_id["us_ndx_index"]["horizon_days"] == 10
    assert by_id["cn_growth_tech"]["horizon_days"] == 10
    assert by_id["us_ndx_etfs"]["horizon_days"] == 20
