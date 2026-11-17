from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "stock-analysis"


def test_provider_statuses() -> None:
    with TestClient(app) as client:
        response = client.get("/api/quality/providers")
    assert response.status_code == 200
    assert {item["name"] for item in response.json()} == {
        "tushare",
        "baostock",
        "akshare",
    }


def test_new_quality_and_analysis_routes_are_exposed() -> None:
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]
    assert "/api/quality/sources/compare" in paths
    assert "/api/stocks/{code}/analysis" in paths
    assert "/api/stocks/{code}/analysis/backtest" in paths
    assert "/api/stocks/{code}/model/ensemble" in paths
    assert "/api/model/groups" in paths
    assert "/api/model/release" in paths
    assert "/api/stocks/{code}/financial/latest" in paths


def test_three_model_groups_are_exposed() -> None:
    with TestClient(app) as client:
        response = client.get("/api/model/groups")
    assert response.status_code == 200
    assert {group["id"] for group in response.json()} == {
        "us_ndx_index",
        "cn_growth_tech",
        "us_ndx_etfs",
    }
    etf_group = next(
        group
        for group in response.json()
        if group["id"] == "us_ndx_etfs"
    )
    assert etf_group["audit_status"] in {"pass", "fail", "not_run"}


