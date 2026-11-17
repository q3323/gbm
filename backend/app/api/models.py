from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.config import BACKEND_ROOT
from app.services.group_bagging import load_group_configs


router = APIRouter(prefix="/api/model", tags=["models"])
GROUP_CONFIG_PATH = BACKEND_ROOT / "config" / "model_groups.json"
STOCK_UNIVERSE_PATH = BACKEND_ROOT / "config" / "stock_universe.json"


@router.get("/groups")
async def model_groups(
    request: Request,
) -> list[dict]:
    try:
        configs = load_group_configs(GROUP_CONFIG_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid model group configuration: {exc}",
        ) from exc

    results = []
    stock_names = _load_stock_names()
    for config in configs:
        group_id = str(config.get("id", "")).strip()
        if not group_id:
            continue
        group_horizon = int(config.get("horizon_days", 10))
        report = request.app.state.store.load_group_bagging_result(
            group_id,
            group_horizon,
        )
        audit = _load_group_audit(
            request.app.state.store.data_dir,
            group_id,
            group_horizon,
        )
        if report is not None:
            group_names = {
                str(code): str(name)
                for code, name in (config.get("names") or {}).items()
            }
            prediction = report.get("group_prediction", {})
            expected_return = prediction.get("expected_return")
            if expected_return is not None:
                prediction["direction"] = (
                    "up"
                    if float(expected_return) > 0
                    else "down"
                    if float(expected_return) < 0
                    else "mixed"
                )
            for ranking in report.get("rankings", []):
                code = str(ranking.get("stock_code", ""))
                ranking["stock_name"] = (
                    str(config.get("name", code))
                    if config.get("type") == "index"
                    else group_names.get(
                        code,
                        stock_names.get(code, code),
                    )
                )
                stock_return = ranking.get("expected_return")
                if stock_return is not None:
                    ranking["direction"] = (
                        "up"
                        if float(stock_return) > 0
                        else "down"
                        if float(stock_return) < 0
                        else "mixed"
                    )
        results.append(
            {
                "id": group_id,
                "name": config.get("name", group_id),
                "market": config.get("market", ""),
                "type": config.get("type", ""),
                "symbol": config.get("symbol"),
                "stock_codes": config.get("stock_codes", []),
                "seeds": config.get("seeds", []),
                "horizon_days": group_horizon,
                "status": "ready" if report is not None else "not_trained",
                "audit_status": (
                    str(audit.get("status"))
                    if audit is not None
                    else "not_run"
                ),
                "audit_checks": (
                    audit.get("checks", {}) if audit is not None else {}
                ),
                "report": report,
            }
        )
    return results


@router.get("/release")
async def model_release(request: Request) -> dict:
    path = (
        request.app.state.store.data_dir
        / "ads"
        / "model_manifest"
        / "latest.json"
    )
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No published model release manifest",
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_stock_names() -> dict[str, str]:
    try:
        payload = json.loads(
            STOCK_UNIVERSE_PATH.read_text(encoding="utf-8")
        )
    except Exception:
        return {}
    return {
        str(item.get("code", "")): str(item.get("name", ""))
        for item in payload.get("stocks", [])
        if item.get("code")
    }


def _load_group_audit(
    data_dir: Path,
    group_id: str,
    horizon: int,
) -> dict | None:
    path = (
        data_dir
        / "quality"
        / "group_audits"
        / group_id
        / f"horizon={horizon}"
        / "latest.json"
    )
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
