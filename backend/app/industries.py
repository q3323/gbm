from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


INDUSTRY_CONFIG = Path(__file__).resolve().parents[1] / "config/stock_industries.json"


@lru_cache(maxsize=1)
def load_industry_map() -> dict[str, str]:
    if not INDUSTRY_CONFIG.exists():
        return {}
    raw = json.loads(INDUSTRY_CONFIG.read_text(encoding="utf-8"))
    return {
        str(item["code"]).zfill(6): str(item.get("industry") or "Unknown")
        for item in raw.get("industries", [])
    }


# Map Tushare fine-grained industries into 8 broad sectors for cross-sectional neutralization.
INDUSTRY_TO_SECTOR: dict[str, str] = {
    "银行": "banking",
    "证券": "non_bank_financials",
    "保险": "non_bank_financials",
    "白酒": "consumer",
    "家用电器": "consumer",
    "食品": "consumer",
    "乳制品": "consumer",
    "中成药": "pharma",
    "化学制药": "pharma",
    "生物制药": "pharma",
    "医疗保健": "pharma",
    "通信设备": "tech",
    "元器件": "tech",
    "软件服务": "tech",
    "IT设备": "tech",
    "半导体": "tech",
    "电池": "tech",
    "电气设备": "tech",
    "汽车整车": "manufacturing",
    "钢加工": "energy_materials",
    "石油加工": "energy_materials",
    "石油开采": "energy_materials",
    "煤炭开采": "energy_materials",
    "铜": "energy_materials",
    "水泥": "energy_materials",
    "化工原料": "energy_materials",
    "机场": "utilities_infra",
    "全国地产": "utilities_infra",
    "电信运营": "utilities_infra",
    "水运": "utilities_infra",
    "建筑工程": "utilities_infra",
}

MIN_SECTOR_SIZE = 5


@lru_cache(maxsize=1)
def load_sector_map() -> dict[str, str]:
    """Return {stock_code: sector} derived from the industry map."""
    industry_map = load_industry_map()
    return {
        code: INDUSTRY_TO_SECTOR.get(industry, "other")
        for code, industry in industry_map.items()
    }
