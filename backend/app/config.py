from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(BACKEND_ROOT / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default.resolve()


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    tushare_token: str
    api_token: str
    host: str
    port: int
    log_level: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        data_dir=_path_from_env("STOCK_DATA_DIR", PROJECT_ROOT / "data"),
        tushare_token=os.getenv("TUSHARE_TOKEN", "").strip(),
        api_token=os.getenv("STOCK_API_TOKEN", "").strip(),
        host=os.getenv("APP_HOST", "0.0.0.0").strip() or "0.0.0.0",
        port=int(os.getenv("APP_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
    )


def ensure_data_dirs(settings: Settings) -> None:
    for relative in (
        "ods",
        "dwd",
        "dws",
        "ads",
        "quality/daily",
        "quality/financial",
        "quality/source_comparison",
        "quality/pipeline_runs",
        "cache",
    ):
        (settings.data_dir / relative).mkdir(parents=True, exist_ok=True)

