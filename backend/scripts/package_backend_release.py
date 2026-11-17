from __future__ import annotations

import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
BUILD_ROOT = PROJECT_ROOT / "build" / "backend_releases"

SERVER_SCRIPTS = (
    "backfill_adjust_type.py",
    "compute_indicators.py",
    "sync_daily.py",
    "sync_financial.py",
    "sync_financial_all.py",
    "sync_universe.py",
)


def main() -> None:
    release_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
        "%Y%m%d_%H%M%S"
    )
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    archive = BUILD_ROOT / f"gupaio-backend-{release_id}.tar.gz"
    include = [
        "app",
        "config",
        "deploy",
        "requirements.txt",
        "pyproject.toml",
        ".env.example",
    ]
    with tarfile.open(archive, "w:gz") as tar:
        for relative in include:
            source = BACKEND_ROOT / relative
            if not source.exists():
                raise SystemExit(f"missing backend release file: {source}")
            tar.add(
                source,
                arcname=relative,
                filter=_exclude_runtime_files,
            )
        for script_name in SERVER_SCRIPTS:
            source = BACKEND_ROOT / "scripts" / script_name
            if not source.is_file():
                raise SystemExit(f"missing backend script: {source}")
            tar.add(
                source,
                arcname=f"scripts/{script_name}",
                filter=_exclude_runtime_files,
            )
    print(archive)


def _exclude_runtime_files(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    if "__pycache__" in info.name or info.name.endswith(".pyc"):
        return None
    return info


if __name__ == "__main__":
    main()
