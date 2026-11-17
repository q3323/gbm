from __future__ import annotations

import argparse
import getpass
import json
import posixpath
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
PREDICTION_SCRIPT = BACKEND_ROOT / "scripts" / "train_model_groups.py"
SYNC_SCRIPT = BACKEND_ROOT / "scripts" / "sync_universe.py"
PUBLISH_CONFIG = BACKEND_ROOT / "publish_config.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train predictions, package reports, and publish over SSH"
    )
    config = _load_publish_config()
    parser.add_argument("--host", default=config.get("host"))
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default=config.get("username", "root"))
    parser.add_argument(
        "--remote-root",
        default=config.get("remote_data_root", "/opt/gupaio/data"),
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--sync",
        action="store_true",
        default=bool(config.get("sync_before_train", False)),
    )
    parser.add_argument("--package-only", action="store_true")
    parser.add_argument(
        "--allow-failed-audit",
        action="store_true",
        help="Allow publishing groups whose audit status is not pass",
    )
    args = parser.parse_args()

    if not args.host:
        raise SystemExit("host is not configured")
    package_root = _build_package(
        skip_train=args.skip_train,
        sync_before_train=args.sync,
        allow_failed_audit=args.allow_failed_audit,
    )
    archive = _create_archive(package_root)
    print(f"generated: {archive}")
    if args.package_only:
        return

    password = getpass.getpass("SSH password: ")
    if not password:
        raise SystemExit("SSH password is required")
    _upload_and_publish(
        archive,
        host=args.host,
        port=args.port,
        user=args.user,
        password=password,
        remote_root=args.remote_root,
    )
    print("prediction report published")


def _build_package(
    *,
    skip_train: bool,
    sync_before_train: bool,
    allow_failed_audit: bool,
) -> Path:
    if sync_before_train:
        subprocess.run(
            [
                sys.executable,
                str(SYNC_SCRIPT),
                "--config",
                str(BACKEND_ROOT / "config" / "stock_universe.json"),
                "--start",
                "20180101",
                "--end",
                datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d"),
                "--source",
                "baostock",
                "--delay",
                "0.2",
                "--skip-features",
            ],
            cwd=str(BACKEND_ROOT),
            check=True,
        )
    if not skip_train:
        command = [sys.executable, str(PREDICTION_SCRIPT), "--all"]
        subprocess.run(
            command,
            cwd=str(BACKEND_ROOT),
            check=True,
        )

    settings = _load_settings()
    groups = _load_groups()
    _validate_group_audits(
        settings.data_dir,
        groups,
        allow_failed=allow_failed_audit,
    )
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
        timespec="seconds"
    )
    report_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
        "%Y%m%d_%H%M%S"
    )
    package_root = (
        PROJECT_ROOT
        / "build"
        / "prediction_releases"
        / report_id
    )
    if package_root.exists():
        raise SystemExit(f"release directory already exists: {package_root}")
    reports_root = package_root / "reports"
    reports_root.mkdir(parents=True)
    manifest_groups = []
    as_of_dates = []
    for group in groups:
        group_id = str(group["id"])
        horizon = int(group["horizon_days"])
        source = (
            settings.data_dir
            / "ads"
            / "group_bagging"
            / group_id
            / f"horizon={horizon}"
            / "latest.json"
        )
        if not source.exists():
            raise SystemExit(f"missing prediction report: {source}")
        target = (
            reports_root
            / "group_bagging"
            / group_id
            / f"horizon={horizon}"
            / "latest.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        report = json.loads(target.read_text(encoding="utf-8"))
        as_of = str(report["metadata"]["as_of_date"])
        as_of_dates.append(as_of)
        manifest_group = {
            "id": group_id,
            "name": str(group.get("name", group_id)),
            "horizon_days": horizon,
            "as_of_date": as_of,
            "file": target.relative_to(package_root).as_posix(),
        }
        if group.get("require_audit"):
            audit_source = (
                settings.data_dir
                / "quality"
                / "group_audits"
                / group_id
                / f"horizon={horizon}"
                / "latest.json"
            )
            audit_target = (
                reports_root
                / "quality"
                / "group_audits"
                / group_id
                / f"horizon={horizon}"
                / "latest.json"
            )
            audit_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audit_source, audit_target)
            audit = json.loads(
                audit_target.read_text(encoding="utf-8")
            )
            manifest_group["audit_status"] = audit.get("status")
            manifest_group["audit_file"] = (
                audit_target.relative_to(package_root).as_posix()
            )
        manifest_groups.append(manifest_group)

    manifest = {
        "report_id": report_id,
        "generated_at": generated_at,
        "as_of_date": sorted(as_of_dates)[-1],
        "model_version": "grouped-gbm-bagging-v1",
        "groups": manifest_groups,
    }
    manifest_root = reports_root / "model_manifest"
    release_manifest = manifest_root / "releases" / f"{report_id}.json"
    release_manifest.parent.mkdir(parents=True, exist_ok=True)
    release_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (manifest_root / "latest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return package_root


def _validate_group_audits(
    data_dir: Path,
    groups: list[dict],
    *,
    allow_failed: bool,
) -> None:
    failures = []
    for group in groups:
        if not group.get("require_audit"):
            continue
        group_id = str(group["id"])
        horizon = int(group["horizon_days"])
        path = (
            data_dir
            / "quality"
            / "group_audits"
            / group_id
            / f"horizon={horizon}"
            / "latest.json"
        )
        if not path.exists():
            failures.append(f"{group_id}: audit report missing")
            continue
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("status") != "pass":
            checks = audit.get("checks", {})
            failed_checks = [
                name for name, passed in checks.items() if not passed
            ]
            failures.append(
                f"{group_id}: audit failed ({', '.join(failed_checks)})"
            )
    if failures and not allow_failed:
        raise SystemExit(
            "Prediction publication blocked:\n- "
            + "\n- ".join(failures)
            + "\nUse --allow-failed-audit only for non-production review."
        )


def _create_archive(package_root: Path) -> Path:
    archive = package_root / "prediction-reports.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(package_root / "reports", arcname="reports")
    return archive


def _upload_and_publish(
    archive: Path,
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    remote_root: str,
) -> None:
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=port,
            username=user,
            password=password,
            timeout=20,
            banner_timeout=20,
            auth_timeout=20,
            look_for_keys=False,
            allow_agent=False,
        )
        remote_incoming = posixpath.join(remote_root, "incoming")
        remote_target = posixpath.join(
            remote_incoming,
            archive.name,
        )
        sftp = client.open_sftp()
        try:
            try:
                sftp.stat(remote_root)
            except OSError as exc:
                raise RuntimeError(
                    "remote_data_root does not exist; update "
                    "backend/publish_config.json"
                ) from exc
            _mkdir_p(sftp, remote_incoming)
            sftp.put(str(archive), remote_target)
        finally:
            sftp.close()

        ads_dir = posixpath.join(remote_root, "ads")
        command = (
            f"set -eu; "
            f"mkdir -p {ads_dir}; "
            f"tmp=$(mktemp -d); "
            f"tar -xzf {remote_target} -C \"$tmp\"; "
            f"test -f \"$tmp/reports/model_manifest/latest.json\"; "
            f"cd \"$tmp/reports\"; "
            f"for f in $(find . -type f -name '*.json' "
            f"! -path './quality/*'); do "
            f"python3 -m json.tool \"$f\" >/dev/null; done; "
            f"for f in $(find . -type f -name '*.json'); do "
            f"mkdir -p \"{ads_dir}/$(dirname \"$f\")\"; "
            f"cp \"$tmp/reports/$f\" \"{ads_dir}/$f.tmp\"; "
            f"mv \"{ads_dir}/$f.tmp\" \"{ads_dir}/$f\"; done; "
            f"if [ -d \"$tmp/reports/quality\" ]; then "
            f"mkdir -p {remote_root}/quality; "
            f"cp -R \"$tmp/reports/quality/.\" {remote_root}/quality/; fi; "
            f"cd /; "
            f"rm -rf \"$tmp\"; "
            f"rm -f {remote_target}"
        )
        stdin, stdout, stderr = client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        output = stderr.read().decode("utf-8", errors="replace")
        if exit_code != 0:
            raise RuntimeError(f"remote publish failed: {output}")
    finally:
        client.close()


def _mkdir_p(sftp, remote_path: str) -> None:
    parts = [part for part in remote_path.split("/") if part]
    current = "/" if remote_path.startswith("/") else ""
    for part in parts:
        current = posixpath.join(current, part)
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def _load_settings():
    sys.path.insert(0, str(BACKEND_ROOT))
    from app.config import get_settings

    return get_settings()


def _load_groups() -> list[dict]:
    sys.path.insert(0, str(BACKEND_ROOT))
    from app.services.group_bagging import load_group_configs

    return load_group_configs(
        BACKEND_ROOT / "config" / "model_groups.json"
    )


def _load_publish_config() -> dict:
    if not PUBLISH_CONFIG.exists():
        return {}
    return json.loads(PUBLISH_CONFIG.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
