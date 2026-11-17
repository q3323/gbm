from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from app.cleaning import DAILY_KEY_COLUMNS, clean_daily_bars
from app.config import Settings
from app.models import FinancialReport
from app.quality import daily_quality_report
from app.quality import financial_quality_report
from app.symbols import normalize_stock_code


def _normalize_query_date(value: str) -> str:
    text = value.strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return text


@dataclass(frozen=True)
class StoredDailyResult:
    raw_report: dict[str, Any]
    clean_report: dict[str, Any]
    ods_path: Path
    dwd_path: Path
    dws_path: Path


class ParquetStore:
    def __init__(self, settings: Settings) -> None:
        self.data_dir = settings.data_dir

    def save_ods_daily(
        self,
        code: str,
        source: str,
        frame: pl.DataFrame,
    ) -> Path:
        normalized_code = normalize_stock_code(code)
        path = (
            self.data_dir
            / "ods"
            / "daily_price"
            / f"source={source}"
            / f"stock_code={normalized_code}"
            / "bars.parquet"
        )
        merged = self._merge_existing(
            path,
            frame,
            keys=DAILY_KEY_COLUMNS,
        )
        self._write_parquet_atomic(path, merged)
        return path

    def ingest_daily(
        self,
        code: str,
        source: str,
        frame: pl.DataFrame,
    ) -> StoredDailyResult:
        normalized_code = normalize_stock_code(code)
        raw_report = daily_quality_report(frame)
        ods_path = self.save_ods_daily(normalized_code, source, frame)

        cleaned = clean_daily_bars(frame)
        dwd_path = (
            self.data_dir
            / "dwd"
            / "daily_price"
            / f"stock_code={normalized_code}"
            / "bars.parquet"
        )
        merged = self._merge_existing(
            dwd_path,
            cleaned,
            keys=DAILY_KEY_COLUMNS,
        )
        self._write_parquet_atomic(dwd_path, merged)

        clean_report = daily_quality_report(merged)
        self.save_quality_report(normalized_code, raw_report, "raw")
        self.save_quality_report(normalized_code, clean_report, "clean")
        from app.indicators import compute_indicators

        indicators = compute_indicators(merged)
        dws_path = self.save_dws_indicators(normalized_code, indicators)
        return StoredDailyResult(
            raw_report=raw_report,
            clean_report=clean_report,
            ods_path=ods_path,
            dwd_path=dwd_path,
            dws_path=dws_path,
        )

    def save_dws_indicators(self, code: str, frame: pl.DataFrame) -> Path:
        normalized_code = normalize_stock_code(code)
        frame = _coerce_trade_date(frame)
        path = (
            self.data_dir
            / "dws"
            / "technical_indicators"
            / f"stock_code={normalized_code}"
            / "indicators.parquet"
        )
        merged = self._merge_existing(
            path,
            frame,
            keys=["stock_code", "trade_date", "adjust_type"],
        )
        self._write_parquet_atomic(path, merged)
        return path

    def load_dws_indicators(
        self,
        code: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        normalized_code = normalize_stock_code(code)
        path = (
            self.data_dir
            / "dws"
            / "technical_indicators"
            / f"stock_code={normalized_code}"
            / "indicators.parquet"
        )
        if not path.exists():
            return pl.DataFrame()
        frame = _coerce_trade_date(pl.read_parquet(path))
        if start_date:
            frame = frame.filter(
                pl.col("trade_date")
                >= pl.lit(_normalize_query_date(start_date)).str.strptime(
                    pl.Date,
                    "%Y-%m-%d",
                )
            )
        if end_date:
            frame = frame.filter(
                pl.col("trade_date")
                <= pl.lit(_normalize_query_date(end_date)).str.strptime(
                    pl.Date,
                    "%Y-%m-%d",
                )
            )
        return frame.sort("trade_date")

    def load_dwd_daily(
        self,
        code: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        normalized_code = normalize_stock_code(code)
        path = (
            self.data_dir
            / "dwd"
            / "daily_price"
            / f"stock_code={normalized_code}"
            / "bars.parquet"
        )
        if not path.exists():
            return pl.DataFrame()
        frame = pl.read_parquet(path)
        if start_date:
            frame = frame.filter(
                pl.col("trade_date")
                >= pl.lit(_normalize_query_date(start_date)).str.strptime(
                    pl.Date,
                    "%Y-%m-%d",
                )
            )
        if end_date:
            frame = frame.filter(
                pl.col("trade_date")
                <= pl.lit(_normalize_query_date(end_date)).str.strptime(
                    pl.Date,
                    "%Y-%m-%d",
                )
            )
        return frame.sort("trade_date")

    def load_all_dwd_daily(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for path in sorted(
            (self.data_dir / "dwd" / "daily_price").glob(
                "stock_code=*/bars.parquet"
            )
        ):
            frame = pl.read_parquet(path)
            if start_date:
                frame = frame.filter(
                    pl.col("trade_date")
                    >= pl.lit(_normalize_query_date(start_date)).str.strptime(
                        pl.Date,
                        "%Y-%m-%d",
                    )
                )
            if end_date:
                frame = frame.filter(
                    pl.col("trade_date")
                    <= pl.lit(_normalize_query_date(end_date)).str.strptime(
                        pl.Date,
                        "%Y-%m-%d",
                    )
                )
            if not frame.is_empty():
                frames.append(frame)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed").sort(
            ["stock_code", "trade_date"]
        )

    def load_all_financial_reports(self) -> pl.DataFrame:
        frames: list[pl.DataFrame] = []
        for path in sorted(
            (self.data_dir / "dwd" / "financial_reports").glob(
                "stock_code=*/reports.parquet"
            )
        ):
            frame = pl.read_parquet(path)
            if not frame.is_empty():
                frames.append(frame)
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed").sort(
            ["stock_code", "report_date"]
        )

    def save_quality_report(
        self,
        code: str,
        report: dict[str, Any],
        stage: str,
    ) -> Path:
        normalized_code = normalize_stock_code(code)
        report_dir = (
            self.data_dir
            / "quality"
            / "daily"
            / f"stock_code={normalized_code}"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"{stage}_latest.json"
        self._write_text_atomic(
            path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return path

    def load_quality_report(self, code: str, stage: str = "clean") -> dict[str, Any] | None:
        normalized_code = normalize_stock_code(code)
        path = (
            self.data_dir
            / "quality"
            / "daily"
            / f"stock_code={normalized_code}"
            / f"{stage}_latest.json"
        )
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def record_pipeline_run(self, record: dict[str, Any]) -> Path:
        path = self.data_dir / "quality" / "pipeline_runs" / "runs.parquet"
        frame = pl.DataFrame([record])
        merged = self._merge_existing(path, frame, keys=["run_id"])
        self._write_parquet_atomic(path, merged)
        return path

    def load_pipeline_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        path = self.data_dir / "quality" / "pipeline_runs" / "runs.parquet"
        if not path.exists():
            return []
        frame = pl.read_parquet(path)
        return frame.tail(limit).sort("started_at", descending=True).to_dicts()

    def save_source_comparison(
        self,
        code: str,
        adjust_type: str,
        rows: pl.DataFrame,
        report: dict[str, Any],
    ) -> Path:
        normalized_code = normalize_stock_code(code)
        report_dir = (
            self.data_dir
            / "quality"
            / "source_comparison"
            / f"stock_code={normalized_code}"
            / f"adjust_type={adjust_type}"
        )
        path = report_dir / "comparison.parquet"
        self._write_parquet_atomic(path, rows)
        self._write_text_atomic(
            report_dir / "latest.json",
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return path

    def save_analysis_report(
        self,
        code: str,
        report: dict[str, Any],
    ) -> Path:
        normalized_code = normalize_stock_code(code)
        path = (
            self.data_dir
            / "ads"
            / "time_series"
            / f"stock_code={normalized_code}"
            / "latest.json"
        )
        self._write_text_atomic(
            path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return path

    def save_gbm_features(self, code: str, frame: pl.DataFrame) -> Path:
        normalized_code = normalize_stock_code(code)
        path = (
            self.data_dir
            / "dws"
            / "gbm_features"
            / f"stock_code={normalized_code}"
            / "features.parquet"
        )
        self._write_parquet_atomic(path, frame)
        return path

    def save_gbm_report(self, code: str, report: dict[str, Any]) -> Path:
        normalized_code = normalize_stock_code(code)
        path = (
            self.data_dir
            / "ads"
            / "gbm"
            / f"stock_code={normalized_code}"
            / "latest.json"
        )
        self._write_text_atomic(
            path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return path

    def save_backtest_report(
        self,
        code: str,
        model: str,
        report: dict[str, Any],
    ) -> Path:
        normalized_code = normalize_stock_code(code)
        path = (
            self.data_dir
            / "ads"
            / "backtest"
            / model
            / f"stock_code={normalized_code}"
            / "latest.json"
        )
        self._write_text_atomic(
            path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return path

    def save_sensitivity_report(
        self,
        code: str,
        model: str,
        report: dict[str, Any],
    ) -> Path:
        normalized_code = normalize_stock_code(code)
        path = (
            self.data_dir
            / "ads"
            / "sensitivity"
            / model
            / f"stock_code={normalized_code}"
            / "latest.json"
        )
        self._write_text_atomic(
            path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return path

    def save_nested_tuning_result(
        self,
        code: str,
        model: str,
        report: dict[str, Any],
        out_of_sample: pl.DataFrame,
    ) -> Path:
        normalized_code = normalize_stock_code(code)
        directory = (
            self.data_dir
            / "ads"
            / "nested_tuning"
            / model
            / f"stock_code={normalized_code}"
        )
        report_path = directory / "latest.json"
        self._write_text_atomic(
            report_path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        self._write_parquet_atomic(
            directory / "out_of_sample.parquet",
            out_of_sample,
        )
        return report_path

    def save_nested_tuning_seed_result(
        self,
        code: str,
        model: str,
        horizon: int,
        seed: int,
        report: dict[str, Any],
        out_of_sample: pl.DataFrame,
    ) -> Path:
        normalized_code = normalize_stock_code(code)
        directory = (
            self.data_dir
            / "ads"
            / "nested_tuning"
            / model
            / f"stock_code={normalized_code}"
            / f"horizon={horizon}"
            / f"seed={seed}"
        )
        report_path = directory / "latest.json"
        self._write_text_atomic(
            report_path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        self._write_parquet_atomic(
            directory / "out_of_sample.parquet",
            out_of_sample,
        )
        return report_path

    def save_nested_tuning_summary(
        self,
        code: str,
        model: str,
        horizon: int,
        summary: dict[str, Any],
    ) -> Path:
        normalized_code = normalize_stock_code(code)
        path = (
            self.data_dir
            / "ads"
            / "nested_tuning"
            / model
            / f"stock_code={normalized_code}"
            / f"horizon={horizon}"
            / "summary.json"
        )
        self._write_text_atomic(
            path,
            json.dumps(summary, ensure_ascii=False, indent=2),
        )
        return path

    def save_ensemble_result(
        self,
        code: str,
        model: str,
        horizon: int,
        report: dict[str, Any],
        out_of_sample: pl.DataFrame,
    ) -> Path:
        normalized_code = normalize_stock_code(code)
        directory = (
            self.data_dir
            / "ads"
            / "ensemble"
            / model
            / f"stock_code={normalized_code}"
            / f"horizon={horizon}"
        )
        report_path = directory / "latest.json"
        self._write_text_atomic(
            report_path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        self._write_parquet_atomic(
            directory / "out_of_sample.parquet",
            out_of_sample,
        )
        return report_path

    def load_ensemble_result(
        self,
        code: str,
        model: str,
        horizon: int,
    ) -> dict[str, Any] | None:
        normalized_code = normalize_stock_code(code)
        path = (
            self.data_dir
            / "ads"
            / "ensemble"
            / model
            / f"stock_code={normalized_code}"
            / f"horizon={horizon}"
            / "latest.json"
        )
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_group_bagging_result(
        self,
        group_id: str,
        horizon: int,
        report: dict[str, Any],
    ) -> Path:
        path = (
            self.data_dir
            / "ads"
            / "group_bagging"
            / group_id
            / f"horizon={horizon}"
            / "latest.json"
        )
        self._write_text_atomic(
            path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return path

    def load_group_bagging_result(
        self,
        group_id: str,
        horizon: int,
    ) -> dict[str, Any] | None:
        path = (
            self.data_dir
            / "ads"
            / "group_bagging"
            / group_id
            / f"horizon={horizon}"
            / "latest.json"
        )
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_universe_sync_report(self, report: dict[str, Any]) -> Path:
        path = (
            self.data_dir
            / "quality"
            / "universe_sync"
            / "latest.json"
        )
        self._write_text_atomic(
            path,
            json.dumps(report, ensure_ascii=False, indent=2),
        )
        return path

    def save_financial_report(self, report: FinancialReport) -> Path:
        normalized_code = normalize_stock_code(report.stock_code)
        path = (
            self.data_dir
            / "dwd"
            / "financial_reports"
            / f"stock_code={normalized_code}"
            / "reports.parquet"
        )
        frame = pl.DataFrame([report.model_dump()])
        merged = self._merge_existing(
            path,
            frame,
            keys=["stock_code", "report_date", "source"],
        )
        self._write_parquet_atomic(path, merged)
        self.save_financial_quality_report(normalized_code, merged)
        return path

    def load_financial_reports(self, code: str) -> pl.DataFrame:
        normalized_code = normalize_stock_code(code)
        path = (
            self.data_dir
            / "dwd"
            / "financial_reports"
            / f"stock_code={normalized_code}"
            / "reports.parquet"
        )
        if not path.exists():
            return pl.DataFrame()
        return pl.read_parquet(path).sort("report_date", descending=True)

    def save_financial_quality_report(self, code: str, frame: pl.DataFrame) -> Path:
        normalized_code = normalize_stock_code(code)
        report_dir = (
            self.data_dir
            / "quality"
            / "financial"
            / f"stock_code={normalized_code}"
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "latest.json"
        self._write_text_atomic(
            path,
            json.dumps(
                financial_quality_report(frame),
                ensure_ascii=False,
                indent=2,
            ),
        )
        return path

    def _merge_existing(
        self,
        path: Path,
        frame: pl.DataFrame,
        keys: list[str],
    ) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        if path.exists():
            existing = pl.read_parquet(path)
            if "adjust_type" in keys:
                if "adjust_type" not in existing.columns:
                    existing = existing.with_columns(
                        pl.lit("qfq").alias("adjust_type")
                    )
                else:
                    existing = existing.with_columns(
                        pl.col("adjust_type").fill_null("qfq")
                    )
            frame = pl.concat([existing, frame], how="diagonal_relaxed")
        return frame.unique(subset=keys, keep="last", maintain_order=True)

    def _write_parquet_atomic(self, path: Path, frame: pl.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp.parquet")
        frame.write_parquet(temp)
        os.replace(temp, path)

    def _write_text_atomic(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(f"{path.suffix}.tmp")
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, path)


def new_run_record(
    *,
    run_id: str,
    code: str,
    source: str,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    row_count: int,
    error: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stock_code": normalize_stock_code(code),
        "source": source,
        "status": status,
        "started_at": started_at.astimezone(timezone.utc).replace(tzinfo=None),
        "finished_at": finished_at.astimezone(timezone.utc).replace(tzinfo=None),
        "duration_seconds": round(
            (finished_at - started_at).total_seconds(),
            3,
        ),
        "row_count": row_count,
        "error": error,
    }


def _coerce_trade_date(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "trade_date" not in frame.columns:
        return frame
    if frame.schema["trade_date"] == pl.Date:
        return frame
    text = pl.col("trade_date").cast(pl.String).str.slice(0, 10)
    normalized = pl.coalesce(
        [
            text.str.strptime(pl.Date, "%Y-%m-%d", strict=False),
            text.str.strptime(pl.Date, "%Y%m%d", strict=False),
        ]
    )
    return frame.with_columns(normalized.alias("trade_date"))
