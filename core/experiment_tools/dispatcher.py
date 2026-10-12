"""Dispatcher that maps a validated protocol to its registered executor."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.experiment_tools.registry import get_tool
from core.experiment_tools.validate_protocol import validate_experiment_protocol
from core.harness_unified import UnifiedExperimentHarness
from core.result_unified import build_metric_delta, build_run_metrics, export_result_bundle
from core.robustness_diagnostics import (
    paired_calendar_block_bootstrap,
    summary_to_robustness_report,
)
from core.unified_schema import (
    DataSourceCoverage,
    ExecutionSummary,
    ExperimentDiagnostic,
    ExperimentProtocol,
    ExperimentResult,
)


class ExperimentDispatcher:
    """Execute validated structured protocols through the shared harness."""

    def __init__(self, harness: UnifiedExperimentHarness) -> None:
        self.harness = harness
        self.project_root = harness.project_root

    def execute(self, protocol: ExperimentProtocol) -> ExperimentResult:
        validation = validate_experiment_protocol(protocol)
        tool = get_tool(protocol.experiment_type)
        if not validation.valid:
            return self._invalid_result(protocol, validation.reasons)
        if tool is None or not tool.available:
            return self._unsupported_result(protocol, validation)

        if protocol.experiment_type == "incremental_prediction":
            result = self.harness.run(protocol)
        elif protocol.experiment_type == "baseline_benchmark":
            result = self.harness.run(protocol)
        elif protocol.experiment_type == "model_sensitivity":
            result = self._run_model_sensitivity(protocol, validation)
        elif protocol.experiment_type == "robustness_diagnostic":
            result = self._run_robustness_diagnostic(protocol, validation)
        else:
            return self._unsupported_result(protocol, validation)
        return result

    def _run_model_sensitivity(self, protocol: ExperimentProtocol, validation) -> ExperimentResult:
        result = self.harness.run_batch(
            protocol,
            run_specs=validation.compiled_runs,
            contrasts_spec=validation.compiled_contrasts,
            diagnostics_spec=validation.compiled_diagnostics,
        )
        gain_rows = []
        for contrast in result.contrasts:
            gain_rows.append(
                {
                    "contrast_id": contrast.contrast_id,
                    "run_ids": contrast.run_ids,
                    "delta_rmse": contrast.metrics_delta.rmse,
                    "delta_pearson_r": contrast.metrics_delta.pearson_r,
                }
            )
        result.diagnostics.append(
            ExperimentDiagnostic(
                diagnostic_id="family_gain_summary",
                name="各模型家族内日影增量汇总",
                source_run_ids=[run.run_id for run in result.runs],
                structured={
                    "gains": gain_rows,
                    "rule": "只比较同一模型家族内部 baseline/treatment；跨家族不得错误归因。",
                },
                status="completed",
            )
        )
        round_dir = self.harness.results_dir / f"round_{protocol.round_id:02d}"
        export_result_bundle(
            result,
            protocol,
            round_dir=round_dir,
            latest_protocol_path=self.harness.latest_protocol_path,
            latest_metrics_path=self.harness.latest_metrics_path,
        )
        return result

    def _run_robustness_diagnostic(
        self,
        protocol: ExperimentProtocol,
        validation,
    ) -> ExperimentResult:
        params = protocol.model.parameters
        source_experiment_id = params.get("source_experiment_id")
        source_prediction_file = params.get("source_prediction_file")
        for run_spec in protocol.runs_spec:
            if run_spec.source_prediction_file:
                source_prediction_file = run_spec.source_prediction_file
                break
        base_frame, treat_frame = self._resolve_source_predictions(
            protocol=protocol,
            source_experiment_id=source_experiment_id,
            source_prediction_file=source_prediction_file,
        )
        started = time.perf_counter()
        bootstrap = paired_calendar_block_bootstrap(
            base_frame,
            treat_frame,
            block_size=7,
            n_draws=4000,
            seed=20261008,
        )
        robustness = summary_to_robustness_report(bootstrap)
        time_segments = self._time_segment_diagnostics(base_frame, treat_frame)
        residual_stats = self._residual_diagnostics(base_frame, treat_frame)
        duration_seconds = time.perf_counter() - started

        diagnostics = [
            ExperimentDiagnostic(
                diagnostic_id="paired_calendar_bootstrap",
                name="7 日配对日历块 bootstrap",
                source_run_ids=["baseline", "treatment"],
                structured=bootstrap,
                status="completed",
            ),
            ExperimentDiagnostic(
                diagnostic_id="time_segments",
                name="时间分段误差诊断",
                source_run_ids=["baseline", "treatment"],
                structured=time_segments,
                status="completed",
            ),
            ExperimentDiagnostic(
                diagnostic_id="residual_diagnostics",
                name="平均偏差与残差诊断",
                source_run_ids=["baseline", "treatment"],
                structured=residual_stats,
                status="completed",
            ),
        ]
        runs = [
            {
                "run_id": "baseline",
                "name": "对照组（复用预测）",
                "prediction_artifact": None,
                "metrics": _metrics_from_frame(base_frame),
                "model_family": str(protocol.model.name or "reused"),
                "feature_labels": [],
                "feature_columns": [],
                "parameters": {"source_experiment_id": source_experiment_id},
                "n_train": None,
                "n_test": int(len(base_frame)),
                "training_mode": "reused",
                "protocol_version": protocol.protocol_version,
            },
            {
                "run_id": "treatment",
                "name": "实验组（复用预测）",
                "prediction_artifact": None,
                "metrics": _metrics_from_frame(treat_frame),
                "model_family": str(protocol.model.name or "reused"),
                "feature_labels": [],
                "feature_columns": [],
                "parameters": {"source_experiment_id": source_experiment_id},
                "n_train": None,
                "n_test": int(len(treat_frame)),
                "training_mode": "reused",
                "protocol_version": protocol.protocol_version,
            },
        ]
        from core.unified_schema import ExperimentRun

        result = ExperimentResult(
            experiment_id=protocol.experiment_id,
            round_id=protocol.round_id,
            status="completed",
            experiment_type=protocol.experiment_type,
            protocol_version=protocol.protocol_version,
            runs=[ExperimentRun.model_validate(item) for item in runs],
            comparison=build_metric_delta(
                _metrics_from_frame(base_frame),
                _metrics_from_frame(treat_frame),
            ),
            diagnostics=diagnostics,
            source_evidence_ids=list(protocol.source_evidence_ids),
            allowed_conclusions=list(protocol.allowed_conclusions),
            unresolved_questions=list(protocol.unresolved_questions),
            validity_status=(
                "valid" if bool(bootstrap.get("stable_rmse")) else "uncertain"
            ),
            warnings=[
                "复用已有预测结果，未重新训练模型。",
                "不确定性的 bootstrap CI 跨越零，结论方向待科学评价。",
            ]
            if not bootstrap.get("stable_rmse")
            else ["复用已有预测结果，未重新训练模型。"],
            visualizations=[],
            data_coverage=[
                DataSourceCoverage(
                    source=f"source:{source_experiment_id}",
                    run_id="baseline",
                    observed_days=int(len(base_frame)),
                    note="复用已有预测文件。",
                )
            ],
            execution=ExecutionSummary(
                duration_seconds=duration_seconds,
                compute_seconds=duration_seconds,
                status="completed",
            ),
            robustness=robustness,
        )
        round_dir = self.harness.results_dir / f"round_{protocol.round_id:02d}"
        export_result_bundle(
            result,
            protocol,
            round_dir=round_dir,
            latest_protocol_path=self.harness.latest_protocol_path,
            latest_metrics_path=self.harness.latest_metrics_path,
        )
        return result

    def _resolve_source_predictions(
        self,
        *,
        protocol: ExperimentProtocol,
        source_experiment_id: str | None,
        source_prediction_file: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        round_dir = self.harness.results_dir / f"round_{protocol.round_id:02d}"
        candidates: list[Path] = []
        if source_prediction_file:
            path = Path(source_prediction_file)
            if not path.is_absolute():
                path = self.project_root / path
            candidates.append(path)
        if source_experiment_id:
            base_dir = round_dir / source_experiment_id / "predictions"
            candidates.append(base_dir / f"{source_experiment_id}_baseline.csv")
            candidates.append(base_dir / f"{source_experiment_id}_treatment.csv")
            candidates.append(round_dir / f"{source_experiment_id}_baseline.csv")
            candidates.append(round_dir / f"{source_experiment_id}_treatment.csv")
            archive_root = round_dir / "archive"
            if archive_root.exists():
                for archive in sorted(archive_root.iterdir()):
                    if not archive.is_dir():
                        continue
                    candidates.append(archive / "predictions.csv")
        existing = [path for path in candidates if path.exists()]
        if not existing:
            raise FileNotFoundError(
                f"robustness_diagnostic 找不到来源预测文件（source={source_experiment_id}, "
                f"file={source_prediction_file}）"
            )
        frames: list[pd.DataFrame] = []
        for path in existing:
            frame = pd.read_csv(path)
            frames.append(frame)
        if len(frames) == 1 and "arm" in frames[0].columns:
            combined = frames[0]
            base = combined[combined["arm"] == "baseline"].reset_index(drop=True)
            treatment = combined[combined["arm"] == "treatment"].reset_index(drop=True)
        elif len(frames) >= 2:
            base, treatment = frames[0], frames[1]
        else:
            raise FileNotFoundError("来源预测文件必须包含配对双臂或 arm 列。")

        def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
            normalized = frame[
                ["sample_id", "issue_date", "target_date", "y_true", "y_pred"]
            ].copy()
            normalized["sample_id"] = normalized["sample_id"].astype(str)
            return normalized.reset_index(drop=True)

        return _normalize(base), _normalize(treatment)

    @staticmethod
    def _time_segment_diagnostics(
        base_frame: pd.DataFrame,
        treat_frame: pd.DataFrame,
    ) -> dict:
        segments = []
        dates = pd.to_datetime(treat_frame["target_date"])
        labels = ["early", "middle", "late"]
        bounds = np.quantile(
            np.arange(len(dates)),
            [1 / 3, 2 / 3],
        ).astype(int)
        for label, start, end in [
            (labels[0], 0, bounds[0]),
            (labels[1], bounds[0], bounds[1]),
            (labels[2], bounds[1], len(dates)),
        ]:
            if end <= start:
                continue
            seg_base = base_frame.iloc[start:end]
            seg_treat = treat_frame.iloc[start:end]
            base_metrics = _metrics_from_frame(seg_base)
            treat_metrics = _metrics_from_frame(seg_treat)
            segments.append(
                {
                    "segment": label,
                    "n_samples": int(end - start),
                    "date_start": str(dates.iloc[start].date()),
                    "date_end": str(dates.iloc[end - 1].date()),
                    "baseline_rmse": base_metrics["rmse"],
                    "treatment_rmse": treat_metrics["rmse"],
                    "baseline_bias": _mean_bias(seg_base),
                    "treatment_bias": _mean_bias(seg_treat),
                }
            )
        return {"segments": segments, "note": "仅按日历顺序分段，不用于删除难预测日期。"}

    @staticmethod
    def _residual_diagnostics(
        base_frame: pd.DataFrame,
        treat_frame: pd.DataFrame,
    ) -> dict:
        base_metrics = _metrics_from_frame(base_frame)
        treat_metrics = _metrics_from_frame(treat_frame)
        return {
            "baseline": {
                **base_metrics,
                "mean_bias": _mean_bias(base_frame),
                "prediction_range": [
                    float(base_frame["y_pred"].min()),
                    float(base_frame["y_pred"].max()),
                ],
            },
            "treatment": {
                **treat_metrics,
                "mean_bias": _mean_bias(treat_frame),
                "prediction_range": [
                    float(treat_frame["y_pred"].min()),
                    float(treat_frame["y_pred"].max()),
                ],
            },
        }

    def _invalid_result(
        self,
        protocol: ExperimentProtocol,
        reasons: list[str],
    ) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=protocol.experiment_id,
            round_id=protocol.round_id,
            status="failed",
            experiment_type=protocol.experiment_type,
            protocol_version=protocol.protocol_version,
            validity_status="invalid",
            warnings=list(reasons),
            execution=ExecutionSummary(
                status="failed",
                message="协议结构校验失败；" + "；".join(reasons),
            ),
        )

    def _unsupported_result(
        self,
        protocol: ExperimentProtocol,
        validation,
    ) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=protocol.experiment_id,
            round_id=protocol.round_id,
            status="failed",
            experiment_type=protocol.experiment_type,
            protocol_version=protocol.protocol_version,
            validity_status="pending_scientific_review",
            warnings=[
                f"实验类型 {protocol.experiment_type} 尚未接通执行器，禁止静默降级。"
                + ("；".join(validation.warnings) if validation.warnings else "")
            ],
            execution=ExecutionSummary(
                status="failed",
                message=f"unsupported experiment_type={protocol.experiment_type}",
            ),
        )


def _metrics_from_frame(frame: pd.DataFrame) -> dict:
    y_true = frame["y_true"].astype(float).to_numpy()
    y_pred = frame["y_pred"].astype(float).to_numpy()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_r": pearson_r,
        "r2": r2,
    }


def _mean_bias(frame: pd.DataFrame) -> float:
    y_true = frame["y_true"].astype(float).to_numpy()
    y_pred = frame["y_pred"].astype(float).to_numpy()
    return float(np.mean(y_true - y_pred))
