from __future__ import annotations

import math
from pathlib import Path

from core.json_safe import json_dumps
from core.unified_schema import (
    DataSourceCoverage,
    ExecutionSummary,
    ExperimentContrast,
    ExperimentDiagnostic,
    ExperimentProtocol,
    ExperimentResult,
    ExperimentRun,
    MetricDelta,
    RunMetrics,
    RobustnessReport,
    VisualizationArtifact,
)


def build_run_metrics(raw_metrics: dict) -> RunMetrics:
    return RunMetrics(
        rmse=raw_metrics.get("test_rmse"),
        mae=raw_metrics.get("test_mae"),
        pearson_r=raw_metrics.get("test_pearson_r"),
        r2=raw_metrics.get("test_r2"),
        unavailable_reasons=dict(raw_metrics.get("unavailable_reasons") or {}),
    )


def build_metric_delta(baseline_metrics: dict, treatment_metrics: dict) -> MetricDelta:
    # 统一约定：正值表示实验组相对对照组“改善”。
    # RMSE/MAE 降低是增益，Pearson r / R2 升高是增益。
    return MetricDelta(
        rmse=_safe_reverse_delta(treatment_metrics.get("test_rmse"), baseline_metrics.get("test_rmse")),
        mae=_safe_reverse_delta(treatment_metrics.get("test_mae"), baseline_metrics.get("test_mae")),
        pearson_r=_safe_delta(treatment_metrics.get("test_pearson_r"), baseline_metrics.get("test_pearson_r")),
        r2=_safe_delta(treatment_metrics.get("test_r2"), baseline_metrics.get("test_r2")),
    )


def build_experiment_result(
    protocol: ExperimentProtocol,
    *,
    baseline_output: dict | None = None,
    treatment_output: dict | None = None,
    runs: list[ExperimentRun] | None = None,
    prediction_paths: dict[str, str | Path] | None = None,
    visualizations: list[VisualizationArtifact] | None = None,
    duration_seconds: float,
    data_coverage: list[DataSourceCoverage] | None = None,
    robustness: RobustnessReport | None = None,
    run_id: str | None = None,
    contrasts: list[ExperimentContrast] | None = None,
    diagnostics: list[ExperimentDiagnostic] | None = None,
    validity_status: str | None = None,
    warnings: list[str] | None = None,
) -> ExperimentResult:
    visualizations = visualizations or []
    if runs is None:
        if baseline_output is None or treatment_output is None:
            raise ValueError("未提供 runs 时，baseline_output 与 treatment_output 均不能为空")
        baseline_metrics = baseline_output["metrics"]
        treatment_metrics = treatment_output["metrics"]
        runs = [
            ExperimentRun(
                run_id="baseline",
                name="baseline",
                prediction_artifact=str((prediction_paths or {}).get("baseline") or ""),
                metrics=build_run_metrics(baseline_metrics),
                model_family=str(protocol.model.name or "ElasticNet"),
                feature_labels=list(protocol.features.control),
                feature_columns=list(protocol.features.control),
                parameters=dict(protocol.model.parameters),
                n_train=int(baseline_output.get("y_train", []).shape[0]) if len(baseline_output.get("y_train", [])) else None,
                n_test=int(baseline_output.get("y_test", []).shape[0]) if len(baseline_output.get("y_test", [])) else None,
                training_mode="retrained",
                protocol_version=protocol.protocol_version,
            ),
            ExperimentRun(
                run_id="treatment",
                name="treatment",
                prediction_artifact=str((prediction_paths or {}).get("treatment") or ""),
                metrics=build_run_metrics(treatment_metrics),
                model_family=str(protocol.model.name or "ElasticNet"),
                feature_labels=list(protocol.features.treatment),
                feature_columns=list(protocol.features.treatment),
                parameters=dict(protocol.model.parameters),
                n_train=int(treatment_output.get("y_train", []).shape[0]) if len(treatment_output.get("y_train", [])) else None,
                n_test=int(treatment_output.get("y_test", []).shape[0]) if len(treatment_output.get("y_test", [])) else None,
                training_mode="retrained",
                protocol_version=protocol.protocol_version,
            ),
        ]

    comparison = MetricDelta()
    if baseline_output is not None and treatment_output is not None:
        comparison = build_metric_delta(baseline_output["metrics"], treatment_output["metrics"])
    elif contrasts:
        comparison = contrasts[0].metrics_delta

    if contrasts is None:
        baseline_run = next((run for run in runs if run.run_id == "baseline"), None)
        treatment_run = next((run for run in runs if run.run_id == "treatment"), None)
        if baseline_run is not None and treatment_run is not None:
            contrasts = [
                ExperimentContrast(
                    contrast_id="baseline_vs_treatment",
                    name="对照组 / 实验组主比较",
                    kind="metric_delta",
                    run_ids=["baseline", "treatment"],
                    metrics_delta=comparison,
                    note="默认双臂主比较",
                )
            ]

    return ExperimentResult(
        experiment_id=protocol.experiment_id,
        round_id=protocol.round_id,
        status="completed",
        run_id=run_id,
        experiment_type=protocol.experiment_type,
        protocol_version=protocol.protocol_version,
        runs=runs,
        comparison=comparison,
        contrasts=contrasts or [],
        diagnostics=diagnostics or [],
        source_evidence_ids=list(protocol.source_evidence_ids),
        allowed_conclusions=list(protocol.allowed_conclusions),
        unresolved_questions=list(protocol.unresolved_questions),
        validity_status=validity_status or ("valid" if not protocol.warnings else "uncertain"),
        warnings=list(protocol.warnings) + (warnings or []),
        visualizations=visualizations,
        data_coverage=data_coverage or [],
        execution=ExecutionSummary(
            duration_seconds=duration_seconds,
            compute_seconds=duration_seconds,
            status="completed",
        ),
        robustness=robustness,
    )


def export_result_bundle(
    result: ExperimentResult,
    protocol: ExperimentProtocol,
    *,
    round_dir: Path,
    latest_protocol_path: Path,
    latest_metrics_path: Path,
    write_latest: bool = True,
) -> None:
    round_dir.mkdir(parents=True, exist_ok=True)
    latest_protocol_path.parent.mkdir(parents=True, exist_ok=True)
    latest_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    protocol_payload = protocol.model_dump(mode="json", exclude_none=True)
    result_payload = result.model_dump(mode="json", exclude_none=True)
    latest_metrics_payload = {
        "experiment_id": result.experiment_id,
        "round_id": result.round_id,
        "runs": [run.metrics.model_dump(exclude_none=True) for run in result.runs],
    }
    if result.comparison is not None:
        latest_metrics_payload["comparison"] = result.comparison.model_dump(
            exclude_none=True
        )
    if len(result.runs) == 2:
        latest_metrics_payload["baseline"] = result.runs[0].metrics.model_dump(
            exclude_none=True
        )
        latest_metrics_payload["treatment"] = result.runs[1].metrics.model_dump(
            exclude_none=True
        )

    (round_dir / "protocol.json").write_text(
        json_dumps(protocol_payload, indent=2),
        encoding="utf-8",
    )
    (round_dir / "result_unified.json").write_text(
        json_dumps(result_payload, indent=2),
        encoding="utf-8",
    )
    if write_latest:
        latest_protocol_path.write_text(
            json_dumps(protocol_payload, indent=2),
            encoding="utf-8",
        )
        latest_metrics_path.write_text(
            json_dumps(latest_metrics_payload, indent=2),
            encoding="utf-8",
        )


def _safe_delta(new_value: float | None, old_value: float | None) -> float | None:
    if new_value is None or old_value is None:
        return None
    try:
        new_float = float(new_value)
        old_float = float(old_value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(new_float) and math.isfinite(old_float)):
        return None
    return new_float - old_float


def _safe_reverse_delta(new_value: float | None, old_value: float | None) -> float | None:
    """误差类指标：正值 = 实验组误差更小（增益）。"""

    if new_value is None or old_value is None:
        return None
    try:
        new_float = float(new_value)
        old_float = float(old_value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(new_float) and math.isfinite(old_float)):
        return None
    return old_float - new_float
