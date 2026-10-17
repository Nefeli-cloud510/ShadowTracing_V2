from __future__ import annotations

import time
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from core.json_safe import json_dumps

from core.calendar_samples import (
    FrozenEvaluationPlan,
    SamplePlan,
    build_sample_plan,
    canonicalize_feature_lags,
    compute_default_test_start,
    freeze_evaluation_plan,
    is_blind_sealed,
    is_legacy_uniform_lags,
    load_or_freeze_evaluation_plan,
    make_calendar_split,
)
from core.experiment_tools.compiled_plan import CompiledExperimentPlan
from core.protocol_bridge import ElasticNetDataSourceConfig, ElasticNetProtocolBridge
from core.result_unified import (
    build_experiment_result,
    build_metric_delta,
    build_run_metrics,
    export_result_bundle,
)
from core.robustness_diagnostics import (
    empty_bootstrap_summary,
    paired_calendar_block_bootstrap,
    summary_to_robustness_report,
)
from core.unified_schema import (
    ExperimentContrast,
    DataSourceCoverage,
    ExperimentProtocol,
    ExperimentResult,
    ExperimentRun,
    ModelSpec,
    RunMetrics,
    RunSpec,
    ContrastSpec,
    DiagnosticSpec,
    VisualizationArtifact,
)
from models import elasticnet_runner as runner


def _normalize_model_family(name: str) -> str | None:
    normalized = str(name).strip().lower()
    if normalized == "enet":
        return "elasticnet"
    if normalized in {"ridge", "elasticnet"}:
        return normalized
    return None


def _canonical_feature_lags(
    protocol: ExperimentProtocol,
    feature_columns: list[str],
    past_lag_days: int,
) -> dict[str, list[int]]:
    """Resolve one authoritative sparse-lag map for the compiled plan.

    An empty protocol ``lags`` keeps the legacy uniform expansion so existing
    non-sparse protocols run unchanged; a non-empty declaration must cover
    every feature column actually trained by the plan.
    """

    columns = list(dict.fromkeys(str(column) for column in feature_columns))
    return canonicalize_feature_lags(
        protocol.features.lags or None,
        columns,
        past_lag_days=past_lag_days,
    )


class UnifiedExperimentHarness:
    """Type-safe harness that executes the real ElasticNet runner via unified schemas."""

    def __init__(
        self,
        *,
        data_source: ElasticNetDataSourceConfig,
        project_root: Path | None = None,
        run_shap: bool = True,
    ) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.latest_protocol_path = self.project_root / "config" / "latest_protocol.json"
        self.latest_metrics_path = self.project_root / "outputs" / "latest_metrics.json"
        self.evaluation_plan_path = self.project_root / "config" / "evaluation_plan.json"
        self.data_source = data_source
        self.bridge = ElasticNetProtocolBridge(data_source)
        self.run_shap = run_shap

    def run(self, protocol: ExperimentProtocol) -> ExperimentResult:
        model_family = str(protocol.model.name or "").strip().lower()
        normalized_family = _normalize_model_family(model_family)
        if normalized_family is None:
            raise ValueError(
                f"当前统一 harness 仅支持 Ridge / ElasticNet，收到: {protocol.model.name}"
            )
        if not protocol.features.treatment:
            raise ValueError("ExperimentProtocol.features.treatment 不能为空")
        is_baseline_single_arm = (
            "experiment_mode:baseline_single_arm" in (protocol.notes or [])
            or not protocol.features.control
        )
        if not is_baseline_single_arm and not protocol.features.control:
            raise ValueError("ExperimentProtocol.features.control 不能为空")

        baseline_features = self._resolve_feature_columns(
            protocol=protocol,
            label="baseline",
            fallback=protocol.features.treatment if is_baseline_single_arm else protocol.features.control,
        )
        treatment_features = self._resolve_feature_columns(
            protocol=protocol,
            label="treatment",
            fallback=protocol.features.treatment,
        )
        shared_overrides = self._resolve_shared_window_overrides(protocol)
        parameters = dict(protocol.model.parameters)
        parameters.update(shared_overrides or {})
        parameters.setdefault("window_size", 3)
        parameters.setdefault(
            "test_split_ratio",
            self.data_source.test_split_ratio,
        )
        forecast_horizon_days = int(
            self.bridge._resolve_forecast_horizon_days(
                protocol=protocol,
                overrides=shared_overrides or {},
                model_parameters=parameters,
            )
        )
        window_size = int(parameters.get("window_size", 3))
        past_lag_days = parameters.get("past_lag_days") or window_size
        time_parameters = {
            "forecast_horizon_days": forecast_horizon_days,
            "past_lag_days": int(past_lag_days),
            "window_size": window_size,
            "test_split_ratio": float(
                parameters.get(
                    "test_split_ratio",
                    self.data_source.test_split_ratio,
                )
            ),
        }
        feature_lags = _canonical_feature_lags(
            protocol,
            [*baseline_features, *treatment_features],
            int(past_lag_days),
        )
        runs = [
            RunSpec(
                run_id="baseline",
                label="baseline",
                feature_columns=baseline_features,
                model_family=normalized_family,
                parameters=dict(parameters),
                retrain=True,
            )
        ]
        contrasts: list[ContrastSpec] = []
        if not is_baseline_single_arm:
            runs.append(
                RunSpec(
                    run_id="treatment",
                    label="treatment",
                    feature_columns=treatment_features,
                    model_family=normalized_family,
                    parameters=dict(parameters),
                    retrain=True,
                )
            )
            contrasts.append(
                ContrastSpec(
                    contrast_id="baseline_vs_treatment",
                    name="对照组 / 实验组主比较",
                    kind="metric_delta",
                    run_ids=["baseline", "treatment"],
                    direction="positive",
                    note="默认双臂主比较",
                )
            )
        compiled_plan = CompiledExperimentPlan(
            experiment_type=protocol.experiment_type,
            target=protocol.target,
            experiment_id=protocol.experiment_id,
            round_id=protocol.round_id,
            model_parameters=dict(parameters),
            time_parameters=time_parameters,
            feature_lags=feature_lags,
            runs=runs,
            contrasts=contrasts,
            diagnostics=[],
            alignment=(
                protocol.alignment.model_copy(deep=True)
                if protocol.alignment is not None
                else None
            ),
        )
        return self.run_compiled(protocol, compiled_plan)

    def _protocol_with_parameters(
        self,
        protocol: ExperimentProtocol,
        parameters: dict,
    ) -> ExperimentProtocol:
        """Copy a protocol with canonical compiled-plan parameters.

        Legacy duplicate configuration that validation already checked
        (feature_groups / arm_time_windows / model_families) is removed so the
        executor cannot silently fall back to a different feature or family
        than the compiled plan declared.
        """

        copied = protocol.model_copy(deep=True)
        merged = dict(parameters)
        merged.pop("feature_groups", None)
        merged.pop("arm_time_windows", None)
        merged.pop("model_families", None)
        family = _normalize_model_family(
            str(merged.get("model_family") or copied.model.name or "")
        )
        if family is None:
            raise ValueError(
                f"编译计划必须显式携带 Ridge / ElasticNet，收到: {merged.get('model_family')}"
            )
        merged["model_family"] = family
        copied.model = ModelSpec(name=family, parameters=merged)
        return copied

    def run_batch(
        self,
        protocol: ExperimentProtocol,
        *,
        run_specs: list,
        contrasts_spec: list,
        diagnostics_spec: list | None = None,
    ) -> ExperimentResult:
        """Execute a validated multi-run protocol on one shared calendar sample."""

        run_models = [
            RunSpec.model_validate(item.model_dump())
            for item in run_specs
        ]
        batch_time_parameters = {
            key: protocol.model.parameters[key]
            for key in (
                "forecast_horizon_days",
                "past_lag_days",
                "window_size",
                "test_split_ratio",
            )
            if protocol.model.parameters.get(key) is not None
        }
        feature_lags = _canonical_feature_lags(
            protocol,
            [
                column
                for run in run_models
                for column in run.feature_columns
            ],
            int(
                batch_time_parameters.get("past_lag_days")
                or protocol.model.parameters.get("window_size", 3)
            ),
        )
        plan = CompiledExperimentPlan(
            experiment_type=protocol.experiment_type,
            target=protocol.target,
            experiment_id=protocol.experiment_id,
            round_id=protocol.round_id,
            model_parameters=dict(protocol.model.parameters),
            time_parameters=batch_time_parameters,
            feature_lags=feature_lags,
            runs=run_models,
            contrasts=[
                ContrastSpec.model_validate(item.model_dump())
                for item in contrasts_spec
            ],
            diagnostics=[
                DiagnosticSpec.model_validate(item.model_dump())
                for item in (diagnostics_spec or [])
            ],
            alignment=(
                protocol.alignment.model_copy(deep=True)
                if protocol.alignment is not None
                else None
            ),
        )
        return self.run_compiled(protocol, plan)

    def run_compiled(
        self,
        protocol: ExperimentProtocol,
        plan: CompiledExperimentPlan,
    ) -> ExperimentResult:
        """Execute a validated protocol strictly from its compiled plan.

        The plan is the only authoritative source for time windows, per-run
        model families and feature columns. Legacy duplicate configuration
        such as ``feature_groups`` / ``arm_time_windows`` is deliberately
        removed before the runner is invoked.
        """

        if not plan.runs:
            raise ValueError("编译计划至少需要一个 run，禁止空执行。")
        run_specs = plan.runs

        canonical_parameters = dict(plan.model_parameters)
        for key, value in plan.time_parameters.items():
            if value is not None:
                canonical_parameters[key] = value
        canonical_parameters.pop("feature_groups", None)
        canonical_parameters.pop("arm_time_windows", None)
        canonical_parameters.pop("model_families", None)

        shared_features = list(
            dict.fromkeys(
                column
                for spec in run_specs
                for column in spec.feature_columns
            )
        )
        if not shared_features:
            raise ValueError("编译计划的特征列为空，无法加载数据。")

        plan = CompiledExperimentPlan(
            experiment_type=protocol.experiment_type,
            target=protocol.target,
            experiment_id=protocol.experiment_id,
            round_id=protocol.round_id,
            model_parameters=canonical_parameters,
            time_parameters=dict(plan.time_parameters),
            feature_lags=dict(plan.feature_lags),
            runs=run_specs,
            contrasts=plan.contrasts,
            diagnostics=plan.diagnostics,
            alignment=(
                plan.alignment.model_copy(deep=True)
                if plan.alignment is not None
                else None
            ),
        )
        shared_config = self.bridge.build_runner_config_from_plan(
            plan,
            run_specs[0],
            shared_features,
        )
        merged = runner.load_and_merge_data(shared_config)
        plan_lags = (
            dict(plan.feature_lags)
            or shared_config.time_window_config.feature_lags
            or None
        )
        if is_legacy_uniform_lags(
            plan_lags,
            shared_config.time_window_config.past_lag_days
            or shared_config.time_window_config.window_size,
        ):
            plan_lags = None
        sample_plan = build_sample_plan(
            merged.X,
            merged.y,
            merged.time,
            feature_columns=shared_features,
            target_name=protocol.target,
            past_lag_days=shared_config.time_window_config.past_lag_days
            or shared_config.time_window_config.window_size,
            forecast_horizon_days=shared_config.time_window_config.forecast_horizon_days,
            feature_lags=plan_lags,
        )
        retained_records = sample_plan.retained_records()
        if not retained_records:
            raise ValueError("共同日历样本为空，无法执行严格预测回测")
        evaluation_plan = self._resolve_evaluation_plan(
            retained_records=retained_records,
            test_ratio=shared_config.time_window_config.test_split_ratio,
        )
        shared_split = make_calendar_split(
            retained_records,
            test_ratio=evaluation_plan.test_ratio,
            n_internal_folds=evaluation_plan.n_internal_folds,
            test_start=(
                pd.Timestamp(evaluation_plan.test_start)
                if evaluation_plan.test_start
                else None
            ),
        )
        shared_keys = {record.compact_key for record in retained_records}

        round_dir = self.results_dir / f"round_{protocol.round_id:02d}"
        execution_dir = round_dir / protocol.experiment_id
        execution_dir.mkdir(parents=True, exist_ok=True)
        run_id = self._next_run_id(protocol, round_dir)

        started = time.perf_counter()
        outputs: dict[str, dict] = {}
        effective_parameters: dict[str, dict] = {}
        for spec in run_specs:
            family = _normalize_model_family(spec.model_family)
            if family is None:
                raise ValueError(
                    f"编译计划仅支持 Ridge / ElasticNet，收到: {spec.model_family}"
                )
            run_parameters = dict(canonical_parameters)
            run_parameters.update(dict(spec.parameters))
            run_parameters["model_family"] = family
            run_protocol = self._protocol_with_parameters(
                protocol,
                run_parameters,
            )
            outputs[spec.run_id] = self._execute_variant(
                protocol=run_protocol,
                feature_columns=list(spec.feature_columns),
                label=spec.label,
                output_dir=execution_dir / spec.run_id,
                shared_keys=shared_keys,
                shared_split=shared_split,
                pre_loaded=merged,
                plan=plan,
                run_spec=spec,
            )
            effective_parameters[spec.run_id] = run_parameters
        duration_seconds = time.perf_counter() - started

        predictions_dir = execution_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        prediction_paths: dict[str, str] = {}
        for spec in run_specs:
            prediction_paths[spec.run_id] = self._to_relative_path(
                self._write_prediction_csv(
                    predictions_dir
                    / f"{protocol.experiment_id}_{spec.run_id}.csv",
                    outputs[spec.run_id],
                )
            )

        visualizations: list[VisualizationArtifact] = []
        for spec in run_specs:
            output = outputs[spec.run_id]
            for name, path in output["figure_paths"].items():
                visualizations.append(
                    VisualizationArtifact(
                        name=f"{protocol.experiment_id}_{spec.run_id}_{name}",
                        path=self._to_relative_path(Path(path)),
                    )
                )
            for name, path in output["shap_paths"].items():
                visualizations.append(
                    VisualizationArtifact(
                        name=f"{protocol.experiment_id}_{spec.run_id}_shap_{name}",
                        path=self._to_relative_path(Path(path)),
                    )
                )

        coverage_entries: list[DataSourceCoverage] = []
        for spec in run_specs:
            for entry in outputs[spec.run_id].get("data_coverage", []):
                coverage_entries.append(DataSourceCoverage.model_validate(entry))

        contrasts: list[ExperimentContrast] = []
        robustness = None
        for contrast_spec in plan.contrasts:
            if len(contrast_spec.run_ids) != 2:
                raise ValueError(
                    f"contrast {contrast_spec.contrast_id} 目前只支持两个 run 之间的配对比较"
                )
            left = outputs[contrast_spec.run_ids[0]]
            right = outputs[contrast_spec.run_ids[1]]
            left_frame = self._prediction_frame(left)
            right_frame = self._prediction_frame(right)
            bootstrap_summary = paired_calendar_block_bootstrap(
                left_frame,
                right_frame,
                block_size=7,
                n_draws=4000,
                seed=20261008,
                min_blocks=2,
            )
            if robustness is None:
                robustness = summary_to_robustness_report(bootstrap_summary)
            contrasts.append(
                ExperimentContrast(
                    contrast_id=contrast_spec.contrast_id,
                    name=contrast_spec.name,
                    kind=contrast_spec.kind,
                    specification={
                        "run_ids": contrast_spec.run_ids,
                        "direction": contrast_spec.direction,
                    },
                    run_ids=list(contrast_spec.run_ids),
                    metrics_delta=build_metric_delta(
                        left["metrics"],
                        right["metrics"],
                    ),
                    bootstrap_ci=summary_to_robustness_report(
                        bootstrap_summary
                    ).bootstrap,
                    note=contrast_spec.note,
                )
            )

        runs: list[ExperimentRun] = []
        for spec in run_specs:
            output = outputs[spec.run_id]
            family = _normalize_model_family(spec.model_family) or spec.model_family
            runs.append(
                ExperimentRun(
                    run_id=spec.run_id,
                    name=spec.label,
                    prediction_artifact=prediction_paths[spec.run_id],
                    metrics=build_run_metrics(output["metrics"]),
                    model_family=family,
                    feature_labels=list(spec.feature_columns),
                    feature_columns=list(spec.feature_columns),
                    parameters=effective_parameters[spec.run_id],
                    n_train=int(len(output["y_train"])),
                    n_test=int(len(output["y_test"])),
                    training_mode="retrained" if spec.retrain else "reused",
                    protocol_version=protocol.protocol_version,
                )
            )

        result = build_experiment_result(
            protocol,
            runs=runs,
            visualizations=visualizations,
            duration_seconds=duration_seconds,
            data_coverage=coverage_entries,
            robustness=robustness,
            run_id=run_id,
            contrasts=contrasts,
            diagnostics=[],
        )
        export_result_bundle(
            result,
            protocol,
            round_dir=round_dir,
            latest_protocol_path=self.latest_protocol_path,
            latest_metrics_path=self.latest_metrics_path,
            write_latest=evaluation_plan.split_label != "final_test_blind",
        )
        self._write_batch_archive(
            protocol=protocol,
            run_id=run_id,
            round_dir=round_dir,
            result=result,
            outputs=outputs,
            run_specs=run_specs,
            shared_config=shared_config,
            plan=sample_plan,
            merged=merged,
            compiled_plan=plan,
            shared_split=shared_split,
            evaluation_plan=evaluation_plan,
        )
        return result

    def _resolve_evaluation_plan(
        self,
        *,
        retained_records: list,
        test_ratio: float,
    ) -> FrozenEvaluationPlan:
        """Load the frozen project partition, deriving and persisting the
        initial ``test_start`` only on the first candidate that runs."""

        plan = load_or_freeze_evaluation_plan(
            self.project_root,
            test_ratio=test_ratio,
            n_internal_folds=2,
            requested_split_label="dev_backtest",
        )
        if not plan.test_start:
            derived_start = compute_default_test_start(
                retained_records,
                test_ratio=plan.test_ratio,
                n_internal_folds=plan.n_internal_folds,
            )
            plan = freeze_evaluation_plan(
                self.project_root,
                plan.with_test_start(derived_start),
            )
        return plan

    def is_blind_sealed(self) -> bool:
        """Whether a sealed final-test plan exists on disk without re-resolving."""

        return is_blind_sealed(self.project_root)

    def _write_batch_archive(
        self,
        *,
        protocol: ExperimentProtocol,
        compiled_plan: CompiledExperimentPlan,
        run_id: str,
        round_dir: Path,
        result: ExperimentResult,
        outputs: dict[str, dict],
        run_specs: list,
        shared_config,
        plan,
        merged,
        shared_split,
        evaluation_plan: FrozenEvaluationPlan,
    ) -> None:
        """Write the complete audit archive for any executed experiment.

        All experiment types go through this single writer so the original
        approved protocol, the actual compiled plan, data provenance, sample
        manifest, per-run tuning record, predictions and diagnostics stay in
        one consistent archive layout.
        """

        archive_dir = round_dir / "archive" / run_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        (archive_dir / "evaluation_plan.json").write_text(
            json_dumps(evaluation_plan.to_dict(), indent=2),
            encoding="utf-8",
        )

        protocol_payload = protocol.model_dump(mode="json", exclude_none=True)
        (archive_dir / "experiment_protocol.json").write_text(
            json_dumps(protocol_payload, indent=2),
            encoding="utf-8",
        )
        (archive_dir / "compiled_plan.json").write_text(
            json_dumps(
                compiled_plan.to_dict(),
                indent=2,
            ),
            encoding="utf-8",
        )
        (archive_dir / "prediction_protocol.json").write_text(
            json_dumps(protocol_payload, indent=2),
            encoding="utf-8",
        )
        self._write_data_manifest(
            archive_dir,
            shared_config,
            merged,
            plan,
            evaluation_plan,
        )
        self._write_sample_manifest(
            archive_dir,
            plan,
            shared_split,
            evaluation_plan,
        )
        self._write_cv_results(
            archive_dir,
            shared_split,
            outputs,
        )
        self._write_predictions(
            archive_dir,
            run_specs,
            outputs,
        )
        self._write_metrics(
            archive_dir,
            result,
            evaluation_plan,
        )
        self._write_diagnostics(
            archive_dir,
            protocol,
            plan,
            shared_split,
            evaluation_plan,
            run_specs,
            outputs,
        )
        bootstrap_summary = None
        if result.robustness is not None and result.robustness.bootstrap is not None:
            bootstrap = result.robustness.bootstrap
            bootstrap_summary = {
                "status": bootstrap.status,
                "n_iterations": bootstrap.n_iterations,
                "block_size": bootstrap.block_size,
                "seed": bootstrap.seed,
                "n_nonempty_blocks": bootstrap.n_nonempty_blocks,
                "min_blocks": bootstrap.min_blocks,
                "insufficient_blocks": bootstrap.insufficient_blocks,
            }
        self._write_run_notes(
            archive_dir,
            protocol,
            shared_config,
            plan,
            bootstrap_summary,
            run_id,
            evaluation_plan,
        )
        for spec in run_specs:
            run_archive_dir = archive_dir / spec.run_id
            run_archive_dir.mkdir(parents=True, exist_ok=True)
            output = outputs[spec.run_id]
            (run_archive_dir / "config.json").write_text(
                json_dumps(output.get("config", {}), indent=2),
                encoding="utf-8",
            )
            coefficients = output.get("coefficients")
            if coefficients is not None and not coefficients.empty:
                coefficients.to_csv(
                    run_archive_dir / "coefficients.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            search_rows = output.get("param_search_rows") or []
            if search_rows:
                pd.DataFrame([dict(row) for row in search_rows]).to_csv(
                    run_archive_dir / "param_search.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

        latest_entry = {
            "run_id": run_id,
            "archived_at": pd.Timestamp.now().isoformat(),
            "experiment_protocol.json": "experiment_protocol.json",
            "compiled_plan.json": "compiled_plan.json",
            "data_manifest.json": "data_manifest.json",
            "sample_manifest.csv": "sample_manifest.csv",
            "cv_results.csv": "cv_results.csv",
            "predictions.csv": "predictions.csv",
            "metrics.json": "metrics.json",
            "diagnostics.json": "diagnostics.json",
            "run_notes.json": "run_notes.json",
            "prediction_protocol.json": "prediction_protocol.json",
        }
        (round_dir / "archive" / "latest_run.json").write_text(
            json_dumps(latest_entry, indent=2),
            encoding="utf-8",
        )
        (round_dir / "latest_run_id.txt").write_text(run_id, encoding="utf-8")

    @staticmethod
    def _write_cv_results(
        archive_dir: Path,
        shared_split,
        outputs: dict[str, dict],
    ) -> None:
        rows: list[dict] = []
        split_dict = shared_split.to_dict()
        for run_label, output in outputs.items():
            fold_rows = []
            for fold in split_dict["cv_folds"]:
                fold_rows.append(
                    {
                        "fold": fold["fold"],
                        "train_count": fold["train_count"],
                        "eval_count": fold["eval_count"],
                        "train_targets_strictly_before_eval_issue": fold[
                            "train_targets_strictly_before_eval_issue"
                        ],
                        "run_id": run_label,
                        "selected_alpha": output.get("selected_alpha"),
                        "selected_l1_ratio": output.get("selected_l1_ratio"),
                        "search_mode": (output.get("param_search") or {}).get("search_mode"),
                        "converged": (output.get("param_search") or {}).get("converged"),
                        "search_completed": (output.get("param_search") or {}).get(
                            "search_completed"
                        ),
                        "boundary_hit": (output.get("param_search") or {}).get(
                            "boundary_hit"
                        ),
                        "solver_converged": (output.get("param_search") or {}).get(
                            "solver_converged"
                        ),
                    }
                )
            rows.extend(fold_rows)
            search_rows = output.get("param_search_rows") or []
            for index, row in enumerate(search_rows):
                merged = dict(row)
                merged["run_id"] = run_label
                merged["order"] = index
                merged["row_type"] = "param_search"
                merged["search_completed"] = (output.get("param_search") or {}).get(
                    "search_completed"
                )
                merged["boundary_hit"] = (output.get("param_search") or {}).get(
                    "boundary_hit"
                )
                merged["solver_converged"] = (output.get("param_search") or {}).get(
                    "solver_converged"
                )
                for fold in split_dict["cv_folds"]:
                    merged[f"fold_{fold['fold']}_train_count"] = fold["train_count"]
                    merged[f"fold_{fold['fold']}_eval_count"] = fold["eval_count"]
                rows.append(merged)
        if not rows:
            return
        pd.DataFrame(rows).to_csv(
            archive_dir / "cv_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

    @staticmethod
    def _write_predictions(
        archive_dir: Path,
        run_specs: list,
        outputs: dict[str, dict],
    ) -> None:
        frames = [
            UnifiedExperimentHarness._prediction_frame_with_arm(
                spec.run_id,
                outputs[spec.run_id],
            )
            for spec in run_specs
        ]
        pd.concat(frames, ignore_index=True).to_csv(
            archive_dir / "predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )

    @staticmethod
    def _write_metrics(
        archive_dir: Path,
        result: ExperimentResult,
        evaluation_plan: FrozenEvaluationPlan,
    ) -> None:
        payload = {
            "run_id": result.run_id,
            "experiment_id": result.experiment_id,
            "round_id": result.round_id,
            "status": result.status,
            "experiment_type": result.experiment_type,
            "split_label": evaluation_plan.split_label,
            "test_start": evaluation_plan.test_start,
            "evaluation_plan": evaluation_plan.to_dict(),
            "runs": [
                run.model_dump(mode="json", exclude_none=True)
                for run in result.runs
            ],
            "comparison": (
                result.comparison.model_dump(exclude_none=True)
                if result.comparison is not None
                else None
            ),
            "contrasts": [
                contrast.model_dump(mode="json", exclude_none=True)
                for contrast in result.contrasts
            ],
            "diagnostics": [
                item.model_dump(mode="json", exclude_none=True)
                for item in result.diagnostics
            ],
            "bootstrap": (
                result.robustness.model_dump(mode="json", exclude_none=True)
                if result.robustness is not None
                else None
            ),
        }
        if len(result.runs) == 2:
            payload["baseline"] = result.runs[0].metrics.model_dump(exclude_none=True)
            payload["treatment"] = result.runs[1].metrics.model_dump(exclude_none=True)
        (archive_dir / "metrics.json").write_text(
            json_dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _write_diagnostics(
        self,
        archive_dir: Path,
        protocol: ExperimentProtocol,
        plan: SamplePlan,
        shared_split,
        evaluation_plan: FrozenEvaluationPlan,
        run_specs: list,
        outputs: dict[str, dict],
    ) -> None:
        arms: dict[str, dict] = {}
        for spec in run_specs:
            output = outputs[spec.run_id]
            arms[spec.run_id] = {
                "label": output.get("label") or spec.label or spec.run_id,
                "n_train": int(len(output["dataset_split"].y_train)),
                "n_test": int(len(output["dataset_split"].y_test)),
                "sliding_window_total": int(output["sliding_window_total"]),
                "sliding_window_dropped": int(output["sliding_window_dropped"]),
                "common_sample_excluded": int(output["common_sample_excluded"]),
                "selected_alpha": output.get("selected_alpha"),
                "selected_l1_ratio": output.get("selected_l1_ratio"),
                "param_search": output.get("param_search"),
                "feature_columns": output["config"]["file_config"]["feature_columns"],
            }
        diagnostics = {
            "run_id": protocol.experiment_id,
            "experiment_id": protocol.experiment_id,
            "round_id": protocol.round_id,
            "sample_plan": plan.stats(),
            "calendar_split": shared_split.to_dict(),
            "evaluation_plan": evaluation_plan.to_dict(),
            "arms": arms,
        }
        (archive_dir / "diagnostics.json").write_text(
            json_dumps(diagnostics, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _resolve_feature_columns(
        *,
        protocol: ExperimentProtocol,
        label: str,
        fallback: list[str],
    ) -> list[str]:
        feature_groups = protocol.model.parameters.get("feature_groups")
        if isinstance(feature_groups, dict):
            configured = feature_groups.get(label)
            if isinstance(configured, list):
                normalized = [str(item).strip() for item in configured if str(item).strip()]
                if normalized:
                    return normalized
        return [str(item).strip() for item in fallback if str(item).strip()]

    @staticmethod
    def _resolve_shared_window_overrides(protocol: ExperimentProtocol) -> dict | None:
        arm_windows = protocol.model.parameters.get("arm_time_windows")
        if not isinstance(arm_windows, dict):
            return None
        labels = ("baseline", "treatment")
        values: list[tuple[int | None, int | None]] = []
        for label in labels:
            override = arm_windows.get(label)
            if not isinstance(override, dict):
                continue
            values.append(
                (
                    override.get("forecast_horizon_days"),
                    override.get("past_lag_days"),
                )
            )
        if len(values) == 2:
            first, second = values
            if first != second:
                raise ValueError(
                    "双臂窗口覆盖必须一致：forecast_horizon_days 与 past_lag_days "
                    "需在共同样本上成对比较"
                )
        return dict(arm_windows.get("baseline") or arm_windows.get("treatment") or {})

    def _execute_variant(
        self,
        *,
        protocol: ExperimentProtocol,
        feature_columns: list[str],
        label: str,
        output_dir: Path,
        shared_keys: set[tuple[str, str, int]],
        shared_split,
        pre_loaded=None,
        shared_overrides: dict | None = None,
        plan=None,
        run_spec=None,
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        if plan is not None and run_spec is not None:
            runner_config = self.bridge.build_runner_config_from_plan(
                plan,
                run_spec,
                feature_columns,
            )
        else:
            runner_config = self.bridge.build_runner_config(
                protocol,
                feature_columns,
                arm_overrides=shared_overrides,
            )

        merged = pre_loaded or runner.load_and_merge_data(runner_config)
        window_result = runner.build_sliding_window(
            merged.X,
            merged.y,
            merged.time,
            runner_config.time_window_config,
            only_keys=shared_keys,
            feature_columns=feature_columns,
        )
        split = runner.split_dataset_by_calendar(
            window_result.X_all,
            window_result.y_all,
            window_result.time_all,
            window_result.origins_all,
            window_result.sample_ids_all,
            window_result.records_all,
            runner_config.time_window_config.test_split_ratio,
            window_result.feature_names,
            calendar_split=shared_split,
        )

        model, param_search = runner.train_elasticnet_model_with_search(
            split.X_train,
            split.y_train,
            runner_config.elasticnet_config,
            cv_folds=split.cv_folds,
        )
        metrics, ytr_pred, yte_pred = runner.evaluate_model(
            model,
            split.X_train,
            split.X_test,
            split.y_train,
            split.y_test,
        )
        selected_alpha = runner.resolve_alpha_for_model(
            model,
            runner_config.elasticnet_config.alpha,
        )
        coef_df = runner.make_coefficient_table(model, window_result.feature_names)
        coef_df.to_csv(output_dir / f"{label}_coefficients.csv", index=False, encoding="utf-8-sig")

        sparse_labels = set(merged.active_sparse_sources)
        coverage_entries: list[dict] = []
        if sparse_labels:
            for cov in merged.coverage:
                if cov.source not in sparse_labels:
                    continue
                entry = cov.to_dict()
                entry["run_id"] = label
                entry["dropped_gap_windows"] = 0
                entry["note"] = "缺失记录保留至日历样本层，按历史特征与目标可用性判定保留或排除"
                coverage_entries.append(entry)

        full_time = np.concatenate([split.time_train, split.time_test])
        x_limits = (full_time.min(), full_time.max()) if len(full_time) > 0 else None
        split_time = split.time_test[0] if len(split.time_test) > 0 else None
        chart_label = f"{protocol.experiment_id} / {label}"
        chart_model_family = runner_config.elasticnet_config.model_family
        figure_paths = {
            "coefficients": runner.plot_feature_coefficients(
                coef_df,
                output_dir,
                model_family=chart_model_family,
            ),
            "scatter": runner.plot_prediction_scatter(
                split.y_test,
                yte_pred,
                metrics,
                output_dir,
                experiment_label=chart_label,
                train_true=split.y_train,
                train_pred=ytr_pred,
                model_family=chart_model_family,
            ),
            "timeseries": runner.plot_prediction_series(
                split.time_test,
                split.y_test,
                yte_pred,
                output_dir,
                experiment_label=chart_label,
                time_train=split.time_train,
                train_true=split.y_train,
                train_pred=ytr_pred,
                x_limits=x_limits,
                split_time=split_time,
                model_family=chart_model_family,
            ),
            "residual_scatter": runner.plot_residual_scatter(
                yte_pred,
                split.y_test - yte_pred,
                output_dir,
                model_family=chart_model_family,
            ),
            "residual_hist": runner.plot_residual_histogram(
                split.y_test - yte_pred,
                output_dir,
                model_family=chart_model_family,
            ),
        }

        shap_paths: dict[str, str] = {}
        shap_importance = None
        if self.run_shap:
            shap_result = runner.run_shap_analysis(
                model,
                split.X_train,
                split.X_test,
                output_dir,
                model_family=chart_model_family,
            )
            shap_importance = shap_result["importance"]
            shap_paths = {
                "summary": str(shap_result["summary_path"]),
                "bar": str(shap_result["bar_path"]),
                "waterfall": str(shap_result["waterfall_path"]),
                "importance_csv": str(shap_result["importance_path"]),
                "variable_importance_csv": str(shap_result["variable_importance_path"]),
                "lag_profile_csv": str(shap_result["lag_profile_path"]),
            }

        return {
            "label": label,
            "config": runner_config.model_dump(mode="json"),
            "metrics": metrics,
            "coefficients": coef_df,
            "model": model,
            "selected_alpha": selected_alpha,
            "selected_l1_ratio": param_search.selected_l1_ratio,
            "param_search": param_search.to_dict(),
            "param_search_rows": param_search.rows,
            "time_train": split.time_train,
            "time_test": split.time_test,
            "origin_test": split.origin_test,
            "sample_ids_test": split.sample_ids_test,
            "y_train": split.y_train,
            "y_test": split.y_test,
            "y_train_pred": ytr_pred,
            "y_test_pred": yte_pred,
            "figure_paths": {name: str(path) for name, path in figure_paths.items()},
            "shap_paths": shap_paths,
            "shap_importance": shap_importance,
            "data_coverage": coverage_entries,
            "sliding_window_total": window_result.total_windows,
            "sliding_window_dropped": window_result.dropped_gap_windows,
            "common_sample_excluded": window_result.common_sample_excluded,
            "sample_plan": window_result.sample_plan,
            "dataset_split": split,
        }

    @staticmethod
    def _assert_common_samples(baseline_output: dict, treatment_output: dict) -> None:
        checks = (
            (
                "sample_id",
                baseline_output["sample_ids_test"],
                treatment_output["sample_ids_test"],
            ),
            (
                "y_true",
                np.asarray(baseline_output["y_test"]),
                np.asarray(treatment_output["y_test"]),
            ),
            (
                "target_date",
                np.asarray(baseline_output["time_test"]),
                np.asarray(treatment_output["time_test"]),
            ),
        )
        for name, left, right in checks:
            if not np.array_equal(left, right):
                raise AssertionError(
                    f"双臂 {name} 不一致：公共样本语义被破坏，禁止比较增益"
                )

    @staticmethod
    def _prediction_frame(output: dict) -> pd.DataFrame:
        origins = pd.to_datetime(pd.Series(output["origin_test"])).dt.normalize()
        targets = pd.to_datetime(pd.Series(output["time_test"])).dt.normalize()
        return pd.DataFrame(
            {
                "sample_id": list(output["sample_ids_test"]),
                "issue_date": origins.dt.strftime("%Y-%m-%d"),
                "target_date": targets.dt.strftime("%Y-%m-%d"),
                "y_true": np.asarray(output["y_test"], dtype=float),
                "y_pred": np.asarray(output["y_test_pred"], dtype=float),
            }
        )

    @classmethod
    def _write_prediction_csv(cls, file_path: Path, execution_output: dict) -> Path:
        frame = cls._prediction_frame(execution_output)
        frame.to_csv(file_path, index=False, encoding="utf-8-sig")
        return file_path

    def _next_run_id(self, protocol: ExperimentProtocol, round_dir: Path) -> str:
        safe_experiment = "".join(
            character if character.isalnum() else "_" for character in protocol.experiment_id
        )
        archive_root = round_dir / "archive"
        existing: list[int] = []
        if archive_root.exists():
            for path in archive_root.iterdir():
                if not path.is_dir() or not path.name.startswith(f"{safe_experiment}_r"):
                    continue
                tail = path.name.rsplit("_r", 1)[-1]
                if tail.isdigit():
                    existing.append(int(tail))
        next_number = max(existing, default=0) + 1
        return f"{safe_experiment}_r{next_number}"

    def _write_data_manifest(
        self,
        archive_dir: Path,
        shared_config,
        merged,
        plan: SamplePlan,
        evaluation_plan: FrozenEvaluationPlan,
    ) -> None:
        file_cfg = shared_config.file_config
        sources: list[dict] = []
        seen_paths: set[str] = set()
        for path in [file_cfg.omni_file_path, file_cfg.lhaaso_file_path, *file_cfg.extra_file_paths]:
            key = str(Path(path).resolve()).lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            sources.append(
                {
                    "path": self._to_relative_path(path),
                    "source": (
                        "omni" if Path(path) == Path(file_cfg.omni_file_path) else
                        "lhaaso" if Path(path) == Path(file_cfg.lhaaso_file_path) else
                        "extra"
                    ),
                }
            )
        manifest = {
            "data_version": plan.data_version,
            "evaluation_plan": evaluation_plan.to_dict(),
            "time_column": file_cfg.time_column,
            "target_column": file_cfg.target_column,
            "feature_columns_union": list(plan.feature_columns),
            "sources": sources,
            "merged_rows": int(len(merged.X)),
            "raw_missing_counts": merged.missing_counts,
            "merged_date_range": {
                "start": plan.date_start,
                "end": plan.date_end,
            },
            "coverage": [cov.to_dict() for cov in merged.coverage],
            "active_sparse_sources": list(merged.active_sparse_sources),
            "merged_sources": list(merged.merged_sources),
        }
        (archive_dir / "data_manifest.json").write_text(
            json_dumps(manifest, indent=2),
            encoding="utf-8",
        )

    def _write_sample_manifest(
        self,
        archive_dir: Path,
        plan: SamplePlan,
        shared_split,
        evaluation_plan: FrozenEvaluationPlan,
    ) -> None:
        split_map: dict[str, str] = {}
        for record in shared_split.train:
            split_map[record.sample_id] = "train"
        for record in shared_split.test:
            split_map[record.sample_id] = "test"
        for record in shared_split.boundary_excluded:
            split_map[record.sample_id] = "boundary_excluded"

        rows = []
        for record in plan.records:
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "issue_date": record.issue_date.date().isoformat(),
                    "target_date": record.target_date.date().isoformat(),
                    "horizon_days": record.horizon_days,
                    "split": split_map.get(record.sample_id, "excluded"),
                    "split_label": evaluation_plan.split_label,
                    "sample_condition": record.sample_condition,
                    "drop_reason": record.drop_reason or "",
                }
            )
        pd.DataFrame(rows, columns=[
            "sample_id",
            "issue_date",
            "target_date",
            "horizon_days",
            "split",
            "split_label",
            "sample_condition",
            "drop_reason",
        ]).to_csv(archive_dir / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    @staticmethod
    def _prediction_frame_with_arm(arm: str, output: dict) -> pd.DataFrame:
        frame = UnifiedExperimentHarness._prediction_frame(output)
        frame.insert(0, "arm", arm)
        return frame

    def _write_run_notes(
        self,
        archive_dir: Path,
        protocol: ExperimentProtocol,
        shared_config,
        plan: SamplePlan,
        bootstrap_summary: dict | None,
        run_id: str,
        evaluation_plan: FrozenEvaluationPlan,
    ) -> None:
        notes = {
            "run_id": run_id,
            "experiment_id": protocol.experiment_id,
            "round_id": protocol.round_id,
            "evaluation_plan": evaluation_plan.to_dict(),
            "time_definition": (
                "issue_date=dataset_id；特征日期=issue_date-滞后天数；"
                "目标日期=issue_date+forecast_horizon_days 日历天"
            ),
            "missing_value_policy": (
                "不做插补；双臂要求共同日历样本上目标与所有联合特征均非空，"
                "任一缺失的日期整样本剔除"
            ),
            "common_sample_policy": (
                "对照组与实验组在完全相同的 issue/target 日历样本上成对训练与测试"
            ),
            "evaluation_partition_policy": (
                f"split_label={evaluation_plan.split_label}；"
                f"test_start={evaluation_plan.test_start}；"
                "评估边界在项目首次执行时冻结，候选实验不得按各自有效样本重切。"
            ),
            "sign_convention": (
                "RMSE/MAE 正向 delta=实验组误差更小；Pearson r/R2 正向 delta=实验组相关更高"
            ),
            "model_family": shared_config.elasticnet_config.model_family,
            "forecast_horizon_days": shared_config.time_window_config.forecast_horizon_days,
            "past_lag_days": plan.past_lag_days,
            "alpha_grid": shared_config.elasticnet_config.alpha_grid,
            "l1_ratio_grid": shared_config.elasticnet_config.l1_ratio_grid,
            "param_search_mode": shared_config.elasticnet_config.param_search_mode,
            "bootstrap": {
                "status": (bootstrap_summary or {}).get("status"),
                "n_iterations": (bootstrap_summary or {}).get("n_iterations"),
                "block_size": (bootstrap_summary or {}).get("block_size"),
                "n_nonempty_blocks": (bootstrap_summary or {}).get(
                    "n_nonempty_blocks"
                ),
                "min_blocks": (bootstrap_summary or {}).get("min_blocks"),
                "insufficient_blocks": (bootstrap_summary or {}).get(
                    "insufficient_blocks"
                ),
                "boundary_note": (
                    "训练入口 bootstrap 固定使用 7 天块、4000 次抽样、"
                    "min_blocks=2；单独 robustness 诊断按 diagnostics_spec 配置执行。"
                ),
            },
            "verified_items": [
                "日期纪元按 dataset_id 解析，TIME 列未被使用",
                "重复日历日期会被拒绝",
                "训练目标日期严格早于测试段首个 issue_date，内部折同样满足",
                "选择 alpha 只在冻结的日历扩张折上完成",
                "两种模型族均通过统一桥接获得参数",
            ],
            "not_verified": [
                "D:/astro_rearrange 与 D:/helioprior 独立脚本与本仓库尚未做逐文件等价性验证",
            ],
            "user_notes": list(protocol.notes or []),
        }
        (archive_dir / "run_notes.json").write_text(
            json_dumps(notes, indent=2),
            encoding="utf-8",
        )

    def _collect_visualizations(
        self,
        protocol: ExperimentProtocol,
        baseline_output: dict,
        treatment_output: dict,
    ) -> list[VisualizationArtifact]:
        visualizations: list[VisualizationArtifact] = []

        for label, output in (("baseline", baseline_output), ("treatment", treatment_output)):
            for name, path in output["figure_paths"].items():
                visualizations.append(
                    VisualizationArtifact(
                        name=f"{protocol.experiment_id}_{label}_{name}",
                        path=self._to_relative_path(Path(path)),
                    )
                )
            for name, path in output["shap_paths"].items():
                visualizations.append(
                    VisualizationArtifact(
                        name=f"{protocol.experiment_id}_{label}_shap_{name}",
                        path=self._to_relative_path(Path(path)),
                    )
                )

        return visualizations

    def _to_relative_path(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.project_root.resolve())
            return relative.as_posix()
        except ValueError:
            return path.as_posix()
