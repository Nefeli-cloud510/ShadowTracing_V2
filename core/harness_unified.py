from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.calendar_samples import (
    SamplePlan,
    build_sample_plan,
    make_calendar_split,
)
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
        self.bridge = ElasticNetProtocolBridge(data_source)
        self.run_shap = run_shap

    def run(self, protocol: ExperimentProtocol) -> ExperimentResult:
        model_family = str(protocol.model.name or "").strip().lower()
        if model_family not in {"elasticnet", "ridge", "enet"}:
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
        shared_features = list(
            dict.fromkeys([*baseline_features, *treatment_features])
        )
        shared_overrides = self._resolve_shared_window_overrides(protocol)
        shared_config = self.bridge.build_runner_config(
            protocol,
            shared_features,
            arm_overrides=shared_overrides,
        )
        merged = runner.load_and_merge_data(shared_config)
        plan = build_sample_plan(
            merged.X,
            merged.y,
            merged.time,
            feature_columns=shared_features,
            target_name=protocol.target,
            past_lag_days=shared_config.time_window_config.past_lag_days
            or shared_config.time_window_config.window_size,
            forecast_horizon_days=shared_config.time_window_config.forecast_horizon_days,
        )
        retained_records = plan.retained_records()
        if not retained_records:
            raise ValueError("双臂共同日历样本为空，无法执行严格预测回测")
        shared_split = make_calendar_split(
            retained_records,
            test_ratio=shared_config.time_window_config.test_split_ratio,
            n_internal_folds=2,
        )
        shared_keys = {record.compact_key for record in retained_records}

        round_dir = self.results_dir / f"round_{protocol.round_id:02d}"
        execution_dir = round_dir / protocol.experiment_id
        execution_dir.mkdir(parents=True, exist_ok=True)
        run_id = self._next_run_id(protocol, round_dir)

        started = time.perf_counter()
        baseline_output = self._execute_variant(
            protocol=protocol,
            feature_columns=baseline_features,
            label="baseline",
            output_dir=execution_dir / "baseline",
            shared_keys=shared_keys,
            shared_split=shared_split,
            pre_loaded=merged,
            shared_overrides=shared_overrides,
        )
        if is_baseline_single_arm:
            treatment_output = baseline_output
        else:
            treatment_output = self._execute_variant(
                protocol=protocol,
                feature_columns=treatment_features,
                label="treatment",
                output_dir=execution_dir / "treatment",
                shared_keys=shared_keys,
                shared_split=shared_split,
                pre_loaded=merged,
                shared_overrides=shared_overrides,
            )
            self._assert_common_samples(baseline_output, treatment_output)
        duration_seconds = time.perf_counter() - started

        predictions_dir = execution_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        prediction_paths = {
            "baseline": self._to_relative_path(
                self._write_prediction_csv(
                    predictions_dir / f"{protocol.experiment_id}_baseline.csv",
                    baseline_output,
                )
            ),
            "treatment": self._to_relative_path(
                self._write_prediction_csv(
                    predictions_dir / f"{protocol.experiment_id}_treatment.csv",
                    treatment_output,
                )
            ),
        }

        visualizations = self._collect_visualizations(protocol, baseline_output, treatment_output)
        coverage_entries: list[DataSourceCoverage] = []
        outputs_to_track = (
            [baseline_output]
            if is_baseline_single_arm
            else [baseline_output, treatment_output]
        )
        for output in outputs_to_track:
            for entry in output.get("data_coverage", []):
                coverage_entries.append(DataSourceCoverage.model_validate(entry))

        bootstrap_summary = None
        robustness = None
        if not is_baseline_single_arm:
            baseline_frame = self._prediction_frame(baseline_output)
            treatment_frame = self._prediction_frame(treatment_output)
            bootstrap_summary = paired_calendar_block_bootstrap(
                baseline_frame,
                treatment_frame,
                block_size=7,
                n_draws=10000,
                seed=20261008,
            )
            robustness = summary_to_robustness_report(bootstrap_summary)
        else:
            bootstrap_summary = empty_bootstrap_summary("单臂对照模式没有可配对的实验组")

        result = build_experiment_result(
            protocol,
            baseline_output=baseline_output,
            treatment_output=treatment_output,
            prediction_paths=prediction_paths,
            visualizations=visualizations,
            duration_seconds=duration_seconds,
            data_coverage=coverage_entries,
            robustness=robustness,
            run_id=run_id,
        )
        export_result_bundle(
            result,
            protocol,
            round_dir=round_dir,
            latest_protocol_path=self.latest_protocol_path,
            latest_metrics_path=self.latest_metrics_path,
        )
        self._write_archive(
            protocol=protocol,
            shared_config=shared_config,
            merged=merged,
            plan=plan,
            shared_split=shared_split,
            run_id=run_id,
            round_dir=round_dir,
            result=result,
            baseline_output=baseline_output,
            treatment_output=treatment_output,
            bootstrap_summary=bootstrap_summary,
        )
        return result

    def _prepare_common_pipeline(
        self,
        protocol: ExperimentProtocol,
    ) -> dict:
        """共享日历样本 / 划分 / 输出目录，供双臂与多 run 实验复用。"""

        baseline_features = self._resolve_feature_columns(
            protocol=protocol,
            label="baseline",
            fallback=protocol.features.control or protocol.features.treatment,
        )
        treatment_features = self._resolve_feature_columns(
            protocol=protocol,
            label="treatment",
            fallback=protocol.features.treatment,
        )
        shared_features = list(
            dict.fromkeys([*baseline_features, *treatment_features])
        )
        shared_overrides = self._resolve_shared_window_overrides(protocol)
        shared_config = self.bridge.build_runner_config(
            protocol,
            shared_features,
            arm_overrides=shared_overrides,
        )
        merged = runner.load_and_merge_data(shared_config)
        plan = build_sample_plan(
            merged.X,
            merged.y,
            merged.time,
            feature_columns=shared_features,
            target_name=protocol.target,
            past_lag_days=shared_config.time_window_config.past_lag_days
            or shared_config.time_window_config.window_size,
            forecast_horizon_days=shared_config.time_window_config.forecast_horizon_days,
        )
        retained_records = plan.retained_records()
        if not retained_records:
            raise ValueError("双臂共同日历样本为空，无法执行严格预测回测")
        shared_split = make_calendar_split(
            retained_records,
            test_ratio=shared_config.time_window_config.test_split_ratio,
            n_internal_folds=2,
        )
        shared_keys = {record.compact_key for record in retained_records}

        round_dir = self.results_dir / f"round_{protocol.round_id:02d}"
        execution_dir = round_dir / protocol.experiment_id
        execution_dir.mkdir(parents=True, exist_ok=True)
        return {
            "baseline_features": baseline_features,
            "treatment_features": treatment_features,
            "shared_features": shared_features,
            "shared_overrides": shared_overrides,
            "shared_config": shared_config,
            "merged": merged,
            "plan": plan,
            "shared_split": shared_split,
            "shared_keys": shared_keys,
            "round_dir": round_dir,
            "execution_dir": execution_dir,
        }

    def run_batch(
        self,
        protocol: ExperimentProtocol,
        *,
        run_specs: list,
        contrasts_spec: list,
        diagnostics_spec: list | None = None,
    ) -> ExperimentResult:
        """Execute a validated multi-run protocol on one shared calendar sample."""

        if not run_specs:
            raise ValueError("run_batch 至少需要一个 run spec")
        pipeline = self._prepare_common_pipeline(protocol)
        shared_keys = pipeline["shared_keys"]
        shared_split = pipeline["shared_split"]
        execution_dir = pipeline["execution_dir"]
        round_dir = pipeline["round_dir"]
        run_id = self._next_run_id(protocol, round_dir)

        started = time.perf_counter()
        outputs: dict[str, dict] = {}
        for spec in run_specs:
            family = _normalize_model_family(spec.model_family)
            if family is None:
                raise ValueError(f"批量执行仅支持 Ridge / ElasticNet，收到: {spec.model_family}")
            variant_protocol = protocol.model_copy(deep=True)
            merged_parameters = {
                **dict(protocol.model.parameters),
                **dict(spec.parameters),
                "model_family": family,
            }
            variant_protocol.model = ModelSpec(
                name=family,
                parameters=merged_parameters,
            )
            outputs[spec.run_id] = self._execute_variant(
                protocol=variant_protocol,
                feature_columns=list(spec.feature_columns),
                label=spec.label,
                output_dir=execution_dir / spec.run_id,
                shared_keys=shared_keys,
                shared_split=shared_split,
                pre_loaded=pipeline["merged"],
                shared_overrides=pipeline["shared_overrides"],
            )
        duration_seconds = time.perf_counter() - started

        predictions_dir = execution_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        prediction_paths: dict[str, str] = {}
        for spec in run_specs:
            prediction_paths[spec.run_id] = self._to_relative_path(
                self._write_prediction_csv(
                    predictions_dir / f"{protocol.experiment_id}_{spec.run_id}.csv",
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
        for contrast_spec in contrasts_spec:
            if contrast_spec.kind == "gain_change":
                contrasts.append(
                    ExperimentContrast(
                        contrast_id=contrast_spec.contrast_id,
                        name=contrast_spec.name,
                        kind="gain_change",
                        specification={
                            "run_ids": contrast_spec.run_ids,
                            "contrast_ids": contrast_spec.contrast_ids,
                        },
                        run_ids=list(contrast_spec.run_ids),
                        contrast_ids=list(contrast_spec.contrast_ids),
                        note=contrast_spec.note,
                    )
                )
                continue
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
            runs.append(
                ExperimentRun(
                    run_id=spec.run_id,
                    name=spec.label,
                    prediction_artifact=prediction_paths[spec.run_id],
                    metrics=build_run_metrics(output["metrics"]),
                    model_family=spec.model_family,
                    feature_labels=list(spec.feature_columns),
                    feature_columns=list(spec.feature_columns),
                    parameters=dict(spec.parameters),
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
        )
        self._write_batch_archive(
            protocol=protocol,
            run_id=run_id,
            round_dir=round_dir,
            result=result,
            outputs=outputs,
            run_specs=run_specs,
            shared_config=pipeline["shared_config"],
            plan=pipeline["plan"],
            shared_split=pipeline["shared_split"],
        )
        return result

    def _write_batch_archive(
        self,
        *,
        protocol: ExperimentProtocol,
        run_id: str,
        round_dir: Path,
        result: ExperimentResult,
        outputs: dict[str, dict],
        run_specs: list,
        shared_config,
        plan,
        shared_split,
    ) -> None:
        archive_dir = round_dir / "archive" / run_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        (archive_dir / "experiment_protocol.json").write_text(
            json.dumps(protocol.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        rows = []
        for spec in run_specs:
            frame = UnifiedExperimentHarness._prediction_frame_with_arm(
                spec.run_id,
                outputs[spec.run_id],
            )
            rows.append(frame)
        pd.concat(rows, ignore_index=True).to_csv(
            archive_dir / "predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        metrics_payload = {
            "run_id": run_id,
            "experiment_id": result.experiment_id,
            "round_id": result.round_id,
            "experiment_type": result.experiment_type,
            "runs": [run.model_dump(mode="json", exclude_none=True) for run in result.runs],
            "contrasts": [contrast.model_dump(mode="json", exclude_none=True) for contrast in result.contrasts],
            "diagnostics": [item.model_dump(mode="json", exclude_none=True) for item in result.diagnostics],
            "sample_plan": plan.stats(),
            "calendar_split": shared_split.to_dict(),
            "model_family": shared_config.elasticnet_config.model_family,
        }
        (archive_dir / "metrics.json").write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        latest_entry = {
            "run_id": run_id,
            "archived_at": pd.Timestamp.now().isoformat(),
            "experiment_protocol.json": "experiment_protocol.json",
            "predictions.csv": "predictions.csv",
            "metrics.json": "metrics.json",
        }
        (round_dir / "archive" / "latest_run.json").write_text(
            json.dumps(latest_entry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (round_dir / "latest_run_id.txt").write_text(run_id, encoding="utf-8")

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
    ) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
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

        model = runner.train_elasticnet_model(
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
                entry["note"] = "缺失记录已按行剔除（inner merge + dropna），窗口直接基于剩余行构造"
                coverage_entries.append(entry)

        full_time = np.concatenate([split.time_train, split.time_test])
        x_limits = (full_time.min(), full_time.max()) if len(full_time) > 0 else None
        split_time = split.time_test[0] if len(split.time_test) > 0 else None
        chart_label = f"{protocol.experiment_id} / {label}"
        figure_paths = {
            "coefficients": runner.plot_feature_coefficients(coef_df, output_dir),
            "scatter": runner.plot_prediction_scatter(
                split.y_test,
                yte_pred,
                metrics,
                output_dir,
                experiment_label=chart_label,
                train_true=split.y_train,
                train_pred=ytr_pred,
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
            ),
            "residual_scatter": runner.plot_residual_scatter(
                yte_pred,
                split.y_test - yte_pred,
                output_dir,
            ),
            "residual_hist": runner.plot_residual_histogram(
                split.y_test - yte_pred,
                output_dir,
            ),
        }

        shap_paths: dict[str, str] = {}
        shap_importance = None
        if self.run_shap:
            shap_result = runner.run_shap_analysis(model, split.X_train, split.X_test, output_dir)
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

    def _write_archive(
        self,
        *,
        protocol: ExperimentProtocol,
        shared_config,
        merged,
        plan: SamplePlan,
        shared_split,
        run_id: str,
        round_dir: Path,
        result: ExperimentResult,
        baseline_output: dict,
        treatment_output: dict,
        bootstrap_summary: dict | None,
    ) -> None:
        archive_dir = round_dir / "archive" / run_id
        archive_dir.mkdir(parents=True, exist_ok=True)

        (archive_dir / "experiment_protocol.json").write_text(
            json.dumps(protocol.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._write_data_manifest(archive_dir, shared_config, merged, plan)
        self._write_sample_manifest(archive_dir, plan, shared_split)
        self._write_cv_results(archive_dir, shared_split, baseline_output, treatment_output)
        self._write_predictions(archive_dir, protocol, baseline_output, treatment_output)
        self._write_metrics(archive_dir, result, bootstrap_summary)
        self._write_diagnostics(
            archive_dir,
            protocol,
            plan,
            shared_split,
            baseline_output,
            treatment_output,
        )
        self._write_run_notes(
            archive_dir,
            protocol,
            shared_config,
            plan,
            bootstrap_summary,
            run_id,
        )

        latest_entry = {
            "run_id": run_id,
            "archived_at": pd.Timestamp.now().isoformat(),
            "experiment_protocol.json": "experiment_protocol.json",
            "data_manifest.json": "data_manifest.json",
            "sample_manifest.csv": "sample_manifest.csv",
            "cv_results.csv": "cv_results.csv",
            "predictions.csv": "predictions.csv",
            "metrics.json": "metrics.json",
            "diagnostics.json": "diagnostics.json",
            "run_notes.json": "run_notes.json",
        }
        (round_dir / "archive" / "latest_run.json").write_text(
            json.dumps(latest_entry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (round_dir / "latest_run_id.txt").write_text(run_id, encoding="utf-8")

    def _write_data_manifest(self, archive_dir: Path, shared_config, merged, plan: SamplePlan) -> None:
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
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_sample_manifest(self, archive_dir: Path, plan: SamplePlan, shared_split) -> None:
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
            "sample_condition",
            "drop_reason",
        ]).to_csv(archive_dir / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    @staticmethod
    def _write_cv_results(
        archive_dir: Path,
        shared_split,
        baseline_output: dict,
        treatment_output: dict,
    ) -> None:
        rows = []
        split_dict = shared_split.to_dict()
        for fold in split_dict["cv_folds"]:
            rows.append(
                {
                    "fold": fold["fold"],
                    "train_count": fold["train_count"],
                    "eval_count": fold["eval_count"],
                    "train_targets_strictly_before_eval_issue": fold[
                        "train_targets_strictly_before_eval_issue"
                    ],
                    "baseline_selected_alpha": baseline_output.get("selected_alpha"),
                    "treatment_selected_alpha": treatment_output.get("selected_alpha"),
                }
            )
        pd.DataFrame(rows).to_csv(
            archive_dir / "cv_results.csv",
            index=False,
            encoding="utf-8-sig",
        )

    @staticmethod
    def _write_predictions(
        archive_dir: Path,
        protocol: ExperimentProtocol,
        baseline_output: dict,
        treatment_output: dict,
    ) -> None:
        frames = [
            UnifiedExperimentHarness._prediction_frame_with_arm("baseline", baseline_output),
            UnifiedExperimentHarness._prediction_frame_with_arm("treatment", treatment_output),
        ]
        pd.concat(frames, ignore_index=True).to_csv(
            archive_dir / "predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        # 保留协议快照便于审计对比。
        (archive_dir / "prediction_protocol.json").write_text(
            json.dumps(protocol.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _prediction_frame_with_arm(arm: str, output: dict) -> pd.DataFrame:
        frame = UnifiedExperimentHarness._prediction_frame(output)
        frame.insert(0, "arm", arm)
        return frame

    @staticmethod
    def _write_metrics(
        archive_dir: Path,
        result: ExperimentResult,
        bootstrap_summary: dict | None,
    ) -> None:
        payload = {
            "run_id": result.run_id,
            "experiment_id": result.experiment_id,
            "round_id": result.round_id,
            "status": result.status,
            "baseline": result.runs[0].metrics.model_dump(exclude_none=True),
            "treatment": result.runs[1].metrics.model_dump(exclude_none=True),
            "comparison": result.comparison.model_dump(exclude_none=True),
            "bootstrap": bootstrap_summary or None,
        }
        (archive_dir / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_diagnostics(
        self,
        archive_dir: Path,
        protocol: ExperimentProtocol,
        plan: SamplePlan,
        shared_split,
        baseline_output: dict,
        treatment_output: dict,
    ) -> None:
        arms: dict[str, dict] = {}
        for output in (baseline_output, treatment_output):
            arms[output["label"]] = {
                "n_train": int(len(output["dataset_split"].y_train)),
                "n_test": int(len(output["dataset_split"].y_test)),
                "sliding_window_total": int(output["sliding_window_total"]),
                "sliding_window_dropped": int(output["sliding_window_dropped"]),
                "common_sample_excluded": int(output["common_sample_excluded"]),
                "selected_alpha": output.get("selected_alpha"),
                "feature_columns": output["config"]["file_config"]["feature_columns"],
            }
        diagnostics = {
            "run_id": protocol.experiment_id,
            "sample_plan": plan.stats(),
            "calendar_split": shared_split.to_dict(),
            "arms": arms,
        }
        (archive_dir / "diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_run_notes(
        self,
        archive_dir: Path,
        protocol: ExperimentProtocol,
        shared_config,
        plan: SamplePlan,
        bootstrap_summary: dict | None,
        run_id: str,
    ) -> None:
        notes = {
            "run_id": run_id,
            "experiment_id": protocol.experiment_id,
            "round_id": protocol.round_id,
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
            "sign_convention": (
                "RMSE/MAE 正向 delta=实验组误差更小；Pearson r/R2 正向 delta=实验组相关更高"
            ),
            "model_family": shared_config.elasticnet_config.model_family,
            "forecast_horizon_days": shared_config.time_window_config.forecast_horizon_days,
            "past_lag_days": plan.past_lag_days,
            "alpha_grid": shared_config.elasticnet_config.alpha_grid,
            "bootstrap": {
                "status": (bootstrap_summary or {}).get("status"),
                "n_iterations": (bootstrap_summary or {}).get("n_iterations"),
                "block_size": (bootstrap_summary or {}).get("block_size"),
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
            json.dumps(notes, ensure_ascii=False, indent=2),
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
