import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.experiment_tools.dispatcher import ExperimentDispatcher
from core.experiment_tools.evaluators import evaluate_experiment_result
from core.experiment_tools.registry import (
    ALL_EXPERIMENT_TYPES,
    EXPERIMENT_TOOL_INDEX,
    candidate_selectable_types,
    get_tool,
)
from core.experiment_tools.validate_protocol import validate_experiment_protocol
from core.harness_unified import UnifiedExperimentHarness
from core.control_unified import CandidateProtocolMapper
from core.protocol_bridge import ElasticNetDataSourceConfig
from core.state_repository import UnifiedStateRepository
from core.state_updater import UnifiedStateUpdater
from core.unified_schema import (
    BootstrapReport,
    CandidateExperiment,
    ContrastSpec,
    DiagnosticSpec,
    EvaluationSpec,
    ExperimentContrast,
    ExecutionSummary,
    ExperimentDesign,
    ExperimentProtocol,
    ExperimentResult,
    ExperimentRun,
    HypothesisNode,
    ModelSpec,
    ResearchQuestion,
    RunSpec,
    ScientificConstraints,
    ScientificTask,
    ScientificTaskPayload,
    VariableBinding,
)


class ExperimentDispatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "config").mkdir()
        (self.root / "outputs").mkdir()
        (self.root / "results").mkdir()

        time_values = pd.date_range("2021-01-01", periods=44, freq="D")
        omni = pd.DataFrame(
            {
                "TIME": time_values.strftime("%Y%m%d"),
                "By": [(-1) ** i * (i % 5 + 1) for i in range(44)],
                "Bz": [(i % 7) - 3 for i in range(44)],
                "Vsw": [350 + i * 2 for i in range(44)],
            }
        )
        lhaaso = pd.DataFrame(
            {
                "TIME": time_values.strftime("%Y%m%d"),
                "DeltaDec": [0.1 * ((i % 6) - 3) for i in range(44)],
            }
        )
        self.omni_path = self.root / "omni.csv"
        self.lhaaso_path = self.root / "lhaaso.csv"
        omni.to_csv(self.omni_path, index=False)
        lhaaso.to_csv(self.lhaaso_path, index=False)

        self.data_source = ElasticNetDataSourceConfig(
            omni_file_path=self.omni_path,
            lhaaso_file_path=self.lhaaso_path,
            time_column="TIME",
            target_column="Vsw",
            test_split_ratio=0.25,
            random_state=7,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _harness(self, data_source=None) -> UnifiedExperimentHarness:
        return UnifiedExperimentHarness(
            data_source=data_source or self.data_source,
            project_root=self.root,
            run_shap=False,
        )

    def _base_protocol(
        self,
        experiment_type: str,
        experiment_id: str,
        **kwargs,
    ) -> ExperimentProtocol:
        fields = {
            "experiment_id": experiment_id,
            "round_id": 1,
            "task_id": "ST_DISPATCH",
            "target": "Vsw",
            "experiment_type": experiment_type,
            "features": ExperimentDesign(
                target="Vsw",
                control=["By", "Bz"],
                treatment=["By", "Bz", "DeltaDec"],
            ),
            "model": ModelSpec(
                name="ElasticNet",
                parameters={
                    "window_size": 3,
                    "use_lag_feature": True,
                    "max_lag_day": 1,
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                    "random_state": 7,
                    "test_split_ratio": 0.25,
                    "forecast_horizon_days": 1,
                    "past_lag_days": 3,
                },
            ),
        }
        fields.update(kwargs)
        return ExperimentProtocol(**fields)

    def test_registry_only_exposes_connected_tools(self) -> None:
        self.assertEqual(
            {item.experiment_type for item in EXPERIMENT_TOOL_INDEX.values()},
            {
                "incremental_prediction",
                "baseline_benchmark",
                "baseline_redundancy",
                "model_sensitivity",
                "temporal_sensitivity",
                "conditional_effect",
                "robustness_diagnostic",
            },
        )
        self.assertEqual(
            set(ALL_EXPERIMENT_TYPES),
            set(EXPERIMENT_TOOL_INDEX.keys()),
        )
        self.assertTrue(get_tool("incremental_prediction").available)
        self.assertTrue(get_tool("baseline_benchmark").available)
        self.assertTrue(get_tool("model_sensitivity").available)
        self.assertTrue(get_tool("robustness_diagnostic").available)
        for experiment_type in [
            "baseline_redundancy",
            "temporal_sensitivity",
            "conditional_effect",
        ]:
            self.assertFalse(get_tool(experiment_type).available)

    def test_candidate_selectable_types_exclude_robustness_entry(self) -> None:
        self.assertEqual(
            set(candidate_selectable_types()),
            {"incremental_prediction", "model_sensitivity", "baseline_benchmark"},
        )

    def _task(self) -> ScientificTask:
        return ScientificTask(
            task_id="ST_CAND_MAP",
            payload=ScientificTaskPayload(
                research_question=ResearchQuestion(
                    text="DeltaDec 是否改善 Vsw 预测",
                    target="Vsw",
                    question_type="forecasting",
                    variables=VariableBinding(x="DeltaDec", y="Vsw"),
                ),
                evaluation=EvaluationSpec(primary_metric="RMSE"),
                constraints=ScientificConstraints(),
            ),
        )

    def test_candidate_mapper_compiles_model_sensitivity_four_runs(self) -> None:
        candidate = CandidateExperiment(
            experiment_id="E_MS_CAND_01",
            type="distinguishing",
            experiment_type="model_sensitivity",
            purpose="检验日影增量是否依赖模型家族",
            design=ExperimentDesign(
                target="Vsw",
                control=["By", "Bz"],
                treatment=["By", "Bz", "DeltaDec"],
                design_focus="DeltaDec",
                model_families=["ridge", "elasticnet"],
                runs_spec=[
                    RunSpec(
                        run_id="ridge_baseline",
                        label="Ridge 对照组",
                        feature_columns=["By", "Bz"],
                        model_family="ridge",
                    ),
                    RunSpec(
                        run_id="ridge_treatment",
                        label="Ridge 实验组",
                        feature_columns=["By", "Bz", "DeltaDec"],
                        model_family="ridge",
                    ),
                    RunSpec(
                        run_id="elasticnet_baseline",
                        label="ElasticNet 对照组",
                        feature_columns=["By", "Bz"],
                        model_family="elasticnet",
                    ),
                    RunSpec(
                        run_id="elasticnet_treatment",
                        label="ElasticNet 实验组",
                        feature_columns=["By", "Bz", "DeltaDec"],
                        model_family="elasticnet",
                    ),
                ],
                contrasts_spec=[
                    ContrastSpec(
                        contrast_id="gain_ridge",
                        name="Ridge 家族内日影增量",
                        kind="incremental_gain",
                        run_ids=["ridge_baseline", "ridge_treatment"],
                    ),
                    ContrastSpec(
                        contrast_id="gain_elasticnet",
                        name="ElasticNet 家族内日影增量",
                        kind="incremental_gain",
                        run_ids=["elasticnet_baseline", "elasticnet_treatment"],
                    ),
                ],
            ),
        )

        protocol = CandidateProtocolMapper().to_protocol(
            task=self._task(),
            candidate=candidate,
            round_id=1,
        )
        self.assertEqual(protocol.experiment_type, "model_sensitivity")
        self.assertEqual(protocol.model.parameters["model_families"], ["ridge", "elasticnet"])
        self.assertEqual(len(protocol.runs_spec), 4)
        self.assertEqual(len(protocol.contrasts_spec), 2)

        result = ExperimentDispatcher(self._harness()).execute(protocol)
        self.assertEqual(result.status, "completed")
        self.assertEqual({run.run_id for run in result.runs}, {
            "ridge_baseline",
            "ridge_treatment",
            "elasticnet_baseline",
            "elasticnet_treatment",
        })

    def test_candidate_mapper_preserves_single_arm_baseline(self) -> None:
        candidate = CandidateExperiment(
            experiment_id="E_BL_CAND_01",
            type="baseline_benchmark",
            experiment_type="baseline_benchmark",
            purpose="建立单臂基准性能快照",
            design=ExperimentDesign(
                target="Vsw",
                treatment=["By", "Bz"],
            ),
        )

        protocol = CandidateProtocolMapper().to_protocol(
            task=self._task(),
            candidate=candidate,
            round_id=1,
        )
        self.assertEqual(protocol.experiment_type, "baseline_benchmark")
        self.assertEqual(protocol.features.control, [])
        self.assertEqual(
            protocol.model.parameters["feature_groups"]["mode"],
            "baseline_single_arm",
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)
        self.assertEqual(result.status, "completed")
        self.assertIn("baseline", {run.run_id for run in result.runs})

    def test_candidate_mapper_rejects_unavailable_type_without_relabel(self) -> None:
        candidate = CandidateExperiment(
            experiment_id="E_TS_CAND_01",
            type="distinguishing",
            experiment_type="temporal_sensitivity",
            purpose="跨超前窗口检验",
            design=ExperimentDesign(
                target="Vsw",
                control=["By", "Bz"],
                treatment=["By", "Bz", "DeltaDec"],
                design_focus="DeltaDec",
            ),
        )
        with self.assertRaisesRegex(ValueError, "当前不可选"):
            CandidateProtocolMapper().to_protocol(
                task=self._task(),
                candidate=candidate,
                round_id=1,
            )
        self.assertEqual(candidate.experiment_type, "temporal_sensitivity")

    def test_invalid_incremental_design_is_rejected_not_rerouted(self) -> None:
        protocol = self._base_protocol(
            "incremental_prediction",
            "E_INVALID_01",
            features=ExperimentDesign(
                target="Vsw",
                control=["By", "Bz"],
                treatment=["By", "Bz"],
            ),
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.validity_status, "invalid")
        self.assertIn("焦点特征", "；".join(result.warnings))

    def test_unconnected_types_return_explicit_unsupported(self) -> None:
        cases = [
            self._base_protocol(
                "baseline_redundancy",
                "E_BR_01",
                model=ModelSpec(
                    name="ElasticNet",
                    parameters={
                        "baseline_features": ["By"],
                        "enhanced_features": ["Bz"],
                        "focus_features": ["DeltaDec"],
                    },
                ),
            ),
            self._base_protocol(
                "temporal_sensitivity",
                "E_TS_01",
                model=ModelSpec(
                    name="ElasticNet",
                    parameters={
                        "change_axis": "forecast_horizon_days",
                        "values": [1, 3],
                    },
                ),
                alignment={"common_issue_date": True, "common_target_date": False},
            ),
            self._base_protocol(
                "conditional_effect",
                "E_CE_01",
                model=ModelSpec(
                    name="ElasticNet",
                    parameters={
                        "condition_field": "By",
                        "threshold_source": "training_data",
                    },
                ),
                contrasts_spec=[
                    ContrastSpec(
                        contrast_id="gain_difference",
                        name="组间增益差",
                        kind="group_gain_difference",
                        run_ids=["baseline", "treatment"],
                    )
                ],
            ),
        ]
        for protocol in cases:
            result = ExperimentDispatcher(self._harness()).execute(protocol)
            self.assertEqual(result.status, "failed", protocol.experiment_id)
            self.assertEqual(
                result.validity_status,
                "pending_scientific_review",
                protocol.experiment_id,
            )
            self.assertTrue(
                any("未接通执行器" in item for item in result.warnings),
                protocol.experiment_id,
            )

    def test_model_sensitivity_runs_four_paired_runs(self) -> None:
        protocol = self._base_protocol(
            "model_sensitivity",
            "E_MS_01",
            model=ModelSpec(
                name="ElasticNet",
                parameters={
                    "model_families": ["ridge", "elasticnet"],
                    "window_size": 3,
                    "use_lag_feature": True,
                    "max_lag_day": 1,
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                    "random_state": 7,
                    "test_split_ratio": 0.25,
                    "forecast_horizon_days": 1,
                    "past_lag_days": 3,
                },
            ),
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.experiment_type, "model_sensitivity")
        self.assertEqual({run.run_id for run in result.runs}, {
            "ridge_baseline",
            "ridge_treatment",
            "elasticnet_baseline",
            "elasticnet_treatment",
        })
        self.assertGreaterEqual(len(result.contrasts), 2)
        for family in ("ridge", "elasticnet"):
            contrast = next(
                item
                for item in result.contrasts
                if item.contrast_id == f"gain_{family}"
            )
            self.assertEqual(
                contrast.run_ids,
                [f"{family}_baseline", f"{family}_treatment"],
            )
            self.assertIsNotNone(contrast.metrics_delta.rmse)
        result_path = self.root / "results" / "round_01" / "result_unified.json"
        self.assertTrue(result_path.exists())
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["experiment_type"], "model_sensitivity")
        self.assertTrue(
            any(
                item["diagnostic_id"] == "family_gain_summary"
                for item in payload["diagnostics"]
            )
        )

    def _sensitivity_protocol(
        self,
        experiment_id: str,
        *,
        runs_spec=None,
        contrasts_spec=None,
    ) -> ExperimentProtocol:
        fields = {
            "model": ModelSpec(
                name="ElasticNet",
                parameters={
                    "model_families": ["ridge", "elasticnet"],
                    "window_size": 3,
                    "use_lag_feature": True,
                    "max_lag_day": 1,
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                    "random_state": 7,
                    "test_split_ratio": 0.25,
                    "forecast_horizon_days": 1,
                    "past_lag_days": 3,
                },
            ),
        }
        if runs_spec is not None:
            fields["runs_spec"] = runs_spec
        if contrasts_spec is not None:
            fields["contrasts_spec"] = contrasts_spec
        return self._base_protocol("model_sensitivity", experiment_id, **fields)

    def _paired_run_specs(self) -> list[RunSpec]:
        return [
            RunSpec(
                run_id="ridge_baseline",
                label="Ridge 对照组",
                feature_columns=["By", "Bz"],
                model_family="ridge",
            ),
            RunSpec(
                run_id="ridge_treatment",
                label="Ridge 实验组",
                feature_columns=["By", "Bz", "DeltaDec"],
                model_family="ridge",
            ),
            RunSpec(
                run_id="elasticnet_baseline",
                label="ElasticNet 对照组",
                feature_columns=["By", "Bz"],
                model_family="elasticnet",
            ),
            RunSpec(
                run_id="elasticnet_treatment",
                label="ElasticNet 实验组",
                feature_columns=["By", "Bz", "DeltaDec"],
                model_family="elasticnet",
            ),
        ]

    def test_model_sensitivity_rejects_mislabeled_four_run_protocol(self) -> None:
        runs = [
            RunSpec(
                run_id="ridge_baseline",
                label="Ridge 对照组",
                feature_columns=["By", "Bz"],
                model_family="ridge",
            ),
            RunSpec(
                run_id="ridge_treatment",
                label="Ridge 实验组",
                feature_columns=["By", "Bz"],
                model_family="ridge",
            ),
            RunSpec(
                run_id="elasticnet_baseline",
                label="ElasticNet 对照组",
                feature_columns=["By", "Bz"],
                model_family="ridge",
            ),
            RunSpec(
                run_id="elasticnet_treatment",
                label="ElasticNet 实验组",
                feature_columns=["By", "Bz"],
                model_family="ridge",
            ),
        ]
        contrasts = [
            ContrastSpec(
                contrast_id="gain_ridge",
                name="Ridge 日影增量",
                kind="incremental_gain",
                run_ids=["ridge_baseline", "ridge_treatment"],
            ),
            ContrastSpec(
                contrast_id="cross_family",
                name="跨家族比较",
                kind="incremental_gain",
                run_ids=["ridge_baseline", "elasticnet_treatment"],
            ),
        ]
        protocol = self._sensitivity_protocol(
            "E_MS_BAD_01",
            runs_spec=runs,
            contrasts_spec=contrasts,
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.validity_status, "invalid")
        warning_text = "；".join(result.warnings)
        self.assertIn("模型家族与 run 命名必须一致", warning_text)
        self.assertIn("跨模型家族比较", warning_text)

    def test_model_sensitivity_rejects_duplicate_run_ids(self) -> None:
        runs = self._paired_run_specs()
        runs[1] = RunSpec(
            run_id="ridge_baseline",
            label="Ridge 对照组重复",
            feature_columns=["By", "Bz"],
            model_family="ridge",
        )
        protocol = self._sensitivity_protocol("E_MS_DUP_01", runs_spec=runs)
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertIn("重复 run_id", "；".join(result.warnings))

    def test_model_sensitivity_rejects_missing_contrast_reference(self) -> None:
        contrasts = [
            ContrastSpec(
                contrast_id="gain_bad",
                name="缺失 run 引用",
                kind="incremental_gain",
                run_ids=["ridge_baseline", "missing_run"],
            )
        ]
        protocol = self._sensitivity_protocol(
            "E_MS_REF_01",
            runs_spec=self._paired_run_specs(),
            contrasts_spec=contrasts,
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertIn(
            "引用了不存在的 run_id",
            "；".join(result.warnings),
        )

    def test_model_sensitivity_rejects_cross_family_contrast(self) -> None:
        contrasts = [
            ContrastSpec(
                contrast_id="gain_cross",
                name="跨家族比较",
                kind="incremental_gain",
                run_ids=["ridge_baseline", "elasticnet_treatment"],
            )
        ]
        protocol = self._sensitivity_protocol(
            "E_MS_CROSS_01",
            runs_spec=self._paired_run_specs(),
            contrasts_spec=contrasts,
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertIn("跨模型家族比较", "；".join(result.warnings))

    def test_incremental_rejects_retrain_false(self) -> None:
        protocol = self._base_protocol(
            "incremental_prediction",
            "E_INC_NORETRAIN",
            runs_spec=[
                RunSpec(
                    run_id="baseline",
                    label="对照组",
                    feature_columns=["By", "Bz"],
                    model_family="elasticnet",
                    retrain=False,
                ),
                RunSpec(
                    run_id="treatment",
                    label="实验组",
                    feature_columns=["By", "Bz", "DeltaDec"],
                    model_family="elasticnet",
                    retrain=False,
                ),
            ],
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.validity_status, "invalid")
        self.assertIn("禁止 retrain=False", "；".join(result.warnings))

    def test_incremental_rejects_diagnostics_spec(self) -> None:
        protocol = self._base_protocol(
            "incremental_prediction",
            "E_INC_DIAG",
            diagnostics_spec=[
                DiagnosticSpec(
                    diagnostic_id="paired_bootstrap",
                    name="配对日历块 bootstrap",
                )
            ],
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertIn(
            "普通训练实验禁止携带 diagnostics_spec",
            "；".join(result.warnings),
        )

    def test_incremental_rejects_feature_groups_conflict(self) -> None:
        params = dict(self._base_protocol("incremental_prediction", "E_X").model.parameters)
        params["feature_groups"] = {
            "mode": "comparative",
            "baseline": ["Bz"],
            "treatment": ["By", "Bz", "DeltaDec"],
        }
        protocol = self._base_protocol(
            "incremental_prediction",
            "E_INC_GROUPS",
            model=ModelSpec(name="ElasticNet", parameters=params),
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertIn("冲突", "；".join(result.warnings))

    def test_compiled_plan_executes_distinct_families_and_features(self) -> None:
        protocol = self._sensitivity_protocol("E_MS_COMPILED")
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "completed")
        by_run = {run.run_id: run for run in result.runs}
        self.assertEqual(by_run["ridge_baseline"].model_family, "ridge")
        self.assertEqual(by_run["ridge_treatment"].model_family, "ridge")
        self.assertEqual(by_run["elasticnet_baseline"].model_family, "elasticnet")
        self.assertEqual(by_run["elasticnet_treatment"].model_family, "elasticnet")
        self.assertEqual(by_run["ridge_baseline"].feature_columns, ["By", "Bz"])
        self.assertEqual(
            by_run["elasticnet_treatment"].feature_columns,
            ["By", "Bz", "DeltaDec"],
        )
        latest_metrics = json.loads(
            (self.root / "outputs" / "latest_metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(latest_metrics["runs"]), 4)
        self.assertNotIn("baseline", latest_metrics)

    def test_incremental_rejects_fractional_horizon(self) -> None:
        params = dict(
            self._base_protocol(
                "incremental_prediction",
                "E_X",
            ).model.parameters
        )
        params["forecast_horizon_days"] = 1.5
        protocol = self._base_protocol(
            "incremental_prediction",
            "E_INC_HORIZON_FRACTION",
            model=ModelSpec(name="ElasticNet", parameters=params),
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertIn("必须是正整数", "；".join(result.warnings))

    def test_incremental_rejects_invalid_test_ratio(self) -> None:
        params = dict(
            self._base_protocol(
                "incremental_prediction",
                "E_X",
            ).model.parameters
        )
        params["test_split_ratio"] = 2
        protocol = self._base_protocol(
            "incremental_prediction",
            "E_INC_TEST_RATIO",
            model=ModelSpec(name="ElasticNet", parameters=params),
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertIn("(0, 1)", "；".join(result.warnings))

    def test_incremental_rejects_asymmetric_fixed_alpha(self) -> None:
        protocol = self._base_protocol(
            "incremental_prediction",
            "E_INC_ASYMMETRIC_ALPHA",
            runs_spec=[
                RunSpec(
                    run_id="baseline",
                    label="对照组",
                    feature_columns=["By", "Bz"],
                    model_family="elasticnet",
                    parameters={"alpha": 1000.0},
                ),
                RunSpec(
                    run_id="treatment",
                    label="实验组",
                    feature_columns=["By", "Bz", "DeltaDec"],
                    model_family="elasticnet",
                    parameters={"alpha": 0.001},
                ),
            ],
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertIn("同一预处理/选参规则", "；".join(result.warnings))

    def test_unknown_model_name_rejected_not_fallback(self) -> None:
        base_params = dict(
            self._base_protocol(
                "incremental_prediction",
                "E_X",
            ).model.parameters
        )
        for name in ("XGBoost", "RandomForest", "ElasticNetX"):
            with self.subTest(name=name):
                protocol = self._base_protocol(
                    "incremental_prediction",
                    f"E_UNKNOWN_MODEL_{name}",
                    model=ModelSpec(name=name, parameters=base_params),
                )
                validation = validate_experiment_protocol(protocol)
                self.assertFalse(validation.valid)
                self.assertIn(
                    "默认模型家族必须是 Ridge 或 ElasticNet",
                    "；".join(validation.reasons),
                )

    def test_model_aliases_compile_to_registered_family(self) -> None:
        base_params = dict(
            self._base_protocol(
                "incremental_prediction",
                "E_X",
            ).model.parameters
        )
        for name, expected_family in (
            ("enet", "elasticnet"),
            ("elastic_net", "elasticnet"),
            ("Ridge", "ridge"),
        ):
            with self.subTest(name=name):
                protocol = self._base_protocol(
                    "incremental_prediction",
                    f"E_ALIAS_{expected_family}_{name}",
                    model=ModelSpec(name=name, parameters=base_params),
                )
                validation = validate_experiment_protocol(protocol)
                self.assertTrue(validation.valid, "；".join(validation.reasons))
                self.assertEqual(
                    {run.model_family for run in validation.compiled_runs},
                    {expected_family},
                )

    def test_unknown_run_model_family_rejected(self) -> None:
        protocol = self._base_protocol(
            "incremental_prediction",
            "E_RUN_XGB",
            runs_spec=[
                RunSpec(
                    run_id="baseline",
                    label="对照组",
                    feature_columns=["By", "Bz"],
                    model_family="XGBoost",
                ),
                RunSpec(
                    run_id="treatment",
                    label="实验组",
                    feature_columns=["By", "Bz", "DeltaDec"],
                    model_family="XGBoost",
                ),
            ],
        )
        validation = validate_experiment_protocol(protocol)
        self.assertFalse(validation.valid)
        self.assertIn("XGBoost", "；".join(validation.reasons))

    def test_model_sensitivity_rejects_unknown_family(self) -> None:
        params = dict(
            self._base_protocol(
                "incremental_prediction",
                "E_X",
            ).model.parameters
        )
        params["model_families"] = ["XGBoost", "elasticnet"]
        protocol = self._base_protocol(
            "model_sensitivity",
            "E_MS_UNKNOWN",
            model=ModelSpec(name="ElasticNet", parameters=params),
        )
        validation = validate_experiment_protocol(protocol)
        self.assertFalse(validation.valid)
        self.assertIn(
            "不支持的模型家族 'XGBoost'",
            "；".join(validation.reasons),
        )

    def test_conflicting_duplicate_time_parameters_rejected(self) -> None:
        params = dict(
            self._base_protocol(
                "incremental_prediction",
                "E_X",
            ).model.parameters
        )
        params["forecast_horizon_days"] = 3
        protocol = self._base_protocol(
            "incremental_prediction",
            "E_INC_TIME_CONFLICT",
            features=ExperimentDesign(
                target="Vsw",
                control=["By", "Bz"],
                treatment=["By", "Bz", "DeltaDec"],
                forecast_horizon_days=1,
            ),
            model=ModelSpec(name="ElasticNet", parameters=params),
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertIn("重复时间参数", "；".join(result.warnings))

    def test_compiled_plan_horizon_matches_prediction_csv(self) -> None:
        protocol = self._base_protocol(
            "incremental_prediction",
            "E_INC_HORIZON_03",
            model=ModelSpec(
                name="ElasticNet",
                parameters={
                    "window_size": 3,
                    "use_lag_feature": True,
                    "max_lag_day": 1,
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                    "random_state": 7,
                    "test_split_ratio": 0.25,
                    "forecast_horizon_days": 3,
                    "past_lag_days": 3,
                },
            ),
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "completed")
        prediction_csv = (
            self.root
            / "results"
            / "round_01"
            / "E_INC_HORIZON_03"
            / "predictions"
            / "E_INC_HORIZON_03_baseline.csv"
        )
        frame = pd.read_csv(prediction_csv)
        self.assertGreater(len(frame), 0)
        issue_date = pd.to_datetime(frame["issue_date"])
        target_date = pd.to_datetime(frame["target_date"])
        self.assertTrue(((target_date - issue_date).dt.days == 3).all())

    def test_executor_uses_plan_horizon_not_legacy_features_field(self) -> None:
        protocol = self._base_protocol(
            "incremental_prediction",
            "E_PLAN_AUTH",
            features=ExperimentDesign(
                target="Vsw",
                control=["By", "Bz"],
                treatment=["By", "Bz", "DeltaDec"],
                forecast_horizon_days=1,
            ),
        )
        validation = validate_experiment_protocol(protocol)
        self.assertTrue(validation.valid, "；".join(validation.reasons))
        plan = validation.compiled_plan
        # 模拟评审场景：编译层已把提前量解析为 3 天，但协议旧字段仍残留 1。
        plan.time_parameters["forecast_horizon_days"] = 3
        plan.model_parameters["forecast_horizon_days"] = 3

        self._harness().run_compiled(protocol, plan)

        prediction_csv = (
            self.root
            / "results"
            / "round_01"
            / "E_PLAN_AUTH"
            / "predictions"
            / "E_PLAN_AUTH_baseline.csv"
        )
        frame = pd.read_csv(prediction_csv)
        self.assertGreater(len(frame), 0)
        issue_date = pd.to_datetime(frame["issue_date"])
        target_date = pd.to_datetime(frame["target_date"])
        self.assertTrue(((target_date - issue_date).dt.days == 3).all())

    def test_robustness_diagnostic_reuses_predictions_without_retrain(self) -> None:
        dates = pd.date_range("2021-02-01", periods=28, freq="D")
        frame = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(28)],
                "issue_date": dates.strftime("%Y-%m-%d"),
                "target_date": (dates + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "y_true": [400 + 3 * i for i in range(28)],
            }
        )
        paired = pd.concat(
            [
                frame.assign(arm="baseline", y_pred=[380 + 2.5 * i for i in range(28)]),
                frame.assign(arm="treatment", y_pred=[385 + 2.7 * i for i in range(28)]),
            ],
            ignore_index=True,
        )
        prediction_file = self.root / "source_predictions.csv"
        paired.to_csv(prediction_file, index=False)

        protocol = self._base_protocol(
            "robustness_diagnostic",
            "E_RD_01",
            model=ModelSpec(
                name="reused",
                parameters={
                    "source_experiment_id": "E_LEGACY_01",
                    "source_prediction_file": str(prediction_file),
                },
            ),
            diagnostics_spec=[
                DiagnosticSpec(
                    diagnostic_id="paired_bootstrap",
                    name="配对日历块 bootstrap",
                )
            ],
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.experiment_type, "robustness_diagnostic")
        self.assertEqual({run.run_id for run in result.runs}, {"baseline", "treatment"})
        self.assertTrue(
            all(run.training_mode == "reused" for run in result.runs)
        )
        self.assertEqual(
            [item.diagnostic_id for item in result.diagnostics],
            ["paired_calendar_bootstrap"],
        )
        self.assertIn(
            "未重新训练模型",
            "; ".join(result.warnings),
        )
        bootstrap = next(
            item
            for item in result.diagnostics
            if item.diagnostic_id == "paired_calendar_bootstrap"
        )
        self.assertEqual(bootstrap.structured["status"], "completed")
        self.assertEqual(bootstrap.structured["n_iterations"], 4000)

    def test_robustness_diagnostic_honors_diagnostics_spec(self) -> None:
        dates = pd.date_range("2021-02-01", periods=28, freq="D")
        frame = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(28)],
                "issue_date": dates.strftime("%Y-%m-%d"),
                "target_date": (dates + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "y_true": [400 + 3 * i for i in range(28)],
            }
        )
        paired = pd.concat(
            [
                frame.assign(arm="baseline", y_pred=[380 + 2.5 * i for i in range(28)]),
                frame.assign(arm="treatment", y_pred=[385 + 2.7 * i for i in range(28)]),
            ],
            ignore_index=True,
        )
        prediction_file = self.root / "source_spec_predictions.csv"
        paired.to_csv(prediction_file, index=False)

        protocol = self._base_protocol(
            "robustness_diagnostic",
            "E_RD_SPEC_01",
            model=ModelSpec(
                name="reused",
                parameters={
                    "source_experiment_id": "E_LEGACY_SPEC",
                    "source_prediction_file": str(prediction_file),
                },
            ),
            diagnostics_spec=[
                DiagnosticSpec(
                    diagnostic_id="paired_bootstrap",
                    name="配对日历块 bootstrap",
                    parameters={"block_size": 2, "n_draws": 100, "seed": 7},
                ),
                DiagnosticSpec(
                    diagnostic_id="time_segments",
                    name="时间分段误差诊断",
                ),
                DiagnosticSpec(
                    diagnostic_id="residual_diagnostics",
                    name="平均偏差与残差诊断",
                ),
            ],
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "completed")
        self.assertEqual(
            [item.diagnostic_id for item in result.diagnostics],
            [
                "paired_calendar_bootstrap",
                "time_segments",
                "residual_diagnostics",
            ],
        )
        bootstrap = next(
            item
            for item in result.diagnostics
            if item.diagnostic_id == "paired_calendar_bootstrap"
        )
        self.assertEqual(bootstrap.structured["n_iterations"], 100)
        self.assertEqual(bootstrap.structured["block_size"], 2)
        self.assertEqual(bootstrap.structured["n_valid_pearson_draws"], 100)

        self.assertIsNotNone(result.robustness)
        self.assertIsNotNone(result.robustness.time_slice)
        self.assertEqual(result.robustness.time_slice.status, "descriptive")
        self.assertEqual(result.robustness.time_slice.metric, "RMSE")
        self.assertGreaterEqual(len(result.robustness.time_slice.splits), 1)
        self.assertIn(
            "已执行描述性分时段诊断",
            result.robustness.recommendation or "",
        )

    def test_robustness_diagnostic_resolves_previous_round_archive(self) -> None:
        dates = pd.date_range("2021-02-01", periods=28, freq="D")
        frame = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(28)],
                "issue_date": dates.strftime("%Y-%m-%d"),
                "target_date": (dates + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "y_true": [400 + 3 * i for i in range(28)],
            }
        )
        paired = pd.concat(
            [
                frame.assign(arm="baseline", y_pred=[380 + 2.5 * i for i in range(28)]),
                frame.assign(arm="treatment", y_pred=[385 + 2.7 * i for i in range(28)]),
            ],
            ignore_index=True,
        )
        archive_dir = self.root / "results" / "round_01" / "archive" / "batch_run_01"
        archive_dir.mkdir(parents=True, exist_ok=True)
        paired.to_csv(archive_dir / "predictions.csv", index=False)
        (archive_dir / "experiment_protocol.json").write_text(
            json.dumps(
                {
                    "experiment_id": "E_SRC_01",
                    "round_id": 1,
                    "model": {"parameters": {"forecast_horizon_days": 1}},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (archive_dir / "metrics.json").write_text(
            json.dumps({"run_id": "batch_run_01", "experiment_id": "E_SRC_01"}),
            encoding="utf-8",
        )

        protocol = self._base_protocol(
            "robustness_diagnostic",
            "E_RD_ARCHIVE_01",
            round_id=2,
            model=ModelSpec(
                name="reused",
                parameters={
                    "source_experiment_id": "E_SRC_01",
                    "source_round_id": 1,
                    "source_run_id": "batch_run_01",
                    "forecast_horizon_days": 1,
                },
            ),
            diagnostics_spec=[
                DiagnosticSpec(
                    diagnostic_id="paired_bootstrap",
                    name="配对日历块 bootstrap",
                    parameters={"n_draws": 100, "block_size": 2, "seed": 9},
                )
            ],
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "completed")
        self.assertEqual({run.run_id for run in result.runs}, {"baseline", "treatment"})
        self.assertEqual(
            [item.diagnostic_id for item in result.diagnostics],
            ["paired_calendar_bootstrap"],
        )
        for run in result.runs:
            self.assertEqual(run.parameters["source_round_id"], 1)
            self.assertEqual(run.parameters["source_experiment_id"], "E_SRC_01")
            self.assertEqual(run.parameters["source_run_id"], "batch_run_01")
            self.assertTrue(run.parameters["prediction_artifact_hash"])
            self.assertIn(
                "SHA-256 已核验",
                "; ".join(
                    entry.note or ""
                    for entry in result.data_coverage
                ),
            )

    def _make_predictions_frame(
        self,
        *,
        base_pred_offset: float = 2.0,
        treatment_pred_offset: float = 4.0,
    ) -> pd.DataFrame:
        dates = pd.date_range("2021-02-01", periods=28, freq="D")
        base = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(28)],
                "issue_date": dates.strftime("%Y-%m-%d"),
                "target_date": (dates + pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
                "y_true": [400 + 3 * i for i in range(28)],
                "y_pred": [
                    round(402 + 3 * i + base_pred_offset, 4)
                    for i in range(28)
                ],
                "arm": "baseline",
            }
        )
        treatment = base.assign(
            y_pred=[
                round(402 + 3 * i + treatment_pred_offset, 4)
                for i in range(28)
            ],
            arm="treatment",
        )
        return pd.concat([base, treatment], ignore_index=True)

    def _write_prediction_archive(
        self,
        *,
        execution_id: str,
        experiment_id: str,
        predictions: pd.DataFrame,
    ) -> Path:
        archive_dir = (
            self.root
            / "results"
            / "round_01"
            / "archive"
            / execution_id
        )
        archive_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(archive_dir / "predictions.csv", index=False)
        (archive_dir / "experiment_protocol.json").write_text(
            json.dumps(
                {
                    "experiment_id": experiment_id,
                    "round_id": 1,
                    "target": "Vsw",
                    "model": {
                        "parameters": {"forecast_horizon_days": 1},
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (archive_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "run_id": execution_id,
                    "experiment_id": experiment_id,
                }
            ),
            encoding="utf-8",
        )
        return archive_dir

    def _robustness_protocol(
        self,
        *,
        experiment_id: str,
        source_experiment_id: str,
        runs_spec: tuple[RunSpec, RunSpec],
        source_execution_id: str | None = None,
    ) -> ExperimentProtocol:
        if source_execution_id:
            runs_spec = tuple(
                RunSpec(
                    run_id=spec.run_id,
                    label=spec.label,
                    retrain=spec.retrain,
                    parameters={
                        **spec.parameters,
                        "source_execution_id": source_execution_id,
                    },
                )
                for spec in runs_spec
            )
        return self._base_protocol(
            "robustness_diagnostic",
            experiment_id,
            round_id=2,
            model=ModelSpec(
                name="reused",
                parameters={
                    "source_experiment_id": source_experiment_id,
                    "source_round_id": 1,
                    "forecast_horizon_days": 1,
                },
            ),
            runs_spec=list(runs_spec),
            diagnostics_spec=[
                DiagnosticSpec(
                    diagnostic_id="paired_bootstrap",
                    name="配对日历块 bootstrap",
                    parameters={"n_draws": 100, "block_size": 2, "seed": 9},
                )
            ],
        )

    def test_source_direct_plus_archive_duplicate_resolves_to_archive(
        self,
    ) -> None:
        predictions = self._make_predictions_frame()
        self._write_prediction_archive(
            execution_id="E_SRC_02_batch",
            experiment_id="E_SRC_02",
            predictions=predictions,
        )
        direct = predictions[predictions["arm"] == "baseline"].drop(
            columns=["arm"]
        )
        direct_path = self.root / "results" / "round_01" / "E_SRC_02_baseline.csv"
        direct.to_csv(direct_path, index=False)

        protocol = self._robustness_protocol(
            experiment_id="E_RD_DUP_01",
            source_experiment_id="E_SRC_02",
            runs_spec=(
                RunSpec(
                    run_id="baseline",
                    label="对照组",
                    retrain=False,
                    parameters={"source_run_id": "baseline"},
                ),
                RunSpec(
                    run_id="treatment",
                    label="实验组",
                    retrain=False,
                    parameters={"source_run_id": "treatment"},
                ),
            ),
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "completed")
        archive_path = (
            self.root
            / "results"
            / "round_01"
            / "archive"
            / "E_SRC_02_batch"
            / "predictions.csv"
        )
        for run in result.runs:
            self.assertEqual(run.prediction_artifact, str(archive_path.resolve()))
            self.assertEqual(run.parameters["source_execution_id"], "E_SRC_02_batch")
            self.assertEqual(
                run.parameters["prediction_artifact_hash"],
                ExperimentDispatcher._file_sha256(archive_path),
            )

    def test_source_two_reruns_differ_requires_execution_id(self) -> None:
        run_1 = self._make_predictions_frame(base_pred_offset=2.0)
        run_2 = self._make_predictions_frame(base_pred_offset=2.5)
        self._write_prediction_archive(
            execution_id="E_SRC_03_r1",
            experiment_id="E_SRC_03",
            predictions=run_1,
        )
        self._write_prediction_archive(
            execution_id="E_SRC_03_r2",
            experiment_id="E_SRC_03",
            predictions=run_2,
        )

        protocol = self._robustness_protocol(
            experiment_id="E_RD_RERUN_AMB",
            source_experiment_id="E_SRC_03",
            runs_spec=(
                RunSpec(
                    run_id="baseline",
                    label="对照组",
                    retrain=False,
                    parameters={"source_run_id": "baseline"},
                ),
                RunSpec(
                    run_id="treatment",
                    label="实验组",
                    retrain=False,
                    parameters={"source_run_id": "treatment"},
                ),
            ),
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertIn(
            "多个科学上不同的预测文件版本",
            "; ".join(result.warnings),
        )

    def test_source_two_reruns_resolves_with_execution_id_or_hash(
        self,
    ) -> None:
        run_1 = self._make_predictions_frame(base_pred_offset=2.0)
        run_2 = self._make_predictions_frame(base_pred_offset=2.5)
        self._write_prediction_archive(
            execution_id="E_SRC_03_r1",
            experiment_id="E_SRC_03",
            predictions=run_1,
        )
        self._write_prediction_archive(
            execution_id="E_SRC_03_r2",
            experiment_id="E_SRC_03",
            predictions=run_2,
        )
        archive_1 = (
            self.root
            / "results"
            / "round_01"
            / "archive"
            / "E_SRC_03_r1"
            / "predictions.csv"
        )
        hash_1 = ExperimentDispatcher._file_sha256(archive_1)

        with self.subTest("by source_execution_id"):
            protocol = self._robustness_protocol(
                experiment_id="E_RD_RERUN_EXEC",
                source_experiment_id="E_SRC_03",
                source_execution_id="E_SRC_03_r1",
                runs_spec=(
                    RunSpec(
                        run_id="baseline",
                        label="对照组",
                        retrain=False,
                        parameters={"source_run_id": "baseline"},
                    ),
                    RunSpec(
                        run_id="treatment",
                        label="实验组",
                        retrain=False,
                        parameters={"source_run_id": "treatment"},
                    ),
                ),
            )
            result = ExperimentDispatcher(self._harness()).execute(protocol)
            self.assertEqual(result.status, "completed")
            for run in result.runs:
                self.assertEqual(
                    run.parameters["source_execution_id"],
                    "E_SRC_03_r1",
                )
                self.assertEqual(
                    run.parameters["prediction_artifact_hash"],
                    hash_1,
                )

        with self.subTest("by prediction_artifact_hash"):
            protocol = self._robustness_protocol(
                experiment_id="E_RD_RERUN_HASH",
                source_experiment_id="E_SRC_03",
                runs_spec=(
                    RunSpec(
                        run_id="baseline",
                        label="对照组",
                        retrain=False,
                        parameters={
                            "source_run_id": "baseline",
                            "prediction_artifact_hash": hash_1,
                        },
                    ),
                    RunSpec(
                        run_id="treatment",
                        label="实验组",
                        retrain=False,
                        parameters={
                            "source_run_id": "treatment",
                            "prediction_artifact_hash": hash_1,
                        },
                    ),
                ),
            )
            result = ExperimentDispatcher(self._harness()).execute(protocol)
            self.assertEqual(result.status, "completed")
            for run in result.runs:
                self.assertEqual(
                    run.prediction_artifact,
                    str(archive_1.resolve()),
                )
                self.assertEqual(
                    run.parameters["prediction_artifact_hash"],
                    hash_1,
                )

    def test_source_multi_model_archive_resolves_each_arm(self) -> None:
        archive_base = self._make_predictions_frame()
        archive_multimodel = pd.concat(
            [
                archive_base[archive_base["arm"] == "baseline"].assign(
                    arm="ridge_baseline"
                ),
                archive_base[archive_base["arm"] == "treatment"].assign(
                    arm="ridge_treatment"
                ),
            ],
            ignore_index=True,
        )
        self._write_prediction_archive(
            execution_id="E_SRC_04_ridge_batch",
            experiment_id="E_SRC_04",
            predictions=archive_multimodel,
        )

        protocol = self._robustness_protocol(
            experiment_id="E_RD_MULTI_ARM",
            source_experiment_id="E_SRC_04",
            runs_spec=(
                RunSpec(
                    run_id="baseline",
                    label="Ridge 对照组",
                    retrain=False,
                    parameters={"source_run_id": "ridge_baseline"},
                ),
                RunSpec(
                    run_id="treatment",
                    label="Ridge 实验组",
                    retrain=False,
                    parameters={"source_run_id": "ridge_treatment"},
                ),
            ),
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "completed")
        archive_path = (
            self.root
            / "results"
            / "round_01"
            / "archive"
            / "E_SRC_04_ridge_batch"
            / "predictions.csv"
        )
        for run in result.runs:
            self.assertEqual(run.prediction_artifact, str(archive_path.resolve()))
            self.assertEqual(
                run.parameters["prediction_artifact_hash"],
                ExperimentDispatcher._file_sha256(archive_path),
            )
        self.assertEqual(
            [run.parameters.get("source_run_id") for run in result.runs],
            ["ridge_baseline", "ridge_treatment"],
        )

    def test_source_wrong_artifact_hash_rejected(self) -> None:
        predictions = self._make_predictions_frame()
        self._write_prediction_archive(
            execution_id="E_SRC_05_batch",
            experiment_id="E_SRC_05",
            predictions=predictions,
        )

        protocol = self._robustness_protocol(
            experiment_id="E_RD_BAD_HASH",
            source_experiment_id="E_SRC_05",
            runs_spec=(
                RunSpec(
                    run_id="baseline",
                    label="对照组",
                    retrain=False,
                    parameters={
                        "source_run_id": "baseline",
                        "prediction_artifact_hash": "0" * 64,
                    },
                ),
                RunSpec(
                    run_id="treatment",
                    label="实验组",
                    retrain=False,
                    parameters={
                        "source_run_id": "treatment",
                        "prediction_artifact_hash": "0" * 64,
                    },
                ),
            ),
        )
        result = ExperimentDispatcher(self._harness()).execute(protocol)

        self.assertEqual(result.status, "failed")
        self.assertIn(
            "SHA-256 与 prediction_artifact_hash 不一致",
            "; ".join(result.warnings),
        )

    def test_new_type_evaluation_does_not_auto_update_support(self) -> None:
        repository = UnifiedStateRepository(self.root)
        task = ScientificTask(
            task_id="ST_DISPATCH_STATE",
            payload=ScientificTaskPayload(
                research_question=ResearchQuestion(
                    text="DeltaDec 是否改善 Vsw 预测",
                    target="Vsw",
                    question_type="forecasting",
                    variables=VariableBinding(x="DeltaDec", y="Vsw"),
                ),
                evaluation=EvaluationSpec(primary_metric="RMSE"),
                constraints=ScientificConstraints(),
            ),
        )
        repository.initialize_state_skeleton(
            task,
            initial_hypotheses=[
                HypothesisNode(
                    hypothesis_id="H1",
                    statement="DeltaDec 带来独立增量",
                    level=1,
                    status="active",
                    support_score=0.5,
                    activation_condition="always",
                )
            ],
            overwrite=True,
        )
        protocol = self._base_protocol("model_sensitivity", "E_MS_STATE")
        protocol.tested_hypotheses = ["H1"]
        result = ExperimentResult(
            experiment_id=protocol.experiment_id,
            round_id=protocol.round_id,
            status="completed",
            experiment_type="model_sensitivity",
            execution=ExecutionSummary(status="completed"),
        )
        evaluation = evaluate_experiment_result(
            result,
            protocol=protocol,
            remaining_uncertainties=[
                {"question": "Ridge 与 ElasticNet 的日影增量是否一致"}
            ],
            main_question=task.payload.research_question.text,
        )

        assessments = evaluation.scientific.hypothesis_assessments
        self.assertEqual(len(assessments), 1)
        self.assertFalse(assessments[0].updated)
        self.assertEqual(assessments[0].support_before, assessments[0].support_after)
        self.assertIsNone(assessments[0].direction_matched)
        self.assertIsNone(assessments[0].magnitude_matched)
        self.assertEqual(assessments[0].evidence_classification, "inconclusive")
        self.assertTrue(assessments[0].evidence_labels)
        self.assertIn(
            "未检出稳定增益",
            evaluation.scientific.three_layer_conclusion.scientific_layer.answer,
        )

        updated = UnifiedStateUpdater(repository).apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
            protocol_path="config/latest_protocol.json",
            result_path=f"results/round_{protocol.round_id:02d}/result_unified.json",
            evaluation_path=f"results/round_{protocol.round_id:02d}/evaluation_unified.json",
        )
        node = updated["hypothesis_tree"].node_index()["H1"]
        self.assertEqual(node.support_score, 0.5)
        self.assertTrue(
            any(
                entry.event.startswith("pending_scientific_review")
                for entry in node.support_history
            )
        )
        self.assertEqual(node.evidence_items[-1].support_weight, 0.0)

    def _sensitivity_result(
        self,
        *,
        run_cis: list[list[float] | None] | None = None,
        run_points: list[float] | None = None,
    ) -> ExperimentResult:
        run_cis = run_cis or [[0.20, 0.30], [0.15, 0.25]]
        run_points = run_points or [0.25, 0.20]
        result = ExperimentResult(
            experiment_id="E_MS_EVAL",
            round_id=1,
            status="completed",
            experiment_type="model_sensitivity",
            execution=ExecutionSummary(status="completed"),
            runs=[
                ExperimentRun(
                    run_id="ridge_baseline",
                    name="Ridge 对照组",
                    model_family="ridge",
                ),
                ExperimentRun(
                    run_id="ridge_treatment",
                    name="Ridge 实验组",
                    model_family="ridge",
                ),
                ExperimentRun(
                    run_id="elasticnet_baseline",
                    name="ElasticNet 对照组",
                    model_family="elasticnet",
                ),
                ExperimentRun(
                    run_id="elasticnet_treatment",
                    name="ElasticNet 实验组",
                    model_family="elasticnet",
                ),
            ],
            contrasts=[
                ExperimentContrast(
                    contrast_id="gain_ridge",
                    name="Ridge 日影增量",
                    kind="incremental_gain",
                    run_ids=["ridge_baseline", "ridge_treatment"],
                    metrics_delta={"rmse": run_points[0]},
                    bootstrap_ci=BootstrapReport(
                        stable=True,
                        rmse_ci=run_cis[0],
                        point_gain_rmse=run_points[0],
                    ),
                ),
                ExperimentContrast(
                    contrast_id="gain_elasticnet",
                    name="ElasticNet 日影增量",
                    kind="incremental_gain",
                    run_ids=["elasticnet_baseline", "elasticnet_treatment"],
                    metrics_delta={"rmse": run_points[1]},
                    bootstrap_ci=BootstrapReport(
                        stable=True,
                        rmse_ci=run_cis[1],
                        point_gain_rmse=run_points[1],
                    ),
                ),
            ],
        )
        return result

    def _evaluate_sensitivity(
        self,
        *,
        run_cis: list[list[float] | None] | None = None,
        run_points: list[float] | None = None,
    ) -> "object":
        protocol = self._base_protocol("model_sensitivity", "E_MS_EVAL")
        protocol.tested_hypotheses = ["H1"]
        return evaluate_experiment_result(
            self._sensitivity_result(run_cis=run_cis, run_points=run_points),
            protocol=protocol,
        )

    def test_model_sensitivity_evaluator_reports_stable_positive_per_family(self) -> None:
        evaluation = self._evaluate_sensitivity()
        conclusion = evaluation.scientific.three_layer_conclusion
        answer = conclusion.scientific_layer.answer
        self.assertIn("稳定正向增益", answer)
        self.assertIn("Ridge", conclusion.data_layer.rmse_attribution)
        assessments = evaluation.scientific.hypothesis_assessments
        self.assertEqual(len(assessments), 1)
        self.assertFalse(assessments[0].updated)
        self.assertEqual(assessments[0].support_before, assessments[0].support_after)
        self.assertIsNone(assessments[0].direction_matched)
        self.assertEqual(
            assessments[0].evidence_classification,
            "stable_improvement",
        )
        self.assertEqual(len(assessments[0].evidence_labels), 2)
        families = conclusion.experiment_layer.model_sensitivity_families
        self.assertEqual([item.family for item in families], ["ridge", "elasticnet"])
        self.assertEqual([item.evaluation for item in families], ["stable_positive", "stable_positive"])
        self.assertIn("稳定正向增益", families[0].evaluation_label)

    def test_model_sensitivity_evaluator_reports_stable_negative(self) -> None:
        evaluation = self._evaluate_sensitivity(
            run_cis=[[-0.30, -0.20], [-0.25, -0.15]],
            run_points=[-0.25, -0.20],
        )
        answer = evaluation.scientific.three_layer_conclusion.scientific_layer.answer
        self.assertIn("稳定负向增益", answer)

    def test_model_sensitivity_evaluator_marks_ci_crossing_zero_as_no_stable_gain(self) -> None:
        evaluation = self._evaluate_sensitivity(
            run_cis=[[-0.10, 0.20], [0.10, 0.30]],
            run_points=[0.05, 0.20],
        )
        conclusion = evaluation.scientific.three_layer_conclusion
        answer = conclusion.scientific_layer.answer
        self.assertIn("未检出稳定增益", answer)
        self.assertIn("点估计方向一致", answer)
        self.assertEqual(
            [item.evaluation for item in conclusion.experiment_layer.model_sensitivity_families],
            ["no_stable_gain", "stable_positive"],
        )

    def test_model_sensitivity_evaluator_handles_missing_ci_without_auto_support(self) -> None:
        evaluation = self._evaluate_sensitivity(
            run_cis=[[0.10, 0.20], None],
            run_points=[0.15, 0.10],
        )
        answer = evaluation.scientific.three_layer_conclusion.scientific_layer.answer
        self.assertIn("未检出稳定增益", answer)


if __name__ == "__main__":
    unittest.main()
