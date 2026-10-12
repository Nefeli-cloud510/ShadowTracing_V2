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
    get_tool,
)
from core.experiment_tools.validate_protocol import validate_experiment_protocol
from core.harness_unified import UnifiedExperimentHarness
from core.protocol_bridge import ElasticNetDataSourceConfig
from core.state_repository import UnifiedStateRepository
from core.state_updater import UnifiedStateUpdater
from core.unified_schema import (
    ContrastSpec,
    DiagnosticSpec,
    EvaluationSpec,
    ExecutionSummary,
    ExperimentDesign,
    ExperimentProtocol,
    ExperimentResult,
    ExperimentRun,
    HypothesisNode,
    ModelSpec,
    ResearchQuestion,
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
        self.assertTrue(get_tool("model_sensitivity").available)
        self.assertTrue(get_tool("robustness_diagnostic").available)
        for experiment_type in [
            "baseline_redundancy",
            "temporal_sensitivity",
            "conditional_effect",
        ]:
            self.assertFalse(get_tool(experiment_type).available)

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
        self.assertGreaterEqual(len(result.diagnostics), 3)
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

        self.assertEqual(evaluation.scientific.hypothesis_assessments, [])
        self.assertIn(
            "待科学评价",
            evaluation.scientific.three_layer_conclusion.data_layer.rmse_attribution,
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


if __name__ == "__main__":
    unittest.main()
