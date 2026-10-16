import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from core.control_unified import HumanControlService
from core.evaluation_unified import evaluate_experiment
from core.scientific_interpreter_llm import InterpreterRoundAnalysis
from core.state_repository import UnifiedStateRepository
from core.state_updater import UnifiedStateUpdater
from core.unified_schema import (
    BootstrapReport,
    ConclusionDataLayer,
    EvaluationSpec,
    ExecutionSummary,
    ExperimentDesign,
    ExperimentProtocol,
    ExperimentResult,
    ExperimentRun,
    HypothesisNode,
    HypothesisPrediction,
    InterpretationEnhancement,
    MetricDelta,
    ResearchQuestion,
    RobustnessOverall,
    RobustnessReport,
    ScientificConstraints,
    ScientificTask,
    ScientificTaskPayload,
    TimeSliceReport,
    VariableBinding,
)


class StateWritebackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.repository = UnifiedStateRepository(self.project_root)

        task = ScientificTask(
            task_id="ST_100",
            payload=ScientificTaskPayload(
                research_question=ResearchQuestion(
                    text="DeltaDec 是否改善 Vsw 预测",
                    target="Vsw",
                    question_type="forecasting",
                    variables=VariableBinding(x="DeltaDec", y="Vsw", m_candidates=["By"]),
                ),
                evaluation=EvaluationSpec(primary_metric="RMSE", secondary_metrics=["Pearson_r"]),
                constraints=ScientificConstraints(),
            ),
        )
        initial_hypotheses = [
            HypothesisNode(
                hypothesis_id="H1",
                statement="DeltaDec 带来独立增量",
                level=1,
                status="active",
                support_score=0.5,
                children_ids=["H1a"],
                activation_condition="always",
            ),
            HypothesisNode(
                hypothesis_id="H1a",
                statement="DeltaDec 通过 By 中介增强预测",
                level=2,
                parent_id="H1",
                status="pending",
                support_score=0.36,
                activation_condition="父假设支持度>=0.40",
            )
        ]
        self.repository.initialize_state_skeleton(
            task,
            initial_hypotheses=initial_hypotheses,
            overwrite=True,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_apply_evaluation_writeback(self) -> None:
        protocol = ExperimentProtocol(
            experiment_id="E100",
            round_id=1,
            task_id="ST_100",
            target="Vsw",
            features=ExperimentDesign(
                target="Vsw",
                control=["By", "Bz"],
                treatment=["By", "Bz", "DeltaDec"],
            ),
            tested_hypotheses=["H1"],
        )
        result = ExperimentResult(
            experiment_id="E100",
            round_id=1,
            status="completed",
            runs=[
                ExperimentRun(run_id="baseline", name="baseline"),
                ExperimentRun(run_id="treatment", name="treatment"),
            ],
            comparison=MetricDelta(rmse=1.2, pearson_r=0.06),
            execution=ExecutionSummary(status="completed", duration_seconds=4.0),
        )
        evaluation = evaluate_experiment(
            result,
            protocol=protocol,
            remaining_uncertainties=[
                {
                    "question": "DeltaDec 的增益是否跨时间段稳定",
                    "priority": "high",
                }
            ],
        )

        updater = UnifiedStateUpdater(self.repository)
        updated = updater.apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
            protocol_path="config/latest_protocol.json",
            result_path="results/round_01/result_unified.json",
            evaluation_path="results/round_01/evaluation_unified.json",
        )

        tree = updated["hypothesis_tree"]
        uncertainties = updated["uncertainties"]
        experiment_memory = updated["experiment_memory"]
        process_state = updated["process_state"]

        self.assertTrue((self.project_root / "state" / "hypothesis_tree.json").exists())
        self.assertTrue((self.project_root / "state" / "uncertainties.json").exists())
        self.assertTrue((self.project_root / "state" / "uncertainty_priority.json").exists())
        self.assertTrue((self.project_root / "state" / "experiment_memory.json").exists())
        self.assertTrue((self.project_root / "state" / "process.json").exists())
        self.assertTrue((self.project_root / "state" / "task.json").exists())
        self.assertTrue((self.project_root / "state" / "decision_log.json").exists())
        # 没有 bootstrap CI 时点估计不能构成稳定证据，支持度保持不变。
        self.assertAlmostEqual(tree.node_index()["H1"].support_score, 0.5)
        # 父假设达标只解锁子假设参与资格，子节点支持度 0.36 仍进入观察期。
        self.assertEqual(tree.node_index()["H1a"].status, "observing")
        self.assertEqual(tree.node_index()["H1a"].activated_at_round, 1)
        self.assertEqual(tree.latest_update.event, "evaluation_writeback")
        self.assertEqual(uncertainties.records[0].priority, "high")
        self.assertEqual(uncertainties.priority_queue.queue[0].priority_score, 0.9)
        self.assertEqual(experiment_memory.entries[0].evaluation_path, "results/round_01/evaluation_unified.json")
        self.assertEqual(process_state.current_stage, "knowledge_memory_updated")
        self.assertEqual(process_state.steps["state_writeback"].status, "completed")

    def _build_protocol_result_evaluation(self) -> tuple:
        protocol = ExperimentProtocol(
            experiment_id="E100",
            round_id=1,
            task_id="ST_100",
            target="Vsw",
            features=ExperimentDesign(
                target="Vsw",
                control=["By", "Bz"],
                treatment=["By", "Bz", "DeltaDec"],
            ),
            tested_hypotheses=["H1"],
        )
        result = ExperimentResult(
            experiment_id="E100",
            round_id=1,
            status="completed",
            run_id="treatment_v1",
            runs=[
                ExperimentRun(run_id="baseline", name="baseline"),
                ExperimentRun(run_id="treatment", name="treatment"),
            ],
            comparison=MetricDelta(rmse=1.2, pearson_r=0.06),
            execution=ExecutionSummary(status="completed", duration_seconds=4.0),
        )
        evaluation = evaluate_experiment(
            result,
            protocol=protocol,
            remaining_uncertainties=[
                {"question": "DeltaDec 的增益是否跨时间段稳定", "priority": "high"}
            ],
        )
        return protocol, result, evaluation

    def test_apply_evaluation_is_idempotent(self) -> None:
        protocol, result, evaluation = self._build_protocol_result_evaluation()
        protocol_file = self.project_root / "config" / "latest_protocol.json"
        protocol_file.parent.mkdir(parents=True, exist_ok=True)
        protocol_file.write_text(json.dumps({"approved": True}), encoding="utf-8")
        updater = UnifiedStateUpdater(self.repository)
        updater.apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
        )
        tree = self.repository.load_hypothesis_tree()
        node = tree.node_index()["H1"]
        history_count = len(node.support_history)
        evidence_count = len(node.evidence_items)
        self.assertEqual(history_count, 1)
        self.assertEqual(evidence_count, 1)

        # 同一份证据重复应用不改变状态、不追加记录。
        updater.apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
        )
        tree = self.repository.load_hypothesis_tree()
        node = tree.node_index()["H1"]
        self.assertEqual(len(node.support_history), history_count)
        self.assertEqual(len(node.evidence_items), evidence_count)
        self.assertEqual(node.support_score, 0.5)

        commit = self.repository.load_update_commit()
        self.assertEqual(
            commit["experiments"]["E100"]["update_types"],
            ["evaluation"],
        )
        # 已批准协议不被 state_updater 覆盖。
        self.assertEqual(json.loads(protocol_file.read_text(encoding="utf-8")), {"approved": True})
        core_outputs = self.project_root / "config" / "latest_protocol.core_outputs.json"
        self.assertTrue(core_outputs.exists())
        self.assertTrue(
            json.loads(core_outputs.read_text(encoding="utf-8")).get("exploratory")
        )

    def test_corrected_evidence_replaces_previous_version(self) -> None:
        protocol, result, evaluation = self._build_protocol_result_evaluation()
        updater = UnifiedStateUpdater(self.repository)
        updater.apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
        )

        tree = self.repository.load_hypothesis_tree()
        initial_version = tree.node_index()["H1"].support_history[0].evidence_version

        corrected = result.model_copy(
            update={
                "comparison": MetricDelta(rmse=2.0, pearson_r=0.12),
            }
        )
        corrected_evaluation = evaluate_experiment(
            corrected,
            protocol=protocol,
            remaining_uncertainties=[
                {"question": "DeltaDec 的增益是否跨时间段稳定", "priority": "high"}
            ],
        )
        updater.apply_evaluation(
            protocol=protocol,
            result=corrected,
            evaluation=corrected_evaluation,
        )

        tree = self.repository.load_hypothesis_tree()
        node = tree.node_index()["H1"]
        self.assertEqual(len(node.support_history), 1, "更正证据应替换而不是追加一票")
        self.assertEqual(len(node.evidence_items), 1)
        self.assertEqual(node.support_history[0].run_id, "treatment_v1")
        self.assertNotEqual(
            node.support_history[0].evidence_version,
            initial_version,
            "更正后的证据应使用新的 evidence_version",
        )

    def _build_protocol_result_evaluation_with_ci(
        self,
        *,
        ci: list[float],
        rmse_delta: float | None,
    ) -> tuple[ExperimentProtocol, ExperimentResult, object]:
        protocol = ExperimentProtocol(
            experiment_id="E100",
            round_id=1,
            task_id="ST_100",
            target="Vsw",
            features=ExperimentDesign(
                target="Vsw",
                control=["By", "Bz"],
                treatment=["By", "Bz", "DeltaDec"],
            ),
            tested_hypotheses=["H1"],
            hypothesis_predictions={
                "H1": HypothesisPrediction(
                    expected_effect="positive",
                    expected_range=[0.02, 1.0],
                )
            },
        )
        result = ExperimentResult(
            experiment_id="E100",
            round_id=1,
            status="completed",
            runs=[
                ExperimentRun(run_id="baseline", name="baseline"),
                ExperimentRun(run_id="treatment", name="treatment"),
            ],
            comparison=MetricDelta(rmse=rmse_delta),
            execution=ExecutionSummary(status="completed", duration_seconds=4.0),
            robustness=RobustnessReport(
                bootstrap=BootstrapReport(rmse_ci=ci, stable=False),
                time_slice=TimeSliceReport(splits=[], stable=False),
                overall=RobustnessOverall(stable=False),
            ),
        )
        evaluation = evaluate_experiment(
            result,
            protocol=protocol,
            remaining_uncertainties=[
                {"question": "DeltaDec 的增益是否跨时间段稳定", "priority": "high"}
            ],
        )
        return protocol, result, evaluation

    def _strong_support_enhancement(self) -> InterpretationEnhancement:
        return InterpretationEnhancement(
            enhancement_id="LIE_INT_001",
            target_hypothesis_id="H1",
            interpretation="科学解释者认为日影增量信号清晰，假设应被强支持。",
            impact_direction="supports",
            impact_strength=1.0,
            confidence=1.0,
        )

    def test_inconclusive_ci_blocks_interpreter_support_writeback(self) -> None:
        protocol, result, evaluation = self._build_protocol_result_evaluation_with_ci(
            ci=[-0.2, 0.3],
            rmse_delta=0.05,
        )
        assessment = evaluation.scientific.hypothesis_assessments[0]
        self.assertEqual(assessment.evidence_classification, "inconclusive")
        self.assertFalse(assessment.updated)

        updater = UnifiedStateUpdater(self.repository)
        updated = updater.apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
            interpretation_enhancements=[self._strong_support_enhancement()],
            protocol_path="config/latest_protocol.json",
            result_path="results/round_01/result_unified.json",
            evaluation_path="results/round_01/evaluation_unified.json",
        )
        node = updated["hypothesis_tree"].node_index()["H1"]
        self.assertAlmostEqual(node.support_score, 0.5)
        self.assertFalse(
            any(
                getattr(entry, "evidence_key", "").startswith("E100:enhancement:")
                for entry in node.support_history
            ),
            "CI 跨零时解释者不得写入增强支持度历史",
        )
        self.assertEqual(
            node.evidence_items[-1].support_weight,
            0.0,
            "解释者意见只作为顾问文本记录，不参与支持度计算",
        )
        self.assertEqual(node.critiques[-1].from_role, "scientific_interpreter")
        self.assertTrue(node.alternative_explanations)

    def test_numeric_degradation_beats_strong_interpreter_support(self) -> None:
        protocol, result, evaluation = self._build_protocol_result_evaluation_with_ci(
            ci=[-0.4, -0.1],
            rmse_delta=-0.5,
        )
        assessment = evaluation.scientific.hypothesis_assessments[0]
        self.assertEqual(assessment.evidence_classification, "stable_degradation")
        self.assertTrue(assessment.updated)

        updater = UnifiedStateUpdater(self.repository)
        updated = updater.apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
            interpretation_enhancements=[self._strong_support_enhancement()],
            protocol_path="config/latest_protocol.json",
            result_path="results/round_01/result_unified.json",
            evaluation_path="results/round_01/evaluation_unified.json",
        )
        node = updated["hypothesis_tree"].node_index()["H1"]
        self.assertAlmostEqual(node.support_score, 0.45)
        self.assertFalse(
            any(
                getattr(entry, "evidence_key", "").startswith("E100:enhancement:")
                for entry in node.support_history
            ),
            "数值证据退化后，解释者强支持不得反向拉高支持度",
        )

    def test_conclusion_overwrite_blocked_for_conservative_evidence(self) -> None:
        protocol, result, evaluation = self._build_protocol_result_evaluation_with_ci(
            ci=[-0.2, 0.3],
            rmse_delta=0.05,
        )
        program_answer = evaluation.scientific.three_layer_conclusion.scientific_layer.answer
        program_path_answer = (
            evaluation.scientific.three_layer_conclusion.scientific_layer.path_answer
        )
        analysis = InterpreterRoundAnalysis(
            enhancements=[],
            data_layer=ConclusionDataLayer(
                rmse_attribution="实际存在明显稳定增益，RMSE 持续下降。",
                anomalies=["无"],
            ),
            conclusion_text="能，确实有稳定增益，日影显著改善太阳风速度预测。",
            path_answer="路径已确认，通过 By 独立传递。",
            hypothesis_layer_summary="解释者综合意见。",
            overall_summary="解释者总体总结。",
        )

        HumanControlService._apply_interpreter_analysis_to_conclusion(
            evaluation,
            analysis,
            protocol=protocol,
        )
        conclusion = evaluation.scientific.three_layer_conclusion
        self.assertEqual(conclusion.scientific_layer.answer, program_answer)
        self.assertEqual(conclusion.scientific_layer.path_answer, program_path_answer)
        self.assertEqual(conclusion.hypothesis_layer_summary, "解释者综合意见。")
        self.assertEqual(conclusion.overall_summary, "解释者总体总结。")

    def test_conclusion_overwrite_allowed_for_decisive_evidence(self) -> None:
        protocol, result, evaluation = self._build_protocol_result_evaluation_with_ci(
            ci=[0.1, 0.4],
            rmse_delta=0.5,
        )
        analysis = InterpreterRoundAnalysis(
            enhancements=[],
            data_layer=ConclusionDataLayer(
                rmse_attribution="LLM 对四图与两组对比的数据层分析。",
                anomalies=["无"],
            ),
            conclusion_text="能，bootstrap RMSE 增益 CI 整体为正，存在稳定前置信息。",
            path_answer="控制 By 后仍观察到独立增量。",
        )

        HumanControlService._apply_interpreter_analysis_to_conclusion(
            evaluation,
            analysis,
            protocol=protocol,
        )
        conclusion = evaluation.scientific.three_layer_conclusion
        self.assertEqual(
            conclusion.scientific_layer.answer,
            "能，bootstrap RMSE 增益 CI 整体为正，存在稳定前置信息。",
        )
        self.assertEqual(
            conclusion.scientific_layer.path_answer,
            "控制 By 后仍观察到独立增量。",
        )

    def _apply_ci_evaluation(
        self,
        protocol: ExperimentProtocol,
        result: ExperimentResult,
        evaluation: object,
    ) -> dict[str, object]:
        updater = UnifiedStateUpdater(self.repository)
        return updater.apply_evaluation(
            protocol=protocol,
            result=result,
            evaluation=evaluation,
            protocol_path="config/latest_protocol.json",
            result_path="results/round_01/result_unified.json",
            evaluation_path="results/round_01/evaluation_unified.json",
        )

    def _evaluate_with_ci(
        self,
        protocol: ExperimentProtocol,
        result: ExperimentResult,
    ) -> object:
        return evaluate_experiment(
            result,
            protocol=protocol,
            remaining_uncertainties=[
                {"question": "DeltaDec 的增益是否跨时间段稳定", "priority": "high"}
            ],
        )

    def test_metadata_only_change_does_not_reapply_support(self) -> None:
        protocol, result, evaluation = self._build_protocol_result_evaluation_with_ci(
            ci=[0.1, 0.4],
            rmse_delta=0.5,
        )
        self._apply_ci_evaluation(protocol, result, evaluation)
        tree = self.repository.load_hypothesis_tree()
        node = tree.node_index()["H1"]
        self.assertAlmostEqual(node.support_score, 0.55)
        self.assertEqual(len(node.support_history), 1)
        first_identity = node.support_history[0].evidence_identity
        first_version = node.support_history[0].evidence_version

        metadata_changed = result.model_copy(
            update={
                "warnings": ["重新生成报告时补充的警告文字"],
                "execution": result.execution.model_copy(
                    update={"message": "人工复核后重新导出"}
                ),
            }
        )
        new_evaluation = self._evaluate_with_ci(protocol, metadata_changed)
        self._apply_ci_evaluation(protocol, metadata_changed, new_evaluation)

        tree = self.repository.load_hypothesis_tree()
        node = tree.node_index()["H1"]
        self.assertAlmostEqual(node.support_score, 0.55)
        self.assertEqual(len(node.support_history), 1)
        entry = node.support_history[0]
        self.assertEqual(entry.evidence_identity, first_identity)
        self.assertNotEqual(entry.evidence_version, first_version)

    def test_scientific_revision_replaces_previous_contribution(self) -> None:
        protocol, result, evaluation = self._build_protocol_result_evaluation_with_ci(
            ci=[0.1, 0.4],
            rmse_delta=0.5,
        )
        self._apply_ci_evaluation(protocol, result, evaluation)
        tree = self.repository.load_hypothesis_tree()
        self.assertAlmostEqual(tree.node_index()["H1"].support_score, 0.55)

        revised = result.model_copy(
            update={
                "comparison": MetricDelta(rmse=-0.5),
                "robustness": RobustnessReport(
                    bootstrap=BootstrapReport(rmse_ci=[-0.4, -0.1], stable=False),
                    time_slice=TimeSliceReport(splits=[], stable=False),
                    overall=RobustnessOverall(stable=False),
                ),
            }
        )
        revised_evaluation = self._evaluate_with_ci(protocol, revised)
        self._apply_ci_evaluation(protocol, revised, revised_evaluation)

        tree = self.repository.load_hypothesis_tree()
        node = tree.node_index()["H1"]
        self.assertAlmostEqual(
            node.support_score,
            0.45,
            "科学结论修订应撤销旧贡献并只应用最终版本",
        )
        self.assertEqual(len(node.support_history), 1)
        entry = node.support_history[0]
        self.assertAlmostEqual(entry.support_before, 0.5)
        self.assertAlmostEqual(entry.support_delta, -0.05)

    def test_revision_to_inconclusive_reverts_to_original_support(self) -> None:
        protocol, result, evaluation = self._build_protocol_result_evaluation_with_ci(
            ci=[0.1, 0.4],
            rmse_delta=0.5,
        )
        self._apply_ci_evaluation(protocol, result, evaluation)
        tree = self.repository.load_hypothesis_tree()
        self.assertAlmostEqual(tree.node_index()["H1"].support_score, 0.55)

        inconclusive = result.model_copy(
            update={
                "comparison": MetricDelta(rmse=0.05),
                "robustness": RobustnessReport(
                    bootstrap=BootstrapReport(rmse_ci=[-0.2, 0.3], stable=False),
                    time_slice=TimeSliceReport(splits=[], stable=False),
                    overall=RobustnessOverall(stable=False),
                ),
            }
        )
        inconclusive_evaluation = self._evaluate_with_ci(protocol, inconclusive)
        self._apply_ci_evaluation(protocol, inconclusive, inconclusive_evaluation)

        node = self.repository.load_hypothesis_tree().node_index()["H1"]
        self.assertAlmostEqual(
            node.support_score,
            0.5,
            "修订为 CI 跨零时，旧贡献应撤销且不产生新加分",
        )
        self.assertEqual(len(node.support_history), 1)
        self.assertAlmostEqual(node.support_history[0].support_delta, 0.0)


if __name__ == "__main__":
    unittest.main()
