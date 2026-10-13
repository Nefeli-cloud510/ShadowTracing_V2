import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from core.evaluation_unified import evaluate_experiment
from core.state_repository import UnifiedStateRepository
from core.state_updater import UnifiedStateUpdater
from core.unified_schema import (
    EvaluationSpec,
    ExecutionSummary,
    ExperimentDesign,
    ExperimentProtocol,
    ExperimentResult,
    ExperimentRun,
    HypothesisNode,
    MetricDelta,
    ResearchQuestion,
    ScientificConstraints,
    ScientificTask,
    ScientificTaskPayload,
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


if __name__ == "__main__":
    unittest.main()
