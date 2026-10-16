import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from core.control_unified import HumanControlService
from core.hypothesis_state_machine import resolve_status
from core.state_repository import UnifiedStateRepository
from core.support_update_rules import compute_support_update_from_reasoning
from core.unified_schema import (
    DataDictionarySummary,
    DecisionLog,
    HypothesisNode,
    HypothesisTreeState,
    LatestTreeUpdate,
    PlannerEvaluationSummary,
    PlannerReasoningTraceItem,
    ProcessState,
    ReasoningPlannerInput,
    SupportHistoryEntry,
)
from tests.test_decision_layer import (
    StubCandidateExperimentDesigner,
    StubCandidateExperimentWriter,
)


class AdvisoryQuestioningRulesTest(unittest.TestCase):
    def test_weakens_without_falsification_basis_becomes_clarifies(self) -> None:
        outcome = compute_support_update_from_reasoning(
            current_support=0.45,
            impact_direction="weakens",
            impact_strength=0.8,
            confidence=0.9,
            evidence_source="advisory",
        )
        self.assertEqual(outcome.support_after, 0.45)
        self.assertEqual(outcome.status, "active")

    def test_advisory_weakens_cannot_cross_prune_threshold_on_first_pass(self) -> None:
        outcome = compute_support_update_from_reasoning(
            current_support=0.25,
            impact_direction="weakens",
            impact_strength=1.0,
            confidence=1.0,
            falsification_basis="与 E_R01 实验残差矛盾",
            previous_status="active",
            evidence_source="advisory",
        )
        self.assertGreaterEqual(outcome.support_after, 0.20)
        self.assertEqual(outcome.status, "observing")

    def test_advisory_observing_never_prunes_on_weak_streak(self) -> None:
        history = [
            SupportHistoryEntry(round=1, score=0.30, event="scientific_questioning"),
            SupportHistoryEntry(round=2, score=0.30, event="scientific_questioning"),
            SupportHistoryEntry(round=3, score=0.30, event="scientific_questioning"),
        ]
        status = resolve_status(
            previous_status="observing",
            support_after=0.30,
            support_history=history,
            current_round=4,
            evidence_source="advisory",
        )
        self.assertEqual(status, "observing")

    def test_advisory_cannot_converge_but_experimental_can(self) -> None:
        history = [
            SupportHistoryEntry(
                round=1,
                score=0.75,
                event="evaluation_writeback:E1",
                evidence_source="experimental",
                evidence_classification="stable_improvement",
                direction_matched=True,
            ),
        ]
        advisory_status = resolve_status(
            previous_status="active",
            support_after=0.78,
            support_history=history,
            current_round=2,
            evidence_source="advisory",
            evidence_classification="stable_improvement",
            direction_matched=True,
        )
        self.assertEqual(advisory_status, "active")

        experimental_status = resolve_status(
            previous_status="active",
            support_after=0.78,
            support_history=history,
            current_round=2,
            evidence_source="experimental",
            evidence_classification="stable_improvement",
            direction_matched=True,
        )
        self.assertEqual(experimental_status, "converged")


class UndoScientificQuestioningTest(unittest.TestCase):
    def _seed_repository(self, root: Path) -> UnifiedStateRepository:
        repo = UnifiedStateRepository(root)
        pre_tree = HypothesisTreeState(
            tree_id="ST_ADVISORY_tree",
            task_id="ST_ADVISORY",
            current_round=1,
            root_question="太阳风速度是否由行星际磁场主导",
            latest_update=LatestTreeUpdate(
                round=1,
                event="hypothesis_tree_confirmed",
                description="确认后冻结",
            ),
            nodes=[
                HypothesisNode(
                    hypothesis_id="H1",
                    display_hypothesis_id="H1",
                    statement="By 对 Vsw 存在独立预测增益",
                    level=1,
                    status="active",
                    support_score=0.55,
                    activation_condition="父假设支持度 >= 0.40",
                )
            ],
            active_hypotheses=["H1"],
        )
        repo.save_hypothesis_tree(
            pre_tree.model_copy(
                update={
                    "nodes": [
                        node.model_copy(update={"support_score": 0.21, "status": "observing"})
                        for node in pre_tree.nodes
                    ],
                    "latest_update": LatestTreeUpdate(
                        round=1,
                        event="scientific_questioning_completed",
                        description="质询完成",
                    ),
                }
            )
        )
        repo.save_pre_questioning_tree(pre_tree)
        repo.save_process_state(
            ProcessState(
                task_id="ST_ADVISORY",
                current_round=1,
                current_stage="awaiting_uncertainty_identification",
                current_phase="hypothesis_generation",
                current_step="step_1",
                progress_percentage=55,
                started_at=datetime.now(),
            )
        )
        repo.save_planner_input(
            ReasoningPlannerInput(
                task_id="ST_ADVISORY",
                source_round_id=1,
                next_round_id=1,
                scientific_question="太阳风速度是否由行星际磁场主导",
                target="Vsw",
                evaluation_summary=PlannerEvaluationSummary(round_id=1),
                data_dictionary_summary=DataDictionarySummary(
                    dictionary_id="DICT_ADVISORY",
                    dataset_name="demo",
                    time_column="TIME",
                    target_candidates=["Vsw"],
                    feature_candidates=["By"],
                ),
                recent_reasoning_traces=[
                    PlannerReasoningTraceItem(
                        trace_id="SQ0101",
                        stage="scientific_questioner",
                        summary="质询追踪",
                    ),
                    PlannerReasoningTraceItem(
                        trace_id="HP001",
                        stage="hypothesis_proposer",
                        summary="假设追踪",
                    ),
                ],
                planner_guidance=["supplemental_llm_hypothesis:H5 补充生成"],
            )
        )
        repo.save_decision_log(
            DecisionLog(
                task_id="ST_ADVISORY",
                decisions=[],
            )
        )
        return repo

    def test_undo_restores_tree_and_backs_out_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._seed_repository(Path(tmp))
            control = HumanControlService(
                repo,
                project_root=Path(tmp),
                experiment_designer=StubCandidateExperimentDesigner(),
                experiment_writer=StubCandidateExperimentWriter(),
            )
            result = control.undo_scientific_questioning()

            self.assertEqual(result["status"], "awaiting_scientific_questioning")
            process = repo.load_process_state()
            self.assertEqual(process.current_stage, "awaiting_scientific_questioning")
            self.assertEqual(process.current_step, "scientific_questioning")
            tree = repo.load_hypothesis_tree()
            self.assertEqual(tree.nodes[0].support_score, 0.55)
            self.assertEqual(tree.nodes[0].status, "active")
            self.assertFalse(tree.nodes[0].questioning_records)
            planner_input = repo.load_planner_input()
            self.assertEqual(
                [trace.stage for trace in planner_input.recent_reasoning_traces],
                ["hypothesis_proposer"],
            )
            self.assertTrue(
                all(
                    not note.startswith("supplemental_llm_hypothesis:")
                    for note in planner_input.planner_guidance
                )
            )
            decision_log = repo.load_decision_log()
            self.assertEqual(
                decision_log.decisions[-1].decision_type,
                "scientific_questioning_undone",
            )


if __name__ == "__main__":
    unittest.main()
