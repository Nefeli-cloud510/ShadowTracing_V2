import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.central_controller_llm import CentralControllerLLM
from core.control_unified import HumanControlService
from core.decision_unified import DecisionLayerService
from core.hypothesis_generation import HypothesisGenerationResult
from core.hypothesis_proposer_llm import HypothesisProposerResponse
from core.planner_unified import PlannerOutputBuilder
from core.rag_service import RAGContextBundle, RAGEvidenceSnippet
from core.round_orchestrator import RoundOrchestrator
from core.scientific_questioner_llm import (
    HypothesisQuestioningUpdate,
    ProposedUncertainty,
    ScientificQuestionerResponse,
)
from core.state_repository import UnifiedStateRepository
from core.unified_schema import (
    ClosureChecklistItem,
    DataDictionary,
    EvaluationSpec,
    ExperimentStep,
    FieldDescriptor,
    HypothesisNode,
    ProtocolRefinementSuggestion,
    ProcessPhase,
    ProcessStep,
    ResearchQuestion,
    RoundHistoryEntry,
    ScientificConstraints,
    ScientificTask,
    ScientificTaskPayload,
    UncertaintyRecord,
    VariableBinding,
)
from tests.test_decision_layer import (
    StubCandidateExperimentDesigner,
    StubCandidateExperimentWriter,
)


class StubLLMGateway:
    def generate_structured(
        self,
        *,
        system_prompt,
        user_prompt,
        response_model,
        fallback_factory=None,
        payload_fixer=None,
        temperature=None,
        image_paths=None,
    ):
        return response_model.model_validate(
            {
                "summary": "LLM 中央控制器已基于上一轮结果生成下一轮聚焦建议。",
                "candidate_focus_ids": ["E_R02_01"],
                "candidate_rationale": "优先围绕 By 的单变量验证路径做收敛式确认。",
                "target_hypothesis_id": "H1",
                "interpretation": "H1 在当前结果下获得初步支持，但仍需通过更聚焦的 By 路径验证稳健性。",
                "related_uncertainty_ids": ["U01"],
                "impact_direction": "supports",
                "impact_strength": 0.62,
                "confidence": 0.81,
                "uncertainty_priority_action": "decrease",
                "feature_focus": ["By"],
                "protocol_notes": ["llm_requests_by_validation"],
                "suggested_model_parameters": {"alpha": 0.18, "l1_ratio": 0.62},
            }
        )


class StubRAGService:
    def build_context_bundle(self, planner_input, **kwargs):
        return RAGContextBundle(
            query=planner_input.scientific_question,
            project_evidence=[
                RAGEvidenceSnippet(
                    source_type="project_memory",
                    source_id="decision_log",
                    title="decision_log.json",
                    excerpt="项目历史显示 By 路径曾多次被人工要求优先验证。",
                    score=0.9,
                    citation="state/decision_log.json",
                )
            ],
            literature_evidence=[
                RAGEvidenceSnippet(
                    source_type="literature_memory",
                    source_id="planner_doc",
                    title="推理规划层_中央与假设+质询llm.txt",
                    excerpt="文档强调中央控制器需要结合知识库证据识别关键科学不确定性。",
                    score=0.86,
                    citation="outputs/pdf_texts/推理规划层_中央与假设+质询llm.txt",
                )
            ],
        )


class StubHypothesisProposer:
    def propose(self, *, planner_input, rag_context):
        return HypothesisProposerResponse(
            focus_features=["By"],
            guidance_notes=["proposer_focus:By 条件路径值得补充为下一轮假设。"],
            proposal_summaries=["By 条件路径可能决定 DeltaDec 增量是否成立。"],
        )


class StubScientificQuestioner:
    def challenge_hypothesis_tree(
        self,
        *,
        planner_input,
        rag_context,
        tree,
        mined_candidates=None,
        targeted_hypothesis_ids=None,
    ):
        return ScientificQuestionerResponse(
            challenge_points=["当前仍需确认 DeltaDec 的增量是否独立于 By。"],
            guidance_notes=["questioner_gap:需要区分独立信息与中介解释。"],
            proposed_uncertainties=[
                ProposedUncertainty(
                    question="DeltaDec 的增量是否独立于 By？",
                    description="需要区分独立信息与中介解释。",
                    priority="high",
                )
            ],
            hypothesis_updates=[
                HypothesisQuestioningUpdate(
                    hypothesis_id=node.hypothesis_id,
                    impact_direction="supports",
                    impact_strength=0.55,
                    confidence=0.8,
                    rationale="质询认为当前证据仍偏向支持该假设。",
                )
                for node in tree.nodes
                if targeted_hypothesis_ids is None
                or node.hypothesis_id in targeted_hypothesis_ids
            ],
        )

    def question(self, *, planner_input, rag_context, mined_candidates=None):
        return ScientificQuestionerResponse(
            challenge_points=["当前仍需确认 DeltaDec 的增量是否独立于 By。"],
            guidance_notes=["questioner_gap:需要区分独立信息与中介解释。"],
            proposed_uncertainties=[
                ProposedUncertainty(
                    question="DeltaDec 的增量是否独立于 By？",
                    description="需要区分独立信息与中介解释。",
                    priority="high",
                )
            ],
        )


class StubExperimentPlanner:
    def refine(
        self,
        *,
        task,
        candidate,
        existing_refinements=None,
        semantic_service=None,
        history_context=None,
    ):
        return ProtocolRefinementSuggestion(
            suggestion_id="EPL_STUB_001",
            target_candidate_id=candidate.experiment_id,
            refinement_type="实验规划者llm",
            rationale=f"为 {candidate.experiment_id} 增加协议级焦点检查。",
            suggested_model_parameters={"alpha": 0.2},
            suggested_feature_focus=candidate.design.treatment[:1],
            suggested_steps=[
                ExperimentStep(
                    step=1,
                    action="planner_hypothesis_focus",
                    parameters={"candidate_id": candidate.experiment_id},
                )
            ],
            protocol_notes=["stub_experiment_planner_note"],
        )


class StubHypothesisGenerator:
    """Offline tree pass-through so round bootstrap can run without LLM proposals."""

    def __init__(self, repository):
        self.repository = repository

    def build_tree(self, **kwargs):
        tree = self.repository.load_hypothesis_tree()
        uncertainties = self.repository.load_uncertainties()
        return HypothesisGenerationResult(
            tree=tree,
            generated_node_ids=(),
            updated_uncertainties=tuple(uncertainties.records),
            mode="llm_next_round",
            source="test_stub",
        )


class HumanControlFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        (self.project_root / "data" / "raw").mkdir(parents=True, exist_ok=True)
        self.repository = UnifiedStateRepository(self.project_root)
        self._create_demo_data()
        self.task = ScientificTask(
            task_id="ST_300",
            payload=ScientificTaskPayload(
                research_question=ResearchQuestion(
                    text="DeltaDec 是否包含能够改善 Vsw 预测的独立信息",
                    target="Vsw",
                    question_type="forecasting",
                    variables=VariableBinding(x="DeltaDec", y="Vsw", m_candidates=["By", "Bz"]),
                    keywords=["DeltaDec", "Vsw"],
                    clarified=True,
                ),
                evaluation=EvaluationSpec(
                    primary_metric="Pearson_r",
                    secondary_metrics=["RMSE", "MAE"],
                    visual_analysis=["prediction_vs_truth", "scatter_plot"],
                ),
                constraints=ScientificConstraints(
                    no_future_information=True,
                    validation_feedback_allowed=True,
                    final_test_blind=True,
                    max_rounds=10,
                    max_experiments_per_round=3,
                    resource_budget={
                        "token_budget": 100000,
                        "max_time_seconds_per_round": 300,
                        "max_candidates": 3,
                    },
                ),
                data_sources={
                    "omni": {"path": "data/raw/omni.csv", "time_column": "TIME"},
                    "lhaaso": {"path": "data/raw/lhaaso.csv", "time_column": "TIME"},
                },
            ),
        )
        self.dictionary = DataDictionary(
            dictionary_id="DICT_300",
            generated_at=datetime.now(),
            version="1.0",
            dataset_name="demo",
            total_samples=128,
            time_column="TIME",
            time_format="YYYYMMDD",
            fields=[
                FieldDescriptor(field_name="TIME", data_type="int", physical_meaning="time"),
                FieldDescriptor(field_name="Vsw", data_type="float", physical_meaning="target", is_target=True),
                FieldDescriptor(field_name="By", data_type="float", physical_meaning="IMF By"),
                FieldDescriptor(field_name="Bz", data_type="float", physical_meaning="IMF Bz"),
                FieldDescriptor(field_name="DeltaDec", data_type="float", physical_meaning="LHAASO DeltaDec"),
            ],
            target_candidates=["Vsw"],
            feature_candidates=["By", "Bz", "DeltaDec"],
        )
        self.repository.initialize_state_skeleton(
            self.task,
            initial_hypotheses=[
                HypothesisNode(
                    hypothesis_id="H1",
                    statement="DeltaDec 提供独立增量信息",
                    level=1,
                    status="active",
                    support_score=0.6,
                    activation_condition="always",
                ),
                HypothesisNode(
                    hypothesis_id="H2",
                    statement="DeltaDec 主要通过 By 起作用",
                    level=1,
                    status="active",
                    support_score=0.45,
                    activation_condition="always",
                ),
            ],
            initial_uncertainties=[
                UncertaintyRecord(
                    uncertainty_id="U01",
                    question="DeltaDec 的贡献是否独立于 By？",
                    description="需要区分独立增量与中介解释。",
                    related_hypotheses=["H1", "H2"],
                    status="active",
                    priority="high",
                    created_at_round=1,
                    created_by="tester",
                ),
                UncertaintyRecord(
                    uncertainty_id="U02",
                    question="DeltaDec 的增益是否跨时间段稳定？",
                    description="需要检验滞后稳定性。",
                    related_hypotheses=["H1"],
                    status="active",
                    priority="medium",
                    created_at_round=1,
                    created_by="tester",
                ),
            ],
            overwrite=True,
        )
        uncertainties = self.repository.load_uncertainties()
        uncertainties.current_round = 1
        self.repository.save_uncertainties(uncertainties)
        DecisionLayerService(
            self.repository,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        ).build_candidate_plan(task=self.task, data_dictionary=self.dictionary)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_demo_data(self) -> None:
        time_values = pd.date_range("2021-01-01", periods=36, freq="D")
        omni = pd.DataFrame(
            {
                "TIME": time_values.strftime("%Y%m%d"),
                "By": [(-1) ** i * (i % 5 + 1) for i in range(36)],
                "Bz": [(i % 7) - 3 for i in range(36)],
                "Vsw": [350 + i * 2 for i in range(36)],
            }
        )
        lhaaso = pd.DataFrame(
            {
                "TIME": time_values.strftime("%Y%m%d"),
                "DeltaDec": [0.1 * ((i % 6) - 3) for i in range(36)],
            }
        )
        omni.to_csv(self.project_root / "data" / "raw" / "omni.csv", index=False)
        lhaaso.to_csv(self.project_root / "data" / "raw" / "lhaaso.csv", index=False)

    def _mark_round_closed(self, round_id: int) -> None:
        """Simulate a completed execution+evaluation cycle for short-circuit round tests."""
        history = self.repository.load_round_history()
        entry = next((item for item in history.entries if item.round_id == round_id), None)
        if entry is None:
            entry = RoundHistoryEntry(round_id=round_id)
            history.entries.append(entry)
        entry.status = "completed"
        entry.gating_ready = True
        entry.closure_checklist = [
            ClosureChecklistItem(item_id="candidate_approval", label="候选实验已审批", completed=True),
            ClosureChecklistItem(
                item_id="experiment_execution",
                label="实验执行已完成或已失败归因",
                completed=True,
            ),
            ClosureChecklistItem(item_id="result_feedback", label="结果解释与反馈已写回", completed=True),
        ]
        entry.updated_at = datetime.now()
        history.last_updated = entry.updated_at
        self.repository.save_round_history(history)

    def test_request_review_and_approve_top_candidate(self) -> None:
        control = HumanControlService(
            self.repository,
            project_root=self.project_root,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        review_payload = control.request_experiment_selection_review()
        self.assertEqual(review_payload["status"], "awaiting_human_review")
        self.assertEqual(review_payload["current_phase"], "experiment_planning")
        self.assertEqual(review_payload["current_step"], "step_4")

        protocol = control.approve_candidate(
            human_notes="同意先执行最高 utility 候选实验",
            auto_continue=False,
            model_parameters={"alpha": 0.2, "l1_ratio": 0.6},
        )
        self.assertEqual(protocol.task_id, "ST_300")
        self.assertEqual(protocol.source_candidate_id, protocol.experiment_id)
        self.assertTrue(protocol.hypothesis_predictions)
        self.assertTrue(protocol.target_uncertainties)
        self.assertTrue(protocol.disagreement_context)
        self.assertEqual(protocol.steps[0].action, "resolve_disagreement")
        top_uncertainty_id = protocol.target_uncertainties[0]
        self.assertTrue(any(note.startswith("目标不确定性：") for note in protocol.notes))
        baseline_step = next(step for step in protocol.steps if step.action == "run_model" and step.parameters.get("label") == "baseline")
        self.assertEqual(baseline_step.parameters["label"], "baseline")
        self.assertTrue((self.project_root / "config" / "latest_protocol.json").exists())

        process_state = self.repository.load_process_state()
        self.assertEqual(process_state.current_stage, "protocol_ready")
        self.assertEqual(process_state.current_step, "step_5")
        self.assertEqual(
            process_state.phases["experiment_planning"].steps["step_4"].status,
            "completed",
        )

        decision_log = self.repository.load_decision_log()
        self.assertEqual(len(decision_log.decisions), 3)
        self.assertEqual(decision_log.decisions[0].decision_type, "experiment_selection_requested")
        self.assertEqual(decision_log.decisions[1].decision_type, "experiment_approved")
        self.assertEqual(decision_log.decisions[2].decision_type, "protocol_generated")

    def test_reject_candidate_updates_state_and_candidate_set(self) -> None:
        control = HumanControlService(
            self.repository,
            project_root=self.project_root,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        control.request_experiment_selection_review()
        rejected = control.reject_candidate(reason="该候选与当前判断不一致")

        self.assertEqual(rejected["status"], "rejected")
        process_state = self.repository.load_process_state()
        self.assertEqual(process_state.current_stage, "candidate_rejected")
        self.assertEqual(process_state.current_step, "step_4")

        candidate_set = self.repository.load_candidate_experiments()
        self.assertTrue(candidate_set.rejected_candidates)
        self.assertGreaterEqual(rejected["remaining_candidates"], 0)

    def test_round_review_and_decision_are_logged(self) -> None:
        control = HumanControlService(
            self.repository,
            project_root=self.project_root,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        payload = control.request_round_review(
            round_id=1,
            experiment_id="E_R02_01",
            evaluation_summary={"pearson_r_delta": 0.05},
            hypothesis_assessments=[
                {
                    "hypothesis_id": "H1",
                    "support_before": 0.5,
                    "support_after": 0.58,
                    "status": "supported",
                    "direction_matched": True,
                    "magnitude_matched": "partial",
                    "reason": "test assessment context",
                }
            ],
            reasoning_traces=[
                {
                    "trace_id": "RT001",
                    "stage": "round_review_context",
                    "summary": "上一轮程序解释认为 H1 获得初步支持",
                    "related_hypotheses": ["H1"],
                    "related_uncertainties": ["U01"],
                    "support_delta": 0.08,
                }
            ],
        )
        self.assertEqual(payload["status"], "awaiting_round_decision")

        self._mark_round_closed(round_id=1)
        recorded = control.record_round_decision(
            round_id=1,
            decision="adjust",
            human_feedback="下一轮先测试 By 单变量路径",
        )
        self.assertEqual(recorded["status"], "recorded")

        decision_log = self.repository.load_decision_log()
        self.assertEqual(decision_log.decisions[-2].decision_type, "round_review_requested")
        self.assertEqual(decision_log.decisions[-1].decision_type, "round_adjusted")
        self.assertEqual(decision_log.human_feedback[-1].content, "下一轮先测试 By 单变量路径")
        self.assertEqual(decision_log.human_feedback[-1].linked_reasoning_trace_ids, ["RT001"])
        self.assertEqual(recorded["planning_status"], "awaiting_data_dictionary")

    def test_round_decision_adjust_rebuilds_candidate_plan(self) -> None:
        control = HumanControlService(
            self.repository,
            project_root=self.project_root,
            planner_mode="llm",
            planner_output_builder=PlannerOutputBuilder(
                mode="llm",
                central_controller=CentralControllerLLM(gateway=StubLLMGateway()),
            ),
            rag_service=StubRAGService(),
            hypothesis_proposer=StubHypothesisProposer(),
            hypothesis_generator=StubHypothesisGenerator(self.repository),
            scientific_questioner=StubScientificQuestioner(),
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        control.request_round_review(
            round_id=1,
            experiment_id="E_R02_01",
            evaluation_summary={"pearson_r_delta": 0.05},
            hypothesis_assessments=[
                {
                    "hypothesis_id": "H1",
                    "support_before": 0.5,
                    "support_after": 0.58,
                    "status": "supported",
                    "direction_matched": True,
                    "magnitude_matched": "partial",
                    "reason": "test assessment context",
                }
            ],
            disagreement_updates=[
                {
                    "uncertainty_id": "U01",
                    "compared_hypotheses": ["H1", "H2"],
                    "support_span_before": 0.20,
                    "support_span_after": 0.11,
                    "narrowed": True,
                    "resolution_status": "partially_resolved",
                    "leading_hypothesis_id": "H1",
                    "unresolved_hypotheses": ["H1", "H2"],
                    "summary": "U01 分歧缩小但仍需下一轮验证",
                }
            ],
            reasoning_traces=[
                {
                    "trace_id": "RT001",
                    "stage": "round_review_context",
                    "summary": "上一轮程序解释认为 H1 获得初步支持",
                    "related_hypotheses": ["H1"],
                    "related_uncertainties": ["U01"],
                    "support_delta": 0.08,
                }
            ],
        )

        self._mark_round_closed(round_id=1)
        recorded = control.record_round_decision(
            round_id=1,
            decision="adjust",
            human_feedback="下一轮先测试 By 单变量路径",
            data_dictionary=self.dictionary,
        )
        self.assertEqual(recorded["planning_status"], "awaiting_hypothesis_confirmation")
        task = self.repository.load_task()
        task.payload.constraints.min_active_hypotheses = 2
        self.repository.save_task(task)
        self.repository.save_data_dictionary(self.dictionary)
        confirmed = control.confirm_hypothesis_tree()
        self.assertEqual(confirmed["status"], "awaiting_scientific_questioning")
        questioned = control.run_hypothesis_scientific_questioning()
        self.assertEqual(questioned["planning_status"], "scientific_questioning_completed")
        recorded = control.start_uncertainty_identification()
        self.assertEqual(recorded["planning_status"], "candidate_plan_rebuilt")
        self.assertEqual(recorded["next_review"]["status"], "awaiting_human_review")
        self.assertEqual(recorded["planner_input"].next_round_id, 2)
        self.assertEqual(recorded["planner_input"].human_feedback, "下一轮先测试 By 单变量路径")
        self.assertTrue(recorded["planner_output"].candidate_supplements)
        self.assertTrue(recorded["planner_output"].protocol_refinements)
        self.assertGreaterEqual(recorded["applied_output"]["hypothesis_updates"], 1)
        self.assertGreaterEqual(recorded["applied_output"]["uncertainty_updates"], 1)

        candidate_set = self.repository.load_candidate_experiments()
        focused = [
            candidate
            for candidate in candidate_set.candidates
            if "human_feedback_focus_single_feature:By" in candidate.design.notes
        ]
        self.assertTrue(focused)
        self.assertIn("By", focused[0].design.treatment)
        self.assertIn("human_feedback_focus_single_feature:By", focused[0].design.notes)

        planner_input = self.repository.load_planner_input()
        self.assertEqual(planner_input.source_round_id, 1)
        self.assertTrue(planner_input.unresolved_uncertainties)
        self.assertTrue(planner_input.recent_disagreement_updates)
        self.assertEqual(planner_input.recent_disagreement_updates[0].resolution_status, "partially_resolved")
        self.assertIn("human_feedback", planner_input.to_experiment_planner_payload())
        self.assertTrue(planner_input.to_central_controller_payload()["planner_guidance"])
        self.assertTrue(planner_input.recent_hypothesis_assessments)
        self.assertEqual(planner_input.recent_human_feedback[-1].linked_reasoning_trace_ids, ["RT001"])
        self.assertTrue(planner_input.recent_reasoning_traces)
        planner_output = self.repository.load_planner_output()
        self.assertTrue(planner_output.to_experiment_planner_payload()["candidate_supplements"])
        self.assertTrue(planner_output.to_scientific_interpreter_payload()["interpretation_enhancements"])

        process_state = self.repository.load_process_state()
        self.assertEqual(process_state.current_phase, "experiment_planning")
        self.assertEqual(process_state.current_stage, "awaiting_human_approval")

        tree = self.repository.load_hypothesis_tree()
        self.assertEqual(tree.latest_update.event, "planner_output_applied")
        self.assertTrue(tree.node_index()["H1"].critiques)
        self.assertEqual(tree.node_index()["H1"].critiques[-1].from_role, "scientific_interpreter")
        self.assertGreater(tree.node_index()["H1"].support_score, 0.6)
        self.assertEqual(tree.node_index()["H1"].support_history[-1].event, "planner_interpretation:IE001")

        uncertainties = self.repository.load_uncertainties()
        self.assertTrue(uncertainties.records[0].history)
        self.assertEqual(uncertainties.records[0].history[-1].event, "planner_interpretation")
        self.assertIsNotNone(uncertainties.records[0].notes)
        self.assertEqual(uncertainties.record_index()["U01"].priority, "medium")
        self.assertEqual(uncertainties.record_index()["U02"].priority, "low")
        self.assertEqual(uncertainties.priority_queue.queue[0].uncertainty_id, "U_LLM_R02_03")

        experiment_memory = self.repository.load_experiment_memory()
        trace_entry = experiment_memory.entry_index()["E_R02_01"]
        self.assertTrue(trace_entry.reasoning_traces)
        self.assertEqual(trace_entry.reasoning_traces[-1].stage, "interpretation_enhancement")
        self.assertIsNotNone(trace_entry.reasoning_traces[-1].priority_delta)

        decision_log = self.repository.load_decision_log()
        self.assertEqual(decision_log.decisions[-6].decision_type, "planner_input_prepared")
        self.assertEqual(decision_log.decisions[-5].decision_type, "hypothesis_tree_confirmed")
        self.assertEqual(decision_log.decisions[-4].decision_type, "scientific_questioning_completed")
        self.assertEqual(decision_log.decisions[-3].decision_type, "planner_output_generated")
        self.assertEqual(decision_log.decisions[-2].decision_type, "planner_output_applied")
        self.assertEqual(decision_log.decisions[-1].decision_type, "experiment_selection_requested")

    def test_round_decision_continue_rebuilds_candidate_plan_without_feedback(self) -> None:
        control = HumanControlService(
            self.repository,
            project_root=self.project_root,
            planner_mode="llm",
            planner_output_builder=PlannerOutputBuilder(
                mode="llm",
                central_controller=CentralControllerLLM(gateway=StubLLMGateway()),
            ),
            rag_service=StubRAGService(),
            hypothesis_proposer=StubHypothesisProposer(),
            hypothesis_generator=StubHypothesisGenerator(self.repository),
            scientific_questioner=StubScientificQuestioner(),
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        control.request_round_review(
            round_id=1,
            experiment_id="E_R02_01",
            evaluation_summary={"pearson_r_delta": 0.02},
        )

        self._mark_round_closed(round_id=1)
        recorded = control.record_round_decision(
            round_id=1,
            decision="continue",
            data_dictionary=self.dictionary,
        )
        self.assertEqual(recorded["planning_status"], "awaiting_hypothesis_confirmation")
        task = self.repository.load_task()
        task.payload.constraints.min_active_hypotheses = 2
        self.repository.save_task(task)
        self.repository.save_data_dictionary(self.dictionary)
        control.confirm_hypothesis_tree()
        control.run_hypothesis_scientific_questioning()
        recorded = control.start_uncertainty_identification()
        self.assertEqual(recorded["planning_status"], "candidate_plan_rebuilt")
        self.assertEqual(recorded["next_review"]["status"], "awaiting_human_review")
        self.assertIsNone(recorded["planner_input"].human_feedback)

    def test_approve_candidate_consumes_planner_output_refinements(self) -> None:
        control = HumanControlService(
            self.repository,
            project_root=self.project_root,
            planner_mode="llm",
            planner_output_builder=PlannerOutputBuilder(
                mode="llm",
                central_controller=CentralControllerLLM(gateway=StubLLMGateway()),
            ),
            rag_service=StubRAGService(),
            hypothesis_proposer=StubHypothesisProposer(),
            hypothesis_generator=StubHypothesisGenerator(self.repository),
            scientific_questioner=StubScientificQuestioner(),
            experiment_planner=StubExperimentPlanner(),
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        control.request_round_review(
            round_id=1,
            experiment_id="E_R02_01",
            evaluation_summary={"pearson_r_delta": -0.01, "stable": False},
        )
        self._mark_round_closed(round_id=1)
        control.record_round_decision(
            round_id=1,
            decision="adjust",
            human_feedback="下一轮先测试 By 单变量路径",
            data_dictionary=self.dictionary,
        )
        task = self.repository.load_task()
        task.payload.constraints.min_active_hypotheses = 2
        self.repository.save_task(task)
        self.repository.save_data_dictionary(self.dictionary)
        control.confirm_hypothesis_tree()
        control.run_hypothesis_scientific_questioning()
        control.start_uncertainty_identification()

        protocol = control.approve_candidate(auto_continue=False)
        self.assertIn("planner_refinement:feature_focus", protocol.notes)
        self.assertIn("planner_feedback_focus:By", protocol.notes)
        self.assertEqual(protocol.model.parameters["alpha"], 0.2)
        self.assertEqual(protocol.steps[0].action, "stability_validation")

    def test_request_pause_is_logged(self) -> None:
        control = HumanControlService(
            self.repository,
            project_root=self.project_root,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        control.request_stop(phase="experiment_execution", reason="先检查数据质量", pause=True)

        process_state = self.repository.load_process_state()
        self.assertTrue(process_state.user_requests.stop_requested)
        self.assertEqual(process_state.user_requests.stop_at_phase, "experiment_execution")

        decision_log = self.repository.load_decision_log()
        self.assertEqual(decision_log.stop_history[0].type, "pause")
        self.assertEqual(decision_log.decisions[0].decision_type, "pause_requested")

    def test_approve_execute_and_prepare_round_review(self) -> None:
        control = HumanControlService(
            self.repository,
            project_root=self.project_root,
            run_shap=False,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        control.request_experiment_selection_review()
        payload = control.approve_and_execute_candidate(
            human_notes="批准后直接执行",
            auto_continue=False,
        )

        self.assertEqual(payload["result"].status, "completed")
        self.assertEqual(payload["evaluation"].metrics.experiment_id, payload["protocol"].experiment_id)
        self.assertEqual(payload["review"]["status"], "awaiting_round_decision")
        self.assertTrue(payload["review"]["hypothesis_assessments"])
        self.assertTrue(payload["review"]["disagreement_updates"])
        self.assertTrue(payload["evaluation"].scientific.disagreement_updates)
        self.assertTrue(
            (self.project_root / "results" / f"round_{payload['protocol'].round_id:02d}" / "evaluation_unified.json").exists()
        )

        process_state = self.repository.load_process_state()
        self.assertEqual(process_state.current_stage, "awaiting_round_decision")
        self.assertEqual(process_state.current_phase, "decision_making")

        experiment_memory = self.repository.load_experiment_memory()
        self.assertEqual(experiment_memory.entries[0].experiment_id, payload["protocol"].experiment_id)

        uncertainties = self.repository.load_uncertainties()
        target_uncertainty_id = payload["protocol"].target_uncertainties[0]
        self.assertTrue(uncertainties.record_index()[target_uncertainty_id].history)
        self.assertIn(
            "disagreement_evaluation",
            [entry.event for entry in uncertainties.record_index()[target_uncertainty_id].history],
        )
        self.assertEqual(
            uncertainties.record_index()[target_uncertainty_id].resolving_experiment,
            payload["protocol"].experiment_id,
        )
        self.assertIsNotNone(uncertainties.record_index()[target_uncertainty_id].notes)

        decision_log = self.repository.load_decision_log()
        self.assertEqual(decision_log.decisions[-1].decision_type, "round_review_requested")

    def test_resume_evaluation_from_persisted_run_without_rerunning(self) -> None:
        control = HumanControlService(
            self.repository,
            project_root=self.project_root,
            run_shap=False,
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        control.request_experiment_selection_review()
        payload = control.approve_and_execute_candidate(
            human_notes="批准后直接执行",
            auto_continue=False,
        )
        protocol = payload["protocol"]
        round_dir = self.project_root / "results" / f"round_{protocol.round_id:02d}"
        evaluation_path = round_dir / "evaluation_unified.json"
        result_path = round_dir / "result_unified.json"
        self.assertTrue(evaluation_path.exists())
        persisted_result = result_path.read_bytes()

        # Simulate the server restarting right after the experiment finished but
        # before result analysis was persisted.
        evaluation_path.unlink()
        process_state = self.repository.load_process_state()
        process_state.current_phase = "result_analysis"
        process_state.current_stage = "evaluating_results"
        process_state.current_step = "evaluation"
        process_state.progress_percentage = 72
        process_state.phases["result_analysis"] = ProcessPhase(
            status="in_progress",
            started_at=datetime.now(),
            steps={
                "evaluation": ProcessStep(name="计算指标与稳健性分析", status="in_progress"),
                "scientific_interpretation": ProcessStep(name="生成科学解释", status="pending"),
            },
        )
        self.repository.save_process_state(process_state)

        recovered = control.resume_evaluation_from_persisted_run(
            round_id=protocol.round_id,
            experiment_id=protocol.experiment_id,
        )

        self.assertEqual(recovered["protocol"].experiment_id, protocol.experiment_id)
        self.assertEqual(recovered["result"].status, "completed")
        self.assertEqual(recovered["review"]["status"], "awaiting_round_decision")
        self.assertTrue(evaluation_path.exists())
        self.assertEqual(result_path.read_bytes(), persisted_result)

        final_state = self.repository.load_process_state()
        self.assertEqual(final_state.current_stage, "awaiting_round_decision")
        self.assertEqual(final_state.current_phase, "decision_making")
        self.assertEqual(final_state.phases["result_analysis"].status, "completed")

        persisted_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted_evaluation["metrics"]["experiment_id"], protocol.experiment_id)


if __name__ == "__main__":
    unittest.main()
