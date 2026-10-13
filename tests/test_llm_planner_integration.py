import unittest
from datetime import datetime, timezone

from core.central_controller_llm import CentralControllerLLM, _rewrite_candidate_design_focus
from core.control_unified import HumanControlService
from core.decision_unified import DecisionLayerService
from core.hypothesis_generation import HypothesisGenerationResult, HypothesisGenerationService
from core.hypothesis_proposer_llm import (
    HypothesisProposerResponse,
    ProposedHypothesis,
    _next_display_hypothesis_id,
)
from core.planner_unified import PlannerOutputBuilder
from core.rag_service import RAGContextBundle, RAGEvidenceSnippet
from core.round_orchestrator import RoundOrchestrator
from core.scientific_interpreter_llm import ScientificInterpreterLLM
from core.scientific_questioner_llm import (
    HypothesisQuestioningUpdate,
    ProposedUncertainty,
    ScientificQuestionerResponse,
)
from core.unified_schema import (
    CandidateExperiment,
    ExperimentDesign,
    ExperimentStep,
    ProtocolRefinementSuggestion,
)
from core.variable_semantic_service import VariableSemanticService
from tests.test_decision_layer import (
    StubCandidateExperimentDesigner,
    StubCandidateExperimentWriter,
)
from tests.test_human_control_flow import HumanControlFlowTest


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
        role=None,
        use_cache=True,
        deadline_seconds=None,
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


class StubScientificInterpreterGateway:
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
        role=None,
        use_cache=True,
        deadline_seconds=None,
    ):
        return response_model.model_validate(
            {
                "target_hypothesis_id": "H1",
                "interpretation": "LLM 科学解释者认为 H1 获得了更可重复的支持，但仍需保持保守解释。",
                "related_uncertainty_ids": ["U01"],
                "impact_direction": "supports",
                "impact_strength": 0.57,
                "confidence": 0.79,
                "uncertainty_priority_action": "decrease",
                "suggested_state_note": "llm scientific interpreter note",
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
            focus_features=["Np"],
            guidance_notes=["proposer_focus:Np 条件路径值得补充为下一轮假设。"],
            proposal_summaries=["Np 条件路径可能决定 DeltaDec 增量是否成立。"],
            proposed_hypotheses=[
                ProposedHypothesis(
                    display_hypothesis_id="H1",
                    statement="宇宙线日影南北偏移对太阳风速度预测提供独立增量信息",
                    level=1,
                    activation_condition="always",
                ),
                ProposedHypothesis(
                    display_hypothesis_id="H2",
                    statement="宇宙线日影南北偏移的预测增益主要来自行星际磁场南北分量的伴随相关",
                    level=1,
                    activation_condition="always",
                ),
                ProposedHypothesis(
                    display_hypothesis_id="H3",
                    statement="宇宙线日影南北偏移在控制行星际磁场后仍保留独立预测信息",
                    level=2,
                    parent_id="H1",
                    activation_condition="父假设支持度>=0.40",
                ),
                ProposedHypothesis(
                    display_hypothesis_id="H4",
                    statement="宇宙线日影南北偏移的增益仅在特定超前预测窗口成立",
                    level=2,
                    parent_id="H1",
                    activation_condition="父假设支持度>=0.40",
                ),
                ProposedHypothesis(
                    display_hypothesis_id="H5",
                    statement="宇宙线日影南北偏移的预测结论对评价期选择保持稳健",
                    level=2,
                    parent_id="H2",
                    activation_condition="父假设支持度>=0.40",
                ),
            ],
        )

    def propose_supplemental(self, *, planner_input, rag_context, tree):
        return {
            "display_hypothesis_id": _next_display_hypothesis_id(tree.nodes),
            "statement": "DeltaDec 的预测增量仅在特定太阳风密度条件下成立，且不依赖已剪枝路径",
            "level": 1,
            "parent_id": None,
            "activation_condition": "始终激活：科学质询后补充的竞争根假设。",
            "falsification_conditions": [
                "控制太阳风密度条件后，DeltaDec 的预测增量消失"
            ],
            "alternative_explanations": [
                "密度条件差异只是伴随相关，不是独立机制"
            ],
            "predictions": [
                {
                    "experiment_condition": "按太阳风密度分层验证 DeltaDec 增量",
                    "expected_observable": "Pearson_r",
                    "expected_direction": "positive",
                }
            ],
            "evidence_items": [
                {
                    "description": "轮次间残留的不确定性仍指向密度条件路径未被区分",
                    "evidence_type": "evidence_gap",
                    "source": "test_stub",
                }
            ],
            "source": "test_stub",
            "model": "stub",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


class StubScientificQuestioner:
    def question(self, *, planner_input, rag_context, mined_candidates=None):
        return ScientificQuestionerResponse(
            challenge_points=["当前仍需确认 Np 条件路径是否只是伴随相关。"],
            guidance_notes=["questioner_gap:需要把 Np 条件路径显式提升为科学不确定性。"],
            proposed_uncertainties=[
                ProposedUncertainty(
                    question="DeltaDec 的增量是否只在特定 Np 条件下成立？",
                    description="需要区分独立信息与密度条件约束路径。",
                    priority="high",
                )
            ],
        )

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
            challenge_points=["当前仍需确认 Np 条件路径是否只是伴随相关。"],
            guidance_notes=["questioner_gap:需要把 Np 条件路径显式提升为科学不确定性。"],
            proposed_uncertainties=[
                ProposedUncertainty(
                    question="DeltaDec 的增量是否只在特定 Np 条件下成立？",
                    description="需要区分独立信息与密度条件约束路径。",
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
            suggested_model_parameters={"alpha": 0.19},
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

    def append_supplemental_node(self, **kwargs):
        return HypothesisGenerationService().append_supplemental_node(**kwargs)


class LLMPlannerIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture_helper = HumanControlFlowTest(methodName="test_round_decision_adjust_rebuilds_candidate_plan")
        fixture_helper.setUp()
        self.helper = fixture_helper
        self.repository = fixture_helper.repository
        self.project_root = fixture_helper.project_root
        self.dictionary = fixture_helper.dictionary
        candidate_set = self.repository.load_candidate_experiments()
        if not candidate_set.candidates:
            DecisionLayerService(
                self.repository,
                experiment_designer=StubCandidateExperimentDesigner(),
                experiment_writer=StubCandidateExperimentWriter(),
            ).build_candidate_plan(
                task=self.repository.load_task(),
                data_dictionary=self.dictionary,
            )

    def tearDown(self) -> None:
        self.helper.tearDown()

    def test_focus_rewrite_applies_llm_feature_to_candidate_design(self) -> None:
        candidate = CandidateExperiment(
            experiment_id="E_R99_01",
            type="distinguishing",
            purpose="test Np vs Vsw",
            tested_hypotheses=["H1"],
            design=ExperimentDesign(
                target="Vsw",
                control=["DeltaDec", "By"],
                treatment=["DeltaDec", "Np"],
                design_focus="Np",
            ),
        )
        changed = _rewrite_candidate_design_focus(
            candidate=candidate,
            feature_focus_raw=["By"],
            primary_x="DeltaDec",
            semantic=VariableSemanticService(),
        )
        self.assertTrue(changed)
        self.assertEqual(candidate.design.design_focus, "By")
        self.assertEqual(candidate.design.control, ["DeltaDec"])
        self.assertEqual(candidate.design.treatment, ["DeltaDec", "By"])
        self.assertIn("llm_controller_focus_rewrite:Np->By", candidate.design.notes)

    def test_llm_mode_rebuilds_candidate_plan_without_skipping_human_review(self) -> None:
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
            scientific_questioner=StubScientificQuestioner(),
            experiment_planner=StubExperimentPlanner(),
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )
        control.hypothesis_generator = StubHypothesisGenerator(self.repository)
        control.request_round_review(
            round_id=1,
            experiment_id="E_R02_01",
            evaluation_summary={"pearson_r_delta": 0.05, "stable": True},
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
        )

        self.helper._mark_round_closed(round_id=1)
        recorded = control.record_round_decision(
            round_id=1,
            decision="adjust",
            human_feedback="下一轮先测试 By 单变量路径",
            data_dictionary=self.dictionary,
        )
        self.repository.save_data_dictionary(self.dictionary)
        task = self.repository.load_task()
        task.payload.constraints.min_active_hypotheses = 2
        self.repository.save_task(task)

        self.assertEqual(recorded["planning_status"], "awaiting_hypothesis_confirmation")
        confirmed = control.confirm_hypothesis_tree()
        self.assertEqual(confirmed["status"], "awaiting_scientific_questioning")
        questioned = RoundOrchestrator(control).run_next_round_scientific_questioning()
        self.assertEqual(questioned["planning_status"], "scientific_questioning_completed")
        recorded = control.start_uncertainty_identification()
        self.assertEqual(recorded["planning_status"], "candidate_plan_rebuilt")
        self.assertEqual(recorded["next_review"]["status"], "awaiting_human_review")
        self.assertEqual(recorded["planner_output"].planner_mode, "llm")
        self.assertEqual(recorded["planner_output"].candidate_supplements[0].source, "central_controller_llm")
        self.assertEqual(
            recorded["planner_output"].protocol_refinements[0].refinement_type,
            "llm_central_controller",
        )

        candidate_set = self.repository.load_candidate_experiments()
        llm_selected = [
            candidate
            for candidate in candidate_set.candidates
            if "llm_controller_selected" in candidate.design.notes
        ]
        self.assertTrue(llm_selected)
        self.assertIn("llm_controller_feature_focus:By", llm_selected[0].design.notes)

        decision_log = self.repository.load_decision_log()
        planner_generated = [
            entry for entry in decision_log.decisions if entry.decision_type == "planner_output_generated"
        ]
        self.assertTrue(planner_generated)
        self.assertEqual(planner_generated[-1].details["planner_mode"], "llm")
        planner_input = self.repository.load_planner_input()
        self.assertTrue(any("rag_project:" in note for note in planner_input.planner_guidance))
        self.assertTrue(any("rag_literature:" in note for note in planner_input.planner_guidance))
        self.assertTrue(any("proposer_focus:Np" in note for note in planner_input.planner_guidance))
        self.assertTrue(any("questioner_gap:" in note for note in planner_input.planner_guidance))
        self.assertTrue(any("Np" in trace.summary for trace in planner_input.recent_reasoning_traces))
        uncertainties = self.repository.load_uncertainties()
        self.assertTrue(
            any("Np 条件下成立" in record.question for record in uncertainties.records)
        )

    def test_experiment_planner_refines_protocol_at_approval_time(self) -> None:
        control = HumanControlService(
            self.repository,
            project_root=self.project_root,
            planner_mode="llm",
            experiment_planner=StubExperimentPlanner(),
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )

        protocol = control.approve_candidate(
            human_notes="测试 llm experiment planner",
            auto_continue=False,
        )

        self.assertTrue(any(note == "planner_refinement:实验规划者llm" for note in protocol.notes))
        self.assertIn("stub_experiment_planner_note", protocol.notes)
        self.assertTrue(any(step.action == "planner_hypothesis_focus" for step in protocol.steps))

        decision_log = self.repository.load_decision_log()
        self.assertIn(
            "实验规划者llm",
            decision_log.decisions[-1].details["refinement_types"],
        )

    def test_llm_scientific_interpreter_is_applied_between_evaluation_and_review(self) -> None:
        control = HumanControlService(
            self.repository,
            project_root=self.project_root,
            planner_mode="llm",
            scientific_interpreter=ScientificInterpreterLLM(gateway=StubScientificInterpreterGateway()),
            experiment_designer=StubCandidateExperimentDesigner(),
            experiment_writer=StubCandidateExperimentWriter(),
        )

        executed = control.approve_and_execute_candidate(
            human_notes="执行并检查解释者写回",
            auto_continue=False,
        )

        self.assertEqual(executed["review"]["status"], "awaiting_round_decision")
        experiment_memory = self.repository.load_experiment_memory()
        entry = experiment_memory.entry_index()[executed["protocol"].experiment_id]
        self.assertTrue(entry.reasoning_traces)
        self.assertEqual(entry.reasoning_traces[-1].stage, "scientific_interpreter")
        self.assertIn("LLM 科学解释者", entry.reasoning_traces[-1].summary)

        tree = self.repository.load_hypothesis_tree()
        self.assertEqual(tree.latest_update.event, "scientific_interpreter_applied")
        self.assertTrue(tree.node_index()["H1"].critiques)
        self.assertEqual(tree.node_index()["H1"].critiques[-1].from_role, "scientific_interpreter")

        decision_log = self.repository.load_decision_log()
        self.assertIn(
            "scientific_interpreter_applied",
            [entry.decision_type for entry in decision_log.decisions],
        )


if __name__ == "__main__":
    unittest.main()
