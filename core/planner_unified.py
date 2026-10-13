from __future__ import annotations

from typing import Literal

from core.state_repository import UnifiedStateRepository
from core.unified_schema import (
    CandidateExperimentSet,
    DataDictionary,
    DataDictionarySummary,
    ExperimentStep,
    HypothesisSnapshot,
    InterpretationEnhancement,
    PlannerDisagreementUpdateItem,
    PlannerHumanFeedbackItem,
    PlannerEvaluationSummary,
    PlannerCandidateSupplement,
    PlannerReasoningTraceItem,
    PlannerUncertaintyItem,
    ReasoningPlannerInput,
    ReasoningPlannerOutput,
    ProtocolRefinementSuggestion,
)
from core.variable_semantic_service import VariableSemanticService


class ReasoningPlannerInputBuilder:
    """Build a unified planner input for the next-round reasoning/planning layer."""

    def __init__(self, repository: UnifiedStateRepository) -> None:
        self.repository = repository

    def build(
        self,
        *,
        data_dictionary: DataDictionary,
        source_round_id: int,
        human_feedback: str | None = None,
    ) -> ReasoningPlannerInput:
        task = self.repository.load_task()
        tree = self.repository.load_hypothesis_tree()
        uncertainties = self.repository.load_uncertainties()
        decision_log = self.repository.load_decision_log()
        experiment_memory = self.repository.load_experiment_memory()

        review_entry = _find_latest_round_review(decision_log, source_round_id)
        evaluation_summary = _build_evaluation_summary(source_round_id, review_entry.details if review_entry else {})
        unresolved_uncertainties = _build_unresolved_uncertainties(uncertainties)
        active_hypotheses = _build_active_hypotheses(tree)
        recent_assessments = _build_recent_hypothesis_assessments(review_entry.details if review_entry else {})
        recent_disagreement_updates = _build_recent_disagreement_updates(review_entry.details if review_entry else {})
        recent_reasoning_traces = _build_recent_reasoning_traces(
            experiment_memory,
            evaluation_summary.experiment_id,
            review_entry.details if review_entry else {},
        )
        recent_human_feedback = _build_recent_human_feedback(decision_log, source_round_id)
        base_dictionary_summary = DataDictionarySummary(
            dictionary_id=data_dictionary.dictionary_id,
            dataset_name=data_dictionary.dataset_name,
            time_column=data_dictionary.time_column,
            target_candidates=data_dictionary.target_candidates,
            feature_candidates=data_dictionary.feature_candidates,
        )
        semantic_service = VariableSemanticService.from_data_dictionary(data_dictionary)
        planner_input = ReasoningPlannerInput(
            task_id=task.task_id,
            source_round_id=source_round_id,
            next_round_id=source_round_id + 1,
            source_experiment_id=evaluation_summary.experiment_id,
            scientific_question=task.payload.research_question.text,
            target=task.payload.research_question.target,
            human_feedback=human_feedback,
            evaluation_summary=evaluation_summary,
            unresolved_uncertainties=unresolved_uncertainties,
            active_hypotheses=active_hypotheses,
            recent_hypothesis_assessments=recent_assessments,
            recent_disagreement_updates=recent_disagreement_updates,
            recent_reasoning_traces=recent_reasoning_traces,
            recent_human_feedback=recent_human_feedback,
            data_dictionary_summary=semantic_service.build_display_summary(base_dictionary_summary),
            planning_constraints=_build_constraints(task),
            planner_guidance=_build_planner_guidance(
                human_feedback=human_feedback,
                evaluation_summary=evaluation_summary,
                unresolved_uncertainties=unresolved_uncertainties,
                recent_disagreement_updates=recent_disagreement_updates,
                recent_reasoning_traces=recent_reasoning_traces,
                recent_human_feedback=recent_human_feedback,
                active_hypotheses=active_hypotheses,
            ),
        )
        return planner_input


class ProgrammaticPlannerOutputBuilder:
    """Generate a standardized planner output from planner_input and current candidate plan."""

    def build(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        candidate_plan: CandidateExperimentSet,
    ) -> ReasoningPlannerOutput:
        candidate_supplements = _build_candidate_supplements(candidate_plan)
        interpretation_enhancements = _build_interpretation_enhancements(planner_input)
        protocol_refinements = _build_protocol_refinements(planner_input, candidate_plan)
        return ReasoningPlannerOutput(
            task_id=planner_input.task_id,
            source_round_id=planner_input.source_round_id,
            next_round_id=planner_input.next_round_id,
            source_experiment_id=planner_input.source_experiment_id,
            planner_mode="programmatic",
            candidate_supplements=candidate_supplements,
            interpretation_enhancements=interpretation_enhancements,
            protocol_refinements=protocol_refinements,
            summary=_build_output_summary(candidate_supplements, interpretation_enhancements, protocol_refinements),
        )


class PlannerOutputBuilder:
    """Route planner_output generation through programmatic or llm mode."""

    def __init__(
        self,
        *,
        mode: Literal["programmatic", "llm"] = "programmatic",
        central_controller=None,
        programmatic_builder: ProgrammaticPlannerOutputBuilder | None = None,
    ) -> None:
        self.mode = mode
        self.central_controller = central_controller
        self.programmatic_builder = programmatic_builder or ProgrammaticPlannerOutputBuilder()

    def build(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        candidate_plan: CandidateExperimentSet,
    ) -> ReasoningPlannerOutput:
        if self.mode == "llm":
            controller = self.central_controller
            if controller is None:
                from core.central_controller_llm import CentralControllerLLM

                controller = CentralControllerLLM()
            return controller.build(
                planner_input=planner_input,
                candidate_plan=candidate_plan,
            )
        return self.programmatic_builder.build(
            planner_input=planner_input,
            candidate_plan=candidate_plan,
        )


def _find_latest_round_review(decision_log, round_id: int):
    for entry in reversed(decision_log.decisions):
        if entry.round_id == round_id and entry.decision_type == "round_review_requested":
            return entry
    return None


def _build_evaluation_summary(round_id: int, details: dict) -> PlannerEvaluationSummary:
    payload = details.get("evaluation_summary", {})
    return PlannerEvaluationSummary(
        experiment_id=details.get("experiment_id") or payload.get("experiment_id"),
        round_id=payload.get("round_id", round_id),
        baseline_rmse=payload.get("baseline_rmse"),
        treatment_rmse=payload.get("treatment_rmse"),
        baseline_pearson_r=payload.get("baseline_pearson_r"),
        treatment_pearson_r=payload.get("treatment_pearson_r"),
        delta_pearson_r=payload.get("delta_pearson_r", payload.get("pearson_r_delta")),
        delta_rmse=payload.get("delta_rmse", payload.get("rmse_delta")),
        pg_actual_signed=payload.get("pg_actual_signed"),
        pg_actual_clipped=payload.get("pg_actual_clipped"),
        key_findings=payload.get("key_findings", []),
        robustness_recommendation=payload.get("robustness_recommendation"),
        stable=payload.get("stable"),
    )


def _build_unresolved_uncertainties(uncertainties) -> list[PlannerUncertaintyItem]:
    score_index = {}
    if uncertainties.priority_queue is not None:
        score_index = {
            item.uncertainty_id: item.priority_score for item in uncertainties.priority_queue.queue
        }

    items: list[PlannerUncertaintyItem] = []
    for record in uncertainties.records:
        if record.status in {"resolved", "deprecated"}:
            continue
        items.append(
            PlannerUncertaintyItem(
                uncertainty_id=record.uncertainty_id,
                question=record.question,
                priority=record.priority,
                priority_score=score_index.get(record.uncertainty_id),
                status=record.status,
                resolution_status=record.resolution_status,
                related_hypotheses=record.related_hypotheses,
            )
        )
    items.sort(key=lambda item: item.priority_score or 0.0, reverse=True)
    return items[:12]


def _build_active_hypotheses(tree) -> list[HypothesisSnapshot]:
    snapshots = [
        HypothesisSnapshot(
            hypothesis_id=node.hypothesis_id,
            statement=(node.display_statement or node.statement),
            status=node.status,
            support_score=node.support_score,
        )
        for node in tree.nodes
        if node.status in {"active", "observing", "converged", "pending", "draft", "weakened"}
    ]
    snapshots.sort(key=lambda item: item.support_score, reverse=True)
    return snapshots[:12]


def _build_constraints(task) -> dict[str, object]:
    constraints = task.payload.constraints
    return {
        "no_future_information": constraints.no_future_information,
        "validation_feedback_allowed": constraints.validation_feedback_allowed,
        "final_test_blind": constraints.final_test_blind,
        "max_rounds": constraints.max_rounds,
        "max_experiments_per_round": constraints.max_experiments_per_round,
        "resource_budget": constraints.resource_budget.model_dump(exclude_none=True) if constraints.resource_budget else {},
    }


def _build_planner_guidance(
    *,
    human_feedback: str | None,
    evaluation_summary: PlannerEvaluationSummary,
    unresolved_uncertainties: list[PlannerUncertaintyItem],
    recent_disagreement_updates: list[PlannerDisagreementUpdateItem] | None = None,
    recent_reasoning_traces: list[PlannerReasoningTraceItem] | None = None,
    recent_human_feedback: list[PlannerHumanFeedbackItem] | None = None,
    active_hypotheses: list[HypothesisSnapshot] | None = None,
) -> list[str]:
    guidance: list[str] = []
    if human_feedback:
        guidance.append(f"human_feedback: {human_feedback}")
    if evaluation_summary.stable is False:
        guidance.append("上一轮结果稳定性不足，下一轮优先设计稳健性验证或更简单的方案。")
    if evaluation_summary.delta_pearson_r is not None and evaluation_summary.delta_pearson_r <= 0:
        guidance.append("上一轮预测增益不明显，下一轮优先探索区分性更强的候选实验。")
    if unresolved_uncertainties:
        guidance.append(f"优先关注未解决不确定性: {unresolved_uncertainties[0].question}")
    if recent_disagreement_updates:
        update = recent_disagreement_updates[0]
        leading_label = "未标记"
        if update.leading_hypothesis_id and active_hypotheses:
            leading_label = next(
                (
                    item.statement if len(item.statement) <= 48 else f"{item.statement[:48]}…"
                    for item in active_hypotheses
                    if item.hypothesis_id == update.leading_hypothesis_id
                ),
                update.leading_hypothesis_id,
            )
        guidance.append(
            f"最近分歧状态: {update.uncertainty_id} -> {update.resolution_status}, leading={leading_label}"
        )
    if recent_reasoning_traces:
        guidance.append(f"上一轮程序解释重点: {recent_reasoning_traces[0].summary}")
    if recent_human_feedback:
        guidance.append(f"最近人工反馈: {recent_human_feedback[0].content}")
    return guidance


def _build_candidate_supplements(
    candidate_plan: CandidateExperimentSet,
) -> list[PlannerCandidateSupplement]:
    supplements: list[PlannerCandidateSupplement] = []
    for index, candidate in enumerate(candidate_plan.candidates[:2], start=1):
        supplements.append(
            PlannerCandidateSupplement(
                proposal_id=f"PC{index:03d}",
                candidate=candidate.model_copy(deep=True),
                rationale=f"候选 {candidate.experiment_id} 在当前程序化排序中靠前，可作为实验规划者继续细化的基础。",
            )
        )
    return supplements


def _build_interpretation_enhancements(
    planner_input: ReasoningPlannerInput,
) -> list[InterpretationEnhancement]:
    enhancements: list[InterpretationEnhancement] = []
    if planner_input.active_hypotheses:
        top = planner_input.active_hypotheses[0]
        assessment = _select_assessment_for_hypothesis(planner_input, top.hypothesis_id)
        if assessment is not None:
            support_delta = assessment.support_after - assessment.support_before
            impact_direction = "supports" if support_delta > 0 else "weakens"
            impact_strength = min(max(abs(support_delta) / 0.12, 0.25), 0.85)
            direction = "与上一轮正式评价一致，出现支持信号" if support_delta > 0 else "与上一轮正式评价一致，支持信号减弱"
        else:
            delta = planner_input.evaluation_summary.delta_pearson_r
            direction = "出现初步支持信号" if (delta or 0.0) > 0 else "尚未形成稳定支持信号"
            impact_direction = "supports" if (delta or 0.0) > 0 else "weakens"
            impact_strength = min(max(abs(delta or 0.03) / 0.10, 0.25), 0.85)
        uncertainty_action = (
            "decrease" if impact_direction == "supports" and planner_input.evaluation_summary.stable is not False else "maintain"
        )
        enhancements.append(
            InterpretationEnhancement(
                enhancement_id="IE001",
                target_hypothesis_id=top.hypothesis_id,
                interpretation=(
                    f"主假设“{top.statement if len(top.statement) <= 56 else top.statement[:56] + '…'}”"
                    f"当前支持度为 {top.support_score:.2f}，结合上一轮表现，{direction}。"
                ),
                evidence_basis=planner_input.evaluation_summary.key_findings[:2],
                related_uncertainties=[item.uncertainty_id for item in planner_input.unresolved_uncertainties[:2]],
                suggested_state_note="可由科学解释者补充方向/幅度/精密度的更细解释。",
                impact_direction=impact_direction,
                impact_strength=impact_strength,
                confidence=0.75 if planner_input.evaluation_summary.stable is not False else 0.55,
                uncertainty_priority_action=uncertainty_action,
            )
        )
    if planner_input.unresolved_uncertainties:
        first = planner_input.unresolved_uncertainties[0]
        enhancements.append(
            InterpretationEnhancement(
                enhancement_id="IE002",
                interpretation=f"未解决不确定性 {first.uncertainty_id} 仍为下一轮核心分歧点，应要求解释层说明其与候选实验的区分关系。",
                evidence_basis=[first.question],
                related_uncertainties=[first.uncertainty_id],
                impact_direction="clarifies",
                impact_strength=0.45,
                confidence=0.7,
                uncertainty_priority_action="increase",
            )
        )
    return enhancements


def _build_protocol_refinements(
    planner_input: ReasoningPlannerInput,
    candidate_plan: CandidateExperimentSet,
) -> list[ProtocolRefinementSuggestion]:
    refinements: list[ProtocolRefinementSuggestion] = []
    top_candidate = candidate_plan.top_candidate() or (candidate_plan.candidates[0] if candidate_plan.candidates else None)
    if top_candidate is None:
        return refinements

    feedback_text = planner_input.human_feedback or ""
    feedback_lower = feedback_text.lower()
    focus_features = [
        feature
        for feature in planner_input.data_dictionary_summary.feature_candidates
        if feature.lower() in feedback_lower
    ]
    if focus_features:
        refinements.append(
            ProtocolRefinementSuggestion(
                suggestion_id="PR001",
                target_candidate_id=top_candidate.experiment_id,
                refinement_type="feature_focus",
                rationale="human_feedback 明确指定了优先测试变量，应在协议中保留该聚焦信息。",
                suggested_model_parameters={"alpha": 0.2},
                suggested_feature_focus=focus_features,
                protocol_notes=[f"planner_feedback_focus:{feature}" for feature in focus_features],
            )
        )

    if planner_input.evaluation_summary.stable is False:
        refinements.append(
            ProtocolRefinementSuggestion(
                suggestion_id="PR002",
                target_candidate_id=top_candidate.experiment_id,
                refinement_type="stability_validation",
                rationale="上一轮稳定性不足，应在协议中显式增加稳健性校验步骤。",
                suggested_steps=[
                    ExperimentStep(
                        step=1,
                        action="stability_validation",
                        parameters={"source": "planner_output", "reason": "previous_round_unstable"},
                    )
                ],
                protocol_notes=["planner_requires_stability_validation"],
            )
        )

    if planner_input.evaluation_summary.delta_pearson_r is not None and planner_input.evaluation_summary.delta_pearson_r <= 0:
        refinements.append(
            ProtocolRefinementSuggestion(
                suggestion_id="PR003",
                target_candidate_id=top_candidate.experiment_id,
                refinement_type="conservative_model_tuning",
                rationale="上一轮预测增益不明显，建议采用更保守的正则配置。",
                suggested_model_parameters={"l1_ratio": 0.6},
                protocol_notes=["planner_suggests_conservative_regularization"],
            )
        )
    return refinements


def _build_output_summary(
    candidate_supplements: list[PlannerCandidateSupplement],
    interpretation_enhancements: list[InterpretationEnhancement],
    protocol_refinements: list[ProtocolRefinementSuggestion],
) -> str:
    return (
        f"生成 {len(candidate_supplements)} 个候选补充、"
        f"{len(interpretation_enhancements)} 条解释增强、"
        f"{len(protocol_refinements)} 条协议细化建议。"
    )


def _build_recent_hypothesis_assessments(details: dict) -> list:
    return details.get("hypothesis_assessments", [])


def _build_recent_disagreement_updates(details: dict) -> list[PlannerDisagreementUpdateItem]:
    return [
        PlannerDisagreementUpdateItem(**item)
        if not isinstance(item, PlannerDisagreementUpdateItem)
        else item
        for item in details.get("disagreement_updates", [])
    ]


def _build_recent_reasoning_traces(
    experiment_memory,
    experiment_id: str | None,
    review_details: dict,
) -> list[PlannerReasoningTraceItem]:
    traces: list[PlannerReasoningTraceItem] = []
    seen_ids: set[str] = set()
    if experiment_id:
        entry = experiment_memory.entry_index().get(experiment_id)
        if entry is not None:
            for trace in entry.reasoning_traces[-5:]:
                traces.append(
                    PlannerReasoningTraceItem(
                        trace_id=trace.trace_id,
                        stage=trace.stage,
                        summary=trace.summary,
                        related_hypotheses=trace.related_hypotheses,
                        related_uncertainties=trace.related_uncertainties,
                        referenced_variables=trace.referenced_variables,
                        support_delta=trace.support_delta,
                        priority_delta=trace.priority_delta,
                    )
                )
                seen_ids.add(trace.trace_id)
    for trace in review_details.get("reasoning_traces", []):
        trace_id = trace.get("trace_id")
        if not trace_id or trace_id in seen_ids:
            continue
        traces.append(
            PlannerReasoningTraceItem(
                trace_id=trace_id,
                stage=trace.get("stage", "round_review_context"),
                summary=trace.get("summary", "review reasoning trace"),
                related_hypotheses=trace.get("related_hypotheses", []),
                related_uncertainties=trace.get("related_uncertainties", []),
                referenced_variables=trace.get("referenced_variables") or [],
                support_delta=trace.get("support_delta"),
                priority_delta=trace.get("priority_delta"),
            )
        )
    return traces[-5:]


def _build_recent_human_feedback(decision_log, round_id: int) -> list[PlannerHumanFeedbackItem]:
    items = []
    for entry in reversed(decision_log.human_feedback):
        if entry.round_number != round_id:
            continue
        items.append(
            PlannerHumanFeedbackItem(
                feedback_id=entry.feedback_id,
                feedback_type=entry.feedback_type,
                content=entry.content,
                linked_reasoning_trace_ids=entry.linked_reasoning_trace_ids,
                related_experiment_id=entry.related_experiment_id,
                context_summary=entry.context_summary,
            )
        )
        if len(items) >= 3:
            break
    return list(reversed(items))


def _select_assessment_for_hypothesis(
    planner_input: ReasoningPlannerInput,
    hypothesis_id: str,
):
    for assessment in planner_input.recent_hypothesis_assessments:
        if assessment.hypothesis_id == hypothesis_id:
            return assessment
    return None
