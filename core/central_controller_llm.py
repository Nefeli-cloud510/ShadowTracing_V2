from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.llm_gateway import LLMGateway
from core.prompt_rules import ELASTIC_NET_EXECUTION_RULE
from core.planner_unified import ProgrammaticPlannerOutputBuilder
from core.unified_schema import (
    CandidateExperimentSet,
    InterpretationEnhancement,
    PlannerCandidateSupplement,
    ProtocolRefinementSuggestion,
    ReasoningPlannerInput,
    ReasoningPlannerOutput,
)
from core.variable_semantic_service import VariableSemanticService


class CentralControllerLLMResponse(BaseModel):
    summary: str = Field(min_length=1)
    candidate_focus_ids: list[str] = Field(default_factory=list)
    candidate_rationale: str = Field(min_length=1)
    target_hypothesis_id: str | None = None
    interpretation: str = Field(min_length=1)
    related_uncertainty_ids: list[str] = Field(default_factory=list)
    impact_direction: str = Field(default="clarifies")
    impact_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    uncertainty_priority_action: str = Field(default="maintain")
    feature_focus: list[str] = Field(default_factory=list)
    protocol_notes: list[str] = Field(default_factory=list)
    suggested_model_parameters: dict[str, Any] = Field(default_factory=dict)


class CentralControllerLLM:
    """LLM-backed central controller that refines planner output on top of a safe baseline."""

    SYSTEM_PROMPT = """你是“逐影 Shadow Tracing”的中央进程控制者。

你的任务是读取统一 planner_input 与候选实验计划，在不绕过 human PI 审批的前提下，
输出下一轮规划的结构化裁决。你不能发明不存在的 candidate_id、hypothesis_id、
uncertainty_id；你只能在给定候选计划范围内聚焦、解释与细化。

解释性文本和 protocol_notes 禁止出现 H_*、U_*、status=、narrowed= 等内部机器标识；
提及假设用 H1..H5 或自然语言名称，内部 ID 只允许出现在指定 id 字段。

输出必须是严格 JSON，且字段必须完整。""" + "\n\n" + ELASTIC_NET_EXECUTION_RULE

    def __init__(
        self,
        *,
        gateway: LLMGateway | None = None,
        baseline_builder: ProgrammaticPlannerOutputBuilder | None = None,
    ) -> None:
        self.gateway = gateway or LLMGateway()
        self.baseline_builder = baseline_builder or ProgrammaticPlannerOutputBuilder()

    def build(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        candidate_plan: CandidateExperimentSet,
    ) -> ReasoningPlannerOutput:
        baseline_output = self.baseline_builder.build(
            planner_input=planner_input,
            candidate_plan=candidate_plan,
        )
        response = self.gateway.generate_structured(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(
                planner_input=planner_input,
                candidate_plan=candidate_plan,
                baseline_output=baseline_output,
            ),
            response_model=CentralControllerLLMResponse,
            fallback_factory=lambda: self._build_fallback_response(
                planner_input=planner_input,
                candidate_plan=candidate_plan,
                baseline_output=baseline_output,
            ),
            role="中央进程控制者llm",
        )
        return self._compose_output(
            planner_input=planner_input,
            candidate_plan=candidate_plan,
            baseline_output=baseline_output,
            llm_response=response,
        )

    def _build_user_prompt(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        candidate_plan: CandidateExperimentSet,
        baseline_output: ReasoningPlannerOutput,
    ) -> str:
        top_candidates = []
        for candidate in candidate_plan.candidates[:3]:
            top_candidates.append(
                {
                    "experiment_id": candidate.experiment_id,
                    "type": candidate.type,
                    "purpose": candidate.purpose,
                    "tested_hypotheses": candidate.tested_hypotheses,
                    "related_uncertainties": candidate.related_uncertainties,
                    "treatment": candidate.design.treatment,
                    "notes": candidate.design.notes[-4:],
                    "utility_score": candidate.utility_score,
                }
            )
        return (
            "请基于以下 planner 输入，为下一轮输出结构化裁决。\n"
            "你只能使用候选计划里已有的 id。\n\n"
            f"planner_input={planner_input.to_central_controller_payload()}\n\n"
            f"candidate_plan_top={top_candidates}\n\n"
            "data_dictionary_display_summary="
            f"{_dictionary_display_block(planner_input.data_dictionary_summary)}\n"
            "注意：feature_focus 与解释性文本中的变量必须使用展示名"
            "（display_feature_candidates/display_target_candidates），禁止直接输出原始 csv 表头。\n"
            f"baseline_summary={baseline_output.summary}\n"
            f"baseline_protocol_refinement_count={len(baseline_output.protocol_refinements)}\n"
            "请输出 JSON，包括：summary、candidate_focus_ids、candidate_rationale、"
            "target_hypothesis_id、interpretation、related_uncertainty_ids、impact_direction、"
            "impact_strength、confidence、uncertainty_priority_action、feature_focus、"
            "protocol_notes、suggested_model_parameters。"
        )

    def _build_fallback_response(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        candidate_plan: CandidateExperimentSet,
        baseline_output: ReasoningPlannerOutput,
    ) -> CentralControllerLLMResponse:
        scored = sorted(
            (
                candidate
                for candidate in candidate_plan.candidates
                if candidate.utility_score is not None
            ),
            key=lambda item: item.utility_score or 0.0,
            reverse=True,
        )
        top_two = scored[:2] or candidate_plan.candidates[:2]
        top_hypothesis = planner_input.active_hypotheses[0] if planner_input.active_hypotheses else None
        top_uncertainty = (
            planner_input.unresolved_uncertainties[0] if planner_input.unresolved_uncertainties else None
        )
        top_hypothesis_ref = (
            top_hypothesis.statement if top_hypothesis and top_hypothesis.statement else "当前主假设"
        )
        if top_hypothesis and top_hypothesis.statement:
            top_hypothesis_ref = (
                top_hypothesis.statement
                if len(top_hypothesis.statement) <= 56
                else f"{top_hypothesis.statement[:56]}…"
            )
        semantic = VariableSemanticService.from_summary(planner_input.data_dictionary_summary)
        feature_focus = []
        feedback_text = (planner_input.human_feedback or "").lower()
        for feature in planner_input.data_dictionary_summary.feature_candidates:
            if semantic.matches_text(feature, feedback_text):
                feature_focus.append(feature)
        return CentralControllerLLMResponse(
            summary=baseline_output.summary or "中央控制器采用本地对照组/实验组策略生成下一轮规划。",
            candidate_focus_ids=[candidate.experiment_id for candidate in top_two],
            candidate_rationale=(
                "优先细化 "
                + "、".join(candidate.experiment_id for candidate in top_two)
                + "，因为它们在当前候选计划中综合价值较高、设计互不相同，且更贴近最近反馈。"
                if top_two
                else "当前无候选可供细化。"
            ),
            target_hypothesis_id=top_hypothesis.hypothesis_id if top_hypothesis else None,
            interpretation=(
                f"结合上一轮评价与当前分歧，继续围绕“{top_hypothesis_ref}”做更保守、可验证的下一轮推进。"
                if top_hypothesis
                else "当前没有明确活跃假设，优先保持探索性规划。"
            ),
            related_uncertainty_ids=[top_uncertainty.uncertainty_id] if top_uncertainty else [],
            impact_direction="supports"
            if planner_input.evaluation_summary.delta_pearson_r and planner_input.evaluation_summary.delta_pearson_r > 0
            else "clarifies",
            impact_strength=0.58 if planner_input.evaluation_summary.stable is not False else 0.42,
            confidence=0.72 if planner_input.evaluation_summary.stable is not False else 0.55,
            uncertainty_priority_action=(
                "decrease"
                if planner_input.evaluation_summary.delta_pearson_r
                and planner_input.evaluation_summary.delta_pearson_r > 0
                and planner_input.evaluation_summary.stable is not False
                else "maintain"
            ),
            feature_focus=feature_focus,
            protocol_notes=["llm_fallback_controller"],
            suggested_model_parameters={"alpha": 0.2} if feature_focus else {},
        )

    def _compose_output(
        self,
        *,
        planner_input: ReasoningPlannerInput,
        candidate_plan: CandidateExperimentSet,
        baseline_output: ReasoningPlannerOutput,
        llm_response: CentralControllerLLMResponse,
    ) -> ReasoningPlannerOutput:
        candidate_index = {candidate.experiment_id: candidate for candidate in candidate_plan.candidates}
        valid_hypothesis_ids = {item.hypothesis_id for item in planner_input.active_hypotheses}
        valid_uncertainty_ids = {item.uncertainty_id for item in planner_input.unresolved_uncertainties}
        semantic = VariableSemanticService.from_summary(planner_input.data_dictionary_summary)
        known_raw_names = set(semantic.raw_names())
        # LLM 的 feature_focus 可能是散文而不是变量名；只有字典中存在真实
        # raw 名称的条目才能进入变量字段，其余只作为审计上下文保留。
        feature_focus_raw = [
            item
            for item in semantic.raw_list(llm_response.feature_focus)
            if item in known_raw_names
        ]

        focused_candidates = [
            candidate_index[candidate_id]
            for candidate_id in llm_response.candidate_focus_ids
            if candidate_id in candidate_index
        ]
        if not focused_candidates:
            focused_candidates = [
                item.candidate.model_copy(deep=True) for item in baseline_output.candidate_supplements[:2]
            ]
        else:
            focused_candidates = [candidate.model_copy(deep=True) for candidate in focused_candidates[:2]]

        primary_candidates: list[str] = []
        if focused_candidates:
            focused_design = focused_candidates[0].design
            primary_candidates.extend(
                [
                    focused_design.design_focus or "",
                    *(focused_design.treatment or []),
                    *(focused_design.control or []),
                ]
            )
        if planner_input.data_dictionary_summary.feature_candidates:
            primary_candidates.append(
                planner_input.data_dictionary_summary.feature_candidates[0]
            )
        primary_x = next(
            (item for item in primary_candidates if item in known_raw_names),
            None,
        )
        candidate_supplements: list[PlannerCandidateSupplement] = []
        for index, candidate in enumerate(focused_candidates, start=1):
            _sanitize_candidate_variables(
                candidate,
                known_raw_names=known_raw_names,
                semantic=semantic,
            )
            _rewrite_candidate_design_focus(
                candidate=candidate,
                feature_focus_raw=feature_focus_raw,
                primary_x=primary_x,
                semantic=semantic,
            )
            for feature in feature_focus_raw:
                note = f"llm_controller_feature_focus:{feature}"
                if note not in candidate.design.notes:
                    candidate.design.notes.append(note)
            if "llm_controller_selected" not in candidate.design.notes:
                candidate.design.notes.append("llm_controller_selected")
            candidate_supplements.append(
                PlannerCandidateSupplement(
                    proposal_id=f"LCP{index:03d}",
                    candidate=candidate,
                    rationale=llm_response.candidate_rationale,
                    source="central_controller_llm",
                )
            )

        target_hypothesis_id = (
            llm_response.target_hypothesis_id
            if llm_response.target_hypothesis_id in valid_hypothesis_ids
            else (planner_input.active_hypotheses[0].hypothesis_id if planner_input.active_hypotheses else None)
        )
        related_uncertainties = [
            uncertainty_id
            for uncertainty_id in llm_response.related_uncertainty_ids
            if uncertainty_id in valid_uncertainty_ids
        ]
        if not related_uncertainties:
            related_uncertainties = [
                item.uncertainty_id for item in planner_input.unresolved_uncertainties[:2]
            ]

        interpretation_enhancements = list(baseline_output.interpretation_enhancements)
        interpretation_enhancements.insert(
            0,
            InterpretationEnhancement(
                enhancement_id="LIE001",
                target_hypothesis_id=target_hypothesis_id,
                interpretation=llm_response.interpretation,
                evidence_basis=planner_input.evaluation_summary.key_findings[:2],
                related_uncertainties=related_uncertainties,
                suggested_state_note="由中央控制 LLM 生成的解释聚焦。",
                impact_direction=_normalize_impact_direction(llm_response.impact_direction),
                impact_strength=llm_response.impact_strength,
                confidence=llm_response.confidence,
                uncertainty_priority_action=_normalize_priority_action(
                    llm_response.uncertainty_priority_action
                ),
            ),
        )

        protocol_refinements = list(baseline_output.protocol_refinements)
        target_candidate_id = (
            candidate_supplements[0].candidate.experiment_id if candidate_supplements else None
        )
        protocol_refinements.insert(
            0,
            ProtocolRefinementSuggestion(
                suggestion_id="LPR001",
                target_candidate_id=target_candidate_id,
                refinement_type="llm_central_controller",
                rationale=llm_response.candidate_rationale,
                suggested_model_parameters=_filter_model_parameters(
                    llm_response.suggested_model_parameters
                ),
                suggested_feature_focus=[
                    feature
                    for feature in feature_focus_raw
                    if feature in planner_input.data_dictionary_summary.feature_candidates
                ],
                protocol_notes=["llm_central_controller"] + llm_response.protocol_notes,
            )
        )

        return ReasoningPlannerOutput(
            task_id=planner_input.task_id,
            source_round_id=planner_input.source_round_id,
            next_round_id=planner_input.next_round_id,
            source_experiment_id=planner_input.source_experiment_id,
            planner_mode="llm",
            candidate_supplements=candidate_supplements,
            interpretation_enhancements=interpretation_enhancements,
            protocol_refinements=protocol_refinements,
            summary=llm_response.summary,
        )


def _sanitize_candidate_variables(
    candidate,
    *,
    known_raw_names: set[str],
    semantic: VariableSemanticService,
) -> None:
    """Remove prose that leaked into variable fields before rewriting focus."""
    design = candidate.design
    control = list(dict.fromkeys(item for item in design.control if item in known_raw_names))
    treatment = list(dict.fromkeys(item for item in design.treatment if item in known_raw_names))
    if set(control) != set(design.control) or set(treatment) != set(design.treatment):
        design.control = control
        design.treatment = treatment
        design.display_control = semantic.display_list(control)
        design.display_treatment = semantic.display_list(treatment)
        design.notes.append("llm_controller_sanitized:removed_unmapped_prose")
    if design.design_focus not in known_raw_names:
        fallback = next(
            (item for item in [*treatment, *control] if item in known_raw_names),
            None,
        )
        if fallback is not None:
            design.design_focus = fallback
            design.display_design_focus = semantic.to_display(fallback)


def _rewrite_candidate_design_focus(
    *,
    candidate,
    feature_focus_raw: list[str],
    primary_x: str | None,
    semantic: VariableSemanticService,
) -> bool:
    """Rebuild a selected candidate around the LLM's feature focus."""
    new_focus = next((feature for feature in feature_focus_raw if feature), None)
    if not new_focus:
        return False

    design = candidate.design
    old_focus = design.design_focus
    if old_focus == new_focus:
        return False

    if primary_x and old_focus == primary_x:
        # feature_focus selects a control-conditioning variable here, not the
        # variable under test; keep the primary tested variable in both arms.
        control = list(design.control)
        treatment = list(design.treatment)
        if new_focus not in control:
            control.append(new_focus)
        if primary_x not in treatment:
            treatment.insert(0, primary_x)
        if new_focus not in treatment:
            treatment.append(new_focus)
        design.control = control
        design.treatment = treatment
        design.design_focus = old_focus
        design.display_design_focus = semantic.to_display(old_focus)
        design.display_control = semantic.display_list(design.control)
        design.display_treatment = semantic.display_list(design.treatment)
        design.notes = [
            note for note in design.notes if not note.startswith("design_focus:")
        ]
        design.notes.append(f"design_focus:{old_focus}")
        design.notes.append(f"llm_controller_focus_rewrite:primary->{new_focus}")
        return True

    control = [
        feature
        for feature in design.control
        if feature not in {old_focus, new_focus}
    ]
    treatment = [
        feature
        for feature in design.treatment
        if feature != old_focus
    ]
    treatment = list(dict.fromkeys([*control, *treatment, new_focus]))
    control = list(dict.fromkeys(control))
    treatment = list(dict.fromkeys(treatment))

    design.control = control
    design.treatment = treatment
    design.design_focus = new_focus
    if old_focus and old_focus in design.lags:
        lag_values = design.lags.pop(old_focus)
        design.lags[new_focus] = lag_values

    display_old = semantic.to_display(old_focus) if old_focus else None
    display_new = semantic.to_display(new_focus)
    if display_old and display_old != display_new:
        if candidate.purpose:
            candidate.purpose = candidate.purpose.replace(display_old, display_new)
        if candidate.distinguishing_insight:
            candidate.distinguishing_insight = candidate.distinguishing_insight.replace(
                display_old,
                display_new,
            )

    design.notes = [
        note for note in design.notes if not note.startswith("design_focus:")
    ]
    design.notes.append(f"design_focus:{new_focus}")
    design.notes.append(f"llm_controller_focus_rewrite:{old_focus or 'none'}->{new_focus}")

    design.display_design_focus = display_new
    design.display_control = semantic.display_list(design.control)
    design.display_treatment = semantic.display_list(design.treatment)
    return True


def _normalize_impact_direction(direction: str) -> str:
    if direction in {"supports", "weakens", "clarifies"}:
        return direction
    return "clarifies"


def _normalize_priority_action(action: str) -> str:
    if action in {"increase", "decrease", "maintain"}:
        return action
    return "maintain"


def _dictionary_display_block(summary) -> str:
    display_features = list(summary.display_feature_candidates or summary.feature_candidates)
    display_targets = list(summary.display_target_candidates or summary.target_candidates)
    display_time = summary.display_time_column or summary.time_column
    return (
        "{"
        f'"dataset_name": "{summary.dataset_name}", '
        f'"display_time_column": "{display_time}", '
        f'"display_target_candidates": {display_targets}, '
        f'"display_feature_candidates": {display_features}, '
        f'"raw_display_map": {dict(summary.raw_display_map or {})}'
        "}"
    )


def _filter_model_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    allowed = {"window_size", "use_lag_feature", "max_lag_day", "alpha", "l1_ratio", "random_state"}
    filtered: dict[str, Any] = {}
    for key, value in parameters.items():
        if key not in allowed:
            continue
        if isinstance(value, (int, float, bool)):
            filtered[key] = value
    return filtered
