from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

from core.state_repository import UnifiedStateRepository
from core.unified_schema import (
    CandidateExperiment,
    CandidateExperimentSet,
    ContrastSpec,
    DataDictionary,
    DecisionLog,
    EstimatedValue,
    ExperimentDesign,
    ExperimentMemoryState,
    HypothesisPrediction,
    HypothesisTreeState,
    MAX_ACTIVE_UNCERTAINTIES,
    PrioritizedUncertainty,
    PriorityFactors,
    RejectedCandidate,
    ReasoningPlannerInput,
    RunSpec,
    ScientificTask,
    UncertaintyDisagreement,
    UncertaintyPriorityQueue,
    UncertaintyRecord,
    UncertaintyState,
)
from core.hypothesis_generation import HypothesisGenerationService
from core.hypothesis_identity import infer_related_hypothesis_ids
from core.experiment_planner_llm import CandidateExperimentDesignerLLM, CandidateExperimentWriterLLM
from core.experiment_tools.registry import normalize_model_family
from core.display_text_cleaner import clean_llm_text
from core.runtime_config import (
    get_bailian_model_prices,
    get_llm_model_for_role,
    get_runtime_setting,
)
from core.variable_semantic_service import VariableSemanticService

EPSILON = 1e-8


DEFAULT_LLM_INPUT_PRICE = 0.80
DEFAULT_LLM_OUTPUT_PRICE = 2.70


WINDOW_PRESETS: tuple[dict[str, int], ...] = (
    {"forecast_horizon_days": 3, "past_lag_days": 3, "window_size": 14},
    {"forecast_horizon_days": 3, "past_lag_days": 10, "window_size": 10},
    {"forecast_horizon_days": 5, "past_lag_days": 7, "window_size": 21},
    {"forecast_horizon_days": 7, "past_lag_days": 5, "window_size": 28},
)

IG_METRIC_PRIORITY: tuple[str, ...] = ("Skill", "Pearson_r", "RMSE", "MAE", "R2")

FROZEN_TREE_EVENTS = frozenset(
    {
        "hypothesis_tree_confirmed",
        "scientific_questioning_completed",
    }
)


def _tree_is_frozen_for(
    tree: HypothesisTreeState,
    target_round: int | None,
    decision_log: DecisionLog | None = None,
) -> bool:
    """A human-confirmed tree is authoritative and must not be regenerated.

    The in-tree confirmation event is the primary signal; the persisted decision
    log is a durable fallback so post-processing (state-machine reconciliation,
    metadata refresh, etc.) can never silently un-freeze the tree.
    """
    latest = tree.latest_update
    expected_round = max(int(target_round if target_round is not None else tree.current_round or 0), 0)
    if (
        latest is not None
        and latest.event in FROZEN_TREE_EVENTS
        and latest.round >= expected_round
    ):
        return True
    if decision_log is not None:
        confirmed_rounds = [
            int(entry.round_id or 0)
            for entry in decision_log.decisions
            if entry.decision_type == "hypothesis_tree_confirmed"
        ]
        return bool(confirmed_rounds) and max(confirmed_rounds) >= expected_round
    return False


@dataclass(frozen=True)
class PlanningContext:
    guidance_text: str | None = None
    preferred_features: tuple[str, ...] = ()
    preferred_keywords: tuple[str, ...] = ()
    prefer_simple_design: bool = False


DIFFERENTIATING_EXPERIMENT_ERROR = "实验组与对照组变量无差异，无法开展区分性对照实验，请重新生成实验方案"


def _is_baseline_candidate(candidate: CandidateExperiment) -> bool:
    return (
        candidate.type == "baseline_benchmark"
        or candidate.experiment_type == "baseline_benchmark"
    )


def _entry_model_families(entry) -> list[str]:
    raw = list(getattr(entry, "model_families", None) or [])
    if raw:
        return [name for name in (normalize_model_family(str(item)) for item in raw) if name]
    metrics = getattr(entry, "metrics_snapshot", None)
    if metrics is not None:
        return list(
            dict.fromkeys(
                name
                for run in getattr(metrics, "runs", []) or []
                for name in [normalize_model_family(getattr(run, "model_family", None) or "")]
                if name
            )
        )
    return []


def _entry_design_params(entry) -> dict[str, Any]:
    """Recover forecast horizon / history length from a completed memory entry."""

    params: dict[str, Any] = {}
    metrics = getattr(entry, "metrics_snapshot", None)
    if metrics is None:
        return params
    for run in getattr(metrics, "runs", []) or []:
        run_params = getattr(run, "parameters", None) or {}
        if not isinstance(run_params, dict):
            continue
        for key in ("forecast_horizon_days", "forecast_horizon", "horizon_days"):
            try:
                params["forecast_horizon_days"] = int(run_params[key])
                break
            except (KeyError, TypeError, ValueError):
                continue
        for key in ("past_lag_days", "past_lag"):
            try:
                params["past_lag_days"] = int(run_params[key])
                break
            except (KeyError, TypeError, ValueError):
                continue
        for key in ("window_size", "window"):
            try:
                params["window_size"] = int(run_params[key])
                break
            except (KeyError, TypeError, ValueError):
                continue
    if getattr(entry, "metrics_snapshot", None) is not None and getattr(metrics, "runs", None):
        feature_set: list[str] = []
        for run in metrics.runs:
            columns = list(getattr(run, "feature_columns", None) or [])
            if not columns:
                columns = list(getattr(run, "feature_labels", None) or [])
            feature_set = list(dict.fromkeys([*feature_set, *columns]))
        if feature_set:
            params["feature_set"] = feature_set
    return params


def _latest_window_params(experiment_memory: ExperimentMemoryState | None) -> dict[str, int]:
    """Return the window configuration of the latest completed experiment."""

    if experiment_memory is None:
        return {}
    completed = [entry for entry in experiment_memory.entries if entry.status == "completed"]
    if not completed:
        return {}
    params = _entry_design_params(completed[-1])
    return {
        key: int(params[key])
        for key in ("forecast_horizon_days", "past_lag_days", "window_size")
        if key in params and params[key] is not None
    }


def _candidate_traceability(
    *,
    record: UncertaintyRecord,
    design: ExperimentDesign,
    experiment_type: str,
    experiment_memory: ExperimentMemoryState | None,
    next_round: int,
) -> dict[str, Any]:
    """Pre-register what this candidate responds to and which factors it alters."""

    held_fixed: list[str] = ["目标变量", "评价指标与日历对齐规则", "训练/评价时间划分"]
    if design.probe_axis == "cross_window_stability":
        held_fixed = [item for item in held_fixed if item != "训练/评价时间划分"]
        held_fixed.append("评价窗口按冻结回测逐窗口分别报告（dev/f1/f2）")
    changed: list[str] = []
    prior_finding = None
    mining_sources = list(getattr(record, "mining_sources", None) or [])
    is_external_diagnostic = (
        str(getattr(record, "uncertainty_id", "")).startswith("U_EXT_")
        or any(str(item).startswith("external_diagnostic:") for item in mining_sources)
    )
    completed = (
        [entry for entry in experiment_memory.entries if entry.status == "completed"]
        if experiment_memory
        else []
    )
    latest_entry = completed[-1] if completed else None
    if latest_entry is not None:
        prior_entry_id = latest_entry.experiment_id
        findings = list(getattr(latest_entry, "key_findings", None) or [])
        prior_finding = (
            f"实验 {prior_entry_id} 的上一轮发现"
            + (f"：{'；'.join(findings[:2])}" if findings else "（以最新完成实验为重点）")
        )
        prior_params = _entry_design_params(latest_entry)
        prior_horizon = prior_params.get("forecast_horizon_days")
        prior_lag = prior_params.get("past_lag_days")
        prior_window = prior_params.get("window_size")
        prior_features = {str(item) for item in prior_params.get("feature_set", [])}
        prior_families = _entry_model_families(latest_entry)
        if prior_horizon is None or design.forecast_horizon_days is None or int(design.forecast_horizon_days) != int(prior_horizon):
            if design.forecast_horizon_days is not None and prior_horizon is not None:
                changed.append(
                    f"预测超前天数（{int(prior_horizon)} 天 → {int(design.forecast_horizon_days)} 天）"
                )
            else:
                changed.append("预测超前天数")
        elif design.forecast_horizon_days is not None:
            held_fixed.append(f"预测超前 {int(design.forecast_horizon_days)} 天")
        if prior_lag is None or design.past_lag_days is None or int(design.past_lag_days) != int(prior_lag):
            if design.past_lag_days is not None and prior_lag is not None:
                changed.append(f"历史长度（{int(prior_lag)} 天 → {int(design.past_lag_days)} 天）")
            else:
                changed.append("历史长度")
        elif design.past_lag_days is not None:
            held_fixed.append(f"历史长度 {int(design.past_lag_days)} 天")
        if prior_window is None or design.window_size is None or int(design.window_size) != int(prior_window):
            if design.window_size is not None and prior_window is not None:
                changed.append(f"时间窗口（{int(prior_window)} 天 → {int(design.window_size)} 天）")
            else:
                changed.append("时间窗口")
        elif design.window_size is not None:
            held_fixed.append(f"时间窗口 {int(design.window_size)} 天")
        current_features = {str(item) for item in [*design.control, *design.treatment]}
        if prior_features and current_features != prior_features:
            changed.append("自变量特征集")
        elif prior_features:
            held_fixed.append("自变量特征集")
        current_families = [
            name
            for name in (normalize_model_family(str(item)) for item in design.model_families)
            if name
        ]
        if current_families and prior_families and current_families != prior_families:
            changed.append("模型家族")
        elif current_families and prior_families:
            held_fixed.append("模型家族")
    else:
        prior_finding = "本轮为首轮实验：以当前科学不确定性为基线，回答其能否被实验证据区分。"
        held_fixed.extend(
            [
                item
                for item in [
                    (
                        f"预测超前 {design.forecast_horizon_days} 天"
                        if design.forecast_horizon_days
                        else None
                    ),
                    (
                        f"历史长度 {design.past_lag_days} 天"
                        if design.past_lag_days
                        else None
                    ),
                    (
                        f"时间窗口 {design.window_size} 天"
                        if design.window_size
                        else None
                    ),
                ]
                if item
            ]
        )
    if is_external_diagnostic:
        prior_finding = (
            "外部补充回测的上一轮发现：dev 段 mag_both（提前 3 天、训练内自动选参）"
            "唯一不跨零的 RMSE 改善未获得跨窗口一致支持。"
            "触发证据：dev 小幅正向；f1 点估计转负；f2 点估计同向但区间跨零。"
            "因此改变：优先核验跨时段可重复性，并保持原始特征协议。"
            "拟检验问题：当前结果不足，主要反映估计不精确，还是存在条件依赖？"
        )
    if not changed:
        changed.append("在本轮固定条件下验证/复测与上一轮相关的不确定性")
    exploratory = False
    meaningful_changed = [item for item in changed if item != "在本轮固定条件下验证/复测与上一轮相关的不确定性"]
    if len(meaningful_changed) > 1:
        exploratory = True

    type_key = experiment_type or "incremental_prediction"
    branches: dict[str, str] = {}
    if design.probe_axis == "cross_window_stability":
        branches = {
            "dev、f1、f2 增益点估计同向且区间均不跨零": "记录跨窗口一致支持，并继续在未参与选择的时段做独立验证。",
            "窗口间方向不一致或点估计同向但区间跨零": "结论写“未获得跨窗口一致支持”，仅保留局部窗口的探索性证据，不将其更新为稳定增益支持。",
            "所有窗口均未检出增益": "保留“未检出稳定增益”的保守结论，下一轮核查数据质量与协议边界。",
        }
    elif type_key == "model_sensitivity":
        branches = {
            "两个模型家族均检出稳定正向日影增益": "把稳定性作为下一轮核心结论的暂定支持证据，继续做固定任务下的正则化敏感性核查。",
            "两个模型家族均检出稳定负向增益": "系统性地加入日影反而带来一致退化，先检查数据对齐、缺失掩码与样本范围，再决定是否调整特征构造。",
            "两个模型家族方向相反或点估计方向一致但 CI 跨零": "先认定“未检出稳定增益”，区分模型设置依赖与数据质量问题，不把点估计方向当作假设支持。",
            "部分家族出现系数被筛零或常数预测": "核查正则会否掩盖信号，并检查预测文件与共同样本，必要时固定任务做单因素敏感性实验。",
        }
    else:
        branches = {
            "CI 全部落在正区间": "把“稳定改善”作为暂定证据，下一轮只变更一个因素复核其稳健性。",
            "CI 全部落在负区间": "把当前方向视为退化证据，下一轮检查数据质量或改回上一轮条件。",
            "CI 跨零": "结论写“未检出稳定增益”，下一轮优先核验跨零原因而不是继续加分。",
        }

    return {
        "responded_prior_finding": prior_finding,
        "target_uncertainty_focus": record.question,
        "changed_factors": list(dict.fromkeys(changed)),
        "held_fixed_factors": list(dict.fromkeys(held_fixed)),
        "expected_outcome_branches": branches,
        "exploratory": exploratory,
    }


def _normalize_design_features(features: list[str]) -> list[str]:
    return _unique_preserve_order([item.strip() for item in features if item and item.strip()])


def _design_arm_time_differs(design: ExperimentDesign) -> bool:
    """True when the two arms differ by time window even if raw features match."""
    return (
        (design.control_lag_days is not None or design.treatment_lag_days is not None)
        and design.control_lag_days != design.treatment_lag_days
    ) or (
        (design.control_forecast_horizon_days is not None or design.treatment_forecast_horizon_days is not None)
        and design.control_forecast_horizon_days != design.treatment_forecast_horizon_days
    )


def _validate_candidate_design(candidate: CandidateExperiment) -> None:
    candidate.design.control = _normalize_design_features(candidate.design.control)
    candidate.design.treatment = _normalize_design_features(candidate.design.treatment)
    if _is_baseline_candidate(candidate):
        if not candidate.design.treatment:
            raise ValueError("对照组实验至少需要保留一组对照组变量。")
        candidate.design.control = []
        if "experiment_mode:baseline_single_arm" not in candidate.design.notes:
            candidate.design.notes.append("experiment_mode:baseline_single_arm")
        return
    if not candidate.design.control or not candidate.design.treatment:
        raise ValueError("区分性实验必须同时提供对照组与实验组变量。")
    if (
        set(candidate.design.control) == set(candidate.design.treatment)
        and not _design_arm_time_differs(candidate.design)
    ):
        raise ValueError(DIFFERENTIATING_EXPERIMENT_ERROR)


class UncertaintyPrioritizer:
    """Programmatic prioritizer that scores open scientific uncertainties."""

    def prioritize(
        self,
        uncertainty_state: UncertaintyState,
        hypothesis_tree: HypothesisTreeState,
        target_round: int | None = None,
    ) -> UncertaintyPriorityQueue:
        _refresh_uncertainty_disagreement(uncertainty_state, hypothesis_tree)
        node_index = hypothesis_tree.node_index()
        queue: list[PrioritizedUncertainty] = []

        for record in uncertainty_state.records:
            if record.status in {"resolved", "deprecated"}:
                continue

            related_scores = [
                node_index[hypothesis_id].support_score
                for hypothesis_id in record.related_hypotheses
                if hypothesis_id in node_index
            ]
            support_gap = (max(related_scores) - min(related_scores)) if len(related_scores) >= 2 else (
                related_scores[0] if related_scores else 0.3
            )
            hypothesis_count = len(record.related_hypotheses)
            expected_information_gain = min(
                1.0,
                0.35 + 0.15 * hypothesis_count + 0.2 * (1.0 if "independent" in record.question.lower() else 0.0),
            )
            estimated_cost = _estimate_uncertainty_cost(record)
            override = record.priority_factors.human_priority_override if record.priority_factors else None

            record.priority_factors = PriorityFactors(
                hypothesis_count=hypothesis_count,
                support_gap=round(min(max(support_gap, 0.0), 1.0), 4),
                expected_information_gain=round(expected_information_gain, 4),
                estimated_cost=round(estimated_cost, 4),
                human_priority_override=override,
            )
            priority_score = _combine_priority(record.priority, record.priority_factors, record.resolution_status)
            queue.append(
                PrioritizedUncertainty(
                    uncertainty_id=record.uncertainty_id,
                    question=record.question,
                    priority_score=priority_score,
                    status=record.status,
                    estimated_resolution_round=(
                        target_round
                        if target_round is not None
                        else uncertainty_state.current_round + 1
                    ),
                )
            )

        queue = sorted(queue, key=lambda item: item.priority_score, reverse=True)
        seen_ids: set[str] = set()
        bounded_queue: list[PrioritizedUncertainty] = []
        for item in queue:
            if item.uncertainty_id in seen_ids:
                continue
            seen_ids.add(item.uncertainty_id)
            bounded_queue.append(item)
            if len(bounded_queue) >= MAX_ACTIVE_UNCERTAINTIES:
                break
        queue = bounded_queue
        uncertainty_state.priority_queue = UncertaintyPriorityQueue(
            last_updated=datetime.now(),
            current_round=uncertainty_state.current_round,
            queue=queue,
            resolved=[item.uncertainty_id for item in uncertainty_state.records if item.status == "resolved"],
            deprecated=[item.uncertainty_id for item in uncertainty_state.records if item.status == "deprecated"],
        )
        return uncertainty_state.priority_queue


class CandidateExperimentGenerator:
    """Generate a minimal set of candidate experiments from prioritized uncertainties."""

    def generate(
        self,
        *,
        task: ScientificTask,
        data_dictionary: DataDictionary,
        hypothesis_tree: HypothesisTreeState,
        uncertainty_state: UncertaintyState,
        experiment_memory: ExperimentMemoryState | None = None,
        planning_context: PlanningContext | None = None,
        limit: int | None = None,
        target_round: int | None = None,
    ) -> CandidateExperimentSet:
        if limit is None:
            budget = task.payload.constraints.resource_budget
            candidate_budget = (
                budget.max_candidates
                if budget and budget.max_candidates
                else task.payload.constraints.max_experiments_per_round or 6
            )
            limit = max(min(candidate_budget, 8), 3)
        planning_context = planning_context or PlanningContext()
        next_round = (
            target_round
            if target_round is not None
            else uncertainty_state.current_round + 1
        )
        semantic = VariableSemanticService.from_data_dictionary(data_dictionary)
        _refresh_uncertainty_disagreement(uncertainty_state, hypothesis_tree)
        variables = task.payload.research_question.variables
        target = task.payload.research_question.target
        primary_x = variables.x if variables else (data_dictionary.feature_candidates[0] if data_dictionary.feature_candidates else target)
        mediator_candidates = variables.m_candidates if variables else []
        # feature_candidates may contain only the user-selected x variable.
        # Build the programmatic baseline from all real feature fields so the
        # first draft can still provide a 对照组 when no secondary candidates
        # were pre-selected.
        feature_pool = _unique_preserve_order(
            [
                *data_dictionary.feature_candidates,
                *[
                    item.field_name
                    for item in data_dictionary.fields
                    if not item.is_target
                    and item.field_name != data_dictionary.time_column
                    and item.physical_meaning not in {"time", "弃用字段"}
                ],
            ]
        )
        feature_pool = [item for item in feature_pool if item != primary_x]
        control_base = _unique_preserve_order(mediator_candidates or feature_pool[:3])
        queue = uncertainty_state.priority_queue or UncertaintyPriorityQueue(
            last_updated=datetime.now(),
            current_round=uncertainty_state.current_round,
            queue=[],
        )
        node_index = hypothesis_tree.node_index()
        simple_focus_override = bool(
            planning_context.prefer_simple_design and planning_context.preferred_features
        )
        candidates: list[CandidateExperiment] = []
        uncertainty_limit = max(limit, 1)
        if simple_focus_override:
            uncertainty_limit = max(uncertainty_limit, 1)
        uncertainty_candidates = self._build_uncertainty_candidates(
            uncertainty_state=uncertainty_state,
            uncertainty_queue=queue,
            hypothesis_tree=hypothesis_tree,
            node_index=node_index,
            task=task,
            data_dictionary=data_dictionary,
            planning_context=planning_context,
            primary_x=primary_x,
            target=target,
            mediator_candidates=mediator_candidates,
            feature_pool=feature_pool,
            semantic=semantic,
            limit=uncertainty_limit,
            experiment_memory=experiment_memory,
            next_round=next_round,
        )
        candidates.extend(uncertainty_candidates)
        rejected_candidates: list[RejectedCandidate] = []
        for candidate in candidates:
            _annotate_candidate_estimates(
                candidate=candidate,
                task=task,
                data_dictionary=data_dictionary,
                hypothesis_tree=hypothesis_tree,
                experiment_memory=experiment_memory,
                current_round=next_round,
            )

        for candidate in candidates:
            _apply_semantic_to_candidate_display(candidate, semantic, node_index)
            _validate_candidate_design(candidate)
            if candidate.estimated_information_gain is None:
                _annotate_candidate_estimates(
                    candidate=candidate,
                    task=task,
                    data_dictionary=data_dictionary,
                    hypothesis_tree=hypothesis_tree,
                    experiment_memory=experiment_memory,
                    current_round=next_round,
                )

        if experiment_memory:
            seen_ids = {entry.experiment_id for entry in experiment_memory.entries}
            for candidate in list(candidates):
                if candidate.experiment_id in seen_ids:
                    candidates.remove(candidate)
                    rejected_candidates.append(
                        RejectedCandidate(
                            experiment_id=candidate.experiment_id,
                            reason="experiment_id already exists in experiment_memory",
                            similar_to=candidate.experiment_id,
                        )
                    )

        return CandidateExperimentSet(
            round=next_round,
            generated_at=datetime.now(),
            candidates=candidates,
            rejected_candidates=rejected_candidates,
            minimum_required=3,
            note=_candidate_set_note(len(candidates), planning_context),
        )

    def _build_uncertainty_candidates(
        self,
        *,
        uncertainty_state: UncertaintyState,
        uncertainty_queue: UncertaintyPriorityQueue,
        hypothesis_tree: HypothesisTreeState,
        node_index: dict[str, object],
        task: ScientificTask,
        data_dictionary: DataDictionary,
        planning_context: PlanningContext,
        primary_x: str,
        target: str,
        mediator_candidates: list[str],
        feature_pool: list[str],
        semantic: VariableSemanticService,
        experiment_memory: ExperimentMemoryState | None = None,
        limit: int,
        next_round: int,
    ) -> list[CandidateExperiment]:
        """Generate one candidate per uncertainty so every open scientific
        question independently drives an experiment."""
        if limit <= 0:
            return []

        control_base = _unique_preserve_order(mediator_candidates or feature_pool[:3])
        record_index = uncertainty_state.record_index()
        ranked_items = _rank_uncertainties_with_feedback(
            uncertainty_queue.top_uncertainties(limit=max(limit * 4, 12)),
            uncertainty_state,
            planning_context,
        )
        candidates: list[CandidateExperiment] = []
        used_signatures = set()

        for item in ranked_items:
            if len(candidates) >= limit:
                break
            record = record_index.get(item.uncertainty_id)
            if record is None or not record.question:
                continue

            design = _design_from_uncertainty(
                record=record,
                target=target,
                primary_x=primary_x,
                control_base=control_base,
                planning_context=planning_context,
                node_index=node_index,
                semantic=semantic,
                feature_pool=feature_pool,
                used_signatures=used_signatures,
            )
            experiment_type = _candidate_experiment_type(record)
            if experiment_type != "model_sensitivity" and _discusses_model_form(record):
                capability_note = (
                    "capability_boundary:当前执行层仅接通 Ridge / ElasticNet 线性模型，"
                    "无法验证非线性机制；本候选保留为线性框架内的增量检验。"
                )
                if capability_note not in design.notes:
                    design.notes.append(capability_note)
            signature = _design_signature(design, experiment_type=experiment_type)
            if signature in used_signatures:
                continue
            used_signatures.add(signature)

            rationale_context = _candidate_rationale_context(record, node_index)
            tested_hypotheses = _focus_hypothesis_ids(
                record=record,
                focus=design.design_focus,
                node_index=node_index,
                semantic=semantic,
            )
            if not tested_hypotheses:
                tested_hypotheses = [
                    hypothesis_id
                    for hypothesis_id in record.related_hypotheses
                    if hypothesis_id in node_index
                ]
            if not tested_hypotheses:
                tested_hypotheses = infer_related_hypothesis_ids(
                    hypothesis_tree,
                    record.question or "",
                    record.description or "",
                    list(record.related_hypotheses),
                )
                tested_hypotheses = [
                    hypothesis_id
                    for hypothesis_id in tested_hypotheses
                    if hypothesis_id in node_index
                ] or _tree_active_hypothesis_ids(hypothesis_tree, node_index)
            candidate = CandidateExperiment(
                experiment_id=f"E_R{next_round:02d}_{len(candidates) + 1:02d}",
                type=_candidate_type(record),
                experiment_type=experiment_type,
                purpose=_candidate_purpose(record, rationale_context),
                scientific_question=record.question,
                tested_hypotheses=tested_hypotheses,
                related_uncertainties=[record.uncertainty_id],
                disagreement_context={
                    uncertainty_id: {
                        hypothesis_id: (
                            disagreement.model_copy(deep=True)
                            if hasattr(disagreement, "model_copy")
                            else disagreement
                        )
                        for hypothesis_id, disagreement in record.disagreement.items()
                    }
                    for uncertainty_id, item in [(record.uncertainty_id, record)]
                    if item.disagreement
                },
                hypothesis_source_context=_candidate_source_context(record, node_index),
                design=design,
                hypothesis_predictions=_build_hypothesis_predictions(
                    record=record,
                    node_index=node_index,
                    design_focus=design.design_focus,
                    semantic=semantic,
                ),
                distinguishing_insight=_distinguishing_insight(
                    record=record,
                    rationale_context=rationale_context,
                    primary_x=primary_x,
                    focus_feature=design.design_focus,
                ),
                estimated_information_gain=EstimatedValue(
                    value=0.5,
                    rationale="候选实验基于单条科学不确定性生成，正式估值在评分前统一重算。",
                ),
                estimated_performance_gain=EstimatedValue(
                    value=0.0,
                    rationale="正式 PG 估值在评分前统一重算。",
                ),
                estimated_risk=EstimatedValue(
                    value=0.4,
                    rationale="正式风险估值在评分前统一重算。",
                ),
                estimated_cost=EstimatedValue(
                    value=0.35,
                    rationale="正式成本估值在评分前统一重算。",
                ),
                requires_human_review=True,
                novelty=_build_candidate_novelty(planning_context),
            )
            candidate.design.notes.extend(_planning_notes(planning_context))
            candidate.design.notes.extend(
                _rationale_design_notes(
                    record,
                    rationale_context,
                    tree=hypothesis_tree,
                    uncertainty_records=uncertainty_state.records,
                )
            )
            inherited_window = _latest_window_params(experiment_memory)
            explicit_window = {
                key: int(getattr(candidate.design, key))
                for key in ("forecast_horizon_days", "past_lag_days", "window_size")
                if getattr(candidate.design, key) is not None
            }
            if explicit_window:
                preset = {**WINDOW_PRESETS[0], **inherited_window, **explicit_window}
                window_note = (
                    "explicit_window_design:axis=uncertainty,"
                    f"forecast_horizon={preset['forecast_horizon_days']},"
                    f"past_lag={preset['past_lag_days']},window={preset['window_size']}"
                )
            elif len(inherited_window) == 3:
                preset = inherited_window
                window_note = (
                    "window_inherited_from_previous_round:axis=uncertainty,"
                    f"forecast_horizon={preset['forecast_horizon_days']},"
                    f"past_lag={preset['past_lag_days']},window={preset['window_size']}"
                )
            else:
                preset = WINDOW_PRESETS[0]
                window_note = (
                    "initial_window_preset:axis=uncertainty,"
                    f"forecast_horizon={preset['forecast_horizon_days']},"
                    f"past_lag={preset['past_lag_days']},window={preset['window_size']}"
                )
            candidate.design.window_size = preset["window_size"]
            candidate.design.past_lag_days = preset["past_lag_days"]
            candidate.design.forecast_horizon_days = preset["forecast_horizon_days"]
            candidate.design.control_lag_days = candidate.design.control_lag_days or preset["past_lag_days"]
            candidate.design.treatment_lag_days = candidate.design.treatment_lag_days or preset["past_lag_days"]
            candidate.design.control_forecast_horizon_days = (
                candidate.design.control_forecast_horizon_days or preset["forecast_horizon_days"]
            )
            candidate.design.treatment_forecast_horizon_days = (
                candidate.design.treatment_forecast_horizon_days or preset["forecast_horizon_days"]
            )
            candidate.design.notes.append(window_note)
            if experiment_type == "model_sensitivity":
                _apply_model_sensitivity_design(candidate.design)
            traceability = _candidate_traceability(
                record=record,
                design=candidate.design,
                experiment_type=experiment_type,
                experiment_memory=experiment_memory,
                next_round=next_round,
            )
            candidate.responded_prior_finding = traceability["responded_prior_finding"]
            candidate.target_uncertainty_focus = traceability["target_uncertainty_focus"]
            candidate.changed_factors = traceability["changed_factors"]
            candidate.held_fixed_factors = traceability["held_fixed_factors"]
            candidate.expected_outcome_branches = traceability["expected_outcome_branches"]
            candidate.exploratory = traceability["exploratory"]
            try:
                _validate_candidate_design(candidate)
            except ValueError as exc:
                raise ValueError(f"{candidate.experiment_id} 设计无效: {exc}") from exc
            candidates.append(candidate)
        return candidates


class UtilityScorer:
    """Compute simplified utility scores for candidate experiments."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {
            "alpha": 0.50,
            "beta": 0.20,
            "gamma": 0.20,
            "delta": 0.10,
        }

    def score(
        self,
        candidate_set: CandidateExperimentSet,
        *,
        experiment_memory: ExperimentMemoryState | None = None,
    ) -> CandidateExperimentSet:
        beta = 0.0 if candidate_set.round <= 1 else self.weights["beta"]
        completed_entries = (
            [entry for entry in experiment_memory.entries if entry.status == "completed"]
            if experiment_memory
            else []
        )
        for candidate in candidate_set.candidates:
            ig = candidate.estimated_information_gain.value if candidate.estimated_information_gain else 0.0
            pg = candidate.estimated_performance_gain.value if candidate.estimated_performance_gain else 0.0
            risk = candidate.estimated_risk.value if candidate.estimated_risk else 0.5
            cost = candidate.estimated_cost.value if candidate.estimated_cost else 0.5
            utility = (
                self.weights["alpha"] * ig
                + beta * pg
                - self.weights["gamma"] * risk
                - self.weights["delta"] * cost
            )
            if candidate.type == "validation_followup":
                utility += 0.08
            if any(
                str(note).startswith("human_feedback_focus_single_feature:")
                for note in candidate.design.notes
            ):
                utility += 0.15
            same_design = candidate.design.control == candidate.design.treatment
            if same_design and not _is_baseline_candidate(candidate):
                utility = min(utility, 0.12)
                if "selection_penalty:control_equals_treatment" not in candidate.design.notes:
                    candidate.design.notes.append("selection_penalty:control_equals_treatment")
            if completed_entries:
                best_similarity = 0.0
                best_entry_id = None
                for entry in completed_entries:
                    similarity = _historical_similarity(candidate, entry)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_entry_id = entry.experiment_id
                if best_similarity >= 0.6 and best_entry_id:
                    penalty = 0.04 + 0.05 * min((best_similarity - 0.6) / 0.4, 1.0)
                    utility -= penalty
                    note = (
                        f"selection_penalty:executed_history_similarity:{best_similarity:.4f}:"
                        f"similar_to:{best_entry_id}"
                    )
                    if note not in candidate.design.notes:
                        candidate.design.notes.append(note)
            candidate.utility_score = round(min(max(utility, 0.0), 1.0), 4)
        # Human-mandated single-feature candidates stay above equal-utility alternatives.
        candidate_set.candidates.sort(
            key=lambda item: (
                0
                if any(
                    str(note).startswith("human_feedback_focus_single_feature:")
                    for note in item.design.notes
                )
                else 1,
                -(item.utility_score or 0.0),
            ),
        )
        return candidate_set


def recompute_candidate_plan_metadata(
    candidate_set: CandidateExperimentSet,
    *,
    task: ScientificTask,
    data_dictionary: DataDictionary,
    hypothesis_tree: HypothesisTreeState,
    uncertainty_state: UncertaintyState,
    experiment_memory: ExperimentMemoryState | None,
    current_round: int,
    rescore: bool = True,
) -> CandidateExperimentSet:
    """Recompute traceability, estimates and utility after a design changed.

    LLM refinements and PI approval overrides can alter the actual feature set,
    model families or time window. Keeping the old pre-registered narrative and
    scores would make the candidate description diverge from the final protocol,
    so this helper derives every derived field from the final design.
    """

    node_index = hypothesis_tree.node_index()
    uncertainty_index = uncertainty_state.record_index()
    semantic = VariableSemanticService.from_data_dictionary(data_dictionary)
    for candidate in candidate_set.candidates:
        record = (
            uncertainty_index.get(candidate.related_uncertainties[0])
            if candidate.related_uncertainties
            else None
        )
        if record is None:
            record = UncertaintyRecord(
                uncertainty_id=(
                    candidate.related_uncertainties[0]
                    if candidate.related_uncertainties
                    else f"TMP_{candidate.experiment_id}"
                ),
                question=(
                    candidate.scientific_question
                    or candidate.purpose
                    or "候选实验待解决的未知问题"
                ),
                description=(
                    candidate.purpose
                    or candidate.scientific_question
                    or "候选实验的生成不确定性"
                ),
                created_at_round=max(current_round, 0),
                created_by="system",
                status="active",
            )
        traceability = _candidate_traceability(
            record=record,
            design=candidate.design,
            experiment_type=candidate.experiment_type or "incremental_prediction",
            experiment_memory=experiment_memory,
            next_round=current_round,
        )
        candidate.responded_prior_finding = traceability["responded_prior_finding"]
        candidate.target_uncertainty_focus = traceability["target_uncertainty_focus"]
        candidate.changed_factors = traceability["changed_factors"]
        candidate.held_fixed_factors = traceability["held_fixed_factors"]
        candidate.expected_outcome_branches = traceability["expected_outcome_branches"]
        candidate.exploratory = traceability["exploratory"]
        _annotate_candidate_estimates(
            candidate=candidate,
            task=task,
            data_dictionary=data_dictionary,
            hypothesis_tree=hypothesis_tree,
            experiment_memory=experiment_memory,
            current_round=current_round,
        )
        _apply_semantic_to_candidate_display(candidate, semantic, node_index)
        try:
            _validate_candidate_design(candidate)
        except ValueError as exc:
            raise ValueError(f"{candidate.experiment_id} 设计在重算后无效: {exc}") from exc
    if rescore:
        return UtilityScorer().score(candidate_set, experiment_memory=experiment_memory)
    return candidate_set


class DecisionLayerService:
    """End-to-end procedural decision layer for the first non-LLM planning stage."""

    def __init__(
        self,
        repository: UnifiedStateRepository,
        prioritizer: UncertaintyPrioritizer | None = None,
        generator: CandidateExperimentGenerator | None = None,
        scorer: UtilityScorer | None = None,
        experiment_designer: CandidateExperimentDesignerLLM | None = None,
        experiment_writer: CandidateExperimentWriterLLM | None = None,
    ) -> None:
        self.repository = repository
        self.prioritizer = prioritizer or UncertaintyPrioritizer()
        self.generator = generator or CandidateExperimentGenerator()
        self.scorer = scorer or UtilityScorer()
        self.experiment_designer = experiment_designer or CandidateExperimentDesignerLLM()
        self.experiment_writer = experiment_writer or CandidateExperimentWriterLLM()

    def build_candidate_plan(
        self,
        *,
        task: ScientificTask,
        data_dictionary: DataDictionary,
        planning_feedback: str | None = None,
        planner_input: ReasoningPlannerInput | None = None,
        target_round: int | None = None,
    ) -> CandidateExperimentSet:
        tree = self.repository.load_hypothesis_tree()
        uncertainties = self.repository.load_uncertainties()
        experiment_memory = self.repository.load_experiment_memory()
        resolved_round = target_round or (planner_input.next_round_id if planner_input else None)
        decision_log = self.repository.load_decision_log()
        if not _tree_is_frozen_for(tree, resolved_round, decision_log):
            proposals = list(planner_input.llm_hypothesis_proposals) if planner_input else []
            if not tree.nodes and proposals:
                generation = HypothesisGenerationService().build_tree(
                    task=task,
                    data_dictionary=data_dictionary,
                    uncertainties=uncertainties,
                    planner_input=planner_input,
                    existing_tree=tree,
                    current_round=tree.current_round,
                )
                tree = generation.tree
                uncertainties = uncertainties.model_copy(
                    update={"records": list(generation.updated_uncertainties)}
                )
                self.repository.save_hypothesis_tree(tree)
        merged_feedback = planning_feedback or (planner_input.merged_guidance_text() if planner_input else None)
        planning_context = _build_planning_context(merged_feedback, data_dictionary)
        prioritized = self.prioritizer.prioritize(
            uncertainties,
            tree,
            target_round=resolved_round,
        )
        candidates = self.generator.generate(
            task=task,
            data_dictionary=data_dictionary,
            hypothesis_tree=tree,
            uncertainty_state=uncertainties,
            experiment_memory=experiment_memory,
            planning_context=planning_context,
            target_round=resolved_round,
        )
        semantic_service = VariableSemanticService.from_data_dictionary(data_dictionary)
        node_index = tree.node_index()
        uncertainty_index = uncertainties.record_index()
        variables = task.payload.research_question.variables
        primary_x = variables.x if variables else (
            data_dictionary.feature_candidates[0]
            if data_dictionary.feature_candidates
            else task.payload.research_question.target
        )

        def _design_one(candidate: CandidateExperiment) -> CandidateExperiment:
            record = (
                uncertainty_index.get(candidate.related_uncertainties[0])
                if candidate.related_uncertainties
                else None
            )
            self.experiment_designer.design(
                candidate=candidate,
                task=task,
                uncertainty_question=(
                    record.question if record is not None else candidate.scientific_question
                ),
                uncertainty_description=record.description if record is not None else None,
                semantic_service=semantic_service,
                node_index=node_index,
                primary_x=primary_x,
                round_id=resolved_round or candidates.round,
            )
            return candidate

        design_deadline = self._design_deadline_seconds()
        design_started = perf_counter()
        design_payloads = list(candidates.candidates)
        with ThreadPoolExecutor(
            max_workers=min(3, max(1, len(design_payloads)))
        ) as pool:
            futures = [pool.submit(_design_one, candidate) for candidate in design_payloads]
            for future in as_completed(futures):
                if design_deadline is not None and (
                    perf_counter() - design_started
                ) >= design_deadline:
                    for pending in futures:
                        pending.cancel()
                    raise TimeoutError(
                        f"候选实验设计超过统一截止时间 {design_deadline:.0f}s"
                    )
                future.result()

        self.experiment_writer.write_all(
            candidates=candidates.candidates,
            task=task,
            semantic_service=semantic_service,
            node_index=node_index,
            round_id=resolved_round or candidates.round,
            uncertainty_index=uncertainty_index,
        )
        scored = self.scorer.score(candidates, experiment_memory=experiment_memory)
        uncertainty_index = uncertainties.record_index()
        for candidate in scored.candidates:
            for uncertainty_id in candidate.related_uncertainties:
                record = uncertainty_index.get(uncertainty_id)
                if record is not None and not record.resolving_experiment:
                    record.resolving_experiment = candidate.experiment_id
        uncertainties.priority_queue = prioritized
        self.repository.save_uncertainties(uncertainties)
        self.repository.save_candidate_experiments(scored)
        return scored

    def _design_deadline_seconds(self) -> float | None:
        raw = get_runtime_setting("llm_batch_deadline_seconds", None)
        if raw is None:
            return None
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0 else None


def _refresh_uncertainty_disagreement(
    uncertainty_state: UncertaintyState,
    hypothesis_tree: HypothesisTreeState,
) -> None:
    node_index = hypothesis_tree.node_index()
    for record in uncertainty_state.records:
        if not record.related_hypotheses:
            continue
        disagreement: dict[str, UncertaintyDisagreement] = {}
        for hypothesis_id in record.related_hypotheses:
            node = node_index.get(hypothesis_id)
            if node is None:
                continue
            disagreement[hypothesis_id] = UncertaintyDisagreement(
                hypothesis_id=hypothesis_id,
                expected=_expected_from_node(node),
                status=node.status,
                support_score=node.support_score,
                rationale_trigger=node.generation_rationale.trigger if node.generation_rationale else None,
                source_summary=node.generation_rationale.summary if node.generation_rationale else None,
            )
        record.disagreement = disagreement


def _combine_priority(
    priority: str,
    factors: PriorityFactors,
    resolution_status: str | None = None,
) -> float:
    base = {"low": 0.35, "medium": 0.6, "high": 0.85}.get(priority, 0.6)
    score = (
        0.35 * base
        + 0.2 * min(1.0, factors.hypothesis_count / 3)
        + 0.2 * factors.support_gap
        + 0.2 * factors.expected_information_gain
        + 0.05 * (1 - factors.estimated_cost)
    )
    if resolution_status == "resolved":
        score *= 0.1
    elif resolution_status == "partially_resolved":
        score *= 0.72
    if factors.human_priority_override is not None:
        score = 0.7 * score + 0.3 * factors.human_priority_override
    return round(min(max(score, 0.0), 1.0), 4)


def _estimate_uncertainty_cost(record: UncertaintyRecord) -> float:
    text = f"{record.question} {record.description}".lower()
    if any(token in text for token in ("lag", "滞后", "跨时间", "time", "stability", "稳定")):
        return 0.65
    if any(token in text for token in ("mediat", "中介", "through", "independent", "独立")):
        return 0.45
    return 0.35


_TEMPORAL_STABILITY_TOKENS = (
    "跨时段",
    "跨窗口",
    "跨时间",
    "temporal stability",
    "cross-window",
    "cross_window",
)


def _mentions_temporal_stability(text: str) -> bool:
    """True when the uncertainty explicitly asks about stability across windows."""

    lowered = text.lower()
    return any(token in lowered for token in _TEMPORAL_STABILITY_TOKENS)


def _candidate_type(record: UncertaintyRecord) -> str:
    if record.resolution_status == "partially_resolved":
        return "validation_followup"
    text = f"{record.question} {record.description}".lower()
    if any(token in text for token in ("mediat", "中介", "through")):
        return "mediator_test"
    if _mentions_temporal_stability(text):
        return "cross_window_stability"
    if any(token in text for token in ("lag", "滞后")):
        return "lag_sensitivity"
    if any(token in text for token in ("independent", "独立")):
        return "incremental_gain"
    return "distinguishing"


_MODEL_SENSITIVITY_TOKENS = (
    "model family",
    "model_family",
    "ridge",
    "elasticnet",
    "elastic net",
    "算法依赖",
    "模型依赖",
    "模型家族",
    "模型稳定性",
    "model stability",
    "算法选择",
)


def _mentions_nonlinearity(text: str) -> bool:
    """True when the uncertainty explicitly asks about a nonlinear form."""

    lowered = text.lower()
    return "非线性" in lowered or "nonlinear" in lowered


def _discusses_model_form(record: UncertaintyRecord) -> bool:
    """Detect uncertainty about linear/nonlinear model form for capability notes."""

    text = f"{record.question} {record.description}".lower()
    return _mentions_nonlinearity(text) or any(
        token in text for token in ("线性", "linear", "模型形式", "model form")
    )


def _candidate_experiment_type(record: UncertaintyRecord) -> str:
    """Map an uncertainty to the registered execution type without inventing types."""

    text = f"{record.question} {record.description}".lower()
    if _mentions_nonlinearity(text):
        return "incremental_prediction"
    if any(token.lower() in text for token in _MODEL_SENSITIVITY_TOKENS):
        return "model_sensitivity"
    return "incremental_prediction"


def _apply_model_sensitivity_design(design: ExperimentDesign) -> None:
    """Compile the shared Ridge/ElasticNet sensitivity structure into the design."""

    families: list[str] = []
    for name in design.model_families or ["ridge", "elasticnet"]:
        normalized = normalize_model_family(name)
        if normalized and normalized not in families:
            families.append(normalized)
    if len(families) < 2:
        families = ["ridge", "elasticnet"]
    design.model_families = families
    design.runs_spec = []
    for family in families:
        design.runs_spec.extend(
            [
                RunSpec(
                    run_id=f"{family}_baseline",
                    label=f"{family} 对照组",
                    feature_columns=list(design.control),
                    model_family=family,
                    retrain=True,
                ),
                RunSpec(
                    run_id=f"{family}_treatment",
                    label=f"{family} 实验组",
                    feature_columns=list(design.treatment),
                    model_family=family,
                    retrain=True,
                ),
            ]
        )
    design.contrasts_spec = [
        ContrastSpec(
            contrast_id=f"gain_{family}",
            name=f"{family} 家族内日影增量",
            kind="incremental_gain",
            run_ids=[f"{family}_baseline", f"{family}_treatment"],
            direction="mixed",
            note="只比较同一模型家族内部加日影前后；禁止跨家族错误归因。",
        )
        for family in families
    ]
    capability_note = (
        "capability_boundary:model_sensitivity_仅限线性家族内的正则化与设置敏感性，"
        "不验证非线性机制"
    )
    if capability_note not in design.notes:
        design.notes.append(capability_note)


def _notes_mining_features(record: UncertaintyRecord) -> list[str]:
    """Extract explicit feature hints written by uncertainty miners."""
    if not record.notes or "mining_features:" not in record.notes:
        return []
    segment = record.notes.split("mining_features:", 1)[1].split(";", 1)[0]
    return [item.strip() for item in segment.split(",") if item.strip()]


def _uncertainty_design_strategy(
    record: UncertaintyRecord,
    text: str,
    has_non_primary_focus: bool,
) -> str:
    """Pick an experiment structure from the uncertainty's scientific signal."""
    has_lag = any(token in text for token in ("lag", "滞后", "时滞"))
    has_mediator = any(token in text for token in ("mediat", "中介", "through", "via", "路径"))
    has_independent = any(token in text for token in ("independent", "独立增量", "独立预测"))
    has_condition = any(
        token in text
        for token in (
            "condition",
            "conditioned",
            "branch",
            "未建模",
            "约束",
            "交互",
            "interaction",
            "条件路径",
            "条件分支",
            "阶段变化",
        )
    )
    has_residual = any(token in text for token in ("residual", "残差", "失败", "failure", "异常", "归因"))
    is_partial = record.resolution_status == "partially_resolved"

    if is_partial:
        return "validation_followup"
    if _mentions_temporal_stability(text):
        return "temporal_stability_probe"
    if has_lag:
        return "lag_stability"
    if has_residual:
        return "residual_probe"

    mining_focus_present = bool(_notes_mining_features(record))
    global_independent = (
        has_independent
        and not mining_focus_present
        and not has_condition
        and not any(token in text for token in ("分歧", "独立于", "_via_", "hypothesisconflict"))
    )
    if global_independent:
        return "global_incremental"
    if has_condition:
        return "conditional_path"
    if has_independent or has_mediator:
        return "incremental_beyond_focus"
    if "null_competition" in text or "competition" in text or "competitive" in text:
        return "competition_probe"
    if not has_non_primary_focus:
        return "global_incremental"
    return "incremental_beyond_focus"


def _lag_stability_values(record: UncertaintyRecord) -> list[int]:
    """Keep time-stability and prediction-horizon questions on distinct axes."""
    text = f"{record.question} {record.description}".lower()
    if any(token in text for token in ("horizon", "lead time", "超前", "提前量")):
        return [1, 3, 5]
    if any(token in text for token in ("lag", "滞后", "时滞")):
        return [1, 2, 3]
    return []


_FROZEN_MAG_BOTH_LAGS: dict[str, list[int]] = {
    "BY, nT (GSE)": [4, 5, 7, 8],
    "BZ, nT (GSE)": [1, 2],
    "Scalar B, nT": [1, 2],
    "p2_y中心值": [1],
    "p1_x中心值": [9],
}


def _build_design_for_strategy(
    *,
    strategy: str,
    focus: str,
    primary_x: str,
    target: str,
    feature_pool: list[str],
    uncertainty_base_features: list[str] | None = None,
    lag_values: list[int] | None = None,
) -> tuple[ExperimentDesign, str]:
    """Build a 对照组/实验组 design that isolates ``primary_x``.

    对照组 always exclude the variable under test (``primary_x``);
    实验组 always add it. ``focus`` only participates in selecting
    which other features constrain the 对照组, never as the tested variable.
    """
    control: list[str]
    treatment: list[str]
    lags: dict[str, list[int]] = {}

    by_feature = _resolve_by_feature(feature_pool)
    baseline_features = list(uncertainty_base_features or [])
    if not baseline_features:
        baseline_features = [
            feature for feature in [by_feature, *feature_pool[:2]]
            if feature not in {primary_x, target}
        ]
    else:
        baseline_features = [
            feature for feature in baseline_features
            if feature not in {primary_x, target}
        ]
    if strategy == "temporal_stability_probe":
        magnetic_base = [
            feature
            for feature in ("BY, nT (GSE)", "BZ, nT (GSE)", "Scalar B, nT")
            if feature in {*baseline_features, *feature_pool}
            and feature not in {primary_x, target}
        ]
        if magnetic_base:
            baseline_features = magnetic_base
        else:
            baseline_features = [
                feature
                for feature in baseline_features
                if feature not in {primary_x, target, "p2_y中心值", "p1_x中心值"}
            ]
    control = baseline_features
    if strategy == "temporal_stability_probe":
        shadow_axes = _unique_preserve_order(
            feature
            for feature in [primary_x, *feature_pool]
            if feature in {"p2_y中心值", "p1_x中心值"}
        )
        treatment = _unique_preserve_order([*baseline_features, *shadow_axes])
        if not shadow_axes:
            treatment = _unique_preserve_order([*baseline_features, primary_x])
        lags = {
            feature: list(_FROZEN_MAG_BOTH_LAGS[feature])
            for feature in treatment
            if feature in _FROZEN_MAG_BOTH_LAGS
        }
    else:
        treatment = _unique_preserve_order([*baseline_features, primary_x])
        lags = (
            {primary_x: lag_values or [1, 2, 3]}
            if strategy == "lag_stability"
            else {}
        )

    return ExperimentDesign(
        target=target,
        control=_normalize_design_features(control),
        treatment=_normalize_design_features(treatment),
        lags=lags,
        probe_axis=(
            "cross_window_stability"
            if strategy == "temporal_stability_probe"
            else None
        ),
        design_focus=primary_x,
    ), primary_x


def _resolve_by_feature(
    feature_pool: list[str],
    mediator_candidates: list[str] | None = None,
) -> str | None:
    """Return the raw feature that represents the IMF By path variable."""
    pool = _unique_preserve_order([*(mediator_candidates or []), *feature_pool])
    return next(
        (
            feature
            for feature in pool
            if "by" in _slug(feature)
            or "行星际磁场" in feature
            or "BY" in feature
            or "By" in feature
        ),
        None,
    )


def _design_from_uncertainty(
    record: UncertaintyRecord,
    target: str,
    primary_x: str,
    control_base: list[str],
    planning_context: PlanningContext,
    node_index: dict[str, object],
    semantic: VariableSemanticService | None = None,
    feature_pool: list[str] | None = None,
    used_signatures: set[
        tuple[
            tuple[str, ...],
            tuple[str, ...],
            tuple[tuple[str, tuple[int, ...]], ...],
            tuple[int | None, int | None, int | None, int | None, int | None, int | None, int | None],
        ]
    ]
    | None = None,
) -> ExperimentDesign:
    semantic = semantic or VariableSemanticService()
    feature_pool = _unique_preserve_order(feature_pool or [primary_x])
    used_signatures = used_signatures or set()

    is_partial = record.resolution_status == "partially_resolved"

    uncertainty_own_focus_candidates = _candidate_focus_features(
        record=record,
        node_index=node_index,
        primary_x=primary_x,
        target=target,
        semantic=semantic,
        feature_pool=feature_pool,
    )
    preferred = _preferred_planning_features(planning_context, semantic, feature_pool)
    if preferred:
        focus_candidates = _unique_preserve_order(
            [*preferred, *uncertainty_own_focus_candidates]
        )
    else:
        focus_candidates = uncertainty_own_focus_candidates
    focus_order = _unique_preserve_order([*focus_candidates, primary_x, *feature_pool])

    if planning_context.prefer_simple_design and preferred:
        focus = primary_x
        design = ExperimentDesign(
            target=target,
            control=_normalize_design_features(
                [item for item in control_base if item != focus]
            ),
            treatment=_unique_preserve_order([*control_base, primary_x]),
            notes=[
                f"generated_from:{record.uncertainty_id}",
                f"design_focus:{focus}",
                "probe_single_feature_path",
                f"human_feedback_focus_single_feature:{preferred[0]}",
            ],
            design_focus=focus,
        )
        if is_partial:
            design.notes.append("validation_followup_for_partially_resolved_disagreement")
        return design

    text = f"{record.notes or ''} {record.question} {record.description}".lower()
    has_non_primary_focus = any(feature != primary_x for feature in focus_candidates)
    strategy = _uncertainty_design_strategy(record, text, has_non_primary_focus)
    fallback_design: ExperimentDesign | None = None
    uncertainty_base_feature = next(
        (feature for feature in uncertainty_own_focus_candidates if feature != primary_x),
        None,
    )

    lag_values = _lag_stability_values(record) if strategy == "lag_stability" else None
    for focus in focus_order:
        design, effective_focus = _build_design_for_strategy(
            strategy=strategy,
            focus=focus,
            primary_x=primary_x,
            target=target,
            feature_pool=feature_pool,
            uncertainty_base_features=(
                [uncertainty_base_feature] if uncertainty_base_feature else []
            ),
            lag_values=lag_values,
        )
        strategy_notes = {
            "global_incremental": "probe_global_incremental_gain:primary variable added to compact baseline",
            "incremental_beyond_focus": f"probe_incremental_gain_beyond_focus:{effective_focus}",
            "competition_probe": f"probe_primary_gain_over_competitor:{effective_focus}",
            "conditional_path": f"probe_conditioned_contribution_with_focus:{effective_focus}",
            "lag_stability": f"probe_lagged_stability_across_windows:{effective_focus}",
            "temporal_stability_probe": f"probe_cross_window_stability:{effective_focus}",
            "residual_probe": f"probe_residual_source_with_focus:{effective_focus}",
            "validation_followup": f"validation_followup_rerun_focus:{effective_focus}",
        }
        notes = [
            f"generated_from:{record.uncertainty_id}",
            f"design_focus:{effective_focus}",
            strategy_notes[strategy],
        ]
        if planning_context.preferred_features and preferred:
            notes.append(f"human_feedback_focus_feature:{preferred[0]}")
        notes.append(
            "compare constrained baseline vs baseline+primary treatment"
        )
        if uncertainty_base_feature:
            notes.append(
                f"uncertainty_baseline_feature:{uncertainty_base_feature}"
            )

        if is_partial:
            notes.append("validation_followup_for_partially_resolved_disagreement")

        rationale_context = _candidate_rationale_context(record, node_index)
        if rationale_context["focus_features"]:
            notes.append(f"rationale_focus_features:{','.join(rationale_context['focus_features'])}")
        if rationale_context["source_types"]:
            notes.append(f"rationale_source_types:{','.join(rationale_context['source_types'])}")

        design.notes = notes
        if fallback_design is None:
            fallback_design = design
        signature = _design_signature(
            design,
            experiment_type=_candidate_experiment_type(record),
        )
        if signature not in used_signatures:
            return design

    if fallback_design is not None:
        fallback_design.notes.append("design_focus_fallback_duplicate_safeguard")
    return fallback_design or ExperimentDesign(
        target=target,
        control=[feature_pool[0]] if feature_pool else [],
        treatment=_unique_preserve_order(
            [feature_pool[0], primary_x] if feature_pool else [primary_x]
        ),
        notes=[
            f"generated_from:{record.uncertainty_id}",
            "design_focus:primary_x",
            "design_focus_fallback_duplicate_safeguard",
        ],
        design_focus=primary_x,
    )


def _run_spec_signature(spec: RunSpec) -> tuple[str, str, tuple[str, ...], tuple[tuple[str, object], ...]]:
    params = tuple(sorted(spec.parameters.items()))
    return (
        spec.run_id,
        normalize_model_family(spec.model_family),
        tuple(spec.feature_columns),
        params,
    )


def _contrast_spec_signature(spec: ContrastSpec) -> tuple[str, str, str, tuple[str, ...], tuple[str, ...]]:
    return (
        spec.contrast_id,
        spec.kind,
        spec.direction,
        tuple(spec.run_ids),
        tuple(spec.contrast_ids),
    )


def _design_signature(
    design: ExperimentDesign,
    *,
    experiment_type: str = "incremental_prediction",
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[tuple[str, tuple[int, ...]], ...],
    tuple[int | None, int | None, int | None, int | None, int | None, int | None, int | None],
    str,
    tuple[str, ...],
    tuple[tuple[str, str, tuple[str, ...], tuple[tuple[str, object], ...]], ...],
    tuple[tuple[str, str, str, tuple[str, ...], tuple[str, ...]], ...],
]:
    return (
        tuple(design.control),
        tuple(design.treatment),
        tuple(sorted((name, tuple(values)) for name, values in design.lags.items())),
        (
            design.forecast_horizon_days,
            design.past_lag_days,
            design.window_size,
            design.control_lag_days,
            design.treatment_lag_days,
            design.control_forecast_horizon_days,
            design.treatment_forecast_horizon_days,
        ),
        experiment_type,
        design.probe_axis,
        tuple(sorted({normalize_model_family(name) for name in design.model_families})),
        tuple(sorted(_run_spec_signature(spec) for spec in design.runs_spec)),
        tuple(sorted(_contrast_spec_signature(spec) for spec in design.contrasts_spec)),
    )


def _preferred_planning_features(
    planning_context: PlanningContext,
    semantic: VariableSemanticService,
    feature_pool: list[str],
) -> list[str]:
    resolved: list[str] = []
    for name in planning_context.preferred_features:
        raw = semantic.to_raw(name)
        if raw in feature_pool:
            resolved.append(raw)
    return _unique_preserve_order(resolved)


def _candidate_focus_features(
    *,
    record: UncertaintyRecord,
    node_index: dict[str, object],
    primary_x: str,
    target: str,
    semantic: VariableSemanticService,
    feature_pool: list[str],
) -> list[str]:
    """Rank raw features this uncertainty should isolate, leading hypothesis first."""
    excluded = {target}
    available = [feature for feature in feature_pool if feature not in excluded]
    weighted: list[tuple[int, str]] = []

    # Uncertainty miners often tag explicit features; those are the strongest signal.
    for feature in _notes_mining_features(record):
        raw = semantic.to_raw(feature)
        if raw and raw in available:
            weighted.append((0, raw))

    text = f"{record.question} {record.description}"
    for feature in available:
        if semantic.matches_text(feature, text):
            weighted.append((1, feature))

    for rank, group in enumerate(
        (record.related_hypotheses[:1], record.related_hypotheses[1:]), start=2
    ):
        for hypothesis_id in group:
            node = node_index.get(hypothesis_id)
            if node is None or node.generation_rationale is None:
                continue
            for feature in node.generation_rationale.derived_features:
                raw = semantic.to_raw(feature)
                if raw and raw in available:
                    weighted.append((rank, raw))

    ordered = _unique_preserve_order(
        raw for _, raw in sorted(weighted, key=lambda item: (item[0], item[1]))
    )
    return _unique_preserve_order([primary_x, *ordered]) if primary_x in available else _unique_preserve_order(ordered)


def _hypothesis_relates_to_focus(
    node: object,
    focus: str | None,
    semantic: VariableSemanticService,
) -> bool:
    """True when this hypothesis is a primary target for the design focus."""
    if not focus:
        return True
    statement = getattr(node, "statement", "") or ""
    rationale = getattr(node, "generation_rationale", None)
    if rationale is not None:
        for feature in rationale.derived_features:
            if semantic.to_raw(feature) == focus:
                return True
        trigger = rationale.trigger or ""
        if trigger in {"task_bootstrap", "competition_bootstrap"}:
            return True
        corpus = f"{statement} {rationale.summary or ''}"
        if semantic.matches_text(focus, corpus):
            return True
    return semantic.matches_text(focus, statement)


def _focus_hypothesis_ids(
    *,
    record: UncertaintyRecord,
    focus: str | None,
    node_index: dict[str, object],
    semantic: VariableSemanticService,
) -> list[str]:
    """Select the competing hypotheses this candidate should actually resolve."""
    selected: list[str] = []
    for hypothesis_id in record.related_hypotheses:
        node = node_index.get(hypothesis_id)
        if node is None:
            continue
        if _hypothesis_relates_to_focus(node, focus, semantic):
            selected.append(hypothesis_id)
    return _unique_preserve_order(selected) or [
        hypothesis_id
        for hypothesis_id in record.related_hypotheses
        if hypothesis_id in node_index
    ]


def _tree_active_hypothesis_ids(
    tree: HypothesisTreeState,
    node_index: dict[str, object],
) -> list[str]:
    return _unique_preserve_order(
        hypothesis_id
        for hypothesis_id in tree.active_hypotheses
        if hypothesis_id in node_index
    )


def _build_hypothesis_predictions(
    record: UncertaintyRecord,
    node_index: dict[str, object],
    design_focus: str | None = None,
    semantic: VariableSemanticService | None = None,
) -> dict[str, HypothesisPrediction]:
    semantic = semantic or VariableSemanticService()
    predictions: dict[str, HypothesisPrediction] = {}
    selected_ids = _focus_hypothesis_ids(
        record=record,
        focus=design_focus,
        node_index=node_index,
        semantic=semantic,
    )
    nodes = [node_index.get(hypothesis_id) for hypothesis_id in selected_ids]
    nodes = [node for node in nodes if node is not None]
    llm_lookup = _select_llm_predictions(nodes)
    for node in nodes:
        disagreement = (
            record.disagreement.get(node.hypothesis_id)
            if record.disagreement is not None
            else None
        )
        llm_prediction = llm_lookup.get(node.hypothesis_id)
        expected_range = (
            list(llm_prediction.expected_range)
            if llm_prediction is not None and llm_prediction.expected_range
            else None
        )
        expected_effect = _expected_effect_for_candidate(
            disagreement_effect=getattr(disagreement, "expected", None),
            node=node,
            llm_prediction=llm_prediction,
            expected_range=expected_range,
        )
        if expected_range is None:
            expected_range = _expected_range_from_effect(
                expected_effect=expected_effect,
                support_score=getattr(node, "support_score", 0.5),
            )
        metric = (
            getattr(llm_prediction, "metric", None)
            if llm_prediction is not None and expected_range is not None
            else None
        )
        predictions[node.hypothesis_id] = HypothesisPrediction(
            expected_effect=expected_effect,
            expected_range=expected_range,
            metric=metric,
        )
    return predictions


def _select_llm_predictions(nodes: list[object]) -> dict[str, object]:
    """Choose one LLM prediction per node, preferring a shared metric.

    The overlap estimate is only meaningful when the hypothesis distributions
    are measured on the same metric, so we first pick the most frequent LLM
    metric with a real expected_range and then re-select that metric when the
    node provides it.
    """
    first_pass: dict[str, object] = {}
    for node in nodes:
        hypothesis_id = getattr(node, "hypothesis_id", "")
        first_pass[hypothesis_id] = _preferred_prediction(node)
    metric_counts: dict[str, int] = {}
    for prediction in first_pass.values():
        metric = getattr(prediction, "metric", None)
        if (
            prediction is not None
            and getattr(prediction, "expected_range", None)
            and metric in IG_METRIC_PRIORITY
        ):
            metric_counts[metric] = metric_counts.get(metric, 0) + 1
    preferred = None
    if metric_counts:
        preferred = max(
            metric_counts,
            key=lambda metric: (metric_counts[metric], IG_METRIC_PRIORITY.index(metric)),
        )
    selected: dict[str, object] = {}
    for node in nodes:
        hypothesis_id = getattr(node, "hypothesis_id", "")
        if preferred is not None:
            selected[hypothesis_id] = _preferred_prediction(node, preferred_metric=preferred)
        else:
            selected[hypothesis_id] = first_pass.get(hypothesis_id)
    return selected


def _preferred_prediction(node: object, preferred_metric: str | None = None) -> object | None:
    node_predictions = getattr(node, "predictions", None) or []
    if preferred_metric is not None:
        for prediction in node_predictions:
            if (
                getattr(prediction, "metric", None) == preferred_metric
                and getattr(prediction, "expected_range", None)
            ):
                return prediction
    for metric in IG_METRIC_PRIORITY:
        for prediction in node_predictions:
            if (
                getattr(prediction, "metric", None) == metric
                and getattr(prediction, "expected_range", None)
            ):
                return prediction
    return next(
        (
            prediction
            for prediction in node_predictions
            if getattr(prediction, "expected_range", None)
        ),
        None,
    )


def _expected_effect_for_candidate(
    *,
    disagreement_effect: str | None,
    node: object,
    llm_prediction: object | None,
    expected_range: list[float] | None,
) -> str:
    if disagreement_effect:
        return disagreement_effect
    if llm_prediction is not None:
        direction = str(getattr(llm_prediction, "expected_direction", "") or "").lower()
        if direction in {"positive", "negative", "near_zero"}:
            return direction
    if expected_range and len(expected_range) == 2:
        center = (expected_range[0] + expected_range[1]) / 2.0
        if center > 0.01:
            return "positive"
        if center < -0.01:
            return "negative"
        return "near_zero"
    return _expected_from_node(node)


def _annotate_candidate_estimates(
    *,
    candidate: CandidateExperiment,
    task: ScientificTask,
    data_dictionary: DataDictionary,
    hypothesis_tree: HypothesisTreeState,
    experiment_memory: ExperimentMemoryState | None,
    current_round: int,
) -> None:
    ig, ig_rationale = _formal_information_gain(candidate, current_round=current_round)
    pg, pg_rationale = _formal_expected_performance_gain(
        candidate=candidate,
        experiment_memory=experiment_memory,
        current_round=current_round,
    )
    risk, risk_rationale = _formal_risk_score(
        candidate=candidate,
        task=task,
        data_dictionary=data_dictionary,
        hypothesis_tree=hypothesis_tree,
        experiment_memory=experiment_memory,
    )
    cost, cost_rationale = _formal_cost_score(
        candidate=candidate,
        task=task,
        data_dictionary=data_dictionary,
        hypothesis_tree=hypothesis_tree,
        experiment_memory=experiment_memory,
    )
    candidate.estimated_information_gain = EstimatedValue(value=ig, rationale=ig_rationale)
    candidate.estimated_performance_gain = EstimatedValue(value=pg, rationale=pg_rationale)
    candidate.estimated_risk = EstimatedValue(value=risk, rationale=risk_rationale)
    candidate.estimated_cost = EstimatedValue(value=cost, rationale=cost_rationale)


def _expected_range_from_effect(*, expected_effect: str, support_score: float) -> list[float]:
    confidence = min(max(support_score, 0.05), 0.95)
    width = max(0.015, 0.08 - confidence * 0.04)
    if expected_effect == "negative":
        center = -max(0.02, 0.03 + confidence * 0.05)
        return [round(center - width, 4), round(center + width, 4)]
    if expected_effect == "near_zero":
        return [round(-width, 4), round(width, 4)]
    center = max(0.02, 0.03 + confidence * 0.05)
    return [round(center - width, 4), round(center + width, 4)]


def _prediction_metric_domain(metric: str | None) -> str | None:
    """Map prediction metrics to semantic domains so units are never compared loosely."""

    if not metric:
        return None
    normalized = metric.strip().lower()
    aliases = {
        "pearson_r": "pearson_r_absolute",
        "pearson r": "pearson_r_absolute",
        "pearson_r_absolute": "pearson_r_absolute",
        "r_abs": "pearson_r_absolute",
        "pearson_r_delta": "pearson_r_delta",
        "delta_pearson_r": "pearson_r_delta",
        "delta_r": "pearson_r_delta",
        "stability_rate": "stability_rate",
        "stable_window_rate": "stability_rate",
        "rmse_delta": "rmse_delta",
        "delta_rmse": "rmse_delta",
    }
    return aliases.get(normalized, f"exact:{normalized}")


def _formal_information_gain(
    candidate: CandidateExperiment,
    *,
    current_round: int,
) -> tuple[float, str]:
    predictions = list(candidate.hypothesis_predictions.values())
    if len(predictions) < 2:
        return (
            0.5,
            "候选实验当前关联的可区分假设不足 2 个，无法按 IG_pair=1-overlap 计算，"
            "按文档约定退化为中性信息增益 0.5；后续关联到至少 2 个假设后，"
            "将改用预测区间重叠度重新计算，不再使用固定低值模板。",
        )

    pair_scores: list[float] = []
    incomparable_pairs = 0
    focus_label = candidate.design.design_focus
    for index, left in enumerate(predictions):
        for right in predictions[index + 1 :]:
            overlap = _prediction_overlap(
                left.expected_range,
                right.expected_range,
                left_metric=left.metric,
                right_metric=right.metric,
            )
            if overlap is None:
                incomparable_pairs += 1
                continue
            score = 1.0 - overlap
            pair_scores.append(score)
    if pair_scores:
        ig = sum(pair_scores) / len(pair_scores)
    elif incomparable_pairs:
        return (
            0.5,
            "候选实验关联假设的预测区间使用不同指标或缺失指标，统计量/语义不一致，"
            "无法直接计算区分度；"
            f"{incomparable_pairs} 组区间因统计量/语义不一致被标记为不可比，按中性 0.5 处理。"
            "请让关联假设在同一指标上给出预测区间后再做严格比较。",
        )
    else:
        return (
            0.5,
            "候选实验预测区间缺失，无法计算区分度，按中性 0.5 处理。",
        )
    focus_text = f"围绕设计焦点 {focus_label}，候选实验针对性区分 " if focus_label else "候选实验共区分 "
    used_metrics = sorted(
        {
            str(prediction.metric)
            for prediction in predictions
            if getattr(prediction, "metric", None)
        }
    )
    metric_text = (
        f"，预测区间来自 LLM 输出（metric={'/'.join(used_metrics)}）"
        if used_metrics
        else "，预测区间为程序回退估计"
    )
    rationale = (
        "按可比的预测区间对计算 IG_pair=1-overlap、IG(E)=avg(IG_pair)；"
        f"{focus_text}{len(predictions)} 个相关竞争假设、{len(pair_scores)} 组可比预测区间"
        f"（{incomparable_pairs} 组因统计量/语义不一致不可比），"
        f"平均区分度为 {ig:.4f}{metric_text}。"
    )
    if current_round > 1:
        rationale += (
            " 本评分是实验前的预期区分度启发式评分，不是严格贝叶斯 EIG；"
            "实验完成后系统以实际观测 Δ 计算后验 KL 散度审计并写入评估结果。"
        )
    return round(min(max(ig, 0.0), 1.0), 4), rationale


def _prediction_overlap(
    left_range: list[float] | None,
    right_range: list[float] | None,
    *,
    left_metric: str | None = None,
    right_metric: str | None = None,
) -> float | None:
    left_domain = _prediction_metric_domain(left_metric)
    right_domain = _prediction_metric_domain(right_metric)
    # Two missing metrics come from the same program fallback heuristic, so their
    # ranges remain mutually comparable. A known metric mixed with an unknown one
    # is rejected because the units are not auditable.
    if left_domain is None and right_domain is None:
        pass
    elif left_domain is None or right_domain is None or left_domain != right_domain:
        return None
    if not left_range or not right_range:
        return 0.5
    left_mu = (left_range[0] + left_range[1]) / 2
    right_mu = (right_range[0] + right_range[1]) / 2
    left_sigma = max(abs(left_range[1] - left_range[0]) / 4, EPSILON)
    right_sigma = max(abs(right_range[1] - right_range[0]) / 4, EPSILON)
    lower = min(left_mu - 4 * left_sigma, right_mu - 4 * right_sigma)
    upper = max(left_mu + 4 * left_sigma, right_mu + 4 * right_sigma)
    steps = 256
    dx = (upper - lower) / steps
    overlap = 0.0
    for step in range(steps):
        x = lower + (step + 0.5) * dx
        overlap += min(_normal_pdf(x, left_mu, left_sigma), _normal_pdf(x, right_mu, right_sigma)) * dx
    return min(max(overlap, 0.0), 1.0)


def _normal_pdf(x: float, mean: float, sigma: float) -> float:
    z = (x - mean) / max(sigma, EPSILON)
    return math.exp(-0.5 * z * z) / (max(sigma, EPSILON) * math.sqrt(2 * math.pi))


def _formal_expected_performance_gain(
    *,
    candidate: CandidateExperiment,
    experiment_memory: ExperimentMemoryState | None,
    current_round: int,
) -> tuple[float, str]:
    current_best_rmse = _current_best_rmse(experiment_memory, candidate=candidate)
    if current_round <= 1 or current_best_rmse is None:
        return (
            0.0,
            "第一轮、尚无历史实验结果、或没有同类型可比的对照组/实验组 RMSE 时，"
            "按规则将 PG_expected 置为 0，不参与综合价值计算。",
        )
    predicted_after_rmse, similarity, historical_gain = _predict_after_rmse_from_history(
        candidate=candidate,
        current_best_rmse=current_best_rmse,
        experiment_memory=experiment_memory,
    )
    pg_expected = (current_best_rmse - predicted_after_rmse) / max(current_best_rmse, EPSILON)
    rationale = (
        "按 PG_expected(E)=(RMSE_current_best-RMSE_predicted_after(E))/RMSE_current_best "
        "仅使用同类型、同对比结构、同时间定义的历史结果计算；"
        f"当前最优 RMSE={current_best_rmse:.4f}，历史相似度={similarity:.4f}，"
        f"历史平均实际增益={historical_gain:.4f}，预测实验后 RMSE={predicted_after_rmse:.4f}。"
    )
    return round(min(max(pg_expected, -1.0), 1.0), 4), rationale


def _predict_after_rmse_from_history(
    *,
    candidate: CandidateExperiment,
    current_best_rmse: float,
    experiment_memory: ExperimentMemoryState | None,
) -> tuple[float, float, float]:
    if experiment_memory is None:
        return current_best_rmse, 0.0, 0.0

    weighted_gain = 0.0
    similarity_total = 0.0
    fallback_gains: list[float] = []
    for entry in experiment_memory.entries:
        metrics = entry.metrics_snapshot
        if not _same_experiment_structure(candidate, entry):
            continue
        gain = _entry_actual_pg(metrics, experiment_type=entry.experiment_type)
        if gain is None:
            continue
        fallback_gains.append(gain)
        similarity = _historical_similarity(candidate, entry)
        if similarity <= 0.0:
            continue
        weighted_gain += similarity * gain
        similarity_total += similarity

    historical_gain = weighted_gain / similarity_total if similarity_total > 0 else (
        sum(fallback_gains) / len(fallback_gains) if fallback_gains else 0.0
    )
    lag_penalty = 0.01 * len(candidate.design.lags)
    predicted_gain = min(historical_gain - lag_penalty, 0.35)
    predicted_after_rmse = max(current_best_rmse * (1.0 - predicted_gain), EPSILON)
    normalized_similarity = min(similarity_total / max(len(fallback_gains), 1), 1.0) if fallback_gains else 0.0
    return predicted_after_rmse, normalized_similarity, historical_gain


def _entry_actual_pg(metrics, *, experiment_type: str | None = None) -> float | None:
    if metrics is None:
        return None
    if metrics.pg_actual_signed is not None:
        return metrics.pg_actual_signed
    if experiment_type in {"model_sensitivity", "baseline_benchmark", "robustness_diagnostic"}:
        return None
    if metrics.baseline_rmse is None or metrics.treatment_rmse is None:
        return None
    return (metrics.baseline_rmse - metrics.treatment_rmse) / max(metrics.baseline_rmse, EPSILON)


def _structure_signature(entry) -> tuple[str, tuple[str, ...]]:
    entry_type = str(entry.experiment_type or "incremental_prediction").strip()
    families = tuple(
        sorted(normalize_model_family(str(name)) for name in (entry.model_families or []))
    )
    return entry_type, families


def _entry_horizon_days(entry) -> int | None:
    metrics = entry.metrics_snapshot
    if metrics is None:
        return None
    horizons = []
    for run in metrics.runs:
        params = run.parameters or {}
        for key in ("forecast_horizon_days", "forecast_horizon", "horizon_days"):
            raw = params.get(key)
            try:
                value = int(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                horizons.append(value)
    return max(horizons) if horizons else None


def _same_experiment_structure(candidate: CandidateExperiment, entry) -> bool:
    candidate_type = str(candidate.experiment_type or candidate.type or "incremental_prediction").strip()
    candidate_families = tuple(
        sorted(normalize_model_family(str(name)) for name in candidate.design.model_families)
    )
    entry_type, entry_families = _structure_signature(entry)
    if entry_type != candidate_type:
        return False
    if candidate_type == "model_sensitivity" and candidate_families and entry_families:
        if set(candidate_families) != set(entry_families):
            return False
    candidate_horizons = {
        candidate.design.control_forecast_horizon_days,
        candidate.design.treatment_forecast_horizon_days,
        candidate.design.forecast_horizon_days,
    }
    non_null_horizons = {value for value in candidate_horizons if value is not None}
    if non_null_horizons:
        candidate_horizon = min(non_null_horizons)
        entry_horizon = _entry_horizon_days(entry)
        if entry_horizon is not None and int(entry_horizon) != int(candidate_horizon):
            return False
    return True


def _historical_similarity(candidate: CandidateExperiment, entry) -> float:
    hypothesis_overlap = _jaccard_similarity(set(candidate.tested_hypotheses), set(entry.tested_hypotheses))
    finding_overlap = _token_overlap(candidate.purpose, " ".join(entry.key_findings))
    status_bonus = 0.1 if entry.status == "completed" else 0.0
    structure = _structure_signature(entry)
    candidate_type = str(candidate.experiment_type or candidate.type or "incremental_prediction").strip()
    family_similarity = 1.0
    entry_families = set(structure[1])
    candidate_families = {
        normalize_model_family(str(name)) for name in candidate.design.model_families
    }
    if candidate_families or entry_families:
        family_similarity = (
            len(candidate_families & entry_families) / max(len(candidate_families | entry_families), 1)
            if (candidate_families or entry_families)
            else 1.0
        )
    type_similarity = 0.2 if structure[0] == candidate_type else 0.0
    return min(
        0.45 * hypothesis_overlap
        + 0.15 * finding_overlap
        + 0.10 * family_similarity
        + type_similarity
        + status_bonus,
        1.0,
    )


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in left.lower().replace("、", " ").replace(",", " ").split() if token}
    right_tokens = {token for token in right.lower().replace("、", " ").replace(",", " ").split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


def _formal_risk_score(
    *,
    candidate: CandidateExperiment,
    task: ScientificTask,
    data_dictionary: DataDictionary,
    hypothesis_tree: HypothesisTreeState,
    experiment_memory: ExperimentMemoryState | None,
) -> tuple[float, str]:
    node_index = hypothesis_tree.node_index()
    supports = [
        node_index[hypothesis_id].support_score
        for hypothesis_id in candidate.tested_hypotheses
        if hypothesis_id in node_index
    ]
    avg_support = sum(supports) / len(supports) if supports else 0.5
    support_span = (max(supports) - min(supports)) if len(supports) >= 2 else 0.0
    d1 = min(max(1.0 - avg_support + (0.1 if support_span > 0.30 else 0.0), 0.0), 1.0)

    sample_risk = 1.0 - min(data_dictionary.total_samples / 200.0, 1.0)
    multi_source_risk = 0.5 if len(task.payload.data_sources) > 1 else 0.0
    leakage_risk = 0.5 if not task.payload.constraints.no_future_information else 0.0
    d2 = min(0.4 * sample_risk + 0.3 * multi_source_risk + 0.3 * leakage_risk, 1.0)

    control_risk = 0.6 if not candidate.design.control else (0.3 if len(candidate.design.control) < 2 else 0.0)
    reproducibility_risk = 0.2 if _has_similar_history(candidate, experiment_memory) else 0.0
    design_complexity_risk = min(0.2 * len(candidate.design.lags) + 0.1 * max(len(candidate.design.treatment) - 3, 0), 0.7)
    d3 = min(0.4 * design_complexity_risk + 0.3 * control_risk + 0.3 * reproducibility_risk, 1.0)

    budget = task.payload.constraints.resource_budget
    token_budget = budget.token_budget if budget and budget.token_budget else 200000
    time_budget = budget.max_time_seconds_per_round if budget and budget.max_time_seconds_per_round else 300
    estimated_tokens = _estimate_llm_tokens(candidate, hypothesis_tree, experiment_memory)
    compute_seconds = _estimate_compute_seconds(candidate, data_dictionary)
    compute_risk = 1.0 if estimated_tokens > token_budget * 0.8 else (0.5 if estimated_tokens > token_budget * 0.5 else 0.0)
    complexity_risk = min(compute_seconds / max(time_budget, 1), 1.0)
    human_risk = 0.5 if candidate.requires_human_review else 0.0
    d4 = min(0.4 * compute_risk + 0.35 * complexity_risk + 0.25 * human_risk, 1.0)

    risk = min(0.35 * d1 + 0.25 * d2 + 0.25 * d3 + 0.15 * d4, 1.0)
    dimensions = {
        "支持度先验": d1,
        "数据质量/对齐": d2,
        "实验设计": d3,
        "资源执行": d4,
    }
    dominant_dimension = max(dimensions, key=dimensions.get)
    rationale = (
        "按 Risk(E)=0.35·D1+0.25·D2+0.25·D3+0.15·D4 计算；"
        f"D1={d1:.4f}（支持度先验：{len(supports)} 条假设平均支持度 {avg_support:.4f}、"
        f"支持度跨度 {support_span:.4f}），"
        f"D2={d2:.4f}（数据质量/对齐：样本量风险 {sample_risk:.4f}、"
        f"多源对齐风险 {multi_source_risk:.2f}、未来信息泄露风险 {leakage_risk:.2f}），"
        f"D3={d3:.4f}（实验设计：设计复杂度 {design_complexity_risk:.4f}、"
        f"对照充分性 {control_risk:.2f}、历史可重复性 {reproducibility_risk:.2f}），"
        f"D4={d4:.4f}（资源执行：估算 tokens {estimated_tokens:.0f}/{token_budget}、"
        f"估算耗时 {compute_seconds:.1f}s/{time_budget}s、人工审核 {human_risk:.2f}）；"
        f"当前最高风险分项为“{dominant_dimension}”。"
    )
    return round(risk, 4), rationale


def _formal_cost_score(
    *,
    candidate: CandidateExperiment,
    task: ScientificTask,
    data_dictionary: DataDictionary,
    hypothesis_tree: HypothesisTreeState,
    experiment_memory: ExperimentMemoryState | None,
) -> tuple[float, str]:
    budget = task.payload.constraints.resource_budget
    token_budget = budget.token_budget if budget and budget.token_budget else 200000
    time_budget = budget.max_time_seconds_per_round if budget and budget.max_time_seconds_per_round else 300
    input_tokens, output_tokens = _estimate_llm_token_usage(
        candidate=candidate,
        hypothesis_tree=hypothesis_tree,
        experiment_memory=experiment_memory,
    )
    total_tokens = input_tokens + output_tokens
    model_name = get_llm_model_for_role("experiment_planner")
    price_rates = _llm_price_rates()
    input_price = price_rates.get("input", DEFAULT_LLM_INPUT_PRICE)
    output_price = price_rates.get("output", DEFAULT_LLM_OUTPUT_PRICE)
    llm_fee_yuan = (input_tokens * input_price + output_tokens * output_price) / 1_000_000.0
    budget_cost_yuan = _llm_cost_budget_yuan(token_budget, price_rates)
    c1 = min(llm_fee_yuan / max(budget_cost_yuan, EPSILON), 1.0)
    c2_raw = _estimate_compute_seconds(candidate, data_dictionary)
    c2 = min(c2_raw / max(time_budget, 1), 1.0)
    c3 = 1.0 if candidate.requires_human_review else 0.0
    cost = max(0.50 * c1 + 0.35 * c2 + 0.15 * c3, 0.05)
    rationale = (
        "按 Cost(E)=max(0.50·C1+0.35·C2+0.15·C3, 0.05) 计算；"
        f"估算输入 tokens={input_tokens:.0f}、输出 tokens={output_tokens:.0f}（合计 {total_tokens:.0f}）；"
        f"{model_name} 真实计费（输入 {input_price:.2f} 元/百万 tokens、"
        f"输出 {output_price:.2f} 元/百万 tokens）估算约 {llm_fee_yuan:.4f} 元，"
        f"本轮费用上限={budget_cost_yuan:.4f} 元；"
        f"C1={c1:.4f}(LLM真实费用占比), C2={c2:.4f}(计算时间占比 {c2_raw:.1f}s/{time_budget}s), "
        f"C3={c3:.4f}(人工审核成本)。"
    )
    return round(min(cost, 1.0), 4), rationale


def _estimate_llm_tokens(
    candidate: CandidateExperiment,
    hypothesis_tree: HypothesisTreeState,
    experiment_memory: ExperimentMemoryState | None,
) -> float:
    input_tokens, output_tokens = _estimate_llm_token_usage(
        candidate=candidate,
        hypothesis_tree=hypothesis_tree,
        experiment_memory=experiment_memory,
    )
    return input_tokens + output_tokens


def _estimate_llm_token_usage(
    candidate: CandidateExperiment,
    hypothesis_tree: HypothesisTreeState,
    experiment_memory: ExperimentMemoryState | None,
) -> tuple[float, float]:
    """Estimate input/output token counts for one candidate experiment."""
    active_hypotheses = [node for node in hypothesis_tree.nodes if node.status in {"active", "converged"}]
    input_tokens = 2000 + len(str(candidate.design.model_dump(mode="json"))) / 3
    input_tokens += sum((len(node.statement) / 3) + 200 for node in active_hypotheses)
    input_tokens += (len(candidate.purpose or "") + len(candidate.distinguishing_insight or "")) / 3
    history_count = len(experiment_memory.entries[-3:]) if experiment_memory else 0
    input_tokens += 300 * history_count
    output_tokens = 1500
    return input_tokens, output_tokens


def _llm_price_rates() -> dict[str, float]:
    return get_bailian_model_prices(get_llm_model_for_role("experiment_planner"))


def _estimate_llm_cost_yuan(
    candidate: CandidateExperiment,
    hypothesis_tree: HypothesisTreeState,
    experiment_memory: ExperimentMemoryState | None,
    price_rates: dict[str, float] | None = None,
) -> float:
    price_rates = price_rates or _llm_price_rates()
    input_tokens, output_tokens = _estimate_llm_token_usage(
        candidate=candidate,
        hypothesis_tree=hypothesis_tree,
        experiment_memory=experiment_memory,
    )
    input_price = price_rates.get("input", DEFAULT_LLM_INPUT_PRICE)
    output_price = price_rates.get("output", DEFAULT_LLM_OUTPUT_PRICE)
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000.0


def _llm_cost_budget_yuan(token_budget: int, price_rates: dict[str, float]) -> float:
    raw_budget = get_runtime_setting("BAILIAN_COST_BUDGET_YUAN")
    if raw_budget:
        try:
            custom_budget = float(raw_budget)
        except ValueError:
            custom_budget = 0.0
        if custom_budget > 0.0:
            return custom_budget
    output_price = price_rates.get("output", DEFAULT_LLM_OUTPUT_PRICE)
    return max(token_budget, 1) * output_price / 1_000_000.0


def _estimate_compute_seconds(candidate: CandidateExperiment, data_dictionary: DataDictionary) -> float:
    n_samples = max(data_dictionary.total_samples, 1)
    n_features = max(len(set(candidate.design.control + candidate.design.treatment)), 1)
    step_count = max(len(candidate.design.lags), 1)
    data_coeff = max(1.0, (n_samples / 1000.0) * (n_features / 20.0))
    step_coeff = 1.0 + (step_count - 1) * 0.2
    run_specs = list(candidate.design.runs_spec or [])
    run_count = max(len(run_specs), 2 if candidate.design.treatment else 1)
    search_candidates = max(
        1,
        sum(
            int(spec.parameters.get("search_candidates") or 0)
            or int(spec.parameters.get("alpha_candidates") or 0)
            or int(spec.parameters.get("l1_ratio_candidates") or 0)
            for spec in run_specs
        ),
    )
    diagnostics = list(candidate.design.diagnostics_spec or [])
    diagnostic_runs = max(
        sum(int(item.parameters.get("n_iterations") or 0) for item in diagnostics)
        or len(diagnostics),
        0,
    )
    cv_folds = max(
        [
            int(item.parameters.get("cv_folds") or 0)
            for item in diagnostics
            if int(item.parameters.get("cv_folds") or 0) > 0
        ]
        or [5],
    )
    run_coeff = 1.0 + (run_count - 2 if run_count > 2 else 0) * 0.45
    search_coeff = 1.0 + max(search_candidates - 1, 0) * 0.08
    cv_coeff = 1.0 + max(int(cv_folds) - 5, 0) * 0.06
    diagnostic_coeff = 1.0 + diagnostic_runs * 0.12
    return round(
        10.0
        * data_coeff
        * step_coeff
        * run_coeff
        * search_coeff
        * cv_coeff
        * diagnostic_coeff,
        2,
    )


def _has_similar_history(candidate: CandidateExperiment, experiment_memory: ExperimentMemoryState | None) -> bool:
    if experiment_memory is None:
        return False
    return any(_historical_similarity(candidate, entry) >= 0.6 for entry in experiment_memory.entries)


def _estimate_performance_gain(
    record: UncertaintyRecord,
    primary_x: str,
    experiment_memory: ExperimentMemoryState | None = None,
) -> float:
    return _estimate_performance_gain_from_text(
        text=f"{record.question} {record.description}",
        primary_x=primary_x,
        experiment_memory=experiment_memory,
        partially_resolved=record.resolution_status == "partially_resolved",
    )


def _estimate_performance_gain_from_text(
    *,
    text: str,
    primary_x: str,
    experiment_memory: ExperimentMemoryState | None = None,
    partially_resolved: bool = False,
) -> float:
    current_best_rmse = _current_best_rmse(experiment_memory)
    if current_best_rmse is None:
        return 0.0
    predicted_after_rmse = _predict_after_rmse(
        text=text,
        primary_x=primary_x,
        current_best_rmse=current_best_rmse,
        experiment_memory=experiment_memory,
        partially_resolved=partially_resolved,
    )
    pg_expected = (current_best_rmse - predicted_after_rmse) / max(current_best_rmse, EPSILON)
    return round(min(max(pg_expected, -1.0), 1.0), 4)


def _performance_gain_rationale(
    record: UncertaintyRecord,
    primary_x: str,
    experiment_memory: ExperimentMemoryState | None = None,
) -> str:
    return _performance_gain_rationale_from_text(
        text=f"{record.question} {record.description}",
        primary_x=primary_x,
        experiment_memory=experiment_memory,
        partially_resolved=record.resolution_status == "partially_resolved",
    )


def _performance_gain_rationale_from_text(
    *,
    text: str,
    primary_x: str,
    experiment_memory: ExperimentMemoryState | None = None,
    partially_resolved: bool = False,
) -> str:
    current_best_rmse = _current_best_rmse(experiment_memory)
    if current_best_rmse is None:
        return "当前无历史实验 RMSE 可供校准，第一轮或无历史时将 PG_expected 视为 0。"
    predicted_after_rmse = _predict_after_rmse(
        text=text,
        primary_x=primary_x,
        current_best_rmse=current_best_rmse,
        experiment_memory=experiment_memory,
        partially_resolved=partially_resolved,
    )
    pg_expected = (current_best_rmse - predicted_after_rmse) / max(current_best_rmse, EPSILON)
    return (
        "按 PG_expected(E) = (RMSE_current_best - RMSE_predicted_after(E)) / RMSE_current_best 估计；"
        f"当前最优 RMSE={current_best_rmse:.4f}，预测实验后 RMSE={predicted_after_rmse:.4f}，"
        f"得到 PG_expected≈{pg_expected:.4f}。"
    )


def _predict_after_rmse(
    *,
    text: str,
    primary_x: str,
    current_best_rmse: float,
    experiment_memory: ExperimentMemoryState | None = None,
    partially_resolved: bool = False,
) -> float:
    text = text.lower()
    estimated_gain = _historical_pg_mean(experiment_memory)
    if primary_x.lower() in text:
        estimated_gain += 0.025
    if any(token in text for token in ("independent", "独立", "incremental", "增量")):
        estimated_gain += 0.035
    if any(token in text for token in ("lag", "滞后", "稳定", "跨时间")):
        estimated_gain += 0.02
    if any(token in text for token in ("mediat", "中介", "through")):
        estimated_gain += 0.015
    if partially_resolved:
        estimated_gain = max(estimated_gain - 0.015, 0.0)
    estimated_gain = min(max(estimated_gain, 0.0), 0.25)
    return max(current_best_rmse * (1.0 - estimated_gain), EPSILON)


def _estimate_baseline_performance_gain(experiment_memory: ExperimentMemoryState | None = None) -> float:
    current_best_rmse = _current_best_rmse(experiment_memory)
    if current_best_rmse is None:
        return 0.0
    predicted_after_rmse = current_best_rmse
    return round((current_best_rmse - predicted_after_rmse) / max(current_best_rmse, EPSILON), 4)


def _baseline_performance_gain_rationale(experiment_memory: ExperimentMemoryState | None = None) -> str:
    current_best_rmse = _current_best_rmse(experiment_memory)
    if current_best_rmse is None:
        return "当前无历史实验 RMSE 可供校准，对照组补位实验的 PG_expected 记为 0。"
    return (
        "按 PG_expected(E) = (RMSE_current_best - RMSE_predicted_after(E)) / RMSE_current_best 估计；"
        f"对照组补位实验预期保持当前最优 RMSE={current_best_rmse:.4f}，因此 PG_expected≈0.0000。"
    )


def _current_best_rmse(
    experiment_memory: ExperimentMemoryState | None,
    *,
    candidate: CandidateExperiment | None = None,
) -> float | None:
    if experiment_memory is None:
        return None
    best: float | None = None
    for entry in experiment_memory.entries:
        if candidate is not None and not _same_experiment_structure(candidate, entry):
            continue
        metrics = entry.metrics_snapshot
        if metrics is None:
            continue
        for value in (metrics.baseline_rmse, metrics.treatment_rmse):
            if value is None:
                continue
            best = value if best is None else min(best, value)
    return best


def _historical_pg_mean(experiment_memory: ExperimentMemoryState | None) -> float:
    if experiment_memory is None:
        return 0.0
    values: list[float] = []
    for entry in experiment_memory.entries:
        metrics = entry.metrics_snapshot
        if metrics is None or metrics.baseline_rmse is None or metrics.treatment_rmse is None:
            continue
        before = metrics.baseline_rmse
        after = metrics.treatment_rmse
        gain = (before - after) / max(before, EPSILON)
        values.append(gain)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _estimate_candidate_risk(record: UncertaintyRecord) -> float:
    if record.resolution_status == "partially_resolved":
        return 0.28
    text = f"{record.question} {record.description}".lower()
    if any(token in text for token in ("lag", "滞后", "跨时间")):
        return 0.6
    if any(token in text for token in ("mediat", "中介")):
        return 0.5
    return 0.35


def _estimate_candidate_cost(record: UncertaintyRecord) -> float:
    if record.resolution_status == "partially_resolved":
        return 0.25
    return _estimate_uncertainty_cost(record)


def _unique_preserve_order(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", str(text).lower())


def _collect_hypothesis_ids(
    top_uncertainties: list[PrioritizedUncertainty],
    uncertainty_state: UncertaintyState,
) -> list[str]:
    ids: list[str] = []
    record_index = uncertainty_state.record_index()
    for item in top_uncertainties:
        record = record_index.get(item.uncertainty_id)
        if record is None:
            continue
        ids.extend(record.related_hypotheses)
    return _unique_preserve_order(ids)


def _build_planning_context(
    planning_feedback: str | None,
    data_dictionary: DataDictionary,
) -> PlanningContext:
    planning_feedback = _sanitize_guidance_text(planning_feedback or "")
    if not planning_feedback:
        return PlanningContext()

    text = planning_feedback.lower()
    preferred_features = tuple(
        feature for feature in data_dictionary.feature_candidates if feature.lower() in text
    )
    keywords = tuple(
        keyword
        for keyword in ("单变量", "独立", "中介", "滞后", "稳定", "baseline", "incremental")
        if keyword.lower() in text
    )
    prefer_simple_design = any(token in text for token in ("单变量", "single", "只测", "先测试"))
    return PlanningContext(
        guidance_text=planning_feedback,
        preferred_features=preferred_features,
        preferred_keywords=keywords,
        prefer_simple_design=prefer_simple_design,
    )


def _sanitize_guidance_text(text: str | None) -> str | None:
    """Drop corrupted or over-long guidance before it reaches candidate notes."""
    if not text:
        return None
    text = str(text).strip()
    if not text:
        return None
    if len(text) > 4000:
        return None
    corrupted_markers = (
        "\\\"",
        "yload",
        "project\": {",
        '"research_question"',
        '"current_round"',
        '"active_hypotheses"',
        '"root_question"',
        '"data_dictionary_summary"',
    )
    lowered = text.lower()
    if any(marker in lowered for marker in corrupted_markers):
        return None
    if " {" in text or text.count("{") > 3:
        return None
    return text


def _rank_uncertainties_with_feedback(
    top_uncertainties: list[PrioritizedUncertainty],
    uncertainty_state: UncertaintyState,
    planning_context: PlanningContext,
) -> list[PrioritizedUncertainty]:
    if not planning_context.guidance_text:
        return top_uncertainties

    record_index = uncertainty_state.record_index()

    def score(item: PrioritizedUncertainty) -> tuple[float, float]:
        record = record_index.get(item.uncertainty_id)
        if record is None:
            return (item.priority_score, item.priority_score)
        boost = 0.0
        source = f"{record.question} {record.description}".lower()
        for feature in planning_context.preferred_features:
            if feature.lower() in source:
                boost += 0.2
        for keyword in planning_context.preferred_keywords:
            if keyword.lower() in source:
                boost += 0.1
        return (round(item.priority_score + boost, 4), item.priority_score)

    return sorted(top_uncertainties, key=score, reverse=True)


def _planning_notes(planning_context: PlanningContext) -> list[str]:
    guidance_text = _sanitize_guidance_text(planning_context.guidance_text)
    if not guidance_text:
        return []
    notes = [f"human_feedback:{guidance_text}"]
    if planning_context.preferred_features:
        notes.append(f"preferred_features:{','.join(planning_context.preferred_features)}")
    if planning_context.prefer_simple_design:
        notes.append("preferred_design:simple")
    return notes


def _build_candidate_novelty(planning_context: PlanningContext) -> str:
    if planning_context.guidance_text:
        return "new uncertainty-focused experiment with human feedback guidance"
    return "new uncertainty-focused experiment"


def _candidate_set_note(candidate_count: int, planning_context: PlanningContext) -> str | None:
    notes: list[str] = []
    if candidate_count < 3:
        notes.append("当前候选实验不足3个，建议后续由实验规划者补充更多候选实验。")
    guidance_text = _sanitize_guidance_text(planning_context.guidance_text)
    if guidance_text:
        notes.append(f"本轮候选已纳入 human_feedback: {guidance_text}")
    return " ".join(notes) if notes else None


def _expected_from_node(node) -> str:
    if node.status in {"weakened", "pruned"} or node.support_score < 0.40:
        return "near_zero"
    if any(token in node.statement.lower() for token in ("伴随", "competition", "null")):
        return "near_zero"
    return "positive"


def _candidate_rationale_context(record: UncertaintyRecord, node_index: dict[str, object]) -> dict[str, list[str] | str]:
    source_types: list[str] = []
    focus_features: list[str] = []
    trigger_tags: list[str] = []
    summaries: list[str] = []
    for hypothesis_id in record.related_hypotheses:
        node = node_index.get(hypothesis_id)
        if node is None or node.generation_rationale is None:
            continue
        rationale = node.generation_rationale
        trigger_tags.append(rationale.trigger)
        label = (
            getattr(node, "display_hypothesis_id", None)
            or (f"H{node.level}" if getattr(node, "level", None) else "")
            or hypothesis_id
        )
        summaries.append(f"{label}:{rationale.summary}")
        focus_features.extend(rationale.derived_features)
        source_types.extend(signal.signal_type for signal in rationale.source_signals)
    return {
        "trigger_tags": _unique_preserve_order(trigger_tags),
        "focus_features": _unique_preserve_order(focus_features),
        "source_types": _unique_preserve_order(source_types),
        "summary": " | ".join(summaries[:3]),
    }


def _candidate_source_context(
    record: UncertaintyRecord,
    node_index: dict[str, object],
) -> dict[str, object]:
    context: dict[str, object] = {}
    for hypothesis_id in record.related_hypotheses:
        node = node_index.get(hypothesis_id)
        if node is None or node.generation_rationale is None:
            continue
        context[hypothesis_id] = node.generation_rationale.model_copy(deep=True)
    return context


def _candidate_purpose(record: UncertaintyRecord, rationale_context: dict[str, list[str] | str]) -> str:
    return f"承接科学不确定性：{record.question}"


def _replace_hypothesis_ids(text: str, node_index: dict[str, object]) -> str:
    """Replace hypothesis ids with their statements in display-only text."""
    display = text
    for hypothesis_id in sorted(node_index, key=len, reverse=True):
        node = node_index.get(hypothesis_id)
        if node is None or not getattr(node, "statement", ""):
            continue
        display = display.replace(hypothesis_id, node.statement)
    return display


def _replace_hypothesis_ids_with_short_labels(text: str, node_index: dict[str, object]) -> str:
    """Replace canonical hypothesis ids with user-facing H1/H2 labels."""
    display = text
    for hypothesis_id in sorted(node_index, key=len, reverse=True):
        node = node_index.get(hypothesis_id)
        if node is None:
            continue
        label = getattr(node, "display_hypothesis_id", None) or (
            f"H{node.level}" if hasattr(node, "level") else None
        )
        if label:
            display = display.replace(hypothesis_id, label)
    return display


def _apply_semantic_to_candidate_display(
    candidate: CandidateExperiment,
    semantic: VariableSemanticService,
    node_index: dict[str, object],
) -> None:
    """Translate user-facing candidate text while keeping execution fields raw."""
    candidate.purpose = _replace_hypothesis_ids(
        semantic.display_text(candidate.purpose),
        node_index,
    )
    if candidate.scientific_question:
        candidate.scientific_question = _replace_hypothesis_ids(
            semantic.display_text(candidate.scientific_question),
            node_index,
        )
    if candidate.distinguishing_insight:
        candidate.distinguishing_insight = _replace_hypothesis_ids_with_short_labels(
            semantic.display_text(candidate.distinguishing_insight),
            node_index,
        )
        candidate.distinguishing_insight = _replace_hypothesis_ids(
            candidate.distinguishing_insight,
            node_index,
        )
    candidate.design.display_target = semantic.to_display(candidate.design.target)
    candidate.design.display_control = semantic.display_list(candidate.design.control)
    candidate.design.display_treatment = semantic.display_list(candidate.design.treatment)
    if candidate.design.design_focus:
        candidate.design.display_design_focus = semantic.to_display(candidate.design.design_focus)


def _distinguishing_insight(
    record: UncertaintyRecord,
    rationale_context: dict[str, list[str] | str],
    primary_x: str,
    focus_feature: str | None = None,
) -> str:
    focus_text = focus_feature or primary_x
    hypothesis_ids = list(record.disagreement.keys()) or record.related_hypotheses[:2]
    if hypothesis_ids:
        return (
            f"通过比较对照组与实验组，检验 {focus_text} 的相对增量贡献："
            f"对照组使用常规日地环境变量但不加入待验证焦点变量 {focus_text}，"
            f"实验组在相同窗口与超前期下加入 {focus_text}，"
            f"以相同窗口和超前期评价太阳风速度预测的 RMSE 与 Pearson_r 变化；"
            f"由此区分假设 { '、'.join(hypothesis_ids[:3]) } 的实证差异。"
        )
    return (
        f"通过比较对照组与实验组，检验 {focus_text} 的相对增量贡献："
        "对照组不含待验证焦点变量，实验组在相同窗口与超前期下加入该变量，"
        "以相同窗口和超前期评价太阳风速度预测的 RMSE 与 Pearson_r 变化。"
    )


def _information_gain_rationale(
    record: UncertaintyRecord,
    rationale_context: dict[str, list[str] | str],
) -> str:
    if rationale_context["trigger_tags"]:
        return (
            "优先级越高，说明该实验更可能消除关键科学不确定性；"
            f"同时可区分由 {','.join(rationale_context['trigger_tags'][:2])} 触发的假设来源。"
        )
    return "优先级越高，说明该实验更可能消除关键科学不确定性。"


def _rationale_design_notes(
    record: UncertaintyRecord,
    rationale_context: dict[str, list[str] | str],
    *,
    tree=None,
    uncertainty_records=None,
) -> list[str]:
    raw_notes: list[str] = []
    if rationale_context["trigger_tags"]:
        raw_notes.append(f"hypothesis_triggers:{','.join(rationale_context['trigger_tags'])}")
    if rationale_context["summary"]:
        raw_notes.append(f"hypothesis_source_summary:{rationale_context['summary']}")
    if record.disagreement:
        raw_notes.append(f"disagreement_hypotheses:{','.join(record.disagreement.keys())}")
    return [
        clean_llm_text(note, tree=tree, uncertainty_records=uncertainty_records)
        for note in raw_notes
        if note
    ]
